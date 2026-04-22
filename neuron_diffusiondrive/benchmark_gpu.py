"""
DiffusionDrive benchmark on GPU (CUDA).
Same model config as the Trainium benchmark for apples-to-apples comparison.
Measures training step time and inference latency.

Usage:
    python benchmark_gpu.py [--batch_size 1] [--steps 100] [--inference_steps 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neuron_diffusiondrive.model_standalone import DiffusionDriveModel, DiffusionDriveConfig


def create_synthetic_batch(config, batch_size, device):
    features = {
        "camera_feature": torch.randn(batch_size, 3, config.camera_height, config.camera_width, device=device),
        "lidar_feature": torch.randn(batch_size, 1, config.lidar_resolution_height, config.lidar_resolution_width, device=device),
        "status_feature": torch.randn(batch_size, 8, device=device),
    }
    targets = {
        "trajectory": torch.randn(batch_size, config.num_poses, 3, device=device),
        "agent_states": torch.randn(batch_size, config.num_bounding_boxes, 5, device=device),
        "agent_labels": torch.randint(0, 2, (batch_size, config.num_bounding_boxes), device=device).float(),
        "bev_semantic_map": torch.randint(0, config.num_bev_classes,
            (batch_size, config.lidar_resolution_height // 2, config.lidar_resolution_width), device=device),
    }
    return features, targets


def freeze_backbone(model):
    frozen = 0
    for name, param in model.named_parameters():
        if '_backbone.' in name:
            param.requires_grad = False
            frozen += 1
    return frozen


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Batch size: {args.batch_size}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    config = DiffusionDriveConfig()
    plan_anchor_path = os.path.join(os.path.dirname(__file__), 'plan_anchors.npy')
    if not os.path.exists(plan_anchor_path):
        np.save(plan_anchor_path, np.random.RandomState(42).randn(20, 8, 2).astype(np.float32) * 5.0)
    config.plan_anchor_path = plan_anchor_path

    model = DiffusionDriveModel(config)
    if args.freeze_backbone:
        frozen = freeze_backbone(model)
        print(f"Frozen {frozen} backbone parameters")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    model = model.to(device)

    # ==================== TRAINING BENCHMARK ====================
    print("\n--- Training Benchmark ---")
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=6e-4, weight_decay=1e-4
    )
    features, targets = create_synthetic_batch(config, args.batch_size, device)

    # Warmup
    for _ in range(3):
        output = model(features, targets=targets)
        loss = output["trajectory_loss"] + config.bev_semantic_weight * F.cross_entropy(
            output["bev_semantic_map"], targets["bev_semantic_map"].long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Benchmark
    step_times = []
    for step in range(args.steps):
        torch.cuda.synchronize()
        start = time.time()

        output = model(features, targets=targets)
        loss = output["trajectory_loss"] + config.bev_semantic_weight * F.cross_entropy(
            output["bev_semantic_map"], targets["bev_semantic_map"].long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        step_times.append(time.time() - start)

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{args.steps}: {step_times[-1]*1000:.1f}ms")

    step_times = step_times[2:]  # drop first 2
    avg_train = np.mean(step_times)
    std_train = np.std(step_times)
    train_sps = args.batch_size / avg_train

    # ==================== INFERENCE BENCHMARK ====================
    print("\n--- Inference Benchmark ---")
    model.eval()
    # Full resolution for GPU (no HBM constraint)
    infer_features = {
        "camera_feature": torch.randn(args.batch_size, 3, config.camera_height, config.camera_width, device=device),
        "lidar_feature": torch.randn(args.batch_size, 1, config.lidar_resolution_height, config.lidar_resolution_width, device=device),
        "status_feature": torch.randn(args.batch_size, 8, device=device),
    }

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            output = model(infer_features, targets=None)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for step in range(args.inference_steps):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(infer_features, targets=None)
        torch.cuda.synchronize()
        latencies.append(time.time() - start)

    latencies = latencies[3:]  # drop first 3
    avg_infer = np.mean(latencies)
    std_infer = np.std(latencies)
    p50_infer = np.percentile(latencies, 50)
    p99_infer = np.percentile(latencies, 99)
    infer_fps = args.batch_size / avg_infer

    # Peak GPU memory
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # ==================== RESULTS ====================
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: DiffusionDrive on {gpu_name}")
    print("=" * 60)
    print(f"Backbone:           ResNet-34 ({'frozen' if args.freeze_backbone else 'trainable'})")
    print(f"Batch size:         {args.batch_size}")
    print(f"Image size:         {config.camera_height}x{config.camera_width}")
    print(f"BEV grid:           {config.lidar_resolution_height}x{config.lidar_resolution_width}")
    print(f"Parameters:         {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"Peak GPU memory:    {peak_mem:.1f} GB")
    print(f"")
    print(f"TRAINING:")
    print(f"  Avg step time:    {avg_train*1000:.1f} +/- {std_train*1000:.1f} ms")
    print(f"  Throughput:       {train_sps:.2f} samples/sec")
    print(f"")
    print(f"INFERENCE (full resolution):")
    print(f"  Avg latency:      {avg_infer*1000:.1f} +/- {std_infer*1000:.1f} ms")
    print(f"  P50 latency:      {p50_infer*1000:.1f} ms")
    print(f"  P99 latency:      {p99_infer*1000:.1f} ms")
    print(f"  Throughput:       {infer_fps:.1f} FPS")
    print("=" * 60)

    results = {
        "gpu": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 1),
        "peak_memory_gb": round(peak_mem, 1),
        "backbone": f"ResNet-34 ({'frozen' if args.freeze_backbone else 'trainable'})",
        "batch_size": args.batch_size,
        "image_size": f"{config.camera_height}x{config.camera_width}",
        "bev_grid": f"{config.lidar_resolution_height}x{config.lidar_resolution_width}",
        "total_params_M": round(total / 1e6, 1),
        "trainable_params_M": round(trainable / 1e6, 1),
        "training": {
            "avg_step_ms": round(avg_train * 1000, 1),
            "std_step_ms": round(std_train * 1000, 1),
            "throughput_sps": round(train_sps, 2),
            "steps": args.steps,
        },
        "inference": {
            "avg_latency_ms": round(avg_infer * 1000, 1),
            "std_latency_ms": round(std_infer * 1000, 1),
            "p50_latency_ms": round(p50_infer * 1000, 1),
            "p99_latency_ms": round(p99_infer * 1000, 1),
            "throughput_fps": round(infer_fps, 1),
            "steps": args.inference_steps,
        },
    }
    results_path = os.path.join(os.path.dirname(__file__), 'benchmark_results_gpu.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionDrive GPU Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--inference_steps", type=int, default=200)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    args = parser.parse_args()
    run_benchmark(args)
