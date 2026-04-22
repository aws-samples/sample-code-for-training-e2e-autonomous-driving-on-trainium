"""
DiffusionDrive benchmark on AWS Trainium (trn1).
Measures single-NeuronCore training step throughput with synthetic data.
Self-contained: no NAVSIM/nuplan dependencies required.

Usage:
    python benchmark_neuron.py [--batch_size 1] [--steps 20] [--freeze_backbone]
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

# Neuron imports
import torch_xla
import torch_xla.core.xla_model as xm

# Add parent to path for imports
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
    """Freeze all backbone parameters to avoid Neuron compiler errors in backward pass.

    The TransfuserBackbone contains AdaptiveAvgPool2d and bilinear F.interpolate whose
    backward passes generate unsupported HLO ops on Neuron (reduce-window dilation,
    PF transpose DAG). Freezing the entire backbone prevents backward graph construction
    through these ops.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if '_backbone.' in name:
            param.requires_grad = False
            frozen += 1
    return frozen


def run_benchmark(args):
    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    config = DiffusionDriveConfig()

    # Generate plan anchors
    plan_anchor_path = os.path.join(os.path.dirname(__file__), 'plan_anchors.npy')
    if not os.path.exists(plan_anchor_path):
        rng = np.random.RandomState(42)
        np.save(plan_anchor_path, rng.randn(20, 8, 2).astype(np.float32) * 5.0)
    config.plan_anchor_path = plan_anchor_path

    print("Creating model...")
    model = DiffusionDriveModel(config)

    if args.freeze_backbone:
        frozen = freeze_backbone(model)
        print(f"Frozen {frozen} backbone parameters")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=6e-4, weight_decay=1e-4
    )

    print("Creating synthetic data...")
    features, targets = create_synthetic_batch(config, args.batch_size, device)

    # Warmup (triggers Neuron compilation)
    print("Warmup step (Neuron compilation — may take several minutes)...")
    compile_start = time.time()

    output = model(features, targets=targets)
    # Compute loss — use gradient checkpointing-style approach to avoid
    # large monolithic backward graph that triggers NCC_ITEN404
    loss = output["trajectory_loss"]
    bev_loss = F.cross_entropy(output["bev_semantic_map"], targets["bev_semantic_map"].long())
    total_loss = loss + config.bev_semantic_weight * bev_loss
    total_loss.backward()
    xm.mark_step()
    optimizer.step()
    xm.mark_step()
    optimizer.zero_grad()

    compile_time = time.time() - compile_start
    print(f"Compilation + warmup: {compile_time:.1f}s")

    # Benchmark loop
    print(f"\nRunning {args.steps} benchmark steps...")
    step_times = []

    for step in range(args.steps):
        step_start = time.time()

        output = model(features, targets=targets)
        loss = output["trajectory_loss"]
        bev_loss = F.cross_entropy(output["bev_semantic_map"], targets["bev_semantic_map"].long())
        total_loss = loss + config.bev_semantic_weight * bev_loss
        total_loss.backward()
        xm.mark_step()
        optimizer.step()
        xm.mark_step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        step_times.append(step_time)

        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}/{args.steps}: {step_time*1000:.1f}ms")

    # Results (drop first 2 steps for stability)
    step_times = step_times[2:] if len(step_times) > 4 else step_times
    avg_time = np.mean(step_times)
    std_time = np.std(step_times)
    sps = args.batch_size / avg_time

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS: DiffusionDrive on Trainium")
    print("=" * 60)
    print(f"Backbone:           ResNet-34 ({'frozen' if args.freeze_backbone else 'trainable'})")
    print(f"Batch size:         {args.batch_size}")
    print(f"Image size:         {config.camera_height}x{config.camera_width}")
    print(f"BEV grid:           {config.lidar_resolution_height}x{config.lidar_resolution_width}")
    print(f"Diffusion steps:    2 (truncated DDIM)")
    print(f"Trajectory modes:   20")
    print(f"Parameters:         {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    print(f"Compilation time:   {compile_time:.1f}s")
    print(f"Avg step time:      {avg_time*1000:.1f} +/- {std_time*1000:.1f} ms")
    print(f"Throughput:         {sps:.2f} samples/sec/NeuronCore")
    print("=" * 60)

    results = {
        "model": "DiffusionDrive",
        "backbone": f"ResNet-34 ({'frozen' if args.freeze_backbone else 'trainable'})",
        "batch_size": args.batch_size,
        "image_size": f"{config.camera_height}x{config.camera_width}",
        "bev_grid": f"{config.lidar_resolution_height}x{config.lidar_resolution_width}",
        "diffusion_steps": 2,
        "trajectory_modes": 20,
        "total_params_M": round(total / 1e6, 1),
        "trainable_params_M": round(trainable / 1e6, 1),
        "compile_time_s": round(compile_time, 1),
        "avg_step_ms": round(avg_time * 1000, 1),
        "std_step_ms": round(std_time * 1000, 1),
        "throughput_sps": round(sps, 2),
        "steps": args.steps,
    }
    results_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionDrive Trainium Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    args = parser.parse_args()
    run_benchmark(args)
