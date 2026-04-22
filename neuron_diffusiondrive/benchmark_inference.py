"""
DiffusionDrive INFERENCE benchmark on AWS Trainium (trn1).
Measures single-NeuronCore inference latency with synthetic data.
Uses eval mode with 2-step DDIM denoising (matching the paper).

Usage:
    python benchmark_inference.py [--batch_size 1] [--steps 50]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import torch_xla
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neuron_diffusiondrive.model_standalone import DiffusionDriveModel, DiffusionDriveConfig


def create_synthetic_input(config, batch_size, device):
    return {
        "camera_feature": torch.randn(batch_size, 3, config.camera_height, config.camera_width, device=device),
        "lidar_feature": torch.randn(batch_size, 1, config.lidar_resolution_height, config.lidar_resolution_width, device=device),
        "status_feature": torch.randn(batch_size, 8, device=device),
    }


def run_benchmark(args):
    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    config = DiffusionDriveConfig()
    # Reduce resolution to fit in 16GB HBM per NeuronCore
    if args.reduced_resolution:
        config.camera_width = 512
        config.camera_height = 128
        config.img_horz_anchors = 512 // 32
        config.img_vert_anchors = 128 // 32
        config.lidar_resolution_width = 128
        config.lidar_resolution_height = 128
        config.lidar_vert_anchors = 128 // 32
        config.lidar_horz_anchors = 128 // 32
        print(f"Using reduced resolution: {config.camera_height}x{config.camera_width} camera, {config.lidar_resolution_height}x{config.lidar_resolution_width} BEV")
    plan_anchor_path = os.path.join(os.path.dirname(__file__), 'plan_anchors.npy')
    if not os.path.exists(plan_anchor_path):
        np.save(plan_anchor_path, np.random.RandomState(42).randn(20, 8, 2).astype(np.float32) * 5.0)
    config.plan_anchor_path = plan_anchor_path

    print("Creating model...")
    model = DiffusionDriveModel(config)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.1f}M")

    model = model.to(device)
    model.eval()

    features = create_synthetic_input(config, args.batch_size, device)

    # Warmup (compilation)
    print("Warmup (Neuron compilation)...")
    compile_start = time.time()
    with torch.no_grad():
        output = model(features, targets=None)
        xm.mark_step()
    compile_time = time.time() - compile_start
    print(f"Compilation: {compile_time:.1f}s")

    # Benchmark
    print(f"\nRunning {args.steps} inference steps...")
    latencies = []

    for step in range(args.steps):
        start = time.time()
        with torch.no_grad():
            output = model(features, targets=None)
            xm.mark_step()
        latency = time.time() - start
        latencies.append(latency)
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{args.steps}: {latency*1000:.1f}ms")

    # Results (drop first 3 for cache warmup)
    latencies = latencies[3:] if len(latencies) > 6 else latencies
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    fps = args.batch_size / avg_latency

    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK: DiffusionDrive on Trainium")
    print("=" * 60)
    print(f"Backbone:           ResNet-34")
    print(f"Batch size:         {args.batch_size}")
    print(f"Image size:         {config.camera_height}x{config.camera_width}")
    print(f"BEV grid:           {config.lidar_resolution_height}x{config.lidar_resolution_width}")
    print(f"Diffusion steps:    2 (truncated DDIM)")
    print(f"Trajectory modes:   20")
    print(f"Parameters:         {total/1e6:.1f}M")
    print(f"Compilation time:   {compile_time:.1f}s")
    print(f"Avg latency:        {avg_latency*1000:.1f} +/- {std_latency*1000:.1f} ms")
    print(f"P50 latency:        {p50*1000:.1f} ms")
    print(f"P99 latency:        {p99*1000:.1f} ms")
    print(f"Throughput:         {fps:.1f} FPS")
    print(f"Paper reference:    45.3 ms on single A100")
    print("=" * 60)

    results = {
        "benchmark_type": "inference",
        "model": "DiffusionDrive",
        "backbone": "ResNet-34",
        "batch_size": args.batch_size,
        "image_size": f"{config.camera_height}x{config.camera_width}",
        "bev_grid": f"{config.lidar_resolution_height}x{config.lidar_resolution_width}",
        "diffusion_steps": 2,
        "total_params_M": round(total / 1e6, 1),
        "compile_time_s": round(compile_time, 1),
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "std_latency_ms": round(std_latency * 1000, 1),
        "p50_latency_ms": round(p50 * 1000, 1),
        "p99_latency_ms": round(p99 * 1000, 1),
        "throughput_fps": round(fps, 1),
        "paper_a100_latency_ms": 45.3,
        "steps": args.steps,
    }
    results_path = os.path.join(os.path.dirname(__file__), 'benchmark_results_inference.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionDrive Inference Benchmark")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--reduced_resolution", action="store_true", default=True)
    parser.add_argument("--full_resolution", dest="reduced_resolution", action="store_false")
    args = parser.parse_args()
    run_benchmark(args)
