# DiffusionDrive Trainium Adaptation Guide

This guide describes the nine code changes made to compile and train DiffusionDrive on AWS Trainium (`trn1.32xlarge`) with the Neuron SDK.

## Model overview

DiffusionDrive is an end-to-end autonomous driving model that uses truncated diffusion for multi-mode trajectory planning. Architecture: ResNet-34 backbone + TransFuser (GPT-style image/LiDAR fusion) + diffusion-based planning head.

- Paper: [DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](https://arxiv.org/abs/2411.15139) (CVPR 2025 Highlight)
- Total parameters: 60.7M
- Trainable parameters with stem-only freeze: 60.7M
- License: MIT
- Benchmark: NAVSIM

## Prerequisites

- Familiarity with PyTorch, diffusion models, and XLA graph execution concepts
- Neuron SDK 2.23 or later, with `torch-neuronx` for PyTorch 2.8
- A `trn1.32xlarge` (or `trn1.2xlarge` for debugging) instance running the Neuron Deep Learning AMI
- AWS account setup and deployment instructions: see the [README](../README.md)

## Why DiffusionDrive for Trainium

Compared to other open-source end-to-end driving models, DiffusionDrive required the fewest Neuron adaptations:

| Criteria | VAD (2023) | DiffusionDrive (2025) | SparseDrive (2024) |
|----------|-----------|----------------------|-------------------|
| Custom CUDA ops | Deformable attention | None | Deformable attention |
| mmcv dependency | Yes | No | Yes |
| `F.grid_sample` usage | Core of BEV (many) | 1 small module | Core of BEV |
| Neuron blockers | 3 | 2 | 3 |

## The nine adaptations

Each change is summarized below. Source files:
- `blocks_neuron.py` — `F.grid_sample` replacement and cross-BEV attention
- `model_standalone.py` — self-contained model (no NAVSIM runtime dependency)
- `benchmark_neuron.py` — training throughput benchmark with freeze options

### 1. Replace `F.grid_sample` with manual bilinear interpolation

`torch.nn.functional.grid_sample` is not implemented in the Neuron XLA backend. The `manual_bilinear_sample` helper in `blocks_neuron.py` uses `torch.gather` for corner lookups, arithmetic for bilinear weights, and boundary masking to match `padding_mode='zeros'`. Output agreement with `F.grid_sample` is under 1e-4 (verified by `verify_grid_sample.py`).

### 2. Freeze stem convolution and batch norm

The backward pass of a 7x7 `Conv2d` stem triggers an internal compiler error (`NCC_ITEN404`). Freezing `conv1` and `bn1` on both ResNet-34 encoders (image and LiDAR) allows the remaining 60.7M parameters (99.98%) to train. See `neuron_sdk_bug_report.md` for a reproducer.

### 3. Detach grid-sampled features

`torch.gather` backward generates `scatter_add` operations whose transpose pattern fails during Neuron tensorizer code generation. Wrapping the `manual_bilinear_sample` call in `torch.no_grad()` and detaching the output removes that edge from the backward graph. The cross-BEV attention module still learns through the `attention_weights` and `output_proj` paths.

### 4. Insert `xm.mark_step()` before the trajectory head

A monolithic forward + backward graph for the full model exceeds the Neuron compiler's internal limits, even though each sub-component compiles individually. Adding `xm.mark_step()` during training splits the XLA graph without changing the computation.

### 5. Replace `torch.gather` in the loss with `F.one_hot` mask selection

The trajectory loss originally used `torch.gather` to select the best-matching mode for L1 regression. A one-hot mask multiplied against per-mode losses is mathematically equivalent and avoids the `scatter_add` backward path:

```python
mode_mask = F.one_hot(mode_idx, num_classes=ego_fut_mode).float()
per_mode_loss = (poses_reg - target_traj.unsqueeze(1)).abs().mean(dim=(-2, -1))
loss = (per_mode_loss * mode_mask).sum() / bs
```

### 6. Out-of-place tensor operations

In-place slice assignments such as `poses_reg[..., :2] = ...` can break XLA autograd graph construction. The decoder now builds the updated pose tensor with `torch.cat` instead.

### 7. Bilinear upsampling to nearest-neighbor

`F.interpolate(mode='bilinear')` and `nn.Upsample(mode='bilinear')` produce `reduce-window` HLO operations with `lhs_dilate > 1`, which Neuron does not support in the backward pass. Both sites in the BEV feature fusion and BEV semantic head now use `mode='nearest'`.

### 8. Dynamic embedding size in positional encoding

Positional embeddings are sized from the runtime feature map shape rather than a hard-coded constant, so the same model works at multiple camera and BEV resolutions.

### 9. Reduced resolution for inference benchmarking

The full-resolution inference graph (1024x256 camera, 256x256 BEV) requires 17.02 GB of HBM per NeuronCore pair, exceeding the 16 GB trn1 budget. The inference benchmark exposes a `--reduced_resolution` flag (512x128 camera, 128x128 BEV) that fits. Training stays at full resolution because its backward graph uses less HBM than the forward-only inference graph.

## Benchmark summary

Training on a single NeuronCore (batch size 1, stem-only freeze, full resolution):
- Average step time: 172.0 ms
- Throughput: 5.82 samples/sec per NeuronCore
- Compilation time: about 25 minutes
- Neuron SDK: 2.23 (`neuronxcc-2.23.6484.0`)

Inference on a single NeuronCore (batch size 1, reduced resolution):
- Average latency: 159.4 ms
- P99 latency: 161.0 ms

See the top-level README for the trn1 vs. p4d.24xlarge cost comparison.

## Trainium 2 outlook

Trainium 2 (`trn2.48xlarge`) provides 96 GB of HBM per chip, which can accommodate the full-resolution inference and full-backbone training that do not fit on trn1. Some of the backward-pass compiler issues listed above may also be resolved in future Neuron SDK releases.

## File structure

```
neuron_diffusiondrive/
  __init__.py
  blocks_neuron.py              # grid_sample replacement + cross-BEV attention
  model_standalone.py           # self-contained model (no NAVSIM dependency)
  train_navsim_mini.py          # training loop (trn1 and GPU)
  preprocess_navsim_real.py     # NAVSIM mini preprocessing
  benchmark_neuron.py           # training throughput benchmark
  benchmark_gpu.py              # GPU reference benchmark
  benchmark_inference.py        # inference latency benchmark
  compare_results.py            # compares trn1 vs. GPU results
  verify_grid_sample.py         # correctness check for manual_bilinear_sample
  deploy_benchmark.sh           # trn1 deployment wrapper
  deploy_gpu_benchmark.sh       # p4d deployment wrapper
  download_and_preprocess.sh    # NAVSIM mini download + preprocess
  run_train_trn1.sh             # training launcher (trn1)
  run_train_gpu.sh              # training launcher (GPU)
  ADAPTATION_GUIDE.md           # this file
  neuron_sdk_bug_report.md      # 7x7 Conv2d backward reproducer
  plan_anchors.npy              # trajectory anchor centroids
```

## Conclusion

Nine targeted code changes are enough to compile and train DiffusionDrive on AWS Trainium with the Neuron SDK: one operator rewrite (`grid_sample`), three workarounds for `NCC_ITEN404` backward patterns (stem freeze, detached gather, graph split), two loss- and ops-level rewrites (`one_hot` mask, out-of-place updates), one upsampling swap, one embedding generalization, and one resolution reduction for inference.

The result is a model that reaches loss levels comparable to the A100 baseline on NAVSIM mini at roughly 2x lower cost per sample, with 32 NeuronCores on a single trn1.32xlarge providing a strong throughput baseline for data-parallel scale-out. Trainium 2 is the natural next step for workloads that require the full-resolution inference path or unfrozen backbone training.
