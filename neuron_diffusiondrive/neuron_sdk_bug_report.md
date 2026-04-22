# Bug Report: NCC_ITEN404 in 7x7 Conv2d Backward Pass

## Environment
- **Instance**: trn1.32xlarge
- **Neuron SDK**: 2.23 (neuronx-cc 2.23.6484.0+3b612583)
- **PyTorch**: 2.8 (torch-neuronx)
- **OS**: Ubuntu 22.04 (Deep Learning AMI)

## Summary

The backward pass of a 7x7 `Conv2d` layer (ResNet-34 stem) fails with `NCC_ITEN404` internal compiler error during NKI kernel code generation. All 3x3, 1x1, and strided convolutions compile successfully — only the 7x7 kernel triggers this error.

## Minimal Reproducer

```python
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

device = xm.xla_device()

# This FAILS:
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
).to(device).train()

x = torch.randn(1, 3, 256, 1024, device=device, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # <-- NCC_ITEN404 here
xm.mark_step()
```

```python
# This PASSES (same structure, 3x3 kernel):
model = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
).to(device).train()

x = torch.randn(1, 64, 128, 512, device=device, requires_grad=True)
out = model(x)
loss = out.sum()
loss.backward()  # <-- PASSES
xm.mark_step()
```

## Error Message

```
[INTERNAL_ERROR] [NCC_ITEN404] Internal tensorizer error: BirCodeGenLoop:tensorcopy
src start_partition(0) or dst start_partition(i_3) is not multiple of partitions_per_bank (32).
tensorcopy: float32<1 x 12288> TongaSB partitions[1] float32 (2, 128, 21609)
%'conv2d_dw_fb01_io01_01bf_rep_nhwc_PcinhConv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh_sg0005_20404_a0_img_local_prefetch'
```

The error occurs in the NKI kernel `Conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh` during backward graph compilation. The specific failure is a `tensorcopy` partition alignment check.

## Systematic Test Results

| Conv Type | Input Shape | Backward | Compile Time |
|-----------|-------------|----------|-------------|
| 7x7, stride=2, 3→64 | (1, 3, 256, 1024) | **FAIL** (NCC_ITEN404) | — |
| 7x7, stride=2, 3→64 | (1, 3, 64, 64) | **FAIL** (NCC_ITEN404) | — |
| 7x7, stride=2, 1→64 | (1, 1, 256, 256) | **FAIL** (NCC_ITEN404) | — |
| 3x3, stride=1, 64→64 | (1, 64, 128, 512) | PASS | 62.4s |
| 3x3, stride=1, 256→256 | (1, 256, 16, 64) | PASS | 30.6s |
| 3x3, stride=1, 512→512 | (1, 512, 8, 32) | PASS | 4.6s |
| 3x3, stride=2, 128→256 | (1, 128, 32, 128) | PASS | 23.5s |
| 1x1, 64→128 | (1, 64, 8, 32) | PASS | 1.8s |
| ResNet-34 layer1-4 (no stem) | (1, 64, 128, 512) | PASS | 60.1s |

## Impact

This blocks full end-to-end training of ResNet-34-based models on Trainium. The workaround is freezing the stem (conv1+bn1, ~9.5K params per encoder), which allows training 99.98% of the model (60.7M out of 60.7M total params). But true full training would require this fix.

## Use Case

Training DiffusionDrive (CVPR 2025, end-to-end autonomous driving) on trn1.32xlarge. The model uses two ResNet-34 backbones (image + LiDAR) with GPT-style fusion. With the stem frozen, we can train 60.7M params; without the fix, only 8.7M params (head-only fine-tuning).

## Workaround

Freeze `conv1` and `bn1` of the ResNet encoder and run them under `torch.no_grad()`. All subsequent layers (layer1-4, including strided downsample blocks) compile and train successfully.
