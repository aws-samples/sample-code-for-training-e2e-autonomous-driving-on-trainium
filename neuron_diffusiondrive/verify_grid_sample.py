"""
Verify that manual_bilinear_sample matches F.grid_sample output.
Run on CPU/GPU (not Neuron) to validate correctness before deploying.
"""
import torch
import torch.nn.functional as F
from blocks_neuron import manual_bilinear_sample


def test_equivalence():
    torch.manual_seed(42)

    bs, C, H, W = 2, 256, 64, 64
    nq, np_ = 20, 8

    value = torch.randn(bs, C, H, W)
    grid = torch.randn(bs, nq, np_, 2) * 0.8  # mostly in [-1, 1]

    # Reference: F.grid_sample
    ref = F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Ours: manual bilinear
    ours = manual_bilinear_sample(value, grid)

    max_diff = (ref - ours).abs().max().item()
    mean_diff = (ref - ours).abs().mean().item()

    print(f"Shape:     ref={ref.shape}, ours={ours.shape}")
    print(f"Max diff:  {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")
    print(f"Match:     {'PASS' if max_diff < 1e-4 else 'FAIL'}")

    # Test edge cases: points outside [-1, 1]
    grid_oob = torch.tensor([[[[2.0, 2.0], [-2.0, -2.0]]]], dtype=torch.float32)
    value_small = torch.ones(1, 1, 4, 4)
    ref_oob = F.grid_sample(value_small, grid_oob, mode='bilinear', padding_mode='zeros', align_corners=False)
    ours_oob = manual_bilinear_sample(value_small, grid_oob)

    print(f"\nOut-of-bounds test:")
    print(f"  ref={ref_oob.flatten().tolist()}, ours={ours_oob.flatten().tolist()}")
    print(f"  Match: {'PASS' if (ref_oob - ours_oob).abs().max() < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    test_equivalence()
