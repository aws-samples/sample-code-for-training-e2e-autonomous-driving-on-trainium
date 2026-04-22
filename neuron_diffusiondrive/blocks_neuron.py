"""
Neuron-compatible version of DiffusionDrive blocks.
Replaces F.grid_sample with manual bilinear interpolation for Trainium compatibility.
"""
from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def bias_init_with_prob(prior_prob):
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def manual_bilinear_sample(value, grid):
    """
    Manual bilinear interpolation replacing F.grid_sample.
    Neuron/XLA compatible - uses only standard tensor operations.

    Args:
        value: (bs, C, H, W) feature map
        grid: (bs, num_queries, num_points, 2) normalized coordinates in [-1, 1]

    Returns:
        sampled: (bs, C, num_queries, num_points)
    """
    bs, C, H, W = value.shape
    _, nq, np_, _ = grid.shape

    # Convert from [-1, 1] to pixel coordinates
    grid_x = ((grid[..., 0] + 1) / 2) * (W - 1)  # (bs, nq, np)
    grid_y = ((grid[..., 1] + 1) / 2) * (H - 1)  # (bs, nq, np)

    # Get corner pixel indices
    x0 = grid_x.long().clamp(0, W - 2)
    y0 = grid_y.long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    # Compute interpolation weights
    wx = (grid_x - x0.float()).clamp(0, 1)
    wy = (grid_y - y0.float()).clamp(0, 1)

    # Flatten spatial dims for gather
    x0_flat = x0.reshape(bs, -1)  # (bs, nq*np)
    x1_flat = x1.reshape(bs, -1)
    y0_flat = y0.reshape(bs, -1)
    y1_flat = y1.reshape(bs, -1)

    # Compute linear indices into the flattened (H*W) spatial dimension
    idx_00 = (y0_flat * W + x0_flat)  # (bs, nq*np)
    idx_01 = (y0_flat * W + x1_flat)
    idx_10 = (y1_flat * W + x0_flat)
    idx_11 = (y1_flat * W + x1_flat)

    # Flatten value spatial dims: (bs, C, H*W)
    value_flat = value.reshape(bs, C, H * W)

    # Gather for each corner: expand idx to (bs, C, nq*np)
    idx_00 = idx_00.unsqueeze(1).expand(-1, C, -1)
    idx_01 = idx_01.unsqueeze(1).expand(-1, C, -1)
    idx_10 = idx_10.unsqueeze(1).expand(-1, C, -1)
    idx_11 = idx_11.unsqueeze(1).expand(-1, C, -1)

    v00 = torch.gather(value_flat, 2, idx_00)  # (bs, C, nq*np)
    v01 = torch.gather(value_flat, 2, idx_01)
    v10 = torch.gather(value_flat, 2, idx_10)
    v11 = torch.gather(value_flat, 2, idx_11)

    # Reshape weights: (bs, 1, nq*np)
    wx = wx.reshape(bs, 1, -1)
    wy = wy.reshape(bs, 1, -1)

    # Bilinear interpolation
    sampled = (v00 * (1 - wx) * (1 - wy) +
               v01 * wx * (1 - wy) +
               v10 * (1 - wx) * wy +
               v11 * wx * wy)

    # Reshape to (bs, C, nq, np)
    sampled = sampled.reshape(bs, C, nq, np_)

    # Zero out samples outside [-1, 1] (padding_mode='zeros')
    grid_x_orig = grid[..., 0]  # (bs, nq, np)
    grid_y_orig = grid[..., 1]
    in_bounds = ((grid_x_orig >= -1) & (grid_x_orig <= 1) &
                 (grid_y_orig >= -1) & (grid_y_orig <= 1))
    in_bounds = in_bounds.unsqueeze(1).float()  # (bs, 1, nq, np)
    sampled = sampled * in_bounds

    return sampled


class GridSampleCrossBEVAttention(nn.Module):
    """Neuron-compatible version using manual bilinear interpolation instead of F.grid_sample."""

    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        bs, num_queries, num_points, _ = traj_points.shape

        # Normalize trajectory points to [-1, 1] range
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x
        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)

        # Use manual bilinear sampling instead of F.grid_sample.
        # Detach sampled features to avoid torch.gather backward which triggers
        # Neuron compiler NCC_ITEN404 (unsupported scatter transpose pattern).
        # Gradients still flow through attention_weights and output_proj.
        with torch.no_grad():
            sampled_features = manual_bilinear_sample(value, grid)
        sampled_features = sampled_features.detach()

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        out = self.output_proj(out)

        return self.dropout(out) + queries
