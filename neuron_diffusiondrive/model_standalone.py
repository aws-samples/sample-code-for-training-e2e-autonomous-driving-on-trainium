"""
Standalone DiffusionDrive model for Trainium benchmarking.
Extracts only the model components needed, without NAVSIM/nuplan dependencies.
"""
import copy
import math
from enum import IntEnum
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from diffusers.schedulers import DDIMScheduler

from neuron_diffusiondrive.blocks_neuron import (
    GridSampleCrossBEVAttention,
    linear_relu_ln,
    bias_init_with_prob,
    gen_sineembed_for_position,
)


# Minimal enums (extracted from navsim to avoid dependencies)
class StateSE2Index(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING


class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        return 5

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING


# ============================================================================
# Config (extracted from transfuser_config.py)
# ============================================================================
class DiffusionDriveConfig:
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    lidar_seq_len: int = 1
    use_ground_plane: bool = False
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30
    num_poses: int = 8
    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0


# ============================================================================
# TransFuser Backbone (copied from transfuser_backbone.py, no changes needed)
# ============================================================================
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1
        self.config = config
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * config.img_vert_anchors * config.img_horz_anchors
                + lidar_time_frames * config.lidar_vert_anchors * config.lidar_horz_anchors,
                self.n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[Block(n_embd, config.n_head, config.block_exp, config.attn_pdrop, config.resid_pdrop)
              for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        image_tensor_out = (
            x[:, :self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz * self.seq_len, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        lidar_tensor_out = (
            x[:, self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors:, :]
            .view(bz, lidar_h, lidar_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        return image_tensor_out, lidar_tensor_out


class TransfuserBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_encoder = timm.create_model(config.image_architecture, pretrained=False, features_only=True)

        in_channels = config.lidar_seq_len
        # Neuron-safe: reshape-mean pooling instead of AdaptiveAvgPool2d
        # (AdaptiveAvgPool2d backward triggers unsupported reduce-window with lhs_dilate)
        self._img_pool_target = (config.img_vert_anchors, config.img_horz_anchors)
        self._lidar_pool_target = (config.lidar_vert_anchors, config.lidar_horz_anchors)
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture, pretrained=False, in_chans=in_channels, features_only=True,
        )

        start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList([
            GPT(n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                config=config, lidar_time_frames=1)
            for i in range(4)
        ])
        self.lidar_channel_to_img = nn.ModuleList([
            nn.Conv2d(self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                      self.image_encoder.feature_info.info[start_index + i]["num_chs"], kernel_size=1)
            for i in range(4)
        ])
        self.img_channel_to_lidar = nn.ModuleList([
            nn.Conv2d(self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                      self.lidar_encoder.feature_info.info[start_index + i]["num_chs"], kernel_size=1)
            for i in range(4)
        ])
        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
        self.num_features = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]

        channel = config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # Neuron-safe: nearest-neighbor upsampling (bilinear backward triggers reduce-window dilation)
        self.upsample = nn.Upsample(scale_factor=config.bev_upsample_factor, mode="nearest")
        self.upsample2 = nn.Upsample(
            size=(config.lidar_resolution_height // config.bev_down_sample_factor,
                  config.lidar_resolution_width // config.bev_down_sample_factor),
            mode="nearest",
        )
        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.c5_conv = nn.Conv2d(self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"], channel, (1, 1))

    @staticmethod
    def _reshape_mean_pool(x, target_h, target_w):
        """Neuron-safe average pooling via reshape + mean.
        Replaces AdaptiveAvgPool2d whose backward triggers unsupported
        reduce-window with lhs_dilate on Neuron. Requires input H, W to be
        exact multiples of target_h, target_w (true for ResNet-34 feature maps).
        """
        B, C, H, W = x.shape
        if H == target_h and W == target_w:
            return x
        kh, kw = H // target_h, W // target_w
        return x.reshape(B, C, target_h, kh, target_w, kw).mean(dim=(3, 5))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def _has_frozen_stem(self):
        """Check if the encoder stems have frozen parameters (7x7 conv backward unsupported on Neuron)."""
        return not self.image_encoder.conv1.weight.requires_grad

    def forward(self, image, lidar):
        image_features, lidar_features = image, lidar
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Run stem (conv1+bn1+act1+maxpool) without gradients if frozen.
        # The 7x7 conv backward triggers NCC_ITEN404 on Neuron — all other layers work.
        frozen_stem = self._has_frozen_stem()
        if frozen_stem and len(self.image_encoder.return_layers) > 4:
            with torch.no_grad():
                image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
            image_features = image_features.detach()
        elif len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)

        if frozen_stem and len(self.lidar_encoder.return_layers) > 4:
            with torch.no_grad():
                lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            lidar_features = lidar_features.detach()
        elif len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        for i in range(4):
            image_features = self._forward_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self._forward_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self._fuse(image_features, lidar_features, i)

        x4 = lidar_features
        fused_features = lidar_features
        features = self.top_down(x4)
        return features, fused_features, None

    def _forward_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _fuse(self, img_feat, lid_feat, idx):
        img_embd = self._reshape_mean_pool(img_feat, *self._img_pool_target)
        lid_embd = self._reshape_mean_pool(lid_feat, *self._lidar_pool_target)
        lid_embd = self.lidar_channel_to_img[idx](lid_embd)
        img_out, lid_out = self.transformers[idx](img_embd, lid_embd)
        lid_out = self.img_channel_to_lidar[idx](lid_out)
        # Neuron-safe: nearest interpolation (bilinear backward triggers reduce-window dilation)
        img_out = F.interpolate(img_out, size=img_feat.shape[2:], mode="nearest")
        lid_out = F.interpolate(lid_out, size=lid_feat.shape[2:], mode="nearest")
        return img_feat + img_out, lid_feat + lid_out


# ============================================================================
# Diffusion UNet components (from conditional_unet1d.py)
# ============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ============================================================================
# Trajectory Head with Diffusion
# ============================================================================
class ModulationLayer(nn.Module):
    def __init__(self, embed_dims, condition_dims):
        super().__init__()
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )

    def forward(self, traj_feature, time_embed, global_cond=None, global_img=None):
        global_feature = time_embed
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return traj_feature * (1 + scale) + shift


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=8, ego_fut_mode=20):
        super().__init__()
        self.ego_fut_ts = ego_fut_ts
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature):
        bs, ego_fut_mode, _ = traj_feature.shape
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)
        return plan_reg, plan_cls


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, num_poses, d_model, d_ffn, config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model, config.tf_num_head, num_points=num_poses, config=config, in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model, config.tf_num_head, dropout=config.tf_dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn), nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model, ego_fut_ts=num_poses, ego_fut_mode=20,
        )

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query, agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature = self.time_modulation(traj_feature, time_embed)
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        # Out-of-place construction to avoid in-place slice assignment (XLA-safe)
        xy = poses_reg[..., :2] + noisy_traj_points
        heading = poses_reg[..., 2:3].tanh() * np.pi
        poses_reg = torch.cat([xy, heading], dim=-1)
        return poses_reg, poses_cls


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img=None):
        poses_reg_list, poses_cls_list = [], []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape,
                                       agents_query, ego_query, time_embed, status_encoding, global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()
        return poses_reg_list, poses_cls_list


class TrajectoryHead(nn.Module):
    def __init__(self, num_poses, d_ffn, d_model, plan_anchor_path, config):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample",
        )

        plan_anchor = torch.from_numpy(np.load(plan_anchor_path)).float()
        self.plan_anchor = nn.Parameter(plan_anchor, requires_grad=False)

        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512), nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model),
        )
        diff_decoder_layer = CustomTransformerDecoderLayer(num_poses, d_model, d_ffn, config)
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

    def norm_odo(self, odo):
        x = 2 * (odo[..., 0:1] + 1.2) / 56.9 - 1
        y = 2 * (odo[..., 1:2] + 20) / 46 - 1
        h = 2 * (odo[..., 2:3] + 2) / 3.9 - 1
        return torch.cat([x, y, h], dim=-1)

    def denorm_odo(self, odo):
        x = (odo[..., 0:1] + 1) / 2 * 56.9 - 1.2
        y = (odo[..., 1:2] + 1) / 2 * 46 - 20
        h = (odo[..., 2:3] + 1) / 2 * 3.9 - 2
        return torch.cat([x, y, h], dim=-1)

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets=None, global_img=None):
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature, bev_spatial_shape,
                                      status_encoding, targets, global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature, bev_spatial_shape,
                                     status_encoding, global_img)

    def forward_train(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                      status_encoding, targets, global_img):
        bs = ego_query.shape[0]
        device = ego_query.device

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(0, 50, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(odo_info_fut, noise, timesteps).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(timesteps).view(bs, 1, -1)

        poses_reg_list, poses_cls_list = self.diff_decoder(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
            agents_query, ego_query, time_embed, status_encoding, global_img
        )

        # Simplified loss: L1 regression on best mode
        # Uses mask-based selection instead of torch.gather to avoid scatter_add
        # in backward pass (triggers Neuron NCC_ITEN404 compiler bug).
        ret_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            target_traj = targets["trajectory"]
            for poses_reg, poses_cls in zip(poses_reg_list, poses_cls_list):
                dist = torch.linalg.norm(target_traj.unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
                mode_idx = torch.argmin(dist, dim=-1)
                # Mask-based mode selection (no gather/scatter in backward)
                mode_mask = F.one_hot(mode_idx, num_classes=ego_fut_mode).float()  # (bs, modes)
                per_mode_loss = (poses_reg - target_traj.unsqueeze(1)).abs().mean(dim=(-2, -1))  # (bs, modes)
                ret_loss = ret_loss + (per_mode_loss * mode_mask).sum() / bs

        # Output trajectory (detached, no gradient needed)
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_mask = F.one_hot(mode_idx, num_classes=ego_fut_mode).float()
        best_reg = (poses_reg_list[-1] * mode_mask[:, :, None, None]).sum(dim=1)
        return {"trajectory": best_reg.detach(), "trajectory_loss": ret_loss}

    def forward_test(self, ego_query, agents_query, bev_feature, bev_spatial_shape,
                     status_encoding, global_img=None):
        """Inference with 2-step truncated DDIM denoising (matches paper)."""
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)

        step_ratio = 20 / step_num
        roll_timesteps = torch.arange(0, step_num, dtype=torch.float64)
        roll_timesteps = (roll_timesteps * step_ratio).round().flip(0).long()
        roll_timesteps = roll_timesteps.to(device)

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = img.shape[1]

        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(bs)

            time_embed = self.time_mlp(timesteps).view(bs, 1, -1)
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape,
                agents_query, ego_query, time_embed, status_encoding, global_img
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = self.norm_odo(poses_reg[..., :2])
            img = self.diffusion_scheduler.step(model_output=x_start, timestep=k, sample=img).prev_sample

        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx_expand = mode_idx[:, None, None, None].expand(-1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx_expand).squeeze(1)
        return {"trajectory": best_reg}


# ============================================================================
# Full Model
# ============================================================================
class DiffusionDriveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._query_splits = [1, config.num_bounding_boxes]
        self._backbone = TransfuserBackbone(config)

        bev_grid_h = config.lidar_resolution_height // 32
        bev_grid_w = config.lidar_resolution_width // 32
        self._keyval_embedding = nn.Embedding(bev_grid_h * bev_grid_w + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(config.bev_features_channels, config.bev_features_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.bev_features_channels, config.num_bev_classes, 1, bias=True),
            nn.Upsample(size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                        mode="nearest"),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model, nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn, dropout=config.tf_dropout, batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._agent_head = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn), nn.ReLU(),
            nn.Linear(config.tf_d_ffn, BoundingBox2DIndex.size()),
        )
        self._agent_label_head = nn.Linear(config.tf_d_model, 1)

        self._trajectory_head = TrajectoryHead(
            num_poses=config.num_poses, d_ffn=config.tf_d_ffn, d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path, config=config,
        )
        self.bev_proj = nn.Sequential(*linear_relu_ln(256, 1, 1, 320))

    def forward(self, features, targets=None):
        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        status_feature = features["status_feature"]
        batch_size = status_feature.shape[0]

        # Backbone uses Neuron-safe ops (reshape-mean pooling, nearest interpolation).
        # Only the stem (7x7 conv) needs torch.no_grad() due to NCC_ITEN404 in its backward.
        # All other layers (1-4, GPT fusion, FPN) run with full gradients.
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1).permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.cat([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous().view(
            batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='nearest')
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)
        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous().view(
            batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output = {"bev_semantic_map": bev_semantic_map}

        # Split XLA graph before trajectory head to prevent the full backward graph
        # from exceeding Neuron compiler limits (NCC_ITEN404). Each sub-graph compiles
        # independently. This is the standard Neuron pattern for large models.
        if self.training:
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except ImportError:
                pass  # Running on GPU, no graph splitting needed

        trajectory = self._trajectory_head(
            trajectory_query, agents_query, cross_bev_feature, bev_spatial_shape,
            status_encoding[:, None], targets=targets, global_img=None
        )
        output.update(trajectory)

        agent_states = self._agent_head(agents_query)
        agent_labels = self._agent_label_head(agents_query).squeeze(-1)
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        return output
