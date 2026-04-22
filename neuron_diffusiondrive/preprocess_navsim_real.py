"""
Preprocess real NAVSIM mini dataset using the navsim-agents package.
Adapted for the current navsim API (data_path + original_sensor_path).

Usage:
  python3 preprocess_navsim_real.py
"""
import os
import sys

# Must set NUPLAN_MAPS_ROOT BEFORE importing navsim (reads env at import time)
MAPS_ROOT = "/tmp/navsim_mini_download/navsim_data/maps"  # nosec B108 — coordinated with download_and_preprocess.sh
os.environ["NUPLAN_MAPS_ROOT"] = MAPS_ROOT

import numpy as np
import torch
import cv2
from pathlib import Path
from torchvision import transforms
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.enums import LidarIndex

DATA_PATH = Path("/tmp/navsim_mini_download/navsim_data/navsim_logs/mini/mini")  # nosec B108
SENSOR_PATH = Path("/tmp/navsim_mini_download/navsim_data/sensor_blobs/mini/mini")  # nosec B108
OUTPUT_DIR = "/tmp/navsim_real"  # nosec B108
TRAIN_RATIO = 0.83
os.environ["NUPLAN_MAPS_ROOT"] = MAPS_ROOT


def process_scene(scene_loader, token):
    scene = scene_loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()

    # Camera: stitch L+F+R -> (3, 256, 1024)
    cams = agent_input.cameras[-1]
    l0 = cams.cam_l0.image[28:-28, 416:-416]
    f0 = cams.cam_f0.image[28:-28]
    r0 = cams.cam_r0.image[28:-28, 416:-416]
    stitched = np.concatenate([l0, f0, r0], axis=1)
    camera_feature = transforms.ToTensor()(cv2.resize(stitched, (1024, 256)))

    # LiDAR BEV histogram: (1, 256, 256)
    pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
    pc = pc[pc[:, 2] < 100]
    above = pc[pc[:, 2] > -0.5]
    xbins = np.linspace(-32, 32, 257)
    ybins = np.linspace(-32, 32, 257)
    hist = np.histogramdd(above[:, :2], bins=(xbins, ybins))[0]
    hist = np.clip(hist, 0, 5) / 5
    lidar_feature = torch.tensor(np.expand_dims(hist, 0).astype(np.float32))

    # Ego status: (8,)
    ego = agent_input.ego_statuses[-1]
    driving_cmd = np.array(ego.driving_command, dtype=np.float32)
    velocity = np.array(ego.ego_velocity, dtype=np.float32)
    acceleration = np.array(ego.ego_acceleration, dtype=np.float32)
    status = np.concatenate([driving_cmd, velocity, acceleration])
    if len(status) < 8:
        status = np.pad(status, (0, 8 - len(status)))
    status_feature = torch.tensor(status[:8], dtype=torch.float32)

    # Trajectory: (8, 3)
    traj = torch.tensor(
        scene.get_future_trajectory(num_trajectory_frames=8).poses,
        dtype=torch.float32,
    )

    # Agent states: (30, 5)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    boxes = scene.frames[frame_idx].annotations.boxes
    n_agents = min(len(boxes), 30)
    agent_states = torch.zeros(30, 5, dtype=torch.float32)
    agent_labels = torch.zeros(30, dtype=torch.float32)
    for j in range(n_agents):
        agent_states[j, 0] = boxes[j][0]
        agent_states[j, 1] = boxes[j][1]
        agent_states[j, 2] = boxes[j][2]
        agent_states[j, 3] = boxes[j][3]
        agent_states[j, 4] = boxes[j][4]
        agent_labels[j] = 1.0

    # BEV semantic map placeholder: (128, 256)
    bev_semantic_map = torch.zeros(128, 256, dtype=torch.long)

    return {
        "camera_feature": camera_feature,
        "lidar_feature": lidar_feature,
        "status_feature": status_feature,
        "trajectory": traj,
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "bev_semantic_map": bev_semantic_map,
    }


def main():
    print(f"Loading scenes from {DATA_PATH}...")
    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=True, cam_l1=False, cam_l2=False,
        cam_r0=True, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=True,
    )
    scene_filter = SceneFilter(has_route=True)

    scene_loader = SceneLoader(
        data_path=DATA_PATH,
        original_sensor_path=SENSOR_PATH,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    tokens = scene_loader.tokens
    print(f"Found {len(tokens)} scenes")

    np.random.seed(42)
    indices = np.random.permutation(len(tokens))
    split_idx = int(len(tokens) * TRAIN_RATIO)
    splits = {
        "train": [tokens[i] for i in indices[:split_idx]],
        "val": [tokens[i] for i in indices[split_idx:]],
    }

    for split_name, split_tokens in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)
        ok, fail = 0, 0
        for i, token in enumerate(split_tokens):
            try:
                sample = process_scene(scene_loader, token)
                # Note: torch.save uses pickle internally. This is acceptable here
                # because we are saving self-generated tensor data (not loading
                # untrusted files). The corresponding loader uses weights_only=True.
                torch.save(sample, os.path.join(out_dir, f"{token}.pt"))
                ok += 1
                if (i + 1) % 100 == 0:
                    print(f"  [{split_name}] {i+1}/{len(split_tokens)}")
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  Warning: {token}: {e}")
        print(f"  [{split_name}] Done: {ok} saved, {fail} failed")

    train_count = len(os.listdir(os.path.join(OUTPUT_DIR, "train")))
    val_count = len(os.listdir(os.path.join(OUTPUT_DIR, "val")))
    print(f"\nReal NAVSIM data saved to {OUTPUT_DIR}")
    print(f"  Train: {train_count} files")
    print(f"  Val:   {val_count} files")

    import subprocess
    subprocess.run(["/usr/bin/du", "-sh", OUTPUT_DIR], check=False)  # nosec B603 B607 — trusted static args


if __name__ == "__main__":
    main()
