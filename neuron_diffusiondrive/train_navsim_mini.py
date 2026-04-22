"""
Train DiffusionDrive on NAVSIM mini dataset with full comparison metrics.
Works on both Neuron (trn1) and CUDA (p4d/A100).

Usage:
  # On trn1:
  NEURON_CC_FLAGS="--optlevel=1" python -u train_navsim_mini.py \
      --data_dir /tmp/navsim_cached --epochs 50 --freeze_stem

  # On GPU:
  python -u train_navsim_mini.py \
      --data_dir /tmp/navsim_cached --epochs 50 --freeze_stem
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from neuron_diffusiondrive.model_standalone import DiffusionDriveModel, DiffusionDriveConfig


# ============================================================================
# Dataset: loads preprocessed .pt tensor files
# ============================================================================
class NavsimMiniDataset(Dataset):
    """Loads preprocessed NAVSIM mini .pt files."""

    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.samples = sorted([
            f for f in os.listdir(self.data_dir) if f.endswith('.pt')
        ])
        if len(self.samples) == 0:
            raise RuntimeError(f"No .pt files found in {self.data_dir}")
        print(f"  Loaded {len(self.samples)} samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.samples[idx]),
                          map_location='cpu', weights_only=True)
        return data


def collate_fn(batch):
    """Collate list of dicts into batched dict."""
    features = {
        "camera_feature": torch.stack([b["camera_feature"] for b in batch]),
        "lidar_feature": torch.stack([b["lidar_feature"] for b in batch]),
        "status_feature": torch.stack([b["status_feature"] for b in batch]),
    }
    targets = {
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
        "agent_states": torch.stack([b["agent_states"] for b in batch]),
        "agent_labels": torch.stack([b["agent_labels"] for b in batch]),
        "bev_semantic_map": torch.stack([b["bev_semantic_map"] for b in batch]),
    }
    return features, targets


# ============================================================================
# Metrics
# ============================================================================
def compute_trajectory_metrics(pred_traj, gt_traj):
    """Compute ADE, FDE for trajectory predictions.
    pred_traj: (B, num_poses, 3) — x, y, heading
    gt_traj:   (B, num_poses, 3)
    """
    # L2 distance on (x, y)
    displacement = torch.sqrt(
        (pred_traj[..., 0] - gt_traj[..., 0]) ** 2 +
        (pred_traj[..., 1] - gt_traj[..., 1]) ** 2
    )  # (B, num_poses)
    ade = displacement.mean(dim=-1).mean().item()  # avg over timesteps and batch
    fde = displacement[:, -1].mean().item()  # final timestep
    l1 = (pred_traj - gt_traj).abs().mean().item()
    return {"ADE": ade, "FDE": fde, "traj_L1": l1}


def compute_bev_metrics(pred_bev, gt_bev, num_classes=7):
    """Compute per-class and mean IoU for BEV semantic map."""
    pred_labels = pred_bev.argmax(dim=1)  # (B, H, W)
    ious = []
    for c in range(num_classes):
        pred_c = (pred_labels == c)
        gt_c = (gt_bev == c)
        intersection = (pred_c & gt_c).sum().float()
        union = (pred_c | gt_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return {"mIoU": np.mean(ious) if ious else 0.0}


# ============================================================================
# Training loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, device, epoch, is_xla=False):
    model.train()
    total_loss_sum = 0.0
    traj_loss_sum = 0.0
    bev_loss_sum = 0.0
    agent_loss_sum = 0.0
    num_batches = 0
    step_times = []

    for batch_idx, (features, targets) in enumerate(dataloader):
        t0 = time.time()

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        output = model(features, targets=targets)

        # Losses
        traj_loss = output["trajectory_loss"]
        bev_loss = F.cross_entropy(output["bev_semantic_map"], targets["bev_semantic_map"].long())
        agent_loss = F.l1_loss(output["agent_states"], targets["agent_states"])
        agent_cls_loss = F.binary_cross_entropy_with_logits(
            output["agent_labels"], targets["agent_labels"]
        )
        total_loss = traj_loss + bev_loss + agent_loss + agent_cls_loss

        total_loss.backward()

        if is_xla:
            import torch_xla.core.xla_model as xm
            xm.mark_step()

        optimizer.step()

        if is_xla:
            xm.mark_step()

        elapsed = time.time() - t0
        step_times.append(elapsed)
        total_loss_sum += total_loss.item()
        traj_loss_sum += traj_loss.item()
        bev_loss_sum += bev_loss.item()
        agent_loss_sum += agent_loss.item()
        num_batches += 1

    avg_step = np.mean(step_times[1:]) if len(step_times) > 1 else step_times[0]
    return {
        "total_loss": total_loss_sum / num_batches,
        "traj_loss": traj_loss_sum / num_batches,
        "bev_loss": bev_loss_sum / num_batches,
        "agent_loss": agent_loss_sum / num_batches,
        "avg_step_ms": avg_step * 1000,
        "samples_per_sec": 1.0 / avg_step if avg_step > 0 else 0,
        "num_batches": num_batches,
    }


def evaluate(model, dataloader, device, is_xla=False):
    # On Neuron/XLA: run eval using the EXACT same code path as training
    # (forward + backward + mark_step) to reuse already-compiled graphs.
    # Forward-only graphs compile to a slightly different layout that exceeds
    # 16GB HBM by ~13MB. By running backward and discarding gradients, we
    # reuse the training graphs that are known to fit.
    # On GPU: standard no_grad eval for efficiency.
    model.train()
    total_loss_sum = 0.0
    traj_loss_sum = 0.0
    bev_loss_sum = 0.0
    all_ade, all_fde, all_l1 = [], [], []
    all_miou = []
    num_batches = 0

    for features, targets in dataloader:
        features = {k: v.to(device) for k, v in features.items()}
        targets = {k: v.to(device) for k, v in targets.items()}

        if is_xla:
            import torch_xla.core.xla_model as xm
            # Run exact same path as training to reuse compiled graphs
            output = model(features, targets=targets)
            traj_loss = output["trajectory_loss"]
            bev_loss = F.cross_entropy(output["bev_semantic_map"], targets["bev_semantic_map"].long())
            agent_loss = F.l1_loss(output["agent_states"], targets["agent_states"])
            total_loss = traj_loss + bev_loss + agent_loss
            total_loss.backward()
            xm.mark_step()
            # Discard gradients — we only want the predictions
            model.zero_grad()
        else:
            with torch.no_grad():
                output = model(features, targets=targets)
            traj_loss = output["trajectory_loss"]
            bev_loss = F.cross_entropy(output["bev_semantic_map"], targets["bev_semantic_map"].long())
            agent_loss = F.l1_loss(output["agent_states"], targets["agent_states"])
            total_loss = traj_loss + bev_loss + agent_loss

        total_loss_sum += total_loss.item()
        traj_loss_sum += traj_loss.item()
        bev_loss_sum += bev_loss.item()

        # Trajectory metrics (on CPU)
        pred_traj = output["trajectory"].detach().cpu()
        gt_traj = targets["trajectory"].cpu()
        metrics = compute_trajectory_metrics(pred_traj, gt_traj)
        all_ade.append(metrics["ADE"])
        all_fde.append(metrics["FDE"])
        all_l1.append(metrics["traj_L1"])

        # BEV metrics (on CPU)
        pred_bev = output["bev_semantic_map"].detach().cpu()
        gt_bev = targets["bev_semantic_map"].cpu()
        bev_m = compute_bev_metrics(pred_bev, gt_bev)
        all_miou.append(bev_m["mIoU"])

        num_batches += 1

    return {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "traj_loss": traj_loss_sum / max(num_batches, 1),
        "bev_loss": bev_loss_sum / max(num_batches, 1),
        "ADE": np.mean(all_ade),
        "FDE": np.mean(all_fde),
        "traj_L1": np.mean(all_l1),
        "mIoU": np.mean(all_miou),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to preprocessed .pt data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_stem", action="store_true",
                        help="Freeze only conv1+bn1 stem (for Neuron compatibility)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze entire backbone (old approach, 8.7M trainable)")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--output", default="training_results.json")
    args = parser.parse_args()

    # Detect accelerator
    is_xla = False
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_xla = True
        accelerator = "trn1"
        print(f"Accelerator: Trainium (XLA), device: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Accelerator: GPU ({gpu_name}), device: {device}")
        else:
            print(f"Accelerator: CPU, device: {device}")

    # Model setup
    config = DiffusionDriveConfig()
    plan_anchor_path = os.path.join(os.path.dirname(__file__), 'plan_anchors.npy')
    if not os.path.exists(plan_anchor_path):
        np.save(plan_anchor_path, np.random.RandomState(42).randn(20, 8, 2).astype(np.float32) * 5.0)
    config.plan_anchor_path = plan_anchor_path

    model = DiffusionDriveModel(config)

    # Freeze strategy
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if '_backbone.' in name:
                param.requires_grad = False
        freeze_mode = "backbone"
    elif args.freeze_stem:
        for encoder in [model._backbone.image_encoder, model._backbone.lidar_encoder]:
            for attr in ['conv1', 'bn1']:
                module = getattr(encoder, attr, None)
                if module:
                    for p in module.parameters():
                        p.requires_grad = False
        freeze_mode = "stem"
    else:
        freeze_mode = "none"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Freeze mode: {freeze_mode}")
    print(f"Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable, "
          f"{frozen_params/1e3:.1f}K frozen")

    model = model.to(device)

    # Data
    train_dataset = NavsimMiniDataset(args.data_dir, split="train")
    val_dataset = NavsimMiniDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    # Training
    results = {
        "accelerator": accelerator,
        "freeze_mode": freeze_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": [],
    }

    total_train_time = 0.0
    best_val_loss = float('inf')

    print(f"\n{'='*70}")
    print(f"Training: {args.epochs} epochs, {len(train_dataset)} train / {len(val_dataset)} val samples")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, is_xla)
        epoch_time = time.time() - t_epoch
        total_train_time += epoch_time

        epoch_data = {
            "epoch": epoch,
            "train": train_metrics,
            "epoch_time_s": epoch_time,
        }

        log = (f"Epoch {epoch:3d} | loss={train_metrics['total_loss']:.4f} "
               f"traj={train_metrics['traj_loss']:.4f} "
               f"bev={train_metrics['bev_loss']:.4f} | "
               f"{train_metrics['avg_step_ms']:.1f}ms/step "
               f"{train_metrics['samples_per_sec']:.2f}sps")

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, is_xla)
            epoch_data["val"] = val_metrics
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
            log += (f" | val_loss={val_metrics['total_loss']:.4f} "
                    f"ADE={val_metrics['ADE']:.3f} FDE={val_metrics['FDE']:.3f} "
                    f"mIoU={val_metrics['mIoU']:.3f}")

        print(log)
        results["epochs"].append(epoch_data)

    # Summary
    final_train = results["epochs"][-1]["train"]
    final_val = results["epochs"][-1].get("val", {})

    # Step time stats (exclude first epoch which includes compilation)
    step_times = [e["train"]["avg_step_ms"] for e in results["epochs"][1:]]
    p50_step = np.percentile(step_times, 50) if step_times else 0
    p99_step = np.percentile(step_times, 99) if step_times else 0

    # Memory
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_mem_gb = 0  # XLA doesn't expose this easily

    summary = {
        "accelerator": accelerator,
        "freeze_mode": freeze_mode,
        "trainable_params_M": trainable_params / 1e6,
        "total_train_time_s": total_train_time,
        "total_train_time_hr": total_train_time / 3600,
        "final_train_loss": final_train["total_loss"],
        "final_traj_loss": final_train["traj_loss"],
        "final_bev_loss": final_train["bev_loss"],
        "best_val_loss": best_val_loss,
        "final_val_ADE": final_val.get("ADE", None),
        "final_val_FDE": final_val.get("FDE", None),
        "final_val_mIoU": final_val.get("mIoU", None),
        "avg_step_ms": np.mean(step_times) if step_times else 0,
        "p50_step_ms": p50_step,
        "p99_step_ms": p99_step,
        "p99_p50_ratio": p99_step / p50_step if p50_step > 0 else 0,
        "throughput_sps": 1000.0 / np.mean(step_times) if step_times else 0,
        "peak_memory_gb": peak_mem_gb,
        "compilation_time_s": results["epochs"][0]["epoch_time_s"] if results["epochs"] else 0,
    }
    results["summary"] = summary

    print(f"\n{'='*70}")
    print(f"SUMMARY — {accelerator} ({freeze_mode} frozen)")
    print(f"{'='*70}")
    print(f"  Trainable params:  {summary['trainable_params_M']:.1f}M")
    print(f"  Total train time:  {summary['total_train_time_hr']:.2f} hr")
    print(f"  Final train loss:  {summary['final_train_loss']:.4f}")
    print(f"  Best val loss:     {summary['best_val_loss']:.4f}")
    if summary['final_val_ADE'] is not None:
        print(f"  Val ADE:           {summary['final_val_ADE']:.3f}")
        print(f"  Val FDE:           {summary['final_val_FDE']:.3f}")
        print(f"  Val mIoU:          {summary['final_val_mIoU']:.3f}")
    print(f"  Avg step time:     {summary['avg_step_ms']:.1f}ms")
    print(f"  P99/P50 ratio:     {summary['p99_p50_ratio']:.3f}")
    print(f"  Throughput:        {summary['throughput_sps']:.2f} samples/sec")
    print(f"  Peak memory:       {summary['peak_memory_gb']:.2f} GB")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
