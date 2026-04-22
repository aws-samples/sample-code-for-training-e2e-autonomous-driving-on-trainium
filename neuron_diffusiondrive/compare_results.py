"""
Compare training results between trn1 and p4d (A100).
Reads JSON output files from train_navsim_mini.py and produces a comparison table.

Usage: python compare_results.py --trn1 results_trn1.json --gpu results_gpu.json
"""
import argparse
import json


# On-Demand pricing (us-east-1)
PRICING = {
    "trn1": {"instance": "trn1.32xlarge", "price_hr": 21.50, "accelerators": 32, "accel_name": "NeuronCores"},
    "gpu":  {"instance": "p4d.24xlarge",  "price_hr": 21.96, "accelerators": 8,  "accel_name": "A100 GPUs"},
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn1", required=True, help="trn1 results JSON")
    parser.add_argument("--gpu", required=True, help="GPU results JSON")
    parser.add_argument("--output", default="comparison_report.json")
    args = parser.parse_args()

    trn1 = load_results(args.trn1)
    gpu = load_results(args.gpu)

    t = trn1["summary"]
    g = gpu["summary"]

    # Cost calculations
    trn1_price = PRICING["trn1"]
    gpu_price = PRICING["gpu"]

    trn1_cost = t["total_train_time_hr"] * trn1_price["price_hr"]
    gpu_cost = g["total_train_time_hr"] * gpu_price["price_hr"]

    # Instance-level throughput (assuming linear scaling)
    trn1_instance_sps = t["throughput_sps"] * trn1_price["accelerators"]
    gpu_instance_sps = g["throughput_sps"] * gpu_price["accelerators"]

    # Per-accelerator cost (instance price / num accelerators)
    trn1_accel_price_hr = trn1_price["price_hr"] / trn1_price["accelerators"]
    gpu_accel_price_hr = gpu_price["price_hr"] / gpu_price["accelerators"]

    # Cost per 1K samples (using per-accelerator pricing)
    trn1_cost_per_1k = (trn1_accel_price_hr / 3600) / t["throughput_sps"] * 1000 if t["throughput_sps"] > 0 else 0
    gpu_cost_per_1k = (gpu_accel_price_hr / 3600) / g["throughput_sps"] * 1000 if g["throughput_sps"] > 0 else 0

    # Instance-level cost per 1K samples
    trn1_inst_cost_per_1k = (trn1_price["price_hr"] / 3600) / trn1_instance_sps * 1000 if trn1_instance_sps > 0 else 0
    gpu_inst_cost_per_1k = (gpu_price["price_hr"] / 3600) / gpu_instance_sps * 1000 if gpu_instance_sps > 0 else 0

    print("=" * 80)
    print("TRAINING COMPARISON: trn1.32xlarge vs p4d.24xlarge (A100)")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'trn1':>18} {'p4d (A100)':>18} {'Ratio':>10}")
    print("-" * 80)

    # Model config
    print(f"\n--- Model Configuration ---")
    print(f"{'Freeze mode':<35} {t['freeze_mode']:>18} {g['freeze_mode']:>18}")
    print(f"{'Trainable params (M)':<35} {t['trainable_params_M']:>18.1f} {g['trainable_params_M']:>18.1f}")

    # Convergence
    print(f"\n--- Loss Convergence ---")
    print(f"{'Final train loss':<35} {t['final_train_loss']:>18.4f} {g['final_train_loss']:>18.4f} "
          f"{t['final_train_loss']/g['final_train_loss']:>10.2f}x")
    print(f"{'Final traj loss':<35} {t['final_traj_loss']:>18.4f} {g['final_traj_loss']:>18.4f} "
          f"{t['final_traj_loss']/g['final_traj_loss']:>10.2f}x")
    print(f"{'Final BEV loss':<35} {t['final_bev_loss']:>18.4f} {g['final_bev_loss']:>18.4f} "
          f"{t['final_bev_loss']/g['final_bev_loss']:>10.2f}x")
    print(f"{'Best val loss':<35} {t['best_val_loss']:>18.4f} {g['best_val_loss']:>18.4f} "
          f"{t['best_val_loss']/g['best_val_loss']:>10.2f}x")

    # Accuracy
    print(f"\n--- Accuracy (Validation) ---")
    if t.get("final_val_ADE") and g.get("final_val_ADE"):
        print(f"{'Trajectory ADE (m)':<35} {t['final_val_ADE']:>18.3f} {g['final_val_ADE']:>18.3f} "
              f"{t['final_val_ADE']/g['final_val_ADE']:>10.2f}x")
        print(f"{'Trajectory FDE (m)':<35} {t['final_val_FDE']:>18.3f} {g['final_val_FDE']:>18.3f} "
              f"{t['final_val_FDE']/g['final_val_FDE']:>10.2f}x")
        print(f"{'BEV mIoU':<35} {t['final_val_mIoU']:>18.3f} {g['final_val_mIoU']:>18.3f} "
              f"{t['final_val_mIoU']/max(g['final_val_mIoU'], 1e-8):>10.2f}x")

    # Throughput
    print(f"\n--- Training Throughput ---")
    print(f"{'Avg step time (ms)':<35} {t['avg_step_ms']:>18.1f} {g['avg_step_ms']:>18.1f} "
          f"{t['avg_step_ms']/g['avg_step_ms']:>10.2f}x")
    print(f"{'Per-accelerator (sps)':<35} {t['throughput_sps']:>18.2f} {g['throughput_sps']:>18.2f} "
          f"{t['throughput_sps']/g['throughput_sps']:>10.2f}x")
    print(f"{'Accelerators/instance':<35} {trn1_price['accelerators']:>18d} {gpu_price['accelerators']:>18d}")
    print(f"{'Instance throughput (sps)*':<35} {trn1_instance_sps:>18.1f} {gpu_instance_sps:>18.1f} "
          f"{trn1_instance_sps/gpu_instance_sps:>10.2f}x")
    print(f"{'P99/P50 step ratio':<35} {t['p99_p50_ratio']:>18.3f} {g['p99_p50_ratio']:>18.3f}")
    print(f"{'Compilation time (s)':<35} {t['compilation_time_s']:>18.1f} {g['compilation_time_s']:>18.1f}")

    # Cost
    print(f"\n--- Per-Accelerator Cost ---")
    print(f"{'Instance price ($/hr)':<35} {trn1_price['price_hr']:>18.2f} {gpu_price['price_hr']:>18.2f}")
    print(f"{'Per-accelerator price ($/hr)':<35} {trn1_accel_price_hr:>18.3f} {gpu_accel_price_hr:>18.3f}")
    print(f"{'Cost per 1K samples ($)':<35} {trn1_cost_per_1k:>18.4f} {gpu_cost_per_1k:>18.4f} "
          f"{gpu_cost_per_1k/trn1_cost_per_1k:>10.2f}x cheaper")

    print(f"\n--- Instance-Level Cost (linear scaling*) ---")
    print(f"{'Instance throughput (sps)*':<35} {trn1_instance_sps:>18.1f} {gpu_instance_sps:>18.1f} "
          f"{trn1_instance_sps/gpu_instance_sps:>10.2f}x")
    print(f"{'Cost per 1K samples ($)':<35} {trn1_inst_cost_per_1k:>18.4f} {gpu_inst_cost_per_1k:>18.4f} "
          f"{gpu_inst_cost_per_1k/trn1_inst_cost_per_1k:>10.2f}x cheaper")
    print(f"{'Samples per dollar*':<35} {1000/trn1_inst_cost_per_1k:>18.0f} {1000/gpu_inst_cost_per_1k:>18.0f} "
          f"{(1000/trn1_inst_cost_per_1k)/(1000/gpu_inst_cost_per_1k):>10.2f}x")

    print(f"\n*Instance throughput assumes linear scaling across all accelerators.")
    print(f" Single-accelerator numbers measured; multi-accelerator not validated.")

    # Save report
    report = {
        "trn1": t,
        "gpu": g,
        "comparison": {
            "trn1_total_cost": trn1_cost,
            "gpu_total_cost": gpu_cost,
            "cost_ratio": gpu_cost / trn1_cost if trn1_cost > 0 else 0,
            "trn1_instance_sps": trn1_instance_sps,
            "gpu_instance_sps": gpu_instance_sps,
            "throughput_ratio": trn1_instance_sps / gpu_instance_sps if gpu_instance_sps > 0 else 0,
            "loss_parity": abs(t["best_val_loss"] - g["best_val_loss"]) / g["best_val_loss"]
                           if g["best_val_loss"] > 0 else 0,
        },
    }
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
