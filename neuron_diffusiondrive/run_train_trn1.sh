#!/bin/bash
# Run trn1 training. Usage: bash neuron_diffusiondrive/run_train_trn1.sh
#
# Honors these environment variables:
#   PROJECT_DIR  - repo root (default: auto-detect; fallback /home/ubuntu/end-2end-AI)
#   DATA_DIR     - preprocessed NAVSIM mini output (default: /tmp/navsim_real)
#   EPOCHS       - number of epochs (default: 50)
#   BATCH_SIZE   - per-accelerator batch size (default: 1)
#   S3_BUCKET    - optional; if set, results are uploaded there
set -e

# Locate the project root (parent of this script's directory)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-/tmp/navsim_real}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"

source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
rm -rf /var/tmp/neuron-compile-cache/

cd "$PROJECT_DIR"
NEURON_CC_FLAGS="--optlevel=1" python3 -u neuron_diffusiondrive/train_navsim_mini.py \
    --data_dir "$DATA_DIR" --epochs "$EPOCHS" --freeze_stem --eval_every 5 \
    --batch_size "$BATCH_SIZE" --output training_results_trn1_real.json \
    2>&1 | tee /tmp/train_trn1_real_output.txt

if [ -n "${S3_BUCKET:-}" ]; then
    aws s3 cp "$PROJECT_DIR/neuron_diffusiondrive/training_results_trn1_real.json" \
        "s3://$S3_BUCKET/training_results_trn1_real.json" 2>/dev/null || true
    aws s3 cp /tmp/train_trn1_real_output.txt \
        "s3://$S3_BUCKET/train_trn1_real_output.txt" 2>/dev/null || true
fi
