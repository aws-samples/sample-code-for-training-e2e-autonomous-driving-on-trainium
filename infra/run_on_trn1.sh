#!/bin/bash
# =============================================================================
# Run on trn1: Deploy code, install deps, run DiffusionDrive benchmark
# Designed to be uploaded to S3 and run via SSM
# =============================================================================
set -euo pipefail
exec > >(tee /home/ubuntu/benchmark.log) 2>&1

S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

echo "============================================================"
echo "  DiffusionDrive Neuron Benchmark Runner"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Instance: $(curl -s -H 'X-aws-ec2-metadata-token: '$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60') http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown)"
echo "============================================================"

# 1. Deploy code from S3 or git
echo ""
echo "=== Deploying DiffusionDrive code ==="
cd /home/ubuntu
if [ -n "${S3_BUCKET}" ]; then
    mkdir -p end-2end-AI
    aws s3 cp "s3://${S3_BUCKET}/deploy/code.tar.gz" /tmp/code.tar.gz --region "${AWS_REGION}"
    tar xzf /tmp/code.tar.gz -C end-2end-AI/
    rm -f /tmp/code.tar.gz
    chown -R ubuntu:ubuntu end-2end-AI
    echo "Code deployed from S3"
elif [ -d "end-2end-AI" ]; then
    echo "Project directory exists, using existing code"
else
    echo "WARNING: No code found. Set REPO_URL and clone your repository first:"
    echo "  git clone \$REPO_URL end-2end-AI"
    exit 1
fi

# 2. Find Neuron venv
echo ""
echo "=== Finding Neuron Virtual Environment ==="
NEURON_VENV=""
for venv in /opt/aws_neuronx_venv_pytorch_2_8/bin/activate \
            /opt/aws_neuronx_venv_pytorch_2_5/bin/activate \
            /opt/aws_neuronx_venv_pytorch/bin/activate; do
    if [ -f "$venv" ]; then
        NEURON_VENV="$venv"
        break
    fi
done

if [ -z "$NEURON_VENV" ]; then
    echo "ERROR: No Neuron virtual environment found"
    exit 1
fi
echo "Using: ${NEURON_VENV}"
source "${NEURON_VENV}"

# 3. Install dependencies
echo ""
echo "=== Installing dependencies ==="
pip install -q timm diffusers 2>&1 | tail -3

# 4. System info
echo ""
echo "=== System Info ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch_xla; print(f'torch_xla: {torch_xla.__version__}')" 2>/dev/null || echo "torch_xla: not found"
python3 -c "import torch_neuronx; print(f'torch_neuronx: {torch_neuronx.__version__}')" 2>/dev/null || echo "torch_neuronx: not found"
neuron-ls 2>/dev/null | head -10 || echo "neuron-ls: not available"

# 5. Set Neuron env
export NEURON_COMPILE_CACHE_URL="/home/ubuntu/neuron_cache"
export NEURON_RT_NUM_CORES=2
export NEURON_CC_FLAGS="--optlevel=1"
export MALLOC_ARENA_MAX=64
mkdir -p /home/ubuntu/neuron_cache

# 6. Run DiffusionDrive training benchmark
echo ""
echo "=== Running DiffusionDrive Training Benchmark ==="
cd /home/ubuntu/end-2end-AI/neuron_diffusiondrive
python3 benchmark_neuron.py --freeze_backbone --steps 20

# 7. Upload results
echo ""
echo "=== Uploading results ==="
TS=$(date -u +%Y%m%d_%H%M%S)
if [ -n "${S3_BUCKET}" ]; then
    aws s3 cp benchmark_results.json "s3://${S3_BUCKET}/benchmarks/diffusiondrive_results_${TS}.json" --region "${AWS_REGION}" 2>/dev/null || echo "S3 upload of results failed"
    aws s3 cp /home/ubuntu/benchmark.log "s3://${S3_BUCKET}/benchmarks/benchmark_${TS}.log" --region "${AWS_REGION}" 2>/dev/null || echo "S3 upload of log failed"
    echo "Uploaded to s3://${S3_BUCKET}/benchmarks/"
fi

echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
