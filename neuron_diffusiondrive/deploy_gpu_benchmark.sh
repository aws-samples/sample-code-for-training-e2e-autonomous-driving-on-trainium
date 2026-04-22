#!/bin/bash
# Deploy and run DiffusionDrive GPU benchmark on p4d instance via SSM
# Usage: ./deploy_gpu_benchmark.sh <instance-id>
set -e

INSTANCE_ID="${1:?Usage: $0 <instance-id>}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="${S3_BUCKET:?Set S3_BUCKET environment variable}"
REMOTE_DIR="/tmp/dd_bench"

echo "=== Packaging files ==="
cd "$PROJECT_DIR"
tar czf /tmp/dd_bench_gpu.tar.gz \
    neuron_diffusiondrive/__init__.py \
    neuron_diffusiondrive/blocks_neuron.py \
    neuron_diffusiondrive/model_standalone.py \
    neuron_diffusiondrive/benchmark_gpu.py

echo "=== Uploading to S3 ==="
aws s3 cp /tmp/dd_bench_gpu.tar.gz "s3://$BUCKET/dd_bench_gpu.tar.gz"

echo "=== Sending command to $INSTANCE_ID ==="
CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[
        'set -ex',
        'aws s3 cp s3://$BUCKET/dd_bench_gpu.tar.gz /tmp/dd_bench_gpu.tar.gz',
        'rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR',
        'cd $REMOTE_DIR && tar xzf /tmp/dd_bench_gpu.tar.gz',
        'pip install -q timm diffusers einops 2>&1 | tail -3',
        'cd $REMOTE_DIR && python -u neuron_diffusiondrive/benchmark_gpu.py --freeze_backbone --batch_size 1 --steps 100 --inference_steps 200 2>&1 | tee /tmp/dd_gpu_output.txt',
        'aws s3 cp /tmp/dd_gpu_output.txt s3://$BUCKET/dd_gpu_output.txt 2>&1 || true',
        'aws s3 cp $REMOTE_DIR/neuron_diffusiondrive/benchmark_results_gpu.json s3://$BUCKET/benchmark_results_gpu.json 2>&1 || true'
    ]" \
    --timeout-seconds 3600 \
    --output text \
    --query "Command.CommandId")

echo ""
echo "Command ID: $CMD_ID"
echo ""
echo "=== Monitor: ==="
echo "aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID --query 'Status' --output text"
echo ""
echo "=== Results: ==="
echo "aws s3 cp s3://$BUCKET/dd_gpu_output.txt -"
echo "aws s3 cp s3://$BUCKET/benchmark_results_gpu.json -"
