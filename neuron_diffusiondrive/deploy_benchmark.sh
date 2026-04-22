#!/bin/bash
# Deploy and run DiffusionDrive benchmark on trn1 instance via SSM
# Usage: ./deploy_benchmark.sh <instance-id> [extra args for benchmark]
#
# Example:
#   ./deploy_benchmark.sh i-0d6c8037359ff5e67
#   ./deploy_benchmark.sh i-0d6c8037359ff5e67 --batch_size 2 --no_freeze_backbone
set -e

INSTANCE_ID="${1:?Usage: $0 <instance-id> [extra benchmark args]}"
shift
EXTRA_ARGS="${@:---freeze_backbone --batch_size 1 --steps 20}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="${S3_BUCKET:?Set S3_BUCKET environment variable}"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_8/bin/activate"
REMOTE_DIR="/tmp/dd_bench"

echo "=== Packaging files ==="
cd "$PROJECT_DIR"
tar czf /tmp/dd_bench.tar.gz \
    neuron_diffusiondrive/__init__.py \
    neuron_diffusiondrive/blocks_neuron.py \
    neuron_diffusiondrive/model_standalone.py \
    neuron_diffusiondrive/benchmark_neuron.py \
    neuron_diffusiondrive/benchmark_inference.py \
    neuron_diffusiondrive/verify_grid_sample.py

echo "=== Uploading to S3 ==="
aws s3 cp /tmp/dd_bench.tar.gz "s3://$BUCKET/dd_bench.tar.gz"

echo "=== Sending command to $INSTANCE_ID ==="
CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[
        'set -ex',
        'aws s3 cp s3://$BUCKET/dd_bench.tar.gz /tmp/dd_bench.tar.gz',
        'rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR',
        'cd $REMOTE_DIR && tar xzf /tmp/dd_bench.tar.gz',
        'rm -rf /var/tmp/neuron-compile-cache/',
        'bash -c \"source $NEURON_VENV && pip install -q timm diffusers einops 2>&1 | tail -3\"',
        'bash -c \"source $NEURON_VENV && cd $REMOTE_DIR && NEURON_CC_FLAGS=\\\"--optlevel=1\\\" python -u neuron_diffusiondrive/benchmark_neuron.py $EXTRA_ARGS 2>&1 | tee /tmp/dd_train_output.txt\"',
        'bash -c \"aws s3 cp /tmp/dd_train_output.txt s3://$BUCKET/dd_train_output.txt 2>&1 || true\"',
        'bash -c \"aws s3 cp $REMOTE_DIR/neuron_diffusiondrive/benchmark_results.json s3://$BUCKET/benchmark_results_diffusiondrive.json 2>&1 || true\"',
        'bash -c \"source $NEURON_VENV && cd $REMOTE_DIR && NEURON_CC_FLAGS=\\\"--optlevel=1\\\" python -u neuron_diffusiondrive/benchmark_inference.py --reduced_resolution --steps 50 2>&1 | tee /tmp/dd_infer_output.txt\"',
        'bash -c \"aws s3 cp /tmp/dd_infer_output.txt s3://$BUCKET/dd_infer_output.txt 2>&1 || true\"',
        'bash -c \"aws s3 cp $REMOTE_DIR/neuron_diffusiondrive/benchmark_results_inference.json s3://$BUCKET/benchmark_results_inference.json 2>&1 || true\"'
    ]" \
    --timeout-seconds 7200 \
    --output text \
    --query "Command.CommandId")

echo ""
echo "Command ID: $CMD_ID"
echo ""
echo "=== Monitor with: ==="
echo "aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID --query 'Status' --output text"
echo ""
echo "=== Get output: ==="
echo "aws ssm get-command-invocation --command-id $CMD_ID --instance-id $INSTANCE_ID --query 'StandardOutputContent' --output text"
echo ""
echo "=== Get results: ==="
echo "aws s3 cp s3://$BUCKET/dd_train_output.txt -"
echo "aws s3 cp s3://$BUCKET/dd_infer_output.txt -"
echo "aws s3 cp s3://$BUCKET/benchmark_results_diffusiondrive.json -"
echo "aws s3 cp s3://$BUCKET/benchmark_results_inference.json -"
