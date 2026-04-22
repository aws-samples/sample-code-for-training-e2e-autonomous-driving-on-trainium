#!/bin/bash
# =============================================================================
# One-Click Deploy: DiffusionDrive Training on AWS Trainium
# =============================================================================
# Usage:
#   ./infra/deploy.sh                    # Deploy with defaults
#   ./infra/deploy.sh --stack-name foo   # Custom stack name
#   ./infra/deploy.sh --delete           # Delete stack
#   ./infra/deploy.sh --status           # Check stack status
#   ./infra/deploy.sh --results          # Fetch benchmark results
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/trn1-diffusiondrive-stack.yaml"

AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="${STACK_NAME:-diffusiondrive-trainium}"
VPC_ID="${VPC_ID:-}"
SUBNET_ID="${SUBNET_ID:-}"
TTL_HOURS="${TTL_HOURS:-4}"
INSTANCE_TYPE="${INSTANCE_TYPE:-trn1.32xlarge}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

cmd_deploy() {
    if [ -z "${VPC_ID}" ] || [ -z "${SUBNET_ID}" ]; then
        log_error "VPC_ID and SUBNET_ID are required."
        log_error "Set via environment variables or --vpc / --subnet flags."
        log_error "  Example: VPC_ID=vpc-xxx SUBNET_ID=subnet-yyy ./deploy.sh deploy"
        log_error "  Or: ./deploy.sh deploy --vpc vpc-xxx --subnet subnet-yyy"
        log_error ""
        log_error "Find your default VPC:"
        log_error "  aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text"
        log_error "Find a subnet:"
        log_error "  aws ec2 describe-subnets --filters Name=vpc-id,Values=\$VPC_ID --query 'Subnets[0].SubnetId' --output text"
        exit 1
    fi

    if [ -z "${REPO_URL:-}" ]; then
        log_error "REPO_URL is required. Export it with the Git URL of your repository."
        log_error "  Example: export REPO_URL=https://github.com/YOUR-ORG/YOUR-REPO.git"
        exit 1
    fi

    log_info "Validating AWS CloudFormation template..."
    aws cloudformation validate-template \
        --template-body "file://${TEMPLATE}" \
        --region "${AWS_REGION}" > /dev/null

    log_ok "Template valid"
    log_info "Deploying stack: ${STACK_NAME}"
    log_info "Instance type: ${INSTANCE_TYPE}"
    log_info "Region: ${AWS_REGION}"
    log_info "VPC: ${VPC_ID} / Subnet: ${SUBNET_ID}"
    log_info "TTL: ${TTL_HOURS} hours"
    echo ""

    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}" \
        --template-body "file://${TEMPLATE}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "${AWS_REGION}" \
        --parameters \
            "ParameterKey=InstanceType,ParameterValue=${INSTANCE_TYPE}" \
            "ParameterKey=VpcId,ParameterValue=${VPC_ID}" \
            "ParameterKey=SubnetId,ParameterValue=${SUBNET_ID}" \
            "ParameterKey=TTLHours,ParameterValue=${TTL_HOURS}" \
            "ParameterKey=GitRepoUrl,ParameterValue=${REPO_URL}" \
            "ParameterKey=RunBenchmark,ParameterValue=true" \
        --tags \
            "Key=Project,Value=end-2end-AI" \
            "Key=Owner,Value=$(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null || echo 'unknown')"

    log_ok "Stack creation initiated: ${STACK_NAME}"
    log_info "Monitor: aws cloudformation describe-stack-events --stack-name ${STACK_NAME} --region ${AWS_REGION}"
    log_info ""
    log_info "Waiting for stack to complete..."

    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" 2>/dev/null &
    WAIT_PID=$!

    # Poll status while waiting (ps -p returns zero while the background job exists)
    while ps -p $WAIT_PID > /dev/null 2>&1; do
        STATUS=$(aws cloudformation describe-stacks \
            --stack-name "${STACK_NAME}" \
            --region "${AWS_REGION}" \
            --query 'Stacks[0].StackStatus' \
            --output text 2>/dev/null || echo "CHECKING")
        echo -ne "\r  Status: ${STATUS}   "
        sleep 15
    done
    echo ""

    # Get outputs
    OUTPUTS=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs' \
        --output json 2>/dev/null)

    echo ""
    log_ok "Stack deployed successfully!"
    echo ""
    echo "=== Stack Outputs ==="
    echo "${OUTPUTS}" | python3 -c "
import sys, json
outputs = json.load(sys.stdin)
for o in outputs:
    print(f\"  {o['OutputKey']}: {o['OutputValue']}\")
" 2>/dev/null || echo "${OUTPUTS}"

    echo ""
    INSTANCE_ID=$(echo "${OUTPUTS}" | python3 -c "
import sys, json
for o in json.load(sys.stdin):
    if o['OutputKey'] == 'InstanceId': print(o['OutputValue'])
" 2>/dev/null)
    log_info "Connect via SSM: aws ssm start-session --target ${INSTANCE_ID} --region ${AWS_REGION}"
    log_info "Benchmark results will be in S3 once UserData completes (~15-30 min)"
}

cmd_status() {
    log_info "Stack status: ${STACK_NAME}"
    aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].{Status:StackStatus,Created:CreationTime,Outputs:Outputs[*].{Key:OutputKey,Value:OutputValue}}' \
        --output json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Status: {data['Status']}\")
print(f\"  Created: {data['Created']}\")
if data.get('Outputs'):
    print('  Outputs:')
    for o in data['Outputs']:
        print(f\"    {o['Key']}: {o['Value']}\")
" 2>/dev/null

    # Check if benchmark is running
    INSTANCE_ID=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
        --output text 2>/dev/null)

    if [ -n "${INSTANCE_ID}" ] && [ "${INSTANCE_ID}" != "None" ]; then
        STATE=$(aws ec2 describe-instances --instance-ids "${INSTANCE_ID}" \
            --region "${AWS_REGION}" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text 2>/dev/null || echo "unknown")
        log_info "Instance ${INSTANCE_ID} state: ${STATE}"
    fi
}

cmd_results() {
    BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs[?OutputKey==`ArtifactBucketName`].OutputValue' \
        --output text 2>/dev/null)

    if [ -z "${BUCKET}" ] || [ "${BUCKET}" = "None" ]; then
        log_error "Could not find S3 bucket. Is the stack deployed?"
        exit 1
    fi

    log_info "Fetching results from s3://${BUCKET}/benchmarks/"
    aws s3 ls "s3://${BUCKET}/benchmarks/" --region "${AWS_REGION}" 2>/dev/null || {
        log_warn "No benchmark results yet. UserData may still be running."
        exit 0
    }

    # Download latest results JSON
    LATEST=$(aws s3 ls "s3://${BUCKET}/benchmarks/" --region "${AWS_REGION}" 2>/dev/null | grep results_ | sort | tail -1 | awk '{print $4}')
    if [ -n "${LATEST}" ]; then
        log_info "Downloading: ${LATEST}"
        aws s3 cp "s3://${BUCKET}/benchmarks/${LATEST}" /tmp/benchmark_results.json --region "${AWS_REGION}" > /dev/null
        echo ""
        echo "=== Benchmark Results ==="
        python3 -c "
import json
with open('/tmp/benchmark_results.json') as f:
    data = json.load(f)
print(f\"Instance: {data.get('instance_type', 'unknown')}\")
print(f\"Timestamp: {data.get('timestamp', 'unknown')}\")
print(f\"SDK: {json.dumps(data.get('neuron_sdk', {}))}\")
print()
for bench_name, bench_data in data.get('benchmarks', {}).items():
    print(f'--- {bench_name} ---')
    if 'error' in bench_data:
        print(f'  ERROR: {bench_data[\"error\"]}')
    elif isinstance(bench_data, dict):
        for k, v in bench_data.items():
            if isinstance(v, dict):
                status = v.get('status', '?')
                fwd_min = v.get('forward_min_ms', '?')
                comp = v.get('compile_time_s', '?')
                print(f'  {k}: {status} | compile={comp}s | fwd_min={fwd_min}ms')
            else:
                print(f'  {k}: {v}')
    print()
" 2>/dev/null || cat /tmp/benchmark_results.json
    fi

    # Also show latest log
    LATEST_LOG=$(aws s3 ls "s3://${BUCKET}/benchmarks/" --region "${AWS_REGION}" 2>/dev/null | grep benchmark_ | grep .log | sort | tail -1 | awk '{print $4}')
    if [ -n "${LATEST_LOG}" ]; then
        echo ""
        log_info "Latest log: s3://${BUCKET}/benchmarks/${LATEST_LOG}"
        echo "  Download: aws s3 cp s3://${BUCKET}/benchmarks/${LATEST_LOG} ./benchmark.log --region ${AWS_REGION}"
    fi
}

cmd_delete() {
    log_warn "Deleting stack: ${STACK_NAME}"
    read -p "Are you sure? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cancelled"
        exit 0
    fi

    aws cloudformation delete-stack \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}"
    log_ok "Stack deletion initiated"
    log_info "Note: S3 bucket is retained (DeletionPolicy: Retain). Delete manually if needed."
}

cmd_connect() {
    INSTANCE_ID=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
        --output text 2>/dev/null)

    if [ -z "${INSTANCE_ID}" ] || [ "${INSTANCE_ID}" = "None" ]; then
        log_error "Could not find instance. Is the stack deployed?"
        exit 1
    fi

    log_info "Connecting to ${INSTANCE_ID} via SSM..."
    aws ssm start-session --target "${INSTANCE_ID}" --region "${AWS_REGION}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local command="${1:-deploy}"
    shift || true

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stack-name) STACK_NAME="$2"; shift 2 ;;
            --region) AWS_REGION="$2"; shift 2 ;;
            --vpc) VPC_ID="$2"; shift 2 ;;
            --subnet) SUBNET_ID="$2"; shift 2 ;;
            --ttl) TTL_HOURS="$2"; shift 2 ;;
            --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
            --delete) command="delete"; shift ;;
            --status) command="status"; shift ;;
            --results) command="results"; shift ;;
            --connect) command="connect"; shift ;;
            *) shift ;;
        esac
    done

    case "${command}" in
        deploy)   cmd_deploy ;;
        status)   cmd_status ;;
        results)  cmd_results ;;
        delete)   cmd_delete ;;
        connect)  cmd_connect ;;
        *)
            echo "Usage: $0 [deploy|--status|--results|--delete|--connect] [options]"
            echo ""
            echo "Options:"
            echo "  --stack-name NAME    CloudFormation stack name (default: diffusiondrive-trainium)"
            echo "  --region REGION      AWS region (default: us-east-1)"
            echo "  --vpc VPC_ID         VPC ID"
            echo "  --subnet SUBNET_ID   Subnet ID"
            echo "  --ttl HOURS          Auto-terminate after N hours (default: 4)"
            echo "  --instance-type TYPE  Instance type (default: trn1.32xlarge)"
            exit 1
            ;;
    esac
}

main "$@"
