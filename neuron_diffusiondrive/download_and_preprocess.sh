#!/bin/bash
# Download NAVSIM mini data, preprocess to .pt files
# Usage: bash download_and_preprocess.sh
#
# NOTE: The URLs below point to the canonical, publicly hosted NAVSIM/OpenScene
# and nuPlan maps datasets. They are required for this workflow and are not
# illustrative placeholders.
set -ex

DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/navsim_mini_download}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/navsim_real}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Metadata
echo "=== Downloading metadata ==="
if [ ! -d "navsim_data/navsim_logs/mini" ]; then
    wget -q https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
    tar -xzf openscene_metadata_mini.tgz && rm openscene_metadata_mini.tgz
fi

# Camera data (32 shards)
echo "=== Downloading camera data ==="
for i in $(seq 0 31); do
    if [ ! -f ".camera_${i}_done" ]; then
        wget -q "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_${i}.tgz"
        tar -xzf "openscene_sensor_mini_camera_${i}.tgz"
        rm "openscene_sensor_mini_camera_${i}.tgz"
        touch ".camera_${i}_done"
        echo "camera shard $i done"
    fi
done

# LiDAR data (32 shards)
echo "=== Downloading LiDAR data ==="
for i in $(seq 0 31); do
    if [ ! -f ".lidar_${i}_done" ]; then
        wget -q "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_lidar/openscene_sensor_mini_lidar_${i}.tgz"
        tar -xzf "openscene_sensor_mini_lidar_${i}.tgz"
        rm "openscene_sensor_mini_lidar_${i}.tgz"
        touch ".lidar_${i}_done"
        echo "lidar shard $i done"
    fi
done

# Maps
echo "=== Downloading maps ==="
if [ ! -d "navsim_data/maps" ] && [ ! -d "nuplan-maps-v1.0" ]; then
    wget -q https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    unzip -q nuplan-maps-v1.1.zip
    rm nuplan-maps-v1.1.zip
fi

# Organize directory structure
echo "=== Organizing ==="
mkdir -p navsim_data/navsim_logs navsim_data/sensor_blobs
[ -d "openscene_v1.1/meta_datas" ] && mv openscene_v1.1/meta_datas navsim_data/navsim_logs/mini && rm -rf openscene_v1.1
[ -d "openscene-v1.1/sensor_blobs" ] && mv openscene-v1.1/sensor_blobs navsim_data/sensor_blobs/mini && rm -rf openscene-v1.1
[ -d "nuplan-maps-v1.0" ] && mv nuplan-maps-v1.0 navsim_data/maps

echo "=== Directory structure ==="
find navsim_data -maxdepth 3 -type d | head -20
du -sh navsim_data/

# Preprocess
echo "=== Preprocessing ==="
cd "$CODE_DIR"
OPENSCENE_DATA_ROOT="$DOWNLOAD_DIR/navsim_data" python3 -u neuron_diffusiondrive/preprocess_navsim_real.py 2>&1 | tee /tmp/preprocess_output.txt

echo "=== Results ==="
du -sh "$OUTPUT_DIR/"
echo "Train files: $(ls "$OUTPUT_DIR/train/" | wc -l)"
echo "Val files: $(ls "$OUTPUT_DIR/val/" | wc -l)"
