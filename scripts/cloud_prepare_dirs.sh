#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/input/image
mkdir -p data/output/depth_image
mkdir -p data/output/sam3_json
mkdir -p data/output/correct
mkdir -p data/temp
mkdir -p data/visualize/image
mkdir -p data/visualize/depth
mkdir -p data/visualize/image_depth
mkdir -p data/visualize/video

mkdir -p models/Depth-Anything-V2/metric_depth/checkpoints
mkdir -p models/sam3/checkpoints
mkdir -p models/Qwen3-VL/checkpoints

echo "Cloud runtime directories are ready."
