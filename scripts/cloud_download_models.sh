#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-compass}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

cd "$PROJECT_ROOT"

source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p models/Depth-Anything-V2/metric_depth/checkpoints
mkdir -p models/sam3/checkpoints
mkdir -p models/Qwen3-VL/checkpoints

pushd models/Depth-Anything-V2/metric_depth/checkpoints >/dev/null
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
wget -nc https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
popd >/dev/null

python -m modelscope download --model facebook/sam3 --local_dir "$PROJECT_ROOT/models/sam3/checkpoints"

pushd models/Qwen3-VL/checkpoints >/dev/null
if [ ! -d Qwen3-VL-8B-Instruct ]; then
    git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
fi
pushd Qwen3-VL-8B-Instruct >/dev/null
git lfs pull || true
popd >/dev/null
popd >/dev/null

echo "Model downloads completed."
