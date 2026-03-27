#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-compass}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

cd "$PROJECT_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found" >&2
    exit 1
fi

source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import numpy, sam3; print(numpy.__version__); print('sam3 import ok')"
python -c "import cv2, yaml, tifffile, transformers; print('basic imports ok')"
python -c "from src.pipeline.pipeline_manager import PipelineManager; print('pipeline import ok')"
python -c "from pathlib import Path; paths = [Path('models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth'), Path('models/sam3/checkpoints/sam3.pt'), Path('models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct')]; print([(str(p), p.exists()) for p in paths])"
