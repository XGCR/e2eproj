#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-compass}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

cd "$PROJECT_ROOT"

# RunPod 容器默认以 root 运行，不需要 sudo
apt-get update || true
apt-get install -y git git-lfs wget curl ca-certificates build-essential pkg-config || true
git lfs install

if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    TMP_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    wget -O "$TMP_INSTALLER" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$TMP_INSTALLER" -b -p "$CONDA_HOME"
fi

source "$CONDA_HOME/etc/profile.d/conda.sh"
conda init bash >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

mkdir -p "$PROJECT_ROOT/models"

if [ ! -d "$PROJECT_ROOT/models/Depth-Anything-V2/.git" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2 "$PROJECT_ROOT/models/Depth-Anything-V2"
fi

if [ ! -d "$PROJECT_ROOT/models/sam3/.git" ]; then
    git clone https://github.com/facebookresearch/sam3.git "$PROJECT_ROOT/models/sam3"
fi

if [ ! -d "$PROJECT_ROOT/models/Qwen3-VL/.git" ]; then
    git clone https://github.com/QwenLM/Qwen3-VL.git "$PROJECT_ROOT/models/Qwen3-VL"
fi

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy>=1.26,<2"
python -m pip install opencv-python pillow pyyaml tifffile einops decord pycocotools transformers accelerate sentencepiece modelscope

if [ -d "$PROJECT_ROOT/models/Qwen3-VL" ]; then
    python -m pip install -r "$PROJECT_ROOT/models/Qwen3-VL/requirements_web_demo.txt"
fi

if [ -d "$PROJECT_ROOT/models/sam3" ]; then
    python -m pip install -e "$PROJECT_ROOT/models/sam3"
fi

if [ -d "$PROJECT_ROOT/models/Depth-Anything-V2/metric_depth" ]; then
    python -m pip install -r "$PROJECT_ROOT/models/Depth-Anything-V2/metric_depth/requirements.txt"
fi

python -m pip install "numpy>=1.26,<2"

echo "Cloud Ubuntu setup completed."
echo "Activate with: conda activate $ENV_NAME"
