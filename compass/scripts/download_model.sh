#!/bin/bash
source /home/duyiyang/anaconda3/etc/profile.d/conda.sh
set -e  # 遇到错误就退出

ENV_NAME="compass"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. 检查并创建环境
if ! conda env list | grep -q "/${ENV_NAME}$"; then
    echo "正在创建虚拟环境 ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.11 -y
fi

# 2. 激活环境
conda activate ${ENV_NAME}
pip install --upgrade pip

# 3. 开始download模型并安装环境依赖
echo "Setting up Multi-Target Pipeline Environment"
cd models/ # 位置从 项目根目录 切换到 models/

# 克隆各个模型仓库
echo "Cloning model repositories..."
# --------------------------------------------------------------------------
# Depth Anything 2 模型
# 当前目录位置：scripts/
if [ ! -d "./Depth-Anything-V2" ]; then
    echo "Cloning Depth Anything 2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    cd Depth-Anything-V2/metric_depth # 位置 切换到 models/Depth-Anything-V2/metric_depth
    # 下载预训练权重
    mkdir checkpoints
    cd checkpoints # models/Depth-Anything-V2/metric_depth/checkpoints/
    # small / in / out
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth
    # base / in / out
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth
    # large / in / out
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
    cd ../../../ # models/
else
    echo "Depth Anything 2 already exists"
fi
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Segment Anything 3 模型
if [ ! -d "./sam3" ]; then
    echo "Cloning Segment Anything 3 models..."
    git clone https://github.com/facebookresearch/sam3.git
    cd sam3
    # 下载预训练权重
    mkdir checkpoints
    cd checkpoints  # models/sam3/checkpoints/
    modelscope download --model facebook/sam3 --local_dir ./  
    cd ../../ # models/
else
    echo "sam3 already exists"
fi
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Qwen3VL 模型
if [ ! -d "./Qwen3-VL" ]; then
    echo "Cloning Qwen3-VL models..."
    git clone https://github.com/QwenLM/Qwen3-VL.git
    cd Qwen3-VL # models/Qwen3-VL
    # 下载预训练权重
    mkdir checkpoints
    cd checkpoints  # models/Qwen3-VL/checkpoints/
    git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
    cd ../../ # models/
else
    echo "Qwen3-VL already exists"
fi

