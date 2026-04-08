#!/bin/bash
set -e  # 遇到错误就退出

ENV_NAME="compass3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. 检查并创建环境
if ! conda env list | grep -q "/${ENV_NAME}$"; then
    echo "正在创建虚拟环境 ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.11 -y
fi

# 2. 激活环境 - 使用 . 代替 source
. $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

pip install --upgrade pip

# 3. 安装所需库依赖，安装顺序从 Qwen3-VL -> SAM3 -> Depth Anything2
#    调换顺序可能会导致依赖包之间不兼容
# 3-1. Qwen3-VL环境
echo "Creating env for Qwen3-VL model..."
cd models/Qwen3-VL # 位置从 项目根目录 切换到 models/Qwen3-VL
pip install -r requirements_web_demo.txt

# 3-2. SAM3环境
echo "Creating env for SAM3 model..."
cd ../sam3
pip install -e .
pip install modelscope # 用于在modelscope下载模型权重

# 3-3. Depth Anything2环境
echo "Creating env for Depth Anything2 model..."
cd ../Depth-Anything-V2/metric_depth
pip install -r requirements.txt

# 其他依赖库
pip install tifffile
pip install einops
pip install decord
pip install pycocotools