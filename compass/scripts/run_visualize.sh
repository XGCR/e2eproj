#!/bin/bash

# 设置环境变量
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 运行推理
cd "$PROJECT_ROOT"
python src/visualize.py \
    --input "data/output/correct"

# 示例用法:
# ./scripts/run_inference.sh data/input/test.jpg
# ./scripts/run_inference.sh data/input/