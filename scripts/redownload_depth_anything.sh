#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "重新下载 Depth Anything V2 模型权重"
echo "========================================="

CHECKPOINT_DIR="models/Depth-Anything-V2/metric_depth/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# 如果模型文件已存在,先询问是否删除
if [ -f "$CHECKPOINT_DIR/depth_anything_v2_metric_vkitti_vitl.pth" ]; then
    echo "发现已存在的模型文件,正在删除..."
    rm -f "$CHECKPOINT_DIR"/depth_anything_v2_metric_*.pth
    echo "已删除旧的模型文件"
fi

cd "$CHECKPOINT_DIR"

echo ""
echo "开始下载模型权重文件..."
echo "下载地址: HuggingFace"
echo ""

# 只下载 outdoor large 模型 (当前配置使用)
echo "下载 outdoor large 模型 (当前使用)..."
wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth

echo ""
echo "========================================="
echo "模型下载完成!"
echo "========================================="
echo ""
echo "模型文件列表:"
ls -lh "$PROJECT_ROOT/$CHECKPOINT_DIR"/*.pth
echo ""
echo "当前配置使用的模型: depth_anything_v2_metric_vkitti_vitl.pth (outdoor large)"
