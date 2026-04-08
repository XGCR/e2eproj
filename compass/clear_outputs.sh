#!/bin/bash

# 清空输出文件夹脚本

echo "开始清空输出文件夹..."

# 清空correct文件夹
echo "清空 data/output/correct/..."
rm -rf data/output/correct/*

# 清空depth_image文件夹
echo "清空 data/output/depth_image/..."
rm -rf data/output/depth_image/*

# 清空sam3_json文件夹
echo "清空 data/output/sam3_json/..."
rm -rf data/output/sam3_json/*

# 清空temp文件夹(可选)
echo "清空 data/temp/..."
rm -rf data/temp/*

echo "✓ 清空完成！"
