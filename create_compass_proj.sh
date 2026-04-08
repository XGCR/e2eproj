#!/bin/bash

# 定义项目结构
PROJECT_NAME="compass"
DIRECTORIES=(
    "models/"
    "src/pipeline"
    "configs/"
    "scripts/"
    "data/"
)

# 创建主项目目录
mkdir -p "$PROJECT_NAME"

# 进入项目目录
cd "$PROJECT_NAME" || exit

# 创建所有子目录
for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
    echo "创建目录: $dir"
done

# # 创建文件
# touch Dockerfile
# touch docker-compose.yml
# touch setup.py
# touch requirements.txt

echo "项目结构创建完成！"