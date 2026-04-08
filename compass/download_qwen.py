#!/usr/bin/env python3
"""
下载 Qwen3-VL 模型
"""
from huggingface_hub import snapshot_download
import os

# 设置模型保存路径
model_id = "Qwen/Qwen3-VL-8B-Instruct"
local_dir = "models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct"

# 确保目录存在
os.makedirs(local_dir, exist_ok=True)

print(f"开始下载模型: {model_id}")
print(f"保存位置: {local_dir}")

try:
    # 下载模型
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
        resume_download=True  # 断点续传
    )
    print("✓ 模型下载完成!")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    exit(1)
