#!/bin/bash
# GitHub 推送脚本

echo "Compass 项目 GitHub 推送助手"
echo "================================"

# 获取用户信息
read -p "请输入你的 GitHub 用户名: " username
read -p "请输入仓库名称 (默认: compass): " repo_name
repo_name=${repo_name:-compass}

# 构建仓库URL
repo_url="https://github.com/${username}/${repo_name}.git"

echo ""
echo "仓库信息:"
echo "用户名: $username"
echo "仓库名: $repo_name"
echo "仓库URL: $repo_url"
echo ""

# 确认
read -p "信息正确吗? (y/N): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "操作已取消。"
    exit 1
fi

# 执行Git命令
echo "正在添加远程仓库..."
git remote add origin "$repo_url"

echo "正在重命名分支为 main..."
git branch -M main

echo "正在推送代码..."
git push -u origin main

echo ""
echo "✅ 项目已成功推送到 GitHub!"
echo "仓库地址: https://github.com/${username}/${repo_name}"