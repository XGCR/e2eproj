# Compass: 多目标深度语义导航管道

一个基于深度学习的多目标深度语义导航系统，能够从图像中检测路牌、识别文字、生成导航指令，并支持室内外场景的导航。

## 功能特性

- **深度估计**: 使用 Depth-Anything-V2 进行单目深度估计
- **目标分割**: 使用 SAM3 (Segment Anything Model 3) 进行精确的目标分割
- **文字识别**: 使用 Qwen3-VL 大语言模型进行多语言文字识别
- **导航生成**: 基于检测结果生成自然语言导航指令
- **可视化**: 提供丰富的可视化输出，包括标注图像和导航视频
- **多格式支持**: 支持 JPG、PNG 图像格式
- **灵活配置**: YAML 配置驱动，支持不同部署环境

## 快速开始

### 环境要求

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (推荐用于GPU加速)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/compass.git
   cd compass
   ```

2. **创建环境**
   ```bash
   conda create -n compass python=3.11
   conda activate compass
   ```

3. **安装依赖**
   ```bash
   # 安装 PyTorch (根据你的CUDA版本调整)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # 安装项目依赖
   pip install -r requirements.txt
   ```

4. **下载模型**
   ```bash
   bash scripts/download_model.sh
   ```

5. **运行测试**
   ```bash
   # 将测试图片放入 data/input/image/
   python src/main.py --config configs/default_config.yaml

   # 生成可视化结果
   python src/visualize.py
   ```

## 项目结构

```
compass/
├── configs/                 # 配置文件
├── data/                    # 数据目录
│   ├── input/image/         # 输入图片
│   ├── output/              # 输出结果
│   ├── temp/                # 临时文件
│   └── visualize/           # 可视化结果
├── models/                  # 模型文件 (需要单独下载)
├── scripts/                 # 部署和工具脚本
├── src/                     # 源代码
│   ├── pipeline/            # 管道组件
│   │   ├── da2_predictor.py        # 深度估计
│   │   ├── sam3_detector.py        # 目标分割
│   │   ├── qwen3vl_interpreter.py  # 文字识别
│   │   ├── navigation_generator.py # 导航生成
│   │   ├── orientation_correct.py  # 角度修正
│   │   └── pipeline_manager.py     # 管道管理
│   ├── main.py              # 主程序
│   └── visualize.py         # 可视化脚本
└── docs/                    # 文档
```

## 配置说明

项目使用 YAML 配置文件，位于 `configs/default_config.yaml`。主要配置项包括：

- **模型设置**: 各组件的启用/禁用、设备选择
- **路径配置**: 输入输出目录
- **参数调优**: 模型参数、阈值设置

## 部署指南

- [云GPU部署](CLOUD_GPU_DEPLOY.md)
- [服务器部署](SERVER_DEPLOY_STEPS.md)
- [Windows部署](WINDOWS_DEPLOY.md)

## 视频处理

项目支持视频处理功能：

- [视频处理指南](VIDEO_PROCESSING_GUIDE.md)
- [快速开始视频](QUICK_START_VIDEO.md)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证。