# Compass 项目部署完整配置清单

## 📋 项目概述

**项目名称**：Compass  
**项目类型**：图像处理 + 深度学习推理 Pipeline  
**核心功能**：三阶段处理流程
- 阶段1：深度估计 (Depth Anything V2)
- 阶段2：对象分割 (SAM3)  
- 阶段3：视觉理解与导航 (Qwen3-VL)

---

## 🖥️ 系统环境需求

### 最低配置
- **操作系统**：Ubuntu 22.04+ 或 Windows 10/11
- **GPU**：NVIDIA GPU (16GB VRAM最低)
- **CPU**：8核+
- **内存**：32GB RAM+
- **磁盘**：200GB+ (用于模型权重)
- **网络**：需要访问 HuggingFace 或 ModelScope

### 推荐配置
- Ubuntu 22.04 或 24.04 (优先)
- NVIDIA GPU 24GB+ VRAM
- 8+ vCPU
- 32GB+ RAM
- 200GB+ SSD

---

## 🔧 核心版本要求

### Python 环境
| 组件 | 版本 | 备注 |
|------|------|------|
| Python | **3.11** | 必需，3.8-3.12 均可支持 |
| pip | 最新版本 | `python -m pip install --upgrade pip` |
| Conda | 最新版本 | Miniconda3 推荐 |

### 深度学习框架
| 组件 | 版本 | CUDA版本 | 备注 |
|------|------|---------|------|
| PyTorch | 最新 LTS | 12.1 | CPU 版本也可（性能差） |
| torchvision | 最新 | CUDA 12.1 | - |
| torchaudio | 最新 | CUDA 12.1 | - |
| CUDA Toolkit | 12.1+ | - | 系统级，非 pip 安装 |
| cuDNN | 8.9+ | - | CUDA 依赖 |

**安装命令**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 核心库版本
| 库名 | 版本 | 用途 |
|------|------|------|
| numpy | **>=1.26, <2** | SAM3 兼容性要求（强制） |
| opencv-python | 最新 | 图像处理 |
| Pillow | 最新 | 图像处理 |
| PyYAML | 最新 | 配置解析 |
| timm | >=1.0.17 | SAM3 转换器骨架 |

### 模型特定依赖

**Depth Anything V2**
```
gradio==4.29.0
matplotlib
opencv-python
torch
torchvision
```

**SAM3 (Segment Anything Model 3)**
```
timm>=1.0.17
numpy>=1.26,<2
tqdm
ftfy==6.1.1
regex
iopath>=0.1.10
typing_extensions
huggingface_hub
```

**Qwen3-VL (视觉语言模型)**
```
gradio==5.49.1
gradio_client==1.13.3
transformers-stream-generator==0.0.5
transformers (HuggingFace 开发版，特定 commit)
torch
torchvision
accelerate
flash-attn (可选，提高推理速度)
```

### 其他依赖库
```
tifffile          # TIFF 图像支持
einops            # 张量操作
decord            # 视频处理
pycocotools       # COCO 数据集工具
transformers      # HuggingFace 模型库
accelerate        # 分布式推理加速
sentencepiece     # 分词器
modelscope        # 模型权重下载（国内源）
```

---

## 📁 系统配置文件

### 配置文件位置
```
configs/default_config.yaml
```

### 完整配置参数

#### 系统级配置
```yaml
system:
  device: "cuda"         # 计算设备: "cuda" 或 "cpu"
  num_workers: 4         # 数据加载器工作线程数
  seed: 42               # 随机种子
  log_level: "INFO"      # 日志级别
```

#### 相机配置 (OV2640 传感器)
```yaml
camera:
  fov_x_deg: 62.0       # 水平视场角 (度)
  sensor_width: 3.2     # 传感器宽度 (mm) - 1/4" ≈ 3.2mm
  sensor_height: 2.4    # 传感器高度 (mm) - 1/4" ≈ 2.4mm
  focal_length: 4.3     # 焦距 (mm)
```

#### Depth Anything V2 配置
```yaml
da2:
  scene: "outdoor"       # 场景: "indoor" 或 "outdoor"
  dataset:
    indoor: "hypersim"   # 室内模型训练数据集
    outdoor: "vkitti"    # 室外模型训练数据集
  encoder: "vitl"        # 编码器型号: "vits", "vitb", "vitl"
  checkpoint_paths:
    hypersim:
      vitl: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
      vitb: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth"
      vits: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
    vkitti:
      vitl: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"
      vitb: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitb.pth"
      vits: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth"
  model_configs:
    vits: {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    vitb: {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
    vitl: {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
  max_depth:
    indoor: 20           # 室内最大深度 (米)
    outdoor: 80          # 室外最大深度 (米)
```

**模型大小参考**:
- vitl: 最大，推荐用于高精度场景
- vitb: 中等，平衡精度与速度
- vits: 最小，快速推理

#### SAM3 配置
```yaml
sam3:
  checkpoint_path: "models/sam3/checkpoints/sam3.pt"
```

#### Qwen3-VL 配置
```yaml
qwen3vl:
  checkpoint_path: "models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct"
```

#### 路径配置
```yaml
paths:
  input_image_folder: "data/input/image"
  output_depth_folder: "data/output/depth_image"
  output_sam3_json_folder: "data/output/sam3_json"
  temp_organized_json_folder: "data/temp"
  output_correct_json_folder: "data/output/correct"
  output_navigation_folder: "data/output/navigation"
```

#### 导航生成器配置
```yaml
navigation:
  enabled: true
  output_language: "zh"   # 输出语言: "zh" 或 "en"
  device: "cuda"
  max_new_tokens: 128
```

#### TTS 配置
```yaml
tts:
  enabled: false          # 是否启用文本到语音
  backend: "system"       # 后端系统
  language: "en"          # 语言
  voice_hint: "English"   # 语音提示
```

---

## 📦 模型权重配置

### 所需模型文件清单

| 模型 | 大小 | 下载源 | 本地路径 |
|------|------|--------|---------|
| Depth-Anything-V2 vitl | ~1.3GB | HuggingFace / ModelScope | `models/Depth-Anything-V2/metric_depth/checkpoints/` |
| SAM3 | ~2.5GB | HuggingFace / ModelScope | `models/sam3/checkpoints/sam3.pt` |
| Qwen3-VL-8B-Instruct | ~16GB | HuggingFace / ModelScope | `models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct/` |

**总体积**：约 20GB+

### 模型下载方式

#### 方式1：自动脚本下载 (推荐)
```bash
bash scripts/cloud_download_models.sh
```

#### 方式2：手动下载
需要访问：
- HuggingFace Hub: https://huggingface.co
- ModelScope (国内): https://www.modelscope.cn

然后放入对应路径。

---

## 🚀 部署步骤速览

### Ubuntu/Linux 部署
```bash
# 1. 克隆项目
cd /workspace/compass

# 2. 环境准备（自动安装系统包、Conda、Python 依赖）
bash scripts/cloud_setup_ubuntu.sh

# 3. 准备运行目录
bash scripts/cloud_prepare_dirs.sh

# 4. 下载模型权重
bash scripts/cloud_download_models.sh

# 5. 验证安装
bash scripts/cloud_validate_setup.sh

# 6. 运行推理
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compass
python src/main.py --config configs/default_config.yaml
```

### Windows 部署
```powershell
# 1. 创建 Conda 环境
conda create -n compass python=3.11 -y
conda activate compass

# 2. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 克隆模型仓库
cd models
git clone https://github.com/DepthAnything/Depth-Anything-V2
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/QwenLM/Qwen3-VL.git

# 4. 安装依赖（按顺序：Qwen3 -> SAM3 -> DA2）
python -m pip install --upgrade pip
pip install -r models/Qwen3-VL/requirements_web_demo.txt
pip install -e models/sam3
pip install -r models/Depth-Anything-V2/metric_depth/requirements.txt

# 5. 下载模型并运行
python src/main.py --config configs/default_config.yaml
```

---

## ✅ 验证清单

部署完成后，运行验证脚本检查以下项目：

- [ ] `nvidia-smi` 命令工作正常
- [ ] `torch.cuda.is_available()` 返回 `True`
- [ ] numpy 版本为 `1.26.x`
- [ ] SAM3 模块可正常导入
- [ ] PipelineManager 可正常导入
- [ ] 模型路径存在：
  - [ ] `models/Depth-Anything-V2/metric_depth/checkpoints/`
  - [ ] `models/sam3/checkpoints/sam3.pt`
  - [ ] `models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct/`

```bash
bash scripts/cloud_validate_setup.sh
```

---

## 📝 现有部署文档

本项目已包含以下针对不同平台的部署文档：

1. **[CLOUD_GPU_DEPLOY.md](CLOUD_GPU_DEPLOY.md)** - Ubuntu GPU 云实例部署
2. **[SERVER_DEPLOY_STEPS.md](SERVER_DEPLOY_STEPS.md)** - 远程服务器部署（手动上传）
3. **[WINDOWS_DEPLOY.md](WINDOWS_DEPLOY.md)** - Windows 本地部署

---

## 🔗 常用命令参考

### 环境激活
```bash
# Linux/Mac
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compass

# Windows PowerShell
conda activate compass
```

### 运行推理
```bash
python src/main.py --config configs/default_config.yaml
```

### 运行可视化
```bash
python src/visualize.py
```

### 常见问题排查

**问题1：numpy 版本冲突**
```bash
pip install "numpy>=1.26,<2"
```

**问题2：CUDA 不可用**
```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
```

**问题3：模型下载失败**
- 使用 ModelScope（国内镜像）：设置环境变量
  ```bash
  export MODELSCOPE_CACHE=/path/to/cache
  ```
- 或手动下载后放入对应路径

---

## 📞 技术支持信息

- Python：https://www.python.org
- PyTorch：https://pytorch.org
- SAM3：https://github.com/facebookresearch/sam3
- Depth Anything V2：https://github.com/DepthAnything/Depth-Anything-V2
- Qwen3-VL：https://github.com/QwenLM/Qwen3-VL

---

**最后更新**：2026-04-01  
**文档版本**：1.0
