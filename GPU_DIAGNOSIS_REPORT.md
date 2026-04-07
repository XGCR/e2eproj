# GPU 诊断报告 & 优化方案

## 📊 硬件环境

- **GPU**: 1× NVIDIA GeForce RTX 2080 Ti (10.57 GB)
- **CUDA**: 12.1
- **cuDNN**: 90100
- **CPU 使用率**: 8.3% ✅
- **内存使用**: 15.6% (9.76/62.47 GB) ✅

**注意**: 你提到是双卡,但系统只识别到 1 张 GPU!

---

## 🔴 发现的严重问题

### 1. ❌ Qwen3-VL 在 CPU 上运行!

**诊断结果**:
```
模型设备: cpu
在 GPU 上: 否 ❌
模型精度: torch.bfloat16
```

**问题原因**: 
- 配置文件 `default_config.yaml` 中 `qwen3vl` 部分**没有 device 字段**
- 代码默认使用 `cuda`,但可能因为显存不足或其他原因回退到 CPU
- **在 CPU 上运行 8B 参数模型,每帧需要 7-8 分钟!**

**解决方案**:
```yaml
# configs/default_config.yaml
qwen3vl:
  checkpoint_path: "models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct"
  device: "cuda:0"  # 添加这一行!
```

---

### 2. ⚠️ 输入分辨率过高

**诊断结果**:
```
测试图像分辨率: 1681×1035
超过 640×640: 是 ❌
建议缩放因子: 0.38
新分辨率: 640×394
像素减少: 85.5%
```

**问题**: 原始分辨率是推荐值的 **4.3 倍**,导致:
- Depth Anything 计算量增加 4-5 倍
- SAM3 计算量增加 4-5 倍

**解决方案**: 在 pipeline 中添加自动缩放

---

### 3. ⚠️ Depth Anything 使用 Large 模型

**配置**:
```yaml
da2:
  encoder: "vitl"  # Large 版本
```

**问题**: 
- `vitl` (Large) 参数量 335M
- `vits` (Small) 参数量 24M
- Large 比 Small 慢 **3-4 倍**,但精度提升 <5%

**解决方案**: 切换到 `vitb` (Base) 或 `vits` (Small)

---

### 4. ❌ SAM3 配置缺少 device 字段

**问题**: 配置文件中 `sam3` 没有指定 device,可能导致设备分配错误

---

## 🎯 优化方案 (按优先级排序)

### 方案 1: 修复 Qwen3-VL GPU 使用 (最重要!)

**预计效果**: 从 7-8 分钟/帧 → 3-5 秒/帧 (加速 100x!)

**步骤**:

1. 修改配置文件:
```yaml
qwen3vl:
  checkpoint_path: "models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct"
  device: "cuda:0"  # 强制使用 GPU
```

2. 在 `qwen3vl_interpreter.py` 中确保模型加载到 GPU:
```python
def _load_model(self):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        self.checkpoint_path,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 加速
        device_map="cuda:0"  # 强制使用 GPU
    )
    return model, processor
```

---

### 方案 2: 降低输入分辨率

**预计效果**: 加速 3-4 倍

**在 `process_video_smart.py` 中添加**:
```python
def resize_frame_if_needed(frame, max_size=640):
    """如果帧超过 max_size,自动缩放"""
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame
```

---

### 方案 3: 切换到 Depth Anything Small

**预计效果**: 加速 3-4 倍

**修改配置**:
```yaml
da2:
  encoder: "vits"  # 从 vitl 改为 vits
  checkpoint_paths:
    vkitti:
      vits: "models/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vits.pth"
```

---

### 方案 4: 使用 4-bit 量化 Qwen3-VL

**预计效果**: 显存减少 60%,加速 2-3 倍

**需要安装**:
```bash
pip install bitsandbytes accelerate
```

**修改代码**:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    checkpoint_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

### 方案 5: 多 GPU 分配 (如果你有双卡)

**当前问题**: 系统只识别到 1 张 GPU

**检查命令**:
```bash
nvidia-smi
```

**如果真的只有 1 张卡**,可以忽略此方案。

**如果有 2 张卡但只识别 1 张**,需要:
1. 检查 PCIe 连接
2. 更新驱动
3. 设置 `CUDA_VISIBLE_DEVICES=0,1`

**多 GPU 分配策略**:
```python
# SAM3 + Depth → GPU 0
sam3.to("cuda:0")
da2.to("cuda:0")

# Qwen3-VL → GPU 1  
qwen3vl.to("cuda:1")
```

---

## 📈 预期性能提升

| 优化项 | 当前 | 优化后 | 提升 |
|--------|------|--------|------|
| **Qwen3-VL 设备** | CPU (420-480s) | GPU (3-5s) | **100x** |
| **输入分辨率** | 1681×1035 | 640×394 | **4x** |
| **Depth Anything** | vitl (0.5s) | vits (0.15s) | **3x** |
| **总计 (单帧)** | **7-8 分钟** | **5-8 秒** | **60-80x** |
| **FPS** | 0.002 | 0.12-0.2 | **达到可用水平** |

**注意**: 要达到 10 FPS 目标,需要:
- 使用更小的模型 (Depth-Anything vits, Qwen3-VL 量化)
- 降低分辨率到 480px
- 或使用更强的 GPU (RTX 4090, A100)

---

## 🚀 立即行动建议

### 第一步 (最重要): 修复 Qwen3-VL GPU 使用

这是最大的瓶颈!只需修改配置文件即可。

### 第二步: 降低输入分辨率

在 pipeline 开始处添加自动缩放,简单且效果显著。

### 第三步: 测试性能

运行诊断脚本验证优化效果:
```bash
python diagnose_gpu.py
```

---

## 📝 检查清单

- [ ] 修复 Qwen3-VL device 配置
- [ ] 添加输入分辨率自动缩放
- [ ] 考虑切换到 Depth Anything vits
- [ ] 测试 Qwen3-VL 4-bit 量化
- [ ] 验证是否真的是单卡 (nvidia-smi)
- [ ] 运行诊断脚本确认优化效果

---

## 🔧 快速修复脚本

我可以帮你自动应用这些优化,只需要你说一声!
