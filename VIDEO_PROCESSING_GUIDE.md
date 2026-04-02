# 视频处理完整指南

快速生成 demo 视频的两个方案

---

## 🎯 两种处理策略对比

| 方案 | 处理模式 | 速度 | 精度 | 适用场景 | 处理时间 |
|------|---------|------|------|---------|----------|
| **快速版 (推荐)** | 关键帧采样 | ⚡⚡⚡ 快 | ⭐⭐⭐ 好 | Demo演示 | 5-10 分钟 |
| **完整版** | 逐帧处理 | ⚡ 慢 | ⭐⭐⭐⭐ 最佳 | 学术研究/生产 | 1-2 小时+ |

**推荐**：使用**快速版** (`process_video_fast.py`)，速度快且效果足够好

---

## ⚡ 快速上手 (30秒)

### 步骤1：准备视频
```bash
# 把你的视频放在项目根目录
# 假设视频文件名为: demo.mp4
```

### 步骤2：运行处理脚本
```bash
# 快速版 (推荐，5-10分钟)
python process_video_fast.py --input demo.mp4

# 或者用完整版 (1-2小时)
python process_video.py --input demo.mp4
```

### 步骤3：查看结果
```bash
# 输出视频在这里：
# 快速版: data/output/video_demo_fast/demo_*.mp4
# 完整版: data/output/video_demo/demo_*.mp4
```

---

## 📖 完整使用说明

### 方案 1️⃣ : 快速版 (关键帧策略) ⭐ 推荐用这个

**原理**：
- 每 15 帧（或指定间隔）提取一个"关键帧"
- 只对关键帧进行深度学习推理 (慢)
- 中间帧直接使用最近的关键帧结果 (快)
- 最后将所有帧合成为视频

**优点**：
- 🚀 速度快 (5-10 分钟处理 2-3 分钟视频)
- 💾 节省内存和磁盘空间
- ✅ 检测效果仍然很好
- 🎬 适合生成 demo 视频

**基本用法**：
```bash
python process_video_fast.py --input your_video.mp4
```

**高级用法**：
```bash
# 指定关键帧间隔（推荐 15-30）
python process_video_fast.py \
  --input your_video.mp4 \
  --keyframe-interval 15

# 指定输出目录
python process_video_fast.py \
  --input your_video.mp4 \
  --output data/output/my_demo \
  --keyframe-interval 20

# 指定配置文件
python process_video_fast.py \
  --input your_video.mp4 \
  --config configs/default_config.yaml
```

**关键帧间隔说明**：
| 间隔 | 帧/秒 | 示例 | 优缺点 |
|------|-------|------|--------|
| 5 | 6 fps | 精细，但巨慢 | ⭐ 精度最高，耗时长 |
| 10 | 3 fps | 精细 | ⭐⭐ 实用性不理想 |
| **15** | **2 fps** | **均衡** | **⭐⭐⭐ 推荐** |
| 20 | 1.5 fps | 较快 | ⭐⭐ 可接受 |
| 30 | 1 fps | 快速 | ⭐ 可能遗漏 |

**处理流程**：
```
1. 读取视频 (内存中)
   ↓
2. 提取关键帧 (N帧采样)
   ↓
3. 处理关键帧:
   - 深度估计 (Depth Anything V2)
   - 对象分割 (SAM3)
   - 文字识别 (Qwen3-VL)
   - 方向识别
   ↓
4. 标注所有帧:
   - 关键帧: 显示检测结果
   - 中间帧: 使用最近关键帧的结果
   ↓
5. 合成视频
   ↓
6. 保存结果
```

**输出文件**：
```
data/output/video_demo_fast/
├── demo_YYYYMMDD_HHMMSS.mp4       ← 最终输出视频 ⭐
├── processing_summary.json         ← 处理统计
└── detection_results/              ← 每个关键帧的检测 JSON
    ├── frame_000000_detections.json
    ├── frame_000015_detections.json
    └── ...
```

---

### 方案 2️⃣ : 完整版 (逐帧处理)

**原理**：
- 处理视频的每一帧
- 完整的深度学习推理
- 最精准但最慢的方案

**优点**：
- 🎯 检测最精准
- 📊 适合学术研究
- 🔍 完整的逐帧结果

**缺点**：
- 🐢 很慢 (1-2 小时或更多)
- 💾 需要大量磁盘空间
- 🖥️ GPU 占用高

**基本用法**：
```bash
python process_video.py --input your_video.mp4
```

**高级用法**：
```bash
# 使用采样加速（处理每N帧）
python process_video.py \
  --input your_video.mp4 \
  --sample 2          # 处理每2帧（速度快2倍）

# 指定输出目录
python process_video.py \
  --input your_video.mp4 \
  --output data/output/full_analysis
```

**输出文件**：
```
data/output/video_demo/
├── demo_YYYYMMDD_HHMMSS.mp4      ← 最终输出视频
├── detection_results/             ← 所有帧的检测结果 JSON
│   ├── frame_000000_detections.json
│   ├── frame_000001_detections.json
│   └── ...
└── processing_summary.json        ← 处理统计
```

---

## 🔄 处理流程详解

### 深度推理部分 (所有关键帧都会执行)

1. **深度估计** (Depth Anything V2)
   - 生成深度地图
   - 用于理解3D场景

2. **对象分割** (SAM3)
   - 分割指示牌/标志牌
   - 生成分割掩码

3. **语义理解** (Qwen3-VL)
   - 识别文字内容
   - 判断方向（上/下/左/右）

4. **角度修正** (Orientation Correcter)
   - 修正视角角度

### 可视化部分 (所有帧都会执行)

- 在图像上绘制检测框 (绿色)
- 绘制方向箭头 (颜色表示方向)
- 显示识别文字

---

## 📊 输出格式说明

### 视频输出格式
- **编码**: H.264 (MP4v)
- **分辨率**: 原视频分辨率
- **帧率**: 原视频帧率

### JSON 检测结果格式
```json
[
  {
    "alpha": -90,           // 方向角度 (0=上, 90=右, 180=下, -90=左)
    "theta": "",            // 预留字段
    "sign_text": "指示文字",  // 识别的文字内容
    "image_path": "...",    // 原图路径
    "depth_path": "...",    // 深度图路径
    "arrow_box": [[x1,y1], [x2,y2]]  // 检测框坐标
  },
  // ... 更多检测结果
]
```

---

## 🎨 输出视频标注说明

### 颜色含义
- **绿框**: 检测到的指示牌/标志牌
- **方向箭头**:
  - 绿色: 上方
  - 青色: 右方
  - 蓝色: 左方
  - 红色: 下方
- **黄色文字**: 识别出的文字内容

### 示例frame
```
┌─────────────────────────────────────┐
│  指示牌 (ID:1)                       │
│  ┌──────────┐                       │
│  │  🟢 ↑   │ (绿色边框+向上箭头)    │
│  │          │                       │
│  │ "向右"   │ (黄色文字)             │
│  └──────────┘                       │
└─────────────────────────────────────┘
```

---

## ⏱️ 时间预估

**快速版** (关键帧间隔 15):
- 30秒视频: 1-2 分钟
- 2分钟视频: 5-10 分钟  
- 5分钟视频: 15-20 分钟

**完整版** (帧率 30fps):
- 30秒视频: 10-20 分钟
- 2分钟视频: 1-2 小时
- 5分钟视频: 2-5 小时

（实际时间取决于硬件和模型）

---

## 🐛 常见问题

### Q: 处理很慢？
**A**: 使用快速版 (`process_video_fast.py`) 或增加关键帧间隔：
```bash
python process_video_fast.py --input video.mp4 --keyframe-interval 30
```

### Q: GPU 内存不足？
**A**: 
1. 都用快速版
2. 减少 batch size (修改 config)
3. 使用 CPU (config: device: "cpu")

### Q: 检测效果不好？
**A**:
1. 使用完整版处理所有帧
2. 调整 config 中的模型参数
3. 确保视频质量足够好

### Q: 如何只处理视频的某一部分？
**A**: 用 ffmpeg 先截取：
```bash
# 提取 10 秒 (从 0 秒开始)
ffmpeg -i input.mp4 -ss 0 -t 10 -c copy segment.mp4

# 然后处理
python process_video_fast.py --input segment.mp4
```

### Q: 输出视频编码错误？
**A**: 确保系统有 ffmpeg/libx264：
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## 🔧 配置优化

编辑 `configs/default_config.yaml` 优化效果：

```yaml
# 深度估计模型选择
da2:
  encoder: "vitl"    # vitl (精准但慢), vitb (平衡), vits (快)

# 设备选择
system:
  device: "cuda"     # "cuda" (快) 或 "cpu" (慢)

# 文字输出语言
navigation:
  output_language: "zh"  # "zh" (中文) 或 "en" (英文)
```

---

## 📝 命令快速参考

```bash
# 最快速 (推荐演示)
python process_video_fast.py --input video.mp4

# 平衡方案
python process_video_fast.py --input video.mp4 --keyframe-interval 15

# 精细方案
python process_video_fast.py --input video.mp4 --keyframe-interval 5

# 完整处理 (学术用)
python process_video.py --input video.mp4

# 加速处理每 2 帧
python process_video.py --input video.mp4 --sample 2

# 自定义输出目录
python process_video_fast.py \
  --input video.mp4 \
  --output my_output \
  --keyframe-interval 15
```

---

## ✅ 使用清单

在运行前检查：
- [ ] 视频文件存在
- [ ] 足够的磁盘空间 (快速版需 500MB-1GB)
- [ ] GPU 可用或 CPU 充足
- [ ] 配置文件正确 (`configs/default_config.yaml`)
- [ ] 模型权重已下载

---

**最后更新**: 2026-04-01
