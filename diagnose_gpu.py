"""
GPU 诊断脚本 - 检查 RTX 2080 Ti 双卡环境下的模型推理性能
"""
import torch
import cv2
import numpy as np
import time
import psutil
from pathlib import Path
from PIL import Image
import sys
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent / "models" / "sam3"))
sys.path.append(str(Path(__file__).parent / "models" / "Depth-Anything-V2"))

print("="*80)
print("🔍 GPU 诊断工具 - RTX 2080 Ti 双卡环境")
print("="*80)

# ========================================
# 1. GPU 基本信息检查
# ========================================
print("\n" + "="*80)
print("1️⃣  GPU 基本信息")
print("="*80)

print(f"\n✅ CUDA 可用: {torch.cuda.is_available()}")
print(f"✅ CUDA 版本: {torch.version.cuda}")
print(f"✅ cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"✅ GPU 数量: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\n📊 GPU {i}:")
    print(f"   名称: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"   总显存: {props.total_memory / 1024**3:.2f} GB")
    print(f"   计算能力: {props.major}.{props.minor}")

# 当前使用的 GPU
current_device = torch.cuda.current_device()
print(f"\n🎯 当前使用 GPU: {current_device} - {torch.cuda.get_device_name(current_device)}")

# ========================================
# 2. 显存使用状态
# ========================================
print("\n" + "="*80)
print("2️⃣  显存使用状态")
print("="*80)

for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    
    print(f"\n📊 GPU {i}:")
    print(f"   已分配: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"   已预留: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"   总容量: {total:.2f} GB")

# ========================================
# 3. 检查测试图像
# ========================================
print("\n" + "="*80)
print("3️⃣  输入图像分辨率检查")
print("="*80)

# 查找测试图像
test_images = list(Path("data/input/image").glob("*.jpg")) + list(Path("data/input/image").glob("*.png"))
if test_images:
    test_image_path = test_images[0]
    print(f"\n📷 测试图像: {test_image_path}")
    
    # 使用 OpenCV 读取
    img_cv = cv2.imread(str(test_image_path))
    h, w, c = img_cv.shape
    print(f"   OpenCV Shape: ({h}, {w}, {c})")
    print(f"   分辨率: {w}×{h}")
    print(f"   ⚠️  超过 640×640: {'是 ❌' if h > 640 or w > 640 else '否 ✅'}")
    
    # 使用 PIL 读取
    img_pil = Image.open(test_image_path)
    print(f"   PIL Size: {img_pil.size} (W×H)")
    
    if h > 640 or w > 640:
        scale_factor = 640 / max(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        print(f"\n💡 建议缩放:")
        print(f"   缩放因子: {scale_factor:.2f}")
        print(f"   新分辨率: {new_w}×{new_h}")
        print(f"   像素减少: {(1 - scale_factor**2)*100:.1f}%")
else:
    print("\n⚠️  未找到测试图像")
    # 创建一个测试图像
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    h, w, c = test_img.shape
    print(f"   创建测试图像: {w}×{h}")
    print(f"   ⚠️  超过 640×640: 是 ❌")

# ========================================
# 4. 加载并检查 Depth Anything V2
# ========================================
print("\n" + "="*80)
print("4️⃣  Depth Anything V2 模型检查")
print("="*80)

try:
    from src.pipeline.da2_predictor import DA2Predictor
    import yaml
    
    # 加载配置
    with open("configs/default_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    da2_config = config['depth_estimation']
    print(f"\n📦 模型配置:")
    print(f"   编码器: {da2_config.get('encoder', 'unknown')}")
    print(f"   场景: {da2_config.get('scene', 'unknown')}")
    print(f"   设备: {da2_config.get('device', 'unknown')}")
    print(f"   检查点: {da2_config.get('checkpoint_path', 'unknown')}")
    
    # 加载模型
    print("\n⏳ 加载 Depth Anything V2 模型...")
    da2_predictor = DA2Predictor(da2_config)
    
    # 检查模型设备
    if hasattr(da2_predictor, 'model') and da2_predictor.model is not None:
        model_device = next(da2_predictor.model.parameters()).device
        print(f"\n✅ 模型设备: {model_device}")
        print(f"   在 GPU 上: {'是 ✅' if 'cuda' in str(model_device) else '否 ❌'}")
        
        # 检查模型精度
        model_dtype = next(da2_predictor.model.parameters()).dtype
        print(f"   模型精度: {model_dtype}")
        print(f"   使用 float32: {'是 ⚠️ ' if model_dtype == torch.float32 else '否 ✅'}")
    
    # 测试推理
    print("\n⏳ 测试 Depth Anything V2 推理速度...")
    if 'test_img' not in locals():
        test_img = cv2.imread(str(test_image_path))
    
    # 测试原始分辨率
    start = time.time()
    depth_map, _ = da2_predictor.predict(test_img, "test", "test")
    elapsed_original = time.time() - start
    print(f"\n📊 原始分辨率推理时间: {elapsed_original:.2f} 秒")
    
    # 测试缩放后分辨率
    scale_factor = 640 / max(test_img.shape[0], test_img.shape[1])
    new_h, new_w = int(test_img.shape[0] * scale_factor), int(test_img.shape[1] * scale_factor)
    test_img_resized = cv2.resize(test_img, (new_w, new_h))
    
    start = time.time()
    depth_map, _ = da2_predictor.predict(test_img_resized, "test", "test")
    elapsed_resized = time.time() - start
    print(f"📊 缩放后推理时间 (640px): {elapsed_resized:.2f} 秒")
    print(f"💡 加速比: {elapsed_original/elapsed_resized:.1f}x")
    
except Exception as e:
    print(f"\n❌ 加载 Depth Anything V2 失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 5. 加载并检查 SAM3
# ========================================
print("\n" + "="*80)
print("5️⃣  SAM3 模型检查")
print("="*80)

try:
    from src.pipeline.sam3_detector import SAM3Detector
    
    # 确保 config 已加载
    if 'config' not in locals():
        import yaml
        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    sam3_config = config['sam3']
    print(f"\n📦 模型配置:")
    print(f"   设备: {sam3_config.get('device', 'unknown')}")
    print(f"   检查点: {sam3_config.get('checkpoint_path', 'unknown')}")
    
    print("\n⏳ 加载 SAM3 模型...")
    sam3_detector = SAM3Detector(sam3_config)
    
    # 检查模型设备
    if hasattr(sam3_detector, 'model') and sam3_detector.model is not None:
        if hasattr(sam3_detector.model, 'model'):
            model_device = next(sam3_detector.model.model.parameters()).device
            print(f"\n✅ 模型设备: {model_device}")
            print(f"   在 GPU 上: {'是 ✅' if 'cuda' in str(model_device) else '否 ❌'}")
            
            model_dtype = next(sam3_detector.model.model.parameters()).dtype
            print(f"   模型精度: {model_dtype}")
            print(f"   使用 float32: {'是 ⚠️ ' if model_dtype == torch.float32 else '否 ✅'}")
    
    # 测试推理
    print("\n⏳ 测试 SAM3 推理速度...")
    if 'img_pil' not in locals():
        img_pil = Image.open(test_image_path)
    
    start = time.time()
    point_series = sam3_detector.segment(img_pil, str(test_image_path), "test")
    elapsed_sam3 = time.time() - start
    
    boxes_arrow = point_series.get("boxes_arrow", [])
    print(f"\n📊 SAM3 推理时间: {elapsed_sam3:.2f} 秒")
    print(f"   检测到箭头框: {len(boxes_arrow)} 个")
    
except Exception as e:
    print(f"\n❌ 加载 SAM3 失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 6. 加载并检查 Qwen3-VL
# ========================================
print("\n" + "="*80)
print("6️⃣  Qwen3-VL-8B 模型检查")
print("="*80)

try:
    from src.pipeline.qwen3vl_interpreter import Qwen3VLInterpreter
    
    # 确保 config 已加载
    if 'config' not in locals():
        import yaml
        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    
    qwen_config = config.get('qwen3vl', {})
    print(f"\n📦 模型配置:")
    print(f"   模型路径: {qwen_config.get('model_path', 'unknown')}")
    print(f"   设备: {qwen_config.get('device', 'unknown')}")
    
    print("\n⏳ 加载 Qwen3-VL 模型...")
    qwen_interpreter = Qwen3VLInterpreter(qwen_config)
    
    # 检查模型设备
    if hasattr(qwen_interpreter, 'model') and qwen_interpreter.model is not None:
        model_device = next(qwen_interpreter.model.parameters()).device
        print(f"\n✅ 模型设备: {model_device}")
        print(f"   在 GPU 上: {'是 ✅' if 'cuda' in str(model_device) else '否 ❌'}")
        
        model_dtype = next(qwen_interpreter.model.parameters()).dtype
        print(f"   模型精度: {model_dtype}")
        print(f"   使用 float32: {'是 ⚠️ ' if model_dtype == torch.float32 else '否 ✅'}")
        print(f"   使用 float16: {'是 ✅' if model_dtype == torch.float16 else '否'}")
        print(f"   使用 bfloat16: {'是 ✅' if model_dtype == torch.bfloat16 else '否'}")
    
    # 测试推理
    print("\n⏳ 测试 Qwen3-VL 推理速度...")
    if 'boxes_arrow' in locals() and len(boxes_arrow) > 0:
        # 创建一个简单的 point_series
        test_point_series = {
            "boxes_arrow": boxes_arrow[:1],  # 只测试一个框
            "masks_sign": []
        }
        
        start = time.time()
        items = qwen_interpreter.interpret_multiple_objects(
            str(test_image_path),
            test_point_series,
            "test_depth.tiff",
            "test_output"
        )
        elapsed_qwen = time.time() - start
        
        print(f"\n📊 Qwen3-VL 推理时间: {elapsed_qwen:.2f} 秒")
        print(f"   返回结果数: {len(items) if items else 0}")
    else:
        print("\n⚠️  未检测到箭头框,跳过 Qwen3-VL 测试")
        
except Exception as e:
    print(f"\n❌ 加载 Qwen3-VL 失败: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 7. CPU 使用率检查
# ========================================
print("\n" + "="*80)
print("7️⃣  CPU 和内存使用状态")
print("="*80)

cpu_percent = psutil.cpu_percent(interval=1)
memory = psutil.virtual_memory()
print(f"\n📊 CPU 使用率: {cpu_percent}%")
print(f"   {'⚠️  CPU使用率过高' if cpu_percent > 50 else '✅  CPU使用率正常'}")
print(f"\n📊 内存使用: {memory.percent}%")
print(f"   已用: {memory.used / 1024**3:.2f} GB")
print(f"   总计: {memory.total / 1024**3:.2f} GB")

# ========================================
# 8. 综合性能分析
# ========================================
print("\n" + "="*80)
print("8️⃣  综合性能分析")
print("="*80)

print("\n📊 预估每帧处理时间:")
if 'elapsed_original' in locals():
    print(f"   Depth Anything V2: {elapsed_original:.2f} 秒")
if 'elapsed_sam3' in locals():
    print(f"   SAM3: {elapsed_sam3:.2f} 秒")
if 'elapsed_qwen' in locals():
    print(f"   Qwen3-VL: {elapsed_qwen:.2f} 秒")

total_time = sum([
    elapsed_original if 'elapsed_original' in locals() else 0,
    elapsed_sam3 if 'elapsed_sam3' in locals() else 0,
    elapsed_qwen if 'elapsed_qwen' in locals() else 0
])

print(f"\n   总计 (单帧): {total_time:.2f} 秒")
if total_time > 0:
    print(f"   每帧FPS: {1/total_time:.2f}")
    print(f"   目标FPS: 10")
    print(f"   {'✅ 达到目标' if total_time <= 0.1 else '❌ 未达到目标'}")
else:
    print(f"   ⚠️  无法计算FPS (某些模型未测试)")

# ========================================
# 9. 优化建议
# ========================================
print("\n" + "="*80)
print("💡 优化建议")
print("="*80)

print("\n1️⃣  输入分辨率优化:")
print("   - 将所有输入图像缩放到最大 640px")
print("   - 可以减少 70-80% 的计算量")

print("\n2️⃣  模型精度优化:")
print("   - 使用 torch.float16 替代 float32")
print("   - 可以加速 2-3x,减少 50% 显存")

print("\n3️⃣  Depth Anything 优化:")
print("   - 如果使用的是 large (vitl),切换到 small (vits)")
print("   - small 版本速度提升 3-4x,精度损失 <5%")

print("\n4️⃣  Qwen3-VL 优化:")
print("   - 使用 4-bit 或 8-bit 量化 (bitsandbytes)")
print("   - 可以加速 2-3x,减少 60% 显存")

print("\n5️⃣  多 GPU 优化:")
print("   - 将不同模型分配到不同 GPU")
print("   - 例如: SAM3 → GPU 0, Depth → GPU 0, Qwen3-VL → GPU 1")

print("\n6️⃣  批量处理优化:")
print("   - 使用 torch.compile() 加速推理")
print("   - 启用 torch.backends.cudnn.benchmark = True")

print("\n" + "="*80)
