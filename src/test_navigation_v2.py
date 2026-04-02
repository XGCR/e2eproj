#!/usr/bin/env python3
"""
测试 Qwen3VL 识别路牌方向和文字，并生成导航语句
"""
import sys
import json
import numpy as np
from pathlib import Path
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.navigation_generator import NavigationGenerator
from src.pipeline.tts_player import TTSPlayer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_depth_image(depth_path):
    """加载深度图像"""
    from PIL import Image
    depth_img = Image.open(depth_path)
    return np.array(depth_img)

def main():
    print("=" * 70)
    print("  测试 Qwen3VL 路牌识别 + 导航生成")
    print("=" * 70)
    
    # 配置
    data_root = Path(__file__).parent.parent / "data"
    test_images = ["test2", "test4", "test5"]
    output_language = "zh"  # 中文输出
    
    # 初始化导航生成器
    nav_config = {
        "enabled": True,
        "output_language": output_language,
        "device": "cuda",
        "max_new_tokens": 128,
    }
    navigation = NavigationGenerator(nav_config)
    
    # 初始化TTS播放器（禁用语音输出，仅显示文字）
    tts_config = {
        "enabled": False,
        "backend": "system",
        "language": "en",
        "voice_hint": "English",
    }
    tts = TTSPlayer(tts_config)
    
    print(f"\n导航输出语言: {'中文' if output_language == 'zh' else '英文'}")
    print(f"使用 NavigationGenerator 模式: {'Qwen3 LLM' if navigation.use_llm else '模板回退'}\n")
    
    for test_name in test_images:
        print("-" * 70)
        print(f"处理图像: {test_name}")
        print("-" * 70)
        
        # 加载已有的正确结果（包含 Qwen3VL 识别结果）
        correct_json_path = data_root / "output" / "correct" / f"{test_name}_correct.json"
        depth_tiff_path = data_root / "output" / "depth_image" / f"{test_name}_depth.tiff"
        image_path = data_root / "input" / "image" / f"{test_name}.jpg"
        
        if not correct_json_path.exists():
            print(f"  [跳过] 结果文件不存在: {correct_json_path}")
            continue
            
        with open(correct_json_path, 'r', encoding='utf-8') as f:
            correct_data = json.load(f)
        
        if not depth_tiff_path.exists():
            print(f"  [跳过] 深度图像不存在: {depth_tiff_path}")
            continue
        
        # 显示 Qwen3VL 识别结果
        results = correct_data.get('correct_results', [])
        if results:
            print(f"\n  Qwen3VL 识别结果:")
            for i, result in enumerate(results):
                alpha = result.get('alpha_deg', result.get('alpha', 'N/A'))
                sign_text = result.get('sign_text', '(无)')
                print(f"    [{i+1}] 方向角度: {alpha}, 路牌文字: {sign_text}")
        
        # 加载深度图
        depth_map = load_depth_image(depth_tiff_path)
        
        # 生成导航
        nav_result = navigation.process(
            str(image_path),
            depth_map,
            correct_data,
            str(data_root / "output" / "navigation")
        )
        
        # 显示导航文字到终端
        navigation_sentence = nav_result.get('navigation_sentence', '')
        
        print(f"\n  【导航播报文字】")
        print(f"  ┌{'─'*68}┐")
        print(f"  │ {navigation_sentence:<66} │")
        print(f"  └{'─'*68}┘")
        
        # TTS状态
        tts_result = tts.speak(navigation_sentence)
        print(f"\n  TTS状态: {tts_result.get('tts_status', 'unknown')}")
        print()
    
    print("=" * 70)
    print("  测试完成!")
    print("=" * 70)
    
    # 清理
    navigation.release()
    tts.release()

if __name__ == "__main__":
    main()
