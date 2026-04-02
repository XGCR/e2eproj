#!/usr/bin/env python3
"""
测试导航生成和语音播报功能
使用已有的结果数据进行测试
"""
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.navigation_generator import NavigationGenerator
from src.pipeline.tts_player import TTSPlayer

def load_depth_image(depth_path):
    """加载深度图像"""
    from PIL import Image
    depth_img = Image.open(depth_path)
    return np.array(depth_img)

def main():
    print("=" * 60)
    print("  测试导航生成和语音播报功能")
    print("=" * 60)
    
    # 配置
    data_root = Path(__file__).parent.parent / "data"
    test_images = ["test2", "test4", "test5"]
    
    # 初始化导航生成器（使用模板模式）
    nav_config = {
        "enabled": True,
        "output_language": "en",
        "device": "cuda",
        "max_new_tokens": 128,
    }
    navigation = NavigationGenerator(nav_config)
    
    # 初始化TTS播放器（禁用语音输出，仅显示文字）
    tts_config = {
        "enabled": False,  # 禁用语音，仅显示文字
        "backend": "system",
        "language": "en",
        "voice_hint": "English",
    }
    tts = TTSPlayer(tts_config)
    
    print(f"\n使用 NavigationGenerator 模式: {'Qwen3 LLM' if navigation.use_llm else '英文模板回退'}\n")
    
    for test_name in test_images:
        print("-" * 60)
        print(f"处理图像: {test_name}")
        print("-" * 60)
        
        # 加载已有的正确结果
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
        
        print(f"\n  原始结果数量: {len(correct_data.get('correct_results', []))}")
        print(f"\n  【导航播报文字】")
        print(f"  {navigation_sentence}")
        
        # TTS文字播报（已禁用语音，仅返回状态）
        tts_result = tts.speak(navigation_sentence)
        print(f"\n  TTS状态: {tts_result.get('tts_status', 'unknown')}")
        
        print()
    
    print("=" * 60)
    print("  测试完成!")
    print("=" * 60)
    
    # 清理
    navigation.release()
    tts.release()

if __name__ == "__main__":
    main()
