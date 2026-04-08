#!/usr/bin/env python3
"""
多目标深度语义Pipeline主程序
"""
import os
import sys

# 在导入 torch 之前设置环境变量，解决显存碎片化问题
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
from pathlib import Path
import logging
from datetime import datetime
import glob
import re 
import torch
import gc
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.pipeline_manager import PipelineManager

def read_filename(image_folder):
    # 遍历所有输入图像（支持多种格式）
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    # 使用正则表达式提取数字部分并排序
    def extract_number(filename):
        # 从文件名中提取数字部分，例如从"XXX1.jpg"提取"1"
        match = re.search(r'(\d+)\.(jpg|jpeg|png|bmp|JPG|JPEG|PNG|BMP)$', filename)
        return int(match.group(1)) if match else 0

    # 按数字序号排序
    image_files.sort(key=extract_number)

    print(f"找到 {len(image_files)} 张图像，按序号排序:")
    for img_file in image_files:
        print(f"  {img_file}")

    return image_files

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多目标深度语义Pipeline"
    )    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )

    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    logger = logging.getLogger(__name__)
    
    # 准备输入输出路径（只需准备一次）
    config_path = args.config
    temp_config = PipelineManager(config_path)
    paths = temp_config.config.get('paths')
    del temp_config  # 删除临时对象
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    output_depth_folder_str = paths.get('output_depth_folder')
    output_depth_folder = Path(output_depth_folder_str)
    output_depth_folder.mkdir(parents=True, exist_ok=True)

    output_sam3_json_folder_str = paths.get('output_sam3_json_folder')
    output_sam3_json_folder = Path(output_sam3_json_folder_str)
    output_sam3_json_folder.mkdir(parents=True, exist_ok=True)

    temp_organized_json_folder_str = paths.get('temp_organized_json_folder')
    temp_organized_json_folder = Path(temp_organized_json_folder_str)
    temp_organized_json_folder.mkdir(parents=True, exist_ok=True)

    output_correct_json_folder_str = paths.get('output_correct_json_folder')
    output_correct_json_folder = Path(output_correct_json_folder_str)
    output_correct_json_folder.mkdir(parents=True, exist_ok=True)

    input_image_folder_str = paths.get('input_image_folder')
    image_files = read_filename(input_image_folder_str)
    try:
        for idx, img_file in enumerate(image_files):
            # 处理图像
            if os.path.isfile(img_file):
                logger.info(f"=== Processing image {idx+1}/{len(image_files)}: {img_file} ===")
                
                # 在创建新 pipeline 前强力清理显存
                if idx > 0:
                    cleanup_start = time.time()
                    logger.info("Aggressive GPU memory cleanup...")
                    for _ in range(3):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        gc.collect()
                    cleanup_time = time.time() - cleanup_start
                    print(f"✓ GPU cleanup took {cleanup_time:.2f}s")
                
                # 为每张图像创建新的 Pipeline（确保显存是干净的）
                pipeline_start = time.time()
                logger.info("Creating fresh pipeline for this image...")
                pipeline = PipelineManager(config_path)
                pipeline_time = time.time() - pipeline_start
                print(f"✓ Pipeline creation took {pipeline_time:.2f}s")
                
                try:
                    process_start = time.time()
                    pipeline.process_single_image(
                                                img_file, 
                                                output_depth_folder_str,
                                                output_sam3_json_folder_str,
                                                temp_organized_json_folder_str,
                                                output_correct_json_folder_str
                                                )
                    process_time = time.time() - process_start
                    print(f"✓ Image processing took {process_time:.2f}s")
                finally:
                    # 处理完这张图像后，立即释放 pipeline
                    release_start = time.time()
                    logger.info("Releasing pipeline to free GPU memory...")
                    pipeline.release()
                    del pipeline
                    release_time = time.time() - release_start
                    print(f"✓ Pipeline release took {release_time:.2f}s")
                    
                    # 激进清理显存，为下一张图像腾出空间
                    logger.info("Clearing GPU memory...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                                    
            else:
                logger.error(f"Input path does not exist: {img_file}")
                sys.exit(1)

        logger.info("Pipeline processing completed successfully!")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # 清理资源
        if 'pipeline' in locals():
            pipeline.release()

if __name__ == "__main__":
    main()