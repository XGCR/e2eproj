#!/usr/bin/env python3
"""
多目标深度语义Pipeline主程序
"""
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import glob
import os
import re 

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.pipeline_manager import PipelineManager

def read_filename(image_folder):
    # 遍历所有输入图像
    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))

    # 使用正则表达式提取数字部分并排序
    def extract_number(filename):
        # 从文件名中提取数字部分，例如从"XXX1.jpg"提取"1"
        match = re.search(r'(\d+)\.jpg$', filename)
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
    
    # try:
    # 创建Pipeline管理器
    logger.info("Initializing Pipeline Manager...")
    pipeline = PipelineManager(args.config)
    
    # 准备输入输出路径
    paths = pipeline.config.get('paths')

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

    output_navigation_folder_str = paths.get('output_navigation_folder')
    output_navigation_folder = Path(output_navigation_folder_str)
    output_navigation_folder.mkdir(parents=True, exist_ok=True)

    input_image_folder_str = paths.get('input_image_folder')
    image_files = read_filename(input_image_folder_str)
    try:
        for img_file in image_files:
            # 处理图像
            if os.path.isfile(img_file):
                # img_file
                logger.info(f"Processing single image: {img_file}")
                pipeline.process_single_image(
                                            img_file, 
                                            output_depth_folder_str,
                                            output_sam3_json_folder_str,
                                            temp_organized_json_folder_str,
                                            output_correct_json_folder_str,
                                            output_navigation_folder_str
                                            )
                                    
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