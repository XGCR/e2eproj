#!/usr/bin/env python3
"""
视频处理脚本：提取视频帧 → 处理 → 合成结果视频
"""
import os
import sys
import cv2
import argparse
import subprocess
import shutil
from pathlib import Path
import logging
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_frames_from_video(video_path, output_folder, fps=None):
    """从视频中提取所有帧"""
    logger.info(f"正在从视频中提取帧: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return 0, 0
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return 0, 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"视频信息:")
    logger.info(f"  总帧数: {total_frames}")
    logger.info(f"  帧率: {video_fps} fps")
    logger.info(f"  分辨率: {video_width}x{video_height}")
    
    if fps is not None and fps > 0:
        frame_interval = max(1, int(video_fps / fps))
        logger.info(f"  将每 {frame_interval} 帧提取 1 帧 (目标: {fps} fps)")
    else:
        frame_interval = 1
        fps = video_fps
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"{extracted_count + 1}.png")
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                logger.info(f"已提取 {extracted_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    
    logger.info(f"✓ 共提取 {extracted_count} 帧")
    return extracted_count, fps

def run_pipeline(config_path=None):
    """运行处理管道"""
    logger.info("启动处理管道...")
    
    cmd = ["python", "src/main.py"]
    
    if config_path:
        cmd.extend(["--config", config_path])
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), check=True)
        logger.info("✓ 处理管道完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ 处理管道失败: {e}")
        return False

def run_visualization():
    """运行可视化"""
    logger.info("启动可视化...")
    
    try:
        result = subprocess.run(["python", "src/visualize.py"], 
                              cwd=os.getcwd(), 
                              check=True)
        logger.info("✓ 可视化完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ 可视化失败: {e}")
        return False

def clean_input_folder():
    """清空输入文件夹"""
    input_folder = "data/input/image"
    
    if os.path.exists(input_folder):
        logger.info(f"清空输入文件夹: {input_folder}")
        for file in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"无法删除文件 {file_path}: {e}")
    else:
        os.makedirs(input_folder, exist_ok=True)

def find_video_files(video_folder="data/input/video"):
    """查找所有视频文件"""
    video_files = []
    
    if not os.path.exists(video_folder):
        logger.error(f"视频文件夹不存在: {video_folder}")
        return video_files
    
    # 查找所有 mp4 文件
    for file in os.listdir(video_folder):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(video_folder, file)
            if os.path.isfile(full_path):
                video_files.append(full_path)
    
    # 按文件名排序
    video_files.sort()
    
    return video_files

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频处理管道：自动处理 data/input/video/ 中的所有视频"
    )
    
    parser.add_argument(
        '--video-folder', '-vf',
        type=str,
        default='data/input/video',
        help='视频文件夹路径 (默认: data/input/video)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=10,
        help='从视频中提取帧的帧率 (默认: 10 fps)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='跳过清空输入文件夹的步骤'
    )
    
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='跳过处理管道步骤'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='跳过可视化步骤'
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    logger.info("=" * 60)
    logger.info("视频处理管道启动")
    logger.info("=" * 60)
    
    total_start_time = time.time()
    
    # 查找所有视频文件
    logger.info(f"\n查找视频文件夹: {args.video_folder}")
    video_files = find_video_files(args.video_folder)
    
    if not video_files:
        logger.error(f"✗ 在 {args.video_folder} 中没有找到视频文件")
        logger.info("提示: 请将 MP4 视频放在 data/input/video/ 文件夹中")
        sys.exit(1)
    
    logger.info(f"✓ 找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        logger.info(f"  {i}. {os.path.basename(video)}")
    
    # 处理每个视频
    total_extracted = 0
    for video_idx, video_path in enumerate(video_files, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"处理视频 {video_idx}/{len(video_files)}: {os.path.basename(video_path)}")
        logger.info("=" * 60)
        
        video_start_time = time.time()
        
        # Step 1: 清空输入文件夹
        if not args.skip_clean:
            logger.info("\n[步骤 1/4] 清空输入文件夹")
            clean_input_folder()
        else:
            logger.info("\n[步骤 1/4] 跳过清空输入文件夹")
        
        # Step 2: 提取视频帧
        logger.info("\n[步骤 2/4] 提取视频帧")
        extract_start = time.time()
        extracted_count, fps = extract_frames_from_video(
            video_path, 
            "data/input/image",
            args.fps
        )
        extract_time = time.time() - extract_start
        logger.info(f"✓ 帧提取耗时: {extract_time:.2f}s")
        
        if extracted_count == 0:
            logger.error(f"✗ 没有提取到任何帧，跳过此视频")
            continue
        
        total_extracted += extracted_count
        
        # Step 3: 运行处理管道
        if not args.skip_pipeline:
            logger.info("\n[步骤 3/4] 运行处理管道")
            pipeline_start = time.time()
            if not run_pipeline(args.config):
                logger.error(f"✗ 处理管道失败，跳过此视频")
                continue
            pipeline_time = time.time() - pipeline_start
            logger.info(f"✓ 处理管道耗时: {pipeline_time:.2f}s")
        else:
            logger.info("\n[步骤 3/4] 跳过处理管道")
        
        # Step 4: 可视化结果
        if not args.skip_visualization:
            logger.info("\n[步骤 4/4] 可视化结果")
            viz_start = time.time()
            if not run_visualization():
                logger.error("✗ 可视化失败，但处理已完成")
            else:
                viz_time = time.time() - viz_start
                logger.info(f"✓ 可视化耗时: {viz_time:.2f}s")
        else:
            logger.info("\n[步骤 4/4] 跳过可视化")
        
        video_time = time.time() - video_start_time
        logger.info(f"\n✓ 视频 {video_idx} 处理完成 (总耗时: {video_time:.2f}s)")
    
    # 完成信息
    total_time = time.time() - total_start_time
    logger.info("\n" + "=" * 60)
    logger.info("✓ 所有视频处理完成！")
    logger.info("=" * 60)
    logger.info(f"\n处理统计:")
    logger.info(f"  - 处理的视频数: {len(video_files)}")
    logger.info(f"  - 提取的总帧数: {total_extracted}")
    logger.info(f"  - 总耗时: {total_time:.2f}s")
    logger.info(f"  - 平均每个视频: {total_time / len(video_files):.2f}s")
    logger.info(f"\n输出文件夹:")
    logger.info(f"  * 深度图: data/visualize/depth/")
    logger.info(f"  * 图像结果: data/visualize/image/")
    logger.info(f"  * 并排结果: data/visualize/image_depth/")
    logger.info(f"  * 视频文件: data/visualize/video/")
    logger.info("")

if __name__ == "__main__":
    main()
