#!/usr/bin/env python3
"""
快速视频演示脚本 - 使用关键帧策略
适合快速生成demo，不需要处理每一帧

用法: python process_video_fast.py --input video.mp4

策略:
- 每隔N帧提取一个关键帧
- 只对关键帧进行深度学习推理（速度快）
- 中间帧使用原始视频或简单插值
- 最后合成输出视频

这样可以将处理时间从数小时降低到几分钟
"""
import argparse
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.pipeline.pipeline_manager import PipelineManager
from src.pipeline.video_processor import VideoProcessor
from src.pipeline.drawing_utils import DrawingUtils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastVideoDemo:
    """快速视频演示Pipeline"""
    
    def __init__(self, config_path: str):
        """初始化"""
        self.pipeline = PipelineManager(config_path)
        logger.info("FastVideoDemo 初始化完成")
    
    def process_video_fast(self,
                          video_path: str,
                          output_dir: str = "data/output/video_demo_fast",
                          keyframe_interval: int = 15) -> Dict:
        """
        快速处理视频（关键帧策略）
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            keyframe_interval: 关键帧间隔（每N帧提取一个)
            
        Returns:
            处理结果信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 初始化视频处理器
        logger.info(f"快速处理视频: {video_path}")
        video_processor = VideoProcessor(video_path, str(output_dir))
        
        # 获取视频信息
        video_info = video_processor.get_video_info()
        logger.info(f"视频信息: 分辨率 {video_info['width']}x{video_info['height']}, "
                   f"FPS {video_info['fps']}, 总帧数 {video_info['total_frames']}")
        logger.info(f"关键帧间隔: 每 {keyframe_interval} 帧处理一次")
        
        # 2. 直接打开视频文件，逐帧读取
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_frames_dir = output_dir / "output_frames"
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        
        results_json_dir = output_dir / "detection_results"
        results_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 读取所有帧（仅保存到内存，不写磁盘，节省空间）
        logger.info("第一步: 读取视频帧到内存...")
        all_frames = []
        frame_data = []
        frame_index = 0
        
        with tqdm(total=total_frames, desc="读取帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                all_frames.append(frame)
                frame_data.append({
                    'index': frame_index,
                    'is_keyframe': (frame_index % keyframe_interval == 0),
                    'detections': None
                })
                frame_index += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"✓ 读取完成，共 {len(all_frames)} 帧")
        
        # 4. 仅处理关键帧
        keyframe_indices = [i for i, fd in enumerate(frame_data) if fd['is_keyframe']]
        logger.info(f"第二步: 处理关键帧 ({len(keyframe_indices)} 个)")
        
        temp_output_depth = output_dir / "temp_depth"
        temp_output_sam3 = output_dir / "temp_sam3"
        temp_output_org = output_dir / "temp_organized"
        temp_output_correct = output_dir / "temp_correct"
        
        for d in [temp_output_depth, temp_output_sam3, temp_output_org, temp_output_correct]:
            d.mkdir(parents=True, exist_ok=True)
        
        keyframe_results = {}  # {frame_index: detections}
        
        with tqdm(total=len(keyframe_indices), desc="处理关键帧") as pbar:
            for kf_index in keyframe_indices:
                frame = all_frames[kf_index]
                
                # 临时保存帧为文件（pipeline需要文件路径）
                temp_frame_path = output_dir / f"temp_frame_{kf_index:06d}.jpg"
                cv2.imwrite(str(temp_frame_path), frame)
                
                try:
                    # 调用pipeline处理
                    detections = self.pipeline.process_single_image(
                        str(temp_frame_path),
                        str(temp_output_depth),
                        str(temp_output_sam3),
                        str(temp_output_org),
                        str(temp_output_correct)
                    )
                    
                    keyframe_results[kf_index] = detections
                    frame_data[kf_index]['detections'] = detections
                    
                    # 保存检测结果
                    results_file = results_json_dir / f"frame_{kf_index:06d}_detections.json"
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(detections, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    logger.warning(f"处理关键帧 {kf_index} 失败: {e}")
                    keyframe_results[kf_index] = []
                
                finally:
                    # 删除临时文件
                    if temp_frame_path.exists():
                        os.remove(temp_frame_path)
                
                pbar.update(1)
        
        logger.info(f"✓ 关键帧处理完成，共获得 {sum(len(d) for d in keyframe_results.values())} 个检测")
        
        # 5. 标注所有帧
        logger.info(f"第三步: 标注所有帧...")
        output_frame_files = []
        
        with tqdm(total=len(all_frames), desc="标注帧") as pbar:
            for i, frame in enumerate(all_frames):
                # 查找最近的关键帧结果
                detections = None
                
                if frame_data[i]['detections'] is not None:
                    # 这是关键帧，使用自己的检测结果
                    detections = frame_data[i]['detections']
                else:
                    # 这不是关键帧，使用最近的关键帧的结果
                    for kf_idx in sorted(keyframe_indices, reverse=True):
                        if kf_idx < i:
                            detections = keyframe_results.get(kf_idx, [])
                            break
                    if detections is None:
                        detections = []
                
                # 标注帧
                if detections:
                    annotated_frame = DrawingUtils.draw_multiple_annotations(frame, detections)
                else:
                    annotated_frame = frame
                
                # 保存标注后的帧
                output_frame_path = output_frames_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(output_frame_path), annotated_frame)
                output_frame_files.append(str(output_frame_path))
                
                pbar.update(1)
        
        logger.info(f"✓ 标注完成")
        
        # 6. 合成输出视频
        logger.info(f"第四步: 合成输出视频...")
        output_video = output_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_processor.frames_list_to_video(
            output_frame_files,
            str(output_video),
            fps=fps
        )
        
        # 7. 清理临时帧文件（保存空间）
        logger.info("清理临时文件...")
        for frame_file in output_frame_files:
            if Path(frame_file).exists():
                os.remove(frame_file)
        
        # 最后删除输出帧目录
        if output_frames_dir.exists():
            import shutil
            shutil.rmtree(output_frames_dir)
        
        # 8. 生成总结报告
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_video": video_path,
            "output_video": str(output_video),
            "video_info": video_info,
            "strategy": "关键帧处理 (快速)",
            "keyframe_interval": keyframe_interval,
            "total_frames": len(all_frames),
            "total_keyframes": len(keyframe_indices),
            "total_detections": sum(len(d) for d in keyframe_results.values()),
            "output_directory": str(output_dir)
        }
        
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 视频处理完成!")
        logger.info(f"📁 输出视频: {output_video}")
        logger.info(f"📊 处理总结: {summary_file}")
        
        return summary


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="快速视频处理和演示 (关键帧策略)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python process_video_fast.py --input my_video.mp4
  python process_video_fast.py --input my_video.mp4 --keyframe-interval 30
  python process_video_fast.py --input my_video.mp4 --output my_output --keyframe-interval 10

注:
  关键帧间隔越小，处理越精细但耗时越长
  - 5-10: 精细，耗时长
  - 15-30: 平衡（推荐）
  - 30+: 快速，可能遗漏检测
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入视频文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/output/video_demo_fast',
        help='输出目录 (默认: data/output/video_demo_fast)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--keyframe-interval',
        type=int,
        default=15,
        help='关键帧间隔 (默认: 15，即每15帧处理一次)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查输入视频存在
    if not os.path.exists(args.input):
        logger.error(f"输入视频不存在: {args.input}")
        sys.exit(1)
    
    try:
        # 创建pipeline
        pipeline = FastVideoDemo(args.config)
        
        # 处理视频
        result = pipeline.process_video_fast(
            video_path=args.input,
            output_dir=args.output,
            keyframe_interval=args.keyframe_interval
        )
        
        print("\n" + "="*60)
        print("🎉 视频演示生成成功！")
        print("="*60)
        print(f"输入视频: {args.input}")
        print(f"输出视频: {result['output_video']}")
        print(f"总帧数: {result['total_frames']}")
        print(f"处理帧数: {result['total_keyframes']} (关键帧)")
        print(f"检测结果: {result['total_detections']} 个目标")
        print(f"策略: {result['strategy']}")
        print(f"输出目录: {result['output_directory']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
