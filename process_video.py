#!/usr/bin/env python3
"""
视频处理完整脚本
用法: python process_video.py --input video.mp4 --config configs/default_config.yaml

功能:
1. 提取视频帧
2. 逐帧处理（深度估计、对象分割、文字识别等）
3. 标注识别结果（框、方向、文字）
4. 合成输出视频
"""
import argparse
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict
import cv2

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
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


class VideoAnalysisPipeline:
    """视频分析完整Pipeline"""
    
    def __init__(self, config_path: str):
        """初始化"""
        self.pipeline = PipelineManager(config_path)
        self.config_path = config_path
        logger.info("VideoAnalysisPipeline 初始化完成")
    
    def process_video(self, 
                     video_path: str,
                     output_dir: str = "data/output/video_demo",
                     sample_rate: int = 1) -> Dict:
        """
        处理视频
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            sample_rate: 采样率（1表示处理每一帧）
            
        Returns:
            处理结果信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 初始化视频处理器
        logger.info(f"处理视频: {video_path}")
        video_processor = VideoProcessor(video_path, str(output_dir))
        
        # 获取视频信息
        video_info = video_processor.get_video_info()
        logger.info(f"视频信息: 分辨率 {video_info['width']}x{video_info['height']}, "
                   f"FPS {video_info['fps']}, 总帧数 {video_info['total_frames']}")
        
        # 2. 提取帧
        frames_dir = output_dir / "extracted_frames"
        logger.info(f"第一步: 提取视频帧到 {frames_dir}")
        frame_files = video_processor.extract_frames(str(frames_dir), sample_rate=sample_rate)
        
        # 3. 准备输出目录
        annotation_frames_dir = output_dir / "annotated_frames"
        annotation_frames_dir.mkdir(parents=True, exist_ok=True)
        
        results_json_dir = output_dir / "detection_results"
        results_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. 逐帧处理
        logger.info(f"第二步: 逐帧处理 ({len(frame_files)} 帧)")
        annotated_frame_files = []
        all_detections = {}
        
        for i, frame_file in enumerate(frame_files):
            try:
                logger.info(f"处理帧 {i+1}/{len(frame_files)}: {os.path.basename(frame_file)}")
                
                # 运行pipeline处理该帧
                frame_name = Path(frame_file).stem
                
                # 使用pipeline处理（这会产生检测结果JSON）
                # 注意：需要临时目录来保存中间结果
                temp_output_depth = output_dir / "temp_depth"
                temp_output_sam3 = output_dir / "temp_sam3"
                temp_output_org = output_dir / "temp_organized"
                temp_output_correct = output_dir / "temp_correct"
                
                for d in [temp_output_depth, temp_output_sam3, temp_output_org, temp_output_correct]:
                    d.mkdir(parents=True, exist_ok=True)
                
                # 调用pipeline处理单张图像
                detections = self.pipeline.process_single_image(
                    frame_file,
                    str(temp_output_depth),
                    str(temp_output_sam3),
                    str(temp_output_org),
                    str(temp_output_correct)
                )
                
                # 5. 标注帧
                image = cv2.imread(frame_file)
                if image is None:
                    logger.warning(f"无法读取帧: {frame_file}")
                    continue
                
                # 如果有检测结果，标注到图像上
                if detections:
                    annotated_image = DrawingUtils.draw_multiple_annotations(image, detections)
                else:
                    annotated_image = image
                    logger.info(f"帧 {frame_name} 无检测结果")
                
                # 保存标注后的帧
                annotated_frame_path = annotation_frames_dir / f"{frame_name}_annotated.jpg"
                cv2.imwrite(str(annotated_frame_path), annotated_image)
                annotated_frame_files.append(str(annotated_frame_path))
                
                # 保存检测结果JSON
                all_detections[frame_name] = detections
                results_file = results_json_dir / f"{frame_name}_detections.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(detections, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"处理帧 {frame_file} 失败: {e}")
                # 使用原始帧继续
                annotated_frame_files.append(frame_file)
                continue
        
        # 6. 合成输出视频
        logger.info(f"第三步: 合成输出视频")
        output_video = output_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_processor.frames_list_to_video(
            annotated_frame_files,
            str(output_video),
            fps=video_processor.fps
        )
        
        # 7. 生成总结报告
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_video": video_path,
            "output_video": str(output_video),
            "video_info": video_info,
            "total_frames_processed": len(annotated_frame_files),
            "total_detections": sum(len(dets) for dets in all_detections.values()),
            "output_directories": {
                "annotated_frames": str(annotation_frames_dir),
                "detection_results": str(results_json_dir),
                "temp_dir": str(output_dir / "temp_*")
            }
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
        description="视频处理和分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python process_video.py --input my_video.mp4
  python process_video.py --input my_video.mp4 --output data/output/my_demo --sample 2
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
        default='data/output/video_demo',
        help='输出目录 (默认: data/output/video_demo)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=1,
        help='采样率：每N帧进行一次处理 (默认: 1，处理每一帧)'
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
        pipeline = VideoAnalysisPipeline(args.config)
        
        # 处理视频
        result = pipeline.process_video(
            video_path=args.input,
            output_dir=args.output,
            sample_rate=args.sample
        )
        
        print("\n" + "="*60)
        print("🎉 视频处理成功!")
        print("="*60)
        print(f"输入视频: {args.input}")
        print(f"输出视频: {result['output_video']}")
        print(f"处理帧数: {result['total_frames_processed']}")
        print(f"检测结果: {result['total_detections']} 个检测目标")
        print(f"输出目录: {args.output}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
