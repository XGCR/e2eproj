#!/usr/bin/env python3
"""
超快速视频演示脚本 - 两阶段处理策略
第一阶段: 快速扫描所有帧,找出有箭头的帧
第二阶段: 只处理10%的箭头帧,其余用相邻帧替代

用法: python process_video_smart.py --input video.mp4

策略:
- 第一阶段: SAM3快速扫描所有帧,标记有箭头的帧(不调用Qwen3-VL)
- 第二阶段: 从有箭头的帧中抽样10%进行完整处理(Depth+Qwen3-VL)
- 未处理的箭头帧使用最近的处理帧的结果
- 无箭头帧使用原始图像
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
from PIL import Image

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


class SmartVideoDemo:
    """超快速视频演示Pipeline - 两阶段处理策略"""
    
    def __init__(self, config_path: str):
        """初始化"""
        self.pipeline = PipelineManager(config_path)
        logger.info("SmartVideoDemo 初始化完成")
    
    def process_video_smart(self,
                           video_path: str,
                           output_dir: str = "data/output/video_demo_smart",
                           sample_ratio: float = 0.05,
                           use_existing_sam3: bool = True,
                           only_largest_arrow: bool = True) -> Dict:
        """
        超快速处理视频（两阶段处理策略）
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            sample_ratio: 抽样比例,从有箭头的帧中抽取多少进行处理 (默认0.05即5%)
            use_existing_sam3: 是否使用已有的SAM3扫描结果 (默认True)
            only_largest_arrow: 是否只处理每帧中最大的箭头框 (默认True)
            
        Returns:
            处理结果信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 初始化视频处理器
        logger.info(f"智能处理视频: {video_path}")
        video_processor = VideoProcessor(video_path, str(output_dir))
        
        # 获取视频信息
        video_info = video_processor.get_video_info()
        logger.info(f"视频信息: 分辨率 {video_info['width']}x{video_info['height']}, "
                   f"FPS {video_info['fps']}, 总帧数 {video_info['total_frames']}")
        logger.info(f"抽样比例: {sample_ratio*100:.0f}% (将只处理有箭头帧的{sample_ratio*100:.0f}%)")
        logger.info(f"只处理最大箭头框: {only_largest_arrow}")
        logger.info(f"使用已有SAM3结果: {use_existing_sam3}")
        
        # 2. 直接打开视频文件，逐帧读取
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_frames_dir = output_dir / "output_frames"
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        
        results_json_dir = output_dir / "detection_results"
        results_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时目录
        temp_output_depth = output_dir / "temp_depth"
        temp_output_sam3 = output_dir / "temp_sam3"
        temp_output_org = output_dir / "temp_organized"
        temp_output_correct = output_dir / "temp_correct"
        
        for d in [temp_output_depth, temp_output_sam3, temp_output_org, temp_output_correct]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 3. 读取所有帧
        logger.info("第一步: 读取视频帧到内存...")
        all_frames = []
        frame_index = 0
        
        with tqdm(total=total_frames, desc="读取帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                all_frames.append(frame)
                frame_index += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"✓ 读取完成,共 {len(all_frames)} 帧")
        
        # ========== 第一阶段: 快速扫描所有帧,找出有箭头的帧 ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"第一阶段: 快速扫描所有帧 (SAM3检测,不调用Qwen3-VL)")
        logger.info(f"{'='*60}")
        
        frames_with_arrow = []  # 存储有箭头的帧索引
        frame_arrow_counts = {}  # 每帧的箭头数量
        frame_arrow_boxes = {}  # 每帧的箭头框信息 (用于只处理最大框)
        
        if use_existing_sam3 and temp_output_sam3.exists() and len(list(temp_output_sam3.glob('*.json'))) > 0:
            logger.info(f"使用已有的SAM3扫描结果: {temp_output_sam3}")
            # 读取已有的SAM3结果
            sam3_files = sorted(temp_output_sam3.glob('temp_frame_*.json'))
            for sam3_file in tqdm(sam3_files, desc="加载SAM3结果"):
                frame_idx = int(sam3_file.stem.split('_')[2])
                try:
                    with open(sam3_file, 'r') as f:
                        point_series = json.load(f)
                    boxes_arrow = point_series.get("boxes_arrow", [])
                    
                    if len(boxes_arrow) > 0:
                        frames_with_arrow.append(frame_idx)
                        frame_arrow_counts[frame_idx] = len(boxes_arrow)
                        frame_arrow_boxes[frame_idx] = boxes_arrow
                except Exception as e:
                    logger.warning(f"读取SAM3文件 {sam3_file} 失败: {e}")
        else:
            # 重新扫描
            logger.info("重新运行SAM3扫描...")
            # 确保目录存在
            temp_output_sam3.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(tqdm(all_frames, desc="扫描箭头")):
                try:
                    # 临时保存帧
                    temp_frame_path = output_dir / f"temp_frame_{i:06d}.jpg"
                    cv2.imwrite(str(temp_frame_path), frame)
                    
                    # SAM3 快速检测箭头
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image)
                    
                    sam3_detector = self.pipeline.components['sam3']
                    point_series = sam3_detector.segment(
                        image_pil,
                        str(temp_frame_path),
                        str(temp_output_sam3)
                    )
                    
                    boxes_arrow = point_series.get("boxes_arrow", [])
                    
                    if len(boxes_arrow) > 0:
                        frames_with_arrow.append(i)
                        frame_arrow_counts[i] = len(boxes_arrow)
                        frame_arrow_boxes[i] = boxes_arrow
                    
                except Exception as e:
                    logger.warning(f"扫描帧 {i} 失败: {e}")
                
                finally:
                    if temp_frame_path.exists():
                        os.remove(temp_frame_path)
        
        # 如果启用了只处理最大箭头框,则只保留最大的框
        if only_largest_arrow:
            logger.info("过滤箭头框: 每帧只保留最大的箭头框...")
            for frame_idx in frames_with_arrow:
                boxes = frame_arrow_boxes.get(frame_idx, [])
                if len(boxes) > 1:
                    # 计算每个框的面积,保留最大的
                    max_area = 0
                    max_box = None
                    for box in boxes:
                        if len(box) == 4:
                            # 计算四边形面积 (使用对角点)
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            area = width * height
                            if area > max_area:
                                max_area = area
                                max_box = box
                    
                    if max_box:
                        frame_arrow_boxes[frame_idx] = [max_box]
                        frame_arrow_counts[frame_idx] = 1
            
            logger.info("✓ 箭头框过滤完成")
        
        logger.info(f"\n✓ 第一阶段完成!")
        logger.info(f"  总帧数: {len(all_frames)}")
        logger.info(f"  有箭头的帧: {len(frames_with_arrow)}")
        logger.info(f"  无箭头的帧: {len(all_frames) - len(frames_with_arrow)}")
        
        # ========== 第二阶段: 从有箭头的帧中抽样处理 ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"第二阶段: 抽样处理箭头帧 (Depth + Qwen3-VL)")
        logger.info(f"{'='*60}")
        
        # 计算需要处理的帧数
        num_frames_to_process = max(1, int(len(frames_with_arrow) * sample_ratio))
        
        # 均匀抽样: 从有箭头的帧中均匀选取
        if len(frames_with_arrow) <= num_frames_to_process:
            # 如果箭头帧太少,全部处理
            selected_frames = frames_with_arrow
        else:
            # 均匀抽样
            step = len(frames_with_arrow) / num_frames_to_process
            selected_frames = [frames_with_arrow[int(i * step)] for i in range(num_frames_to_process)]
        
        logger.info(f"  需要处理的帧: {len(selected_frames)} (从 {len(frames_with_arrow)} 个箭头帧中抽样)")
        logger.info(f"  抽样帧索引: {selected_frames[:10]}... (显示前10个)")
        
        # 处理选中的帧
        processed_results = {}  # {frame_index: detections}
        
        for idx, frame_idx in enumerate(tqdm(selected_frames, desc="处理选中帧")):
            try:
                frame = all_frames[frame_idx]
                
                # 临时保存帧
                temp_frame_path = output_dir / f"temp_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(temp_frame_path), frame)
                
                # 完整处理 (Depth + Qwen3-VL + Correct)
                # 方案A: 直接调用各个组件,避免 pipeline 重复调用 SAM3
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image)
                image_np = np.array(image_pil)
                
                # 1. Depth Anything
                da2_predictor = self.pipeline.components['da2']
                depth_map, depth_image_path = da2_predictor.predict(
                    image_np,
                    str(temp_frame_path),
                    str(temp_output_depth)
                )
                
                # 2. 使用已有的SAM3结果(已过滤最大箭头框)
                if only_largest_arrow:
                    sam3_detector = self.pipeline.components['sam3']
                    point_series = sam3_detector.segment(
                        image_pil,
                        str(temp_frame_path),
                        str(temp_output_sam3)
                    )
                    
                    boxes_arrow = point_series.get("boxes_arrow", [])
                    if len(boxes_arrow) > 1:
                        # 找到最大的框
                        max_area = 0
                        max_box = None
                        for box in boxes_arrow:
                            if len(box) == 4:
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                width = max(x_coords) - min(x_coords)
                                height = max(y_coords) - min(y_coords)
                                area = width * height
                                if area > max_area:
                                    max_area = area
                                    max_box = box
                        
                        if max_box:
                            # 只保留最大的框
                            point_series["boxes_arrow"] = [max_box]
                            logger.debug(f"帧 {frame_idx}: 从 {len(boxes_arrow)} 个箭头框中保留最大的1个")
                    
                    # 保存过滤后的 SAM3 结果
                    base_name = os.path.splitext(os.path.basename(temp_frame_path))[0]
                    output_point = f"{base_name}_point.json"
                    output_point_file_path = os.path.join(temp_output_sam3, output_point)
                    with open(output_point_file_path, 'w', encoding='utf-8') as f:
                        json.dump(point_series, f, indent=2, ensure_ascii=False)
                else:
                    # 不使用过滤,直接读取已有的SAM3结果
                    base_name = os.path.splitext(os.path.basename(temp_frame_path))[0]
                    output_point = f"{base_name}_point.json"
                    output_point_file_path = os.path.join(temp_output_sam3, output_point)
                    if os.path.exists(output_point_file_path):
                        with open(output_point_file_path, 'r') as f:
                            point_series = json.load(f)
                    else:
                        # 如果没有现成的,重新运行SAM3
                        sam3_detector = self.pipeline.components['sam3']
                        point_series = sam3_detector.segment(
                            image_pil,
                            str(temp_frame_path),
                            str(temp_output_sam3)
                        )
                
                # 3. Qwen3-VL 解释 (使用过滤后的 point_series)
                items = self.pipeline.components['qwen3vl'].interpret_multiple_objects(
                    str(temp_frame_path),
                    point_series,
                    depth_image_path,
                    str(temp_output_org)
                )
                
                # 4. 角度修正
                correct_result = self.pipeline.components['correct'].process(
                    str(temp_frame_path),
                    image_np,
                    depth_map,
                    items,
                    str(temp_output_correct)
                )
                
                detections = correct_result if correct_result else []
                
                processed_results[frame_idx] = detections if detections else []
                
            except Exception as e:
                logger.warning(f"处理帧 {frame_idx} 失败: {e}")
                processed_results[frame_idx] = []
            
            finally:
                if temp_frame_path.exists():
                    os.remove(temp_frame_path)
        
        logger.info(f"\n✓ 第二阶段完成! 处理了 {len(processed_results)} 帧")
        
        # ========== 第三阶段: 为所有帧分配检测结果 ==========
        logger.info(f"\n{'='*60}")
        logger.info(f"第三阶段: 为所有帧分配检测结果")
        logger.info(f"{'='*60}")
        
        all_detections = []
        stats = {
            'total_frames': len(all_frames),
            'frames_with_arrow': len(frames_with_arrow),
            'frames_without_arrow': len(all_frames) - len(frames_with_arrow),
            'qwen_calls': len(processed_results),
            'qwen_skipped': len(frames_with_arrow) - len(processed_results)
        }
        
        for i in tqdm(range(len(all_frames)), desc="分配检测结果"):
            if i in processed_results:
                # 这是被选中的处理帧
                all_detections.append(processed_results[i])
            elif i in frame_arrow_counts:
                # 这是有箭头但未被选中的帧,使用最近的处理帧
                # 向前查找最近的处理帧
                nearest_processed = None
                for check_idx in range(i - 1, -1, -1):
                    if check_idx in processed_results:
                        nearest_processed = check_idx
                        break
                
                if nearest_processed is not None:
                    all_detections.append(processed_results[nearest_processed])
                else:
                    all_detections.append([])
            else:
                # 无箭头帧
                all_detections.append([])
        
        logger.info(f"\n✓ 检测结果分配完成!")
        logger.info(f"  总帧数: {stats['total_frames']}")
        logger.info(f"  有箭头帧: {stats['frames_with_arrow']}")
        logger.info(f"  无箭头帧: {stats['frames_without_arrow']}")
        logger.info(f"  Qwen3-VL实际调用: {stats['qwen_calls']} 次")
        logger.info(f"  节省调用: {stats['qwen_skipped']} 次")
        
        # 保存所有检测结果到JSON
        logger.info(f"保存检测结果到文件...")
        for i, detections in enumerate(tqdm(all_detections, desc="保存JSON")):
            results_file = results_json_dir / f"frame_{i:06d}_detections.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detections, f, indent=2, ensure_ascii=False)
        
        # 5. 标注所有帧
        logger.info(f"\n第四步: 标注所有帧...")
        output_frame_files = []
        
        with tqdm(total=len(all_frames), desc="标注帧") as pbar:
            for i, frame in enumerate(all_frames):
                detections = all_detections[i]
                
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
        logger.info(f"第五步: 合成输出视频...")
        output_video = output_dir / f"demo_smart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_processor.frames_list_to_video(
            output_frame_files,
            str(output_video),
            fps=fps
        )
        
        # 7. 清理临时文件
        logger.info("清理临时文件...")
        for frame_file in output_frame_files:
            if Path(frame_file).exists():
                os.remove(frame_file)
        
        if output_frames_dir.exists():
            import shutil
            shutil.rmtree(output_frames_dir)
        
        # 清理临时目录
        for d in [temp_output_depth, temp_output_sam3, temp_output_org, temp_output_correct]:
            if d.exists():
                shutil.rmtree(d)
        
        # 8. 生成总结报告
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_video": video_path,
            "output_video": str(output_video),
            "video_info": video_info,
            "strategy": f"两阶段处理 (扫描所有帧 + 抽样{sample_ratio*100:.0f}%处理)",
            "statistics": stats,
            "sample_ratio": sample_ratio,
            "total_detections": sum(len(d) for d in all_detections if d),
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
        description="智能视频处理和演示 (动态检测策略)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python process_video_smart.py --input my_video.mp4
  python process_video_smart.py --input my_video.mp4 --skip-threshold 10

说明:
  - 每帧都运行 SAM3 检测箭头(快速,~0.5秒)
  - 只对检测到箭头的帧调用 Qwen3-VL(慢速,~2-3秒)
  - 没有箭头的帧复用上一帧的检测结果
  - skip-threshold: 连续多少帧无箭头就清空检测(默认5)
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
        default='data/output/video_demo_smart',
        help='输出目录 (默认: data/output/video_demo_smart)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=0.05,
        help='抽样比例,从有箭头的帧中抽取多少进行处理 (默认: 0.05即5%%)'
    )
    
    parser.add_argument(
        '--only-largest',
        action='store_true',
        default=True,
        help='只处理每帧中最大的箭头框 (默认: True)'
    )
    
    parser.add_argument(
        '--use-existing-sam3',
        action='store_true',
        default=True,
        help='使用已有的SAM3扫描结果,不重新扫描 (默认: True)'
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
        pipeline = SmartVideoDemo(args.config)
        
        # 处理视频
        result = pipeline.process_video_smart(
            video_path=args.input,
            output_dir=args.output,
            sample_ratio=args.sample_ratio,
            use_existing_sam3=args.use_existing_sam3,
            only_largest_arrow=args.only_largest
        )
        
        print("\n" + "="*60)
        print("🎉 超快速视频演示生成成功!")
        print("="*60)
        print(f"输入视频: {args.input}")
        print(f"输出视频: {result['output_video']}")
        print(f"总帧数: {result['statistics']['total_frames']}")
        print(f"有箭头帧: {result['statistics']['frames_with_arrow']}")
        print(f"无箭头帧: {result['statistics']['frames_without_arrow']}")
        print(f"Qwen3-VL实际调用: {result['statistics']['qwen_calls']} 次")
        print(f"抽样比例: {result['sample_ratio']*100:.0f}%")
        print(f"只处理最大箭头框: {args.only_largest}")
        if result['statistics']['frames_with_arrow'] > 0:
            print(f"节省比例: {(1 - result['statistics']['qwen_calls']/result['statistics']['frames_with_arrow'])*100:.1f}%")
        print(f"检测结果: {result['total_detections']} 个目标")
        print(f"策略: {result['strategy']}")
        print(f"输出目录: {result['output_directory']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
