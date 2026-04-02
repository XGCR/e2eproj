"""
视频处理模块：从视频提取帧 -> 逐帧处理 -> 输出标注视频
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, video_path: str, output_dir: str = "data/output/video"):
        """
        初始化视频处理器
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证视频文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 获取视频信息
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()
        
        logger.info(f"视频信息 - 分辨率: {self.width}x{self.height}, FPS: {self.fps}, 总帧数: {self.total_frames}")
    
    def extract_frames(self, output_frames_dir: str, sample_rate: int = 1) -> List[str]:
        """
        提取视频帧
        
        Args:
            output_frames_dir: 输出帧目录
            sample_rate: 采样率（每N帧提取一帧），1表示全部提取
            
        Returns:
            提取的帧文件路径列表
        """
        frames_dir = Path(output_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        frame_files = []
        frame_count = 0
        extracted_count = 0
        
        logger.info(f"开始提取视频帧 (采样率: {sample_rate})...")
        
        with tqdm(total=self.total_frames, desc="提取帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按采样率提取
                if frame_count % sample_rate == 0:
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frame_files.append(str(frame_path))
                    extracted_count += 1
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"共提取 {extracted_count} 帧")
        return frame_files
    
    def frames_to_video(self, 
                       frames_with_annotations: List[tuple],
                       output_video_path: str,
                       fps: Optional[int] = None) -> str:
        """
        将标注后的帧合成为视频
        
        Args:
            frames_with_annotations: [(frame_path, annotations_dict), ...] 列表
            output_video_path: 输出视频路径
            fps: 输出视频帧率（默认使用原视频fps）
            
        Returns:
            输出视频路径
        """
        if not frames_with_annotations:
            raise ValueError("没有可用的帧")
        
        fps = fps or self.fps
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(frames_with_annotations[0][0])
        height, width = first_frame.shape[:2]
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        logger.info(f"开始合成视频: {output_video_path}...")
        
        with tqdm(total=len(frames_with_annotations), desc="合成视频") as pbar:
            for frame_path, annotations in frames_with_annotations:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"无法读取帧: {frame_path}")
                    continue
                
                # 这里的annotations在外部绘制后已经标注在frame上
                out.write(frame)
                pbar.update(1)
        
        out.release()
        logger.info(f"视频合成完成: {output_video_path}")
        return output_video_path
    
    def frames_list_to_video(self, 
                            frame_files: List[str],
                            output_video_path: str,
                            fps: Optional[int] = None) -> str:
        """
        将帧列表直接合成为视频（帧已标注）
        
        Args:
            frame_files: 帧文件路径列表
            output_video_path: 输出视频路径
            fps: 输出视频帧率
            
        Returns:
            输出视频路径
        """
        if not frame_files:
            raise ValueError("没有可用的帧")
        
        fps = fps or self.fps
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            raise ValueError(f"无法读取第一帧: {frame_files[0]}")
        
        height, width = first_frame.shape[:2]
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        logger.info(f"开始合成视频: {output_video_path}...")
        
        with tqdm(total=len(frame_files), desc="合成视频") as pbar:
            for frame_path in frame_files:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"无法读取帧: {frame_path}")
                    continue
                
                out.write(frame)
                pbar.update(1)
        
        out.release()
        logger.info(f"视频合成完成: {output_video_path}")
        return output_video_path
    
    def get_video_info(self) -> Dict[str, Any]:
        """获取视频信息"""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": self.total_frames / self.fps
        }
