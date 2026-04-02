"""
可视化工具：在图像上绘制识别结果（框、方向箭头、文字等）
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DrawingUtils:
    """绘制工具类"""
    
    # 颜色定义 (BGR格式，OpenCV使用)
    COLORS = {
        'box': (0, 255, 0),           # 绿色 - 检测框
        'arrow': (255, 0, 0),         # 蓝色 - 方向箭头
        'text': (0, 255, 255),        # 黄色 - 文字
        'text_bg': (0, 0, 0),         # 黑色 - 文字背景
        'direction_up': (0, 255, 0),  # 绿色 - 上
        'direction_down': (0, 0, 255),# 红色 - 下
        'direction_left': (255, 0, 0),# 蓝色 - 左
        'direction_right': (255, 255, 0), # 青色 - 右
    }
    
    @staticmethod
    def draw_detection_box(image: np.ndarray, 
                          box: List[List[int]],
                          label: str = "",
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        绘制检测框
        
        Args:
            image: 输入图像
            box: 边框４个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 或 [[x1,y1],[x2,y2]]
            label: 标签文字
            color: 颜色 (BGR)
            thickness: 线条粗细
            
        Returns:
            标注后的图像
        """
        image_copy = image.copy()
        
        if len(box) == 4:
            # 四个点的情况
            points = np.array(box, dtype=np.int32)
            cv2.polylines(image_copy, [points], True, color, thickness)
        elif len(box) == 2:
            # 两个点的情况（左上、右下）
            pt1 = tuple(box[0])
            pt2 = tuple(box[1])
            cv2.rectangle(image_copy, pt1, pt2, color, thickness)
        
        # 绘制标签
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness_text = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
            
            # 文字背景位置
            if len(box) >= 2:
                x, y = int(box[0][0]), int(box[0][1])
            else:
                x, y = 10, 30
            
            # 绘制背景
            cv2.rectangle(image_copy, 
                         (x - 5, y - text_size[1] - 5),
                         (x + text_size[0] + 5, y + 5),
                         DrawingUtils.COLORS['text_bg'], -1)
            
            # 绘制文字
            cv2.putText(image_copy, label, (x, y),
                       font, font_scale, DrawingUtils.COLORS['text'], thickness_text)
        
        return image_copy
    
    @staticmethod
    def draw_direction_arrow(image: np.ndarray,
                            center: Tuple[int, int],
                            angle_deg: float,
                            length: int = 50,
                            color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        绘制方向箭头
        
        Args:
            image: 输入图像
            center: 箭头中心点 (x, y)
            angle_deg: 方向角度（度）
                       0=上, 90=右, 180=下, -90=左
            length: 箭头长度
            color: 颜色 (BGR)，None则根据方向自动选择
            
        Returns:
            标注后的图像
        """
        image_copy = image.copy()
        
        # 自动选择颜色
        if color is None:
            if -45 <= angle_deg <= 45:
                color = DrawingUtils.COLORS['direction_up']      # 上
            elif 45 < angle_deg <= 135:
                color = DrawingUtils.COLORS['direction_right']   # 右
            elif -135 <= angle_deg < -45:
                color = DrawingUtils.COLORS['direction_left']    # 左
            else:
                color = DrawingUtils.COLORS['direction_down']    # 下
        
        # 计算箭头端点
        angle_rad = np.radians(angle_deg)
        end_x = center[0] + length * np.cos(angle_rad)
        end_y = center[1] + length * np.sin(angle_rad)
        end_point = (int(end_x), int(end_y))
        
        # 绘制箭头线
        cv2.arrowedLine(image_copy, center, end_point, color, 2, tipLength=0.3)
        
        # 绘制中心点
        cv2.circle(image_copy, center, 3, color, -1)
        
        return image_copy
    
    @staticmethod
    def draw_text_info(image: np.ndarray,
                      text: str,
                      position: Tuple[int, int] = (10, 30),
                      font_scale: float = 0.7,
                      color: Tuple[int, int, int] = (0, 255, 255),
                      thickness: int = 2,
                      bg_color: Tuple[int, int, int] = (0, 0, 0),
                      with_bg: bool = True) -> np.ndarray:
        """
        绘制文字信息（带背景）
        
        Args:
            image: 输入图像
            text: 文字内容
            position: 文字位置 (x, y)
            font_scale: 字体大小
            color: 文字颜色 (BGR)
            thickness: 文字粗细
            bg_color: 背景颜色
            with_bg: 是否绘制背景
            
        Returns:
            标注后的图像
        """
        image_copy = image.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        x, y = position
        
        if with_bg:
            # 绘制背景矩形
            cv2.rectangle(image_copy,
                         (x - 5, y - text_size[1] - 5),
                         (x + text_size[0] + 5, y + 5),
                         bg_color, -1)
        
        # 绘制文字
        cv2.putText(image_copy, text, position, font, font_scale, color, thickness)
        
        return image_copy
    
    @staticmethod
    def draw_annotation(image: np.ndarray,
                       detection_data: Dict[str, Any]) -> np.ndarray:
        """
        在图像上绘制完整的标注信息
        
        Args:
            image: 输入图像
            detection_data: 检测数据字典，包含:
                {
                    "arrow_box": [[x1,y1],[x2,y2]],    # 方向箭头的检测框
                    "alpha": angle_deg,                 # 方向角度
                    "sign_text": "识别文字"              # 识别的文字
                }
            
        Returns:
            标注后的图像
        """
        image_copy = image.copy()
        
        # 绘制检测框
        if "arrow_box" in detection_data:
            box = detection_data["arrow_box"]
            label = f"指示牌 (ID:{detection_data.get('id', '?')})"
            image_copy = DrawingUtils.draw_detection_box(
                image_copy, box, label, 
                color=DrawingUtils.COLORS['box'], thickness=2
            )
        
        # 绘制方向箭头
        if "alpha" in detection_data and "arrow_box" in detection_data:
            box = detection_data["arrow_box"]
            # 计算框的中心
            if len(box) == 2:
                center_x = (box[0][0] + box[1][0]) // 2
                center_y = (box[0][1] + box[1][1]) // 2
            else:
                center_x = np.mean([p[0] for p in box])
                center_y = np.mean([p[1] for p in box])
            
            alpha = detection_data["alpha"]
            # 角度可能是 0, 90, 180, -90，需要转换
            # API中: 0=上, 90=右, 180=下, -90=左
            # 如果alpha是这样的值
            image_copy = DrawingUtils.draw_direction_arrow(
                image_copy, 
                (int(center_x), int(center_y)),
                angle_deg=alpha,
                length=40,
                color=None  # 自动选择颜色
            )
        
        # 绘制识别文字
        if "sign_text" in detection_data and detection_data["sign_text"]:
            sign_text = detection_data["sign_text"]
            # 移除HTML标签
            import re
            sign_text = re.sub(r'<[^>]+>', '', sign_text)
            sign_text = sign_text.replace('&nbsp;', ' ')
            
            if sign_text.strip() and sign_text != "无文字":
                image_copy = DrawingUtils.draw_text_info(
                    image_copy,
                    f"文字: {sign_text}",
                    position=(10, 60),
                    font_scale=0.7,
                    color=DrawingUtils.COLORS['text'],
                    thickness=2,
                    with_bg=True
                )
        
        return image_copy
    
    @staticmethod
    def draw_multiple_annotations(image: np.ndarray,
                                 detections_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        在图像上绘制多个检测结果
        
        Args:
            image: 输入图像
            detections_list: 检测数据列表
            
        Returns:
            标注后的图像
        """
        image_copy = image.copy()
        
        for idx, detection in enumerate(detections_list):
            detection["id"] = idx + 1
            image_copy = DrawingUtils.draw_annotation(image_copy, detection)
        
        return image_copy
