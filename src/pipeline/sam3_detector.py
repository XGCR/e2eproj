"""
SAM3目标检测器包装器
"""
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import logging
import cv2
import os
import json

# 添加SAM3路径
sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "sam3"))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logger = logging.getLogger(__name__)

class SAM3Detector:
    """SAM3目标检测器"""
    
    def __init__(self, config: dict):
        """
        初始化SAM3检测器
        
        Args:
            config: SAM3配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device'))
        
        # 初始化模型
        self.model = self._load_model()        
        logger.info(f"SAM3 detector initialized on {self.device}")
    
    def _load_model(self):
        """加载SAM3模型"""
        checkpoint_path = self.config.get('checkpoint_path')
        device = self.config.get('device', 'cpu')  # 确保有默认值

        # 使用 bfloat16 精度加载模型（修复 dtype 不匹配问题）
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path, 
            device=device
        )
        processor = Sam3Processor(model, device=device)
        
        # 将模型转换为 bfloat16 并移动到指定设备
        processor.model = processor.model.to(device).to(torch.bfloat16)
        
        return processor
    
    def segment(self, image, image_path, output_sam3_json_folder_str):
        """
        自动检测图像中的所有目标
        
        Args:
            image: 输入图像     Image
            
        Returns:
            masks: 掩码列表
            boxes: 边界框列表
        """
        try:
            # 确保模型在CPU上
            self.model.model = self.model.model.to(self.device)
            
            inference_state = self.model.set_image(image)
            prompt_sign, prompt_arrow = "sign", "arrow"

            # sign
            output_sign = self.model.set_text_prompt(state=inference_state, prompt=prompt_sign)
            masks_sign, boxes_sign, scores_sign = \
                output_sign["masks"], output_sign["boxes"], output_sign["scores"]

            # arrow
            output_arrow = self.model.set_text_prompt(state=inference_state, prompt=prompt_arrow)
            masks_arrow, boxes_arrow, scores_arrow = \
                output_arrow["masks"], output_arrow["boxes"], output_arrow["scores"]

            # 保存点序结果
            point_series = self.reroganize(masks_sign, boxes_arrow)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_point = f"{base_name}_point.json"
            output_point_file_path = os.path.join(output_sam3_json_folder_str,output_point)
            with open(output_point_file_path, 'w', encoding='utf-8') as f:
                json.dump(point_series, f, indent=2, ensure_ascii=False)
                
            return point_series
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                logger.warning(f"SAM3 device error: {e}. Returning empty results.")
                # 返回空结果
                point_series = {"boxes_arrow": [], "masks_sign": []}
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_point = f"{base_name}_point.json"
                output_point_file_path = os.path.join(output_sam3_json_folder_str,output_point)
                with open(output_point_file_path, 'w', encoding='utf-8') as f:
                    json.dump(point_series, f, indent=2, ensure_ascii=False)
                    
                return point_series
            else:
                raise
        print(f"SAM3 Model: 目标轮廓点序已保存: {output_point_file_path}")

        return point_series
    
    def reroganize(self, masks_sign, boxes_arrow):
        """重新组织 box 和 mask 的格式"""

        point_series = {"boxes_arrow":[],"masks_sign":[]}

        boxes_nps = boxes_arrow.cpu().numpy().tolist()
        for index, bn in enumerate(boxes_nps):
            
            # 确保坐标非负并取整
            x1 = max(0, int(round(bn[0])))  # 左
            y1 = max(0, int(round(bn[1])))  # 上
            x2 = max(0, int(round(bn[2])))  # 右
            y2 = max(0, int(round(bn[3])))  # 下
            
            # 确保 x2 >= x1, y2 >= y1
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            boxes_4p = [
                [x1, y1],  # 左上
                [x2, y1],  # 右上
                [x2, y2],  # 右下
                [x1, y2]   # 左下
            ]
            
            # 检查box是否有效（面积大于0）
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                print(f"警告: Box {index} 尺寸无效: {width}x{height}")
                continue
            
            point_series["boxes_arrow"].append(boxes_4p)

        masks_sign = masks_sign.cpu().numpy()
        for index, ms in enumerate(masks_sign):
            # 确保掩膜是二值图像（0和255）
            # 先将掩膜转换为uint8类型
            mask_uint8 = ms.astype(np.uint8)[0]
            h, w = mask_uint8.shape
            # print(masks_sign.shape,mask_uint8.shape)
            # print(np.unique(mask_uint8))
            
            # 提取轮廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 取最大的轮廓（如果多个轮廓，通常取最大的）
                contour = max(contours, key=cv2.contourArea)
                
                # 简化轮廓（减少点数）
                # epsilon参数控制简化程度，值越大越简化
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # 提取轮廓点并转换为整数
                contour_points = []
                for point in approx_contour:
                    x, y = point[0]
                    x = max(0, min(w-1, int(round(x))))
                    y = max(0, min(h-1, int(round(y))))
                    contour_points.append([int(round(x)), int(round(y))])
                
                point_series["masks_sign"].append(contour_points)

        return point_series

    def release(self):
        """释放模型资源"""
        self.predictor = None
        if hasattr(self, 'model'):
            try:
                del self.model
            except Exception as e:
                logger.warning(f"Failed to delete SAM3 model attribute: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()