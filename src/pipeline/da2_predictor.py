"""
Depth Anything模型包装器
"""
import cv2
import torch
import numpy as np
import tifffile
import glob
import os 
import re 
from PIL import Image
from pathlib import Path
import sys
import logging

# 添加Depth Anything路径
sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "Depth-Anything-V2" / "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2

logger = logging.getLogger(__name__)

class DA2Predictor:
    """Depth Anything模型包装器"""
    
    def __init__(self, config: dict):
        """
        初始化深度预测模型
        
        Args:
            config: 深度模型配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device'))
        
        # 初始化模型
        self.model = self._load_model()
        logger.info(f"Depth predictor initialized on {self.device}")

    
    def _load_model(self):
        """加载Depth Anything模型"""
        
        scene = self.config.get('scene')
        dataset = self.config.get('dataset').get(scene)
        encoder = self.config.get('encoder')
        pretrained = self.config.get('checkpoint_paths').get(dataset).get(encoder)
        model_config = self.config.get('model_configs').get(encoder)
        max_depth = self.config.get('max_depth').get(scene)
        
        model = DepthAnythingV2(**{**model_config, 'max_depth': max_depth, 'device': str(self.device)})
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
        model = model.to(self.device).eval()  # 关键：将模型移动到设备

        return model
    
    def predict(self, image_np, image_path, output_depth_folder_str):
        """
        预测深度图
        
        Args:
            image_np: 图像 numpy.array
            
        Returns:
            depth_map: 深度图 (H, W)
        """
        try:        
            # 推理
            with torch.no_grad():
                depth_map = self.model.infer_image(image_np)
            
            # 从原始文件名提取序号（不依赖循环变量i）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_name = f"{base_name}_depth.tiff"
            output_depth_path = os.path.join(output_depth_folder_str,output_name)
            
            # 保存深度图
            tifffile.imwrite(output_depth_path, depth_map)
            print(f"DA2 Model: 深度图已保存: {output_depth_path}")
            
            return depth_map, output_depth_path
            
        except Exception as e:
            logger.error(f"Depth prediction failed: {e}")
            raise
    
    def release(self):
        """释放模型资源"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                # 如果模型在GPU上，先移回CPU再删除
                if torch.cuda.is_available():
                    self.model = self.model.cpu()
                
                del self.model
                self.model = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("DA2Predictor resources released")
            else:
                logger.warning("No model to release")
        except Exception as e:
            logger.error(f"Error releasing DA2Predictor: {e}")