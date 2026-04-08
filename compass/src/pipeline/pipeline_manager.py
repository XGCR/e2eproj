"""
Pipeline管理器：整合所有模块
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

from .da2_predictor import DA2Predictor
from .sam3_detector import SAM3Detector
from .qwen3vl_interpreter import Qwen3VLInterpreter
from .orientation_correct import Correcter

logger = logging.getLogger(__name__)

class PipelineManager:
    """端到端Pipeline管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化Pipeline管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.components = {}
        self._init_components()
        
        logger.info("PipelineManager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        import yaml
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config
    
    def _init_components(self):
        """初始化所有组件"""
        try:
            # 初始化深度预测器
            da2_config = self.config.get('da2', {})
            da2_config['device'] = self.config['system']['device']
            self.components['da2'] = DA2Predictor(da2_config)
            
            # 初始化SAM3检测器
            sam3_config = self.config.get('sam3', {})
            sam3_config['device'] = self.config['system']['device']
            self.components['sam3'] = SAM3Detector(sam3_config)
            
            # 初始化Qwen3VL解释器
            qwen_config = self.config.get('qwen3vl', {})
            qwen_config['device'] = self.config['system']['device']
            self.components['qwen3vl'] = Qwen3VLInterpreter(qwen_config)
            
            # 初始化角度修正器
            self.components['correct'] = Correcter()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_single_image(
                    self, 
                    image_path, 
                    output_depth_folder_str, 
                    output_sam3_json_folder_str,
                    temp_organized_json_folder_str,
                    output_correct_json_folder_str):
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            result: 处理结果
        """
        logger.info(f"Processing image: {image_path}")
        
        # 1. 加载图像
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # 2. da2
        logger.info("Running Depth Anything V2...")
        depth_map, depth_image_path = self.components['da2'].predict(
                                            image_np, 
                                            image_path, 
                                            output_depth_folder_str
                                            )

        # 3. sam3
        logger.info("Running Segment Anything 3...")
        point_series = self.components['sam3'].segment(
                                            image, 
                                            image_path, 
                                            output_sam3_json_folder_str
                                            )
        
        # 3.5 立即释放 DA2 和 SAM3 显存，为 Qwen3-VL 和后续处理腾出空间
        logger.info("Releasing DA2 and SAM3 GPU memory...")
        self.components['da2'].release()
        self.components['sam3'].release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 4. qwen3-vl
        logger.info("Running Qwen3-VL...")
        items = self.components['qwen3vl'].interpret_multiple_objects(
                                            image_path, 
                                            point_series,
                                            depth_image_path,
                                            temp_organized_json_folder_str
                                            )
            
        # 7. 角度修正
        logger.info("Running Orientation Correcter...")
        result = self.components['correct'].process(
                                            image_path, 
                                            image_np, 
                                            depth_map, 
                                            items,
                                            output_correct_json_folder_str
                                            )
            
        # 8. 生成行走指导
        logger.info("Generating walking instruction...")
        walking_instruction = self.components['qwen3vl'].generate_walking_instruction(
                                            image_path, 
                                            items
                                            )
        logger.info(f"Walking instruction: {walking_instruction}")
    
            
        return point_series
            
        # except Exception as e:
        #     logger.error(f"Error processing image {image_path}: {e}")
        #     return self._create_error_result(image_path, str(e))
    
    def _extract_depth_stats(self, depth_map: np.ndarray, 
                            mask: np.ndarray) -> Dict[str, float]:
        """从深度图中提取目标区域的深度统计信息"""
        target_depth = depth_map[mask > 0]
        
        if len(target_depth) == 0:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        return {
            'mean': float(np.mean(target_depth)),
            'min': float(np.min(target_depth)),
            'max': float(np.max(target_depth)),
            'std': float(np.std(target_depth)),
            'median': float(np.median(target_depth)),
            'percentile_25': float(np.percentile(target_depth, 25)),
            'percentile_75': float(np.percentile(target_depth, 75))
        }
    
    def _create_empty_result(self, image_name: str) -> Dict:
        """创建空结果"""
        return {
            'image_info': {
                'name': image_name,
                'status': 'no_targets_detected'
            },
            'targets': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_error_result(self, image_name: str, error_msg: str) -> Dict:
        """创建错误结果"""
        return {
            'image_info': {
                'name': image_name,
                'status': 'error',
                'error': error_msg
            },
            'targets': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_intermediate_result(self, result: Dict, image_path: Path):
        """保存中间结果"""
        output_dir = Path(self.config['paths']['output_dir'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{image_path.stem}_{timestamp}.json"
        
        # 简化结果以节省空间
        simplified = result.copy()
        if 'visualization' in simplified:
            del simplified['visualization']
        
        for target in simplified.get('targets', []):
            if 'target_image' in target:
                del target['target_image']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    def save_results(self, results: List[Dict], output_dir: Path):
        """保存所有结果"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存汇总结果
        summary = {
            'pipeline_version': '1.0',
            'processing_timestamp': timestamp,
            'total_images': len(results),
            'successful_images': len([r for r in results if r['image_info'].get('status') != 'error']),
            'results': results
        }
        
        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 保存每个图像的结果
        for result in results:
            image_name = result['image_info']['name']
            result_path = output_dir / f"{Path(image_name).stem}_result.json"
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 生成可视化报告
        self._generate_report(results, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def release(self):
        """释放所有资源"""
        logger.info("Releasing pipeline resources...")
        
        for name, component in self.components.items():
            if hasattr(component, 'release'):
                component.release()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Pipeline resources released")
    
    def __del__(self):
        """析构函数"""
        self.release()