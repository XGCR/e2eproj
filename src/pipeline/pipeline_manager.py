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
from .navigation_generator import NavigationGenerator
from .tts_player import TTSPlayer

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
            # 获取相机配置
            camera_config = self.config.get('camera', {})
            fov_x_deg = float(camera_config.get('fov_x_deg', 90.0))
            
            # 初始化深度预测器
            da2_config = self.config.get('da2', {})
            da2_config['device'] = self.config['system']['device']
            self.components['da2'] = DA2Predictor(da2_config)
            
            # 初始化SAM3检测器
            sam3_config = self.config.get('sam3', {})
            sam3_config['device'] = self.config['system']['device']
            self.components['sam3'] = SAM3Detector(sam3_config)
            
            # 初始化Qwen3VL解释器（根据config可选）
            qwen_config = self.config.get('qwen3vl', {})
            if qwen_config.get('enabled', True):
                qwen_config['device'] = qwen_config.get('device', self.config['system']['device'])
                qwen_config['fov_x_deg'] = fov_x_deg  # 传递FOV参数
                self.components['qwen3vl'] = Qwen3VLInterpreter(qwen_config)
            else:
                logger.info("Qwen3VL is disabled in config, skipping model load")
            
            # 初始化角度修正器（传递相机配置）
            self.components['correct'] = Correcter(camera_config)
            
            # 初始化导航生成器
            nav_config = self.config.get('navigation', {})
            nav_config['device'] = self.config['system']['device']
            nav_config['fov_x_deg'] = fov_x_deg  # 传递FOV参数
            self.components['navigation'] = NavigationGenerator(nav_config)
            
            # 初始化TTS播放器（禁用语音输出，仅显示文字）
            tts_config = self.config.get('tts', {})
            tts_config['enabled'] = False  # 禁用语音，仅显示文字
            self.components['tts'] = TTSPlayer(tts_config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _prepare_qwen3vl(self):
        """根据配置准备Qwen3VL解释器（按需加载/重载）。"""
        qwen_config = self.config.get('qwen3vl', {})
        if not qwen_config.get('enabled', True):
            logger.info("Qwen3VL disabled in config")
            return None
        
        # 如果已存在且keep_loaded为True，则直接复用
        if qwen_config.get('keep_loaded', False) and self.components.get('qwen3vl'):
            return self.components['qwen3vl']
        
        # 如果已经存在且keep_loaded为False，先释放后重新加载
        if 'qwen3vl' in self.components and not qwen_config.get('keep_loaded', False):
            try:
                self.components['qwen3vl'].release()
            except Exception as e:
                logger.warning(f"Releasing old Qwen3VL instance failed: {e}")
            finally:
                self.components.pop('qwen3vl', None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 设置设备参数后加载模型
        qwen_config['device'] = qwen_config.get('device', self.config['system']['device'])
        qwen_config['fov_x_deg'] = float(self.config.get('camera', {}).get('fov_x_deg', 90.0))
        self.components['qwen3vl'] = Qwen3VLInterpreter(qwen_config)
        return self.components['qwen3vl']

    def process_single_image(
                    self, 
                    image_path, 
                    output_depth_folder_str, 
                    output_sam3_json_folder_str,
                    temp_organized_json_folder_str,
                    output_correct_json_folder_str,
                    output_navigation_folder_str=None):
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            output_navigation_folder_str: 导航输出文件夹（可选）
            
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
            
        # 4. qwen3-vl
        qwen3vl_comp = self._prepare_qwen3vl()
        if qwen3vl_comp is None:
            items = []
            logger.warning("Qwen3VL is disabled or failed to load; skipping text recognition")
        else:
            logger.info("Running Qwen3-VL...")
            items = qwen3vl_comp.interpret_multiple_objects(
                                            image_path, 
                                            point_series,
                                            depth_image_path,
                                            temp_organized_json_folder_str
                                            )

        # 7. 角度修正
        logger.info("Running Orientation Correcter...")
        correct_result = self.components['correct'].process(
                                            image_path, 
                                            image_np, 
                                            depth_map, 
                                            items,
                                            output_correct_json_folder_str
                                            )
        
        # 8. 导航生成
        if output_navigation_folder_str:
            nav_output_folder = Path(output_navigation_folder_str)
            nav_output_folder.mkdir(parents=True, exist_ok=True)
            
            # 加载修正后的结果
            base_name = Path(image_path).stem
            correct_json_path = Path(output_correct_json_folder_str) / f"{base_name}_correct.json"
            
            if correct_json_path.exists():
                with open(correct_json_path, 'r', encoding='utf-8') as f:
                    correct_data = json.load(f)
                
                logger.info("Running Navigation Generator...")
                nav_result = self.components['navigation'].process(
                    image_path,
                    depth_map,
                    correct_data,
                    output_navigation_folder_str
                )
                
                # 显示导航文字到终端
                navigation_sentence = nav_result.get('navigation_sentence', '')
                print("\n" + "="*60)
                print("  导航播报文字 (Navigation Voice Message)")
                print("="*60)
                print(f"  {navigation_sentence}")
                print("="*60 + "\n")
                
                # TTS文字播报（已禁用语音，仅返回状态）
                tts_result = self.components['tts'].speak(navigation_sentence)
                if tts_result.get('tts_status') == 'disabled':
                    logger.info("TTS is disabled, navigation text displayed in terminal only.")
            else:
                logger.warning(f"Correct result file not found: {correct_json_path}")

        # 重新释放Qwen3VL（如果配置要求不保持加载）
        qwen_config = self.config.get('qwen3vl', {})
        if not qwen_config.get('keep_loaded', False) and 'qwen3vl' in self.components:
            try:
                self.components['qwen3vl'].release()
                logger.info("Qwen3VL has been released after processing this image")
            except Exception as e:
                logger.warning(f"Failed to release Qwen3VL after image: {e}")
            finally:
                self.components.pop('qwen3vl', None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return correct_result
            
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
                try:
                    component.release()
                except Exception as e:
                    logger.warning(f"Failed to release component {name}: {e}")
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Pipeline resources released")
    
    def __del__(self):
        """析构函数"""
        self.release()