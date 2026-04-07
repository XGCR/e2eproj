"""
Qwen3VL语义解释器包装器
"""
import torch
from PIL import Image
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import logging
from typing import List, Dict, Any
import os 
import glob 
import re 
import json 

logger = logging.getLogger(__name__)

class Qwen3VLInterpreter:
    """Qwen3VL语义解释器"""
    
    def __init__(self, config: dict):
        """
        初始化Qwen3VL解释器
        
        Args:
            config: Qwen3VL配置字典
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # 初始化模型和处理器
        self.model, self.processor = self._load_model() 
        logger.info(f"Qwen3VL interpreter initialized with model: {self.config.get('checkpoint_path')}")
    
    def _load_model(self):
        """加载Qwen3VL模型"""

        self.checkpoint_path = self.config.get('checkpoint_path')
        
        # 使用 GPU + CPU offload 策略 (避免显存不足)
        logger.info(f"加载 Qwen3-VL 模型...")
        logger.info("使用 bfloat16 精度加速推理...")
        logger.info("使用 GPU + CPU offload 策略...")
        
        # 直接使用 CPU offload 模式,避免先尝试全 GPU 导致显存碎片
        from accelerate import infer_auto_device_map
        
        # 先用 meta 设备计算模型大小
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "8GiB", "cpu": "40GiB"},  # GPU 8GB, 其余放 CPU
            offload_folder="offload",  # 使用文件夹而不是磁盘
            offload_state_dict=True
        )

        processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        
        # 验证模型设备分布
        model_devices = set(p.device for p in model.parameters())
        logger.info(f"✅ Qwen3-VL 模型已加载")
        logger.info(f"   模型分布在: {model_devices}")
        logger.info(f"   模型精度: {model.dtype}")
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"   GPU 显存使用: {gpu_mem:.2f} GB")
            
        # 判断是否完全在 GPU 上
        if len(model_devices) == 1 and 'cuda' in str(list(model_devices)[0]):
            logger.info("   🚀 模型完全在 GPU 上 - 最快!")
        else:
            logger.info("   ⚡ 模型使用 GPU + CPU offload - 比纯 CPU 快 10-20 倍")
            
        return model, processor

    def parse_text(self, text):
        lines = text.split('\n')
        lines = [line for line in lines if line != '']
        count = 0
        for i, line in enumerate(lines):
            if '```' in line:
                count += 1
                items = line.split('`')
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = '<br></code></pre>'
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace('`', r'\`')
                        line = line.replace('<', '&lt;')
                        line = line.replace('>', '&gt;')
                        line = line.replace(' ', '&nbsp;')
                        line = line.replace('*', '&ast;')
                        line = line.replace('_', '&lowbar;')
                        line = line.replace('-', '&#45;')
                        line = line.replace('.', '&#46;')
                        line = line.replace('!', '&#33;')
                        line = line.replace('(', '&#40;')
                        line = line.replace(')', '&#41;')
                        line = line.replace('$', '&#36;')
                    lines[i] = '<br>' + line
        text = ''.join(lines)
        return text
        
    def build_prompt(self, box_4pts):
        """
        将边框点转换为指定格式的文本，同时询问箭头方向和标志牌文字
        
        Args:
            box_4pts: 边框列表，每个边框是4个点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
        Returns:
            格式化的文本字符串, 2点边框 [[x1,y1],[x2,y2]]
        """
        top_left, bottom_right = box_4pts[0], box_4pts[2]
        tl = (int((top_left[0])/1024*1000), int((top_left[1])/1024*1000))
        br = (int((bottom_right[0])/1024*1000), int((bottom_right[1])/1024*1000))

        # 同时询问箭头方向和标志牌上的文字
        prompt = f"""请分析图像中边框 {[{tl},{br}]} 内的内容：
1. 这个标志牌/指示牌的箭头方向是什么？（回答：上/下/左/右）
2. 标志牌上的文字内容是什么？（请完整识别所有文字，如果没有文字则回答“无文字”）
请按以下JSON格式回答：
{{"direction": "方向", "text": "文字内容"}}"""

        return prompt, [top_left, bottom_right]
    
    def interpret_multiple_objects(self, 
                            image_path, 
                            point_series, 
                            depth_image_path,
                            temp_organized_json_folder_str) :
        """
        解释单个目标
        
        Args:
            image_path: 输入图像路径
            point_series: {"boxes_arrow":[],"masks_sign":[]}
            
        Returns:
            interpretation: 解释结果
        """
        # 构建prompt  
        items = []      
        for box_4pts in point_series["boxes_arrow"]:
            message_text, box_2pts = self.build_prompt(box_4pts)
            messages = [
                {
                    "role": "user",
                    "content": [  
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {   "type": "text", 
                            "text": message_text
                        },
                    ],
                }
            ]
            # Preparation for inference
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            parse_result = self.parse_text(output_text[0])
            
            # 解析方向和文字
            direction = None
            sign_text = ""
            
            # 解析方向
            if "左" in parse_result: 
                direction = -90
            elif "右" in parse_result: 
                direction = 90
            elif "上" in parse_result: 
                direction = 0
            elif "下" in parse_result: 
                direction = 180
            
            # 尝试从输出中提取文字信息
            # 查找 JSON 格式的 text 字段
            import re
            text_match = re.search(r'"text"[:\s]*"?([^"}]+)"?', parse_result)
            if text_match:
                sign_text = text_match.group(1).strip()
                # 过滤掉 HTML 标签
                sign_text = re.sub(r'<[^>]+>', '', sign_text)
            
            # 如果没有提取到文字，检查是否包含"无文字"
            if not sign_text or sign_text == "无文字":
                sign_text = ""
            
            item = {
                    "alpha": direction if direction else 0,  # 方向角度
                    "theta": "",  # 新增theta，留空
                    "sign_text": sign_text,  # 标志牌上的文字
                    "image_path": image_path,  # 只保留文件名
                    "depth_path": depth_image_path,  # 只保留文件名
                    "arrow_box": box_2pts  # 原始的4个角点
                }
            items.append(item)
            
            logger.info(f"识别结果 - 方向: {direction}, 文字: {sign_text or '(无文字)'}")

        # 保存结果  
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        organized_point = f"{base_name}_organized.json"
        organized_point_file_path = os.path.join(temp_organized_json_folder_str,organized_point)
        with open(organized_point_file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"Qwen3-VL Model: 解译已完成, 组织结果已保存: {organized_point_file_path}")
        return items
            
    def release(self):
        """释放模型资源"""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()