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
        # 使用 float16 降低显存占用（从 ~6GB 降到 ~3GB）
        logger.info(f"Loading Qwen3VL model in float16 mode to save GPU memory...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
                                    self.checkpoint_path,
                                    torch_dtype=torch.float16,
                                    device_map="auto"
                                )

        processor = AutoProcessor.from_pretrained(self.checkpoint_path)
            
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
        将边框点转换为指定格式的文本
        
        Args:
            box_4pts: 边框列表，每个边框是4个点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
        Returns:
            格式化的文本字符串
        """
        top_left, bottom_right = box_4pts[0], box_4pts[2]
        tl = (int((top_left[0])/1024*1000), int((top_left[1])/1024*1000))
        br = (int((bottom_right[0])/1024*1000), int((bottom_right[1])/1024*1000))

        return f"图像中边框{[{tl},{br}]}内的箭头方向是什么? 请回答：上/下/左/右", [top_left,bottom_right]
    
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
            if "左" in parse_result: alpha=-90
            elif "右" in parse_result: alpha=90
            elif "上" in parse_result: alpha=0
            elif "下" in parse_result: alpha=180

            item = {
                    "alpha": alpha,  # 改为alpha，留空
                    "theta": "",  # 新增theta，留空
                    "image_path": image_path,  # 只保留文件名
                    "depth_path": depth_image_path,  # 只保留文件名
                    "arrow_box": box_2pts  # 原始的4个角点
                }
            items.append(item)

        # 保存结果  
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        organized_point = f"{base_name}_organized.json"
        organized_point_file_path = os.path.join(temp_organized_json_folder_str,organized_point)
        with open(organized_point_file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"Qwen3-VL Model: 解译已完成, 组织结果已保存: {organized_point_file_path}")
        return items
            
    def generate_walking_instruction(self, image_path: str, items: List[Dict]) -> str:
        """
        基于箭头方向生成行走指导句子
        
        Args:
            image_path: 图像路径
            items: 箭头方向信息列表
            
        Returns:
            walking_instruction: 行走指导句子
        """
        # 构建方向描述
        directions = []
        for item in items:
            alpha = item.get('alpha', 0)
            if alpha == 0:
                directions.append("上")
            elif alpha == 90:
                directions.append("右")
            elif alpha == 180:
                directions.append("下")
            elif alpha == -90:
                directions.append("左")
        
        direction_text = ", ".join(directions) if directions else "无方向"
        
        # 构建prompt
        message_text = f"图像中有箭头指向{direction_text}方向。请生成一句简短的行走指导句子，指导用户如何行走。"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": message_text
                    },
                ],
            }
        ]
        
        # 准备推理
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # 生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        walking_instruction = self.parse_text(output_text[0])
        return walking_instruction
            
    def release(self):
        """释放模型资源"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()