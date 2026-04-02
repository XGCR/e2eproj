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
        # 根据配置选择加载方式
        if self.device == 'cpu':
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                device_map={'': 'cpu'},
                torch_dtype=torch.float32,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                dtype='auto',
                device_map='auto',
            )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     "model-8b-inst",
        #     dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

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
        Convert bounding box points to an English prompt for text and arrow direction recognition.
        Args:
            box_4pts: List of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        Returns:
            prompt string, 2-point box [[x1,y1],[x2,y2]]
        """
        top_left, bottom_right = box_4pts[0], box_4pts[2]
        tl = (int((top_left[0])/1024*1000), int((top_left[1])/1024*1000))
        br = (int((bottom_right[0])/1024*1000), int((bottom_right[1])/1024*1000))

        # English prompt for both arrow direction and sign text
        prompt = f"""Please analyze the content inside the bounding box {[{tl},{br}]} in the image:
1. What is the arrow direction of this sign/indicator? (Answer: up/down/left/right)
2. What is the text content on the sign? (Please recognize all text completely. If there is no text, answer 'no text'.)
Please reply in the following JSON format:
{{"direction": "direction", "text": "text content"}}"""

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
        if hasattr(self, 'model'):
            try:
                del self.model
            except Exception as e:
                logger.warning(f"Failed to delete Qwen3VL model attribute: {e}")
        if hasattr(self, 'processor'):
            try:
                del self.processor
            except Exception as e:
                logger.warning(f"Failed to delete Qwen3VL processor attribute: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()