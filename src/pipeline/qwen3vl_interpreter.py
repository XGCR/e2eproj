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
        # default: Load the model on the available device(s)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
                                    self.checkpoint_path, dtype="auto", device_map="auto"
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

    def extract_json_block(self, text: str) -> Dict[str, Any]:
        """从模型输出中尽量提取 JSON 对象，失败时返回空字典。"""
        if not text:
            return {}

        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return {}

        json_candidate = match.group(0)
        try:
            return json.loads(json_candidate)
        except Exception:
            return {}
        
    def build_prompt(self, box_4pts, sign_box_4pts=None):
        """
        将箭头框和路牌框转换为指定格式的提示词。
        
        Args:
            box_4pts: 边框列表，每个边框是4个点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
        Returns:
            格式化的文本字符串
        """
        top_left, bottom_right = box_4pts[0], box_4pts[2]
        tl = (int((top_left[0])/1024*1000), int((top_left[1])/1024*1000))
        br = (int((bottom_right[0])/1024*1000), int((bottom_right[1])/1024*1000))

        if sign_box_4pts is not None:
            sign_top_left, sign_bottom_right = sign_box_4pts[0], sign_box_4pts[2]
            sign_tl = (int((sign_top_left[0])/1024*1000), int((sign_top_left[1])/1024*1000))
            sign_br = (int((sign_bottom_right[0])/1024*1000), int((sign_bottom_right[1])/1024*1000))
            prompt = (
                f"请阅读图像内容。箭头位于边框{[{tl}, {br}]}，"
                f"对应路牌位于边框{[{sign_tl}, {sign_br}]}。"
                "请严格输出 JSON，不要输出额外解释，格式为："
                "{\"arrow_direction\":\"上/下/左/右之一\","
                "\"sign_text\":\"路牌上的主要文字，没有就写空字符串\"}"
            )
        else:
            prompt = (
                f"图像中边框{[{tl}, {br}]}内的箭头方向是什么？"
                "请严格输出 JSON，不要输出额外解释，格式为："
                "{\"arrow_direction\":\"上/下/左/右之一\","
                "\"sign_text\":\"如果图中可读到相关路牌文字就填写，否则写空字符串\"}"
            )

        return prompt, [top_left, bottom_right]

    def contour_to_box(self, contour_points):
        """将轮廓点转为四点矩形框。"""
        if not contour_points:
            return None

        pts = np.array(contour_points, dtype=np.float32)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        return [
            [float(x_min), float(y_min)],
            [float(x_max), float(y_min)],
            [float(x_max), float(y_max)],
            [float(x_min), float(y_max)]
        ]

    def box_center(self, box_4pts):
        pts = np.array(box_4pts, dtype=np.float32)
        center = np.mean(pts, axis=0)
        return float(center[0]), float(center[1])

    def match_sign_box(self, arrow_box_4pts, sign_masks):
        """
        为箭头框匹配最近的路牌区域。

        当前使用简单的中心点最近邻策略，足够作为最小可用版本。
        """
        if not sign_masks:
            return None

        arrow_cx, arrow_cy = self.box_center(arrow_box_4pts)
        best_sign_box = None
        best_dist = None

        for contour_points in sign_masks:
            sign_box = self.contour_to_box(contour_points)
            if sign_box is None:
                continue

            sign_cx, sign_cy = self.box_center(sign_box)
            dist = (arrow_cx - sign_cx) ** 2 + (arrow_cy - sign_cy) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_sign_box = sign_box

        return best_sign_box

    def parse_arrow_direction(self, text: str, parsed_json: Dict[str, Any]) -> int:
        """把模型输出解析为项目内部使用的 alpha 角度。"""
        arrow_text = str(parsed_json.get("arrow_direction", "")).strip()
        merged_text = f"{arrow_text} {text}"

        if "左" in merged_text:
            return -90
        if "右" in merged_text:
            return 90
        if "上" in merged_text:
            return 0
        if "下" in merged_text:
            return 180

        raise ValueError(f"无法从模型输出解析箭头方向: {text}")

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
            sign_box_4pts = self.match_sign_box(box_4pts, point_series.get("masks_sign", []))
            message_text, box_2pts = self.build_prompt(box_4pts, sign_box_4pts)
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
            parsed_json = self.extract_json_block(output_text[0])
            alpha = self.parse_arrow_direction(parse_result, parsed_json)
            sign_text = str(parsed_json.get("sign_text", "")).strip()

            item = {
                    "alpha": alpha,  # 改为alpha，留空
                    "theta": "",  # 新增theta，留空
                    "image_path": image_path,  # 只保留文件名
                    "depth_path": depth_image_path,  # 只保留文件名
                    "arrow_box": box_2pts,  # 原始的2个对角点
                    "arrow_box_4pts": box_4pts,
                    "sign_box": sign_box_4pts,
                    "sign_text": sign_text
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
            
    def release(self):
        """释放模型资源"""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
