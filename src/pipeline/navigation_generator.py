"""
English navigation sentence generator.

This module converts structured perception results into a single
user-facing English navigation sentence. It can use a Qwen3 text model
when available, and falls back to a deterministic English template when not.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


class NavigationGenerator:
    """Summarize perception results into one English navigation sentence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda")
        self.enabled = bool(self.config.get("enabled", True))
        self.output_language = self.config.get("output_language", "zh")
        self.use_llm = False
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = int(self.config.get("max_new_tokens", 128))
        # 相机FOV配置
        self.fov_x_deg = float(self.config.get("fov_x_deg", 90.0))

        checkpoint_path = self.config.get("checkpoint_path")
        if self.enabled and checkpoint_path and os.path.exists(checkpoint_path):
            self._try_load_model(checkpoint_path)
        else:
            logger.warning("NavigationGenerator will use English template fallback because Qwen3 checkpoint is unavailable.")

    def _try_load_model(self, checkpoint_path: str):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            logger.warning("transformers text-generation dependencies are unavailable, fallback to template mode.")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self.use_llm = True
            logger.info("NavigationGenerator initialized with Qwen3 text model: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to load Qwen3 text model, fallback to template mode: %s", e)
            self.model = None
            self.tokenizer = None
            self.use_llm = False

    def box_center(self, box) -> Optional[List[float]]:
        if not box:
            return None
        pts = np.array(box, dtype=np.float32)
        center = np.mean(pts, axis=0)
        return [float(center[0]), float(center[1])]

    def sample_depth_at_point(self, depth_map: np.ndarray, x: float, y: float) -> float:
        """Use a 3x3 median depth to reduce single-pixel noise."""
        h, w = depth_map.shape[:2]
        px = int(np.clip(round(x), 0, w - 1))
        py = int(np.clip(round(y), 0, h - 1))

        x1 = max(0, px - 1)
        x2 = min(w, px + 2)
        y1 = max(0, py - 1)
        y2 = min(h, py + 2)
        patch = depth_map[y1:y2, x1:x2]
        valid = patch[patch > 0.1]
        if valid.size == 0:
            return float(depth_map[py, px])
        return float(np.median(valid))

    def pixel_to_view_angle(self, x: float, image_width: int, fov_x_deg: float = 90.0) -> float:
        """Convert horizontal pixel coordinate to camera-relative horizontal angle."""
        fx = image_width / (2.0 * np.tan(np.radians(fov_x_deg) / 2.0))
        cx = image_width / 2.0
        return float(np.degrees(np.arctan2(x - cx, fx)))

    def describe_relative_position(self, angle_deg: float) -> str:
        abs_angle = abs(angle_deg)
        
        if self.output_language == "zh":
            # 中文位置描述
            if abs_angle < 10:
                return "正前方"
            if angle_deg < 0:
                return f"左前方 {int(round(abs_angle))} 度"
            return f"右前方 {int(round(abs_angle))} 度"
        else:
            # 英文位置描述
            if abs_angle < 10:
                return "directly ahead"
            if angle_deg < 0:
                return f"{int(round(abs_angle))} degrees to your front-left"
            return f"{int(round(abs_angle))} degrees to your front-right"

    def describe_turn(self, theta_deg: float) -> str:
        theta = float(theta_deg)
        
        if self.output_language == "zh":
            # 中文转向描述
            if -15 <= theta <= 15:
                return "直行"
            if 15 < theta < 165:
                return f"向右转 {int(round(theta))} 度"
            if -165 < theta < -15:
                return f"向左转 {int(round(abs(theta)))} 度"
            return "向后掉头"
        else:
            # 英文转向描述
            if -15 <= theta <= 15:
                return "go straight"
            if 15 < theta < 165:
                return f"turn right {int(round(theta))} degrees"
            if -165 < theta < -15:
                return f"turn left {int(round(abs(theta)))} degrees"
            return "turn around"

    def is_mostly_english(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        return ascii_chars / max(len(text), 1) > 0.8

    def normalize_sign_text_for_template(self, text: str) -> str:
        """
        Keep English sign text in template mode.
        If the text is Chinese or mixed and we have no LLM, fall back to a generic phrase.
        """
        # 清理 HTML 实体
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)  # 移除 HTML 标签
        text = re.sub(r"\s+", " ", text.strip())  # 规范化空格
        if not text:
            return ""
        if self.is_mostly_english(text):
            return text
        return ""

    def build_object_summary(self, result: Dict[str, Any], depth_map: np.ndarray) -> Dict[str, Any]:
        image_width = int(result.get("image_width", depth_map.shape[1]))
        # 使用配置的 FOV
        fov_x_deg = self.fov_x_deg

        # 尝试多种方式获取边界框
        anchor_box = (
            result.get("sign_box") or 
            result.get("arrow_box_4pts") or 
            result.get("points_2d") or  # 兼容 compass 项目的数据格式
            result.get("arrow_box")
        )
        
        # 计算中心点
        if anchor_box and len(anchor_box) >= 2:
            if isinstance(anchor_box[0], list) and len(anchor_box[0]) == 2:
                # points_2d 格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                center = self.box_center(anchor_box)
            elif len(anchor_box) == 2 and isinstance(anchor_box[0], (int, float)):
                # 2pts 格式：[x1, y1], [x2, y2]
                center = [(anchor_box[0][0] + anchor_box[1][0]) / 2, 
                         (anchor_box[0][1] + anchor_box[1][1]) / 2]
            else:
                center = [image_width / 2.0, depth_map.shape[0] / 2.0]
        else:
            center = [image_width / 2.0, depth_map.shape[0] / 2.0]
            
        center_depth = self.sample_depth_at_point(depth_map, center[0], center[1])
        rel_angle = self.pixel_to_view_angle(center[0], image_width, fov_x_deg)

        # 兼容 compass 项目：sign_text 可能不存在
        sign_text = str(result.get("sign_text", "")).strip()
        if not sign_text:
            sign_text = str(result.get("text", "")).strip()  # 尝试其他可能的字段
        
        # 清理 HTML 实体（如 &nbsp;）
        import re
        sign_text = re.sub(r'&nbsp;', ' ', sign_text)  # HTML 空格转普通空格
        sign_text = re.sub(r'<[^>]+>', '', sign_text)  # 移除其他 HTML 标签
        sign_text = sign_text.strip()
        
        # 获取 theta_deg
        theta_deg = float(result.get("theta_deg", 0.0))
        
        # alpha_deg 可能存在于 compass 数据中
        if "alpha_deg" in result:
            alpha_deg = float(result.get("alpha_deg", 0.0))
            if alpha_deg != 0:
                theta_deg = alpha_deg

        return {
            "sign_text": sign_text,
            "sign_text_for_template": self.normalize_sign_text_for_template(sign_text),
            "theta_deg": theta_deg,
            "distance_m": round(center_depth, 2),
            "relative_angle_deg": round(rel_angle, 2),
            "relative_position_text": self.describe_relative_position(rel_angle),
            "turn_text": self.describe_turn(theta_deg),
            "on_ground_ceiling": bool(result.get("on_ground_ceiling", False)),
            "arrow_box": anchor_box,
            "sign_box": result.get("sign_box") or result.get("points_2d"),
        }

    def build_prompt(self, image_path: str, summaries: List[Dict[str, Any]]) -> str:
        summaries_json = json.dumps(summaries, ensure_ascii=False, indent=2)
        return (
            "You are an English navigation assistant.\n"
            "Given the structured scene understanding results below, produce exactly one natural English navigation sentence.\n"
            "Requirements:\n"
            "1. Output one sentence only, in English.\n"
            "2. Prefer the most useful and nearest sign.\n"
            "3. Include relative position, distance, and the directional instruction when possible.\n"
            "4. If sign_text is Chinese or mixed-language, translate it into natural English before using it.\n"
            "5. If sign_text is empty or unreadable, do not invent a destination name.\n\n"
            f"Image: {image_path}\n"
            f"Candidates:\n{summaries_json}\n"
        )

    def generate_with_llm(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        return " ".join(text.strip().split())

    def generate_with_template(self, summaries: List[Dict[str, Any]]) -> str:
        best = sorted(summaries, key=lambda item: item["distance_m"])[0]
        sign_text = best["sign_text_for_template"]
        distance = best['distance_m']
        position = best['relative_position_text']
        turn_text = best['turn_text']
            
        # 根据输出语言生成导航语句
        if self.output_language == "zh":
            # 中文导航
            if sign_text:
                # 有路牌文字的情况
                return f"在你{position} {distance:.1f} 米处有一个路牌，路牌文字为\"{sign_text}\"，指示{turn_text}。"
            else:
                # 没有路牌文字的情况
                return f"在你{position} {distance:.1f} 米处有一个路牌，指示{turn_text}。"
        else:
            # 英文导航
            if sign_text:
                sign_clause = f'the sign reads "{sign_text}"'
            else:
                sign_clause = f'the sign indicates that you should {turn_text}'
    
            return (
                f"There is a signboard about {distance:.1f} meters away "
                f"{position}, and {sign_clause}."
            )

    def save_result(self, image_path: str, output_navigation_folder_str: str, payload: Dict[str, Any]):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name = f"{base_name}_navigation.json"
        output_path = os.path.join(output_navigation_folder_str, output_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Navigation output saved: {output_path}")

    def process(
        self,
        image_path: str,
        depth_map: np.ndarray,
        correct_data: Dict[str, Any],
        output_navigation_folder_str: str,
    ) -> Dict[str, Any]:
        correct_results = correct_data.get("correct_results", [])
        if not correct_results:
            payload = {
                "image_path": image_path,
                "navigation_sentence": "No usable sign information was detected for navigation.",
                "navigation_candidates": [],
                "generator_mode": "template_fallback",
            }
            self.save_result(image_path, output_navigation_folder_str, payload)
            return payload

        summaries = [self.build_object_summary(result, depth_map) for result in correct_results if "error" not in result]
        if not summaries:
            payload = {
                "image_path": image_path,
                "navigation_sentence": "The current image does not contain enough reliable sign information for navigation.",
                "navigation_candidates": [],
                "generator_mode": "template_fallback",
            }
            self.save_result(image_path, output_navigation_folder_str, payload)
            return payload

        if self.use_llm:
            prompt = self.build_prompt(image_path, summaries)
            try:
                navigation_sentence = self.generate_with_llm(prompt)
            except Exception as e:
                logger.warning("Qwen3 navigation generation failed, fallback to English template mode: %s", e)
                navigation_sentence = self.generate_with_template(summaries)
        else:
            navigation_sentence = self.generate_with_template(summaries)

        payload = {
            "image_path": image_path,
            "navigation_sentence": navigation_sentence,
            "navigation_candidates": summaries,
            "generator_mode": "qwen3" if self.use_llm else "template_fallback",
        }
        self.save_result(image_path, output_navigation_folder_str, payload)
        return payload

    def release(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
