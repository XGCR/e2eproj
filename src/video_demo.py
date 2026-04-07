#!/usr/bin/env python3
"""
Create a continuous-looking demo video from sparse keyframe inference.

The expensive image pipeline only runs on sampled keyframes. Intermediate
video frames reuse the most recent keyframe result for lightweight overlays.
"""
import argparse
import logging
import os
import shutil
import sys
from bisect import bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a sparse-inference video demo.")
    parser.add_argument("--config", "-c", default="configs/default_config.yaml")
    parser.add_argument("--video", "-v", default="../outdoor2.mp4")
    parser.add_argument("--output-root", default="data/video_demo")
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--max-keyframes", type=int, default=8)
    parser.add_argument("--output-fps", type=float, default=12.0)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap

    cap.release()
    ascii_copy = project_root / "data" / "temp" / "video_demo_input.mp4"
    ascii_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, ascii_copy)

    cap = cv2.VideoCapture(str(ascii_copy))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> Tuple[float, int, int, int, float]:
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / source_fps if source_fps > 0 else 0.0

    if source_fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid; cannot create demo output.")

    return source_fps, frame_count, width, height, duration


def build_run_dirs(output_root: Path) -> Dict[str, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    pipeline_dir = run_dir / "pipeline_outputs"
    dirs = {
        "run": run_dir,
        "keyframes": run_dir / "keyframes",
        "depth": pipeline_dir / "depth_image",
        "sam3": pipeline_dir / "sam3_json",
        "temp": pipeline_dir / "temp",
        "correct": pipeline_dir / "correct",
        "navigation": pipeline_dir / "navigation",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    return dirs


def build_keyframe_times(duration: float, max_duration: float, interval: float, max_keyframes: int) -> List[float]:
    usable_duration = min(duration, max_duration) if duration > 0 else max_duration
    usable_duration = max(0.0, usable_duration)
    interval = max(0.1, interval)
    max_keyframes = max(1, max_keyframes)

    times = []
    current = 0.0
    while current <= usable_duration + 1e-6 and len(times) < max_keyframes:
        times.append(round(current, 3))
        current += interval
    return times or [0.0]


def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame at {time_sec:.2f}s")
    return frame


def write_keyframe(frame: np.ndarray, folder: Path, index: int, time_sec: float) -> Path:
    path = folder / f"keyframe_{index:03d}_{int(time_sec * 1000):06d}ms.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Unable to write keyframe: {path}")
    return path


def run_keyframe_pipeline(
    pipeline: Any,
    video_path: Path,
    cap: cv2.VideoCapture,
    keyframe_times: Sequence[float],
    dirs: Dict[str, Path],
) -> List[Dict[str, Any]]:
    keyframes = []
    for index, time_sec in enumerate(keyframe_times):
        logger.info("Processing keyframe %s/%s at %.2fs", index + 1, len(keyframe_times), time_sec)
        frame = read_frame_at_time(cap, time_sec)
        image_path = write_keyframe(frame, dirs["keyframes"], index, time_sec)
        result = pipeline.process_single_image(
            str(image_path),
            str(dirs["depth"]),
            str(dirs["sam3"]),
            str(dirs["temp"]),
            str(dirs["correct"]),
            str(dirs["navigation"]),
            play_tts=False,
        )
        keyframes.append({
            "index": index,
            "time_sec": time_sec,
            "image_path": str(image_path),
            "source_video": str(video_path),
            "result": result,
        })
    return keyframes


def as_points(value: Any) -> Optional[np.ndarray]:
    if not value:
        return None
    points = np.array(value, dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    return points[:, :2]


def draw_box(frame: np.ndarray, points_value: Any, color: Tuple[int, int, int], label: str):
    points = as_points(points_value)
    if points is None:
        return
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label_at = (int(x1), max(20, int(y1) - 8))
    else:
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        label_at = tuple(points[0])
        label_at = (int(label_at[0]), max(20, int(label_at[1]) - 8))
    cv2.putText(frame, label, label_at, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int = 72) -> List[str]:
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_text_panel(frame: np.ndarray, lines: Sequence[str]):
    if not lines:
        return
    height, width = frame.shape[:2]
    panel_height = 36 + 28 * len(lines)
    y1 = max(0, height - panel_height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = y1 + 28
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def overlay_keyframe_result(frame: np.ndarray, keyframe: Dict[str, Any], current_time: float) -> np.ndarray:
    output = frame.copy()
    result = keyframe["result"]
    correct_results = result.get("correct_result", {}).get("correct_results", [])
    navigation = result.get("navigation_result", {})

    cv2.putText(
        output,
        f"Demo time {current_time:05.2f}s | keyframe {keyframe['time_sec']:05.2f}s",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for item in correct_results:
        if "error" in item:
            continue
        draw_box(output, item.get("sign_box"), (255, 170, 0), "sign")
        draw_box(output, item.get("arrow_box_4pts") or item.get("arrow_box"), (0, 255, 0), "arrow")

        theta = item.get("theta_deg")
        anchor = as_points(item.get("arrow_box_4pts") or item.get("arrow_box") or item.get("sign_box"))
        if theta is not None and anchor is not None:
            center = np.mean(anchor, axis=0).astype(int)
            cv2.putText(
                output,
                f"3D dir: {float(theta):.0f} deg",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    sentence = navigation.get("navigation_sentence", "")
    draw_text_panel(output, wrap_text(sentence))
    return output


def find_active_keyframe(keyframes: Sequence[Dict[str, Any]], time_sec: float) -> Dict[str, Any]:
    times = [item["time_sec"] for item in keyframes]
    index = bisect_right(times, time_sec) - 1
    index = max(0, min(index, len(keyframes) - 1))
    return keyframes[index]


def write_demo_video(
    video_path: Path,
    keyframes: Sequence[Dict[str, Any]],
    output_path: Path,
    max_duration: float,
    output_fps: float,
):
    cap = open_video(video_path)
    try:
        source_fps, _, width, height, duration = get_video_info(cap)
        usable_duration = min(duration, max_duration) if duration > 0 else max_duration
        total_output_frames = max(1, int(round(usable_duration * output_fps)))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {output_path}")

        try:
            for output_index in range(total_output_frames):
                time_sec = output_index / output_fps
                frame_index = int(round(time_sec * source_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                active_keyframe = find_active_keyframe(keyframes, time_sec)
                writer.write(overlay_keyframe_result(frame, active_keyframe, time_sec))
        finally:
            writer.release()
    finally:
        cap.release()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()

    video_path = resolve_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = resolve_path(args.output_root)
    dirs = build_run_dirs(output_root)

    cap = open_video(video_path)
    try:
        _, _, width, height, duration = get_video_info(cap)
        keyframe_times = build_keyframe_times(
            duration=duration,
            max_duration=args.max_duration,
            interval=args.sample_interval,
            max_keyframes=args.max_keyframes,
        )
        logger.info("Video: %s (%sx%s, %.2fs)", video_path, width, height, duration)
        logger.info("Keyframes: %s", ", ".join(f"{item:.2f}s" for item in keyframe_times))

        from src.pipeline.pipeline_manager import PipelineManager

        pipeline = PipelineManager(str(resolve_path(args.config)))
        try:
            keyframes = run_keyframe_pipeline(pipeline, video_path, cap, keyframe_times, dirs)
        finally:
            pipeline.release()
    finally:
        cap.release()

    output_video = dirs["run"] / "demo_video.mp4"
    write_demo_video(
        video_path=video_path,
        keyframes=keyframes,
        output_path=output_video,
        max_duration=args.max_duration,
        output_fps=args.output_fps,
    )

    print(f"Video demo saved: {output_video}")


if __name__ == "__main__":
    main()
