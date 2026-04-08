import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import glob
import logging
from pathlib import Path
import sys
import re

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def images_to_video_opencv(image_folder, output_video, fps=30, image_pattern="*.jpg", 
                           sort_by_number=True, reverse=False):
    """
    使用OpenCV将图片合成为视频
    
    参数:
        image_folder: 图片文件夹路径
        output_video: 输出视频文件路径
        fps: 帧率（每秒帧数）
        image_pattern: 图片文件模式（如 "*.jpg", "*.png"）
        sort_by_number: 是否按数字排序
        reverse: 是否反向排序
    """
    
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(image_folder, image_pattern))
    
    if not image_files:
        print(f"在 {image_folder} 中没有找到 {image_pattern} 格式的图片")
        return
    
    # 按数字排序（如果文件名包含数字）
    if sort_by_number:
        def extract_number(filename):
            # 从文件名中提取第一个出现的数字序列（不限制在扩展名）
            match = re.search(r'(\d+)', os.path.basename(filename))
            return int(match.group(1)) if match else 0
        
        image_files.sort(key=extract_number)
    
    # 排序（如果需要反向）
    if reverse:
        image_files.reverse()
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"无法读取第一张图片: {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    frame_size = (width, height)
    print(f"图片尺寸: {width} x {height}")
    
    # 创建视频写入器
    # 根据输出文件扩展名选择编码器
    video_ext = os.path.splitext(output_video)[1].lower()
    
    if video_ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    elif video_ext == '.mp4':
        # 尝试不同的MP4编码器
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4编码
        except:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
    elif video_ext == '.mov':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(f"未识别的视频格式 {video_ext}，使用AVI格式")
    
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    if not video_writer.isOpened():
        print(f"无法创建视频文件: {output_video}")
        print("请尝试不同的编码器或输出格式")
        return
    
    # 逐帧写入图片
    for i, image_file in enumerate(image_files):
        print(f"处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_file)}")
        
        img = cv2.imread(image_file)
        if img is None:
            print(f"  警告: 无法读取图片 {image_file}，跳过")
            continue
        
        # 检查图片尺寸是否一致
        if img.shape[:2] != (height, width):
            print(f"  警告: 图片尺寸不一致 ({img.shape[1]}x{img.shape[0]})，将调整为 {width}x{height}")
            img = cv2.resize(img, frame_size)
        
        video_writer.write(img)
    
    # 释放资源
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"视频已保存到: {output_video}")
    print(f"总帧数: {len(image_files)}")
    print(f"帧率: {fps} fps")
    print(f"时长: {len(image_files)/fps:.2f} 秒")


def load_json(json_path):
    """加载JSON文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_text_dimensions(text, font_size=15):
    """获取文本的尺寸"""
    # 创建一个临时图像来测量文本尺寸
    img = Image.new('RGB', (100, 100))
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # 获取文本边界框
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_size = draw.textsize(text, font=font)
        text_width, text_height = text_size
    
    return text_width, text_height

def draw_text_with_background(img, text, position, font_size=15, 
                              bg_color=(0, 255, 0), text_color=(0, 0, 0),
                              padding=5):
    """在图像上绘制带背景的文本"""
    # 将OpenCV图像转换为PIL图像
    if len(img.shape) == 2:
        pil_image = Image.fromarray(img)
    else:
        # BGR转RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
    
    # 创建Draw对象
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # 获取文本边界框
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        text_size = draw.textsize(text, font=font)
        text_bbox = (0, 0, text_size[0], text_size[1])
    
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 计算背景矩形位置
    bg_x1 = position[0] - padding
    bg_y1 = position[1] - padding
    bg_x2 = position[0] + text_width + padding
    bg_y2 = position[1] + text_height + padding
    
    # 绘制背景矩形
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=text_color)
    
    # 将PIL图像转换回OpenCV格式
    if len(img.shape) == 2:
        result = np.array(pil_image)
    else:
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result, (bg_x1, bg_y1, bg_x2, bg_y2)

def create_box_region_mask(points, img_width, img_height, expansion=5):
    """
    创建线框区域的掩码（包括内部和外部安全距离）
    expansion: 安全距离（像素），确保文本离线框足够远
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # 绘制线框内部区域的掩码
    cv2.fillPoly(mask, [points], 255)
    
    # 如果线框太小，只考虑内部区域
    # 否则，添加安全距离
    if expansion > 0:
        # 创建一个膨胀核
        kernel = np.ones((expansion, expansion), np.uint8)
        # 对线框掩码进行膨胀
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask

def is_position_overlap_with_boxes(text_rect, forbidden_masks):
    """检查文本位置是否与任何线框掩码重叠"""
    x1, y1, x2, y2 = text_rect
    
    # 检查是否与任何线框掩码重叠
    for mask in forbidden_masks:
        # 提取文本区域对应的掩码部分
        mask_height, mask_width = mask.shape
        text_region_mask = mask[max(0, y1):min(mask_height, y2+1), 
                                max(0, x1):min(mask_width, x2+1)]
        if np.any(text_region_mask > 0):
            return True
    
    return False

def is_position_overlap_with_other_texts(text_rect, used_text_rectangles):
    """检查文本位置是否与其他文本重叠"""
    x1, y1, x2, y2 = text_rect
    
    for used_rect in used_text_rectangles:
        used_x1, used_y1, used_x2, used_y2 = used_rect
        # 检查是否有重叠
        if not (x2 < used_x1 or x1 > used_x2 or y2 < used_y1 or y1 > used_y2):
            return True
    
    return False

def is_position_within_bounds(text_rect, img_width, img_height):
    """检查文本位置是否在图像边界内"""
    x1, y1, x2, y2 = text_rect
    return not (x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height)

def find_closest_text_position(points, img_width, img_height, text_width, text_height,
                               forbidden_masks, used_text_rectangles, padding=5,
                               arrow_direction=None):
    """
    为指定线框寻找最近的文本位置
    策略：尝试线框周围的多个位置，找到最近的有效位置
    参数：
        arrow_direction: (dx, dy) 法向量箭头的方向向量，用于避免与箭头重叠
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    center_x, center_y = np.mean(points, axis=0).astype(int)
    
    # 计算线框的宽高
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    # 定义搜索位置：优先考虑靠近线框的位置
    # 格式：(位置类型, x, y, 距离权重)
    # 距离权重越小，位置越优先
    
    search_positions = []
    
    # 根据箭头方向调整优先级
    # 如果有箭头信息，优先选择与箭头方向相反的位置
    above_weight = 1
    below_weight = 2
    left_weight = 3
    right_weight = 4
    
    if arrow_direction is not None:
        arrow_dx, arrow_dy = arrow_direction
        # 如果箭头主要指向下方，优先选择上方（降低上方权重）
        if arrow_dy > 0.3:
            above_weight = 0.5  # 优先上方
            below_weight = 3
        # 如果箭头主要指向上方，优先选择下方
        elif arrow_dy < -0.3:
            above_weight = 3
            below_weight = 0.5  # 优先下方
        
        # 如果箭头主要指向左方，优先选择右方
        if arrow_dx < -0.3:
            left_weight = 3
            right_weight = 0.5  # 优先右方
        # 如果箭头主要指向右方，优先选择左方
        elif arrow_dx > 0.3:
            left_weight = 0.5  # 优先左方
            right_weight = 3
    
    # 1. 紧贴线框上方（中心对齐）
    y_above = min_y - text_height - padding - 3  # 3像素的安全距离
    if y_above >= 0:
        search_positions.append(("above_center", center_x - text_width//2, y_above, above_weight))
    
    # 2. 紧贴线框下方（中心对齐）
    y_below = max_y + 3  # 3像素的安全距离
    if y_below + text_height + padding < img_height:
        search_positions.append(("below_center", center_x - text_width//2, y_below, below_weight))
    
    # 3. 紧贴线框左侧（垂直居中）
    x_left = min_x - text_width - padding - 3
    if x_left >= 0:
        search_positions.append(("left_center", x_left, center_y - text_height//2, left_weight))
    
    # 4. 紧贴线框右侧（垂直居中）
    x_right = max_x + 3
    if x_right + text_width + padding < img_width:
        search_positions.append(("right_center", x_right, center_y - text_height//2, right_weight))
    
    # 5. 线框左上角附近
    search_positions.append(("corner_top_left", min_x, min_y - text_height - padding - 3, 5))
    
    # 6. 线框右上角附近
    search_positions.append(("corner_top_right", max_x - text_width, min_y - text_height - padding - 3, 6))
    
    # 7. 线框左下角附近
    search_positions.append(("corner_bottom_left", min_x, max_y + 3, 7))
    
    # 8. 线框右下角附近
    search_positions.append(("corner_bottom_right", max_x - text_width, max_y + 3, 8))
    
    # 9. 如果上述位置都无效，尝试在更近的位置搜索
    # 沿着线框周围搜索
    search_radius = 50  # 最大搜索半径
    step_size = 5
    
    # 搜索上方区域
    for offset in range(0, search_radius, step_size):
        y = min_y - text_height - padding - offset
        if y >= 0:
            for x_offset in range(0, min(box_width, 100), step_size):
                x = min_x + x_offset
                if x + text_width + padding < img_width:
                    search_positions.append((f"search_above_{offset}", x, y, 9 + offset))
    
    # 搜索下方区域
    for offset in range(0, search_radius, step_size):
        y = max_y + offset
        if y + text_height + padding < img_height:
            for x_offset in range(0, min(box_width, 100), step_size):
                x = min_x + x_offset
                if x + text_width + padding < img_width:
                    search_positions.append((f"search_below_{offset}", x, y, 10 + offset))
    
    # 搜索左侧区域
    for offset in range(0, search_radius, step_size):
        x = min_x - text_width - padding - offset
        if x >= 0:
            for y_offset in range(0, min(box_height, 100), step_size):
                y = min_y + y_offset
                if y + text_height + padding < img_height:
                    search_positions.append((f"search_left_{offset}", x, y, 11 + offset))
    
    # 搜索右侧区域
    for offset in range(0, search_radius, step_size):
        x = max_x + offset
        if x + text_width + padding < img_width:
            for y_offset in range(0, min(box_height, 100), step_size):
                y = min_y + y_offset
                if y + text_height + padding < img_height:
                    search_positions.append((f"search_right_{offset}", x, y, 12 + offset))
    
    # 按距离权重排序（权重越小越优先）
    search_positions.sort(key=lambda x: x[3])
    
    # 尝试所有位置
    for pos_type, x, y, weight in search_positions:
        text_rect = (x - padding, y - padding, 
                    x + text_width + padding, 
                    y + text_height + padding)
        
        # 检查位置是否有效
        if not is_position_within_bounds(text_rect, img_width, img_height):
            continue
        
        if is_position_overlap_with_boxes(text_rect, forbidden_masks):
            continue
        
        if is_position_overlap_with_other_texts(text_rect, used_text_rectangles):
            continue
        
        return pos_type, x, y
    
    # 如果所有位置都无效，尝试强制放置（可能与其他文本重叠，但不与线框重叠）
    # 优先选择靠近线框的位置
    for pos_type, x, y, weight in search_positions[:20]:  # 只检查前20个最近的位置
        text_rect = (x - padding, y - padding, 
                    x + text_width + padding, 
                    y + text_height + padding)
        
        if not is_position_within_bounds(text_rect, img_width, img_height):
            continue
        
        if is_position_overlap_with_boxes(text_rect, forbidden_masks):
            continue
        
        # 即使与其他文本重叠也接受
        return f"forced_{pos_type}", x, y
    
    # 最后手段：放在图像角落
    corner_x = padding
    corner_y = padding
    return "corner", corner_x, corner_y

def compute_normal_endpoint_improved(center_2d, normal_3d, base_scale=30, min_length=10, max_length=50):
    """
    改进的法向量终点计算
    考虑法向量的3D特性和可视化直觉
    """
    # 3D坐标系：x右，y上，z朝向相机
    # 2D图像坐标系：x右，y下
    
    # 1. 提取3D法向量分量
    normal_x = normal_3d[0]  # x: 右（正值为向右）
    normal_y = -normal_3d[1]  # y: 下（图像y轴向下，所以取负）
    normal_z = normal_3d[2]  # z: 朝向相机（正值为朝向相机）
    
    # 2. 计算法向量的2D投影（忽略z分量，只考虑方向）
    norm_2d = np.sqrt(normal_x**2 + normal_y**2)
    
    # 3. 计算动态长度：基于法向量方向和z分量
    #    - 如果法向量主要朝向相机（z接近1），应该显示较短
    #    - 如果法向量主要侧向（z接近0），应该显示正常长度
    #    - 如果法向量主要远离相机（z接近-1），应该显示较长
    
    # 计算法向量与相机方向的夹角余弦
    # z分量就是法向量与相机方向（z轴）的点积
    
    # 计算动态缩放因子
    # 当z=1（完全朝向相机）：缩放因子小
    # 当z=0（垂直方向）：缩放因子中等
    # 当z=-1（完全远离相机）：缩放因子大
    z_factor = 1.0 - normal_z * 0.5  # z从-1到1，因子从1.5到0.5
    
    # 计算最终长度
    if norm_2d > 0:
        # 归一化x和y分量
        normal_x /= norm_2d
        normal_y /= norm_2d
        
        # 计算动态长度
        dynamic_length = base_scale * z_factor
        
        # 确保在最小和最大范围内
        dynamic_length = max(min_length, min(max_length, dynamic_length))
        
        # 计算终点
        end_x = int(center_2d[0] + normal_x * dynamic_length)
        end_y = int(center_2d[1] + normal_y * dynamic_length)
    else:
        # 如果x和y分量都为0，法向量完全垂直（只沿z方向）
        # 这种情况下，我们无法在2D上显示方向，可以显示一个点或使用默认方向
        normal_x = 0.0
        normal_y = 0.0
        dynamic_length = 0  # 长度为0，不显示箭头
        end_x = center_2d[0]
        end_y = center_2d[1]
        
    return (end_x, end_y), (normal_x, normal_y, normal_z), dynamic_length

def add_3d_visualization_hint(img, center, endpoint, normal_components, length):
    """添加3D可视化提示"""
    # 提取法向量分量
    normal_x, normal_y, normal_z = normal_components
    
    # 只有在长度大于0时才添加提示
    if length > 0:
        # # 在法向量旁边添加文本信息
        # text = f"n=({normal_x:.2f},{normal_y:.2f},{normal_z:.2f})"
        # cv2.putText(img, text, (endpoint[0] + 10, endpoint[1]), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # # 添加长度信息
        # length_text = f"len={length:.0f}"
        # cv2.putText(img, length_text, (endpoint[0] + 10, endpoint[1] + 15), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 根据z分量添加颜色提示
        # 朝向相机（z>0）：红色更深
        # 远离相机（z<0）：红色更浅
        alpha = 0.5 + normal_z * 0.3  # alpha从0.2（z=-1）到0.8（z=1）
        alpha = max(0.2, min(0.8, alpha))
        
        # 绘制半透明圆表示z分量
        radius = 5
        overlay = img.copy()
        color = (0, 0, int(255 * (0.5 + normal_z * 0.5)))  # 根据z调整蓝色分量
        # cv2.circle(overlay, endpoint, radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def visualize_normal_analysis(normals_list, output_path="normal_analysis.png"):
    """创建法向量分析图，帮助理解可视化效果"""
    img_size = 800
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 中心点
    center = (img_size // 2, img_size // 2)
    
    # 绘制坐标轴
    cv2.arrowedLine(img, center, (img_size - 50, center[1]), (0, 0, 0), 2, tipLength=0.05)
    cv2.arrowedLine(img, center, (center[0], 50), (0, 0, 0), 2, tipLength=0.05)
    
    cv2.putText(img, "X (Right)", (img_size - 100, center[1] + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Y (Down)", (center[0] - 50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Camera", (center[0] - 30, center[1] + 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 绘制示例法向量
    examples = [
        # 名称, 法向量, 颜色, 描述
        ("Right", [1, 0, 0], (0, 255, 0), "指向右侧"),
        ("Left", [-1, 0, 0], (0, 200, 0), "指向左侧"),
        ("Up", [0, -1, 0], (255, 0, 0), "指向上方（3D坐标系）"),
        ("Down", [0, 1, 0], (200, 0, 0), "指向下方（3D坐标系）"),
        ("Toward Camera", [0, 0, 1], (0, 0, 255), "朝向相机"),
        ("Away from Camera", [0, 0, -1], (0, 0, 200), "远离相机"),
        ("Diagonal 1", [0.7, 0, 0.7], (255, 0, 255), "右+朝向相机"),
        ("Diagonal 2", [0, 0.7, -0.7], (0, 255, 255), "下+远离相机"),
    ]
    
    y_offset = 250
    for i, (name, normal, color, desc) in enumerate(examples):
        # 使用改进的方法计算终点
        endpoint, normal_components, length = compute_normal_endpoint_improved(center, normal, 40)
        
        # 只在长度大于0时绘制箭头
        if length > 0:
            cv2.arrowedLine(img, center, endpoint, color, 2, tipLength=0.2)
        
        # 添加说明文本
        text_y = y_offset + i * 60
        cv2.putText(img, f"{name}: {normal}", (50, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, desc, (300, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"2D投影: ({normal_components[0]:.2f},{normal_components[1]:.2f})", 
                   (500, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"长度: {length:.0f}", (700, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 添加标题和说明
    cv2.putText(img, "法向量可视化分析", (50, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "坐标系：X右，Y下（图像坐标系），Z朝向相机", (50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "法向量长度动态调整：朝向相机时较短，远离相机时较长", (50, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, img)
    print(f"法向量分析图已保存: {output_path}")
    
    return img

def visualize_on_image(image_path, results_for_image, output_path):
    """在图像上可视化多个结果"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    img_height, img_width = img.shape[:2]
    
    # 准备数据
    points_list = []
    texts_list = []
    
    for result in results_for_image:
        points_2d = result["points_2d"]
        alpha_deg = result["alpha_deg"]
        theta_deg = result["theta_deg"]
        
        points = np.array(points_2d, dtype=np.int32)
        points_list.append(points)
        
        # 准备文本内容
        alpha_text = f"2D [{alpha_deg:.0f}°]"
        theta_text = f"3D [{theta_deg:.0f}°]"
        texts_list.append((alpha_text, theta_text))
    
    # 第一步：绘制所有线框
    for points in points_list:
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 第二步：为所有线框创建禁止区域掩码（安全距离=5像素）
    forbidden_masks = []
    for points in points_list:
        mask = create_box_region_mask(points, img_width, img_height, expansion=5)
        forbidden_masks.append(mask)
    
    # 第三步：按线框大小排序（先处理小线框，因为大线框有更多空间）
    box_areas = []
    for points in points_list:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        box_areas.append(area)
    
    # 按面积排序（从小到大）
    sorted_indices = np.argsort(box_areas)
    
    # 第四步：为每个线框寻找文本位置并绘制
    used_text_rectangles = []
    
    for idx in sorted_indices:
        points = points_list[idx]
        alpha_text, theta_text = texts_list[idx]
        
        # 获取文本尺寸
        font_size = 20
        padding = 5
        
        alpha_width, alpha_height = get_text_dimensions(alpha_text, font_size)
        theta_width, theta_height = get_text_dimensions(theta_text, font_size)
        text_width = max(alpha_width, theta_width)
        text_height = alpha_height + theta_height +5  # 5像素行间距
        
        # 寻找最近的文本位置
        pos_type, text_x, text_y = find_closest_text_position(
            points, img_width, img_height, text_width, text_height,
            forbidden_masks, used_text_rectangles, padding
        )
        
        # 绘制带背景的文本
        img, bbox_alpha = draw_text_with_background(
            img, alpha_text, (text_x, text_y), 
            font_size=font_size, bg_color=(0, 255, 0), text_color=(0, 0, 0),
            padding=padding
        )
        
        img, bbox_theta = draw_text_with_background(
            img, theta_text, (text_x, text_y + alpha_height + 5), 
            font_size=font_size, bg_color=(0, 255, 0), text_color=(0, 0, 0),
            padding=padding
        )
        
        # 将两个文本框合并为一个区域
        combined_x1 = min(bbox_alpha[0], bbox_theta[0])
        combined_y1 = min(bbox_alpha[1], bbox_theta[1])
        combined_x2 = max(bbox_alpha[2], bbox_theta[2])
        combined_y2 = max(bbox_alpha[3], bbox_theta[3])
        
        # 记录已使用的文本区域
        used_text_rectangles.append((combined_x1, combined_y1, combined_x2, combined_y2))
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"图像已保存: {output_path}")
    return img

def visualize_on_depth_image(depth_path, results_for_depth, output_path):
    """改进的深度图像可视化，包含更好的法向量表示"""
    # 读取深度图像
    try:
        depth_pil = Image.open(depth_path)
        depth_array = np.array(depth_pil)
    except Exception as e:
        depth_array = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_array is None:
        print(f"无法读取深度图像: {depth_path}")
        return None
    
    # 将深度图像转换为3通道以便绘制彩色
    if len(depth_array.shape) == 2:
        depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2BGR)
    else:
        depth_rgb = depth_array.copy()
    
    # 归一化深度值以便显示
    if depth_rgb.dtype != np.uint8:
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_rgb = cv2.cvtColor(depth_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    depth_height, depth_width = depth_rgb.shape[:2]
    
    # 自动计算字体大小：根据图像高度
    # 图像高度的 1/18 作为字体大小（比原来更小），最小 12，最大 50
    font_size = max(12, min(50, int(depth_height / 24)))
    
    # 准备数据
    points_list = []
    texts_list = []
    normals_list = []
    
    for result in results_for_depth:
        points_2d = result["points_2d"]
        normal_3d = result["normal"]
        azimuth_n = result["azimuth_n"]
        pitch = result["pitch"]
        points = np.array(points_2d, dtype=np.int32)
        points_list.append(points)
        
        # 准备文本内容
        if result["on_ground_ceiling"]:
            N_deg_text = f"N_deg [{pitch:.0f}°]"
        else:
            N_deg_text = f"N_deg [{azimuth_n:.0f}°]"
        texts_list.append(N_deg_text)
        normals_list.append(normal_3d)
    
    # 第一步：绘制所有线框和法向量
    for i, (points, normal_3d) in enumerate(zip(points_list, normals_list)):
        # 绘制绿色色线框
        cv2.polylines(depth_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 计算线框的中心点
        center = np.mean(points, axis=0).astype(int)
        
        # 使用改进的方法绘制法向量
        normal_endpoint, normal_components, length = compute_normal_endpoint_improved(
            center, normal_3d, base_scale=30, min_length=10, max_length=50
        )
        
        # 确保法向量终点在图像范围内
        normal_endpoint = (
            max(0, min(normal_endpoint[0], depth_width - 1)),
            max(0, min(normal_endpoint[1], depth_height - 1))
        )
        
        # 只在长度大于0时绘制箭头
        if length > 0:
            # 根据法向量z分量调整箭头粗细（朝向相机时较细，远离相机时较粗）
            normal_z = normal_3d[2]
            arrow_thickness = max(1, int(2 + normal_z * 2))  # z从-1到1，线宽从1到3
            
            # 将法向量末端点连接到线框的4个角点（白色连接线）
            for point in points:
                cv2.line(depth_rgb, normal_endpoint, tuple(point), 
                        color=(255, 255, 0), thickness=2)
            
            # 绘制法向量（黄色箭头）
            cv2.arrowedLine(depth_rgb, tuple(center), normal_endpoint, 
                           color=(0, 255, 255), thickness=2, tipLength=0.3)
            
            # 添加3D可视化提示
            depth_rgb = add_3d_visualization_hint(depth_rgb, center, normal_endpoint, 
                                                 normal_components, length)
        
        # # 绘制中心点
        # cv2.circle(depth_rgb, tuple(center), 3, (0, 255, 255), -1)
    
    # 第二步：为所有线框创建禁止区域掩码（安全距离=5像素）
    forbidden_masks = []
    for points in points_list:
        mask = create_box_region_mask(points, depth_width, depth_height, expansion=5)
        forbidden_masks.append(mask)
    
    # 第三步：按线框大小排序
    box_areas = []
    for points in points_list:
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)
        box_areas.append(area)
    
    sorted_indices = np.argsort(box_areas)
    
    # 第四步：为每个线框寻找文本位置并绘制
    used_text_rectangles = []
    
    for idx in sorted_indices:
        points = points_list[idx]
        N_deg_text = texts_list[idx]
        normal_3d = normals_list[idx]
        
        # 获取文本尺寸
        padding = 5
        
        N_deg_width, N_deg_height = get_text_dimensions(N_deg_text, font_size)
        
        # 计算箭头方向（用于避免文本与箭头重叠）
        # 提取法向量的x和y分量（在2D图像坐标系中）
        arrow_dx = normal_3d[0]  # x: 右为正
        arrow_dy = -normal_3d[1]  # y: 下为正（取负是因为3D y向上，但2D y向下）
        
        # 归一化方向向量
        arrow_norm = np.sqrt(arrow_dx**2 + arrow_dy**2)
        if arrow_norm > 0:
            arrow_direction = (arrow_dx / arrow_norm, arrow_dy / arrow_norm)
        else:
            arrow_direction = None
        
        # 寻找最近的文本位置
        pos_type, text_x, text_y = find_closest_text_position(
            points, depth_width, depth_height, N_deg_width, N_deg_height,
            forbidden_masks, used_text_rectangles, padding,
            arrow_direction=arrow_direction
        )
        
        # 绘制带背景的文本（黄色背景）
        depth_rgb, bbox_N_deg = draw_text_with_background(
            depth_rgb, N_deg_text, (text_x, text_y),  # 使用 N_deg_text 而不是 N_deg_width
            font_size=font_size, bg_color=(0, 255, 255), text_color=(0, 0, 0),
            padding=padding
        )
        
        # 记录已使用的文本区域
        used_text_rectangles.append(bbox_N_deg)
    
    # 保存结果    
    cv2.imwrite(output_path, depth_rgb)
    print(f"深度图像已保存: {output_path}")
    return depth_rgb

def visualize_on_combined(image_vis, depth_vis, results_for_combined, output_path):
    """
    创建并排可视化结果：图像在左，深度图像在右
    individual_output_dir: 保存单独可视化结果的目录
    """

    # 获取图像尺寸
    img_h, img_w = image_vis.shape[:2]
    depth_h, depth_w = depth_vis.shape[:2]
    
    # 调整图像大小，使它们具有相同的高度
    # 使用两个图像中较大的高度
    target_height = max(img_h, depth_h)
    
    # 计算调整后的宽度，保持宽高比
    img_scale = target_height / img_h
    img_new_w = int(img_w * img_scale)
    img_new_h = target_height
    
    depth_scale = target_height / depth_h
    depth_new_w = int(depth_w * depth_scale)
    depth_new_h = target_height
    
    # 调整图像大小
    image_resized = cv2.resize(image_vis, (img_new_w, img_new_h))
    depth_resized = cv2.resize(depth_vis, (depth_new_w, depth_new_h))
    
    # 创建并排图像
    # 在两张图像之间添加一条白色分隔线
    separator_width = 5
    combined_width = img_new_w + depth_new_w + separator_width
    combined_height = target_height
    
    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 将图像放在左侧
    combined_image[0:img_new_h, 0:img_new_w] = image_resized
    
    # 添加白色分隔线
    combined_image[:, img_new_w:img_new_w+separator_width] = [255, 255, 255]
    
    # 将深度图像放在右侧
    combined_image[0:depth_new_h, img_new_w+separator_width:img_new_w+separator_width+depth_new_w] = depth_resized
    
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_color = (0, 0, 0)  # 黑色
    
    # 左侧标题：图像
    left_title = "Image"
    left_title_size = cv2.getTextSize(left_title, font, 1, 2)[0]
    left_title_x = img_new_w // 2 - left_title_size[0] // 2
    cv2.putText(combined_image, left_title, (left_title_x, 40), 
                font, 1, title_color, 2)
    
    # 右侧标题：深度图像
    right_title = "Depth Image"
    right_title_size = cv2.getTextSize(right_title, font, 1, 2)[0]
    right_title_x = img_new_w + separator_width + depth_new_w // 2 - right_title_size[0] // 2
    cv2.putText(combined_image, right_title, (right_title_x, 40), 
                font, 1, title_color, 2)
    
    # 添加线框说明
    legend_y = combined_height - 30
    
    # 图像线框说明
    image_legend = "Image: Green box"
    cv2.putText(combined_image, image_legend, (20, legend_y), 
                font, 0.7, (0, 255, 0), 2)
    
    # 深度图像线框说明
    depth_legend = "Depth: Yellow box + Red normal"
    cv2.putText(combined_image, depth_legend, (img_new_w + separator_width + 20, legend_y), 
                font, 0.7, (0, 255, 255), 2)
    
    # 保存并排图像
    cv2.imwrite(output_path, combined_image)
    print(f"并排图像已保存: {output_path}")
    return combined_image

def group_results_by_image_path(results):
    """按图像路径分组结果"""
    image_groups = {}
    depth_groups = {}
    
    for result in results:
        image_path = result["image_path"]
        depth_path = result["depth_path"]
        
        if image_path not in image_groups:
            image_groups[image_path] = []
        if depth_path not in depth_groups:
            depth_groups[depth_path] = []
        
        image_groups[image_path].append(  )
        depth_groups[depth_path].append(result)
    
    return image_groups, depth_groups

def visualize_all_results(  
                    json_files, 
                    visualize_i_folder_str,
                    visualize_d_folder_str,    
                    visualize_id_folder_str):

    """可视化JSON中的所有数据"""
    processed_images = set()
    processed_depths = set()
    
    # 遍历所有JSON文件
    for json_file in json_files:
        # 加载JSON数据
        data = load_json(json_file)
        correct_results = data["correct_results"]
        
        # 如果没有结果，跳过
        if not correct_results:
            print(f"JSON文件 {json_file} 中没有结果，跳过")
            continue
        
        print(f"从 {json_file} 加载了 {len(correct_results)} 个结果")
        
        # 1 图像结果                
        image_path = correct_results[0]["image_path"]
        # 创建输出路径
        base_image_name = os.path.basename(image_path)
        image_name_without_ext = os.path.splitext(base_image_name)[0]
        output_image_path = os.path.join(visualize_i_folder_str, f"annotated_{image_name_without_ext}.jpg")
        # 在图像上绘制所有结果
        img_vis = visualize_on_image(image_path, correct_results, output_image_path)
        # 添加到已处理集合
        processed_images.add(image_path)

        # 2 深度结果
        depth_path = correct_results[0]["depth_path"]
        # 创建输出路径
        base_depth_name = os.path.basename(depth_path)
        depth_name_without_ext = os.path.splitext(base_depth_name)[0]
        output_depth_path = os.path.join(visualize_d_folder_str, f"annotated_{depth_name_without_ext}_depth.png")
        # 在深度图像上绘制所有结果
        depth_vis = visualize_on_depth_image(depth_path, correct_results, output_depth_path)
        # 添加到已处理集合
        processed_depths.add(depth_path)
        
        # 3 图像&深度结果
        # 创建输出路径
        id_base_name = os.path.basename(image_path)
        id_name_without_ext = os.path.splitext(id_base_name)[0]
        output_id_path = os.path.join(visualize_id_folder_str, f"annotated_{id_name_without_ext}_combined.jpg")
        visualize_on_combined(img_vis, depth_vis, correct_results, output_id_path)

    print(f"\n完成！")
    
    return {
        "image_outputs": len(processed_images),
        "depth_outputs": len(processed_depths)
    }


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="修正结果可视化"
    )    

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/output/correct',
        help='可视化json文件目录'
    )

    parser.add_argument(
        '--visualize_i', '-vi',
        type=str,
        default='data/visualize/image',
        help='可视化json文件目录'
    )

    parser.add_argument(
        '--visualize_d', '-vd',
        type=str,
        default='data/visualize/depth',
        help='可视化json文件目录'
    )
    
    parser.add_argument(
        '--visualize_id', '-vid',
        type=str,
        default='data/visualize/image_depth',
        help='可视化json文件目录'
    )

    parser.add_argument(
        '--visualize_v', '-vv',
        type=str,
        default='data/visualize/video',
        help='可视化json文件目录'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=30,
        help='视频帧率 (默认: 30)'
    )

    return parser.parse_args()

def read_filename(json_folder):
    # 遍历所有输入图像
    json_files = glob.glob(os.path.join(json_folder, '*.json'))

    # 使用正则表达式提取数字部分并排序
    def extract_number(filename):
        # 从文件名中提取第一个出现的数字序列
        match = re.search(r'(\d+)', os.path.basename(filename))
        return int(match.group(1)) if match else 0

    # 按数字序号排序
    json_files.sort(key=extract_number)

    print(f"找到 {len(json_files)} 张图像，按序号排序:")
    for json_file in json_files:
        print(f"  {json_file}")

    return json_files
    
if __name__ == "__main__":
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.info("开始可视化...")

    # 可视化目录初始化
    visualize_i_folder_str = args.visualize_i
    visualize_i_folder = Path(visualize_i_folder_str)
    visualize_i_folder.mkdir(parents=True, exist_ok=True)

    visualize_d_folder_str = args.visualize_d
    visualize_d_folder = Path(visualize_d_folder_str)
    visualize_d_folder.mkdir(parents=True, exist_ok=True)

    visualize_id_folder_str = args.visualize_id
    visualize_id_folder = Path(visualize_id_folder_str)
    visualize_id_folder.mkdir(parents=True, exist_ok=True)
    
    visualize_v_folder_str = args.visualize_v
    visualize_v_folder = Path(visualize_v_folder_str)
    visualize_v_folder.mkdir(parents=True, exist_ok=True)

    # 1. 首先读取可视化json文件目录
    json_folder = args.input
    json_files = read_filename(json_folder)

    print("开始处理所有结果...")
    print("=" * 50)
    
    # 2. 创建标准可视化
    print("\n创建标准可视化...")
    results = visualize_all_results(
                            json_files, 
                            visualize_i_folder_str,
                            visualize_d_folder_str,    
                            visualize_id_folder_str
                            )
    
    print("\n" + "=" * 50)
    print("=== 可视化完成 ===")
    print(f"处理了 {results['image_outputs']} 张图像")
    print(f"处理了 {results['depth_outputs']} 张深度图像")

    images_to_video_opencv(
        image_folder=visualize_i_folder_str,  # 图片文件夹
        output_video=os.path.join(visualize_v_folder_str, "output_i_video.mp4"),  # 输出视频
        fps=args.fps,                           # 帧率
        image_pattern="*.jpg"             # 图片格式
    )

    images_to_video_opencv(
        image_folder=visualize_id_folder_str,  # 图片文件夹
        output_video=os.path.join(visualize_v_folder_str, "output_id_video.mp4"),  # 输出视频
        fps=args.fps,                           # 帧率
        image_pattern="*.jpg"             # 图片格式
    )