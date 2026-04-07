"""
后处理器：整合所有信息
"""
import numpy as np
from PIL import Image
import json
import os
import cv2
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class Correcter:
    """后处理器：整合深度、检测和语义信息"""
    
    def __init__(self, camera_config=None):
        """
        初始化后处理器
        
        Args:
            camera_config: 相机配置字典，包含 fov_x_deg 等参数
        """
        self.fov_x_deg = 90.0
        if camera_config:
            self.fov_x_deg = float(camera_config.get('fov_x_deg', 90.0))
        logger.info(f"Correcter initialized with FOV: {self.fov_x_deg}°")
    
    def process(self, image_path, image, depth_map, items, output_correct_json_folder_str):
        """
        处理单张图像的所有信息
        
        Args:
            image: numpy.array
            depth_map: 深度图 (H, W)
            
        Returns:
            result: 修正后的结果
        """      

        H, W = image.shape[:2]
        correct_results = []
        for idx, entry in enumerate(items):
            try:
                # 获取必要参数
                image_path = entry['image_path']
                depth_path = entry['depth_path']
                arrow_box = entry['arrow_box']
                arrow_box, _ = self.shrink_bounding_box(arrow_box, M=10, N=1)
                alpha_deg = entry['alpha']
                theta_gt = entry['theta']
                # 使用相机配置的 FOV
                fov_x_deg = self.fov_x_deg
                
                # 读取图像获取尺寸         
                
                # 从深度图像读取深度值
                corners_with_depth = self.load_depth_from_box(depth_map, arrow_box)
                
                # 估计箭头方向
                correct_result = self.estimate_arrow_orientation(
                    corners=corners_with_depth,
                    alpha_deg=alpha_deg,
                    image_width=W,
                    image_height=H,
                    depth_map=depth_map,
                    fov_x_deg=fov_x_deg
                )
                
                # 添加额外信息
                correct_result['image_path'] = image_path
                correct_result['depth_path'] = depth_path
                correct_result['image_width'] = W
                correct_result['image_height'] = H
                
                # 添加路牌文字信息（从Qwen3VL识别结果中传递）
                if 'sign_text' in entry:
                    correct_result['sign_text'] = entry['sign_text']
                
                correct_results.append(correct_result)
                
                print(f"角度修正 [{idx}] 成功处理: {os.path.basename(image_path)}")
                # print(f"    法向量: {[round(x, 3) for x in result['normal']]}")
                # print(f"    方位角: {result['azimuth_n']:.2f}°, 俯仰角: {result['pitch']:.2f}°")
                # print(f"    地面/天花板: {result['on_ground_ceiling']}")
                # print(f"    法向量方位角θ_n: {result['azimuth_n']}°")
                # print(f"    2D方向α: {result['alpha_deg']}°")
                # print(f"    3D方向θ: {result['theta_deg']}°")
                # print(f"    3D方向θ_gt: {theta_gt}°")
                # print("~"*100,"\r")
                
            except Exception as e:
                print(f"角度修正 [{idx}] 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
                correct_results.append({
                    'image_path': entry.get('image_path', ''),
                    'error': str(e)
                })
        
        # 保存结果
        correct_data = {
            'correct_results': correct_results
        }
        
        correct_data_serializable = self.convert_to_serializable(correct_data)
        
        # 保存结果  
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        correct_info = f"{base_name}_correct.json"
        correct_info_file_path = os.path.join(output_correct_json_folder_str,correct_info)
        with open(correct_info_file_path, 'w', encoding='utf-8') as f:
            json.dump(correct_data_serializable, f, indent=2, ensure_ascii=False)
        print(f"角度修正已完成, 修正结果已保存: {correct_info_file_path}")
        
        # 返回修正后的结果列表,供后续使用
        return correct_results
    
    
    def shrink_bounding_box(self,
        corners: List[List[float]], 
        M: float, 
        N: float
    ) -> Tuple[List[List[float]], bool]:
        """
        缩小边框到1/N，如果满足条件则缩小，否则返回原始边框
        
        参数:
            corners: 边框角点，2个点（对角点）或4个点
            M: 系数，用于计算最小允许边长（最小边长 > M × N）
            N: 缩小倍数
            always_return_four_points: 是否总是返回四个角点
            
        返回:
            (边框点列表, 是否成功缩小)
        """
        # 参数验证
        if not corners or len(corners) not in [2, 4]:
            return corners, False
        
        # 转换为numpy数组以便计算
        corners_array = np.array(corners, dtype=np.float64)
        
        # 计算矩形的边界
        x_min, y_min = np.min(corners_array, axis=0)
        x_max, y_max = np.max(corners_array, axis=0)
        
        # 计算宽度和高度
        width = x_max - x_min
        height = y_max - y_min
        
        # 计算最小边长
        min_side = min(width, height)
        
        # 检查是否满足缩小条件
        if min_side <= M * N:
            # 不满足条件
            return corners, False
        
        # 满足条件，执行缩小操作
        # 保持两个点
        center = np.mean(corners_array, axis=0)
        shrunk_corners = center + (corners_array - center) / N
        result = shrunk_corners.tolist()
        
        return result, True
    
    def load_depth_from_box(self, depth_map, box):
        """
        从深度图像读取角点的深度值
        
        参数:
            depth_map: 深度图像
            boxes: 角点列表，每个元素是[x, y]坐标
            
        返回:
            更新了深度值的角点列表
        """
        # 读取深度图像        
        H, W = depth_map.shape
        
        # 输入是两个对角点
        if len(box) == 2:
            x_tl, y_tl = box[0]
            x_bl, y_bl = box[1]
            # 生成4个角点（顺时针方向）
            arrow_box = [
                [x_tl, y_tl],           # 左上角
                [x_bl, y_tl],           # 右上角
                [x_bl, y_bl],           # 右下角
                [x_tl, y_bl]            # 左下角
            ]
        elif len(box) == 4:
            arrow_box = box
        else:
            raise ValueError(f"需要2个或4个角点，得到{len(box)}个")
        
        corners_with_depth = []
        for corner in arrow_box:
            x = int(round(corner[0]))
            y = int(round(corner[1]))
            
            # 确保坐标在图像范围内
            x = np.clip(x, 0, W-1)
            y = np.clip(y, 0, H-1)
            
            # 读取深度值
            depth_pixel = depth_map[y, x]
            
            corners_with_depth.append({
                'x': float(corner[0]),
                'y': float(corner[1]),
                'depth': float(depth_pixel)
            })
        
        return corners_with_depth
    
    def estimate_arrow_orientation(self,
        corners,
        alpha_deg,
        image_width,
        image_height,
        depth_map,
        fov_x_deg: float = 90.0,
    ):
        """
        估计箭头在三维空间下的真实水平方位角
        
        参数:
            corners: 4个角点的列表，每个元素是包含'x','y','depth'的字典
            alpha_deg: 箭头2D方向角度
            image_width: 图像宽度
            image_height: 图像高度
            fov_x_deg: 水平视场角
            
        返回:
            包含结果信息的字典
        """
        # 1. 将角点转换为3D点
        points_3d_corners = []
        corners_2d_list = []
        for corner in corners:
            x, y, depth = corner['x'], corner['y'], corner['depth']
            corners_2d_list.append([x, y])
            
            ray = self.pixel_to_ray_direction(x, y, image_width, image_height, fov_x_deg)
            point_3d = ray * depth
            points_3d_corners.append(point_3d)
        
        points_3d_corners = np.array(points_3d_corners, dtype=np.float64)
        corners_2d = np.array(corners_2d_list, dtype=np.float32)
        
        # --- 新增的采样流程 ---
        points_3d_sampled = None
        # print("  尝试从深度图采样更多点...")
        points_3d_sampled = self.sample_points_and_get_3d_vectorized(
            corners_2d, depth_map, image_width, image_height, fov_x_deg, sample_step=5
        )

        # 决定使用哪个点集
        if points_3d_sampled is not None:
            # print(f"  采样成功，使用 {len(points_3d_sampled)} 个点进行平面拟合。")
            points_3d_for_fit = points_3d_sampled
        else:
            # print("  采样失败或点数不足，回退到仅使用4个角点。")
            points_3d_for_fit = points_3d_corners
        
        # 调试输出点坐标
        # print(f"3D点深度:")
        # print(f"corners {corners}")
        # print(f"3D点坐标:")
        # for i, pt in enumerate(points_3d_corners):
        #     print(f"  P{i}: {pt}")
        
        # 2. 计算平面法向量
        normal = self.compute_plane_normal(points_3d_for_fit)
        
        # 3. 计算法向量的方位角和俯仰角
        azimuth_n = self.compute_azimuth_from_vector(normal)
        pitch = self.compute_pitch_from_vector(normal)
        
        # 4. 判断是否贴合地面/天花板
        on_ground_ceiling = self.is_on_ground_or_ceiling(pitch)
        
        # 5. 计算最终theta角度
        if on_ground_ceiling:
            # 地面/天花板：直接使用alpha
            theta_deg = alpha_deg
            # print(f"  地面/天花板情况，直接使用alpha: {alpha_deg}°")
        else:
            # 墙面/空中：需要修正
            # print(f"  墙面/空中情况，使用修正公式:")
            # print(f"    法向量方位角θ_n: {azimuth_n:.2f}°")
            # print(f"    2D方向α: {alpha_deg}°")
            theta_deg = self.compute_theta_simple(azimuth_n, alpha_deg)
            # print(f"    修正后θ: {theta_deg:.2f}°")
        
        # 准备返回结果
        estimate_result = {
            'normal': normal.tolist(),
            'azimuth_n': float(azimuth_n),
            'pitch': float(pitch),
            'on_ground_ceiling': on_ground_ceiling,
            'theta_deg': float(theta_deg),
            'alpha_deg': float(alpha_deg),
            'points_3d': points_3d_corners.tolist(),  # 添加调试信息
            'points_2d':corners_2d.tolist()
        }
    
        return estimate_result

    def pixel_to_ray_direction(self, x, y, W, H, fov_x_deg):
        fx = W / (2.0 * np.tan(np.radians(fov_x_deg) / 2.0))
        fy = fx
        cx = W / 2.0
        cy = H / 2.0
        
        # x和y可以是标量，也可以是numpy数组
        X = (x - cx) / fx
        Y = -(y - cy) / fy
        
        # 如果x,y是数组，需要构建一个(N,3)的数组
        if isinstance(x, np.ndarray):
            # np.ones_like(X) 创建一个和X形状相同，值全为1的数组
            # np.stack 将三个数组按最后一维堆叠起来
            return np.stack([X, Y, np.ones_like(X)], axis=-1)
        else:
            return np.array([X, Y, 1.0], dtype=np.float64)

    
    def sample_points_and_get_3d_vectorized(self,
        corners_2d: np.ndarray,
        depth_image: np.ndarray,
        image_width: int,
        image_height: int,
        fov_x_deg: float,
        sample_step: int = 5
    ) -> Optional[np.ndarray]:
        """
        【向量化版本】在2D角点定义的矩形内采样，并将有效点转换为3D点云。

        参数:
            corners_2d: 4x2的2D角点数组。
            depth_image: 深度图。
            image_width, image_height: 图像尺寸。
            fov_x_deg: 水平视场角。
            sample_step: 采样步长（像素）。

        返回:
            Nx3的3D点云数组，如果有效点太少则返回None。
        """
        # 1. 找到2D矩形框的包围盒 (bounding box)
        x_min, y_min = np.min(corners_2d, axis=0).astype(int)
        x_max, y_max = np.max(corners_2d, axis=0).astype(int)

        # 确保包围盒在图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width - 1, x_max)
        y_max = min(image_height - 1, y_max)

        if x_min >= x_max or y_min >= y_max:
            return None

        # 2. 生成采样网格坐标
        # np.mgrid 创建一个密集的坐标网格，[start:end:step]
        # y_coords 和 x_coords 将是2D矩阵
        y_coords, x_coords = np.mgrid[y_min:y_max+1:sample_step, x_min:x_max+1:sample_step]

        # 3. 提取对应的深度值
        # 使用高级索引一次性获取所有采样点的深度
        sampled_depths = depth_image[y_coords, x_coords]

        # 4. 创建有效深度的掩码
        # 假设深度单位是米，0.1米（10cm）作为最小有效距离
        valid_mask = sampled_depths > 0.1
        
        # 获取所有有效点的坐标和深度
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_depths = sampled_depths[valid_mask]

        # 5. 检查是否有足够的点用于拟合
        num_valid_points = len(valid_depths)
        if num_valid_points < 9:  # 至少需要10个点才认为采样有效
            print(f"  采样点过少 ({num_valid_points})，回退。")
            return None

        # 6. 将所有有效点一次性转换为3D坐标
        # 将 valid_x 和 valid_y 转换为浮点数
        valid_x_f = valid_x.astype(np.float64)
        valid_y_f = valid_y.astype(np.float64)
        
        # a. 一次性计算所有射线的方向
        # pixel_to_ray_direction 需要能处理数组输入
        rays = self.pixel_to_ray_direction(valid_x_f, valid_y_f, image_width, image_height, fov_x_deg)
        
        # b. 使用广播机制进行乘法
        # rays 的形状是 (N, 3)，valid_depths 的形状是 (N,)
        # 我们需要将 depths 变成 (N, 1) 才能正确广播
        # points_3d.shape 将是 (N, 3)
        points_3d = rays * valid_depths[:, np.newaxis]

        return points_3d
    
    def compute_plane_normal(self, points_3d: np.ndarray) -> np.ndarray:
        """
        【已升级】通过N个3D点计算平面法向量，使用SVD拟合平面。
        
        参数:
            points_3d: Nx3的3D点数组 (N>=3)
            
        返回:
            单位法向量，指向相机原点。
        """
        if points_3d.shape[0] < 3:
            raise ValueError(f"至少需要3个点来拟合平面，但只得到 {points_3d.shape[0]} 个。")
        
        # 函数体保持不变
        centroid = np.mean(points_3d, axis=0)
        A = points_3d - centroid
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        normal = Vt[-1]
        
        if np.dot(normal, centroid) > 0:
            normal = -normal

    #     # 5. 返回归一化后的单位法向量
    #     norm_val = np.linalg.norm(normal)

    #     if norm_val < 1e-12:
    #         return normal
    #     return normal / norm_val

        return self.normalize(normal)
        
    def compute_azimuth_from_vector(self, v: np.ndarray) -> float:
        """
        计算向量在x-z平面上的投影的方位角
        
        坐标系: X-right, Y-up, Z-forward
        方位角范围: (-180, 180]度
            0°: 正前方 (Z正方向)
            90°: 正右方 (X正方向)
            -90°: 正左方 (X负方向)
            180°: 正后方 (Z负方向)
        
        参数:
            v: 输入向量
            
        返回:
            方位角（度）
        """
        # 投影到x-z平面
        proj_x = v[0]
        proj_z = v[2]
        
        # 计算与Z轴正方向的夹角
        azimuth_rad = np.arctan2(proj_x, proj_z)
        azimuth_deg = np.degrees(azimuth_rad)
        
        return self.wrap_to_180(azimuth_deg)

    def compute_pitch_from_vector(self, v: np.ndarray) -> float:
        
        
        """
        计算向量与y轴的夹角（俯仰角）
        
        参数:
            v: 输入向量
            
        返回:
            俯仰角（度），范围[-90, 90]
        """
        # 计算与水平面（x-z平面）的夹角
        horizontal_norm = np.sqrt(v[0]**2 + v[2]**2)
        if horizontal_norm < 1e-12:
            # 向量垂直向上或向下
            return 90.0 if v[1] > 0 else -90.0
        
        pitch_rad = np.arctan2(v[1], horizontal_norm)
        pitch_deg = np.degrees(pitch_rad)

        # 确保在[-90, 90]范围内
        pitch_deg = np.clip(pitch_deg, -90.0, 90.0)
        
        return pitch_deg

    def is_on_ground_or_ceiling(self, pitch_deg: float) -> bool:
        """
        判断是否贴合地面或天花板
        
        参数:
            pitch_deg: 俯仰角
            
        返回:
            True如果贴合地面或天花板，否则False
        """
        # 使用更严格的阈值，只有当法向量几乎垂直时（与垂直方向夹角小于10°）
        # 才认为是地面/天花板
        return abs(pitch_deg) >45
        # return abs(abs(pitch_deg) - 90) < 10

    def wrap_to_180(self, deg: float) -> float:
        """
        将角度包装到(-180, 180]范围
        
        参数:
            deg: 输入角度
            
        返回:
            包装后的角度
        """
        a = (deg + 180.0) % 360.0 - 180.0
        if np.isclose(a, -180.0):
            return 180.0
        return a

    # 归一化向量
    def normalize(self, v: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        参数:
            v: 输入向量
            
        返回:
            归一化后的向量
        """
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
        
    def compute_theta_simple(self, theta_n, theta_a):
        """
        分段实现，更直观
        """
        if theta_a == 90 or theta_a == -90:
            return (theta_n - theta_a + 180) % 360 - 180
        elif theta_a == 0:
            if theta_n < 0:
                return 180 + theta_n
            else:
                return theta_n - 180
        elif theta_a == 180:
            return theta_n
        else:
            raise ValueError(f"不支持的theta_a值: {theta_a}")
        
    def convert_to_serializable(self, obj):
        """将 NumPy 数据类型转换为 Python 原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

