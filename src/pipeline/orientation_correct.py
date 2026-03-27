"""
朝向修正模块。

这个文件承担的是整条 pipeline 的最后一步：
1. 读取 Qwen3-VL 给出的 2D 箭头方向 `alpha`；
2. 结合 Depth Anything 输出的深度图，把箭头框从 2D 提升到 3D；
3. 对箭头所在平面做拟合，求出该平面的法向量；
4. 根据平面朝向，判断箭头更可能贴在地面/天花板，还是贴在墙面；
5. 最终把 2D 方向修正为 3D 世界中的水平朝向 `theta_deg`。

可以把它理解成：
"前面的模块告诉我箭头在图片里朝哪边，这个模块进一步判断它在真实空间里到底朝哪边。"
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
    """
    朝向修正器。

    输入:
    - 深度图 `depth_map`
    - 箭头框 `arrow_box`
    - 多模态模型预测得到的 2D 箭头方向 `alpha`

    输出:
    - 箭头所在平面的法向量
    - 法向量的方位角 / 俯仰角
    - 是否属于地面 / 天花板场景
    - 修正后的 3D 朝向 `theta_deg`
    """
    
    def __init__(self):
        """
        初始化后处理器
        
        Args:
            config: 后处理配置字典
        """
        logger.info(f"Correcter initialized successfully!")
    
    def process(self, image_path, image, depth_map, items, output_correct_json_folder_str):
        """
        处理单张图像中所有箭头目标的朝向修正。

        Args:
            image_path: 当前原始图像路径。
            image: RGB 图像的 numpy 数组，主要用于获取宽高。
            depth_map: 深度图，形状为 (H, W)。
            items: 上游整理好的箭头条目列表。每个条目通常包含：
                - image_path
                - depth_path
                - arrow_box
                - alpha
                - theta
                - fov_x_deg（可选）
            output_correct_json_folder_str: 修正结果输出目录。

        Returns:
            当前实现不显式 return，而是把结果写入 json 文件。

        说明:
            这里是该类的总控函数。它逐个处理 `items` 里的箭头，
            对每个箭头调用：
            - `load_depth_from_box` 读取角点深度
            - `estimate_arrow_orientation` 推断 3D 朝向
            最后统一保存到 `_correct.json`。
        """

        H, W = image.shape[:2]
        correct_results = []
        for idx, entry in enumerate(items):
            try:
                # 获取必要参数
                image_path = entry['image_path']
                depth_path = entry['depth_path']
                arrow_box = entry['arrow_box']
                # 这里预留了“缩框”能力，意图是避免框边界处的噪声深度影响拟合。
                # 当前参数 N=1，实际不会缩小，相当于沿用原框。
                arrow_box, _ = self.shrink_bounding_box(arrow_box, M=10, N=1)
                alpha_deg = entry['alpha']
                theta_gt = entry['theta']
                fov_x_deg = entry.get('fov_x_deg', 90.0)
                
                # 从深度图中取出箭头框角点对应的深度，形成 2D+depth 的结构。
                corners_with_depth = self.load_depth_from_box(depth_map, arrow_box)
                
                # 把 2D 框和深度信息提升到 3D，并据此推断真实朝向。
                correct_result = self.estimate_arrow_orientation(
                    corners=corners_with_depth,
                    alpha_deg=alpha_deg,
                    image_width=W,
                    image_height=H,
                    depth_map=depth_map,
                    fov_x_deg=fov_x_deg
                )
                
                # 为结果补充一些追踪字段，便于后续分析与排错。
                correct_result['image_path'] = image_path
                correct_result['depth_path'] = depth_path
                correct_result['image_width'] = W
                correct_result['image_height'] = H
                correct_result['fov_x_deg'] = fov_x_deg
                correct_result['arrow_box'] = entry.get('arrow_box')
                correct_result['arrow_box_4pts'] = entry.get('arrow_box_4pts')
                correct_result['sign_box'] = entry.get('sign_box')
                correct_result['sign_text'] = entry.get('sign_text', '')
                
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
        
        # 收敛为统一输出结构，后面再做一次 numpy 类型转换，确保可 json 序列化。
        correct_data = {
            'correct_results': correct_results
        }
        
        correct_data_serializable = self.convert_to_serializable(correct_data)
        
        # 以当前图像名命名输出文件。
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        correct_info = f"{base_name}_correct.json"
        correct_info_file_path = os.path.join(output_correct_json_folder_str,correct_info)
        with open(correct_info_file_path, 'w', encoding='utf-8') as f:
            json.dump(correct_data_serializable, f, indent=2, ensure_ascii=False)
        print(f"角度修正已完成, 修正结果已保存: {correct_info_file_path}")
        return correct_data_serializable
    
    
    def shrink_bounding_box(self,
        corners: List[List[float]], 
        M: float, 
        N: float
    ) -> Tuple[List[List[float]], bool]:
        """
        按中心点把边框缩小为原来的 1/N。

        这个函数的目的，是在箭头框较大时，只保留更靠近中心的区域，
        减少边缘深度噪声、遮挡或背景像素对后续平面拟合的影响。
        但如果框本身太小，就直接返回原框，避免“越缩越不可用”。

        参数:
            corners: 边框角点，2个点（对角点）或4个点
            M: 系数，用于计算最小允许边长（最小边长 > M × N）
            N: 缩小倍数
            
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
        
        # 以边框中心为基准，把每个点向中心收缩。
        # 这个公式是一个标准的“围绕中心缩放点坐标”的写法。
        center = np.mean(corners_array, axis=0)
        shrunk_corners = center + (corners_array - center) / N
        result = shrunk_corners.tolist()
        
        return result, True
    
    def load_depth_from_box(self, depth_map, box):
        """
        从深度图中读取箭头框角点的深度值。

        这个函数做的事情很直接：
        - 如果输入是两个对角点，就先补成四个角点；
        - 对每个角点，到深度图上取对应像素值；
        - 返回形如 `{'x': ..., 'y': ..., 'depth': ...}` 的列表。

        后面 `estimate_arrow_orientation` 会用这些角点，把框还原到 3D 空间。
        
        参数:
            depth_map: 深度图像
            box: 角点列表，每个元素是[x, y]坐标
            
        返回:
            更新了深度值的角点列表
        """
        # 读取深度图像        
        H, W = depth_map.shape
        
        # 兼容两种输入格式：
        # 1. 两个对角点
        # 2. 已经给好的四个角点
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
            
            # 深度图索引必须是整数，因此先 round 再 clip。
            # clip 是为了避免框越界时直接索引报错。
            x = np.clip(x, 0, W-1)
            y = np.clip(y, 0, H-1)
            
            # 注意这里取的是“角点位置”的深度，不是整个框的平均深度。
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
        估计箭头在三维空间中的真实水平朝向。

        这是整个文件最核心的函数，主思路是：
        1. 用深度把 2D 角点投回 3D；
        2. 在箭头框内部采样更多深度点，提高平面拟合稳定性；
        3. 用这些 3D 点拟合箭头所在平面，得到法向量；
        4. 从法向量计算该平面的朝向；
        5. 结合 2D 箭头方向 `alpha_deg`，得到修正后的 3D 朝向 `theta_deg`。

        参数:
            corners: 4个角点的列表，每个元素是包含'x','y','depth'的字典
            alpha_deg: 箭头2D方向角度
            image_width: 图像宽度
            image_height: 图像高度
            fov_x_deg: 水平视场角
            
        返回:
            包含结果信息的字典
        """
        # 第一步：把每个 2D 角点转换为 3D 点。
        # 做法是：
        # 像素点 -> 相机射线方向 -> 乘以深度 -> 相机坐标系下的 3D 点
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
        
        # 第二步：不仅用 4 个角点，还尝试在箭头框内部采样更多点。
        # 这样做的好处是，平面拟合会比只用 4 个角点稳定很多。
        points_3d_sampled = None
        points_3d_sampled = self.sample_points_and_get_3d_vectorized(
            corners_2d, depth_map, image_width, image_height, fov_x_deg, sample_step=5
        )

        # 优先使用采样得到的大量点；如果采样失败，再退回到四个角点。
        if points_3d_sampled is not None:
            points_3d_for_fit = points_3d_sampled
        else:
            points_3d_for_fit = points_3d_corners
        
        # 调试输出点坐标
        # print(f"3D点深度:")
        # print(f"corners {corners}")
        # print(f"3D点坐标:")
        # for i, pt in enumerate(points_3d_corners):
        #     print(f"  P{i}: {pt}")
        
        # 第三步：拟合箭头所在平面，得到法向量。
        normal = self.compute_plane_normal(points_3d_for_fit)
        
        # 第四步：把法向量分解成“水平朝向”和“上下倾斜程度”。
        azimuth_n = self.compute_azimuth_from_vector(normal)
        pitch = self.compute_pitch_from_vector(normal)
        
        # 第五步：根据 pitch 判断这个平面更像地面/天花板还是墙面。
        on_ground_ceiling = self.is_on_ground_or_ceiling(pitch)
        
        # 第六步：得到最终箭头朝向 theta。
        #
        # 直觉上：
        # - 如果箭头贴在地面/天花板，图像中的方向和真实水平朝向基本一致；
        # - 如果箭头贴在墙面，2D 方向会受到墙面姿态影响，需要结合平面法向量修正。
        if on_ground_ceiling:
            theta_deg = alpha_deg
        else:
            theta_deg = self.compute_theta_simple(azimuth_n, alpha_deg)
        
        # 返回值里保留了中间量，便于你后续检查每一步是否合理。
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
        """
        把像素坐标转换为相机坐标系下的“射线方向”。

        这里用的是一个非常常见的 pinhole camera（小孔成像）模型近似：
        - 图像中心 `(cx, cy)` 视作主点；
        - 由水平视场角 `fov_x_deg` 反推出焦距 `fx`；
        - 再把像素坐标归一化为相机坐标系中的方向向量 `[X, Y, 1]`。

        注意:
        - 这里只返回方向，不包含真实距离；
        - 真实 3D 点坐标需要再乘以深度值。
        """
        fx = W / (2.0 * np.tan(np.radians(fov_x_deg) / 2.0))
        fy = fx
        cx = W / 2.0
        cy = H / 2.0
        
        # X 向右为正，Y 向上为正，所以图像坐标里的 y 需要取负号。
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

        这一步的意义很重要：
        只靠 4 个角点做平面拟合，容易受噪声影响；
        如果能在框内部多取一些点，SVD 拟合出来的平面会稳定很多。

        “向量化”意味着这里尽量使用 numpy 批量计算，而不是 Python for 循环，
        所以速度和可扩展性都更好。
        """
        # 1. 先求 2D 框的轴对齐包围盒。
        # 这里默认箭头框基本接近矩形，直接在包围盒内采样。
        x_min, y_min = np.min(corners_2d, axis=0).astype(int)
        x_max, y_max = np.max(corners_2d, axis=0).astype(int)

        # 确保包围盒在图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width - 1, x_max)
        y_max = min(image_height - 1, y_max)

        if x_min >= x_max or y_min >= y_max:
            return None

        # 2. 在包围盒内按固定步长构造规则采样网格。
        # np.mgrid 创建一个密集的坐标网格，[start:end:step]
        # y_coords 和 x_coords 将是2D矩阵
        y_coords, x_coords = np.mgrid[y_min:y_max+1:sample_step, x_min:x_max+1:sample_step]

        # 3. 提取对应的深度值
        # 使用高级索引一次性获取所有采样点的深度
        sampled_depths = depth_image[y_coords, x_coords]

        # 4. 过滤掉无效深度。
        # 这里用 > 0.1 作为经验阈值，默认深度单位为米。
        valid_mask = sampled_depths > 0.1
        
        # 获取所有有效点的坐标和深度
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_depths = sampled_depths[valid_mask]

        # 5. 点太少就不做拟合，直接让上层回退到角点方案。
        num_valid_points = len(valid_depths)
        if num_valid_points < 9:  # 至少需要10个点才认为采样有效
            print(f"  采样点过少 ({num_valid_points})，回退。")
            return None

        # 6. 批量完成“像素 -> 射线 -> 3D 点”的转换。
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
        通过多个 3D 点拟合平面，并计算其法向量。

        参数:
            points_3d: Nx3的3D点数组 (N>=3)
            
        返回:
            单位法向量，指向相机原点。

        核心原理:
        - 先对点云去中心化；
        - 再做 SVD 分解；
        - 最小奇异值对应的方向，就是“最垂直于该平面”的方向，也就是法向量。

        额外处理:
        - 法向量有正反两个方向，这里用 `np.dot(normal, centroid)` 判断朝向；
        - 如果它背离相机原点，就翻转过来，让结果尽量保持一致。
        """
        if points_3d.shape[0] < 3:
            raise ValueError(f"至少需要3个点来拟合平面，但只得到 {points_3d.shape[0]} 个。")
        
        # 质心可以理解为这堆点的“中心位置”。
        centroid = np.mean(points_3d, axis=0)

        # 去中心化之后，SVD 更容易提取点云的主方向结构。
        A = points_3d - centroid
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        normal = Vt[-1]
        
        # 统一法向量朝向：尽量让它指向相机。
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
        计算向量在 x-z 平面投影后的方位角。
        
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
        # 忽略竖直分量，只保留水平面内的朝向。
        proj_x = v[0]
        proj_z = v[2]
        
        # arctan2(x, z) 的结果可直接理解为“相对正前方的水平转角”。
        azimuth_rad = np.arctan2(proj_x, proj_z)
        azimuth_deg = np.degrees(azimuth_rad)
        
        return self.wrap_to_180(azimuth_deg)

    def compute_pitch_from_vector(self, v: np.ndarray) -> float:
        """
        计算向量相对水平面的俯仰角。

        这里虽然注释里提到了 y 轴，但实际计算的是：
        “向量与水平面（x-z 平面）的夹角”。
        所以：
        - 向上时接近 +90°
        - 向下时接近 -90°
        - 水平时接近 0°
        
        参数:
            v: 输入向量
            
        返回:
            俯仰角（度），范围[-90, 90]
        """
        # 先看它在水平面上的投影长度。
        horizontal_norm = np.sqrt(v[0]**2 + v[2]**2)
        if horizontal_norm < 1e-12:
            # 水平分量几乎为 0，说明向量几乎竖直。
            return 90.0 if v[1] > 0 else -90.0
        
        pitch_rad = np.arctan2(v[1], horizontal_norm)
        pitch_deg = np.degrees(pitch_rad)

        # 确保在[-90, 90]范围内
        pitch_deg = np.clip(pitch_deg, -90.0, 90.0)
        
        return pitch_deg

    def is_on_ground_or_ceiling(self, pitch_deg: float) -> bool:
        """
        判断当前平面是否更像地面 / 天花板。

        参数:
            pitch_deg: 俯仰角
            
        返回:
            True如果贴合地面或天花板，否则False

        直觉解释:
        - 若法向量 pitch 很大，说明法向量接近竖直方向；
        - 而地面 / 天花板的法向量正是竖直的；
        - 因此可以据此把“水平面”与“竖直墙面”分开。

        当前阈值是 45°，属于比较宽松的经验规则。
        """
        # 注意：注释里说“更严格”，但实际阈值 45° 并不算特别严格。
        # 如果你后面发现误判较多，这里会是一个重点调参位置。
        return abs(pitch_deg) >45
        # return abs(abs(pitch_deg) - 90) < 10

    def wrap_to_180(self, deg: float) -> float:
        """
        把任意角度归一化到 (-180, 180] 范围内。

        例如:
        - 190 -> -170
        - -190 -> 170
        - 360 -> 0
        
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
        把向量归一化为单位向量。

        单位向量只保留方向，不保留长度。
        在本文件里，法向量最终只关心方向，因此通常需要做归一化。
        
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
        根据平面法向量方位角 `theta_n` 和 2D 箭头方向 `theta_a`，
        推出最终 3D 水平朝向 `theta`。

        这里使用的是一个“分段规则”而不是统一公式，
        前提是假设 Qwen3-VL 输出的箭头方向只会是四个离散值：
        - 上 -> 0
        - 右 -> 90
        - 左 -> -90
        - 下 -> 180

        换句话说，这个函数不是一个通用连续角度映射，
        而是针对本项目当前标签体系写的业务规则。
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
        """
        将 NumPy 数据类型递归转换为 Python 原生类型。

        这是为了让结果能够安全地被 `json.dump` 写出。
        否则像 `np.float32`、`np.ndarray` 这类对象会直接报序列化错误。
        """
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
