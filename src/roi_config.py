"""
ROI配置管理器

功能:
    - 加载和管理ROI配置
    - 提供ROI裁剪功能
    - 坐标转换（ROI空间 <-> 原图空间）
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

ROI_CONFIG_FILE = "src/roi_config.json"


class ROIConfigManager:
    """ROI配置管理器"""

    def __init__(self, config_file: str = ROI_CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """加载ROI配置文件"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print(f"已加载ROI配置: {len(config)} 个流")
                return config
            except Exception as e:
                print(f"加载ROI配置失败: {e}")
                return {}
        else:
            print(f"ROI配置文件不存在: {config_path}")
            return {}

    def get_roi(self, stream_url: str) -> Optional[Tuple[int, int, int, int]]:
        """
        获取指定流的ROI配置

        Args:
            stream_url: RTSP流地址

        Returns:
            (x, y, width, height) 或 None（如果未配置）
        """
        if stream_url in self.config:
            roi_data = self.config[stream_url]
            return (roi_data["x"], roi_data["y"], roi_data["width"], roi_data["height"])
        return None

    def get_roi_info(self, stream_url: str) -> Optional[Dict]:
        """
        获取指定流的完整ROI信息

        Args:
            stream_url: RTSP流地址

        Returns:
            包含ROI所有信息的字典，或None
        """
        return self.config.get(stream_url)

    def has_roi(self, stream_url: str) -> bool:
        """检查指定流是否配置了ROI"""
        return stream_url in self.config

    def list_all_rois(self) -> List[Dict]:
        """列出所有配置的ROI"""
        return [
            {"stream_url": url, **roi_data} for url, roi_data in self.config.items()
        ]

    @staticmethod
    def crop_roi(frame, roi: Tuple[int, int, int, int]):
        """
        裁剪ROI区域

        Args:
            frame: 原始帧
            roi: (x, y, width, height)

        Returns:
            裁剪后的帧
        """
        x, y, w, h = roi

        # 边界检查
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        return frame[y : y + h, x : x + w]

    @staticmethod
    def translate_detections_to_original(
        detections: List[Dict], roi: Tuple[int, int, int, int]
    ) -> List[Dict]:
        """
        将ROI空间的检测结果坐标转换到原图空间

        Args:
            detections: 检测结果列表（ROI空间坐标）
            roi: (x, y, width, height)

        Returns:
            转换后的检测结果列表（原图空间坐标）
        """
        roi_x, roi_y, _, _ = roi

        translated_detections = []
        for det in detections:
            translated_det = det.copy()
            box = translated_det["box"]

            # 平移中心点坐标
            box["cx"] += roi_x
            box["cy"] += roi_y

            translated_det["box"] = box
            translated_detections.append(translated_det)

        return translated_detections

    @staticmethod
    def calculate_quadrant_stats_with_roi(
        detections: List[Dict], roi: Tuple[int, int, int, int]
    ) -> Dict:
        """
        基于ROI区域计算四象限统计（在ROI坐标系中）

        Args:
            detections: 检测结果列表（ROI空间坐标）
            roi: (x, y, width, height)

        Returns:
            四象限统计字典
        """
        from collections import defaultdict

        _, _, roi_width, roi_height = roi
        mid_x = roi_width / 2
        mid_y = roi_height / 2

        quadrant_stats = {
            1: defaultdict(int),  # 左上
            2: defaultdict(int),  # 右上
            3: defaultdict(int),  # 左下
            4: defaultdict(int),  # 右下
        }

        for detection in detections:
            box = detection["box"]
            # 注意：这里的坐标是相对于ROI的，不需要减去roi_x和roi_y
            cx, cy = box["cx"], box["cy"]
            class_name = detection["class_name"]

            # 判断象限（在ROI坐标系中）
            if cx < mid_x and cy < mid_y:
                quadrant = 1
            elif cx >= mid_x and cy < mid_y:
                quadrant = 2
            elif cx < mid_x and cy >= mid_y:
                quadrant = 3
            else:
                quadrant = 4

            quadrant_stats[quadrant][class_name] += 1

        # 格式化输出
        formatted_stats = {}
        for quad_num in [1, 2, 3, 4]:
            quad_name = {1: "q1", 2: "q2", 3: "q3", 4: "q4"}[quad_num]
            formatted_stats[quad_name] = dict(quadrant_stats[quad_num])

        return formatted_stats


class ROIProcessor:
    """ROI处理器 - 便捷的封装类"""

    def __init__(
        self, stream_url: str, config_manager: Optional[ROIConfigManager] = None
    ):
        self.stream_url = stream_url
        self.config_manager = config_manager or ROIConfigManager()
        self.roi = self.config_manager.get_roi(stream_url)

        if self.roi:
            x, y, w, h = self.roi
            print(f"{stream_url} - ROI已启用: ({x}, {y}, {w}x{h})")
        else:
            print(f"{stream_url} - 未配置ROI，将使用全帧")

    def has_roi(self) -> bool:
        """是否配置了ROI"""
        return self.roi is not None

    def process_frame(self, frame):
        """
        处理帧：如果配置了ROI则裁剪，否则返回原帧

        Returns:
            (processed_frame, roi_used)
            - processed_frame: 处理后的帧
            - roi_used: 实际使用的ROI（如果未配置则为None）
        """
        if self.roi:
            cropped_frame = ROIConfigManager.crop_roi(frame, self.roi)
            return cropped_frame, self.roi
        else:
            return frame, None

    def translate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        将检测结果坐标从ROI空间转换到原图空间
        如果未使用ROI，则直接返回
        """
        if self.roi:
            return ROIConfigManager.translate_detections_to_original(
                detections, self.roi
            )
        else:
            return detections

    def calculate_quadrant_stats(
        self, detections: List[Dict], frame_shape: Tuple
    ) -> Dict:
        """
        计算四象限统计

        Args:
            detections: 检测结果（应该是ROI空间坐标）
            frame_shape: 处理帧的形状 (height, width)

        Returns:
            四象限统计字典
        """
        if self.roi:
            return ROIConfigManager.calculate_quadrant_stats_with_roi(
                detections, self.roi
            )
        else:
            # 未使用ROI，基于整个帧计算
            from collections import defaultdict

            height, width = frame_shape[:2]
            mid_x = width / 2
            mid_y = height / 2

            quadrant_stats = {
                1: defaultdict(int),
                2: defaultdict(int),
                3: defaultdict(int),
                4: defaultdict(int),
            }

            for detection in detections:
                box = detection["box"]
                cx, cy = box["cx"], box["cy"]
                class_name = detection["class_name"]

                if cx < mid_x and cy < mid_y:
                    quadrant = 1
                elif cx >= mid_x and cy < mid_y:
                    quadrant = 2
                elif cx < mid_x and cy >= mid_y:
                    quadrant = 3
                else:
                    quadrant = 4

                quadrant_stats[quadrant][class_name] += 1

            formatted_stats = {}
            for quad_num in [1, 2, 3, 4]:
                quad_name = {1: "q1", 2: "q2", 3: "q3", 4: "q4"}[quad_num]
                formatted_stats[quad_name] = dict(quadrant_stats[quad_num])

            return formatted_stats


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("ROI配置管理器测试")
    print("=" * 60)

    manager = ROIConfigManager()

    if manager.config:
        print("\n已配置的ROI:")
        for roi_info in manager.list_all_rois():
            print(f"\n流: {roi_info['stream_url']}")
            print(f"  名称: {roi_info['stream_name']}")
            print(f"  位置: ({roi_info['x']}, {roi_info['y']})")
            print(f"  尺寸: {roi_info['width']}x{roi_info['height']}")
            print(f"  宽高比: {roi_info['aspect_ratio']}")
    else:
        print("\n未找到ROI配置")
        print("请先运行 'python src/roi_selector.py' 配置ROI")
