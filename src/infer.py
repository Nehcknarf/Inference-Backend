import math
import asyncio
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import openvino as ov
import openvino.properties.hint as hints
# from openvino import Core
# print(Core().available_devices)

from config import confidence_threshold, nms_threshold, class_names, num_streams


class AsyncYoloInference:
    """封装了基于AsyncInferQueue的异步推理逻辑，支持多流独立推理"""

    def __init__(
        self,
        model_path="model/best.xml",
        device="AUTO",
        num_streams=num_streams,
        requests_per_stream=2,
    ):
        """
        初始化多流异步推理模型

        参数:
            model_path: 模型文件路径
            device: 推理设备 (GPU/CPU)
            num_streams: 支持的并发流数量
            requests_per_stream: 每个流的异步请求队列大小
        """
        # 为每个类别随机生成一个颜色
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(class_names), 3))

        # --- OpenVINO™ 模型初始化 ---
        core = ov.Core()
        ov_model = core.read_model(model_path)

        cache_path = Path("model/model_cache")
        cache_path.mkdir(exist_ok=True)
        config_dict = {
            "CACHE_DIR": str(cache_path),
            hints.performance_mode: hints.PerformanceMode.LATENCY,
        }

        self.compiled_model = core.compile_model(ov_model, device, config_dict)
        self.input_layer = self.compiled_model.input(0)
        _N, _C, self.H, self.W = self.input_layer.shape

        # 为每个流创建独立的异步推理队列
        self.num_streams = num_streams
        self.infer_queues = [
            ov.AsyncInferQueue(self.compiled_model, requests_per_stream)
            for _ in range(num_streams)
        ]

        print(
            f"✓ 已创建 {num_streams} 个独立推理队列，每个队列支持 {requests_per_stream} 个并发请求"
        )

    def preprocess(self, image, roi):
        """图像预处理 - 保持宽高比的缩放和填充"""
        original_height, original_width = image.shape[:2]
        roi_x, roi_y, roi_width, roi_height = roi
        # ROI 边界检查
        x = max(0, min(roi_x, original_width - 1))
        y = max(0, min(roi_y, original_height - 1))
        w = min(roi_width, original_width - roi_x)
        h = min(roi_height, original_height - roi_y)
        crop_image = image[y : y + h, x : x + w]
        crop_height, crop_width = crop_image.shape[:2]
        ratio = min(self.W / crop_width, self.H / crop_height)
        new_unpad_w, new_unpad_h = (
            int(crop_width * ratio),
            int(crop_height * ratio),
        )

        resized_image = cv2.resize(
            image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR
        )
        padded_image = np.full((self.H, self.W, 3), 114, dtype=np.uint8)
        padded_image[0:new_unpad_h, 0:new_unpad_w] = resized_image

        image_data = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32) / 255.0
        return image_data, ratio

    def postprocess(self, model_output, scale_ratio, roi):
        """对模型输出进行后处理，返回结构化的检测结果"""
        detections = model_output.T
        inv_scale_ratio = 1.0 / scale_ratio

        box_params = detections[:, :4]
        angles_rad = detections[:, -1]
        class_scores = detections[:, 4:-1]

        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        valid_mask = max_scores > confidence_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            empty_stats = self.calculate_quadrant_stats([], roi)
            return [], empty_stats

        valid_boxes = box_params[valid_indices]
        valid_angles = angles_rad[valid_indices]
        valid_scores = max_scores[valid_indices]
        valid_class_ids = class_ids[valid_indices]

        valid_boxes[:, 0:4] *= inv_scale_ratio

        angle_mask = (0.5 * math.pi <= valid_angles) & (valid_angles <= 0.75 * math.pi)
        valid_angles[angle_mask] -= math.pi
        valid_angles_deg = valid_angles * (180 / math.pi)

        boxes_for_nms = [
            ((b[0], b[1]), (b[2], b[3]), a)
            for b, a in zip(valid_boxes, valid_angles_deg)
        ]

        try:
            indices = cv2.dnn.NMSBoxesRotated(
                boxes_for_nms,
                valid_scores.tolist(),
                confidence_threshold,
                nms_threshold,
            )
        except Exception as e:
            print(f"NMSBoxesRotated 发生错误: {e}")
            indices = range(len(boxes_for_nms))

        results = []
        if len(indices) > 0:
            if (
                isinstance(indices, (list, tuple))
                and len(indices) > 0
                and isinstance(indices[0], (list, tuple))
            ):
                indices = [i[0] for i in indices]

            for i in indices:
                results.append(
                    {
                        "class_id": int(valid_class_ids[i]),
                        "class_name": class_names[valid_class_ids[i]],
                        "confidence": float(valid_scores[i]),
                        "box": {
                            "cx": float(valid_boxes[i][0]),
                            "cy": float(valid_boxes[i][1]),
                            "width": float(valid_boxes[i][2]),
                            "height": float(valid_boxes[i][3]),
                            "angle": float(valid_angles_deg[i]),
                        },
                    }
                )
        # 调用统计函数
        quadrant_stats = self.calculate_quadrant_stats(results, roi)
        # 返回检测结果和统计结果
        return results, quadrant_stats

    async def predict_async(self, image, roi, stream_id=0):
        """
        异步执行单帧预测流程

        参数:
            image: 输入图像
            stream_id: 流ID (0 到 num_streams-1)，用于选择对应的推理队列
        """
        if stream_id >= self.num_streams:
            raise ValueError(
                f"stream_id {stream_id} 超出范围，当前支持 {self.num_streams} 个流"
            )

        preprocessed_image, scale_ratio = self.preprocess(image, roi)

        # 使用指定流的推理队列（每个流独立，无需加锁）
        infer_queue = self.infer_queues[stream_id]

        # 在主线程中获取当前事件循环的引用
        loop = asyncio.get_running_loop()

        # 创建一个Future来等待推理完成
        future = loop.create_future()

        def callback(infer_request, user_data):
            """推理完成后的回调函数（在OpenVINO后台线程中执行）"""
            fut, s_ratio, event_loop = user_data
            output_tensor = infer_request.get_output_tensor(0).data[0]
            detections = self.postprocess(output_tensor, s_ratio)

            # 使用保存的事件循环引用，从后台线程安全地设置结果
            event_loop.call_soon_threadsafe(fut.set_result, detections)

        # 设置回调并启动异步推理（将事件循环引用传递给回调）
        infer_queue.set_callback(callback)
        infer_queue.start_async({0: preprocessed_image}, (future, scale_ratio, loop))

        # 等待推理完成
        result = await future
        return result

    def draw_boxes(self, image, results):
        """在图像上绘制旋转目标框"""
        for res in results:
            box = res["box"]
            rotated_box = (
                (box["cx"], box["cy"]),
                (box["width"], box["height"]),
                box["angle"],
            )

            points = cv2.boxPoints(rotated_box)
            points = np.intp(points)

            color = self.colors[res["class_id"]].tolist()
            cv2.drawContours(image, [points], 0, color=color, thickness=2)

            label = f"{res['class_name']}: {res['confidence']:.2f}"
            label_pos = (int(points[1][0]), int(points[1][1] - 10))
            cv2.putText(
                image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        return image

    def calculate_quadrant_stats(self, detections, roi):
        """
        计算四象限统计：将图像分为四等分，统计每个象限中各类别的数量

        象限划分：
        +-------+-------+
        |   1   |   2   |  (左上 | 右上)
        +-------+-------+
        |   3   |   4   |  (左下 | 右下)
        +-------+-------+
        """
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
