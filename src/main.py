import math
from pathlib import Path
import base64
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager

import cv2
import numpy as np
import openvino as ov
import openvino.properties.hint as hints
# from openvino import Core
import uvicorn
from fastapi import FastAPI, HTTPException, Request

# camera.py 应该和本文件在同一个目录下
from ipc import ThreadedVideoCapture
from models import DualStreamInferenceResponse
from roi_config import ROIProcessor, ROIConfigManager

# --- 1. 配置部分 ---

# print(cv2.getBuildInformation())
# print(Core().available_devices)

# 是否启用ROI（如果roi_config.json存在且配置了流，将自动使用）
ENABLE_ROI = True

# 双路RTSP视频流地址
RTSP_STREAM_URLS = [
    "rtsp://admin:@192.168.0.10:554",  # 第一路流
    "rtsp://admin:@192.168.0.11:554",  # 第二路流
]

# 类别名称
CLASS_NAMES = ["保留", "ASLAT", "RFTN", "FTN", "SFTN", "MF", "MHJ", "PTD"]

# 后处理置信度和NMS阈值
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- 2. 核心推理逻辑封装（改为异步推理） ---


class AsyncYoloInference:
    """封装了基于AsyncInferQueue的异步推理逻辑，支持多流独立推理"""

    def __init__(
        self,
        model_path="model/best.xml",
        device="AUTO",
        num_streams=2,
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
        self.colors = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

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

    def preprocess(self, image):
        """图像预处理 - 保持宽高比的缩放和填充"""
        original_height, original_width = image.shape[:2]
        ratio = min(self.W / original_width, self.H / original_height)
        new_unpad_w, new_unpad_h = (
            int(original_width * ratio),
            int(original_height * ratio),
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

    def postprocess(self, model_output, scale_ratio):
        """对模型输出进行后处理，返回结构化的检测结果"""
        detections = model_output.T
        inv_scale_ratio = 1.0 / scale_ratio

        box_params = detections[:, :4]
        angles_rad = detections[:, -1]
        class_scores = detections[:, 4:-1]

        class_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)

        valid_mask = max_scores > CONFIDENCE_THRESHOLD
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

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
                CONFIDENCE_THRESHOLD,
                NMS_THRESHOLD,
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
                        "class_name": CLASS_NAMES[valid_class_ids[i]],
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
        return results

    async def predict_async(self, image, stream_id=0):
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

        preprocessed_image, scale_ratio = self.preprocess(image)

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

    def calculate_quadrant_stats(self, results, image_width, image_height):
        """
        计算四象限统计：将图像分为四等分，统计每个象限中各类别的数量

        象限划分：
        +-------+-------+
        |   1   |   2   |  (左上 | 右上)
        +-------+-------+
        |   3   |   4   |  (左下 | 右下)
        +-------+-------+
        """
        mid_x = image_width / 2
        mid_y = image_height / 2

        # 初始化四个象限的统计字典
        quadrant_stats = {
            1: defaultdict(int),  # 左上
            2: defaultdict(int),  # 右上
            3: defaultdict(int),  # 左下
            4: defaultdict(int),  # 右下
        }

        for detection in results:
            box = detection["box"]
            cx, cy = box["cx"], box["cy"]
            # class_name = detection["class_name"]
            class_id = detection["class_id"]

            # 根据中心点坐标判断所属象限
            if cx < mid_x and cy < mid_y:
                quadrant = 1  # 左上
            elif cx >= mid_x and cy < mid_y:
                quadrant = 2  # 右上
            elif cx < mid_x and cy >= mid_y:
                quadrant = 3  # 左下
            else:
                quadrant = 4  # 右下

            quadrant_stats[quadrant][class_id] += 1

        # 转换为更友好的输出格式
        formatted_stats = {}
        for quad_num in [1, 2, 3, 4]:
            quad_name = {1: "q1", 2: "q2", 3: "q3", 4: "q4"}[quad_num]
            formatted_stats[quad_name] = dict(quadrant_stats[quad_num])

        return formatted_stats


# --- 3. FastAPI 应用设置 ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器：负责初始化模型、视频流和ROI处理器，
    并在应用关闭时清理资源。
    """
    print("正在启动资源...")
    try:
        # --- 将所有共享资源挂载到 app.state ---
        # 为每个流创建独立的推理队列
        num_streams = len(RTSP_STREAM_URLS)
        app.state.yolo_model = AsyncYoloInference(
            num_streams=num_streams, requests_per_stream=2
        )

        app.state.rtsp_readers = [ThreadedVideoCapture(url) for url in RTSP_STREAM_URLS]
        for reader in app.state.rtsp_readers:
            reader.start()

        app.state.roi_processors = []
        if ENABLE_ROI:
            roi_manager = ROIConfigManager()
            for url in RTSP_STREAM_URLS:
                app.state.roi_processors.append(ROIProcessor(url, roi_manager))
        else:
            app.state.roi_processors = [None] * len(RTSP_STREAM_URLS)
            print("ℹ️  ROI功能已禁用")

        print("资源启动完毕。")
        yield  # 应用在这里运行

    except Exception as e:
        print(f"初始化失败: {e}")
    finally:
        print("正在停止视频流读取...")
        if hasattr(app.state, "rtsp_readers"):  # 确保 rtsp_readers 确实被初始化了
            for reader in app.state.rtsp_readers:
                reader.stop()
        print("视频流已停止。")


app = FastAPI(title="Async Inference Service", lifespan=lifespan)


@app.post("/predict", response_model=DualStreamInferenceResponse)
async def predict_dual_stream(request: Request):
    """从双路RTSP流获取最新帧，异步推理，返回推理结果"""

    # --- 从 app.state 获取共享资源 ---
    yolo_model = request.app.state.yolo_model
    rtsp_readers = request.app.state.rtsp_readers
    roi_processors = request.app.state.roi_processors

    # 1. 从两路流获取最新帧
    ret1, frame1 = rtsp_readers[0].read(timeout=10)
    ret2, frame2 = rtsp_readers[1].read(timeout=10)

    if not ret1 or frame1 is None:
        raise HTTPException(status_code=503, detail="无法从第一路RTSP流读取帧。")
    if not ret2 or frame2 is None:
        raise HTTPException(status_code=503, detail="无法从第二路RTSP流读取帧。")

    # 保存原始帧用于后续绘制
    original_frame1 = frame1.copy()
    original_frame2 = frame2.copy()

    # 1.5 应用ROI裁剪（如果配置了）
    processing_frame1 = frame1
    processing_frame2 = frame2

    if roi_processors[0]:
        processing_frame1, _ = roi_processors[0].process_frame(frame1)
    if roi_processors[1]:
        processing_frame2, _ = roi_processors[1].process_frame(frame2)

    # 2. 并行执行异步推理（使用ROI裁剪后的帧）
    # 为每个流指定独立的推理队列ID
    results1_roi, results2_roi = await asyncio.gather(
        yolo_model.predict_async(processing_frame1, stream_id=0),
        yolo_model.predict_async(processing_frame2, stream_id=1),
    )

    # 2.5 坐标转换：将ROI空间的检测结果转换回原图空间
    # if roi_processors[0]:
    #     results1 = roi_processors[0].translate_detections(results1_roi)
    # else:
    #     results1 = results1_roi
    #
    # if roi_processors[1]:
    #     results2 = roi_processors[1].translate_detections(results2_roi)
    # else:
    #     results2 = results2_roi

    # 3. 计算四象限统计（基于处理帧，不是原图）
    if roi_processors[0]:
        quadrant_stats_1 = roi_processors[0].calculate_quadrant_stats(
            results1_roi, processing_frame1.shape
        )
    # else:
    #     h1, w1 = frame1.shape[:2]
    #     quadrant_stats_1 = yolo_model.calculate_quadrant_stats(results1, w1, h1)

    if roi_processors[1]:
        quadrant_stats_2 = roi_processors[1].calculate_quadrant_stats(
            results2_roi, processing_frame2.shape
        )
    # else:
    #     h2, w2 = frame2.shape[:2]
    #     quadrant_stats_2 = yolo_model.calculate_quadrant_stats(results2, w2, h2)

    # 4. 绘制检测框（在原图上绘制，使用转换后的坐标）
    image_with_boxes1 = yolo_model.draw_boxes(processing_frame1.copy(), results1_roi)
    image_with_boxes2 = yolo_model.draw_boxes(processing_frame2.copy(), results2_roi)

    # 5. 画面拼接（垂直拼接）
    # 确保两张图宽度一致
    h1, w1 = image_with_boxes1.shape[:2]
    h2, w2 = image_with_boxes2.shape[:2]
    if w1 != w2:
        target_w = min(w1, w2)
        if w1 > target_w:
            new_h1 = int(h1 * target_w / w1)
            image_with_boxes1 = cv2.resize(image_with_boxes1, (target_w, new_h1))
        else:
            new_h2 = int(h2 * target_w / w2)
            image_with_boxes2 = cv2.resize(image_with_boxes2, (target_w, new_h2))

    stitched_image = cv2.vconcat([image_with_boxes1, image_with_boxes2])

    # 5. 画面拼接（水平拼接）
    # 确保两张图高度一致
    # h1, w1 = image_with_boxes1.shape[:2]
    # h2, w2 = image_with_boxes2.shape[:2]
    # if h1 != h2:
    #     target_h = min(h1, h2)
    #     if h1 > target_h:
    #         new_w1 = int(w1 * target_h / h1)
    #         image_with_boxes1 = cv2.resize(image_with_boxes1, (new_w1, target_h))
    #     else:
    #         new_w2 = int(w2 * target_h / h2)
    #         image_with_boxes2 = cv2.resize(image_with_boxes2, (new_w2, target_h))
    #
    # stitched_image = cv2.hconcat([image_with_boxes1, image_with_boxes2])

    _, buffer = cv2.imencode(".jpg", stitched_image)
    image_bytes = buffer.tobytes()
    # 将图片字节编码为 base64 字符串
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "stream_1": {"detections": results1_roi, "quadrant_stats": quadrant_stats_1},
        "stream_2": {"detections": results2_roi, "quadrant_stats": quadrant_stats_2},
        "annotated_image": image_base64,
    }


@app.get("/health")
async def health_check(request: Request):
    """健康检查端点"""
    rtsp_readers = request.app.state.rtsp_readers

    stream_status = []
    for idx, reader in enumerate(rtsp_readers):
        stream_status.append(
            {f"stream_{idx + 1}": "connected" if reader.isOpened() else "disconnected"}
        )

    return {"status": "healthy", "model": "loaded", "streams": stream_status}


# --- 4. 启动服务 ---
if __name__ == "__main__":
    # 使用 uvicorn 启动 web 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
