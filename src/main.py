import base64
import asyncio
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException, Request

# camera.py 应该和本文件在同一个目录下
from stream import ThreadedVideoCapture
from infer import AsyncYoloInference
from models import DualStreamInferenceResponse
from config import num_streams, stream_url, roi


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
        app.state.yolo_model = AsyncYoloInference(
            num_streams=num_streams, requests_per_stream=2
        )

        app.state.rtsp_readers = [ThreadedVideoCapture(url) for url in stream_url]
        for reader in app.state.rtsp_readers:
            reader.start()

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

    # 1. 从两路流获取最新帧
    ret1, frame1 = rtsp_readers[0].read(timeout=10)
    ret2, frame2 = rtsp_readers[1].read(timeout=10)

    if not ret1 or frame1 is None:
        raise HTTPException(status_code=503, detail="无法从第一路RTSP流读取帧。")
    if not ret2 or frame2 is None:
        raise HTTPException(status_code=503, detail="无法从第二路RTSP流读取帧。")

    # 2. 并行执行异步推理（使用ROI裁剪后的帧）
    # 为每个流指定独立的推理队列ID
    task1 = yolo_model.predict_async(frame1, roi[0], stream_id=0)
    task2 = yolo_model.predict_async(frame2, roi[1], stream_id=1)

    (
        (results1_roi, quadrant_stats_1),
        (results2_roi, quadrant_stats_2),
    ) = await asyncio.gather(task1, task2)

    # 绘制检测框（在原图上绘制，使用转换后的坐标）
    image_with_boxes1 = yolo_model.draw_boxes(frame1.copy(), results1_roi)
    image_with_boxes2 = yolo_model.draw_boxes(frame2.copy(), results2_roi)

    # 画面拼接（垂直拼接）
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

    # 画面拼接（水平拼接）
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
