import base64
import asyncio
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, HTTPException, Request, Response, status, Query

# camera.py 应该和本文件在同一个目录下
from stream import ThreadedVideoCapture
from infer import AsyncYoloInference, get_available_devices
from models import DualStreamInferenceResponse, HealthResponse
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


@app.post(
    "/predict",
    response_model=DualStreamInferenceResponse,
    response_model_exclude_none=True,
)
async def predict_dual_stream(
    request: Request,
    return_detections: bool = Query(False, description="是否返回详细的目标检测框信息"),
    return_image: bool = Query(True, description="是否返回绘制了检测框的Base64图片"),
):
    """
    从双路RTSP流获取最新帧，异步推理，返回推理结果。
    可以通过参数控制是否返回 detections 和 annotated_image。
    """

    # --- 从 app.state 获取共享资源 ---
    yolo_model = request.app.state.yolo_model
    rtsp_readers = request.app.state.rtsp_readers

    # 1. 从两路流获取最新帧
    ret1, frame1 = rtsp_readers[0].read(timeout=2)
    ret2, frame2 = rtsp_readers[1].read(timeout=2)

    if not ret1 or frame1 is None:
        raise HTTPException(status_code=503, detail="无法从第一路RTSP流读取帧。")
    if not ret2 or frame2 is None:
        raise HTTPException(status_code=503, detail="无法从第二路RTSP流读取帧。")

    crop_frame1 = yolo_model.crop(frame1, roi[0])
    crop_frame2 = yolo_model.crop(frame2, roi[1])
    # 2. 并行执行异步推理（使用ROI裁剪后的帧）
    # 为每个流指定独立的推理队列ID
    task1 = yolo_model.predict_async(crop_frame1, roi[0], stream_id=0)
    task2 = yolo_model.predict_async(crop_frame2, roi[1], stream_id=1)

    (
        (results_1, quadrant_stats_1),
        (results_2, quadrant_stats_2),
    ) = await asyncio.gather(task1, task2)

    # 3. 根据参数决定是否处理图像
    image_base64 = None
    if return_image:
        # 绘制检测框（在原图上绘制，使用转换后的坐标）
        image_with_boxes1 = yolo_model.draw_boxes(crop_frame1, results_1)
        image_with_boxes2 = yolo_model.draw_boxes(crop_frame2, results_2)

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

        _, buffer = cv2.imencode(".jpg", stitched_image)
        image_bytes = buffer.tobytes()
        # 将图片字节编码为 base64 字符串
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # 4. 根据参数决定是否返回检测详情
    # 如果 return_detections 为 False，则将 detections 字段设为 None
    out_results_1 = results_1 if return_detections else None
    out_results_2 = results_2 if return_detections else None

    return {
        "stream_1": {"detections": out_results_1, "quadrant_stats": quadrant_stats_1},
        "stream_2": {"detections": out_results_2, "quadrant_stats": quadrant_stats_2},
        "annotated_image": image_base64,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request, response: Response):
    """
    健康检查
    """
    rtsp_readers = request.app.state.rtsp_readers

    # 收集所有流的状态
    connected_count = 0

    for idx, reader in enumerate(rtsp_readers):
        is_connected = reader.is_opened  # 调用那个 property
        if is_connected:
            connected_count += 1

    # 核心业务逻辑判断：
    if connected_count == len(rtsp_readers):
        overall_status = "healthy"
        response.status_code = status.HTTP_200_OK
    else:
        # 如果所有流都断了，服务实质上无法工作，返回 503 Service Unavailable
        overall_status = "unhealthy"
        # 注意：即使返回 503，JSON body 依然会被返回，方便排查原因
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": overall_status,
        "active_streams": f"{connected_count}/{len(rtsp_readers)}",
        "available_infer_devices": get_available_devices(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
