import os

os.add_dll_directory("C:/Program Files/gstreamer/1.0/msvc_x86_64/bin")

import httpx
import cv2
import numpy as np
import base64
import sys


def fetch_and_display_image():
    """
    使用 httpx 获取图像数据并使用 OpenCV 显示。
    """
    API_URL = "http://localhost:8000/predict"

    # 定义显示窗口的最大尺寸
    # 你可以根据你的屏幕分辨率调整这些值
    MAX_WIDTH = 895
    MAX_HEIGHT = 1000

    try:
        # 1. 使用 httpx 同步发送 POST 请求
        # httpx 的 API 与 requests 非常相似
        print(f"正在向 {API_URL} 发送请求...")
        response = httpx.post(API_URL)

        # 2. 检查 HTTP 状态码，如果不是 2xx，则引发异常
        response.raise_for_status()
        print("请求成功，正在处理响应...")

        # 3. 解析 JSON 响应
        data = response.json()

        # 4. 提取 base64 编码的图像字符串
        if "annotated_image" not in data:
            print("错误：响应 JSON 中未找到 'annotated_image' 键。", file=sys.stderr)
            return

        image_data_base64 = data["annotated_image"]

        # 5. 解码 base64 字符串
        # 这会返回原始的图像文件字节（例如 JPEG 或 PNG 的字节流）
        try:
            image_data_bytes = base64.b64decode(image_data_base64)
        except base64.binascii.Error as e:
            print(f"错误：Base64 解码失败: {e}", file=sys.stderr)
            return

        # 6. 使用 OpenCV 从内存中解码图像
        # 6.1. 将原始字节缓冲区转换为 1D numpy 数组
        #      OpenCV 的 imdecode 需要这种格式
        nparr = np.frombuffer(image_data_bytes, np.uint8)

        # 6.2. 使用 cv2.imdecode 将 numpy 数组解码为图像
        #      cv2.IMREAD_COLOR 表示加载为3通道 BGR 彩色图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("错误：OpenCV 无法解码图像。", file=sys.stderr)
            print("数据可能已损坏，或者图像格式 OpenCV 不支持。", file=sys.stderr)
            return

        # 7. 检查图像大小并根据需要调整
        h, w = image.shape[:2]
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            # 计算缩放比例，保持纵横比
            ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            print(f"图像原始尺寸: ({w}x{h})。太大，正在缩放至 ({new_w}x{new_h})...")
            # 使用 INTER_AREA 插值法缩小图像，效果较好
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 8. 使用 OpenCV 显示图像
        print("解码成功，正在显示图像... 按任意键关闭窗口。")
        window_name = "Annotated Image (OpenCV)"
        cv2.imshow(window_name, image)

        # 等待用户按键，0 表示无限期等待
        cv2.waitKey(0)

    except httpx.RequestError as exc:
        # 处理连接错误、超时等
        print(f"发起 HTTP 请求时发生错误: {exc}", file=sys.stderr)
    except httpx.HTTPStatusError as exc:
        # 处理非 2xx 的 HTTP 响应
        print(
            f"HTTP 错误: {exc.response.status_code} - {exc.response.text}",
            file=sys.stderr,
        )
    except Exception as e:
        # 捕获其他潜在错误，例如 JSON 解析失败
        print(f"发生未知错误: {e}", file=sys.stderr)
    finally:
        # 确保 OpenCV 窗口在程序结束或出错时关闭
        cv2.destroyAllWindows()
        print("清理并关闭窗口。")


if __name__ == "__main__":
    fetch_and_display_image()
