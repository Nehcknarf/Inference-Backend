"""
交互式ROI选择工具

使用方法:
    python src/roi_selector.py

功能:
    - 从RTSP流实时显示画面
    - 鼠标拖拽选择矩形ROI区域
    - 自动保存ROI配置到JSON文件
    - 支持多路流的独立ROI配置
"""

import os

os.add_dll_directory("C:/Program Files/gstreamer/1.0/msvc_x86_64/bin")

import cv2
import json
from pathlib import Path
from stream import ThreadedVideoCapture

# ROI配置文件路径
ROI_CONFIG_FILE = "roi_config.json"


class ROISelector:
    """交互式ROI选择工具"""

    def __init__(self, stream_url, stream_name="stream_1"):
        self.stream_url = stream_url
        self.stream_name = stream_name
        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None

        # 加载现有配置
        self.config = self.load_config()

    def load_config(self):
        """加载现有的ROI配置"""
        if Path(ROI_CONFIG_FILE).exists():
            with open(ROI_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_config(self):
        """保存ROI配置到文件"""
        with open(ROI_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"ROI配置已保存到: {ROI_CONFIG_FILE}")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始选择
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # 更新选择框
            if self.selecting:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # 完成选择
            self.selecting = False
            self.end_point = (x, y)

            # 计算ROI矩形
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            # 确保ROI有效
            width = x2 - x1
            height = y2 - y1
            if width > 50 and height > 50:  # 最小尺寸限制
                self.roi = (x1, y1, width, height)
                print(f"ROI已选择: x={x1}, y={y1}, width={width}, height={height}")
                print(f"   宽高比: {width / height:.2f}")
            else:
                print("ROI太小，请重新选择（最小尺寸: 50x50）")

    def draw_roi_overlay(self, frame):
        """在帧上绘制ROI覆盖层"""
        overlay = frame.copy()

        # 绘制当前正在选择的矩形
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(overlay, self.start_point, self.end_point, (0, 255, 0), 2)

        # 绘制已确认的ROI
        if self.roi:
            x, y, w, h = self.roi
            # 绘制ROI边框
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 绘制半透明遮罩（突出ROI区域）
            mask = frame.copy()
            cv2.rectangle(mask, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            overlay = cv2.addWeighted(overlay, 0.7, mask, 0.3, 0)

            # 显示ROI信息
            info_text = [
                f"ROI: {w}x{h}",
                f"Position: ({x}, {y})",
                f"Ratio: {w / h:.2f}",
                "Press 's' to save",
            ]
            y_offset = y - 10
            for i, text in enumerate(info_text):
                y_pos = y_offset - (len(info_text) - i) * 25
                cv2.putText(
                    overlay,
                    text,
                    (x, max(30, y_pos)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        return overlay

    def run(self):
        """运行ROI选择工具"""
        print("=" * 60)
        print(f"ROI选择工具 - {self.stream_name}")
        print("=" * 60)
        print(f"视频流: {self.stream_url}")
        print("\n操作说明:")
        print("  1. 鼠标左键拖拽选择矩形ROI区域")
        print("  2. 按 's' 键保存当前ROI配置")
        print("  3. 按 'r' 键重置ROI")
        print("  4. 按 'c' 键清除此流的ROI配置")
        print("  5. 按 'q' 键退出")
        print("=" * 60)

        # 创建视频捕获
        try:
            video_capture = ThreadedVideoCapture(self.stream_url)
            video_capture.start()
        except Exception as e:
            print(f"无法连接视频流: {e}")
            return

        # 创建窗口和鼠标回调
        window_name = f"ROI Selector - {self.stream_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # 加载已保存的ROI（如果存在）
        if self.stream_url in self.config:
            saved_roi = self.config[self.stream_url]
            self.roi = (
                saved_roi["x"],
                saved_roi["y"],
                saved_roi["width"],
                saved_roi["height"],
            )
            print(f"已加载保存的ROI配置: {self.roi}")

        try:
            while True:
                ret, frame = video_capture.read(timeout=5)
                if not ret or frame is None:
                    print("等待视频帧...")
                    continue

                self.current_frame = frame.copy()

                # 绘制ROI覆盖层
                display_frame = self.draw_roi_overlay(frame)

                # 显示提示信息
                help_text = (
                    "Drag to select ROI | 's':Save | 'r':Reset | 'c':Clear | 'q':Quit"
                )
                cv2.putText(
                    display_frame,
                    help_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(window_name, display_frame)

                # 键盘事件处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    # 退出
                    break

                elif key == ord("s"):
                    # 保存ROI配置
                    if self.roi:
                        x, y, w, h = self.roi
                        self.config[self.stream_url] = {
                            "stream_name": self.stream_name,
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "aspect_ratio": round(w / h, 2),
                        }
                        self.save_config()
                        print(f"ROI已保存: {self.stream_name}")
                    else:
                        print("请先选择ROI区域")

                elif key == ord("r"):
                    # 重置ROI
                    self.roi = None
                    self.start_point = None
                    self.end_point = None
                    print("ROI已重置，请重新选择")

                elif key == ord("c"):
                    # 清除此流的ROI配置
                    if self.stream_url in self.config:
                        del self.config[self.stream_url]
                        self.save_config()
                        print(f"已清除 {self.stream_name} 的ROI配置")
                    self.roi = None

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            video_capture.stop()
            cv2.destroyAllWindows()
            print("ROI选择工具已退出")


def select_roi_for_multiple_streams():
    """为多路流依次选择ROI"""
    # 从api_async.py读取配置的流地址
    streams = [
        ("rtsp://admin:@192.168.0.10:554", "stream_1"),
        ("rtsp://admin:@192.168.0.11:554", "stream_2"),
    ]

    print("\n" + "=" * 60)
    print("多路流ROI配置工具")
    print("=" * 60)
    print(f"将为 {len(streams)} 路流配置ROI")
    print("=" * 60)

    for stream_url, stream_name in streams:
        print(f"\n正在配置: {stream_name}")
        choice = input(f"是否为此流配置ROI? (y/n, 默认y): ").strip().lower()

        if choice == "n":
            print(f"跳过 {stream_name}")
            continue

        selector = ROISelector(stream_url, stream_name)
        selector.run()

    print("\n" + "=" * 60)
    print("所有流的ROI配置完成")
    print("=" * 60)
    print(f"配置文件: {ROI_CONFIG_FILE}")
    print("现在可以启动 api_async.py 使用ROI配置")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 单流模式
        stream_url = sys.argv[1]
        stream_name = sys.argv[2] if len(sys.argv) > 2 else "stream_1"
        selector = ROISelector(stream_url, stream_name)
        selector.run()
    else:
        # 多流模式
        select_roi_for_multiple_streams()
