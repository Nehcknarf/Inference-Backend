from pathlib import Path

import cv2
import tomlkit

# ---------------------------------------------------------
# 交互式ROI选择工具 (保留格式版)
# 依赖: pip install tomlkit opencv-python
# ---------------------------------------------------------

CONFIG_FILE = "config.toml"


class ROISelector:
    def __init__(self, stream_id, stream_config, full_config):
        self.stream_id = stream_id
        self.stream_config = stream_config
        self.full_config = (
            full_config  # 这是一个 tomlkit 的 Document 对象，包含格式信息
        )
        self.stream_url = stream_config.get("url")

        self.roi = None
        self.selecting = False
        self.start_point = None
        self.end_point = None

        # 初始化 ROI (如果配置文件中已有)
        if "roi_x" in stream_config:
            self.roi = (
                stream_config.get("roi_x", 0),
                stream_config.get("roi_y", 0),
                stream_config.get("roi_width", 0),
                stream_config.get("roi_height", 0),
            )

    def save_config(self):
        """保存配置到 TOML 文件，保留原始格式"""
        if self.roi:
            x, y, w, h = self.roi

            # 直接修改 tomlkit 对象，它会自动维护原有格式（缩进、数组换行等）
            # 注意：stream_id 在 toml 中通常是字符串键
            current_stream = self.full_config["stream"][self.stream_id]

            current_stream["roi_x"] = int(x)
            current_stream["roi_y"] = int(y)
            current_stream["roi_width"] = int(w)
            current_stream["roi_height"] = int(h)

            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    # tomlkit.dump 会保留原始的注释和排版
                    tomlkit.dump(self.full_config, f)
                print(f"ROI 配置已更新: Stream {self.stream_id}")
            except Exception as e:
                print(f"保存失败: {e}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                width = x2 - x1
                height = y2 - y1

                if width > 10 and height > 10:
                    self.roi = (x1, y1, width, height)
                    print(f"ROI Selected: {self.roi}")

    def draw_roi_overlay(self, frame):
        overlay = frame.copy()
        # 绘制正在选择的框
        if self.selecting and self.start_point and self.end_point:
            cv2.rectangle(overlay, self.start_point, self.end_point, (0, 255, 0), 2)

        # 绘制已确定的 ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 遮罩效果
            mask = frame.copy()
            cv2.rectangle(mask, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            overlay = cv2.addWeighted(overlay, 0.7, mask, 0.3, 0)

            label = f"Stream {self.stream_id} | {w}x{h}"
            cv2.putText(
                overlay,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return overlay

    def run(self):
        print(f"启动流配置: ID={self.stream_id} URL={self.stream_url}")

        # 原生 OpenCV 读取
        cap = cv2.VideoCapture(self.stream_url)

        if not cap.isOpened():
            print(f"无法打开流 {self.stream_url}")
            return

        window_name = f"Config: Stream {self.stream_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("按 's' 保存, 'q' 退出当前流, 'r' 重置")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 简单的防空转
                cv2.waitKey(100)
                continue

            display = self.draw_roi_overlay(frame)

            cv2.putText(
                display,
                "'S': Save | 'Q': Next/Quit | 'R': Reset",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if self.roi:
                    self.save_config()
                else:
                    print("请先选择区域")
            elif key == ord("r"):
                self.roi = None
                print("ROI 重置")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    if not Path(CONFIG_FILE).exists():
        print(f"配置文件 {CONFIG_FILE} 不存在")
        return

    try:
        # 使用 tomlkit 读取，它会保留注释、空行和多行数组格式
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = tomlkit.load(f)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return

    if "stream" not in full_config:
        print("配置中未找到 [stream] 部分")
        return

    streams = full_config["stream"]
    print(f"找到 {len(streams)} 个流配置")

    # tomlkit 的 items() 返回顺序与文件一致
    for stream_id, stream_info in streams.items():
        print(f"\n-----------------------------------")
        print(f"准备配置 Stream {stream_id}...")

        selector = ROISelector(stream_id, stream_info, full_config)
        selector.run()

    print("\n所有配置任务结束。")


if __name__ == "__main__":
    main()
