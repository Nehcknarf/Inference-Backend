import time
import queue
import threading
import cv2


class ThreadedVideoCapture:
    def __init__(self, source_url, api_preference=cv2.CAP_GSTREAMER):
        self.source_url = source_url
        self.api_preference = api_preference
        self._cap = None
        self._thread = None
        self.queue = queue.Queue(maxsize=1)  # 始终只保留最新帧
        self._stop_event = (
            threading.Event()
        )  # 使用 Event 控制停止比 bool 标志更线程安全

    def _build_pipeline(self):
        """构建 GStreamer 字符串，独立出来方便修改"""
        return (
            f"rtspsrc location={self.source_url} latency=0 ! "
            "decodebin ! videoconvert ! "
            "appsink drop=true sync=false"
        )

    def _connect(self):
        """尝试建立连接"""
        pipeline = self._build_pipeline()
        print(f"[{self.source_url}] 连接中...")

        # 注意：如果不是 GStreamer，这里可能需要降级处理，或者由外部传入 pipeline
        cap = cv2.VideoCapture(pipeline, self.api_preference)

        if cap.isOpened():
            print(f"[{self.source_url}] 连接成功")
            return cap
        else:
            print(f"[{self.source_url}] 连接失败")
            return None

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            print(f"[{self.source_url}] 线程已启动")

    def _update_loop(self):
        reconnect_delay = 1

        while not self._stop_event.is_set():
            # 1. 确保连接存在
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._connect()
                if self._cap is None:
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)
                    continue
                reconnect_delay = 1  # 连接成功，重置延迟

            # 2. 读取帧
            try:
                ret, frame = self._cap.read()
                if not ret:
                    print(f"[{self.source_url}] 帧读取失败，触发重连")
                    self._release_cap()
                    continue

                # 3. 更新队列 (非阻塞写入，保证最新)
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()  # 丢弃旧帧
                    except queue.Empty:
                        pass
                self.queue.put(frame)

            except Exception as e:
                print(f"[{self.source_url}] 异常: {e}")
                self._release_cap()
                time.sleep(1)  # 发生异常时的简单冷却

    def read(self, timeout=1.0):
        try:
            return True, self.queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._release_cap()
        print(f"[{self.source_url}] 已停止")

    def _release_cap(self):
        if self._cap:
            self._cap.release()
        self._cap = None

    @property
    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()
