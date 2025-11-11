import time
import queue
import threading

import cv2


class ThreadedVideoCapture:
    """
    一个健壮的、线程化的视频捕获类。

    该类在一个独立的后台线程中从视频源（如RTSP流）读取帧，
    并提供了自动重连机制，以应对网络中断等问题。
    它只在内部队列中保留最新的一帧，以避免画面延迟。
    """

    def __init__(self, source_url):
        self.source_url = source_url
        self._cap = None
        self._thread = None
        self.queue = queue.Queue(maxsize=1)
        self._is_running = False
        self._connect()

    def _connect(self):
        """尝试连接到视频源。"""
        print(f"正在尝试连接到视频流: {self.source_url}")

        pipeline = (
            f"rtspsrc location={self.source_url} latency=0 ! "
            "decodebin ! videoconvert ! "
            f"appsink drop=true sync=false"
        )
        # pipeline = (
        #     "mfvideosrc ! "
        #     "videoconvert ! appsink drop=true sync=false"
        # )
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            print(f"无法打开视频流: {self.source_url}")
            self._cap = None
        else:
            print(f"视频流连接成功: {self.source_url}")

    def start(self):
        """启动后台读取线程。"""
        if self._is_running:
            print(f"线程已经为 {self.source_url} 启动，请勿重复启动。")
            return self

        self._is_running = True
        self._thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name=f"VideoCapture-{self.source_url}",
        )
        self._thread.start()
        print(f"[{self.source_url}] 视频流读取线程已启动。")
        return self

    def _update_loop(self):
        """
        后台循环，负责读取帧和处理重连。
        """
        reconnect_delay = 1
        while self._is_running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    # 读取成功，重置重连延迟
                    reconnect_delay = 1
                    # 使用非阻塞方式放入队列，如果队列满了（说明上一帧还没被消费），就扔掉旧的，放入新的
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.queue.put(frame)
                else:
                    # 帧读取失败，可能连接已断开
                    print(f"从 {self.source_url} 读取帧失败，可能连接已断开。")
                    self._cap.release()
                    self._cap = None
                    # 短暂休眠，避免CPU空转
                    time.sleep(0.01)
            else:
                # 连接丢失，尝试重连
                print(f"流连接丢失，正在尝试重新连接: {self.source_url}")
                if self._cap:
                    self._cap.release()
                    self._cap = None

                # 带指数退避的重连逻辑
                self._connect()
                if self._cap is None:
                    print(f"重连失败，将在 {reconnect_delay} 秒后重试...")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)  # 指数退避，最长30秒
                else:
                    print(f"流重连成功: {self.source_url}")
                    reconnect_delay = 1  # 重连成功后，重置延迟

    def read(self, timeout=1.0):
        """
        从队列中获取最新的一帧。

        Args:
            timeout (float): 等待新帧的超时时间（秒）。

        Returns:
            tuple: (True, frame) 如果成功获取到帧。
                   (False, None) 如果超时或线程已停止。
        """
        if not self._is_running and self.queue.empty():
            return False, None
        try:
            return True, self.queue.get(timeout=timeout)
        except queue.Empty:
            print(f"从 {self.source_url} 读取帧超时。")
            return False, None

    def stop(self):
        """停止后台线程并释放资源。"""
        print(f"正在停止视频读取线程 [{self.source_url}]...")
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)  # 等待线程结束
        if self._cap:
            self._cap.release()
        self._cap = None
        print(f"视频读取线程 [{self.source_url}] 已停止。")

    def isOpened(self):
        """检查视频捕获是否已打开。"""
        return self._cap is not None and self._cap.isOpened()


if __name__ == "__main__":
    camera = ThreadedVideoCapture("rtsp://admin:@192.168.0.10:554")
    camera.start()
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
