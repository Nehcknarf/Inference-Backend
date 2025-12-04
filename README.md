# Inference backend

推理后端服务，由FastAPI构建

# 构建支持 GStreamer 后端的 opencv-python-headless
docker build --output type=local,dest=./wheel --target=exporter -f Dockerfile.opencv .

# 导出镜像
docker save recycle-bin-inference-server:latest | gzip > inference-server.tar.gz

# 导入镜像
docker load -i inference-server.tar.gz

# 打包校准工具
pyinstaller -F --clean --noconfirm --name roi_selector roi_selector.py