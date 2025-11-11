# Inference backend

推理后端服务，由FastAPI构建

# 构建支持 GStreamer 后端的 opencv-python-headless
docker build --output type=local,dest=./wheel --target=exporter -f Dockerfile.opencv .