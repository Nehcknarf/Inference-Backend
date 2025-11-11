FROM python:3.13-slim-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m pip install --upgrade pip

RUN python -m pip wheel --no-binary opencv-python-headless opencv-python-headless==4.12.0.88

FROM scratch AS exporter

COPY --from=builder /build/*.whl /