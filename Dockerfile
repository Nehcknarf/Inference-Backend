FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

ENV DEBIAN_FRONTEND=noninteractive UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    ocl-icd-libopencl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Intel >= Gen12
#RUN wget \
#    https://github.com/intel/intel-graphics-compiler/releases/download/v2.22.2/intel-igc-core-2_2.22.2+20121_amd64.deb \
#    https://github.com/intel/intel-graphics-compiler/releases/download/v2.22.2/intel-igc-opencl-2_2.22.2+20121_amd64.deb \
#    https://github.com/intel/compute-runtime/releases/download/25.44.36015.5/intel-ocloc_25.44.36015.5-0_amd64.deb \
#    https://github.com/intel/compute-runtime/releases/download/25.44.36015.5/intel-opencl-icd_25.44.36015.5-0_amd64.deb \
#    https://github.com/intel/compute-runtime/releases/download/25.44.36015.5/libigdgmm12_22.8.2_amd64.deb \
#    https://github.com/intel/compute-runtime/releases/download/25.44.36015.5/libze-intel-gpu1_25.44.36015.5-0_amd64.deb \
#    && dpkg -i *.deb \
#    && rm *.deb

# Intel Gen8-11
RUN wget \
    https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb \
    https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb \
    https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-level-zero-gpu-legacy1_1.5.30872.36_amd64.deb \
    https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb \
    https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/libigdgmm12_22.5.0_amd64.deb \
    && dpkg -i *.deb \
    && rm *.deb

# Setup a non-root user
RUN groupadd --system --gid 1001 nonroot \
 && useradd --system --gid 1001 --uid 1001 --create-home nonroot

WORKDIR /app

COPY wheel/ /app/wheel/

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN chown -R nonroot:nonroot /app

USER nonroot

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "--host", "0.0.0.0"]
CMD ["src/main.py"]