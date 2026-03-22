# PhysTwin — UV-based image (no conda). GPU use requires NVIDIA Container Toolkit on the host.
# Build: docker build -t phystwin:uv .
# Run (GPU): docker run --gpus all -it --rm -v $PWD:/work -w /work phystwin:uv bash
#
# Vendored gaussian_splatting extensions (simple-knn, etc.) are not built in this minimal image;
# run `pip install` from `gaussian_splatting/submodules/...` on the host or extend this Dockerfile.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /PhysTwin
COPY . .

RUN uv python install 3.10 && uv sync --python 3.10 --frozen --extra dev

ENV PATH="/PhysTwin/.venv/bin:$PATH"
ENV PYTHONPATH=/PhysTwin/scripts/shims:/PhysTwin

CMD ["/bin/bash"]
