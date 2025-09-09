# Dockerfile for FramePack-eichi
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    build-essential \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip && pip install packaging wheel setuptools ninja && \
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 && \
pip install xformers==0.0.29.post3 --no-deps --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir  && \
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  && \
pip install triton==2.2.0 --no-cache-dir  && \
pip install -U "huggingface_hub[cli]" && \
pip install https://ghfast.top/github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
&& pip install sageattention pynvml "jinja2>=3.1.2" peft

# Clone original FramePack repository (needed for the base functionality)
RUN git clone https://ghfast.top/github.com/lllyasviel/FramePack.git /app/framepack \
&& git clone https://ghfast.top/github.com/tanbw/FramePack-eichi /tmp/framepack-eichi \
&& cp -rf /tmp/framepack-eichi/webui/* /app/framepack/
# Install dependencies
RUN pip install -r /app/framepack/requirements.txt && pip install gradio-client uvicorn fastapi python-multipart

WORKDIR /app/framepack
# Create a simple startup script with better error handling
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'cd /app/framepack' >> /app/start.sh && \
    echo 'echo "Starting api.py..."' >> /app/start.sh && \
    echo 'python api.py &' >> /app/start.sh && \
    echo 'SERVER_PID=$!' >> /app/start.sh && \
    echo 'sleep 5' >> /app/start.sh && \
    echo 'if [ -f "endframe_ichi.py" ]; then' >> /app/start.sh && \
    echo '  echo "Starting endframe_ichi.py..."' >> /app/start.sh && \
    echo '  python endframe_ichi.py --server 0.0.0.0 --port 7862 --lang zh-tw' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    chmod +x /app/start.sh
COPY ./webui/api.py /app/framepack/api.py

# Command to run when container starts
ENTRYPOINT ["/app/start.sh"]