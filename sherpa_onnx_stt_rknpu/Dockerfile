# Home Assistant Add-on: RK3566 NPU STT
ARG BUILD_FROM
# 使用 Debian-based 镜像以支持 manylinux wheels
FROM ghcr.io/home-assistant/aarch64-base-debian:bookworm

# 设置环境变量
ENV LANG=C.UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive
# 优化 Python 内存使用
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MALLOC_TRIM_THRESHOLD_=100000

# sherpa-onnx RKNN 版本
ARG SHERPA_ONNX_VERSION=1.12.23
# 是否使用中国镜像源 (true/false)
ARG USE_CN_MIRROR=true

# 安装系统依赖 - 最小化安装
RUN echo "=== Installing system packages ===" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-numpy \
        tar \
        bzip2 \
        curl \
        libasound2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo "=== System packages installed ==="

# 下载并安装 RKNN 运行时库
# 从 rockchip-linux/rknpu2 获取 librknnrt.so
ARG RKNPU2_VERSION=2.3.0
RUN echo "=== Installing RKNN Runtime ===" && \
    curl -L -o /tmp/rknpu2.tar.gz \
        "https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" && \
    mv /tmp/rknpu2.tar.gz /usr/lib/librknnrt.so && \
    chmod 755 /usr/lib/librknnrt.so && ldconfig && \
    echo "=== RKNN Runtime installed ==="

# 安装 wyoming 和 sherpa-onnx Python 绑定
# 根据 USE_CN_MIRROR 选择安装源
# 参考: https://k2-fsa.github.io/sherpa/onnx/rknn/install.html#install
RUN pip3 install --no-cache-dir --break-system-packages wyoming==1.5.4 && \
    if [ "${USE_CN_MIRROR}" = "true" ]; then \
        echo "=== Installing sherpa-onnx from CN mirror ===" && \
        pip3 install --no-cache-dir --break-system-packages sherpa-onnx==${SHERPA_ONNX_VERSION} \
            -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html; \
    else \
        echo "=== Installing sherpa-onnx from default mirror ===" && \
        pip3 install --no-cache-dir --break-system-packages sherpa-onnx==${SHERPA_ONNX_VERSION} \
            -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html; \
    fi

# 复制应用文件
COPY rknn_loader.py /app/
COPY wyoming_server.py /app/
COPY vad_asr.py /app/
COPY test_audio.py /app/

# 复制启动脚本
COPY run.sh /
RUN chmod a+x /run.sh

WORKDIR /app

CMD ["/run.sh"]
