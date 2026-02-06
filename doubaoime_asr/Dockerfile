ARG BUILD_FROM
FROM ${BUILD_FROM}

# 安装系统依赖
RUN apk add --no-cache \
    python3 \
    py3-pip \
    opus \
    opus-dev \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev \
    git

# 创建虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装 Python 依赖
RUN pip3 install --no-cache-dir \
    wyoming==1.5.4 \
    cryptography \
    git+https://github.com/starccy/doubaoime-asr.git

# 复制应用文件
COPY rootfs /

# 设置工作目录
WORKDIR /app

# 设置权限
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
