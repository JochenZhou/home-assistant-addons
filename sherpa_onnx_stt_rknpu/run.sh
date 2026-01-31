#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
set -e

# Read configuration
MODEL_DIR=$(bashio::config 'model_dir')
DEBUG=$(bashio::config 'debug')
CUSTOM_MODEL=$(bashio::config 'custom_model' || echo "")
OUTPUT_EMOTION=$(bashio::config 'output_emotion')

# 确定使用的模型：自定义模型或默认模型
if [ -n "${CUSTOM_MODEL}" ]; then
    STT_MODEL="${CUSTOM_MODEL}"
    bashio::log.info "Using custom model: ${STT_MODEL}"
else
    STT_MODEL="sense-voice-zh-5s-rk3566"
    bashio::log.info "Using default model: ${STT_MODEL}"
fi

# 确保 VAD 模型存在（如果启用 VAD 或需要处理长音频）
VAD_MODEL_PATH="${MODEL_DIR}/silero_vad.onnx"
if [ ! -f "${VAD_MODEL_PATH}" ]; then
    bashio::log.info "Downloading Silero VAD model..."
    curl -L -o "${VAD_MODEL_PATH}" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
        || bashio::log.warning "Failed to download VAD model"
    if [ -f "${VAD_MODEL_PATH}" ]; then
        bashio::log.info "VAD model downloaded successfully"
    fi
fi

# Build arguments
ARGS=(
    --stt-model "${STT_MODEL}"
    --model-dir "${MODEL_DIR}"
    --port 10400
    --enable-vad
    --vad-model "${VAD_MODEL_PATH}"
)

if [ "${DEBUG}" = "true" ]; then
    ARGS+=(--debug)
fi

if [ "${OUTPUT_EMOTION}" = "true" ]; then
    ARGS+=(--output-emotion)
fi

bashio::log.info "Starting Sherpa-Onnx STT RKNPU service..."
bashio::log.info "STT Model: ${STT_MODEL}"
bashio::log.info "Output Emotion: ${OUTPUT_EMOTION}"

exec python3 /app/wyoming_server.py "${ARGS[@]}"
