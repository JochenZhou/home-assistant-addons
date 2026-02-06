#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
set -e

# Read configuration
CREDENTIAL_PATH=$(bashio::config 'credential_path')
ENABLE_PUNCTUATION=$(bashio::config 'enable_punctuation')
DEBUG=$(bashio::config 'debug')

# Build arguments
ARGS=(
    --port 10300
    --credential-path "${CREDENTIAL_PATH}"
)

if [ "${ENABLE_PUNCTUATION}" = "true" ]; then
    ARGS+=(--enable-punctuation true)
else
    ARGS+=(--enable-punctuation false)
fi

if [ "${DEBUG}" = "true" ]; then
    ARGS+=(--debug)
fi

bashio::log.info "Starting Doubao ASR Wyoming service..."
bashio::log.info "Credential path: ${CREDENTIAL_PATH}"
bashio::log.info "Enable punctuation: ${ENABLE_PUNCTUATION}"

exec python3 /app/wyoming_doubao.py "${ARGS[@]}"
