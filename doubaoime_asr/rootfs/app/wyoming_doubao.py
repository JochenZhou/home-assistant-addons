#!/usr/bin/env python3
"""Wyoming protocol server for Doubao IME ASR."""
from __future__ import annotations

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from doubaoime_asr import ASRConfig, transcribe

_LOGGER = logging.getLogger(__name__)


class DoubaoASREventHandler(AsyncEventHandler):
    """Event handler for Doubao ASR."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.cli_args = cli_args
        self._audio_buffer = bytes()
        self._converter: Optional[AudioChunkConverter] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Transcribe.is_type(event.type):
            # 开始新的转录
            self._audio_buffer = bytes()
            _LOGGER.debug("Starting new transcription")
            return True

        if AudioStart.is_type(event.type):
            self._audio_buffer = bytes()
            # 创建转换器以确保音频格式正确
            self._converter = AudioChunkConverter(
                rate=16000,
                width=2,
                channels=1,
            )
            return True

        if AudioChunk.is_type(event.type):
            # 收集音频数据
            chunk = AudioChunk.from_event(event)
            if self._converter:
                chunk = self._converter.convert(chunk)
            self._audio_buffer += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            # 音频结束，开始识别
            _LOGGER.debug(
                "Audio stopped, processing %d bytes", len(self._audio_buffer)
            )

            if self._audio_buffer:
                try:
                    # 创建 ASR 配置
                    config = ASRConfig(
                        credential_path=self.cli_args.credential_path,
                        sample_rate=16000,
                        channels=1,
                        enable_punctuation=self.cli_args.enable_punctuation,
                    )

                    # 执行语音识别
                    text = await transcribe(self._audio_buffer, config=config)
                    _LOGGER.info("Transcription result: %s", text)

                    # 发送结果
                    await self.write_event(Transcript(text=text or "").event())

                except Exception as err:
                    _LOGGER.exception("Error during transcription: %s", err)
                    await self.write_event(Transcript(text="").event())
            else:
                await self.write_event(Transcript(text="").event())

            # 清空缓冲区
            self._audio_buffer = bytes()
            return True

        return True


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Doubao ASR Wyoming Server")
    parser.add_argument(
        "--port",
        type=int,
        default=10300,
        help="Server port (default: 10300)",
    )
    parser.add_argument(
        "--credential-path",
        default="/data/credentials.json",
        help="Path to credential file",
    )
    parser.add_argument(
        "--enable-punctuation",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Enable punctuation in transcription",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _LOGGER.info("Starting Doubao ASR Wyoming Server")
    _LOGGER.info("Port: %s", args.port)
    _LOGGER.info("Credential path: %s", args.credential_path)
    _LOGGER.info("Enable punctuation: %s", args.enable_punctuation)

    # Wyoming 服务信息
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="doubaoime-asr",
                description="豆包输入法语音识别",
                version="1.0.0",
                attribution=Attribution(
                    name="Doubao IME",
                    url="https://github.com/starccy/doubaoime-asr",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name="doubao",
                        description="豆包语音识别模型",
                        version="1.0.0",
                        attribution=Attribution(
                            name="ByteDance",
                            url="https://www.doubao.com/",
                        ),
                        installed=True,
                        languages=["zh", "en"],
                    )
                ],
            )
        ],
    )

    # 启动服务器
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{args.port}")
    _LOGGER.info("Server started, listening on port %s", args.port)

    try:
        await server.run(
            partial(
                DoubaoASREventHandler,
                wyoming_info,
                args,
            )
        )
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
