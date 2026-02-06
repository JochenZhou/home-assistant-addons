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

from doubaoime_asr import ASRConfig, transcribe_realtime, ResponseType

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
        self._converter: Optional[AudioChunkConverter] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._final_text: str = ""

    async def _audio_generator(self):
        """异步生成器，从队列中读取音频数据"""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:  # 结束信号
                break
            yield chunk

    async def _run_stream(self, config: ASRConfig):
        """运行流式识别，处理中间和最终结果"""
        try:
            async for response in transcribe_realtime(
                self._audio_generator(),
                config=config
            ):
                match response.type:
                    case ResponseType.FINAL_RESULT:
                        self._final_text = response.text
                    case ResponseType.ERROR:
                        _LOGGER.error("Stream error: %s", response.error_msg)
        except Exception as err:
            _LOGGER.exception("Error in stream: %s", err)

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming events."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if Transcribe.is_type(event.type):
            # 开始新的转录
            _LOGGER.debug("Starting new transcription")
            return True

        if AudioStart.is_type(event.type):
            # 创建转换器以确保音频格式正确
            self._converter = AudioChunkConverter(
                rate=16000,
                width=2,
                channels=1,
            )
            # 初始化流式识别
            self._audio_queue = asyncio.Queue()
            self._final_text = ""
            # 创建 ASR 配置
            config = ASRConfig(
                credential_path=self.cli_args.credential_path,
                sample_rate=16000,
                channels=1,
                enable_punctuation=self.cli_args.enable_punctuation,
            )
            # 启动流式识别任务
            self._stream_task = asyncio.create_task(self._run_stream(config))
            _LOGGER.debug("Started streaming recognition")
            return True

        if AudioChunk.is_type(event.type):
            # 将音频数据放入队列
            chunk = AudioChunk.from_event(event)
            if self._converter:
                chunk = self._converter.convert(chunk)
            if self._audio_queue:
                await self._audio_queue.put(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            # 音频结束，发送结束信号并等待结果
            _LOGGER.debug("Audio stopped, waiting for final result")

            if self._audio_queue and self._stream_task:
                # 发送结束信号
                await self._audio_queue.put(None)
                # 等待流式识别完成
                try:
                    await self._stream_task
                except Exception as err:
                    _LOGGER.exception("Error waiting for stream task: %s", err)

                _LOGGER.info("Transcription result: %s", self._final_text)
                await self.write_event(Transcript(text=self._final_text or "").event())
            else:
                await self.write_event(Transcript(text="").event())

            # 清理状态
            self._audio_queue = None
            self._stream_task = None
            self._final_text = ""
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
