"""
Wyoming Protocol Server for RK3566 NPU STT
支持 Home Assistant 的 Wyoming 协议语音识别服务

仅支持 SenseVoice 模型（对短音频识别效果最好）

注意：当前 RKNN 版本的 SenseVoice 模型不输出情绪信息，只输出纯文本
"""
import argparse
import asyncio
import logging
import os
from functools import partial
from typing import Optional

import numpy as np

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from rknn_loader import (
    RKNNConfig,
    STTModelLoader,
    check_rk3566_npu,
)

# VAD 模块可选导入
try:
    from vad_asr import VADWithASR, VADConfig, create_vad_asr
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    VADWithASR = None
    VADConfig = None
    create_vad_asr = None

_LOGGER = logging.getLogger(__name__)

# 情绪到 emoji 的映射（中性不输出）
EMOTION_EMOJI = {
    "HAPPY": "😊",
    "SAD": "😢",
    "ANGRY": "😠",
}


class STTEventHandler(AsyncEventHandler):
    """Wyoming STT 事件处理器"""

    # 类级别共享的 VAD 实例（延迟加载，所有 handler 共享）
    _shared_vad_asr: Optional[VADWithASR] = None
    _vad_config: Optional[dict] = None

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        recognizer,
        model_info: dict,
        vad_config: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info = wyoming_info
        self.recognizer = recognizer
        self.model_info = model_info
        # 保存 VAD 配置用于延迟加载
        if vad_config:
            STTEventHandler._vad_config = vad_config
        self._audio_buffer = bytes()
        self._sample_rate = 16000
        self._sample_width = 2
        self._channels = 1
        self._converter: Optional[AudioChunkConverter] = None

    def _get_vad_asr(self) -> Optional[VADWithASR]:
        """延迟加载 VAD（只在第一次需要时加载）"""
        if STTEventHandler._shared_vad_asr is not None:
            return STTEventHandler._shared_vad_asr

        if STTEventHandler._vad_config is None:
            return None

        config = STTEventHandler._vad_config
        _LOGGER.info("Lazy loading VAD for long audio processing...")
        STTEventHandler._shared_vad_asr = create_vad_asr(
            recognizer=self.recognizer,
            vad_model_path=config["vad_model_path"],
            threshold=config["threshold"],
            min_silence_duration=config["min_silence_duration"],
            max_segment_duration=config["max_segment_duration"],
        )
        _LOGGER.info(f"VAD loaded (threshold: {config['threshold']}, max segment: {config['max_segment_duration']}s)")
        return STTEventHandler._shared_vad_asr

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True

        if AudioStart.is_type(event.type):
            self._audio_buffer = bytes()
            audio_start = AudioStart.from_event(event)
            self._sample_rate = audio_start.rate
            self._sample_width = audio_start.width
            self._channels = audio_start.channels

            # 创建转换器以确保音频格式正确
            self._converter = AudioChunkConverter(
                rate=16000,
                width=2,
                channels=1,
            )
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._converter:
                chunk = self._converter.convert(chunk)
            self._audio_buffer += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            if not self._audio_buffer:
                _LOGGER.warning("No audio data received")
                await self.write_event(Transcript(text="").event())
                return True

            # 转换为numpy数组
            audio_array = np.frombuffer(self._audio_buffer, dtype=np.int16)
            audio_float = audio_array.astype(np.float32, copy=False)
            audio_float *= (1.0 / 32768.0)

            # 执行识别
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    partial(self._transcribe, audio_float),
                )
                _LOGGER.info(f"Transcribed: {text}")
            except Exception as e:
                _LOGGER.error(f"Transcription error: {e}")
                text = ""

            await self.write_event(Transcript(text=text).event())
            # 清理缓冲区
            self._audio_buffer = bytes()
            del audio_array, audio_float
            return True

        if Transcribe.is_type(event.type):
            # 处理 Transcribe 事件（可能包含语言设置）
            return True

        return True

    def _transcribe(self, audio: np.ndarray) -> str:
        """执行 SenseVoice 语音识别

        SenseVoice 模型支持情绪识别，结果包含：
        - text: 识别文本
        - lang: 语言 (如 <|zh|>)
        - emotion: 情绪 (如 <|NEUTRAL|>, <|HAPPY|>, <|SAD|>, <|ANGRY|>)
        - event: 事件 (如 <|Speech|>, <|BGM|>)
        """
        import time
        start_time = time.time()

        audio_duration = len(audio) / 16000.0
        _LOGGER.info(f"Audio duration: {audio_duration:.2f}s")

        # 获取模型的最大时长限制
        model_max_duration = self.model_info.get("max_duration", 5.0)

        emotion_emoji = ""

        # 如果启用了 VAD 且音频超过模型最大时长，使用 VAD 分段处理
        vad_asr = self._get_vad_asr()
        if vad_asr and audio_duration > model_max_duration:
            _LOGGER.info(f"Using VAD for long audio ({audio_duration:.2f}s > {model_max_duration:.1f}s)")
            text = vad_asr.transcribe(audio, 16000)
            # VAD 模式下暂不支持情绪输出
        else:
            # SenseVoice 离线识别
            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self.recognizer.decode_stream(stream)
            result = stream.result
            text = result.text.strip() if result else ""

            # 提取情绪和事件信息
            if result:
                emotion = getattr(result, 'emotion', None)
                event = getattr(result, 'event', None)
                lang = getattr(result, 'lang', None)
                if emotion or event:
                    # 清理标记格式 <|XXX|> -> XXX
                    emotion_clean = emotion.replace('<|', '').replace('|>', '') if emotion else None
                    event_clean = event.replace('<|', '').replace('|>', '') if event else None
                    lang_clean = lang.replace('<|', '').replace('|>', '') if lang else None
                    _LOGGER.info(f"SenseVoice: lang={lang_clean}, emotion={emotion_clean}, event={event_clean}")

                    # 获取情绪 emoji
                    if emotion_clean and self.cli_args.output_emotion:
                        emotion_emoji = EMOTION_EMOJI.get(emotion_clean, "")

        elapsed = time.time() - start_time
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        _LOGGER.info(f"Recognition time: {elapsed:.3f}s, RTF: {rtf:.3f}")

        # 过滤无效识别结果（误唤醒时可能只识别出单个标点符号）
        if text and len(text) <= 2 and all(c in '.,。，、!！?？…' for c in text):
            _LOGGER.info(f"Filtered invalid result: '{text}' (likely false wake)")
            return ""

        # 如果启用情绪输出且有情绪 emoji，添加到文本前
        if emotion_emoji and text:
            text = f"{emotion_emoji} {text}"

        return text


async def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="RK3566 NPU Wyoming STT Server (SenseVoice)")

    parser.add_argument(
        "--stt-model",
        default="sense-voice-zh-5s-rk3566",
        help="SenseVoice model to use (default: sense-voice-zh-5s-rk3566)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10400,
        help="Wyoming STT server port",
    )
    parser.add_argument(
        "--model-dir",
        default="/share/models",
        help="Model directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--enable-vad",
        action="store_true",
        help="Enable VAD for long audio processing",
    )
    parser.add_argument(
        "--vad-model",
        default="/share/models/silero_vad.onnx",
        help="Path to Silero VAD model",
    )
    parser.add_argument(
        "--vad-speech-threshold",
        type=float,
        default=0.5,
        help="VAD speech detection threshold (0-1)",
    )
    parser.add_argument(
        "--vad-min-silence",
        type=float,
        default=0.5,
        help="Minimum silence duration to split segments (seconds)",
    )
    parser.add_argument(
        "--output-emotion",
        action="store_true",
        help="Output emotion emoji prefix in transcription",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 检查 NPU 状态
    npu_info = check_rk3566_npu()
    _LOGGER.info(f"NPU Status: {npu_info}")

    # 创建配置 - RKNN 默认启用
    config = RKNNConfig(
        model_dir=args.model_dir,
        enable_rknn=True,
    )

    # 加载 SenseVoice STT 模型
    _LOGGER.info(f"Loading SenseVoice model: {args.stt_model}")
    stt_loader = STTModelLoader(config, args.stt_model)
    model_info = stt_loader.model_info
    recognizer = stt_loader.load()
    _LOGGER.info("SenseVoice model loaded")

    # 准备 VAD 配置（延迟加载，只在需要时才加载 VAD 模型）
    vad_config = None
    if args.enable_vad:
        if not VAD_AVAILABLE:
            _LOGGER.warning("VAD requested but vad_asr module not available")
        elif not os.path.exists(args.vad_model):
            _LOGGER.warning(f"VAD model not found: {args.vad_model}")
            _LOGGER.info("Download with: curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx -o " + args.vad_model)
        else:
            max_duration = model_info.get("max_duration", 5.0)
            vad_config = {
                "vad_model_path": args.vad_model,
                "threshold": args.vad_speech_threshold,
                "min_silence_duration": args.vad_min_silence,
                "max_segment_duration": max_duration,
            }
            _LOGGER.info(f"VAD enabled (lazy loading, threshold: {args.vad_speech_threshold}, max segment: {max_duration}s)")

    # 构建 Wyoming Info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="Sherpa-Onnx-SenseVoice-RKNPU",
                description=f"SenseVoice STT with RKNPU ({model_info['description']})",
                attribution=Attribution(
                    name="sherpa-onnx",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=args.stt_model,
                        description=model_info["description"],
                        attribution=Attribution(
                            name="k2-fsa",
                            url="https://github.com/k2-fsa/sherpa-onnx",
                        ),
                        installed=True,
                        languages=model_info.get("languages", ["zh", "en", "ja", "ko", "yue"]),
                        version="1.0.0",
                    )
                ],
            )
        ],
    )

    # 启动服务器
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{args.port}")
    _LOGGER.info(f"Wyoming STT server listening on port {args.port}")

    await server.run(
        partial(
            STTEventHandler,
            wyoming_info,
            args,
            recognizer,
            model_info,
            vad_config,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
