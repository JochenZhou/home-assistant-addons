"""
VAD + ASR 长音频处理模块
使用 Silero VAD 将长音频切分成短片段，然后逐段用 SenseVoice 模型识别

原理：
1. Silero VAD 检测音频中的语音活动区域
2. 将检测到的语音片段逐个送入 SenseVoice 模型（5秒/10秒/20秒）
3. 合并所有片段的识别结果

这样即使使用 5 秒模型，也能处理任意长度的音频

注意：当前 RKNN 版本的 SenseVoice 模型不输出情绪信息，只输出纯文本
"""
import os
import logging
import numpy as np
from typing import Optional, List, Any
from dataclasses import dataclass

_LOGGER = logging.getLogger("vad_asr")


@dataclass
class SpeechSegment:
    """语音片段"""
    start: float  # 开始时间（秒）
    end: float    # 结束时间（秒）
    samples: np.ndarray  # 音频数据
    text: str = ""  # 识别结果


@dataclass
class VADConfig:
    """VAD 配置"""
    # Silero VAD 模型路径
    model_path: str = "/models/silero_vad.onnx"
    # 语音检测阈值 (0-1)，越高越严格
    # 降低阈值可以检测到更多语音，但可能包含更多噪音
    threshold: float = 0.3  # 从 0.5 降低到 0.3，更敏感
    # 最小静音时长（秒），用于分割语音片段
    # 降低此值可以更快地分割语音片段
    min_silence_duration: float = 0.3  # 从 0.5 降低到 0.3
    # 最小语音时长（秒），过短的片段会被忽略
    min_speech_duration: float = 0.1  # 从 0.25 降低到 0.1
    # 采样率
    sample_rate: int = 16000
    # VAD 窗口大小（样本数）
    window_size: int = 512
    # 语音前后的填充时间（秒），避免截断语音开头和结尾
    speech_pad_ms: int = 100  # 100ms 填充


class VADProcessor:
    """VAD 处理器 - 使用 Silero VAD 检测语音活动"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self._vad = None
        self._sherpa_onnx = None
        
    def _ensure_vad_model(self):
        """确保 VAD 模型存在"""
        if not os.path.exists(self.config.model_path):
            _LOGGER.info(f"Downloading Silero VAD model to {self.config.model_path}")
            import subprocess
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
            subprocess.check_call(["curl", "-L", url, "-o", self.config.model_path])
            _LOGGER.info("Silero VAD model downloaded")
    
    def _init_vad(self):
        """初始化 VAD"""
        if self._vad is not None:
            return
            
        self._ensure_vad_model()
        
        import sherpa_onnx
        self._sherpa_onnx = sherpa_onnx
        
        # 配置 VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = self.config.model_path
        vad_config.silero_vad.threshold = self.config.threshold
        vad_config.silero_vad.min_silence_duration = self.config.min_silence_duration
        vad_config.silero_vad.min_speech_duration = self.config.min_speech_duration
        vad_config.silero_vad.window_size = self.config.window_size
        vad_config.sample_rate = self.config.sample_rate
        
        # 创建 VAD 检测器，buffer 设置为 30 秒（智能家居场景足够）
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        _LOGGER.info(f"VAD initialized (threshold={self.config.threshold})")
    
    def detect_speech(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SpeechSegment]:
        """
        检测音频中的语音片段

        Args:
            samples: 音频数据 (float32, 范围 -1 到 1)
            sample_rate: 采样率


        Returns:
            语音片段列表
        """
        # 初始化或重置 VAD（避免重复加载模型）
        if self._vad is None:
            self._init_vad()
        else:
            # 使用 reset() 重置状态，比重新创建实例更高效
            self._vad.reset()
        
        # 确保采样率匹配
        if sample_rate != self.config.sample_rate:
            _LOGGER.warning(f"Sample rate mismatch: {sample_rate} vs {self.config.sample_rate}")
            # 这里可以添加重采样逻辑
        
        # 确保数据类型正确
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        # 归一化到 -1 到 1
        if np.abs(samples).max() > 1.0:
            samples = samples / 32768.0
        
        # 分块处理音频
        window_size = self.config.window_size
        segments = []
        
        # 喂入音频数据
        for i in range(0, len(samples), window_size):
            chunk = samples[i:i + window_size]
            if len(chunk) < window_size:
                # 填充最后一个块
                chunk = np.pad(chunk, (0, window_size - len(chunk)))
            self._vad.accept_waveform(chunk)
        
        # 标记输入结束
        self._vad.flush()
        
        # 提取检测到的语音片段
        while not self._vad.empty():
            speech = self._vad.front
            segment = SpeechSegment(
                start=speech.start / sample_rate,
                end=(speech.start + len(speech.samples)) / sample_rate,
                samples=np.array(speech.samples, dtype=np.float32)
            )
            segments.append(segment)
            self._vad.pop()
        
        _LOGGER.info(f"Detected {len(segments)} speech segments")
        for i, seg in enumerate(segments):
            _LOGGER.debug(f"  Segment {i}: {seg.start:.2f}s - {seg.end:.2f}s ({seg.end - seg.start:.2f}s)")
        
        return segments


class VADWithASR:
    """
    VAD + SenseVoice ASR 组合处理器

    使用 VAD 将长音频切分成短片段，然后用 SenseVoice 模型逐段识别
    """

    def __init__(self, recognizer: Any, vad_config: Optional[VADConfig] = None,
                 max_segment_duration: float = 5.0):
        """
        Args:
            recognizer: sherpa_onnx 的 SenseVoice OfflineRecognizer 实例
            vad_config: VAD 配置，如果为 None 则使用默认配置
            max_segment_duration: 单个片段最大时长（秒），超过会进一步切分
        """
        self.recognizer = recognizer
        self.vad_config = vad_config or VADConfig()
        self.vad_processor = VADProcessor(self.vad_config)
        self.max_segment_duration = max_segment_duration

    def _split_long_segment(self, segment: SpeechSegment, sample_rate: int) -> List[SpeechSegment]:
        """将超长片段切分成多个短片段"""
        duration = segment.end - segment.start
        if duration <= self.max_segment_duration:
            return [segment]

        # 计算需要切分成几段
        max_samples = int(self.max_segment_duration * sample_rate)
        sub_segments = []
        offset = 0

        while offset < len(segment.samples):
            end_offset = min(offset + max_samples, len(segment.samples))
            sub_start = segment.start + offset / sample_rate
            sub_end = segment.start + end_offset / sample_rate

            sub_segment = SpeechSegment(
                start=sub_start,
                end=sub_end,
                samples=segment.samples[offset:end_offset]
            )
            sub_segments.append(sub_segment)
            offset = end_offset

        _LOGGER.info(f"Split long segment ({duration:.2f}s) into {len(sub_segments)} sub-segments")
        return sub_segments

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """
        转录长音频
        
        Args:
            samples: 音频数据
            sample_rate: 采样率
            
        Returns:
            完整的转录文本
        """
        # 检测语音片段
        segments = self.vad_processor.detect_speech(samples, sample_rate)
        
        if not segments:
            _LOGGER.warning("No speech detected in audio")
            return ""

        # 对超长片段进行二次切分
        all_segments = []
        for seg in segments:
            all_segments.extend(self._split_long_segment(seg, sample_rate))

        # 逐段识别（使用 SenseVoice 模型）
        results = []
        for i, segment in enumerate(all_segments):
            _LOGGER.debug(f"Transcribing segment {i+1}/{len(all_segments)}: "
                        f"{segment.start:.2f}s - {segment.end:.2f}s")

            # 创建识别流
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, segment.samples)

            # 识别
            self.recognizer.decode_stream(stream)

            # 获取结果
            result = stream.result
            text = result.text.strip() if result else ""

            # 输出情绪和事件信息（如果可用）
            if result:
                emotion = getattr(result, 'emotion', None)
                event = getattr(result, 'event', None)
                if emotion or event:
                    emotion_clean = emotion.replace('<|', '').replace('|>', '') if emotion else None
                    event_clean = event.replace('<|', '').replace('|>', '') if event else None
                    _LOGGER.debug(f"  Emotion: {emotion_clean}, Event: {event_clean}")

            # 释放片段音频数据，减少内存占用
            segment.samples = None

            if text:
                results.append(text)
                _LOGGER.debug(f"  Result: {text}")
        
        # 合并结果
        full_text = " ".join(results)
        return full_text
    
    def transcribe_with_timestamps(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SpeechSegment]:
        """
        转录长音频并返回带时间戳的结果
        
        Args:
            samples: 音频数据
            sample_rate: 采样率
            
        Returns:
            带时间戳和文本的语音片段列表
        """
        # 检测语音片段
        segments = self.vad_processor.detect_speech(samples, sample_rate)
        
        if not segments:
            _LOGGER.warning("No speech detected in audio")
            return []
        
        # 逐段识别
        for i, segment in enumerate(segments):
            _LOGGER.info(f"Transcribing segment {i+1}/{len(segments)}: "
                        f"{segment.start:.2f}s - {segment.end:.2f}s")

            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, segment.samples)
            self.recognizer.decode_stream(stream)

            # 获取结果
            segment.text = stream.result.text.strip() if stream.result else ""
            _LOGGER.info(f"  [{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
        
        return segments


def create_vad_asr(recognizer: Any,
                   vad_model_path: str = "/models/silero_vad.onnx",
                   threshold: float = 0.5,
                   min_silence_duration: float = 0.5,
                   min_speech_duration: float = 0.1,
                   max_segment_duration: float = 5.0) -> VADWithASR:
    """
    创建 VAD + SenseVoice ASR 处理器的便捷函数

    Args:
        recognizer: sherpa_onnx 的 SenseVoice OfflineRecognizer 实例
        vad_model_path: Silero VAD 模型路径
        threshold: 语音检测阈值 (0.3 更敏感，0.5 更严格)
        min_silence_duration: 最小静音时长 (秒)
        min_speech_duration: 最小语音时长 (秒)
        max_segment_duration: 单个片段最大时长（秒），超过会进一步切分

    Returns:
        VADWithASR 实例
    """
    vad_config = VADConfig(
        model_path=vad_model_path,
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        min_speech_duration=min_speech_duration
    )
    return VADWithASR(recognizer, vad_config, max_segment_duration)


# 示例用法
if __name__ == "__main__":
    import wave
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # 读取音频文件
    def read_wave(filename: str) -> Tuple[np.ndarray, int]:
        with wave.open(filename, 'rb') as f:
            sample_rate = f.getframerate()
            samples = f.readframes(f.getnframes())
            samples = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sample_rate
    
    if len(sys.argv) < 2:
        print("Usage: python vad_asr.py <audio_file>")
        print("\n示例:")
        print("  python vad_asr.py long_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # 加载模型
    from rknn_loader import RKNNConfig, STTModelLoader
    
    config = RKNNConfig(model_dir="/models/stt")
    loader = STTModelLoader(config, model_key="sense-voice-zh-en")  # 使用 5 秒模型
    recognizer = loader.load()
    
    # 创建 VAD + ASR 处理器
    vad_asr = create_vad_asr(recognizer)
    
    # 读取音频
    samples, sample_rate = read_wave(audio_file)
    duration = len(samples) / sample_rate
    print(f"\n音频文件: {audio_file}")
    print(f"时长: {duration:.2f} 秒")
    
    # 转录
    print("\n开始转录...")
    segments = vad_asr.transcribe_with_timestamps(samples, sample_rate)
    
    # 输出结果
    print("\n=== 转录结果 ===")
    for seg in segments:
        print(f"[{seg.start:.2f}s - {seg.end:.2f}s]: {seg.text}")
    
    print("\n=== 完整文本 ===")
    full_text = " ".join(seg.text for seg in segments if seg.text)
    print(full_text)
