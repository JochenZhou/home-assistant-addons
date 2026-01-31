"""
RK3566 NPU Loader for sherpa-onnx
支持RK3566 NPU加速的语音识别加载器

仅支持 SenseVoice 模型（对短音频识别效果最好）
"""
import os
import logging
import subprocess
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

_LOGGER = logging.getLogger("rknn_loader")

# RK3566支持的NPU核心数
RK3566_NPU_CORES = 1


@dataclass
class RKNNConfig:
    """RKNN配置"""
    # 模型目录
    model_dir: str = "/models"
    # NPU核心数 (RK3566只有1个NPU核心)
    num_threads: int = 1
    # 是否启用RKNN加速
    enable_rknn: bool = True
    # librknnrt.so路径
    rknn_lib_path: str = "/lib/librknnrt.so"
    # 采样率
    sample_rate: int = 16000
    # 额外参数
    extra_params: Dict[str, Any] = field(default_factory=dict)


class RKNNModelLoader:
    """RKNN模型加载器"""

    def __init__(self, config: RKNNConfig):
        self.config = config
        self._sherpa_onnx = None
        self._check_rknn_support()

    def _check_rknn_support(self) -> bool:
        """检查RKNN支持"""
        # 检查librknnrt.so是否存在
        if not os.path.exists(self.config.rknn_lib_path):
            alt_paths = [
                "/usr/lib/librknnrt.so",
                "/usr/local/lib/librknnrt.so",
                "/lib/aarch64-linux-gnu/librknnrt.so",
                "/lib/librknnrt.so",
                "/usr/lib/aarch64-linux-gnu/librknnrt.so",
                "/vendor/lib64/librknnrt.so",
                "/system/lib64/librknnrt.so"
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    self.config.rknn_lib_path = path
                    break
            else:
                _LOGGER.warning(
                    "librknnrt.so not found. RKNN acceleration disabled. "
                    "Please install RKNN runtime from: "
                    "https://github.com/airockchip/rknn-toolkit2"
                )
                self.config.enable_rknn = False
                return False

        # 检查版本
        try:
            result = subprocess.run(
                ["strings", self.config.rknn_lib_path],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if "librknnrt version" in line:
                    _LOGGER.info(f"RKNN Runtime: {line.strip()}")
                    break
        except Exception as e:
            _LOGGER.warning(f"Could not check RKNN version: {e}")

        return True

    def _import_sherpa_onnx(self):
        """延迟导入sherpa_onnx"""
        if self._sherpa_onnx is None:
            try:
                import sherpa_onnx
                self._sherpa_onnx = sherpa_onnx

                # 验证RKNN支持
                if self.config.enable_rknn:
                    exe_path = subprocess.run(
                        ["which", "sherpa-onnx"],
                        capture_output=True, text=True
                    ).stdout.strip()
                    if exe_path:
                        ldd_result = subprocess.run(
                            ["ldd", exe_path],
                            capture_output=True, text=True
                        )
                        if "librknnrt.so" not in ldd_result.stdout:
                            _LOGGER.warning(
                                "sherpa-onnx not compiled with RKNN support. "
                                "Please install RKNN version: "
                                "pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html"
                            )
            except ImportError:
                _LOGGER.error(
                    "sherpa_onnx not installed. Please install: "
                    "pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html"
                )
                raise
        return self._sherpa_onnx


class STTModelLoader(RKNNModelLoader):
    """语音识别模型加载器 - 仅支持 SenseVoice 模型"""

    # 支持的 SenseVoice 模型 (2024-07-17 版本，语言检测更准确)
    SUPPORTED_MODELS = {
        # 5秒模型 - 最快
        "sense-voice-zh-5s-rk3566": {
            "name": "sherpa-onnx-rk3566-5-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 5.0,
            "description": "SenseVoice多语言模型 5秒 (支持情绪识别)"
        },
        # 10秒模型 - 推荐
        "sense-voice-zh-10s-rk3566": {
            "name": "sherpa-onnx-rk3566-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 10.0,
            "description": "SenseVoice多语言模型 10秒 (支持情绪识别) - 推荐"
        },
        # 默认别名 - 使用 10 秒模型
        "sense-voice-zh-en": {
            "name": "sherpa-onnx-rk3566-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 10.0,
            "description": "SenseVoice多语言模型 10秒 (中英日韩粤，支持情绪识别) - 默认"
        },
        "sense-voice-zh-rk3566": {
            "name": "sherpa-onnx-rk3566-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 10.0,
            "description": "SenseVoice多语言模型 10秒 (支持情绪识别)"
        },
        # 15秒模型
        "sense-voice-zh-15s-rk3566": {
            "name": "sherpa-onnx-rk3566-15-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 15.0,
            "description": "SenseVoice多语言模型 15秒 (支持情绪识别)"
        },
        # 20秒模型
        "sense-voice-zh-20s-rk3566": {
            "name": "sherpa-onnx-rk3566-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 20.0,
            "description": "SenseVoice多语言模型 20秒 (支持情绪识别)"
        },
        # 25秒模型
        "sense-voice-zh-25s-rk3566": {
            "name": "sherpa-onnx-rk3566-25-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 25.0,
            "description": "SenseVoice多语言模型 25秒 (支持情绪识别)"
        },
        # 30秒模型
        "sense-voice-zh-30s-rk3566": {
            "name": "sherpa-onnx-rk3566-30-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "type": "sense_voice",
            "languages": ["zh", "en", "ja", "ko", "yue"],
            "max_duration": 30.0,
            "description": "SenseVoice多语言模型 30秒 (支持情绪识别)"
        },
    }

    def __init__(self, config: RKNNConfig, model_key: str = "sense-voice-zh-5s-rk3566"):
        super().__init__(config)
        self.model_key = model_key
        self.model_info = self.SUPPORTED_MODELS.get(model_key)

        if not self.model_info:
            # 尝试作为模型文件夹名称直接使用
            self.model_info = self._detect_custom_model(model_key)
            if self.model_info:
                _LOGGER.info(f"Using custom model folder: {model_key}")
            else:
                raise ValueError(
                    f"Unsupported model: {model_key}. "
                    f"Available models: {list(self.SUPPORTED_MODELS.keys())}. "
                    f"Or use SenseVoice model folder name directly."
                )

    def _detect_custom_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """检测自定义 SenseVoice 模型文件夹"""
        model_path = os.path.join(self.config.model_dir, model_name)

        if not os.path.exists(model_path):
            _LOGGER.warning(f"Custom model folder not found: {model_path}")
            return None

        # 扫描文件夹内容
        files = os.listdir(model_path)
        _LOGGER.info(f"Detecting model type from files: {files}")

        # 检查是否有 RKNN 文件
        has_rknn = any(f.endswith('.rknn') for f in files)
        has_tokens = any(f == 'tokens.txt' for f in files)

        model_name_lower = model_name.lower()

        # 只支持 SenseVoice 模型
        if 'sense-voice' in model_name_lower or 'sensevoice' in model_name_lower:
            if not has_rknn:
                _LOGGER.warning(f"No RKNN model file found in {model_path}")
                return None
            if not has_tokens:
                _LOGGER.warning(f"No tokens.txt found in {model_path}")
                return None

            # 从文件夹名称提取时长限制
            max_duration = 5.0  # 默认 5 秒
            if '5-seconds' in model_name_lower or '5s' in model_name_lower:
                max_duration = 5.0
            elif '10-seconds' in model_name_lower or '10s' in model_name_lower:
                max_duration = 10.0
            elif '20-seconds' in model_name_lower or '20s' in model_name_lower:
                max_duration = 20.0

            model_info = {
                "name": model_name,
                "type": "sense_voice",
                "languages": ["zh", "en", "ja", "ko", "yue"],
                "max_duration": max_duration,
                "description": f"Custom SenseVoice model: {model_name}"
            }

            _LOGGER.info(f"Detected model info: {model_info}")
            return model_info
        else:
            _LOGGER.error(
                f"Only SenseVoice models are supported. "
                f"Model name must contain 'sense-voice' or 'sensevoice'. "
                f"Got: {model_name}"
            )
            return None

    def get_model_path(self) -> str:
        """获取模型路径"""
        return os.path.join(self.config.model_dir, self.model_info["name"])

    def download_model(self) -> str:
        """下载模型"""
        model_path = self.get_model_path()
        model_name = self.model_info["name"]

        if os.path.exists(model_path):
            _LOGGER.info(f"Model already exists: {model_path}")
            return model_path

        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_name}.tar.bz2"

        _LOGGER.info(f"Downloading model from: {url}")
        os.makedirs(self.config.model_dir, exist_ok=True)

        try:
            archive_path = os.path.join(self.config.model_dir, f"{model_name}.tar.bz2")
            subprocess.check_call(["curl", "-L", url, "-o", archive_path])
            subprocess.check_call(["tar", "-xjf", archive_path, "-C", self.config.model_dir])
            os.remove(archive_path)
            _LOGGER.info(f"Model downloaded and extracted to: {model_path}")
        except subprocess.CalledProcessError as e:
            _LOGGER.error(f"Failed to download model: {e}")
            raise

        return model_path

    def load(self) -> Any:
        """加载 SenseVoice STT 模型"""
        sherpa_onnx = self._import_sherpa_onnx()
        model_path = self.download_model()

        _LOGGER.info(f"Loading SenseVoice model: {self.model_info['name']}")

        # 查找RKNN模型文件
        rknn_model = None
        onnx_model = None

        for f in os.listdir(model_path):
            if f.endswith(".rknn"):
                rknn_model = os.path.join(model_path, f)
            elif f.endswith(".onnx") and "model" in f:
                onnx_model = os.path.join(model_path, f)

        tokens = os.path.join(model_path, "tokens.txt")

        # 优先使用RKNN模型
        if self.config.enable_rknn and rknn_model:
            _LOGGER.info(f"Using RKNN model: {rknn_model}")
            model_file = rknn_model
            provider = "rknn"
            num_threads = 1  # RK3566 只有1个NPU核心
        elif onnx_model:
            _LOGGER.info(f"Using ONNX model: {onnx_model}")
            model_file = onnx_model
            provider = "cpu"
            num_threads = self.config.num_threads
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")

        _LOGGER.info(f"Creating OfflineRecognizer with provider={provider}, num_threads={num_threads}")

        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_file,
                tokens=tokens,
                num_threads=num_threads,
                use_itn=True,
                language="auto",  # 自动检测语言，可选: auto, zh, en, ja, ko, yue
                debug=False,
                provider=provider,
            )
            _LOGGER.info(f"SenseVoice model loaded successfully with provider={provider}")
            return recognizer
        except Exception as e:
            error_msg = str(e).lower()
            if provider == "rknn" and ("npu" in error_msg or "rknn" in error_msg or "busy" in error_msg or "device" in error_msg):
                _LOGGER.error(
                    "NPU 初始化失败！可能原因：\n"
                    "  1. NPU 正被其他进程占用（如其他 STT/TTS 服务）\n"
                    "  2. NPU 驱动未正确加载\n"
                    "  3. 设备不支持 RKNN\n"
                    f"原始错误: {e}"
                )
            raise


def check_rk3566_npu() -> Dict[str, Any]:
    """检查RK3566 NPU状态"""
    info = {
        "available": False,
        "driver_version": None,
        "runtime_version": None,
        "cores": RK3566_NPU_CORES,
    }

    # 检查NPU设备
    npu_device = "/dev/rknpu"
    if os.path.exists(npu_device):
        info["available"] = True

    # 检查驱动版本
    try:
        result = subprocess.run(
            ["cat", "/sys/kernel/debug/rknpu/version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip()
    except Exception:
        pass

    # 检查运行时版本
    for lib_path in ["/lib/librknnrt.so", "/usr/lib/librknnrt.so"]:
        if os.path.exists(lib_path):
            try:
                result = subprocess.run(
                    ["strings", lib_path],
                    capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if "librknnrt version" in line:
                        info["runtime_version"] = line.strip()
                        break
            except Exception:
                pass
            break

    return info


# 便捷函数
def create_stt_recognizer(
    model: str = "sense-voice-zh-5s-rk3566",
    model_dir: str = "/models",
    enable_rknn: bool = True
) -> Any:
    """创建STT识别器"""
    config = RKNNConfig(model_dir=model_dir, enable_rknn=enable_rknn)
    loader = STTModelLoader(config, model)
    return loader.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 检查NPU状态
    npu_info = check_rk3566_npu()
    print("RK3566 NPU Status:")
    for k, v in npu_info.items():
        print(f"  {k}: {v}")

    # 列出支持的模型
    print("\nSupported SenseVoice Models:")
    for key, info in STTModelLoader.SUPPORTED_MODELS.items():
        print(f"  {key}: {info['description']}")
