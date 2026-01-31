# Sherpa-Onnx-STT-RKNPU - Home Assistant 插件

基于 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 RK3566 NPU 加速语音识别服务，支持 [Wyoming 协议](https://github.com/rhasspy/wyoming)。

## 功能特性

- 语音识别 (STT) - 支持中英日韩粤多语言
- RK3566/RK3588 NPU 硬件加速（默认启用）
- 情感识别 - SenseVoice 模型支持识别说话人情感（NEUTRAL/HAPPY/SAD/ANGRY）
- 语音活动检测 (VAD) - 自动处理长音频分段（延迟加载，节省内存）
- Wyoming 协议支持，与 Home Assistant 无缝集成

## 支持的硬件

- RK3566 (1 NPU核心, 0.8 TOPS)
- RK3568
- RK3588 (3 NPU核心, 6 TOPS)
- RK3576
- RK3562

## 安装

### 方式一：通过 Home Assistant Add-on Store

1. 在 Home Assistant 中，进入 **设置** > **加载项** > **加载项商店**
2. 点击右上角菜单，选择 **仓库**
3. 添加仓库地址（如果有自定义仓库）
4. 刷新后找到 **Sherpa-Onnx-STT-RKNPU** 并安装

### 方式二：本地安装

1. 将此目录复制到 Home Assistant 的 `/addons/` 目录
2. 在 **设置** > **加载项** > **加载项商店** 中刷新
3. 找到 **Sherpa-Onnx-STT-RKNPU** 并安装

## 配置选项

```yaml
# 模型存储目录
model_dir: /share/models

# 调试模式
debug: false

# 自定义模型（留空使用默认 sense-voice-zh-5s-rk3566）
custom_model: ""

# 输出情绪 emoji（在识别文本前添加情绪 emoji）
output_emotion: false
```

### 配置选项详解

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | 字符串 | `/share/models` | 模型文件存储路径 |
| `debug` | 布尔值 | `false` | 调试模式，输出详细日志 |
| `custom_model` | 字符串 | `""` | 自定义模型名称，留空使用默认模型 |
| `output_emotion` | 布尔值 | `false` | 在识别文本前输出情绪 emoji |

## Home Assistant 集成

安装并启动加载项后，在 Home Assistant 中配置 Wyoming 集成：

### 配置 STT

1. 进入 **设置** > **设备与服务** > **添加集成**
2. 搜索 **Wyoming Protocol**
3. 输入主机：`localhost` 或加载项IP
4. 端口：`10400`

### 在 Assist 中使用

配置完成后，可以在 **设置** > **语音助手** 中选择：
- 语音转文字：`Sherpa-Onnx-SenseVoice-RKNPU`

## 支持的模型

本插件仅支持 **SenseVoice** 模型，该模型对短音频识别效果最好，且支持情感识别。

### SenseVoice NPU 加速模型

| 模型 | 语言 | 时长限制 | 说明 |
|------|------|----------|------|
| `sense-voice-zh-5s-rk3566` | 中英日韩粤 | 5秒 | 最快，适合智能家居命令 |
| `sense-voice-zh-10s-rk3566` | 中英日韩粤 | 10秒 | 推荐，平衡速度和时长 |
| `sense-voice-zh-15s-rk3566` | 中英日韩粤 | 15秒 | 适合较长语音 |
| `sense-voice-zh-20s-rk3566` | 中英日韩粤 | 20秒 | 适合长语音 |
| `sense-voice-zh-25s-rk3566` | 中英日韩粤 | 25秒 | 适合长语音 |
| `sense-voice-zh-30s-rk3566` | 中英日韩粤 | 30秒 | 最长时长支持 |

### 模型选择建议

默认使用 `sense-voice-zh-5s-rk3566` 模型，适合大多数智能家居场景。

如需使用其他模型，在 `custom_model` 中填写模型名称：

1. **短语音命令** (< 5秒): `sense-voice-zh-5s-rk3566` (默认，最快)
2. **一般对话** (5-10秒): `sense-voice-zh-10s-rk3566` (推荐)
3. **长语音** (> 10秒): `sense-voice-zh-20s-rk3566` 或更长时长模型

**注意**: 超过模型时长限制的音频会自动通过 VAD 分段处理。

## Wyoming 协议端口

| 服务 | 端口 | 说明 |
|------|------|------|
| STT | 10400 | 语音识别服务 |

## 性能说明

### 内存占用

- SenseVoice 模型: ~650-700 MB
- VAD 模型: ~10-20 MB（延迟加载，仅在处理长音频时加载）
- 总内存占用: ~700-750 MB

### NPU 性能

RK3566 只有1个NPU核心 (0.8 TOPS)，相比RK3588 (6 TOPS) 性能较低。

典型识别速度 (RTF - Real Time Factor):
- 5秒模型: RTF ~0.1-0.2 (识别1秒音频需要0.1-0.2秒)
- 10秒模型: RTF ~0.15-0.25

### 模型时长限制

SenseVoice 模型有固定的输入时长限制：
- 如果输入音频小于模型时长，会自动填充
- 如果输入音频大于模型时长，VAD 会自动将长音频分段处理

VAD 功能默认启用，会在需要时延迟加载，节省内存。

### SenseVoice 情感识别

SenseVoice 模型支持识别说话人的情感状态：
- 😊 `HAPPY` - 开心
- 😢 `SAD` - 悲伤
- 😠 `ANGRY` - 愤怒
- `NEUTRAL` - 中性（不输出 emoji）

启用 `output_emotion: true` 后，识别结果会在文本前添加情绪 emoji，例如：
- `😊 打开客厅的灯`
- `😠 关掉空调`
- `现在几点了`（中性情绪无 emoji）

可以在 HA 自动化中使用模板提取情绪：
```yaml
# 提取情绪 emoji
{% set text = states('sensor.stt_text') %}
{% if text[:1] == '😊' %}happy{% elif text[:1] == '😠' %}angry{% elif text[:1] == '😢' %}sad{% else %}neutral{% endif %}
```

**注意**: 当前 RKNN 版本的 SenseVoice 模型可能返回 `EMO_UNKNOWN`，无法识别具体情绪。这是 RKNN 模型的限制，非 RKNN 版本（CPU/GPU）支持完整的情绪识别。

## 目录结构

```
sherpa_onnx_stt_rknpu/
├── config.yaml         # HA加载项配置
├── config.json         # 备用配置格式
├── Dockerfile          # Docker构建文件
├── build.yaml          # 构建配置
├── run.sh              # 启动脚本
├── rknn_loader.py      # RKNN模型加载器（仅支持SenseVoice）
├── vad_asr.py          # VAD + ASR 长音频处理模块
├── wyoming_server.py   # Wyoming协议服务器
└── README.md           # 说明文档
```

## 故障排除

### NPU 未检测到

确保设备映射正确：
```yaml
devices:
  - /dev/rknpu
```

### 模型下载失败

检查网络连接，或手动下载模型到 `/share/models/` 目录。

模型下载地址：
```
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-rk3566-5-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

### 识别效果不佳

1. 确保音频采样率为 16kHz
2. 尝试使用不同时长的模型
3. 检查麦克风质量
4. 确保说话清晰，减少背景噪音

### 内存不足

如果遇到内存不足问题：
1. 使用较小时长的模型（如 5s 模型）
2. 确保系统有足够的可用内存（建议 1GB 以上）

## 参考资料

- [Wyoming Protocol](https://github.com/rhasspy/wyoming)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [sherpa-onnx RKNN文档](https://k2-fsa.github.io/sherpa/onnx/rknn/index.html)
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [SenseVoice 模型](https://github.com/FunAudioLLM/SenseVoice)
