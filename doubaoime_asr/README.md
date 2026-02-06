# Doubao ASR Add-on for Home Assistant

基于豆包输入法的语音识别服务，通过 Wyoming 协议提供 STT 功能。

## 功能

- 高质量中文语音识别
- 支持英文识别
- 自动设备注册和凭据管理
- 通过 Wyoming 协议与 Home Assistant 集成

## 安装

1. 在 Home Assistant 中添加本地 Add-on 仓库
2. 安装 "豆包语音识别" Add-on
3. 启动 Add-on
4. 在 Home Assistant 中添加 Wyoming 集成，指向 `localhost:10300`

## 配置

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `credential_path` | 凭据文件路径 | `/config/credentials.json` |
| `enable_punctuation` | 启用标点符号 | `true` |

## 更新上游仓库

重新构建 Add-on 即可获取 [doubaoime-asr](https://github.com/starccy/doubaoime-asr) 的最新更新。

## 免责声明

本项目基于非官方 API，仅供学习和研究目的，不保证未来的可用性和稳定性。
