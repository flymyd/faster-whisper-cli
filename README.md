# Faster-Whisper CLI（基于 CTranslate2）

[English](README.en.md) | [中文](README.md)

### 本仓库说明与致谢
- 本项目是对上游 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的 CLI 包装，提供跨平台命令行工具 `fwhisper`，默认以 CPU 推理运行，并集成 VAD 语音活动检测。
- 核心推理引擎、模型与大部分 API 来自上游项目；特此致谢 SYSTRAN 团队与所有贡献者。

### Faster Whisper模型列表

| 模型名称              | 完整名称                                     | 文件大小  |
| --------------------- | -------------------------------------------- | --------- |
| **tiny**              | Systran/faster-whisper-tiny                  | 72.04 MB  |
| **base**              | Systran/faster-whisper-base                  | 138.49 MB |
| **small**             | Systran/faster-whisper-small                 | 461.15 MB |
| **medium**            | Systran/faster-whisper-medium                | 1.42 GB   |
| **large-v1**          | Systran/faster-whisper-large-v1              | 2.87 GB   |
| **large-v2**          | Systran/faster-whisper-large-v2              | 2.87 GB   |
| **large-v3**          | Systran/faster-whisper-large-v3              | 2.88 GB   |
| **large**             | Systran/faster-whisper-large-v3              | 2.88 GB   |
| **distil-large-v2**   | Systran/faster-distil-whisper-large-v2       | 1.41 GB   |
| **distil-large-v3**   | Systran/faster-distil-whisper-large-v3       | 1.41 GB   |
| **distil-large-v3.5** | distil-whisper/distil-large-v3.5-ct2         | 1.41 GB   |
| **large-v3-turbo**    | mobiuslabsgmbh/faster-whisper-large-v3-turbo | 1.51 GB   |
| **turbo**             | mobiuslabsgmbh/faster-whisper-large-v3-turbo | 1.51 GB   |

我们提供了跨平台 CLI 工具 `fwhisper`，默认以 CPU 推理运行，且默认开启 VAD（语音活动检测），可将音频转写为 txt/srt/vtt/jsonl。

### 使用示例

```bash
fwhisper ./jfk.flac --model tiny --device cpu --compute-type int8 --output ./out.srt --format srt
```

### 常用示例

```bash
# 最简用法：输出到 stdout（txt）
fwhisper input.mp3 --model base --device cpu --compute-type int8

# 生成 SRT/带词级时间戳（默认启用 VAD；如需关闭可加 --no-vad-filter）
fwhisper input.wav --word-timestamps --output out.srt --format srt

# 批量推理
fwhisper input.m4a --batch-size 16 --output out.vtt --format vtt

# 指定语言/任务
fwhisper input.flac --language zh --task transcribe --output out.txt
```

关键参数（与 `WhisperModel.transcribe`/`BatchedInferencePipeline.transcribe` 保持一致）：
- 模型与推理：`--model`、`--device`（默认 cpu）、`--compute-type`（默认 int8，CPU 推荐）、`--cpu-threads`、`--num-workers`、`--batch-size`
- 语言与任务：`--language`、`--task`（transcribe/translate）、`--multilingual`
- 生成控制：`--beam-size`、`--best-of`、`--temperature`、`--length-penalty`、`--repetition-penalty`、`--no-repeat-ngram-size`、`--max-new-tokens`
- 时间戳与热词：`--word-timestamps`、`--without-timestamps`、`--hotwords`
- VAD：默认开启；`--no-vad-filter` 可关闭；`--vad-params '{...}'` 可自定义参数
- 输出：`--output`、`--format`（txt/srt/vtt/jsonl）

说明：
- CLI 默认 CPU 推理（可显式指定 `--device cuda`，但本仓库提供的构建产物面向 CPU）。
- 模型权重不内置，首次运行会自动从 Hugging Face Hub 下载到缓存。

### 参数说明（与 API 一致）

- 模型与设备
  - **--model**: 模型大小或本地 CT2 模型目录（如 `tiny`、`base`、`large-v3`）。
  - **--device**: 计算设备，默认 `cpu`；可选 `auto`、`cuda`。
  - **--compute-type**: 计算精度（默认 `int8`，CPU 推荐）。更多取值参见 CTranslate2 文档。
  - **--cpu-threads**: CPU 线程数，0 表示不覆盖（由后端自行决定）。
  - **--num-workers**: 并行 workers（多线程转写时可提升吞吐，增内存占用）。

- 语言与任务
  - **--language**: 显式指定语言代码（如 `zh`、`en`）；不指定则自动检测（多语模型）。
  - **--task**: `transcribe`（转写）或 `translate`（翻译为英文）。
  - **--multilingual**: 对每个片段做语言检测（多语场景下提升鲁棒性）。

- 生成控制
  - **--beam-size**: Beam 搜索宽度，默认 5。
  - **--best-of**: 采样温度>0时的候选数，默认 5。
  - **--temperature**: 采样温度，默认 0.0（贪心/beam）。
  - **--length-penalty**: 长度惩罚（>1 更偏短）。
  - **--repetition-penalty**: 重复惩罚，默认 1。
  - **--no-repeat-ngram-size**: 禁止重复 n-gram，默认 0 关闭。
  - **--max-new-tokens**: 每段最大新生成 token 数（超长内容可限制泄洪）。
  - **--without-timestamps**: 仅文本 token（不含时间戳），默认关闭。
  - **--word-timestamps**: 词级时间戳，默认关闭（开启更耗时）。
  - **--hotwords**: 热词提示，用于提升特定词命中率。

- VAD（语音活动检测）
  - 默认开启；关闭请使用 **--no-vad-filter**。
  - **--vad-params**: 传入 JSON 字符串自定义参数（例：`'{"min_silence_duration_ms": 500}'`）。

- 性能/分片
  - **--batch-size**: 批量推理大小；>1 时启用批量管线（高并行，提速但占内存）。
  - **--chunk-length**: 输入音频分块长度（秒），默认由特征提取器决定。

- 输出与日志
  - **--output**: 输出路径；不指定则输出到标准输出。
  - **--format**: 输出格式，`txt|srt|vtt|jsonl`；未显式指定时根据 `--output` 扩展名推断。
  - **--log-progress**: 显示进度条。

输入文件说明
- 输入既可为音频也可为视频文件（通过 PyAV/FFmpeg 读取第一个音轨），常见容器与编码如 mp4/mov/mkv/webm/avi/flv/m4a/wav/flac/mp3/ogg、AAC/PCM/MP3/Opus/Vorbis/FLAC 等。部分加密/专有封装可能不支持。

### 常见场景示例

- 基础转写（txt 输出到 stdout）
  ```bash
  fwhisper input.mp3 --model base --device cpu --compute-type int8
  ```

- 生成字幕 SRT（默认启用 VAD）
  ```bash
  fwhisper input.wav --output out.srt --format srt
  ```

- 带词级时间戳字幕（更精细的对齐）
  ```bash
  fwhisper input.wav --word-timestamps --output out.vtt --format vtt
  ```

- 指定语言为中文、仅输出纯文本
  ```bash
  fwhisper input.m4a --language zh --without-timestamps --output out.txt --format txt
  ```

- 翻译为英文字幕
  ```bash
  fwhisper input.mov --task translate --output out_en.srt --format srt
  ```

- 批量推理（提升吞吐）
  ```bash
  fwhisper input.flac --batch-size 16 --output out.srt --format srt
  ```

- 自定义 VAD 参数（缩短静音判定，减小黏连）
  ```bash
  fwhisper input.wav --vad-params '{"min_silence_duration_ms": 500, "speech_pad_ms": 200}' \
    --output out.srt --format srt
  ```

- 关闭 VAD（极短音频或排查问题时使用）
  ```bash
  fwhisper input.mp4 --no-vad-filter --output out.txt --format txt
  ```

- 热词提示（提升专有名词命中率）
  ```bash
  fwhisper input.wav --hotwords "ComfyUI, SYSTRAN" --output out.srt --format srt
  ```

- 性能调参（CPU 线程与 workers）
  ```bash
  fwhisper input.mp3 --cpu-threads 8 --num-workers 2 --output out.srt --format srt
  ```

- 处理视频文件（自动抽取音轨）
  ```bash
  fwhisper movie.mp4 --word-timestamps --output subs.srt --format srt
  ```

- 批处理目录（示例：macOS/Linux）
  ```bash
  find ./inputs -type f \( -name "*.mp3" -o -name "*.mp4" -o -name "*.wav" \) -print0 \
    | xargs -0 -I{} sh -c 'fwhisper "$1" --output "${1%.*}.srt" --format srt' sh {}
  ```

### 本地构建方法（CPU 版）

要求：Python 3.9+，Git。建议在对应目标平台本机构建（PyInstaller 不支持跨编译）。

1) macOS（Apple Silicon / Intel）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel pyinstaller
pip install -e .

# Apple Silicon 架构
ARCH=arm64 ONEFILE=1 ./scripts/build_macos.sh
# Intel 架构
ARCH=x86_64 ONEFILE=1 ./scripts/build_macos.sh

# 产物目录：
# - dist/fwhisper-macos-arm64
# - dist/fwhisper-macos-x86_64
```

2) Windows x64（PowerShell）

```powershell
python -m venv .venv
.\.venv\\Scripts\Activate.ps1
pip install -U pip setuptools wheel pyinstaller
pip install -e .
# .\scripts\build_windows.ps1 -Python python
powershell -File scripts/build_windows.ps1 -Arch x64 -OneFile
# 产物目录：
# - dist/fwhisper-windows-x64
```

运行产物示例：

```bash
# 输出字幕（默认启用 VAD、可词级时间戳）：
./dist/fwhisper-macos-arm64/fwhisper input.mp3 --model tiny --device cpu --compute-type int8 --output out.srt --format srt
fwhisper movie.mp4 --word-timestamps --output subs.srt --format srt
# 关闭VAD
fwhisper input.wav --no-vad-filter --output out.txt
```

注意：
- 若启用 `--vad-filter`，需保证 onnxruntime 已正确收集（本仓库脚本已包含 `--collect-all onnxruntime`）。
- 不内置模型权重，首次运行会联网下载。
