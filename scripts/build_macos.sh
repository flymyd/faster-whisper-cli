#!/usr/bin/env bash
set -euo pipefail

# macOS 通用打包脚本
# 用法：
#   ARCH=arm64 ./scripts/build_macos.sh
#   ARCH=x86_64 ./scripts/build_macos.sh

ARCH=${ARCH:-}
if [[ -z "${ARCH}" ]]; then
  echo "请通过环境变量 ARCH 指定架构：arm64 或 x86_64" >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python3}

${PYTHON_BIN} -m pip install -U pip setuptools wheel pyinstaller || true
${PYTHON_BIN} -m pip install -e .

# 仅 CPU 目标，收集依赖与 assets
pyinstaller -y -n fwhisper \
  --collect-all av \
  --collect-all tokenizers \
  --collect-all huggingface_hub \
  --collect-all ctranslate2 \
  --collect-all onnxruntime \
  --add-data "faster_whisper/assets/*:faster_whisper/assets" \
  faster_whisper/cli.py

# 重命名产物目录，带上架构
OUT_DIR="dist/fwhisper-macos-${ARCH}"
rm -rf "${OUT_DIR}" || true
mv dist/fwhisper "${OUT_DIR}"
echo "构建完成：${OUT_DIR}"


