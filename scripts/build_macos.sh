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

# 选择正确的 Python 解释器（优先使用 Conda/venv），避免误用 Homebrew 系统 Python
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "未找到可用的 Python 解释器，请安装 Python 或通过 PYTHON_BIN 指定路径" >&2
    exit 2
  fi
fi

"${PYTHON_BIN}" -m pip install -U pip setuptools wheel pyinstaller
"${PYTHON_BIN}" -m pip install -e .

# 仅 CPU 目标，收集依赖与 assets
"${PYTHON_BIN}" -m PyInstaller -y -n fwhisper \
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


