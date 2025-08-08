#!/usr/bin/env bash
set -euo pipefail

# macOS 通用打包脚本
# 用法：
#   ARCH=arm64 ./scripts/build_macos.sh
#   ARCH=x86_64 ./scripts/build_macos.sh

ARCH=${ARCH:-}
# 是否打包为单文件（1 为启用），默认关闭：生成包含可执行文件和 `_internal` 目录的单目录结构
ONEFILE=${ONEFILE:-0}
# 自定义 one-folder 模式下的内部目录名称（PyInstaller 默认 `_internal`）
CONTENTS_DIR=${CONTENTS_DIR:-_internal}
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

build_opts=(
  -y
  -n fwhisper
  --collect-all av
  --collect-all tokenizers
  --collect-all huggingface_hub
  --collect-all ctranslate2
  --collect-all onnxruntime
  --add-data "faster_whisper/assets/*:faster_whisper/assets"
  --contents-directory "${CONTENTS_DIR}"
)

if [[ "${ONEFILE}" == "1" ]]; then
  build_opts+=(--onefile)
fi

# 仅 CPU 目标，收集依赖与 assets
"${PYTHON_BIN}" -m PyInstaller "${build_opts[@]}" faster_whisper/cli.py

if [[ "${ONEFILE}" == "1" ]]; then
  # 单文件模式：重命名二进制文件
  OUT_FILE="dist/fwhisper-macos-${ARCH}"
  rm -f "${OUT_FILE}" || true
  mv "dist/fwhisper" "${OUT_FILE}"
  chmod +x "${OUT_FILE}"
  echo "构建完成（单文件）：${OUT_FILE}"
else
  # 单目录模式：重命名产物目录
  OUT_DIR="dist/fwhisper-macos-${ARCH}"
  rm -rf "${OUT_DIR}" || true
  mv dist/fwhisper "${OUT_DIR}"
  echo "构建完成（单目录）：${OUT_DIR}（内部目录：${CONTENTS_DIR}）"
fi


