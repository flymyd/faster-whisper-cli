Param(
  [string]$Python = "python",
  [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Clean) {
  if (Test-Path dist) { Remove-Item -Recurse -Force dist }
  if (Test-Path build) { Remove-Item -Recurse -Force build }
}

& $Python -m pip install -U pip setuptools wheel pyinstaller | Out-Null
& $Python -m pip install -e . | Out-Null

# 仅 CPU 目标，收集依赖与 assets
pyinstaller -y -n fwhisper `
  --collect-all av `
  --collect-all tokenizers `
  --collect-all huggingface_hub `
  --collect-all ctranslate2 `
  --collect-all onnxruntime `
  --add-data "faster_whisper/assets/*;faster_whisper/assets" `
  faster_whisper/cli.py

$outDir = "dist/fwhisper-windows-x64"
if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
Move-Item dist/fwhisper $outDir
Write-Host "构建完成：$outDir"


