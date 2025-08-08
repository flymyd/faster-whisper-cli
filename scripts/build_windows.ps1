Param(
  [string]$Python = "python",
  [switch]$Clean,
  [string]$Arch = "x64",
  [switch]$OneFile,
  [string]$ContentsDir = "_internal"
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
 $opts = @(
  "-y",
  "-n", "fwhisper",
  "--collect-all", "av",
  "--collect-all", "tokenizers",
  "--collect-all", "huggingface_hub",
  "--collect-all", "ctranslate2",
  "--collect-all", "onnxruntime",
  "--add-data", "faster_whisper/assets/*;faster_whisper/assets",
  "--contents-directory", $ContentsDir
 )
 if ($OneFile) {
   $opts += "--onefile"
 }
 & $Python -m PyInstaller @opts faster_whisper/cli.py

if ($OneFile) {
  $outFile = "dist/fwhisper-windows-$Arch.exe"
  if (Test-Path $outFile) { Remove-Item -Force $outFile }
  Move-Item "dist/fwhisper.exe" $outFile
  Write-Host "构建完成（单文件）：$outFile"
} else {
  $outDir = "dist/fwhisper-windows-$Arch"
  if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
  Move-Item dist/fwhisper $outDir
  Write-Host "构建完成（单目录）：$outDir（内部目录：$ContentsDir）"
}


