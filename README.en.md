# Faster-Whisper CLI (CTranslate2-based)

[English](README.en.md) | [中文](README.md)

### About this repository
- This project is a thin CLI wrapper around the upstream repository [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper). It ships a cross-platform command line tool `fwhisper`, defaults to CPU inference, and integrates VAD (voice activity detection).
- The core inference engine, models, and most APIs come from the upstream project.

### Acknowledgements
- Sincere thanks to the SYSTRAN team and all contributors to the upstream project.

### Install (dev/tryout)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e .

# Smoke test (CPU+INT8)
fwhisper tests/data/jfk.flac --model tiny --device cpu --compute-type int8 --output out.srt --format srt
```

### Quick examples
```bash
# Minimal usage: print plain text to stdout
fwhisper input.mp3 --model base --device cpu --compute-type int8

# Generate SRT with word-level timestamps (VAD enabled by default; disable with --no-vad-filter)
fwhisper input.wav --word-timestamps --output out.srt --format srt

# Batched inference
fwhisper input.m4a --batch-size 16 --output out.vtt --format vtt

# Specify language and task
fwhisper input.flac --language en --task transcribe --output out.txt
```

Key options (aligned with `WhisperModel.transcribe` / `BatchedInferencePipeline.transcribe`):
- Model & inference: `--model`, `--device` (default cpu), `--compute-type` (default int8 for CPU), `--cpu-threads`, `--num-workers`, `--batch-size`
- Language & task: `--language`, `--task` (transcribe/translate), `--multilingual`
- Decoding: `--beam-size`, `--best-of`, `--temperature`, `--length-penalty`, `--repetition-penalty`, `--no-repeat-ngram-size`, `--max-new-tokens`
- Timestamps & hotwords: `--word-timestamps`, `--without-timestamps`, `--hotwords`
- VAD: enabled by default; `--no-vad-filter` to disable; `--vad-params '{...}'` to customize
- Output: `--output`, `--format` (txt/srt/vtt/jsonl)

Notes:
- CLI defaults to CPU inference (you may pass `--device cuda`, but this repo focuses on CPU builds).
- Model weights are not bundled. They will be downloaded from the Hugging Face Hub on first use.

