import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from faster_whisper import (
    BatchedInferencePipeline,
    WhisperModel,
)
from faster_whisper.transcribe import Segment
from faster_whisper.utils import format_timestamp


def _infer_format_from_path(output_path: Optional[str]) -> str:
    if not output_path:
        return "txt"
    ext = os.path.splitext(output_path)[1].lower()
    if ext in {".srt"}:
        return "srt"
    if ext in {".vtt"}:
        return "vtt"
    if ext in {".jsonl"}:
        return "jsonl"
    return "txt"


def _to_srt_timestamp(seconds: float) -> str:
    ts = format_timestamp(seconds, always_include_hours=True, decimal_marker=",")
    return ts


def _to_vtt_timestamp(seconds: float) -> str:
    ts = format_timestamp(seconds, always_include_hours=True, decimal_marker=".")
    return ts


def _write_txt(segments: Iterable[Segment], fp):
    for seg in segments:
        if seg.text:
            fp.write(seg.text.strip() + "\n")


def _write_srt(segments: Iterable[Segment], fp):
    index = 1
    for seg in segments:
        if not seg.text:
            continue
        start = _to_srt_timestamp(seg.start)
        end = _to_srt_timestamp(seg.end)
        fp.write(f"{index}\n{start} --> {end}\n{seg.text.strip()}\n\n")
        index += 1


def _write_vtt(segments: Iterable[Segment], fp):
    fp.write("WEBVTT\n\n")
    for seg in segments:
        if not seg.text:
            continue
        start = _to_vtt_timestamp(seg.start)
        end = _to_vtt_timestamp(seg.end)
        fp.write(f"{start} --> {end}\n{seg.text.strip()}\n\n")


def _segment_to_dict(seg: Segment) -> dict:
    data = {
        "id": seg.id,
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "avg_logprob": seg.avg_logprob,
        "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob,
    }
    if seg.words is not None:
        data["words"] = [
            {
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "probability": w.probability,
            }
            for w in seg.words
        ]
    return data


def _write_jsonl(segments: Iterable[Segment], fp):
    for seg in segments:
        if not seg.text:
            continue
        fp.write(json.dumps(_segment_to_dict(seg), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faster-Whisper CLI (CPU 默认), 将音频转写为文本/字幕。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument(
        "--output",
        "-o",
        help="输出文件路径（缺省则输出到标准输出）。根据扩展名自动推断格式。",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt", "jsonl"],
        help="显式指定输出格式（覆盖扩展名推断）",
    )
    # 模型与设备
    parser.add_argument("--model", default="base", help="模型大小或本地路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="计算设备")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="计算精度（CPU 推荐 int8，详见 CTranslate2 文档）",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU 线程数，0 表示不覆盖")
    parser.add_argument("--num-workers", type=int, default=1, help="并行 worker 数（多线程转写时生效）")
    # 解码与控制
    parser.add_argument("--language", help="显式指定语言代码，例如 zh/en 等")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="任务类型")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--without-timestamps", action="store_true", help="仅输出文本 token，不含时间戳")
    parser.add_argument("--word-timestamps", action="store_true", help="输出词级时间戳")
    parser.add_argument("--multilingual", action="store_true", help="逐段语言检测（多语模型）")
    parser.add_argument("--hotwords", help="热词提示，提升特定词识别概率")
    # VAD（默认启用，可用 --no-vad-filter 关闭）
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument(
        "--vad-filter",
        dest="vad_filter",
        action="store_true",
        default=True,
        help="启用 VAD 过滤静音/无语音片段（默认启用）",
    )
    vad_group.add_argument(
        "--no-vad-filter",
        dest="vad_filter",
        action="store_false",
        help="关闭 VAD 过滤",
    )
    parser.add_argument(
        "--vad-params",
        help="VAD 参数，JSON 字符串，例如 '{\"min_silence_duration_ms\": 500}'",
    )
    # 批量/速度
    parser.add_argument("--batch-size", type=int, default=8, help="批量推理大小（>1 使用批量管线）")
    parser.add_argument("--chunk-length", type=int, help="音频分块长度（秒）")
    # 其他
    parser.add_argument("--log-progress", action="store_true")
    parser.add_argument(
        "--no-progress-events",
        action="store_true",
        help="关闭进度事件输出（默认输出 JSON 行到标准错误）",
    )
    return parser


def _open_output(path: Optional[str]):
    if not path or path == "-":
        return sys.stdout
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return open(path, "w", encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    output_format = args.format or _infer_format_from_path(args.output)

    # 仅 CPU 为默认；允许用户显式选择 auto/cuda，但当前构建主要面向 CPU。
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )

    use_batched = args.batch_size and args.batch_size > 1
    pipeline = BatchedInferencePipeline(model) if use_batched else None

    vad_params = None
    if args.vad_params:
        try:
            vad_params = json.loads(args.vad_params)
        except json.JSONDecodeError as e:
            print(f"VAD 参数解析失败: {e}", file=sys.stderr)
            return 2

    common_kwargs = dict(
        language=args.language,
        task=args.task,
        log_progress=args.log_progress,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=1.0,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        without_timestamps=args.without_timestamps,
        word_timestamps=args.word_timestamps,
        multilingual=args.multilingual,
        vad_filter=args.vad_filter,
        vad_parameters=vad_params,
        chunk_length=args.chunk_length,
        hotwords=args.hotwords,
    )

    if pipeline is not None:
        segments, info = pipeline.transcribe(
            args.input,
            batch_size=args.batch_size,
            **common_kwargs,
        )
    else:
        segments, info = model.transcribe(
            args.input,
            **common_kwargs,
        )

    # 进度事件：开始
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    start_ts = time.time()
    if not args.no_progress_events:
        print(
            json.dumps(
                {
                    "event": "start",
                    "time": _now_iso(),
                    "input": args.input,
                    "model": args.model,
                    "device": args.device,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    total_seconds = getattr(info, "duration", None) or 0.0
    # 若存在有效原始总时长，使用它进行进度估计
    if total_seconds and total_seconds > 0 and isinstance(total_seconds, (int, float)):
        pass
    else:
        total_seconds = None

    with _open_output(args.output) as fp:
        # 写入头部（VTT）
        if output_format == "vtt":
            fp.write("WEBVTT\n\n")

        srt_index = 1
        last_progress_time = 0.0

        for seg in segments:
            # 写出分段
            if output_format == "txt":
                if seg.text:
                    fp.write(seg.text.strip() + "\n")
            elif output_format == "jsonl":
                if seg.text:
                    fp.write(json.dumps(_segment_to_dict(seg), ensure_ascii=False) + "\n")
            elif output_format == "srt":
                if seg.text:
                    fp.write(
                        f"{srt_index}\n{_to_srt_timestamp(seg.start)} --> {_to_srt_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
                    )
                    srt_index += 1
            elif output_format == "vtt":
                if seg.text:
                    fp.write(
                        f"{_to_vtt_timestamp(seg.start)} --> {_to_vtt_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
                    )

            # 进度事件：按段更新
            if not args.no_progress_events:
                processed_seconds = seg.end if isinstance(seg.end, (int, float)) else last_progress_time
                if total_seconds:
                    # 防回退
                    processed_seconds = max(last_progress_time, min(processed_seconds, total_seconds))
                else:
                    processed_seconds = max(last_progress_time, processed_seconds)
                last_progress_time = processed_seconds
                elapsed = time.time() - start_ts
                payload = {
                    "event": "progress",
                    "time": _now_iso(),
                    "elapsed_seconds": round(elapsed, 3),
                    "processed_seconds": round(processed_seconds, 3),
                }
                if total_seconds:
                    payload.update(
                        {
                            "total_seconds": round(total_seconds, 3),
                            "progress": round(min(processed_seconds / total_seconds, 1.0), 4),
                        }
                    )
                print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    # 进度事件：结束
    if not args.no_progress_events:
        end_ts = time.time()
        print(
            json.dumps(
                {
                    "event": "end",
                    "time": _now_iso(),
                    "elapsed_seconds": round(end_ts - start_ts, 3),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    return 0


if __name__ == "__main__":
    # 兼容 PyInstaller / macOS spawn：防止子进程重复进入 argparse 主逻辑
    try:
        import multiprocessing as _mp  # noqa: WPS433 (allowed in main guard)

        _mp.freeze_support()
    except Exception:
        pass
    raise SystemExit(main())


