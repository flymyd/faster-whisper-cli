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
    # 日志控制
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="日志级别（debug 最详细）",
    )
    logs_group = parser.add_mutually_exclusive_group()
    logs_group.add_argument(
        "--logs",
        dest="logs",
        action="store_true",
        default=True,
        help="启用日志输出（默认启用）",
    )
    logs_group.add_argument(
        "--no-logs",
        dest="logs",
        action="store_false",
        help="关闭所有日志输出（包括开始/进度/结束事件）",
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

    # 进度事件：尽早输出开始，避免用户在模型加载/预处理期间无反馈
    def _now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    level_order = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    threshold = level_order.get(getattr(args, "log_level", "info"), 20)
    emit_logs = bool(getattr(args, "logs", True))
    emit_events = emit_logs and not getattr(args, "no_progress_events", False)

    def _emit_log(level: str, message: str, data: Optional[dict] = None):
        if not emit_logs:
            return
        if level_order.get(level, 999) < threshold:
            return
        payload = {"event": "log", "level": level, "time": _now_iso(), "message": message}
        if data:
            payload["data"] = data
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    def _emit_event(payload: dict):
        if not emit_events:
            return
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    start_ts = time.time()
    _emit_event(
        {
            "event": "start",
            "time": _now_iso(),
            "input": args.input,
            "model": args.model,
            "device": args.device,
        }
    )
    _emit_log(
        "info",
        "运行参数",
        {
            "input": args.input,
            "output": args.output,
            "output_format": output_format,
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "cpu_threads": args.cpu_threads,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "chunk_length": args.chunk_length,
            "vad_filter": args.vad_filter,
            "vad_params": bool(args.vad_params),
            "word_timestamps": args.word_timestamps,
            "without_timestamps": args.without_timestamps,
            "multilingual": args.multilingual,
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "temperature": args.temperature,
        },
    )

    # 仅 CPU 为默认；允许用户显式选择 auto/cuda，但当前构建主要面向 CPU。
    t0 = time.time()
    _emit_log(
        "info",
        "初始化模型",
        {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
        },
    )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )
    _emit_log("debug", "模型初始化完成", {"took_seconds": round(time.time() - t0, 3)})

    use_batched = args.batch_size and args.batch_size > 1
    pipeline = None
    if use_batched:
        t1 = time.time()
        _emit_log("info", "构建批量推理管线", {"batch_size": args.batch_size})
        pipeline = BatchedInferencePipeline(model)
        _emit_log("debug", "批量管线构建完成", {"took_seconds": round(time.time() - t1, 3)})

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

    _emit_log("info", "开始转写", {"batched": bool(pipeline), "batch_size": args.batch_size or 1})
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
    _emit_log(
        "debug",
        "转写调用返回（开始流式读取分段）",
        {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        },
    )

    # 已在转写前输出 start 事件

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
        seg_count = 0

        for seg in segments:
            seg_count += 1
            _emit_log(
                "debug",
                "分段就绪",
                {
                    "id": getattr(seg, "id", seg_count),
                    "start": getattr(seg, "start", None),
                    "end": getattr(seg, "end", None),
                    "has_text": bool(getattr(seg, "text", "")),
                },
            )
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
    end_ts = time.time()
    _emit_event(
        {
            "event": "end",
            "time": _now_iso(),
            "elapsed_seconds": round(end_ts - start_ts, 3),
        }
    )
    _emit_log(
        "info",
        "转写完成",
        {"elapsed_seconds": round(end_ts - start_ts, 3), "segments": seg_count},
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


