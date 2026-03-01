from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import zstandard as zstd

from minicrunch.backends import load_prior
from minicrunch.codec import compress_text, decompress_archive, unpack_archive
from minicrunch.wiki import fetch_wikipedia_extract


DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
BACKEND_CHOICES = ["transformers", "llamacpp", "vllm"]


def _ratio_percent(size: int, original: int) -> float:
    if original == 0:
        return 0.0
    return (size / original) * 100.0


def _speed(tokens: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return tokens / elapsed


def _progress(stage: str, done: int, total: int) -> None:
    print(f"[{stage}] {done}/{total} tokens", flush=True)


def _read_text(args: argparse.Namespace) -> str:
    if args.input:
        return Path(args.input).read_text(encoding="utf-8")
    if args.wikipedia_title:
        return fetch_wikipedia_extract(args.wikipedia_title)
    raise ValueError("Provide --input or --wikipedia-title")


def _load_prior_from_args(args: argparse.Namespace, model_id: str, backend: str, dtype: str):
    return load_prior(
        model_id=model_id,
        backend=backend,
        device=args.device,
        dtype=dtype,
        llama_n_ctx=args.llama_n_ctx,
        llama_n_batch=args.llama_n_batch,
        llama_n_threads=args.llama_n_threads,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
        vllm_url=args.vllm_url,
        vllm_top_k=args.vllm_top_k,
        vllm_timeout_seconds=args.vllm_timeout_seconds,
        vllm_fallback_logit=args.vllm_fallback_logit,
    )


def cmd_compress(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")

    prior = _load_prior_from_args(
        args=args,
        model_id=args.model_id,
        backend=args.backend,
        dtype=args.dtype,
    )
    result = compress_text(
        text=text,
        prior=prior,
        total_freq=args.total_freq,
        progress_every=args.progress_every,
        progress_callback=_progress if args.progress_every > 0 else None,
    )

    output_path = Path(args.output)
    output_path.write_bytes(result.archive)

    source_bytes = len(text.encode("utf-8"))
    archive_bytes = len(result.archive)
    payload_bytes = math.ceil(result.payload_bits / 8)

    print(f"Wrote archive: {output_path}")
    print(f"Model: {prior.model_id}")
    print(f"Tokens: {result.token_count}")
    print(f"Payload bits: {result.payload_bits}")
    print(
        f"Payload bytes: {payload_bytes} ({_ratio_percent(payload_bytes, source_bytes):.2f}% of original)"
    )
    print(
        f"Archive bytes: {archive_bytes} ({_ratio_percent(archive_bytes, source_bytes):.2f}% of original)"
    )
    print(f"Compress time: {result.elapsed_seconds:.2f}s")
    print(f"Compress speed: {_speed(result.token_count, result.elapsed_seconds):.2f} tok/s")
    return 0


def cmd_decompress(args: argparse.Namespace) -> int:
    archive_path = Path(args.input)
    archive = archive_path.read_bytes()
    header, _ = unpack_archive(archive)

    model_id = args.model_id or header["model_id"]
    backend = args.backend or header.get("backend", "transformers")
    dtype = args.dtype or header.get("dtype", "auto")

    prior = _load_prior_from_args(
        args=args,
        model_id=model_id,
        backend=backend,
        dtype=dtype,
    )
    result = decompress_archive(
        archive=archive,
        prior=prior,
        progress_every=args.progress_every,
        progress_callback=_progress if args.progress_every > 0 else None,
        verify_hash=not args.no_verify,
    )

    output_path = Path(args.output)
    output_path.write_text(result.text, encoding="utf-8")

    print(f"Wrote text: {output_path}")
    print(f"Model: {prior.model_id}")
    print(f"Tokens: {result.token_count}")
    print(f"Decompress time: {result.elapsed_seconds:.2f}s")
    print(f"Decompress speed: {_speed(result.token_count, result.elapsed_seconds):.2f} tok/s")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    text = _read_text(args)
    source_bytes = text.encode("utf-8")

    gzip_bytes = gzip.compress(source_bytes, compresslevel=9)
    zstd_bytes = zstd.ZstdCompressor(level=19).compress(source_bytes)

    prior = _load_prior_from_args(
        args=args,
        model_id=args.model_id,
        backend=args.backend,
        dtype=args.dtype,
    )

    compress_result = compress_text(
        text=text,
        prior=prior,
        total_freq=args.total_freq,
        progress_every=args.progress_every,
        progress_callback=_progress if args.progress_every > 0 else None,
    )
    decompress_result = decompress_archive(
        archive=compress_result.archive,
        prior=prior,
        progress_every=args.progress_every,
        progress_callback=_progress if args.progress_every > 0 else None,
        verify_hash=True,
    )

    payload_bytes = math.ceil(compress_result.payload_bits / 8)
    archive_bytes = len(compress_result.archive)

    roundtrip_ok = decompress_result.text == text
    if not roundtrip_ok:
        raise RuntimeError("Decoded text does not match source")

    if args.output_archive:
        Path(args.output_archive).write_bytes(compress_result.archive)
    if args.output_decoded:
        Path(args.output_decoded).write_text(decompress_result.text, encoding="utf-8")

    print("Benchmark results")
    print(f"Source bytes: {len(source_bytes)}")
    print(
        f"gzip -9: {len(gzip_bytes)} bytes ({_ratio_percent(len(gzip_bytes), len(source_bytes)):.2f}% of source)"
    )
    print(
        f"zstd -19: {len(zstd_bytes)} bytes ({_ratio_percent(len(zstd_bytes), len(source_bytes)):.2f}% of source)"
    )
    print(
        f"LLM payload: {payload_bytes} bytes / {compress_result.payload_bits} bits "
        f"({_ratio_percent(payload_bytes, len(source_bytes)):.2f}% of source)"
    )
    print(
        f"LLM archive (with header): {archive_bytes} bytes "
        f"({_ratio_percent(archive_bytes, len(source_bytes)):.2f}% of source)"
    )
    print(f"Roundtrip exact: {'PASS' if roundtrip_ok else 'FAIL'}")
    print(
        f"Compress: {compress_result.elapsed_seconds:.2f}s "
        f"({_speed(compress_result.token_count, compress_result.elapsed_seconds):.2f} tok/s)"
    )
    print(
        f"Decompress: {decompress_result.elapsed_seconds:.2f}s "
        f"({_speed(decompress_result.token_count, decompress_result.elapsed_seconds):.2f} tok/s)"
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minicrunch",
        description="LLM arithmetic-coding demo with mistralai/Ministral-3-3B-Instruct-2512",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_args(subparser: argparse.ArgumentParser, with_default: bool = True) -> None:
        if with_default:
            subparser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
            subparser.add_argument("--backend", default="transformers", choices=BACKEND_CHOICES)
            subparser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
        else:
            subparser.add_argument("--model-id")
            subparser.add_argument("--backend", choices=BACKEND_CHOICES)
            subparser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
        subparser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    def add_llama_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--llama-n-ctx",
            type=int,
            default=8192,
            help="Context window for llama.cpp backend",
        )
        subparser.add_argument(
            "--llama-n-batch",
            type=int,
            default=512,
            help="Batch size for llama.cpp decode calls",
        )
        subparser.add_argument(
            "--llama-n-threads",
            type=int,
            default=0,
            help="CPU threads for llama.cpp (0 = auto)",
        )
        subparser.add_argument(
            "--llama-n-gpu-layers",
            type=int,
            default=None,
            help="GPU layers for llama.cpp (default: auto by device)",
        )

    def add_vllm_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--vllm-url",
            help="Base URL for remote vLLM logits server (for backend=vllm)",
        )
        subparser.add_argument(
            "--vllm-top-k",
            type=int,
            default=256,
            help="Top-K logprobs requested from vLLM server for backend=vllm",
        )
        subparser.add_argument(
            "--vllm-timeout-seconds",
            type=float,
            default=60.0,
            help="HTTP timeout per vLLM request",
        )
        subparser.add_argument(
            "--vllm-fallback-logit",
            type=float,
            default=-50.0,
            help="Logit assigned to tokens outside vLLM top-k set",
        )

    compress_parser = subparsers.add_parser("compress", help="Compress a UTF-8 text file")
    compress_parser.add_argument("--input", required=True, help="Input UTF-8 text file")
    compress_parser.add_argument("--output", required=True, help="Output archive path")
    compress_parser.add_argument("--total-freq", type=int, default=1 << 20)
    compress_parser.add_argument("--progress-every", type=int, default=100)
    add_model_args(compress_parser, with_default=True)
    add_llama_args(compress_parser)
    add_vllm_args(compress_parser)
    compress_parser.set_defaults(func=cmd_compress)

    decompress_parser = subparsers.add_parser("decompress", help="Decompress an archive")
    decompress_parser.add_argument("--input", required=True, help="Input archive path")
    decompress_parser.add_argument("--output", required=True, help="Output UTF-8 text file")
    decompress_parser.add_argument("--progress-every", type=int, default=100)
    decompress_parser.add_argument("--no-verify", action="store_true")
    add_model_args(decompress_parser, with_default=False)
    add_llama_args(decompress_parser)
    add_vllm_args(decompress_parser)
    decompress_parser.set_defaults(func=cmd_decompress)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run compression benchmark")
    benchmark_group = benchmark_parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument("--input", help="Input UTF-8 text file")
    benchmark_group.add_argument("--wikipedia-title", help="Wikipedia page title")
    benchmark_parser.add_argument("--total-freq", type=int, default=1 << 20)
    benchmark_parser.add_argument("--progress-every", type=int, default=100)
    benchmark_parser.add_argument("--output-archive", help="Optional archive output path")
    benchmark_parser.add_argument("--output-decoded", help="Optional decoded text output path")
    add_model_args(benchmark_parser, with_default=True)
    add_llama_args(benchmark_parser)
    add_vllm_args(benchmark_parser)
    benchmark_parser.set_defaults(func=cmd_benchmark)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
