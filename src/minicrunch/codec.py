from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from typing import Callable

from minicrunch.arithmetic import ArithmeticDecoder, ArithmeticEncoder, BitInput
from minicrunch.backends import PriorModel
from minicrunch.distributions import logits_to_cumulative


MAGIC = b"MCZ1"


@dataclass
class CompressResult:
    archive: bytes
    payload_bits: int
    token_count: int
    elapsed_seconds: float
    header: dict


@dataclass
class DecompressResult:
    text: str
    token_count: int
    elapsed_seconds: float
    header: dict


def pack_archive(header: dict, payload: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload


def unpack_archive(archive: bytes) -> tuple[dict, bytes]:
    if len(archive) < 8:
        raise ValueError("Archive too small")
    if archive[:4] != MAGIC:
        raise ValueError("Invalid archive magic")

    header_len = struct.unpack("<I", archive[4:8])[0]
    header_start = 8
    header_end = header_start + header_len
    if header_end > len(archive):
        raise ValueError("Corrupt archive: header length exceeds file size")

    header = json.loads(archive[header_start:header_end].decode("utf-8"))
    payload = archive[header_end:]
    return header, payload


def compress_text(
    text: str,
    prior: PriorModel,
    total_freq: int = 1 << 20,
    progress_every: int = 0,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> CompressResult:
    source_bytes = text.encode("utf-8")
    token_ids = prior.encode_text(text)

    prior.reset()
    encoder = ArithmeticEncoder()
    start = time.perf_counter()

    for index, token_id in enumerate(token_ids, start=1):
        logits = prior.next_logits()
        cumulative = logits_to_cumulative(logits, total_freq=total_freq)
        encoder.encode_symbol(
            cum_low=int(cumulative[token_id]),
            cum_high=int(cumulative[token_id + 1]),
            total=total_freq,
        )
        prior.accept_token(token_id)

        if progress_every and index % progress_every == 0 and progress_callback:
            progress_callback("compress", index, len(token_ids))

    encoded = encoder.finish()

    header = {
        "version": 1,
        "backend": prior.backend,
        "model_id": prior.model_id,
        "dtype": prior.dtype_name,
        "total_freq": total_freq,
        "token_count": len(token_ids),
        "bit_length": encoded.bit_length,
        "original_bytes": len(source_bytes),
        "sha256_utf8": hashlib.sha256(source_bytes).hexdigest(),
    }
    archive = pack_archive(header=header, payload=encoded.payload)
    elapsed = time.perf_counter() - start

    return CompressResult(
        archive=archive,
        payload_bits=encoded.bit_length,
        token_count=len(token_ids),
        elapsed_seconds=elapsed,
        header=header,
    )


def decompress_archive(
    archive: bytes,
    prior: PriorModel,
    progress_every: int = 0,
    progress_callback: Callable[[str, int, int], None] | None = None,
    verify_hash: bool = True,
) -> DecompressResult:
    header, payload = unpack_archive(archive)

    if header.get("backend") != prior.backend:
        raise ValueError(
            f"Archive backend {header.get('backend')} != loaded backend {prior.backend}"
        )
    if header.get("model_id") != prior.model_id:
        raise ValueError(
            f"Archive model {header.get('model_id')} != loaded model {prior.model_id}"
        )

    total_freq = int(header["total_freq"])
    token_count = int(header["token_count"])
    bit_length = int(header["bit_length"])

    prior.reset()
    bit_input = BitInput(payload=payload, bit_length=bit_length)
    decoder = ArithmeticDecoder(bit_input)

    start = time.perf_counter()
    token_ids: list[int] = []
    for index in range(1, token_count + 1):
        logits = prior.next_logits()
        cumulative = logits_to_cumulative(logits, total_freq=total_freq)
        token_id = decoder.decode_symbol(cumulative, total=total_freq)
        token_ids.append(token_id)
        prior.accept_token(token_id)

        if progress_every and index % progress_every == 0 and progress_callback:
            progress_callback("decompress", index, token_count)

    text = prior.decode_tokens(token_ids)
    elapsed = time.perf_counter() - start

    if verify_hash:
        expected = header.get("sha256_utf8")
        observed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected and observed != expected:
            raise ValueError(
                "Hash mismatch after decode. Compression/decompression priors were not identical."
            )

    return DecompressResult(
        text=text,
        token_count=token_count,
        elapsed_seconds=elapsed,
        header=header,
    )
