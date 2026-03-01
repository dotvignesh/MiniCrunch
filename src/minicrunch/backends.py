from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol
import warnings

import numpy as np
import torch


class PriorModel(Protocol):
    backend: str
    model_id: str
    dtype_name: str
    bos_token_id: int
    vocab_size: int

    def reset(self) -> None:
        ...

    def next_logits(self) -> torch.Tensor:
        ...

    def accept_token(self, token_id: int) -> None:
        ...

    def encode_text(self, text: str) -> list[int]:
        ...

    def decode_tokens(self, token_ids: list[int]) -> str:
        ...


@dataclass
class LoadConfig:
    model_id: str
    vllm_url: str
    vllm_top_k: int = 256
    vllm_timeout_seconds: float = 60.0
    vllm_fallback_logit: float = -50.0
    vllm_max_context: int = 8192


class VllmHttpPrior:
    backend = "vllm"

    def __init__(self, config: LoadConfig) -> None:
        if not config.vllm_url:
            raise ValueError("vLLM URL is required.")
        if config.vllm_top_k <= 0:
            raise ValueError("--vllm-top-k must be > 0")

        self._requests = self._import_requests()
        self._session = self._requests.Session()
        self._base_url = config.vllm_url.rstrip("/")
        self._timeout_seconds = max(1.0, float(config.vllm_timeout_seconds))
        self._top_k = int(config.vllm_top_k)
        self._fallback_logit = float(config.vllm_fallback_logit)
        self._warned_top_k_clamp = False
        self.dtype_name = "vllm-http"

        meta = self._request_json("GET", "/meta")
        remote_model_id = str(meta.get("model_id") or "").strip()
        self.model_id = remote_model_id or config.model_id
        if remote_model_id and config.model_id and config.model_id != remote_model_id:
            warnings.warn(
                f"Configured model_id `{config.model_id}` differs from remote vLLM model "
                f"`{remote_model_id}`. Using remote model_id for archive metadata.",
                stacklevel=2,
            )

        vocab_size = int(meta.get("vocab_size", 0))
        if vocab_size <= 0:
            raise RuntimeError("vLLM /meta response did not include a positive vocab_size.")
        self.vocab_size = vocab_size

        self.max_context_tokens = int(meta.get("max_model_len", config.vllm_max_context))
        if self.max_context_tokens <= 0:
            self.max_context_tokens = max(1, config.vllm_max_context)

        remote_max_logprobs = meta.get("max_logprobs")
        try:
            remote_max_logprobs_int = int(remote_max_logprobs)
        except (TypeError, ValueError):
            remote_max_logprobs_int = -1
        if remote_max_logprobs_int > 0 and self._top_k > remote_max_logprobs_int:
            self._top_k = remote_max_logprobs_int
            warnings.warn(
                f"Remote vLLM /meta max_logprobs={remote_max_logprobs_int}; "
                "clamping --vllm-top-k to that value.",
                stacklevel=2,
            )
            self._warned_top_k_clamp = True

        bos_token_id = int(meta.get("bos_token_id", -1))
        if bos_token_id < 0:
            bos_token_id = int(meta.get("eos_token_id", -1))
        if bos_token_id < 0:
            raise ValueError("Remote vLLM tokenizer has no BOS/EOS token.")
        self.bos_token_id = bos_token_id

        self._next_token_id = self.bos_token_id
        self._cached_context: list[int] = []

    def _import_requests(self) -> Any:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency is core for this project
            raise RuntimeError("requests is required for vLLM backend.") from exc
        return requests

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict:
        url = f"{self._base_url}{path}"
        try:
            if method == "GET":
                response = self._session.get(url, timeout=self._timeout_seconds)
            else:
                response = self._session.post(url, json=payload, timeout=self._timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            detail_suffix = ""
            response = getattr(exc, "response", None)
            if response is not None:
                parsed: Any = None
                try:
                    parsed = response.json()
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and "detail" in parsed:
                    detail_suffix = f" | server detail: {parsed['detail']!r}"
                elif isinstance(parsed, dict):
                    detail_suffix = f" | server json: {parsed!r}"
                else:
                    text = str(getattr(response, "text", "")).strip()
                    if text:
                        detail_suffix = f" | server body: {text[:500]!r}"
            raise RuntimeError(
                f"vLLM backend request failed ({method} {url}): {exc}{detail_suffix}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}.")
        return data

    def _extract_max_allowed_logprobs(self, error_text: str) -> int | None:
        match = re.search(r"max allowed:\s*(\d+)", error_text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def reset(self) -> None:
        self._next_token_id = self.bos_token_id
        self._cached_context = []

    def next_logits(self) -> torch.Tensor:
        full_context = self._cached_context + [self._next_token_id]
        if len(full_context) > self.max_context_tokens:
            full_context = full_context[-self.max_context_tokens :]

        payload = {
            "token_ids": full_context,
            "top_k": self._top_k,
        }
        try:
            response = self._request_json("POST", "/next-token-logprobs", payload=payload)
        except RuntimeError as exc:
            max_allowed = self._extract_max_allowed_logprobs(str(exc))
            if max_allowed is None or max_allowed >= self._top_k:
                raise
            self._top_k = max_allowed
            if not self._warned_top_k_clamp:
                warnings.warn(
                    f"Remote vLLM limited --vllm-top-k to {self._top_k}; "
                    "continuing with the lower value.",
                    stacklevel=2,
                )
                self._warned_top_k_clamp = True
            payload["top_k"] = self._top_k
            response = self._request_json("POST", "/next-token-logprobs", payload=payload)
        entries = response.get("top_token_logprobs")
        if not isinstance(entries, list):
            raise RuntimeError("vLLM response missing `top_token_logprobs` list.")

        logits = np.full(self.vocab_size, self._fallback_logit, dtype=np.float32)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            token_id = int(entry.get("token_id", -1))
            logprob = float(entry.get("logprob", self._fallback_logit))
            if 0 <= token_id < self.vocab_size:
                logits[token_id] = logprob

        self._cached_context = full_context
        return torch.from_numpy(logits.copy())

    def accept_token(self, token_id: int) -> None:
        self._next_token_id = int(token_id)

    def encode_text(self, text: str) -> list[int]:
        response = self._request_json("POST", "/tokenize", payload={"text": text})
        token_ids = response.get("token_ids")
        if not isinstance(token_ids, list):
            raise RuntimeError("vLLM response missing `token_ids` list.")
        return [int(token_id) for token_id in token_ids]

    def decode_tokens(self, token_ids: list[int]) -> str:
        response = self._request_json(
            "POST",
            "/detokenize",
            payload={"token_ids": [int(token_id) for token_id in token_ids]},
        )
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("vLLM response missing `text` string.")
        return text


def load_prior(
    model_id: str,
    vllm_url: str,
    vllm_top_k: int = 256,
    vllm_timeout_seconds: float = 60.0,
    vllm_fallback_logit: float = -50.0,
    vllm_max_context: int = 8192,
) -> PriorModel:
    config = LoadConfig(
        model_id=model_id,
        vllm_url=vllm_url,
        vllm_top_k=vllm_top_k,
        vllm_timeout_seconds=vllm_timeout_seconds,
        vllm_fallback_logit=vllm_fallback_logit,
        vllm_max_context=vllm_max_context,
    )
    return VllmHttpPrior(config)
