from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse
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
    server_url: str
    top_k: int = 256
    timeout_seconds: float = 60.0
    fallback_logit: float = -50.0
    max_context: int = 8192


class TransformersWebsocketPrior:
    backend = "transformers-ws"

    def __init__(self, config: LoadConfig) -> None:
        if not config.server_url:
            raise ValueError("Server URL is required.")
        if config.top_k <= 0:
            raise ValueError("--top-k must be > 0")

        self._timeout_seconds = max(1.0, float(config.timeout_seconds))
        self._top_k = int(config.top_k)
        self._fallback_logit = float(config.fallback_logit)
        self._ws_url = _normalize_ws_url(config.server_url)
        self._conn: Any | None = None

        self._conn = self._connect_ws()
        init_response = self._request(
            {
                "op": "init",
                "top_k": self._top_k,
                "max_context_tokens": int(config.max_context),
            }
        )

        remote_model_id = str(init_response.get("model_id") or "").strip()
        self.model_id = remote_model_id or config.model_id
        if remote_model_id and config.model_id and config.model_id != remote_model_id:
            warnings.warn(
                f"Configured model_id `{config.model_id}` differs from remote model "
                f"`{remote_model_id}`. Using remote model_id for archive metadata.",
                stacklevel=2,
            )

        self.dtype_name = str(init_response.get("dtype") or "unknown")

        vocab_size = int(init_response.get("vocab_size", 0))
        if vocab_size <= 0:
            raise RuntimeError("WebSocket init response did not include a positive vocab_size.")
        self.vocab_size = vocab_size

        bos_token_id = int(init_response.get("bos_token_id", -1))
        if bos_token_id < 0:
            bos_token_id = int(init_response.get("eos_token_id", -1))
        if bos_token_id < 0:
            raise ValueError("Remote tokenizer has no BOS/EOS token.")
        self.bos_token_id = bos_token_id

        max_top_k = init_response.get("max_top_k")
        if max_top_k is not None:
            try:
                max_top_k_int = int(max_top_k)
            except (TypeError, ValueError):
                max_top_k_int = -1
            if max_top_k_int > 0 and self._top_k > max_top_k_int:
                self._top_k = max_top_k_int
                warnings.warn(
                    f"Remote server max_top_k={max_top_k_int}; clamping --top-k to that value.",
                    stacklevel=2,
                )

        self._next_token_id = self.bos_token_id

    def _import_websockets(self) -> Any:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - dependency is core for this project
            raise RuntimeError("websockets is required for WebSocket backend.") from exc
        return connect

    def _connect_ws(self) -> Any:
        connect = self._import_websockets()
        try:
            return connect(
                self._ws_url,
                open_timeout=self._timeout_seconds,
                close_timeout=self._timeout_seconds,
                max_size=None,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to WebSocket server at {self._ws_url}: {exc}") from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._conn.send(json.dumps(payload, separators=(",", ":")))
            raw = self._conn.recv(timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(
                f"WebSocket request timed out after {self._timeout_seconds:.1f}s: {payload.get('op')}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"WebSocket request failed for op={payload.get('op')!r}: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Server returned non-JSON response: {raw!r}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object response, got {type(data).__name__}.")

        if not bool(data.get("ok", False)):
            error = data.get("error")
            code = data.get("code")
            raise RuntimeError(
                f"WebSocket server error for op={payload.get('op')!r}: {code or 'error'}: {error}"
            )
        return data

    def reset(self) -> None:
        self._request({"op": "reset"})
        self._next_token_id = self.bos_token_id

    def next_logits(self) -> torch.Tensor:
        response = self._request(
            {
                "op": "step",
                "token_id": int(self._next_token_id),
                "top_k": self._top_k,
            }
        )
        entries = response.get("top_token_logprobs")
        if not isinstance(entries, list):
            raise RuntimeError("WebSocket response missing `top_token_logprobs` list.")

        logits = np.full(self.vocab_size, self._fallback_logit, dtype=np.float32)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            token_id = int(entry.get("token_id", -1))
            logprob = float(entry.get("logprob", self._fallback_logit))
            if 0 <= token_id < self.vocab_size:
                logits[token_id] = logprob
        return torch.from_numpy(logits.copy())

    def accept_token(self, token_id: int) -> None:
        self._next_token_id = int(token_id)

    def encode_text(self, text: str) -> list[int]:
        response = self._request({"op": "tokenize", "text": text})
        token_ids = response.get("token_ids")
        if not isinstance(token_ids, list):
            raise RuntimeError("WebSocket response missing `token_ids` list.")
        return [int(token_id) for token_id in token_ids]

    def decode_tokens(self, token_ids: list[int]) -> str:
        response = self._request(
            {
                "op": "detokenize",
                "token_ids": [int(token_id) for token_id in token_ids],
            }
        )
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("WebSocket response missing `text` string.")
        return text

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        try:
            self._conn.close(timeout=self._timeout_seconds)
        except Exception:
            pass
        self._conn = None

    def __del__(self) -> None:
        self.close()


def _normalize_ws_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    if not parsed.scheme:
        raise ValueError("Server URL must include a scheme (http/https/ws/wss).")

    if parsed.scheme in {"http", "https"}:
        scheme = "wss" if parsed.scheme == "https" else "ws"
    elif parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    else:
        raise ValueError(f"Unsupported URL scheme for server URL: {parsed.scheme}")

    path = parsed.path or ""
    if path in {"", "/"}:
        path = "/ws"

    return urlunparse((scheme, parsed.netloc, path, "", parsed.query, ""))


def load_prior(
    model_id: str,
    server_url: str,
    top_k: int = 256,
    timeout_seconds: float = 60.0,
    fallback_logit: float = -50.0,
    max_context: int = 8192,
) -> PriorModel:
    config = LoadConfig(
        model_id=model_id,
        server_url=server_url,
        top_k=top_k,
        timeout_seconds=timeout_seconds,
        fallback_logit=fallback_logit,
        max_context=max_context,
    )
    return TransformersWebsocketPrior(config)
