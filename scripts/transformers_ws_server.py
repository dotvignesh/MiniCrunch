#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
from typing import Any
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minicrunch.codec import compress_text, decompress_archive


def _resolve_dtype(dtype_name: str, device_type: str) -> torch.dtype:
    dtype_key = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_key == "auto":
        if device_type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device_type == "cuda":
            return torch.float16
        return torch.float32
    if dtype_key not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Allowed: auto, {allowed}")
    return mapping[dtype_key]


@dataclass
class SessionState:
    session_id: str
    initialized: bool = False
    context_tokens: list[int] = field(default_factory=list)
    past_key_values: Any | None = None
    max_context_tokens: int = 8192
    default_top_k: int = 256

    def reset(self) -> None:
        self.context_tokens = []
        self.past_key_values = None


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_id = args.model_id
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = _resolve_dtype(args.dtype, self.device_type)
        self.default_max_context = max(1, int(args.max_context))
        self.default_top_k = max(1, int(args.top_k_default))

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )

        model, loader_name = _load_text_generation_model(
            model_id=args.model_id,
            dtype=self.dtype,
            device_type=self.device_type,
            trust_remote_code=args.trust_remote_code,
        )
        if self.device_type != "cuda":
            model.to(self.device_type)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.model_loader_name = loader_name
        self._text_model = _select_text_forward_model(model)
        self.input_device = next(model.parameters()).device

        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            raise RuntimeError("Unable to determine tokenizer/model vocab size.")
        self.vocab_size = vocab_size

        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if bos_token_id is None:
            bos_token_id = eos_token_id
        if bos_token_id is None:
            raise RuntimeError("Tokenizer has no BOS/EOS token.")
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else -1

        cfg_max_ctx = getattr(model.config, "max_position_embeddings", None)
        if isinstance(cfg_max_ctx, int) and cfg_max_ctx > 0:
            self.default_max_context = min(self.default_max_context, cfg_max_ctx)

    @property
    def dtype_name(self) -> str:
        if self.dtype == torch.bfloat16:
            return "bfloat16"
        if self.dtype == torch.float16:
            return "float16"
        if self.dtype == torch.float32:
            return "float32"
        return str(self.dtype)

    def tokenize(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(token_id) for token_id in token_ids]

    def detokenize(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _advance_and_get_logits(self, state: SessionState, token_id: int) -> torch.Tensor:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"token_id out of range: {token_id}")

        state.context_tokens.append(int(token_id))
        rebuild_cache = False
        if len(state.context_tokens) > state.max_context_tokens:
            state.context_tokens = state.context_tokens[-state.max_context_tokens :]
            state.past_key_values = None
            rebuild_cache = True

        with torch.inference_mode():
            if state.past_key_values is None or rebuild_cache:
                input_ids = torch.tensor(
                    [state.context_tokens],
                    device=self.input_device,
                    dtype=torch.long,
                )
                outputs = self._run_text_forward(
                    input_ids=input_ids,
                    past_key_values=None,
                )
            else:
                input_ids = torch.tensor([[int(token_id)]], device=self.input_device, dtype=torch.long)
                outputs = self._run_text_forward(
                    input_ids=input_ids,
                    past_key_values=state.past_key_values,
                )

        state.past_key_values = outputs.past_key_values
        return outputs.logits[0, -1]

    def _top_logprobs(self, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        request_top_k = max(1, min(int(top_k), self.vocab_size))
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top_logprobs, top_indices = torch.topk(logprobs, k=request_top_k)
        return top_logprobs, top_indices

    def step(self, state: SessionState, token_id: int, top_k: int) -> list[dict[str, float]]:
        logits = self._advance_and_get_logits(state=state, token_id=token_id)
        top_logprobs, top_indices = self._top_logprobs(logits=logits, top_k=top_k)

        entries: list[dict[str, float]] = []
        for idx, logprob in zip(top_indices.tolist(), top_logprobs.tolist()):
            entries.append({"token_id": int(idx), "logprob": float(logprob)})
        return entries

    def step_dense_logits(
        self,
        state: SessionState,
        token_id: int,
        top_k: int,
        fallback_logit: float,
    ) -> torch.Tensor:
        logits = self._advance_and_get_logits(state=state, token_id=token_id)
        top_logprobs, top_indices = self._top_logprobs(logits=logits, top_k=top_k)

        dense = torch.full(
            (self.vocab_size,),
            float(fallback_logit),
            dtype=torch.float32,
        )
        dense[top_indices.to(device="cpu", dtype=torch.long)] = top_logprobs.to(
            device="cpu",
            dtype=torch.float32,
        )
        return dense

    def _run_text_forward(self, input_ids: torch.Tensor, past_key_values: Any | None) -> Any:
        attempts = [self._text_model]
        if self._text_model is not self.model:
            attempts.append(self.model)

        last_error: Exception | None = None
        for candidate in attempts:
            try:
                if past_key_values is None:
                    return candidate(input_ids=input_ids, use_cache=True)
                return candidate(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            except TypeError as exc:
                # Some wrappers may not expose past_key_values in text-only mode.
                if past_key_values is not None:
                    try:
                        return candidate(input_ids=input_ids, use_cache=True)
                    except Exception as inner_exc:
                        last_error = inner_exc
                        continue
                last_error = exc
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Model forward failed for text-only next-token scoring. "
            f"Loader={self.model_loader_name}, last_error={last_error}"
        )


class RuntimeLocalPrior:
    """In-process PriorModel adapter so compression work stays on the model host."""

    backend = "transformers-ws"

    def __init__(
        self,
        runtime: Runtime,
        *,
        top_k: int,
        max_context: int,
        fallback_logit: float,
    ) -> None:
        self._runtime = runtime
        self._top_k = max(1, min(int(top_k), runtime.vocab_size))
        self._max_context = max(1, int(max_context))
        self._fallback_logit = float(fallback_logit)

        self.model_id = runtime.model_id
        self.dtype_name = runtime.dtype_name
        self.vocab_size = runtime.vocab_size
        self.bos_token_id = runtime.bos_token_id

        self._state = SessionState(
            session_id=str(uuid.uuid4()),
            initialized=True,
            max_context_tokens=self._max_context,
            default_top_k=self._top_k,
        )
        self._next_token_id = self.bos_token_id

    def reset(self) -> None:
        self._state.reset()
        self._next_token_id = self.bos_token_id

    def next_logits(self) -> torch.Tensor:
        return self._runtime.step_dense_logits(
            state=self._state,
            token_id=int(self._next_token_id),
            top_k=self._top_k,
            fallback_logit=self._fallback_logit,
        )

    def accept_token(self, token_id: int) -> None:
        self._next_token_id = int(token_id)

    def encode_text(self, text: str) -> list[int]:
        return self._runtime.tokenize(text)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self._runtime.detokenize(token_ids)


def _build_model_kwargs(
    *,
    dtype: torch.dtype,
    device_type: str,
    trust_remote_code: bool,
    use_legacy_torch_dtype: bool,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if use_legacy_torch_dtype:
        model_kwargs["torch_dtype"] = dtype
    else:
        model_kwargs["dtype"] = dtype
    if device_type == "cuda":
        model_kwargs["device_map"] = "auto"
    return model_kwargs


def _load_with_loader(
    *,
    loader: Any,
    model_id: str,
    dtype: torch.dtype,
    device_type: str,
    trust_remote_code: bool,
) -> Any:
    try:
        kwargs = _build_model_kwargs(
            dtype=dtype,
            device_type=device_type,
            trust_remote_code=trust_remote_code,
            use_legacy_torch_dtype=False,
        )
        return loader.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs = _build_model_kwargs(
            dtype=dtype,
            device_type=device_type,
            trust_remote_code=trust_remote_code,
            use_legacy_torch_dtype=True,
        )
        return loader.from_pretrained(model_id, **kwargs)


def _load_text_generation_model(
    *,
    model_id: str,
    dtype: torch.dtype,
    device_type: str,
    trust_remote_code: bool,
) -> tuple[Any, str]:
    attempted_errors: list[str] = []

    def _attempt(loader: Any, loader_name: str) -> tuple[Any, str] | None:
        try:
            model = _load_with_loader(
                loader=loader,
                model_id=model_id,
                dtype=dtype,
                device_type=device_type,
                trust_remote_code=trust_remote_code,
            )
            return model, loader_name
        except Exception as exc:
            attempted_errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            return None

    result = _attempt(AutoModelForCausalLM, "AutoModelForCausalLM")
    if result is not None:
        return result

    image_text_loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if image_text_loader is not None:
        result = _attempt(image_text_loader, "AutoModelForImageTextToText")
        if result is not None:
            return result

    vision2seq_loader = getattr(transformers, "AutoModelForVision2Seq", None)
    if vision2seq_loader is not None:
        result = _attempt(vision2seq_loader, "AutoModelForVision2Seq")
        if result is not None:
            return result

    details = " | ".join(attempted_errors[-3:])
    raise RuntimeError(
        "Unable to load model for text next-token scoring with available Transformers loaders. "
        f"model_id={model_id!r}. Recent errors: {details}"
    )


def _select_text_forward_model(model: Any) -> Any:
    if hasattr(model, "language_model"):
        language_model = getattr(model, "language_model")
        if callable(getattr(language_model, "forward", None)):
            return language_model
    return model


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def build_app(args: argparse.Namespace) -> FastAPI:
    runtime = Runtime(args)
    app = FastAPI(title="MiniCrunch Transformers WebSocket server", version="2.0")
    runtime_lock = threading.Lock()

    def _make_local_prior(payload: dict[str, Any]) -> RuntimeLocalPrior:
        top_k = int(payload.get("top_k", runtime.default_top_k))
        max_context = int(payload.get("max_context", runtime.default_max_context))
        fallback_logit = float(payload.get("fallback_logit", -50.0))
        return RuntimeLocalPrior(
            runtime=runtime,
            top_k=top_k,
            max_context=max_context,
            fallback_logit=fallback_logit,
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": runtime.model_id,
            "dtype": runtime.dtype_name,
            "loader": runtime.model_loader_name,
            "vocab_size": runtime.vocab_size,
            "ws_path": "/ws",
            "api_paths": ["/api/compress", "/api/decompress"],
        }

    @app.post("/api/compress")
    def api_compress(payload: dict[str, Any]) -> dict[str, Any]:
        text = payload.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="`text` must be a string.")

        total_freq = int(payload.get("total_freq", 1 << 20))
        if total_freq <= 0:
            raise HTTPException(status_code=400, detail="`total_freq` must be > 0.")

        prior = _make_local_prior(payload)
        with runtime_lock:
            result = compress_text(
                text=text,
                prior=prior,
                total_freq=total_freq,
                progress_every=0,
                progress_callback=None,
            )

        return {
            "ok": True,
            "archive_b64": base64.b64encode(result.archive).decode("ascii"),
            "payload_bits": result.payload_bits,
            "token_count": result.token_count,
            "elapsed_seconds": result.elapsed_seconds,
            "header": result.header,
            "compressed_size": len(result.archive),
            "original_size": int(result.header.get("original_bytes", 0)),
        }

    @app.post("/api/decompress")
    def api_decompress(payload: dict[str, Any]) -> dict[str, Any]:
        archive_b64 = payload.get("archive_b64")
        if not isinstance(archive_b64, str) or not archive_b64:
            raise HTTPException(status_code=400, detail="`archive_b64` must be a non-empty string.")
        try:
            archive = base64.b64decode(archive_b64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 archive: {exc}") from exc

        verify_hash = bool(payload.get("verify_hash", True))
        prior = _make_local_prior(payload)
        with runtime_lock:
            result = decompress_archive(
                archive=archive,
                prior=prior,
                progress_every=0,
                progress_callback=None,
                verify_hash=verify_hash,
            )

        return {
            "ok": True,
            "text": result.text,
            "token_count": result.token_count,
            "elapsed_seconds": result.elapsed_seconds,
            "header": result.header,
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        state = SessionState(session_id=str(uuid.uuid4()))

        async def send_ok(op: str, **payload: Any) -> None:
            body = {"ok": True, "op": op}
            body.update(payload)
            await websocket.send_text(_json(body))

        async def send_error(op: str, message: str, code: str = "bad-request") -> None:
            await websocket.send_text(
                _json(
                    {
                        "ok": False,
                        "op": op,
                        "code": code,
                        "error": message,
                    }
                )
            )

        try:
            while True:
                raw_message = await websocket.receive_text()
                op = "unknown"
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        raise ValueError("Message must be a JSON object")
                    op = str(message.get("op") or "")

                    if op == "init":
                        state.initialized = True
                        state.max_context_tokens = max(
                            1,
                            int(message.get("max_context_tokens", runtime.default_max_context)),
                        )
                        state.default_top_k = max(
                            1,
                            int(message.get("top_k", runtime.default_top_k)),
                        )
                        state.default_top_k = min(state.default_top_k, runtime.vocab_size)
                        state.reset()
                        await send_ok(
                            "init",
                            session_id=state.session_id,
                            model_id=runtime.model_id,
                            dtype=runtime.dtype_name,
                            loader=runtime.model_loader_name,
                            vocab_size=runtime.vocab_size,
                            bos_token_id=runtime.bos_token_id,
                            eos_token_id=runtime.eos_token_id,
                            max_context_tokens=state.max_context_tokens,
                            max_top_k=runtime.vocab_size,
                            top_k=state.default_top_k,
                        )
                        continue

                    if not state.initialized:
                        await send_error(op, "Session is not initialized. Send op=init first.")
                        continue

                    if op == "reset":
                        state.reset()
                        await send_ok("reset")
                        continue

                    if op == "close":
                        await send_ok("close")
                        await websocket.close(code=1000)
                        return

                    if op == "tokenize":
                        text = message.get("text")
                        if not isinstance(text, str):
                            raise ValueError("text must be a string")
                        await send_ok("tokenize", token_ids=runtime.tokenize(text))
                        continue

                    if op == "detokenize":
                        token_ids = message.get("token_ids")
                        if not isinstance(token_ids, list):
                            raise ValueError("token_ids must be a list of integers")
                        clean_ids = [int(token_id) for token_id in token_ids]
                        await send_ok("detokenize", text=runtime.detokenize(clean_ids))
                        continue

                    if op == "step":
                        if "token_id" not in message:
                            raise ValueError("token_id is required")
                        token_id = int(message["token_id"])
                        top_k = int(message.get("top_k", state.default_top_k))
                        top_k = max(1, min(top_k, runtime.vocab_size))
                        entries = runtime.step(state=state, token_id=token_id, top_k=top_k)
                        await send_ok(
                            "step",
                            top_token_logprobs=entries,
                            context_length=len(state.context_tokens),
                        )
                        continue

                    await send_error(op, f"Unsupported op: {op}", code="unsupported-op")
                except Exception as exc:
                    await send_error(op, str(exc), code="runtime-error")
        except WebSocketDisconnect:
            return

    return app


def maybe_start_tunnel(args: argparse.Namespace) -> tuple[str | None, Any | None]:
    if args.tunnel == "none":
        return None, None
    if args.tunnel != "ngrok":
        raise RuntimeError(f"Unsupported tunnel provider: {args.tunnel}")

    try:
        from pyngrok import ngrok
    except ImportError as exc:
        raise RuntimeError(
            "Tunnel provider 'ngrok' requires pyngrok. Install it with: pip install pyngrok"
        ) from exc

    authtoken = args.ngrok_authtoken or os.getenv("NGROK_AUTHTOKEN")
    if authtoken:
        ngrok.set_auth_token(authtoken)

    connect_kwargs = {"addr": str(args.port), "proto": "http"}
    if args.ngrok_domain:
        connect_kwargs["domain"] = args.ngrok_domain

    tunnel = ngrok.connect(**connect_kwargs)
    return str(tunnel.public_url), ngrok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host Transformers WebSocket endpoint for MiniCrunch.")
    parser.add_argument(
        "--model-id",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Model to load via Transformers.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype (auto, bfloat16, float16, float32).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument(
        "--top-k-default",
        type=int,
        default=256,
        help="Default top-k if not provided per WS step request.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--tunnel",
        default="none",
        choices=["none", "ngrok"],
        help="Optional public tunnel provider (use 'ngrok' for Colab public URL).",
    )
    parser.add_argument(
        "--ngrok-authtoken",
        help="Optional ngrok auth token. If omitted, reads NGROK_AUTHTOKEN env var.",
    )
    parser.add_argument(
        "--ngrok-domain",
        help="Optional reserved ngrok domain (paid plans).",
    )
    parser.add_argument(
        "--public-url-file",
        help="Optional path to write the tunnel public URL.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print startup config and derived public WebSocket URL hints.",
    )
    return parser.parse_args()


def _to_public_ws_url(public_http_url: str) -> str:
    if public_http_url.startswith("https://"):
        return "wss://" + public_http_url.removeprefix("https://").rstrip("/") + "/ws"
    if public_http_url.startswith("http://"):
        return "ws://" + public_http_url.removeprefix("http://").rstrip("/") + "/ws"
    return public_http_url.rstrip("/") + "/ws"


def main() -> None:
    args = parse_args()
    public_url, ngrok_mod = maybe_start_tunnel(args)

    if public_url and args.public_url_file:
        with open(args.public_url_file, "w", encoding="utf-8") as handle:
            handle.write(public_url)

    if args.print_config:
        print("MiniCrunch Transformers server config")
        print(f"MODEL_ID={args.model_id}")
        print(f"DTYPE={args.dtype}")
        print(f"HOST={args.host}")
        print(f"PORT={args.port}")
        print(f"MAX_CONTEXT={args.max_context}")
        print(f"TOP_K_DEFAULT={args.top_k_default}")
        if public_url:
            print(f"PUBLIC_URL={public_url}")
            print(f"PUBLIC_WS_URL={_to_public_ws_url(public_url)}")

    app = build_app(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if ngrok_mod is not None:
            ngrok_mod.kill()


if __name__ == "__main__":
    main()
