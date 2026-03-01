#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any
import uuid

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if self.device_type == "cuda":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        if self.device_type != "cuda":
            model.to(self.device_type)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
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

    def step(self, state: SessionState, token_id: int, top_k: int) -> list[dict[str, float]]:
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
                outputs = self.model(input_ids=input_ids, use_cache=True)
            else:
                input_ids = torch.tensor([[int(token_id)]], device=self.input_device, dtype=torch.long)
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )

        state.past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1]

        request_top_k = max(1, min(int(top_k), self.vocab_size))
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top_logprobs, top_indices = torch.topk(logprobs, k=request_top_k)

        entries: list[dict[str, float]] = []
        for idx, logprob in zip(top_indices.tolist(), top_logprobs.tolist()):
            entries.append({"token_id": int(idx), "logprob": float(logprob)})
        return entries


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def build_app(args: argparse.Namespace) -> FastAPI:
    runtime = Runtime(args)
    app = FastAPI(title="MiniCrunch Transformers WebSocket server", version="2.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": runtime.model_id,
            "dtype": runtime.dtype_name,
            "vocab_size": runtime.vocab_size,
            "ws_path": "/ws",
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
