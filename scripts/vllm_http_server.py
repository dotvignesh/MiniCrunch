#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams


class TokenizeRequest(BaseModel):
    text: str


class DetokenizeRequest(BaseModel):
    token_ids: list[int]


class NextTokenLogprobsRequest(BaseModel):
    token_ids: list[int]
    top_k: int = Field(default=256)


def _extract_top_logprobs(raw: Any) -> list[dict[str, float]]:
    if raw is None:
        return []

    items: list[tuple[int, float]] = []
    if isinstance(raw, dict):
        iterator = raw.items()
    else:
        iterator = []
        for maybe in raw:
            token_id = getattr(maybe, "token_id", None)
            logprob = getattr(maybe, "logprob", None)
            if token_id is None and isinstance(maybe, tuple) and len(maybe) == 2:
                token_id, value = maybe
                logprob = getattr(value, "logprob", value)
            if token_id is None or logprob is None:
                continue
            iterator.append((token_id, logprob))

    for token_id, value in iterator:
        logprob = getattr(value, "logprob", value)
        try:
            items.append((int(token_id), float(logprob)))
        except (TypeError, ValueError):
            continue

    items.sort(key=lambda pair: pair[1], reverse=True)
    return [{"token_id": token_id, "logprob": logprob} for token_id, logprob in items]


def _is_signature_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "unexpected keyword",
            "missing",
            "positional argument",
            "keyword-only",
            "takes",
            "argument",
            "prompt",
            "input",
            "token",
        )
    )


def _generate_with_token_ids(
    llm: LLM, token_ids: list[int], sampling_params: SamplingParams
) -> list[Any]:
    generate_fn = llm.generate
    signature = inspect.signature(generate_fn)
    parameter_names = set(signature.parameters)

    base_kwargs: dict[str, Any] = {"sampling_params": sampling_params}
    if "use_tqdm" in parameter_names:
        base_kwargs["use_tqdm"] = False

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_attempt(*args: Any, **kwargs: Any) -> None:
        payload = base_kwargs.copy()
        payload.update(kwargs)
        attempts.append((args, payload))

    tokens_prompt = {"prompt_token_ids": token_ids}

    # vLLM<=0.7 style APIs (prompt_token_ids kwarg).
    if "prompt_token_ids" in parameter_names:
        add_attempt(prompt_token_ids=[token_ids])
        if "prompts" in parameter_names:
            add_attempt(prompts=[""], prompt_token_ids=[token_ids])

    # vLLM>=0.8 style APIs (prompts/inputs expect PromptType entries).
    if "prompts" in parameter_names:
        add_attempt(prompts=[tokens_prompt])
        add_attempt(prompts=[token_ids])
    if "inputs" in parameter_names:
        add_attempt(inputs=[tokens_prompt])
        add_attempt(inputs=[token_ids])

    # Positional fallbacks for versions with unstable naming.
    add_attempt([tokens_prompt])
    add_attempt([token_ids])

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return generate_fn(*args, **kwargs)
        except TypeError as exc:
            if not _is_signature_mismatch_error(exc):
                raise
            last_error = exc
        except ValueError as exc:
            if not _is_signature_mismatch_error(exc):
                raise
            last_error = exc

    raise RuntimeError(
        "Unable to call vLLM generate() with token IDs for this installed version."
    ) from last_error


def build_app(args: argparse.Namespace) -> FastAPI:
    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    engine = getattr(llm, "llm_engine", None)
    model_config = getattr(engine, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", args.max_model_len))

    app = FastAPI(title="MiniCrunch vLLM logits server", version="1.0")

    @app.get("/meta")
    def meta() -> dict[str, Any]:
        return {
            "model_id": args.model_id,
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "max_model_len": max_model_len,
        }

    @app.post("/tokenize")
    def tokenize(req: TokenizeRequest) -> dict[str, Any]:
        token_ids = tokenizer.encode(req.text, add_special_tokens=False)
        return {"token_ids": [int(token_id) for token_id in token_ids]}

    @app.post("/detokenize")
    def detokenize(req: DetokenizeRequest) -> dict[str, Any]:
        try:
            text = tokenizer.decode(
                req.token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            text = tokenizer.decode(req.token_ids, skip_special_tokens=False)
        return {"text": text}

    @app.post("/next-token-logprobs")
    def next_token_logprobs(req: NextTokenLogprobsRequest) -> dict[str, Any]:
        if not req.token_ids:
            raise HTTPException(status_code=400, detail="token_ids must not be empty.")
        if req.top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be > 0.")

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=req.top_k,
            detokenize=False,
        )
        try:
            outputs = _generate_with_token_ids(llm, req.token_ids, sampling_params)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query vLLM generate() for token logprobs: {exc}",
            ) from exc
        if not outputs:
            raise HTTPException(status_code=500, detail="vLLM returned no outputs.")
        result = outputs[0]
        output_items = getattr(result, "outputs", None) or []
        if not output_items:
            raise HTTPException(status_code=500, detail="vLLM returned empty output list.")
        logprobs_list = getattr(output_items[0], "logprobs", None) or []
        if not logprobs_list:
            raise HTTPException(
                status_code=500,
                detail="vLLM did not return logprobs. Ensure logprobs support is enabled.",
            )
        top_token_logprobs = _extract_top_logprobs(logprobs_list[0])
        return {"top_token_logprobs": top_token_logprobs}

    @app.get("/")
    def root() -> dict[str, str]:
        return {"status": "ok", "message": "MiniCrunch vLLM logits server"}

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
    parser = argparse.ArgumentParser(description="Host vLLM logits endpoint for MiniCrunch.")
    parser.add_argument(
        "--model-id",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Model to load in vLLM.",
    )
    parser.add_argument("--dtype", default="auto", help="vLLM dtype (for example: auto, float16).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
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
        help="Print startup config as JSON before starting the server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    public_url, ngrok_module = maybe_start_tunnel(args)

    if public_url:
        print(f"PUBLIC_URL={public_url}")
        if args.public_url_file:
            with open(args.public_url_file, "w", encoding="utf-8") as handle:
                handle.write(public_url + "\n")

    if args.print_config:
        config = vars(args).copy()
        config["public_url"] = public_url
        print(json.dumps(config, indent=2, sort_keys=True))

    app = build_app(args)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if public_url and ngrok_module is not None:
            ngrok_module.disconnect(public_url)


if __name__ == "__main__":
    main()
