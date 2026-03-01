#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
            outputs = llm.generate(
                prompt_token_ids=[req.token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except TypeError:
            outputs = llm.generate(
                prompts=[""],
                prompt_token_ids=[req.token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Host vLLM logits endpoint for MiniCrunch.")
    parser.add_argument(
        "--model-id",
        default="mistralai/Ministral-3B-Instruct-2410",
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
        "--print-config",
        action="store_true",
        help="Print startup config as JSON before starting the server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
