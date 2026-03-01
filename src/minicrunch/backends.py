from __future__ import annotations

import importlib.util
import os
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None


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


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    return mapping[dtype]


def iter_model_configs(model_config: object):
    yield model_config
    for attr in ("text_config", "language_config", "llm_config"):
        nested = getattr(model_config, attr, None)
        if nested is not None:
            yield nested


def resolve_max_context_tokens(model_config: object) -> int | None:
    for cfg in iter_model_configs(model_config):
        candidates = [
            getattr(cfg, "max_position_embeddings", None),
            getattr(cfg, "n_positions", None),
            getattr(cfg, "max_seq_len", None),
            getattr(cfg, "seq_length", None),
        ]
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
    return None


def select_automodel_loader(model_config: object):
    model_type = getattr(model_config, "model_type", None)
    if model_type == "mistral3":
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Model type 'mistral3' requires AutoModelForImageTextToText, "
                "which is unavailable in this transformers version."
            )
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def resolve_vocab_size(model_config: object) -> int:
    for cfg in iter_model_configs(model_config):
        value = getattr(cfg, "vocab_size", None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine vocabulary size from model config.")


def is_fp8_quantized_model(model_config: object) -> bool:
    quantization_config = getattr(model_config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return quantization_config.get("quant_method") == "fp8"
    return bool(getattr(quantization_config, "quant_method", None) == "fp8")


def install_triton_stub_if_missing() -> bool:
    if "triton" in sys.modules:
        return False
    if importlib.util.find_spec("triton") is not None:
        return False

    triton = types.ModuleType("triton")
    tl = types.ModuleType("triton.language")

    class _Constexpr:
        pass

    class _Dtype:
        pass

    def _jit(fn=None, **kwargs):
        if fn is None:
            def _decorator(inner):
                return inner

            return _decorator
        return fn

    def _cdiv(x: int, y: int) -> int:
        return (x + y - 1) // y

    triton.jit = _jit
    triton.cdiv = _cdiv
    triton.Config = object
    triton.runtime = types.SimpleNamespace(
        driver=types.SimpleNamespace(
            active=types.SimpleNamespace(get_current_target=lambda: None),
        )
    )
    tl.constexpr = _Constexpr
    tl.dtype = _Dtype
    triton.language = tl

    sys.modules["triton"] = triton
    sys.modules["triton.language"] = tl
    return True


@contextmanager
def patch_torch_compile_identity(enabled: bool):
    if not enabled:
        yield
        return
    original = getattr(torch, "compile", None)
    if original is None:
        yield
        return

    def _compile_identity(fn=None, **kwargs):
        if fn is None:
            def _decorator(inner):
                return inner

            return _decorator
        return fn

    torch.compile = _compile_identity
    try:
        yield
    finally:
        torch.compile = original


def _import_llama_cpp_llama() -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "llama-cpp-python is not installed. Run: uv sync --extra llamacpp"
        ) from exc
    return Llama


def resolve_llama_n_gpu_layers(device: str, configured: int | None) -> int:
    if configured is not None:
        return configured
    return 0 if device == "cpu" else -1


def resolve_llama_n_threads(configured: int) -> int:
    if configured > 0:
        return configured
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def read_llama_n_ctx(llama: object, fallback: int) -> int:
    value = getattr(llama, "n_ctx", None)
    if callable(value):
        try:
            maybe = int(value())
        except Exception:
            maybe = 0
        if maybe > 0:
            return maybe
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def _raise_llama_load_error(model_ref: str, exc: ValueError) -> None:
    raise ValueError(
        "llama.cpp failed to load GGUF model "
        f"`{model_ref}`. Common causes include unsupported GGUF architecture "
        "in your installed llama.cpp (for example, `mistral3` from Ministral "
        "GGUF models on llama-cpp-python 0.3.16) or a corrupted download. "
        "Try `--backend transformers` for Ministral models, pick a different "
        "GGUF model, or rebuild llama.cpp/llama-cpp-python from newer source."
    ) from exc


def load_llama_model(model_ref: str, kwargs: dict[str, Any]) -> Any:
    """Load GGUF model from local path or HF reference `repo_id::filename.gguf`."""
    Llama = _import_llama_cpp_llama()

    if "::" in model_ref:
        repo_id, filename = model_ref.split("::", 1)
        if not repo_id or not filename:
            raise ValueError(
                "Invalid llama model reference. Use `repo_id::filename.gguf` "
                "or a local .gguf path."
            )
        try:
            return Llama.from_pretrained(repo_id=repo_id, filename=filename, **kwargs)
        except ValueError as exc:
            _raise_llama_load_error(model_ref, exc)

    model_path = Path(model_ref).expanduser().resolve()
    if model_path.exists():
        try:
            return Llama(model_path=str(model_path), **kwargs)
        except ValueError as exc:
            _raise_llama_load_error(str(model_path), exc)

    raise ValueError(
        "For backend=llamacpp, --model-id must be a local GGUF file path "
        "or `repo_id::filename.gguf` for Hugging Face download."
    )


@dataclass
class LoadConfig:
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    llama_n_ctx: int = 8192
    llama_n_batch: int = 512
    llama_n_threads: int = 0
    llama_n_gpu_layers: int | None = None
    vllm_url: str | None = None
    vllm_top_k: int = 256
    vllm_timeout_seconds: float = 60.0
    vllm_fallback_logit: float = -50.0


class TransformersPrior:
    backend = "transformers"

    def __init__(self, config: LoadConfig) -> None:
        self.model_id = config.model_id
        self.device = resolve_device(config.device)
        self.torch_dtype = resolve_dtype(config.dtype, self.device)
        self.dtype_name = dtype_name(self.torch_dtype)
        self.config = AutoConfig.from_pretrained(self.model_id)
        model_loader = select_automodel_loader(self.config)
        fp8_on_non_cuda = is_fp8_quantized_model(self.config) and self.device in {"cpu", "mps"}
        triton_stub_installed = False
        if fp8_on_non_cuda:
            triton_stub_installed = install_triton_stub_if_missing()
            warnings.warn(
                "Model config uses FP8 quantization, which requires Triton kernels not "
                "available on this runtime/device. Transformers will dequantize to "
                "bf16 on CPU/MPS during load.",
                stacklevel=2,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        with patch_torch_compile_identity(enabled=triton_stub_installed):
            self.model = model_loader.from_pretrained(
                self.model_id,
                config=self.config,
                dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            loaded_dtype = first_param.dtype
            if loaded_dtype in {torch.float16, torch.bfloat16, torch.float32}:
                self.dtype_name = dtype_name(loaded_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.max_context_tokens = resolve_max_context_tokens(self.model.config)

        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.eos_token_id
        if bos_token_id is None:
            raise ValueError(
                "Tokenizer has no BOS/EOS token. A start token is required for decoding."
            )
        self.bos_token_id = int(bos_token_id)
        self.vocab_size = resolve_vocab_size(self.model.config)

        self._past_key_values = None
        self._next_token_id = self.bos_token_id
        self._cached_context: list[int] = []

    def reset(self) -> None:
        self._past_key_values = None
        self._next_token_id = self.bos_token_id
        self._cached_context = []

    def next_logits(self) -> torch.Tensor:
        full_context = self._cached_context + [self._next_token_id]
        max_context = self.max_context_tokens
        over_limit = max_context is not None and len(full_context) > max_context

        # Fast path uses a 1-token incremental update while context is within limit.
        # If we exceed context length, rebuild KV cache from the latest window.
        if self._past_key_values is not None and not over_limit:
            input_ids = torch.tensor([[self._next_token_id]], device=self.device)
        else:
            if max_context is not None and len(full_context) > max_context:
                full_context = full_context[-max_context:]
            input_ids = torch.tensor([full_context], device=self.device)
            self._past_key_values = None

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        self._past_key_values = outputs.past_key_values
        self._cached_context = full_context
        return outputs.logits[0, -1, :]

    def accept_token(self, token_id: int) -> None:
        self._next_token_id = int(token_id)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )


class LlamaCppPrior:
    backend = "llamacpp"

    def __init__(self, config: LoadConfig) -> None:
        self.model_id = config.model_id
        self.device = resolve_device(config.device)
        self.dtype_name = "gguf"

        n_gpu_layers = resolve_llama_n_gpu_layers(self.device, config.llama_n_gpu_layers)
        n_threads = resolve_llama_n_threads(config.llama_n_threads)
        n_batch = max(1, min(config.llama_n_batch, config.llama_n_ctx))

        self.model = load_llama_model(
            self.model_id,
            kwargs={
                "n_ctx": config.llama_n_ctx,
                "n_batch": n_batch,
                "n_threads": n_threads,
                "n_gpu_layers": n_gpu_layers,
                "logits_all": True,
                "verbose": False,
            },
        )

        self.vocab_size = int(self.model.n_vocab())
        bos_token_id = int(self.model.token_bos())
        if bos_token_id < 0:
            bos_token_id = int(self.model.token_eos())
        if bos_token_id < 0:
            raise ValueError("GGUF tokenizer has no BOS/EOS token.")

        self.bos_token_id = bos_token_id
        self.max_context_tokens = read_llama_n_ctx(self.model, config.llama_n_ctx)

        self._next_token_id = self.bos_token_id
        self._cached_context: list[int] = []

    def reset(self) -> None:
        self.model.reset()
        self._next_token_id = self.bos_token_id
        self._cached_context = []

    def next_logits(self) -> torch.Tensor:
        full_context = self._cached_context + [self._next_token_id]
        if len(full_context) > self.max_context_tokens:
            full_context = full_context[-self.max_context_tokens :]

        can_incremental = (
            bool(self._cached_context)
            and len(full_context) == len(self._cached_context) + 1
            and full_context[:-1] == self._cached_context
            and int(getattr(self.model, "n_tokens", 0)) > 0
        )

        if can_incremental:
            self.model.eval([self._next_token_id])
        else:
            self.model.reset()
            self.model.eval(full_context)

        self._cached_context = full_context
        last_row = int(self.model.n_tokens) - 1
        logits = np.asarray(
            self.model.scores[last_row, : self.vocab_size],
            dtype=np.float32,
        )
        return torch.from_numpy(logits.copy())

    def accept_token(self, token_id: int) -> None:
        self._next_token_id = int(token_id)

    def encode_text(self, text: str) -> list[int]:
        token_ids = self.model.tokenize(
            text.encode("utf-8"),
            add_bos=False,
            special=False,
        )
        return [int(token_id) for token_id in token_ids]

    def decode_tokens(self, token_ids: list[int]) -> str:
        data = self.model.detokenize([int(token_id) for token_id in token_ids], special=False)
        return data.decode("utf-8")


class VllmHttpPrior:
    backend = "vllm"

    def __init__(self, config: LoadConfig) -> None:
        if not config.vllm_url:
            raise ValueError("backend=vllm requires --vllm-url")
        if config.vllm_top_k <= 0:
            raise ValueError("--vllm-top-k must be > 0")

        self._requests = self._import_requests()
        self._session = self._requests.Session()
        self._base_url = config.vllm_url.rstrip("/")
        self._timeout_seconds = max(1.0, float(config.vllm_timeout_seconds))
        self._top_k = int(config.vllm_top_k)
        self._fallback_logit = float(config.vllm_fallback_logit)
        self.device = "remote"
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

        self.max_context_tokens = int(meta.get("max_model_len", config.llama_n_ctx))
        if self.max_context_tokens <= 0:
            self.max_context_tokens = max(1, config.llama_n_ctx)

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
            raise RuntimeError("requests is required for backend=vllm.") from exc
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
            raise RuntimeError(f"vLLM backend request failed ({method} {url}): {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}.")
        return data

    def reset(self) -> None:
        self._next_token_id = self.bos_token_id
        self._cached_context = []

    def next_logits(self) -> torch.Tensor:
        full_context = self._cached_context + [self._next_token_id]
        if len(full_context) > self.max_context_tokens:
            full_context = full_context[-self.max_context_tokens :]

        response = self._request_json(
            "POST",
            "/next-token-logprobs",
            payload={
                "token_ids": full_context,
                "top_k": self._top_k,
            },
        )
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
    backend: str = "transformers",
    device: str = "auto",
    dtype: str = "auto",
    llama_n_ctx: int = 8192,
    llama_n_batch: int = 512,
    llama_n_threads: int = 0,
    llama_n_gpu_layers: int | None = None,
    vllm_url: str | None = None,
    vllm_top_k: int = 256,
    vllm_timeout_seconds: float = 60.0,
    vllm_fallback_logit: float = -50.0,
) -> PriorModel:
    config = LoadConfig(
        model_id=model_id,
        device=device,
        dtype=dtype,
        llama_n_ctx=llama_n_ctx,
        llama_n_batch=llama_n_batch,
        llama_n_threads=llama_n_threads,
        llama_n_gpu_layers=llama_n_gpu_layers,
        vllm_url=vllm_url,
        vllm_top_k=vllm_top_k,
        vllm_timeout_seconds=vllm_timeout_seconds,
        vllm_fallback_logit=vllm_fallback_logit,
    )

    if backend == "transformers":
        return TransformersPrior(config)
    if backend == "llamacpp":
        return LlamaCppPrior(config)
    if backend == "vllm":
        return VllmHttpPrior(config)
    raise ValueError(
        f"Unsupported backend '{backend}'. Supported backends: transformers, llamacpp, vllm."
    )
