from __future__ import annotations

from types import SimpleNamespace

import pytest

from minicrunch import backends
from minicrunch.backends import AutoModelForCausalLM, resolve_max_context_tokens, resolve_vocab_size


def test_resolve_max_context_prefers_max_position_embeddings() -> None:
    cfg = SimpleNamespace(max_position_embeddings=32768, n_positions=4096)
    assert resolve_max_context_tokens(cfg) == 32768


def test_resolve_max_context_falls_back_to_n_positions() -> None:
    cfg = SimpleNamespace(n_positions=2048)
    assert resolve_max_context_tokens(cfg) == 2048


def test_resolve_max_context_returns_none_when_absent() -> None:
    cfg = SimpleNamespace()
    assert resolve_max_context_tokens(cfg) is None


def test_resolve_max_context_reads_text_config() -> None:
    cfg = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=65536))
    assert resolve_max_context_tokens(cfg) == 65536


def test_resolve_vocab_size_prefers_nested_text_config() -> None:
    cfg = SimpleNamespace(text_config=SimpleNamespace(vocab_size=32000))
    assert resolve_vocab_size(cfg) == 32000


def test_resolve_vocab_size_errors_when_missing() -> None:
    cfg = SimpleNamespace()
    with pytest.raises(ValueError, match="vocabulary size"):
        resolve_vocab_size(cfg)


def test_select_automodel_loader_defaults_to_causal() -> None:
    cfg = SimpleNamespace(model_type="ministral3")
    assert backends.select_automodel_loader(cfg) is AutoModelForCausalLM


def test_select_automodel_loader_uses_imagetext_for_mistral3(monkeypatch) -> None:
    fake_loader = object()
    monkeypatch.setattr(backends, "AutoModelForImageTextToText", fake_loader)
    cfg = SimpleNamespace(model_type="mistral3")
    assert backends.select_automodel_loader(cfg) is fake_loader


def test_select_automodel_loader_errors_when_imagetext_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(backends, "AutoModelForImageTextToText", None)
    cfg = SimpleNamespace(model_type="mistral3")
    with pytest.raises(RuntimeError):
        backends.select_automodel_loader(cfg)


def test_is_fp8_quantized_model_from_dict() -> None:
    cfg = SimpleNamespace(quantization_config={"quant_method": "fp8"})
    assert backends.is_fp8_quantized_model(cfg) is True


def test_is_fp8_quantized_model_from_object() -> None:
    cfg = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="fp8"))
    assert backends.is_fp8_quantized_model(cfg) is True


def test_is_fp8_quantized_model_ignores_other_methods() -> None:
    cfg = SimpleNamespace(quantization_config={"quant_method": "bitsandbytes"})
    assert backends.is_fp8_quantized_model(cfg) is False


def test_patch_torch_compile_identity_restores_original() -> None:
    original = backends.torch.compile
    with backends.patch_torch_compile_identity(enabled=True):
        assert backends.torch.compile is not original
    assert backends.torch.compile is original


def test_resolve_llama_n_gpu_layers_auto_by_device() -> None:
    assert backends.resolve_llama_n_gpu_layers("cpu", None) == 0
    assert backends.resolve_llama_n_gpu_layers("mps", None) == -1
    assert backends.resolve_llama_n_gpu_layers("cuda", None) == -1
    assert backends.resolve_llama_n_gpu_layers("mps", 17) == 17


def test_read_llama_n_ctx_from_attribute() -> None:
    llama = SimpleNamespace(n_ctx=1234)
    assert backends.read_llama_n_ctx(llama, fallback=42) == 1234


def test_read_llama_n_ctx_from_callable() -> None:
    llama = SimpleNamespace(n_ctx=lambda: 5678)
    assert backends.read_llama_n_ctx(llama, fallback=42) == 5678


def test_load_llama_model_local_path(monkeypatch, tmp_path) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            self.kind = "local"
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, **kwargs):
            obj = cls.__new__(cls)
            obj.kind = "hf"
            obj.kwargs = kwargs
            return obj

    monkeypatch.setattr(backends, "_import_llama_cpp_llama", lambda: FakeLlama)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    model = backends.load_llama_model(str(model_path), kwargs={"n_ctx": 2048})
    assert model.kind == "local"
    assert model.kwargs["n_ctx"] == 2048
    assert model.kwargs["model_path"].endswith("model.gguf")


def test_load_llama_model_hf_reference(monkeypatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            self.kind = "local"
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, **kwargs):
            obj = cls.__new__(cls)
            obj.kind = "hf"
            obj.kwargs = kwargs
            return obj

    monkeypatch.setattr(backends, "_import_llama_cpp_llama", lambda: FakeLlama)

    model = backends.load_llama_model(
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF::model.gguf",
        kwargs={"n_ctx": 4096},
    )
    assert model.kind == "hf"
    assert model.kwargs["repo_id"] == "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
    assert model.kwargs["filename"] == "model.gguf"
    assert model.kwargs["n_ctx"] == 4096


def test_load_llama_model_wraps_unsupported_architecture_error(monkeypatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, **kwargs):
            raise ValueError("Failed to load model from file: /tmp/model.gguf")

    monkeypatch.setattr(backends, "_import_llama_cpp_llama", lambda: FakeLlama)

    with pytest.raises(ValueError, match="unsupported GGUF architecture"):
        backends.load_llama_model(
            "mistralai/Ministral-3-3B-Instruct-2512-GGUF::model.gguf",
            kwargs={"n_ctx": 4096},
        )


def test_load_prior_routes_llamacpp(monkeypatch) -> None:
    captured = {}

    class FakeLlamaPrior:
        def __init__(self, config) -> None:
            captured["config"] = config

    monkeypatch.setattr(backends, "LlamaCppPrior", FakeLlamaPrior)
    prior = backends.load_prior(
        model_id="model.gguf",
        backend="llamacpp",
        device="mps",
        llama_n_ctx=8192,
        llama_n_batch=1024,
        llama_n_threads=8,
        llama_n_gpu_layers=-1,
    )
    assert isinstance(prior, FakeLlamaPrior)
    assert captured["config"].model_id == "model.gguf"
    assert captured["config"].device == "mps"
    assert captured["config"].llama_n_ctx == 8192
    assert captured["config"].llama_n_batch == 1024
    assert captured["config"].llama_n_threads == 8
    assert captured["config"].llama_n_gpu_layers == -1


def test_load_prior_routes_vllm(monkeypatch) -> None:
    captured = {}

    class FakeVllmPrior:
        def __init__(self, config) -> None:
            captured["config"] = config

    monkeypatch.setattr(backends, "VllmHttpPrior", FakeVllmPrior)
    prior = backends.load_prior(
        model_id="mistralai/Ministral-3B-Instruct-2410",
        backend="vllm",
        vllm_url="https://example.ngrok-free.app",
        vllm_top_k=128,
        vllm_timeout_seconds=30.0,
        vllm_fallback_logit=-42.0,
    )
    assert isinstance(prior, FakeVllmPrior)
    assert captured["config"].vllm_url == "https://example.ngrok-free.app"
    assert captured["config"].vllm_top_k == 128
    assert captured["config"].vllm_timeout_seconds == 30.0
    assert captured["config"].vllm_fallback_logit == -42.0
