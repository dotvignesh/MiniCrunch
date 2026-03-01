from __future__ import annotations

from types import SimpleNamespace

import pytest

from minicrunch import backends


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            raise RuntimeError(f"http {self._status_code}")

    def json(self) -> dict:
        return self._payload


def _make_fake_requests(
    *,
    meta: dict | None = None,
    token_ids: list[int] | None = None,
    decoded_text: str = "decoded",
    top_token_logprobs: list[dict] | None = None,
    calls: list[tuple[str, str, dict | None, float]] | None = None,
):
    meta_payload = meta or {
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "vocab_size": 6,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_model_len": 16,
    }
    tokenize_payload = {"token_ids": token_ids or [4, 5]}
    logits_payload = {
        "top_token_logprobs": top_token_logprobs
        or [
            {"token_id": 2, "logprob": -0.25},
            {"token_id": 4, "logprob": -1.0},
        ]
    }

    class _FakeSession:
        def get(self, url: str, timeout: float):
            if calls is not None:
                calls.append(("GET", url, None, timeout))
            if url.endswith("/meta"):
                return _FakeResponse(meta_payload)
            return _FakeResponse({}, status_code=404)

        def post(self, url: str, json: dict | None, timeout: float):
            if calls is not None:
                calls.append(("POST", url, json, timeout))
            if url.endswith("/tokenize"):
                return _FakeResponse(tokenize_payload)
            if url.endswith("/detokenize"):
                return _FakeResponse({"text": decoded_text})
            if url.endswith("/next-token-logprobs"):
                return _FakeResponse(logits_payload)
            return _FakeResponse({}, status_code=404)

    return SimpleNamespace(Session=lambda: _FakeSession())


def test_load_prior_routes_vllm(monkeypatch) -> None:
    captured = {}

    class FakeVllmPrior:
        def __init__(self, config) -> None:
            captured["config"] = config

    monkeypatch.setattr(backends, "VllmHttpPrior", FakeVllmPrior)
    prior = backends.load_prior(
        model_id="mistralai/Ministral-3-3B-Instruct-2512",
        vllm_url="https://example.ngrok-free.app",
        vllm_top_k=128,
        vllm_timeout_seconds=30.0,
        vllm_fallback_logit=-42.0,
        vllm_max_context=4096,
    )
    assert isinstance(prior, FakeVllmPrior)
    assert captured["config"].vllm_url == "https://example.ngrok-free.app"
    assert captured["config"].vllm_top_k == 128
    assert captured["config"].vllm_timeout_seconds == 30.0
    assert captured["config"].vllm_fallback_logit == -42.0
    assert captured["config"].vllm_max_context == 4096


def test_vllm_prior_roundtrip_methods(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None, float]] = []
    fake_requests = _make_fake_requests(calls=calls)
    monkeypatch.setattr(backends.VllmHttpPrior, "_import_requests", lambda _self: fake_requests)

    prior = backends.VllmHttpPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            vllm_url="https://example.ngrok-free.app",
            vllm_top_k=3,
            vllm_fallback_logit=-55.0,
        )
    )

    assert prior.backend == "vllm"
    assert prior.model_id == "mistralai/Ministral-3-3B-Instruct-2512"
    assert prior.bos_token_id == 1
    assert prior.vocab_size == 6

    assert prior.encode_text("hello") == [4, 5]
    logits_1 = prior.next_logits()
    assert logits_1.shape[0] == 6
    assert float(logits_1[0]) == pytest.approx(-55.0)
    assert float(logits_1[2]) == pytest.approx(-0.25)

    prior.accept_token(2)
    logits_2 = prior.next_logits()
    assert logits_2.shape[0] == 6

    next_calls = [payload for method, url, payload, _ in calls if method == "POST" and url.endswith("/next-token-logprobs")]
    assert next_calls[0] == {"token_ids": [1], "top_k": 3}
    assert next_calls[1] == {"token_ids": [1, 2], "top_k": 3}

    assert prior.decode_tokens([4, 5]) == "decoded"


def test_vllm_prior_warns_on_model_id_mismatch(monkeypatch) -> None:
    fake_requests = _make_fake_requests(
        meta={
            "model_id": "remote-model",
            "vocab_size": 6,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "max_model_len": 16,
        }
    )
    monkeypatch.setattr(backends.VllmHttpPrior, "_import_requests", lambda _self: fake_requests)

    with pytest.warns(UserWarning, match="differs from remote vLLM model"):
        prior = backends.VllmHttpPrior(
            backends.LoadConfig(
                model_id="local-model",
                vllm_url="https://example.ngrok-free.app",
            )
        )
    assert prior.model_id == "remote-model"


def test_vllm_prior_requires_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top-k"):
        backends.VllmHttpPrior(
            backends.LoadConfig(
                model_id="mistralai/Ministral-3-3B-Instruct-2512",
                vllm_url="https://example.ngrok-free.app",
                vllm_top_k=0,
            )
        )


def test_vllm_prior_requires_bos_or_eos(monkeypatch) -> None:
    fake_requests = _make_fake_requests(
        meta={
            "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
            "vocab_size": 6,
            "bos_token_id": -1,
            "eos_token_id": -1,
            "max_model_len": 16,
        }
    )
    monkeypatch.setattr(backends.VllmHttpPrior, "_import_requests", lambda _self: fake_requests)

    with pytest.raises(ValueError, match="BOS/EOS"):
        backends.VllmHttpPrior(
            backends.LoadConfig(
                model_id="mistralai/Ministral-3-3B-Instruct-2512",
                vllm_url="https://example.ngrok-free.app",
            )
        )


def test_vllm_prior_clamps_top_k_on_server_limit(monkeypatch) -> None:
    fake_requests = _make_fake_requests()
    monkeypatch.setattr(backends.VllmHttpPrior, "_import_requests", lambda _self: fake_requests)

    prior = backends.VllmHttpPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            vllm_url="https://example.ngrok-free.app",
            vllm_top_k=256,
        )
    )

    observed_top_ks: list[int] = []

    def fake_request_json(method: str, path: str, payload: dict | None = None) -> dict:
        assert method == "POST"
        assert path == "/next-token-logprobs"
        assert payload is not None
        observed_top_ks.append(int(payload["top_k"]))
        if len(observed_top_ks) == 1:
            raise RuntimeError(
                "vLLM backend request failed (POST /next-token-logprobs): "
                "500 ... Requested sample logprobs of 256, "
                "which is greater than max allowed: 20"
            )
        return {"top_token_logprobs": [{"token_id": 2, "logprob": -0.25}]}

    monkeypatch.setattr(prior, "_request_json", fake_request_json)

    with pytest.warns(UserWarning, match="limited --vllm-top-k to 20"):
        logits = prior.next_logits()

    assert logits.shape[0] == 6
    assert observed_top_ks == [256, 20]
