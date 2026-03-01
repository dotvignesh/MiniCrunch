from __future__ import annotations

import json

import pytest

from minicrunch import backends
from minicrunch.codec import compress_text, decompress_archive


class _FakeWsConnection:
    def __init__(self, responder) -> None:
        self._responder = responder
        self._queued: list[dict | BaseException] = []
        self.closed = False
        self.sent_messages: list[dict] = []

    def send(self, payload: str) -> None:
        message = json.loads(payload)
        self.sent_messages.append(message)
        response = self._responder(message)
        if response is not None:
            self._queued.append(response)

    def recv(self, timeout: float | None = None) -> str:
        _ = timeout
        if not self._queued:
            raise RuntimeError("no queued response")
        item = self._queued.pop(0)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)

    def close(self, timeout: float | None = None) -> None:
        _ = timeout
        self.closed = True


def _init_payload() -> dict:
    return {
        "ok": True,
        "op": "init",
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
        "dtype": "bfloat16",
        "vocab_size": 8,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_context_tokens": 64,
        "max_top_k": 256,
        "top_k": 3,
    }


def test_load_prior_routes_transformers_ws(monkeypatch) -> None:
    captured = {}

    class FakePrior:
        def __init__(self, config) -> None:
            captured["config"] = config

    monkeypatch.setattr(backends, "TransformersWebsocketPrior", FakePrior)
    prior = backends.load_prior(
        model_id="mistralai/Ministral-3-3B-Instruct-2512",
        server_url="https://example.ngrok-free.app",
        top_k=128,
        timeout_seconds=30.0,
        fallback_logit=-42.0,
        max_context=4096,
    )

    assert isinstance(prior, FakePrior)
    assert captured["config"].server_url == "https://example.ngrok-free.app"
    assert captured["config"].top_k == 128
    assert captured["config"].timeout_seconds == 30.0
    assert captured["config"].fallback_logit == -42.0
    assert captured["config"].max_context == 4096


def test_websocket_happy_path(monkeypatch) -> None:
    calls: list[dict] = []

    def responder(message: dict) -> dict:
        calls.append(message)
        op = message["op"]
        if op == "init":
            return _init_payload()
        if op == "tokenize":
            return {"ok": True, "op": "tokenize", "token_ids": [4, 5]}
        if op == "step":
            return {
                "ok": True,
                "op": "step",
                "top_token_logprobs": [
                    {"token_id": 2, "logprob": -0.25},
                    {"token_id": 4, "logprob": -1.0},
                ],
            }
        if op == "detokenize":
            return {"ok": True, "op": "detokenize", "text": "decoded"}
        if op == "reset":
            return {"ok": True, "op": "reset"}
        if op == "close":
            return {"ok": True, "op": "close"}
        raise AssertionError(f"unexpected op {op}")

    connection = _FakeWsConnection(responder)
    monkeypatch.setattr(backends.TransformersWebsocketPrior, "_connect_ws", lambda self: connection)

    prior = backends.TransformersWebsocketPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            server_url="https://example.ngrok-free.app",
            top_k=3,
            fallback_logit=-55.0,
        )
    )

    assert prior.backend == "transformers-ws"
    assert prior.model_id == "mistralai/Ministral-3-3B-Instruct-2512"
    assert prior.bos_token_id == 1
    assert prior.vocab_size == 8

    assert prior.encode_text("hello") == [4, 5]

    logits_1 = prior.next_logits()
    assert logits_1.shape[0] == 8
    assert float(logits_1[0]) == pytest.approx(-55.0)
    assert float(logits_1[2]) == pytest.approx(-0.25)

    prior.accept_token(2)
    logits_2 = prior.next_logits()
    assert logits_2.shape[0] == 8

    step_calls = [payload for payload in calls if payload.get("op") == "step"]
    assert step_calls[0] == {"op": "step", "token_id": 1, "top_k": 3}
    assert step_calls[1] == {"op": "step", "token_id": 2, "top_k": 3}

    assert prior.decode_tokens([4, 5]) == "decoded"
    prior.close()


def test_session_reset_and_close(monkeypatch) -> None:
    calls: list[dict] = []

    def responder(message: dict) -> dict:
        calls.append(message)
        op = message["op"]
        if op == "init":
            return _init_payload()
        if op == "reset":
            return {"ok": True, "op": "reset"}
        if op == "step":
            return {
                "ok": True,
                "op": "step",
                "top_token_logprobs": [{"token_id": 2, "logprob": -0.1}],
            }
        if op == "close":
            return {"ok": True, "op": "close"}
        raise AssertionError(f"unexpected op {op}")

    connection = _FakeWsConnection(responder)
    monkeypatch.setattr(backends.TransformersWebsocketPrior, "_connect_ws", lambda self: connection)

    prior = backends.TransformersWebsocketPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            server_url="wss://example.ngrok-free.app/ws",
            top_k=4,
        )
    )

    prior.accept_token(7)
    prior.reset()
    prior.next_logits()

    step_calls = [payload for payload in calls if payload.get("op") == "step"]
    assert step_calls[0]["token_id"] == prior.bos_token_id

    prior.close()
    assert connection.closed
    assert calls[-1]["op"] == "close"


def test_error_handling_and_timeout(monkeypatch) -> None:
    def error_responder(message: dict) -> dict:
        if message["op"] == "init":
            return _init_payload()
        if message["op"] == "step":
            return {
                "ok": False,
                "op": "step",
                "code": "bad-request",
                "error": "invalid token id",
            }
        raise AssertionError(f"unexpected op {message['op']}")

    error_conn = _FakeWsConnection(error_responder)
    monkeypatch.setattr(backends.TransformersWebsocketPrior, "_connect_ws", lambda self: error_conn)
    prior = backends.TransformersWebsocketPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            server_url="https://example.ngrok-free.app",
        )
    )

    with pytest.raises(RuntimeError, match="bad-request"):
        prior.next_logits()

    def timeout_responder(message: dict):
        if message["op"] == "init":
            return _init_payload()
        if message["op"] == "step":
            return TimeoutError("timed out")
        raise AssertionError(f"unexpected op {message['op']}")

    timeout_conn = _FakeWsConnection(timeout_responder)
    monkeypatch.setattr(backends.TransformersWebsocketPrior, "_connect_ws", lambda self: timeout_conn)
    timeout_prior = backends.TransformersWebsocketPrior(
        backends.LoadConfig(
            model_id="mistralai/Ministral-3-3B-Instruct-2512",
            server_url="https://example.ngrok-free.app",
            timeout_seconds=5,
        )
    )

    with pytest.raises(TimeoutError, match="timed out"):
        timeout_prior.next_logits()


class _ToyWsServer:
    def __init__(self) -> None:
        self.vocab_size = 16
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._context: list[int] = []
        self._table = {"a": 3, "b": 4, " ": 5, "c": 6}
        self._reverse = {v: k for k, v in self._table.items()}

    def handle(self, message: dict) -> dict:
        op = message["op"]
        if op == "init":
            self._context = []
            return {
                "ok": True,
                "op": "init",
                "model_id": "toy-model",
                "dtype": "float32",
                "vocab_size": self.vocab_size,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "max_context_tokens": 128,
                "max_top_k": self.vocab_size,
                "top_k": int(message.get("top_k", 4)),
            }
        if op == "reset":
            self._context = []
            return {"ok": True, "op": "reset"}
        if op == "tokenize":
            text = str(message["text"])
            return {"ok": True, "op": "tokenize", "token_ids": [self._table[ch] for ch in text]}
        if op == "detokenize":
            token_ids = [int(token_id) for token_id in message["token_ids"]]
            return {
                "ok": True,
                "op": "detokenize",
                "text": "".join(self._reverse[token_id] for token_id in token_ids),
            }
        if op == "step":
            token_id = int(message["token_id"])
            self._context.append(token_id)
            top_k = int(message.get("top_k", 4))
            main = (token_id + 1) % self.vocab_size
            second = (token_id + 2) % self.vocab_size
            third = (token_id + 3) % self.vocab_size
            entries = [
                {"token_id": main, "logprob": -0.05},
                {"token_id": second, "logprob": -0.40},
                {"token_id": third, "logprob": -0.80},
            ][: max(1, top_k)]
            return {"ok": True, "op": "step", "top_token_logprobs": entries}
        if op == "close":
            return {"ok": True, "op": "close"}
        raise AssertionError(f"unexpected op {op}")


def test_roundtrip_still_passes_with_websocket_prior(monkeypatch) -> None:
    toy_server = _ToyWsServer()
    connection = _FakeWsConnection(toy_server.handle)
    monkeypatch.setattr(backends.TransformersWebsocketPrior, "_connect_ws", lambda self: connection)

    prior = backends.load_prior(
        model_id="toy-model",
        server_url="https://example.ngrok-free.app",
        top_k=3,
        timeout_seconds=30.0,
        fallback_logit=-25.0,
        max_context=128,
    )

    text = "abba cab"
    compressed = compress_text(text=text, prior=prior, total_freq=1 << 12)
    restored = decompress_archive(archive=compressed.archive, prior=prior, verify_hash=True)

    assert compressed.header["backend"] == "transformers-ws"
    assert restored.text == text


def test_normalize_ws_url() -> None:
    assert backends._normalize_ws_url("https://example.ngrok-free.app") == "wss://example.ngrok-free.app/ws"
    assert backends._normalize_ws_url("http://localhost:8000") == "ws://localhost:8000/ws"
    assert backends._normalize_ws_url("wss://example.ngrok-free.app/ws") == "wss://example.ngrok-free.app/ws"

    with pytest.raises(ValueError, match="scheme"):
        backends._normalize_ws_url("example.ngrok-free.app")
