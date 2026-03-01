from __future__ import annotations

import requests

from minicrunch import wiki


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json_data: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> dict:
        return self._json_data


def test_fetch_wikipedia_extract_uses_rest_first(monkeypatch) -> None:
    def fake_get(url: str, **_: object) -> _FakeResponse:
        assert url.startswith("https://en.wikipedia.org/api/rest_v1/page/plain/")
        return _FakeResponse(text="rest-content")

    monkeypatch.setattr(wiki.requests, "get", fake_get)

    assert wiki.fetch_wikipedia_extract("Large language model") == "rest-content"


def test_fetch_wikipedia_extract_falls_back_to_mediawiki_api(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_get(url: str, **_: object) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(status_code=403)
        assert url == "https://en.wikipedia.org/w/api.php"
        return _FakeResponse(
            json_data={
                "query": {
                    "pages": [
                        {
                            "title": "Large language model",
                            "extract": "api-content",
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(wiki.requests, "get", fake_get)

    assert wiki.fetch_wikipedia_extract("Large language model") == "api-content"


def test_fetch_wikipedia_extract_reports_dual_failure(monkeypatch) -> None:
    def fake_get(_: str, **__: object) -> _FakeResponse:
        return _FakeResponse(status_code=403)

    monkeypatch.setattr(wiki.requests, "get", fake_get)

    try:
        wiki.fetch_wikipedia_extract("Large language model")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        message = str(exc)
        assert "REST page/plain failed" in message
        assert "MediaWiki API failed" in message
