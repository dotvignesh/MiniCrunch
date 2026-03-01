from __future__ import annotations

from urllib.parse import quote

import requests

WIKIPEDIA_HEADERS = {
    "User-Agent": "MiniCrunch/0.1 (LLM compression demo; contact: local-runner)",
    "Accept": "text/plain, application/json;q=0.9, */*;q=0.1",
}


def _fetch_rest_plain(title_key: str, timeout_seconds: int) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/plain/{title_key}"
    response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text.strip()


def _fetch_mediawiki_extract(title: str, timeout_seconds: int) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": title,
    }
    response = requests.get(
        url,
        params=params,
        headers=WIKIPEDIA_HEADERS,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""

    page = pages[0]
    if "missing" in page:
        return ""
    return str(page.get("extract", "")).strip()


def fetch_wikipedia_extract(title: str, timeout_seconds: int = 30) -> str:
    title_key = quote(title.replace(" ", "_"), safe="")
    errors: list[str] = []

    try:
        text = _fetch_rest_plain(title_key, timeout_seconds)
        if text:
            return text
        errors.append("REST page/plain returned empty content")
    except requests.RequestException as exc:
        errors.append(f"REST page/plain failed: {exc}")

    try:
        text = _fetch_mediawiki_extract(title, timeout_seconds)
        if text:
            return text
        errors.append("MediaWiki API extract returned empty content")
    except requests.RequestException as exc:
        errors.append(f"MediaWiki API failed: {exc}")

    details = "; ".join(errors) if errors else "unknown error"
    raise ValueError(f"Wikipedia page '{title}' could not be fetched: {details}")
