#!/usr/bin/env python3
"""MiniCrunch Web UI — Wormhole-style file sharing with LLM compression.

Usage:
    python scripts/web_server.py --server-url ws://localhost:8000/ws --port 8080
"""
from __future__ import annotations

import base64
import gzip
import html as _html
import os
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
import zstandard as zstd

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    raise SystemExit(
        "Install server extras: pip install 'fastapi>=0.115' 'uvicorn>=0.30'"
    )

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minicrunch.backends import load_prior
from minicrunch.codec import compress_text, decompress_archive

# ── Config ─────────────────────────────────────────────────────────────────────
_cfg: dict = {
    "server_url": os.environ.get("MINICRUNCH_SERVER_URL", "ws://localhost:8000/ws"),
    "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
    "top_k": int(os.environ.get("MINICRUNCH_TOP_K", "64")),
    "max_context": int(os.environ.get("MINICRUNCH_MAX_CONTEXT", "4096")),
}

_STORE = Path(tempfile.gettempdir()) / "minicrunch_web"
_STORE.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_files: dict[str, dict] = {}

app = FastAPI(title="MiniCrunch")


def _pct_of_original(size: int, original: int) -> float:
    if original <= 0:
        return 0.0
    return (size / original) * 100.0


def _reduction_pct(size: int, original: int) -> float:
    return max(0.0, 100.0 - _pct_of_original(size=size, original=original))


def _build_benchmark_stats(source_bytes: bytes, minicrunch_size: int) -> dict:
    original_size = len(source_bytes)
    gzip_size = len(gzip.compress(source_bytes, compresslevel=9))
    zstd_size = len(zstd.ZstdCompressor(level=19).compress(source_bytes))

    minicrunch_reduction = _reduction_pct(size=minicrunch_size, original=original_size)
    gzip_reduction = _reduction_pct(size=gzip_size, original=original_size)
    zstd_reduction = _reduction_pct(size=zstd_size, original=original_size)

    return {
        "minicrunch_reduction_pct": minicrunch_reduction,
        "gzip_reduction_pct": gzip_reduction,
        "zstd_reduction_pct": zstd_reduction,
        "vs_gzip_pp": minicrunch_reduction - gzip_reduction,
        "vs_zstd_pp": minicrunch_reduction - zstd_reduction,
    }


def _api_base_url(server_url: str) -> str:
    parsed = urlparse(server_url.strip())
    if not parsed.scheme:
        raise ValueError("Server URL must include a scheme.")

    scheme = parsed.scheme
    if scheme == "ws":
        scheme = "http"
    elif scheme == "wss":
        scheme = "https"

    path = parsed.path or ""
    if path.endswith("/ws"):
        path = path[: -len("/ws")]

    return urlunparse((scheme, parsed.netloc, path.rstrip("/"), "", "", ""))


def _remote_compress(text: str, *, top_k: int, max_context: int) -> dict:
    base = _api_base_url(_cfg["server_url"])
    response = requests.post(
        f"{base}/api/compress",
        json={
            "text": text,
            "top_k": int(top_k),
            "max_context": int(max_context),
            "fallback_logit": -50.0,
            "total_freq": 1 << 20,
        },
        timeout=600,
    )
    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(f"Remote compress failed ({response.status_code}): {detail}")
    payload = response.json()
    archive_b64 = payload.get("archive_b64")
    if not isinstance(archive_b64, str):
        raise RuntimeError("Remote compress response missing `archive_b64`.")
    archive = base64.b64decode(archive_b64)
    return {
        "archive": archive,
        "payload_bits": int(payload.get("payload_bits", 0)),
        "token_count": int(payload.get("token_count", 0)),
        "header": payload.get("header") or {},
    }


def _remote_decompress(archive: bytes, *, top_k: int, max_context: int) -> str:
    base = _api_base_url(_cfg["server_url"])
    response = requests.post(
        f"{base}/api/decompress",
        json={
            "archive_b64": base64.b64encode(archive).decode("ascii"),
            "top_k": int(top_k),
            "max_context": int(max_context),
            "fallback_logit": -50.0,
            "verify_hash": True,
        },
        timeout=600,
    )
    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(f"Remote decompress failed ({response.status_code}): {detail}")
    payload = response.json()
    text = payload.get("text")
    if not isinstance(text, str):
        raise RuntimeError("Remote decompress response missing `text`.")
    return text


def _make_prior():
    return load_prior(
        model_id=_cfg["model_id"],
        server_url=_cfg["server_url"],
        top_k=int(_cfg["top_k"]),
        timeout_seconds=120.0,
        fallback_logit=-50.0,
        max_context=int(_cfg["max_context"]),
    )


# ── Background Jobs ────────────────────────────────────────────────────────────
def _compress_job(job_id: str, text: str, original_name: str) -> None:
    try:
        _jobs[job_id]["status"] = "compressing"
        try:
            remote = _remote_compress(
                text,
                top_k=int(_cfg["top_k"]),
                max_context=int(_cfg["max_context"]),
            )
            archive = remote["archive"]
            payload_bits = int(remote["payload_bits"])
            token_count = int(remote["token_count"])
            header = remote.get("header") or {}
            original_size = int(header.get("original_bytes", len(text.encode("utf-8"))))
        except Exception:
            # Backward-compatible fallback when remote API is unavailable.
            prior = _make_prior()
            result = compress_text(text=text, prior=prior)
            archive = result.archive
            payload_bits = result.payload_bits
            token_count = result.token_count
            header = result.header
            original_size = int(result.header.get("original_bytes", len(text.encode("utf-8"))))

        source_bytes = text.encode("utf-8")
        benchmark = _build_benchmark_stats(
            source_bytes=source_bytes,
            minicrunch_size=len(archive),
        )
        fid = str(uuid.uuid4())
        path = _STORE / f"{fid}.mcz"
        path.write_bytes(archive)
        _files[fid] = {
            "mcz_path": str(path),
            "original_name": original_name,
            "compressed_size": len(archive),
            "original_size": original_size,
            "token_count": token_count,
            "benchmark": benchmark,
        }
        _jobs[job_id] = {
            "status": "done",
            "file_id": fid,
            "result": {
                "file_id": fid,
                "original_name": original_name,
                "compressed_size": len(archive),
                "original_size": original_size,
                "token_count": token_count,
                "payload_bits": payload_bits,
                "benchmark": benchmark,
            },
        }
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}


def _decompress_job(job_id: str, file_id: str) -> None:
    try:
        _jobs[job_id]["status"] = "decompressing"
        info = _files.get(file_id)
        if not info:
            raise ValueError("File not found")
        archive = Path(info["mcz_path"]).read_bytes()
        try:
            text = _remote_decompress(
                archive,
                top_k=int(_cfg["top_k"]),
                max_context=int(_cfg["max_context"]),
            )
        except Exception:
            # Backward-compatible fallback when remote API is unavailable.
            prior = _make_prior()
            result = decompress_archive(archive=archive, prior=prior)
            text = result.text

        out_path = _STORE / f"{job_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        _jobs[job_id] = {
            "status": "done",
            "out_path": str(out_path),
            "filename": info["original_name"],
        }
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_UPLOAD_PAGE)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    fname = file.filename or "file.txt"
    if not fname.lower().endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported")
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be valid UTF-8 text")
    jid = str(uuid.uuid4())
    _jobs[jid] = {"status": "pending"}
    threading.Thread(
        target=_compress_job, args=(jid, text, fname), daemon=True
    ).start()
    return JSONResponse({"job_id": jid})


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


@app.get("/d/{file_id}", response_class=HTMLResponse)
async def dl_page(file_id: str):
    info = _files.get(file_id)
    if not info:
        return HTMLResponse(_NOT_FOUND_PAGE, status_code=404)
    return HTMLResponse(_render_download_page(file_id, info))


@app.post("/start-download/{file_id}")
async def start_download(file_id: str):
    if file_id not in _files:
        raise HTTPException(404, "File not found")
    jid = str(uuid.uuid4())
    _jobs[jid] = {"status": "pending"}
    threading.Thread(
        target=_decompress_job, args=(jid, file_id), daemon=True
    ).start()
    return JSONResponse({"job_id": jid})


@app.get("/get-file/{job_id}")
async def get_file(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(400, f"Job not complete: {job.get('status')}")
    out = Path(job["out_path"])
    if not out.exists():
        raise HTTPException(404, "Output file missing")
    return FileResponse(
        path=str(out),
        filename=job["filename"],
        media_type="text/plain; charset=utf-8",
    )


# ── Shared CSS ─────────────────────────────────────────────────────────────────
_SHARED_CSS = """
  *, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }

  :root {
    --bg: #080808;
    --mistral-orange: #ff7a00;
    --mistral-orange-light: #ffb066;
    --mistral-orange-deep: #d95e00;
    --mistral-ink: #141c2e;
    --mistral-blue: #385d9f;
    --mistral-blue-light: #6f8dc2;
  }

  html, body {
    height: 100%;
    background: var(--bg);
    color: #fff;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  /* ─── Pacman Background ─── */
  #pac-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
  }
  .pac-dot {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle, var(--mistral-blue-light) 0%, var(--mistral-blue) 100%);
    box-shadow: 0 0 8px rgba(56, 93, 159, 0.28);
    transform: translate(-50%, -50%);
  }
  .pac-man {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(
      circle at 35% 32%,
      var(--mistral-orange-light) 0%,
      var(--mistral-orange) 60%,
      var(--mistral-orange-deep) 100%
    );
    transform: translate(-50%, -50%);
    filter: drop-shadow(0 0 10px rgba(255, 122, 0, 0.35));
  }
  .pac-man::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 58%;
    height: 58%;
    background: var(--bg);
    clip-path: polygon(0 50%, 100% 0, 100% 100%);
    transform-origin: left center;
    transform: translate(0, -50%) rotate(32deg);
    animation: pac-mouth 0.26s ease-in-out infinite;
  }
  .pac-man::after {
    content: '';
    position: absolute;
    top: 24%;
    left: 60%;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--mistral-ink);
    opacity: 0.95;
  }

  /* ─── Layout ─── */
  .page {
    position: relative;
    z-index: 10;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 20px;
  }

  /* ─── Header ─── */
  .logo {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #FFD700;
    text-shadow: 0 0 40px rgba(255,215,0,0.4);
    margin-bottom: 6px;
    text-align: center;
  }
  .tagline {
    font-size: 13px;
    color: #484848;
    letter-spacing: 0.4px;
    text-align: center;
    margin-bottom: 44px;
  }

  /* ─── Card ─── */
  .card {
    background: rgba(18, 18, 18, 0.92);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 28px;
    padding: 60px 52px;
    width: min(540px, calc(100vw - 40px));
    text-align: center;
    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);
    box-shadow:
      0 0 0 1px rgba(255,215,0,0.04),
      0 24px 80px rgba(0,0,0,0.6),
      0 4px 20px rgba(0,0,0,0.4);
    animation: card-in 0.55s cubic-bezier(0.23, 1, 0.32, 1) both;
  }
  @keyframes card-in {
    from { opacity: 0; transform: translateY(24px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1);    }
  }
  .card.drag-over {
    border-color: rgba(255,215,0,0.45);
    background: rgba(255,215,0,0.03);
  }

  /* ─── Upload Icon ─── */
  .upload-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 22px;
    background: rgba(255,215,0,0.07);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1.5px solid rgba(255,215,0,0.15);
  }
  .upload-icon svg { width: 30px; height: 30px; fill: #FFD700; }

  /* ─── Download Icon ─── */
  .download-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 22px;
    background: rgba(78,255,145,0.07);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1.5px solid rgba(78,255,145,0.18);
  }
  .download-icon svg { width: 30px; height: 30px; fill: #4eff91; }

  /* ─── Typography ─── */
  .card-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.4px;
    margin-bottom: 8px;
  }
  .card-sub {
    font-size: 14px;
    color: #5a5a5a;
    margin-bottom: 30px;
    line-height: 1.6;
  }
  .card-sub code {
    background: rgba(255,255,255,0.07);
    padding: 1px 7px;
    border-radius: 5px;
    font-size: 12px;
    color: #999;
  }
  .file-name {
    font-size: 18px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
    word-break: break-all;
  }
  .file-meta {
    font-size: 13px;
    color: #555;
    margin-bottom: 30px;
  }

  /* ─── Buttons ─── */
  .btn-primary {
    background: #FFD700;
    color: #080808;
    border: none;
    border-radius: 14px;
    padding: 14px 36px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: -0.2px;
    transition: opacity 0.15s, transform 0.12s;
  }
  .btn-primary:hover  { opacity: 0.86; transform: translateY(-2px); }
  .btn-primary:active { transform: translateY(0); opacity: 1; }

  .btn-green {
    background: #4eff91;
    color: #080808;
    border: none;
    border-radius: 14px;
    padding: 14px 36px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: -0.2px;
    transition: opacity 0.15s, transform 0.12s;
  }
  .btn-green:hover  { opacity: 0.86; transform: translateY(-2px); }
  .btn-green:active { transform: translateY(0); opacity: 1; }

  .btn-secondary {
    background: rgba(255,255,255,0.07);
    color: #bbb;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 11px 28px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 16px;
    transition: background 0.15s;
  }
  .btn-secondary:hover { background: rgba(255,255,255,0.11); }

  /* ─── Progress ─── */
  .progress-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
  }
  .mini-pac {
    position: relative;
    display: inline-block;
    width: 52px;
    height: 52px;
    background: radial-gradient(
      circle at 35% 32%,
      var(--mistral-orange-light) 0%,
      var(--mistral-orange) 60%,
      var(--mistral-orange-deep) 100%
    );
    border-radius: 50%;
    box-shadow: 0 0 22px rgba(255, 122, 0, 0.33);
  }
  .mini-pac::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    width: 58%;
    height: 58%;
    background: var(--bg);
    clip-path: polygon(0 50%, 100% 0, 100% 100%);
    transform-origin: left center;
    transform: translate(0, -50%) rotate(32deg);
    animation: pac-mouth 0.28s ease-in-out infinite;
  }
  .mini-pac::after {
    content: '';
    position: absolute;
    top: 24%;
    left: 60%;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--mistral-ink);
    opacity: 0.95;
  }
  @keyframes pac-mouth {
    0%, 100% { transform: translate(0, -50%) rotate(32deg); }
    50%       { transform: translate(0, -50%) rotate(8deg); }
  }
  .progress-label {
    font-size: 15px;
    color: #888;
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1;   }
  }

  /* ─── Done / Link ─── */
  .done-check {
    font-size: 42px;
    line-height: 1;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 12px rgba(78,255,145,0.5));
  }
  .link-row {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 13px 16px;
    margin: 14px 0 8px;
    text-align: left;
  }
  .link-text {
    flex: 1;
    font-size: 13px;
    color: #FFD700;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  }
  .copy-btn {
    background: rgba(255,215,0,0.1);
    color: #FFD700;
    border: 1px solid rgba(255,215,0,0.18);
    border-radius: 9px;
    padding: 6px 15px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    letter-spacing: 0.3px;
    transition: background 0.15s;
  }
  .copy-btn:hover { background: rgba(255,215,0,0.2); }
  .link-hint { font-size: 12px; color: #3c3c3c; }
  .benchmark-box {
    margin-top: 14px;
    padding: 12px 14px;
    border-radius: 12px;
    border: 1px solid rgba(255, 122, 0, 0.2);
    background:
      linear-gradient(180deg, rgba(255, 122, 0, 0.08) 0%, rgba(20, 28, 46, 0.18) 100%);
    text-align: left;
  }
  .benchmark-title {
    color: #ffcf9f;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .benchmark-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 13px;
    color: #a9b7d0;
    line-height: 1.5;
  }
  .benchmark-row strong { color: #fff; }
  .benchmark-delta { color: #ffcf9f; }

  /* ─── Error ─── */
  .error-icon  { font-size: 38px; color: #ff6b6b; margin-bottom: 12px; }
  .error-text  { font-size: 14px; color: #ff8a8a; line-height: 1.5; }

  /* ─── Footer ─── */
  .footer {
    margin-top: 36px;
    font-size: 12px;
    color: #2e2e2e;
    text-align: center;
  }

  #file-input { display: none; }
"""

# ── Shared Pacman JS ────────────────────────────────────────────────────────────
_PACMAN_BG_JS = """
(function () {
  var bg = document.getElementById('pac-bg');
  if (!bg) return;

  var ROWS         = 6;
  var DOT_SPACING  = 56;   // px between dots
  var PAC_SIZE     = 26;   // px
  var DOT_SIZE     = 7;    // px
  var PAC_DUR      = 13;   // seconds to cross screen
  var W            = window.innerWidth || 1200;
  var numDots      = Math.ceil(W / DOT_SPACING) + 2;

  var styleEl  = document.createElement('style');
  var cssRules = [];

  cssRules.push(
    '@keyframes pac-move {' +
    '  from { left: -40px; }' +
    '  to   { left: calc(100vw + 40px); }' +
    '}'
  );

  for (var r = 0; r < ROWS; r++) {
    var yPct     = ((r + 0.5) / ROWS * 100).toFixed(2);
    var rowOff   = (r / ROWS * PAC_DUR).toFixed(3);

    /* Pacman */
    var pac = document.createElement('div');
    pac.className = 'pac-man';
    pac.style.cssText =
      'top:' + yPct + 'vh;' +
      'width:' + PAC_SIZE + 'px;' +
      'height:' + PAC_SIZE + 'px;' +
      'opacity:0.52;' +
      'animation:pac-move ' + PAC_DUR + 's linear -' + rowOff + 's infinite;';
    bg.appendChild(pac);

    /* Dots */
    for (var c = 0; c < numDots; c++) {
      var xPct    = (c * DOT_SPACING / W * 100).toFixed(2);
      var eatFrac = Math.min(c * DOT_SPACING / W, 0.999);
      var eatPct  = (eatFrac * 100).toFixed(1);
      var endPct  = Math.min(parseFloat(eatPct) + 2.5, 99).toFixed(1);
      var aName   = 'de_' + r + '_' + c;

      cssRules.push(
        '@keyframes ' + aName + ' {' +
        '  0%,' + eatPct + '% { opacity:0.32; }' +
        '  ' + endPct + '%   { opacity:0;    }' +
        '  99.9%              { opacity:0;    }' +
        '  100%               { opacity:0.32; }' +
        '}'
      );

      var dot = document.createElement('div');
      dot.className = 'pac-dot';
      dot.style.cssText =
        'left:' + xPct + 'vw;' +
        'top:' + yPct + 'vh;' +
        'width:' + DOT_SIZE + 'px;' +
        'height:' + DOT_SIZE + 'px;' +
        'animation:' + aName + ' ' + PAC_DUR + 's linear -' + rowOff + 's infinite;';
      bg.appendChild(dot);
    }
  }

  styleEl.textContent = cssRules.join('\\n');
  document.head.appendChild(styleEl);
})();
"""

# ── Upload Page ────────────────────────────────────────────────────────────────
_UPLOAD_PAGE = (
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MiniCrunch — LLM Text Compression</title>
  <style>"""
    + _SHARED_CSS
    + """
  </style>
</head>
<body>

<div id="pac-bg"></div>

<div class="page">
  <div class="logo">MiniCrunch</div>
  <div class="tagline">LLM-powered lossless text compression &middot; share compressed files via link</div>

  <div class="card" id="card">
    <input type="file" id="file-input" accept=".txt">

    <!-- Default state -->
    <div id="state-default">
      <div class="upload-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/>
        </svg>
      </div>
      <div class="card-title">Drop a text file here</div>
      <div class="card-sub">
        Drag &amp; drop a <code>.txt</code> file, or click to browse.<br>
        Your file will be compressed with an LLM and a share link generated.
      </div>
      <button class="btn-primary" onclick="document.getElementById('file-input').click()">
        Choose file
      </button>
    </div>

    <!-- Progress state -->
    <div id="state-progress" style="display:none">
      <div class="progress-wrap">
        <span class="mini-pac"></span>
        <div class="progress-label" id="progress-label">Compressing&hellip;</div>
      </div>
    </div>

    <!-- Done state -->
    <div id="state-done" style="display:none">
      <div class="done-check">&#10003;</div>
      <div class="card-title" style="color:#4eff91">Compressed!</div>
      <div class="card-sub" id="done-meta" style="margin-bottom:4px"></div>
      <div class="benchmark-box" id="benchmark-box" style="display:none">
        <div class="benchmark-title">Reduction vs benchmark</div>
        <div class="benchmark-row"><span>MiniCrunch</span><strong id="bench-minicrunch">--</strong></div>
        <div class="benchmark-row"><span>gzip -9</span><strong id="bench-gzip">--</strong></div>
        <div class="benchmark-row"><span>zstd -19</span><strong id="bench-zstd">--</strong></div>
        <div class="benchmark-row benchmark-delta"><span>Lead vs gzip / zstd</span><strong id="bench-delta">--</strong></div>
      </div>
      <div class="link-row">
        <span class="link-text" id="share-link-text"></span>
        <button class="copy-btn" id="copy-btn" onclick="copyLink()">Copy</button>
      </div>
      <div class="link-hint">Send this link to anyone to let them download the file</div>
    </div>

    <!-- Error state -->
    <div id="state-error" style="display:none">
      <div class="error-icon">&#9888;</div>
      <div class="error-text" id="error-text">Something went wrong.</div>
      <button class="btn-secondary" onclick="resetUI()">Try again</button>
    </div>
  </div>

  <div class="footer">Files are stored temporarily on this server &middot; MiniCrunch</div>
</div>

<script>
"""
    + _PACMAN_BG_JS
    + """

var fileInput = document.getElementById('file-input');
var card      = document.getElementById('card');
var shareUrl  = '';
var pollTimer = null;

function triggerFileDownload(jobId) {
  var frame = document.getElementById('download-frame');
  if (!frame) {
    frame = document.createElement('iframe');
    frame.id = 'download-frame';
    frame.style.display = 'none';
    document.body.appendChild(frame);
  }
  frame.src = '/get-file/' + jobId + '?ts=' + Date.now();
}

function formatPct(value) {
  return (typeof value === 'number' && isFinite(value)) ? value.toFixed(1) + '%' : '--';
}

function formatDelta(value) {
  if (typeof value !== 'number' || !isFinite(value)) return '--';
  return (value >= 0 ? '+' : '') + value.toFixed(1) + ' pp';
}

/* ── Drag-and-drop ── */
document.addEventListener('dragover', function(e){ e.preventDefault(); card.classList.add('drag-over'); });
document.addEventListener('dragleave', function(){ card.classList.remove('drag-over'); });
document.addEventListener('drop', function(e){
  e.preventDefault();
  card.classList.remove('drag-over');
  var f = e.dataTransfer && e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', function(){
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
  fileInput.value = '';
});

/* ── File handling ── */
function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.txt')) {
    setState('error');
    document.getElementById('error-text').textContent = 'Please select a .txt file.';
    return;
  }
  setState('progress');
  document.getElementById('progress-label').textContent = 'Uploading\u2026';

  var fd = new FormData();
  fd.append('file', file);

  fetch('/upload', { method: 'POST', body: fd })
    .then(function(r) {
      if (!r.ok) return r.json().then(function(e){ throw new Error(e.detail || 'Upload failed'); });
      return r.json();
    })
    .then(function(data) {
      document.getElementById('progress-label').textContent = 'Compressing with LLM\u2026';
      pollJob(data.job_id, 'compress');
    })
    .catch(function(err) {
      setState('error');
      document.getElementById('error-text').textContent = 'Upload error: ' + err.message;
    });
}

/* ── Job polling ── */
function pollJob(jobId, mode) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(function() {
    fetch('/job/' + jobId)
      .then(function(r){ return r.json(); })
      .then(function(job) {
        if (job.status === 'done') {
          clearInterval(pollTimer);
          if (mode === 'compress') {
            var r = job.result;
            shareUrl = window.location.origin + '/d/' + r.file_id;
            var origKB = (r.original_size  / 1024).toFixed(1);
            var compKB = (r.compressed_size / 1024).toFixed(1);
            var sourceSize = r.original_size > 0 ? r.original_size : 1;
            var ratioPct = (r.compressed_size / sourceSize) * 100;
            var pct = ratioPct.toFixed(1);
            var benchmark = r.benchmark || {};
            var defaultReduction = r.original_size > 0 ? (100 - ratioPct) : 0;
            var miniReduction = (typeof benchmark.minicrunch_reduction_pct === 'number')
              ? benchmark.minicrunch_reduction_pct
              : defaultReduction;
            document.getElementById('done-meta').textContent =
              r.original_name + ' \u00b7 ' + origKB + ' KB \u2192 ' + compKB + ' KB (' + pct + '%) \u00b7 ' + r.token_count.toLocaleString() + ' tokens';
            document.getElementById('bench-minicrunch').textContent = formatPct(miniReduction);
            document.getElementById('bench-gzip').textContent = formatPct(benchmark.gzip_reduction_pct);
            document.getElementById('bench-zstd').textContent = formatPct(benchmark.zstd_reduction_pct);
            document.getElementById('bench-delta').textContent =
              formatDelta(benchmark.vs_gzip_pp) + ' / ' + formatDelta(benchmark.vs_zstd_pp);
            document.getElementById('benchmark-box').style.display = '';
            document.getElementById('share-link-text').textContent = shareUrl;
            setState('done');
          } else {
            setState('default');
            triggerFileDownload(jobId);
          }
        } else if (job.status === 'error') {
          clearInterval(pollTimer);
          setState('error');
          document.getElementById('error-text').textContent = job.error || 'Compression failed.';
        }
      })
      .catch(function(){});
  }, 2000);
}

/* ── UI helpers ── */
function setState(s) {
  ['default','progress','done','error'].forEach(function(id){
    var el = document.getElementById('state-' + id);
    if (el) el.style.display = (s === id) ? '' : 'none';
  });
}

function resetUI() {
  if (pollTimer) clearInterval(pollTimer);
  document.getElementById('benchmark-box').style.display = 'none';
  setState('default');
}

function copyLink() {
  navigator.clipboard.writeText(shareUrl).then(function() {
    var btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!';
    btn.style.color = '#4eff91';
    setTimeout(function(){ btn.textContent = 'Copy'; btn.style.color = ''; }, 2500);
  }).catch(function() {
    var el = document.createElement('textarea');
    el.value = shareUrl;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
  });
}
</script>
</body>
</html>"""
)

# ── Download Page ──────────────────────────────────────────────────────────────
_NOT_FOUND_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MiniCrunch — Not Found</title>
  <style>
    body { background:#080808; color:#fff; font-family:sans-serif;
           display:flex; align-items:center; justify-content:center;
           min-height:100vh; flex-direction:column; gap:12px; }
    .logo { font-size:28px; font-weight:800; color:#FFD700; }
    p { color:#555; font-size:14px; }
  </style>
</head>
<body>
  <div class="logo">MiniCrunch</div>
  <p>This file link has expired or does not exist.</p>
  <a href="/" style="color:#FFD700; font-size:14px;">Send a new file</a>
</body>
</html>"""


def _render_download_page(file_id: str, info: dict) -> str:
    safe_name = _html.escape(info["original_name"])
    orig_kb = f"{info['original_size'] / 1024:.1f}"
    comp_kb = f"{info['compressed_size'] / 1024:.1f}"
    ratio_value = (
        (info["compressed_size"] / info["original_size"]) * 100.0
        if info["original_size"]
        else 0.0
    )
    ratio = f"{ratio_value:.1f}"
    benchmark = info.get("benchmark") or {}

    def _as_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        return None

    minicrunch_reduction = _as_float(benchmark.get("minicrunch_reduction_pct"))
    if minicrunch_reduction is None:
        minicrunch_reduction = max(0.0, 100.0 - ratio_value)
    gzip_reduction = _as_float(benchmark.get("gzip_reduction_pct"))
    zstd_reduction = _as_float(benchmark.get("zstd_reduction_pct"))

    vs_gzip = _as_float(benchmark.get("vs_gzip_pp"))
    if vs_gzip is None and gzip_reduction is not None:
        vs_gzip = minicrunch_reduction - gzip_reduction

    vs_zstd = _as_float(benchmark.get("vs_zstd_pp"))
    if vs_zstd is None and zstd_reduction is not None:
        vs_zstd = minicrunch_reduction - zstd_reduction

    min_reduction_text = f"{minicrunch_reduction:.1f}%"
    gzip_reduction_text = f"{gzip_reduction:.1f}%" if gzip_reduction is not None else "n/a"
    zstd_reduction_text = f"{zstd_reduction:.1f}%" if zstd_reduction is not None else "n/a"
    vs_gzip_text = f"{vs_gzip:+.1f} pp" if vs_gzip is not None else "n/a"
    vs_zstd_text = f"{vs_zstd:+.1f} pp" if vs_zstd is not None else "n/a"
    lead_text = f"{vs_gzip_text} / {vs_zstd_text}"
    tokens = f"{info['token_count']:,}"
    safe_fid = _html.escape(file_id)

    page = (
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MiniCrunch \u2014 Download """
        + safe_name
        + """</title>
  <style>"""
        + _SHARED_CSS
        + """
  </style>
</head>
<body>

<div id="pac-bg"></div>

<div class="page">
  <div class="logo">MiniCrunch</div>
  <div class="tagline">Someone shared a compressed file with you</div>

  <div class="card" id="card">

    <!-- Default state -->
    <div id="state-default">
      <div class="download-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M19 9h-4V3H9v6H5l7 7 7-7zm-14 9v2h14v-2H5z"/>
        </svg>
      </div>
      <div class="file-name">"""
        + safe_name
        + """</div>
      <div class="file-meta">"""
        + comp_kb
        + """ KB compressed &middot; """
        + orig_kb
        + """ KB original &middot; """
        + ratio
        + """% ratio &middot; """
        + tokens
        + """ tokens</div>
      <div class="benchmark-box">
        <div class="benchmark-title">Reduction vs benchmark</div>
        <div class="benchmark-row"><span>MiniCrunch</span><strong>"""
        + min_reduction_text
        + """</strong></div>
        <div class="benchmark-row"><span>gzip -9</span><strong>"""
        + gzip_reduction_text
        + """</strong></div>
        <div class="benchmark-row"><span>zstd -19</span><strong>"""
        + zstd_reduction_text
        + """</strong></div>
        <div class="benchmark-row benchmark-delta"><span>Lead vs gzip / zstd</span><strong>"""
        + lead_text
        + """</strong></div>
      </div>
      <button class="btn-green" onclick="startDownload('"""
        + safe_fid
        + """')">
        Download
      </button>
    </div>

    <!-- Progress state -->
    <div id="state-progress" style="display:none">
      <div class="progress-wrap">
        <span class="mini-pac"></span>
        <div class="progress-label" id="progress-label">Decompressing&hellip;</div>
      </div>
    </div>

    <!-- Error state -->
    <div id="state-error" style="display:none">
      <div class="error-icon">&#9888;</div>
      <div class="error-text" id="error-text">Something went wrong.</div>
      <button class="btn-secondary" onclick="resetDl()">Try again</button>
    </div>

  </div>

  <div class="footer">MiniCrunch &mdash; LLM-powered lossless text compression</div>
</div>

<script>
"""
        + _PACMAN_BG_JS
        + """

var pollTimer = null;

function triggerFileDownload(jobId) {
  var frame = document.getElementById('download-frame');
  if (!frame) {
    frame = document.createElement('iframe');
    frame.id = 'download-frame';
    frame.style.display = 'none';
    document.body.appendChild(frame);
  }
  frame.src = '/get-file/' + jobId + '?ts=' + Date.now();
}

function setState(s) {
  ['default','progress','error'].forEach(function(id){
    var el = document.getElementById('state-' + id);
    if (el) el.style.display = (s === id) ? '' : 'none';
  });
}

function resetDl() {
  if (pollTimer) clearInterval(pollTimer);
  setState('default');
}

function startDownload(fileId) {
  setState('progress');
  document.getElementById('progress-label').textContent = 'Decompressing with LLM\u2026';

  fetch('/start-download/' + fileId, { method: 'POST' })
    .then(function(r) {
      if (!r.ok) return r.json().then(function(e){ throw new Error(e.detail || 'Failed'); });
      return r.json();
    })
    .then(function(data) {
      pollJob(data.job_id);
    })
    .catch(function(err) {
      setState('error');
      document.getElementById('error-text').textContent = 'Error: ' + err.message;
    });
}

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(function() {
    fetch('/job/' + jobId)
      .then(function(r){ return r.json(); })
      .then(function(job) {
        if (job.status === 'done') {
          clearInterval(pollTimer);
          setState('default');
          triggerFileDownload(jobId);
        } else if (job.status === 'error') {
          clearInterval(pollTimer);
          setState('error');
          document.getElementById('error-text').textContent = job.error || 'Decompression failed.';
        }
      })
      .catch(function(){});
  }, 2000);
}
</script>
</body>
</html>"""
    )
    return page


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MiniCrunch Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--server-url",
        default=_cfg["server_url"],
        help=(
            "Transformers server URL. Supports ws(s)://.../ws for legacy WS mode, "
            "or http(s)://... for offloaded /api/compress + /api/decompress mode "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--model-id",
        default=_cfg["model_id"],
        help="Model ID to use for compression (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(_cfg["top_k"]),
        help="Top-k used for model scoring (default: %(default)s)",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=int(_cfg["max_context"]),
        help="Max context window used on model server (default: %(default)s)",
    )
    args = parser.parse_args()

    _cfg["server_url"] = args.server_url
    _cfg["model_id"] = args.model_id
    _cfg["top_k"] = max(1, int(args.top_k))
    _cfg["max_context"] = max(1, int(args.max_context))

    print("  MiniCrunch Web UI")
    print(f"  URL    : http://localhost:{args.port}")
    print(f"  Server : {args.server_url}")
    print(f"  Model  : {args.model_id}")
    print(f"  Top-k  : {_cfg['top_k']}")
    print(f"  MaxCtx : {_cfg['max_context']}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
