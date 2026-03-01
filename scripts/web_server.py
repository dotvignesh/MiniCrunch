#!/usr/bin/env python3
"""MiniCrunch Web UI — Wormhole-style file sharing with LLM compression.

Usage:
    python scripts/web_server.py --vllm-url http://localhost:8001 --port 8080
"""
from __future__ import annotations

import html as _html
import os
import tempfile
import threading
import uuid
from pathlib import Path

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
    "vllm_url": os.environ.get("MINICRUNCH_VLLM_URL", "http://localhost:8001"),
    "model_id": "mistralai/Ministral-3-3B-Instruct-2512",
}

_STORE = Path(tempfile.gettempdir()) / "minicrunch_web"
_STORE.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict] = {}
_files: dict[str, dict] = {}

app = FastAPI(title="MiniCrunch")


def _make_prior():
    return load_prior(
        model_id=_cfg["model_id"],
        vllm_url=_cfg["vllm_url"],
        vllm_top_k=256,
        vllm_timeout_seconds=120.0,
        vllm_fallback_logit=-50.0,
        vllm_max_context=8192,
    )


# ── Background Jobs ────────────────────────────────────────────────────────────
def _compress_job(job_id: str, text: str, original_name: str) -> None:
    try:
        _jobs[job_id]["status"] = "compressing"
        prior = _make_prior()
        result = compress_text(text=text, prior=prior)
        fid = str(uuid.uuid4())
        path = _STORE / f"{fid}.mcz"
        path.write_bytes(result.archive)
        _files[fid] = {
            "mcz_path": str(path),
            "original_name": original_name,
            "compressed_size": len(result.archive),
            "original_size": result.header["original_bytes"],
            "token_count": result.token_count,
        }
        _jobs[job_id] = {
            "status": "done",
            "file_id": fid,
            "result": {
                "file_id": fid,
                "original_name": original_name,
                "compressed_size": len(result.archive),
                "original_size": result.header["original_bytes"],
                "token_count": result.token_count,
                "payload_bits": result.payload_bits,
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
        prior = _make_prior()
        result = decompress_archive(archive=archive, prior=prior)
        out_path = _STORE / f"{job_id}.txt"
        out_path.write_text(result.text, encoding="utf-8")
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

  html, body {
    height: 100%;
    background: #080808;
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
    background: #FFD700;
    transform: translate(-50%, -50%);
  }
  .pac-man {
    position: absolute;
    border-radius: 50%;
    background: #FFD700;
    transform: translate(-50%, -50%);
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
    display: inline-block;
    width: 52px;
    height: 52px;
    background: #FFD700;
    border-radius: 50%;
    box-shadow: 0 0 20px rgba(255,215,0,0.35);
    animation: ui-chomp 0.28s linear infinite;
  }
  @keyframes ui-chomp {
    0%, 100% { clip-path: polygon(100% 40%, 0% 0%, 0% 100%, 100% 60%); }
    50%       { clip-path: polygon(100% 50%, 0% 0%, 0% 100%, 100% 50%); }
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
    '@keyframes pac-chomp {' +
    '  0%,100%{ clip-path: polygon(100% 40%, 0% 0%, 0% 100%, 100% 60%); }' +
    '  50%    { clip-path: polygon(100% 50%, 0% 0%, 0% 100%, 100% 50%); }' +
    '}' +
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
      'opacity:0.45;' +
      'animation:pac-chomp 0.26s linear infinite,' +
      '           pac-move ' + PAC_DUR + 's linear -' + rowOff + 's infinite;';
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
        '  0%,' + eatPct + '% { opacity:0.17; }' +
        '  ' + endPct + '%   { opacity:0;    }' +
        '  99.9%              { opacity:0;    }' +
        '  100%               { opacity:0.17; }' +
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
            var pct    = (r.compressed_size / r.original_size * 100).toFixed(1);
            document.getElementById('done-meta').textContent =
              r.original_name + ' \u00b7 ' + origKB + ' KB \u2192 ' + compKB + ' KB (' + pct + '%) \u00b7 ' + r.token_count.toLocaleString() + ' tokens';
            document.getElementById('share-link-text').textContent = shareUrl;
            setState('done');
          } else {
            window.location.href = '/get-file/' + jobId;
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
    ratio = (
        f"{info['compressed_size'] / info['original_size'] * 100:.1f}"
        if info["original_size"]
        else "0.0"
    )
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
          window.location.href = '/get-file/' + jobId;
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
        "--vllm-url",
        default=_cfg["vllm_url"],
        help="Base URL of the vLLM logits server (default: %(default)s)",
    )
    parser.add_argument(
        "--model-id",
        default=_cfg["model_id"],
        help="Model ID to use for compression (default: %(default)s)",
    )
    args = parser.parse_args()

    _cfg["vllm_url"] = args.vllm_url
    _cfg["model_id"] = args.model_id

    print(f"  MiniCrunch Web UI")
    print(f"  URL    : http://localhost:{args.port}")
    print(f"  vLLM   : {args.vllm_url}")
    print(f"  Model  : {args.model_id}")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
