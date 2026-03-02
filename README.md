# MiniCrunch

Lossless UTF-8 text compression powered by a *remote* language-model prior.

MiniCrunch treats "what would the model expect next?" as a probability distribution and feeds that into arithmetic coding. When the text is predictable, it costs fewer bits. When it's surprising, you pay for the surprise. Either way, decode is exact as long as both sides use the same settings.

## Why this is interesting

Most compressors look only at the bytes you've already seen. MiniCrunch also leverages a shared world model.

- Better compression on natural language and structured text that a model can strongly predict
- Works over a thin client: your laptop can compress/decompress while a GPU box does the scoring
- Archives include guardrails (backend/model/settings + SHA-256) so mismatches fail fast instead of silently corrupting output

## What's implemented here

This repo is intentionally small and demo-ready:

- A fast arithmetic codec for token IDs with a deterministic archive format (`.mcz`)
- A Transformers-based scoring server exposing WebSocket `/ws` (persistent sessions + KV cache) and HTTP `/api/compress` + `/api/decompress`
- A lightweight Web UI that does: upload `.txt` -> compress -> share link -> decompress/download

## Layout

- `src/minicrunch/arithmetic.py`: arithmetic encoder/decoder
- `src/minicrunch/distributions.py`: logits -> integer cumulative distribution
- `src/minicrunch/codec.py`: archive pack/unpack + compress/decompress flow
- `src/minicrunch/backends.py`: WebSocket prior client
- `src/minicrunch/cli.py`: `minicrunch` CLI
- `scripts/transformers_ws_server.py`: remote scoring server (GPU-friendly)
- `scripts/web_server.py`: Web UI (runs locally, points at remote scorer)

## Quick start (GPU server in Colab, client locally)

You'll run the model server on a GPU instance and point your local CLI/Web UI at its public URL.

### 1) Start the GPU scoring server (Colab)

1. In Colab: Runtime -> Change runtime type -> Hardware accelerator: GPU.
2. Pick a GPU with enough memory for your chosen model (16-24 GB recommended).

Then run (or upload this repo into the Colab filesystem and `cd` into it):

```bash
git clone <this-repo-url>
cd MiniCrunch

pip install "torch>=2.4" "transformers>=4.50" "fastapi>=0.115" "uvicorn>=0.30" "pyngrok>=7.2"
```

If you're using ngrok (recommended for a clean public URL):

```bash
export NGROK_AUTHTOKEN="<your-ngrok-token>"
```

Launch the server:

```bash
python scripts/transformers_ws_server.py \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --dtype bfloat16 \
  --max-context 8192 \
  --top-k-default 256 \
  --host 0.0.0.0 \
  --port 8000 \
  --tunnel ngrok \
  --print-config
```

Notes:

- If you're using the repo defaults, you can omit `--model-id` on both server and client (the default is `mistralai/Ministral-3-3B-Instruct-2512`).
- Decompress can omit `--model-id` to use the model stored in the archive header.

You'll see output like:

```text
PUBLIC_URL=https://abc123.ngrok-free.app
PUBLIC_WS_URL=wss://abc123.ngrok-free.app/ws
```

Keep this Colab cell running.

### 2) Install locally (CLI + optional Web UI)

On your machine:

```bash
uv venv
uv sync
```

If you want to run the Web UI locally:

```bash
uv sync --extra server
```

### 3) Connect your local client to the remote server

MiniCrunch accepts `http(s)` or `ws(s)` URLs.

- If you pass `https://abc123...`, the client will automatically use `wss://abc123.../ws`.
- If you pass `wss://abc123.../ws`, it's used as-is.

#### CLI: compress / decompress

Compress:

```bash
uv run minicrunch compress \
  --input article.txt \
  --output article.mcz \
  --server-url https://<your-public-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --top-k 256 \
  --timeout-seconds 60 \
  --fallback-logit -50 \
  --max-context 8192
```

Decompress:

```bash
uv run minicrunch decompress \
  --input article.mcz \
  --output roundtrip.txt \
  --server-url https://<your-public-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --top-k 256 \
  --timeout-seconds 60 \
  --fallback-logit -50 \
  --max-context 8192
```

#### Web UI: run locally, use the remote GPU for scoring

```bash
uv run python scripts/web_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --server-url https://<your-public-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512
```

Open `http://localhost:8080`.

## Web UI flow (the "send someone a file" path)

1. Upload a UTF-8 `.txt` file.
2. Wait for compression to finish.
3. Copy the generated link (`/d/<file_id>`).
4. Open it elsewhere and click download.
5. The recipient triggers decompression and gets the exact original text back.

## A simple demo script

1. Upload a known text sample and show the compression reduction stats in the UI (it compares against gzip and zstd).
2. Open the share link in a fresh browser profile and download the recovered text.
3. Compare original vs downloaded file byte-for-byte.

## How the codec stays exact

Compression and decompression are exact when these match:

- model (`--model-id`)
- backend (`transformers-ws`)
- `--top-k`, `--fallback-logit`, and `--max-context`
- arithmetic `--total-freq` (stored in the archive header)

Archives include a SHA-256 of the original UTF-8 bytes, so if the prior differs at decode time you get a hard failure (unless you explicitly disable verification).

## Practical defaults

These are good starting points for remote GPU scoring:

- server `--dtype bfloat16` (use `float16` if BF16 isn't supported)
- client `--top-k 256`
- client `--fallback-logit -50`
- client/server `--max-context 8192`

Tradeoffs:

- higher `top-k` can improve compression ratio but increases compute and payload size per token
- lower `top-k` is faster but can degrade compression

## Troubleshooting

### WebSocket timeouts

Increase `--timeout-seconds` on the client and check the tunnel is still up.

### CUDA OOM on the GPU instance

Use a smaller model, reduce `--max-context`, or switch dtype (`bfloat16`/`float16`).

### "Archive model != loaded model"

Decode must use the same model as encode. If you compressed with one model and try to decode with another, MiniCrunch will reject it by design.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```
