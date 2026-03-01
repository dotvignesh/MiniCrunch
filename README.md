# MiniCrunch

MiniCrunch is a hackathon-friendly demo of lossless text compression using a remote LLM prior.

The sender and receiver both query the same model for next-token probabilities. Those probabilities drive arithmetic coding, so predictable text costs fewer bits while remaining exactly recoverable.

Default model used in examples: `mistralai/Ministral-3-3B-Instruct-2512`

## Why this version exists

This repository now uses a Transformers-only serving path with a persistent WebSocket session protocol.

- No vLLM dependency in the runtime path
- Persistent session with KV cache (`past_key_values`) for fast incremental scoring
- Top-k next-token logprobs returned for each token step
- Same codec behavior and exact roundtrip guarantees as long as compression and decompression use matching settings

## Features

- Lossless UTF-8 text compression and decompression
- Remote model scoring over WebSocket (optimized for Colab + ngrok)
- CLI for compress, decompress, and benchmark
- Web UI for upload, compress, share-link generation, and download-time decompression
- Deterministic archive header metadata checks (backend + model guardrails)

## Repository layout

- `src/minicrunch/arithmetic.py`: arithmetic encoder/decoder
- `src/minicrunch/distributions.py`: logits -> integer cumulative distribution
- `src/minicrunch/codec.py`: archive pack/unpack + compress/decompress flow
- `src/minicrunch/backends.py`: Transformers WebSocket prior client
- `src/minicrunch/cli.py`: CLI entrypoint and command wiring
- `scripts/transformers_ws_server.py`: remote Transformers WebSocket scoring server
- `scripts/web_server.py`: MiniCrunch Web UI

## How compression works

### High-level flow

1. Input text is tokenized by the remote model tokenizer.
2. For each token position, MiniCrunch asks the model for a next-token distribution.
3. MiniCrunch converts that distribution into integer frequencies.
4. Arithmetic coding encodes the observed token ID into a compact bitstream.
5. The archive stores the bitstream plus metadata needed to verify deterministic decode assumptions.
6. Decompression replays the same model-driven distributions and decodes the exact original token sequence.

### Distribution detail (important)

MiniCrunch uses top-k next-token logprobs plus a fixed fallback logit for tokens outside top-k.

- Top-k tokens from server keep their returned logprobs.
- All other tokens get the configured fallback logit.
- Softmax over this full vector is then converted into integer frequencies.

This keeps runtime practical over remote links while preserving deterministic behavior if both sides use the same settings.

### Arithmetic coding detail

At each step:

- `logits_to_cumulative()` transforms logits to integer cumulative counts with total `total_freq`.
- Every token gets at least frequency 1 (no zero-probability symbols).
- Encoder narrows the interval using the selected token's cumulative range.
- Decoder performs the exact inverse using the same cumulative tables.

### Why roundtrip is exact

Roundtrip stays exact when all of the following match between encode/decode:

- same backend (`transformers-ws`)
- same remote model/tokenizer behavior
- same top-k/fallback/max-context settings
- same arithmetic parameters stored in archive metadata

If model/backend assumptions differ, hash verification or metadata checks fail fast.

## WebSocket protocol (primary API)

Endpoint: `/ws`

### Lifecycle

1. Client connects.
2. Client sends `init`.
3. Client may send `tokenize`, `step`, `detokenize`, `reset` repeatedly.
4. Client sends `close` and disconnects.

### Request ops

#### `init`

```json
{"op":"init","top_k":256,"max_context_tokens":8192}
```

#### `step`

`token_id` is the token that extends context before predicting the next token distribution.

```json
{"op":"step","token_id":1234,"top_k":256}
```

#### `tokenize`

```json
{"op":"tokenize","text":"hello world"}
```

#### `detokenize`

```json
{"op":"detokenize","token_ids":[101,202,303]}
```

#### `reset`

```json
{"op":"reset"}
```

#### `close`

```json
{"op":"close"}
```

### Success response shape

```json
{"ok":true,"op":"step","top_token_logprobs":[{"token_id":42,"logprob":-0.31}]}
```

### Error response shape

```json
{"ok":false,"op":"step","code":"runtime-error","error":"..."}
```

## Setup

## Local machine

### 1) Create environment and install

```bash
uv venv
uv sync
```

### 2) Install dev tools (optional)

```bash
uv sync --extra dev
```

### 3) Install server/web extras locally (optional)

Needed only if you plan to host the model server or run the Web UI from this machine.

```bash
uv sync --extra server
```

## Colab L4 remote server with ngrok (recommended for speed/cost)

### 1) Start a GPU runtime

Use Colab with an L4 (24 GB) GPU.

### 2) Clone repo and install dependencies

```bash
pip install "torch>=2.4" "transformers>=4.50" "fastapi>=0.115" "uvicorn>=0.30" "pyngrok>=7.2"
```

### 3) Set ngrok auth token

```bash
export NGROK_AUTHTOKEN="<your-ngrok-token>"
```

### 4) Launch server

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

Expected output includes:

```text
PUBLIC_URL=https://abc123.ngrok-free.app
PUBLIC_WS_URL=wss://abc123.ngrok-free.app/ws
```

You can pass either URL to MiniCrunch client `--server-url`.

- If you pass `https://...`, client auto-converts to `wss://.../ws`.
- If you pass `wss://.../ws`, it is used as-is.

## Fast defaults for Colab L4 (24 GB)

Recommended baseline:

- server `--dtype bfloat16`
- client `--top-k 256`
- client `--fallback-logit -50`
- client/server `--max-context 8192`
- keep a persistent process and session per compression/decompression job

Tradeoffs:

- higher `top-k` can improve compression quality but increases compute/network payload
- lower `top-k` is faster but may hurt compression ratio
- very small fallback logit can over-penalize non-top-k tail tokens

## CLI usage

## Benchmark (Wikipedia input)

```bash
uv run minicrunch benchmark \
  --wikipedia-title "Large language model" \
  --server-url https://<your-ngrok-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --top-k 256 \
  --timeout-seconds 60 \
  --fallback-logit -50 \
  --max-context 8192 \
  --output-archive llm.mcz \
  --output-decoded decoded.txt
```

## Compress local file

```bash
uv run minicrunch compress \
  --input article.txt \
  --output article.mcz \
  --server-url https://<your-ngrok-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --top-k 256 \
  --timeout-seconds 60 \
  --fallback-logit -50 \
  --max-context 8192
```

## Decompress archive

```bash
uv run minicrunch decompress \
  --input article.mcz \
  --output roundtrip.txt \
  --server-url https://<your-ngrok-url> \
  --top-k 256 \
  --timeout-seconds 60 \
  --fallback-logit -50 \
  --max-context 8192
```

## Web UI workflow (compression + share + decompression)

This section is for your hackathon demo path.

### What the Web UI does

- Accepts UTF-8 `.txt` upload
- Compresses using the same MiniCrunch codec and remote Transformers prior
- Stores `.mcz` archive in temporary server storage
- Generates a share link `/d/<file_id>`
- On open, recipient can trigger decompression and download the original `.txt`

### Run Web UI on your machine

Use the same remote Colab model server URL:

```bash
uv run python scripts/web_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --server-url https://<your-ngrok-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512
```

Open:

- `http://localhost:8080`

### Demo flow for judges

1. Upload a text file on the home page.
2. Wait until compression finishes.
3. Copy the generated share link (`/d/<file_id>`).
4. Open that link in another browser/session.
5. Click to start decompression and download recovered `.txt`.
6. Show original vs downloaded file byte-for-byte match.

### If you need publicly accessible Web UI

The built-in server is local by default. Expose port `8080` using your preferred tunnel to share the UI URL externally.

## Exactness checklist

Before comparing ratios/timings, make sure:

- same `--model-id`
- same `--top-k`
- same `--fallback-logit`
- same `--max-context`
- same backend (`transformers-ws`)

If these differ, exact verification can fail by design.

## Performance notes

Current bottlenecks in this architecture:

- one network round trip per token step (`op=step`)
- per-step full-vocab `log_softmax` and `topk` compute on server
- cache rebuild cost when context window rolls over
- JSON serialization overhead for message payloads

For hackathon demos, prioritize stability and reproducibility over aggressive micro-optimizations.

## Troubleshooting

### Timeout errors

Increase client `--timeout-seconds` and verify ngrok tunnel health.

### CUDA OOM on Colab

Reduce `--max-context`, use smaller model, or keep `--dtype bfloat16`.

### Model mismatch error during decompress

Pass the same model used during compress, or omit `--model-id` on decompress to use archive header model.

### Backend mismatch error

Archive created with different backend metadata. Re-compress using current `transformers-ws` path.

## Run tests and lint

```bash
uv run pytest -q
uv run ruff check .
```

## Hackathon pitch tips

- Measure compression ratio and tokens/sec live on a known text sample.
- Show end-to-end: upload -> compress -> share link -> decompress -> exact recovery.
- Highlight deterministic safety checks (header + SHA-256 verification).
- Explain that this is an LLM-prior codec, not a generic entropy coder alone.
