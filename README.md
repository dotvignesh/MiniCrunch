# MiniCrunch

LLM-based lossless text compression demo using a remote vLLM prior.

This project treats a language model as a shared prior between sender and receiver: each token is arithmetic-coded using the model's predicted distribution, so predictable text takes fewer bits.

Default model: `mistralai/Ministral-3-3B-Instruct-2512`

## What it does

- Compresses UTF-8 text with arithmetic coding driven by next-token probabilities.
- Decompresses exactly (byte-for-byte) when using the same model + vLLM settings.
- Benchmarks against `gzip -9` and `zstd -19`.
- Can fetch a Wikipedia article directly for a demo run.

## Setup (local laptop)

```bash
uv venv
uv sync
```

Dev tools:

```bash
uv sync --extra dev
```

## Run vLLM server (Colab single launch)

In Colab (GPU runtime), clone this repo and run:

```bash
pip install "vllm>=0.7" "fastapi>=0.115" "uvicorn>=0.30" "pyngrok>=7.2"
export NGROK_AUTHTOKEN="<your-ngrok-token>"
python scripts/vllm_http_server.py \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --tunnel ngrok \
  --print-config
```

The server prints a public URL:

```text
PUBLIC_URL=https://abc123.ngrok-free.app
```

Use that URL as `--vllm-url` from your laptop.

## CLI

### Benchmark (Wikipedia)

```bash
uv run minicrunch benchmark \
  --wikipedia-title "Large language model" \
  --vllm-url https://<your-tunnel-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512 \
  --output-archive llm.mcz \
  --output-decoded decoded.txt
```

### Compress a local file

```bash
uv run minicrunch compress \
  --input article.txt \
  --output article.mcz \
  --vllm-url https://<your-tunnel-url> \
  --model-id mistralai/Ministral-3-3B-Instruct-2512
```

### Decompress

```bash
uv run minicrunch decompress \
  --input article.mcz \
  --output roundtrip.txt \
  --vllm-url https://<your-tunnel-url>
```

## Notes on exact roundtrip

Arithmetic decoding must use the same prior configuration as encoding:

- Same model/tokenizer (`--model-id`).
- Same backend settings (`--vllm-top-k`, `--vllm-fallback-logit`).
- Same archive metadata and matching vLLM behavior.

Important: this vLLM mode uses top-K logprobs plus a fixed fallback logit for non-top-K tokens. It remains lossless for roundtrip when settings match, but compression quality may be lower than exact full-logit backends.

## Run tests

```bash
uv run pytest -q
```
