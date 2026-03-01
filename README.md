# MiniCrunch

LLM-based lossless text compression demo.

This project treats a language model as a shared prior between sender and receiver: each token is arithmetic-coded using the model's predicted distribution, so highly predictable text takes fewer bits.

Default model: `mistralai/Ministral-3-3B-Instruct-2512`

## What it does

- Compresses UTF-8 text with arithmetic coding driven by next-token probabilities.
- Decompresses exactly (byte-for-byte) when using the same model/backend settings.
- Benchmarks against `gzip -9` and `zstd -19`.
- Can fetch a Wikipedia article directly for a demo run.

## Setup (uv)

```bash
uv venv
uv sync
```

Install dev tools:

```bash
uv sync --extra dev
```

First model load will download `mistralai/Ministral-3-3B-Instruct-2512` from Hugging Face.

On Apple Silicon (`--device mps`) and CPU, this FP8 checkpoint is loaded via a
compatibility path that lets `transformers` dequantize weights to `bf16` when
Triton kernels are not available.

## CLI

### Benchmark (Wikipedia)

```bash
uv run minicrunch benchmark \
  --wikipedia-title "Large language model" \
  --device auto \
  --dtype auto \
  --output-archive llm.mcz \
  --output-decoded decoded.txt
```

### Compress a local file

```bash
uv run minicrunch compress \
  --input article.txt \
  --output article.mcz \
  --device auto \
  --dtype auto
```

### Decompress

```bash
uv run minicrunch decompress \
  --input article.mcz \
  --output roundtrip.txt \
  --device auto
```

## Notes on exact roundtrip

Arithmetic decoding must use the exact same prior as encoding.

- Same model weights/tokenizer.
- Same backend and dtype.
- Prefer the same hardware/runtime for deterministic behavior.

The archive stores model metadata and a SHA-256 checksum of decoded UTF-8 to catch mismatches.

## Optional: llama.cpp path

This demo supports both `transformers` and `llamacpp` backends.

If you want llama.cpp runtime, install optional dependencies:

```bash
uv sync --extra llamacpp
```

Use `--backend llamacpp` and pass either:

- Local GGUF path via `--model-id /absolute/path/to/model.gguf`
- Hugging Face GGUF reference via `--model-id repo_id::filename.gguf`

Example:

```bash
uv run minicrunch benchmark \
  --input article.txt \
  --backend llamacpp \
  --model-id /absolute/path/to/model.gguf \
  --device mps \
  --llama-n-gpu-layers -1 \
  --llama-n-ctx 8192
```

Or download directly from Hugging Face via `repo::filename`:

```bash
uv run minicrunch benchmark \
  --input article.txt \
  --backend llamacpp \
  --model-id <repo_id>::<filename.gguf> \
  --device mps \
  --llama-n-gpu-layers -1 \
  --llama-n-ctx 8192
```

Note (March 1, 2026): `llama-cpp-python==0.3.16` cannot load
`mistralai/Ministral-3-3B-Instruct-2512-GGUF` because that GGUF reports
architecture `mistral3`, which is not supported in that release. Use
`--backend transformers` for Ministral models unless your local llama.cpp build
explicitly supports `mistral3`.

You can also build native `llama.cpp` manually with:

```bash
./scripts/setup_llamacpp.sh
```

`--dtype` is used by `transformers`; for `llamacpp`, quantization is defined by the GGUF file.

## Optional: vLLM HTTP path (Colab-friendly)

You can host a remote vLLM server (for example on Google Colab), expose it via a public URL,
and run MiniCrunch locally with `--backend vllm`.

### 1) Start vLLM server in Colab

Clone this repo in Colab and run:

```bash
pip install "vllm>=0.7" "fastapi>=0.115" "uvicorn>=0.30"
python scripts/vllm_http_server.py \
  --model-id mistralai/Ministral-3B-Instruct-2410 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --print-config
```

Expose port `8000` publicly using your preferred tunnel (for example `cloudflared` or `ngrok`),
then copy the HTTPS base URL.

### 2) Compress locally using the Colab URL

```bash
uv run minicrunch compress \
  --input article.txt \
  --output article.mcz \
  --backend vllm \
  --vllm-url https://<your-tunnel-url> \
  --model-id mistralai/Ministral-3B-Instruct-2410 \
  --vllm-top-k 256 \
  --vllm-fallback-logit -50.0
```

### 3) Decompress (must use same backend/model URL behavior)

```bash
uv run minicrunch decompress \
  --input article.mcz \
  --output roundtrip.txt \
  --backend vllm \
  --vllm-url https://<your-tunnel-url>
```

Important: this vLLM mode uses top-K logprobs plus a fixed fallback logit for non-top-K tokens.
It remains lossless for roundtrip when encode/decode settings match, but compression quality may be
lower than exact full-logit backends.

## Run tests

```bash
uv run pytest -q
```
