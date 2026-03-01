#!/usr/bin/env bash
set -euo pipefail

# Optional helper to build llama.cpp locally.
# Usage:
#   ./scripts/setup_llamacpp.sh [target_dir]

TARGET_DIR="${1:-$PWD/.vendor/llama.cpp}"

if [ -d "$TARGET_DIR/.git" ]; then
  git -C "$TARGET_DIR" pull --ff-only
else
  git clone https://github.com/ggerganov/llama.cpp "$TARGET_DIR"
fi

cmake -S "$TARGET_DIR" -B "$TARGET_DIR/build" -DGGML_METAL=ON
cmake --build "$TARGET_DIR/build" -j

echo

echo "llama.cpp built at: $TARGET_DIR/build"
echo "Binary example: $TARGET_DIR/build/bin/llama-cli"
