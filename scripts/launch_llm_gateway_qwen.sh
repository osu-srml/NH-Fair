#!/usr/bin/env bash
# Example: serve a vision-language model with an OpenAI-compatible API (vLLM).
# Point NH-Fair zeroshot at it with:
#   --vlm_backend gateway --llm_gateway_url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct
#
# Requires: pip install vllm  (and a GPU with enough memory for the chosen model)

set -euo pipefail

PORT="${1:-8000}"
MODEL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --trust-remote-code
