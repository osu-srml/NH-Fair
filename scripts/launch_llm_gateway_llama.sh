#!/usr/bin/env bash
# Example vLLM OpenAI server for Llama 3.2 Vision (adjust model id to your cache).
# Zeroshot: --method llama --vlm_backend gateway --llm_gateway_url http://127.0.0.1:8001/v1 --model meta-llama/Llama-3.2-11B-Vision-Instruct

set -euo pipefail

PORT="${1:-8001}"
MODEL="${2:-meta-llama/Llama-3.2-11B-Vision-Instruct}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --trust-remote-code
