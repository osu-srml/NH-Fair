"""Resolve Hugging Face transformers vs OpenAI-compatible HTTP gateway (vLLM, TGI, etc.)."""

from __future__ import annotations


def resolve_vlm_backend(args) -> str:
    """
    Return 'transformers' (local HF) or 'gateway' (OpenAI-compatible HTTP API).

    The backend is decided **only** by the CLI arg ``--vlm_backend``.
    """
    explicit = getattr(args, "vlm_backend", None)
    if explicit is None:
        return "transformers"
    if explicit not in ("transformers", "gateway"):
        raise ValueError(f"Invalid vlm_backend: {explicit}")
    return explicit


def llm_gateway_base_url(args) -> str:
    """OpenAI client base_url, e.g. http://127.0.0.1:8000/v1"""
    u = getattr(args, "llm_gateway_url", None)
    if u:
        return u.rstrip("/")
    port = getattr(args, "vllm_port", 8000)
    return f"http://127.0.0.1:{int(port)}/v1"
