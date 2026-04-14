"""OpenAI-compatible chat-completions client for vision (any server: vLLM, TGI, custom)."""

from __future__ import annotations

import base64
import io
import traceback
from typing import Any

from release_benchmark.methods.lvlm.backend_config import llm_gateway_base_url


class OpenAIGatewayClient:
    """Minimal wrapper: one image + text -> assistant text."""

    def __init__(self, args: Any):
        from openai import OpenAI

        api_key = getattr(args, "llm_gateway_api_key", None) or "EMPTY"
        self.client = OpenAI(api_key=api_key, base_url=llm_gateway_base_url(args))
        self._args = args

    def generate_from_pil(
        self,
        image,
        user_text: str,
        *,
        max_tokens: int,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str | None:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        uri = f"data:image/png;base64,{img_b64}"
        mid = model or getattr(self._args, "model", None) or "gpt-4o-mini"
        try:
            chat = self.client.chat.completions.create(
                model=mid,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": uri}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                max_tokens=max_tokens,
            )
            return chat.choices[0].message.content
        except Exception as e:
            err = str(e)
            if (
                "400" in err
                or "BadRequestError" in err
                or "longer than the maximum model length" in err
            ):
                print(f"[gateway] skip sample (prompt/model length): {err[:200]}")
            else:
                print(f"[gateway] API error: {err[:200]}")
            print(traceback.format_exc())
            return None
