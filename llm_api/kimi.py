"""Moonshot Kimi LLM (OpenAI compatible)"""

import os
from .base import BaseLLM, is_enabled

MODEL_NAME = "kimi-k2-0905-preview"
BASE_URL = "https://api.moonshot.ai/v1"


class KimiLLM(BaseLLM):

    model_name = MODEL_NAME

    def __init__(self):
        if not is_enabled("moonshot"):
            raise ValueError("Moonshot API is disabled")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["timeline"] + "\n\n" + prompt_data["task"]}
        ]

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        u = response.usage
        usage = {
            "cached": getattr(u.prompt_tokens_details, 'cached_tokens', 0) if u.prompt_tokens_details else 0,
            "prompt": u.prompt_tokens,
            "completion": u.completion_tokens
        }
        return response.choices[0].message.content, usage
