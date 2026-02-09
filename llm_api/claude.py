"""Anthropic Claude LLM"""

import os
from .base import BaseLLM, is_enabled

MODEL_NAME = "claude-haiku-4-5-20251001"


class ClaudeLLM(BaseLLM):

    model_name = MODEL_NAME

    def __init__(self):
        if not is_enabled("anthropic"):
            raise ValueError("Anthropic API is disabled")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": prompt_data["system"],
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_data["timeline"],
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": prompt_data["task"]
                        }
                    ]
                }
            ]
        )

        u = response.usage
        cache_create = getattr(u, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(u, 'cache_read_input_tokens', 0) or 0
        usage = {
            "cached": cache_read,
            "prompt": u.input_tokens + cache_create + cache_read,
            "completion": u.output_tokens
        }
        return response.content[0].text, usage
