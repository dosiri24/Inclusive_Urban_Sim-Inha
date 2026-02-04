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
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_data["timeline"],
                            "cache_control": {"type": "ephemeral", "ttl": "5m"}
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
        usage = {
            "cached": u.cache_read_input_tokens,
            "prompt": u.input_tokens,
            "completion": u.output_tokens
        }
        return response.content[0].text, usage
