"""OpenAI ChatGPT LLM"""

import os
from .base import BaseLLM, is_enabled

MODEL_NAME = "gpt-5-mini"


class ChatGPTLLM(BaseLLM):

    model_name = MODEL_NAME

    def __init__(self):
        if not is_enabled("openai"):
            raise ValueError("OpenAI API is disabled")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

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
            "cached": u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0,
            "prompt": u.prompt_tokens,
            "completion": u.completion_tokens
        }
        return response.choices[0].message.content, usage
