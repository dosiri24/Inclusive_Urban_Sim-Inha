"""Moonshot Kimi LLM (OpenAI compatible)"""

import os
from .base import BaseLLM, logger, is_enabled

MODEL_NAME = "kimi-k2-0905-preview"
BASE_URL = "https://api.moonshot.ai/v1"


class KimiLLM(BaseLLM):

    def __init__(self):
        if not is_enabled("moonshot"):
            raise ValueError("Moonshot API is disabled")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        logger.debug("Kimi client initialized")

    def chat(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        return response.choices[0].message.content
