"""LG EXAONE LLM (OpenAI compatible via Friendli)"""

import os
from .base import BaseLLM, logger, is_enabled

MODEL_NAME = os.getenv("MODEL_NAME", "LGAI-EXAONE/K-EXAONE-236B-A23B")


class ExaoneLLM(BaseLLM):

    def __init__(self):
        if not is_enabled("exaone"):
            raise ValueError("EXAONE API is disabled")

        api_key = os.getenv("FRENDLI_TOKEN")
        base_url = os.getenv("EXAONE_BASE_URL")
        if not api_key:
            raise ValueError("FRENDLI_TOKEN is not set")
        if not base_url:
            raise ValueError("EXAONE_BASE_URL is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.debug("EXAONE client initialized")

    def chat(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        return response.choices[0].message.content
