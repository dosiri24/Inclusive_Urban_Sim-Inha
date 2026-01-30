"""Anthropic Claude LLM"""

import os
from .base import BaseLLM, logger, is_enabled

MODEL_NAME = "claude-haiku-4-5-20251001"


class ClaudeLLM(BaseLLM):

    def __init__(self):
        if not is_enabled("anthropic"):
            raise ValueError("Anthropic API is disabled")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        logger.debug("Anthropic client initialized")

    def chat(self, messages: list) -> str:
        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=messages
        )
        return response.content[0].text
