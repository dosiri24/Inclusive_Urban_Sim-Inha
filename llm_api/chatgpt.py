"""OpenAI ChatGPT LLM"""

import os
from .base import BaseLLM, logger, is_enabled

MODEL_NAME = "gpt-5-mini"


class ChatGPTLLM(BaseLLM):

    def __init__(self):
        if not is_enabled("openai"):
            raise ValueError("OpenAI API is disabled")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialized")

    def chat(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        return response.choices[0].message.content
