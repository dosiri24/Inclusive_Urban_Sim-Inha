"""Google Gemini LLM"""

import os
from .base import BaseLLM, logger, is_enabled

MODEL_NAME = "gemini-3-flash-preview"


class GeminiLLM(BaseLLM):

    def __init__(self):
        if not is_enabled("google"):
            raise ValueError("Google API is disabled")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        logger.debug("Gemini client initialized")

    def chat(self, messages: list) -> str:
        contents = self._to_gemini_format(messages)
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=self.config
        )
        return response.text

    def _to_gemini_format(self, messages: list) -> list:
        """Gemini uses role: user/model, parts: [{text}]"""
        result = []
        system_text = ""

        # extract system message
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
                break

        # convert messages
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]

            # prepend system to first user message
            if role == "user" and system_text and not result:
                content = system_text + "\n\n" + content

            result.append({"role": role, "parts": [{"text": content}]})

        return result
