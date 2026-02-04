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

    def chat(self, prompt_data: dict, agent_id: str = None) -> str:
        """
        prompt_data: {"system": str, "timeline": str, "task": str}
        """
        user_content = prompt_data["system"] + "\n\n" + prompt_data["timeline"] + "\n\n" + prompt_data["task"]

        contents = [{"role": "user", "parts": [{"text": user_content}]}]

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=self.config
        )
        u = response.usage_metadata
        cached = getattr(u, 'cached_content_token_count', 0) or 0
        logger.info(f"[{agent_id}] cached={cached}, prompt={u.prompt_token_count}, completion={u.candidates_token_count}")
        return response.text
