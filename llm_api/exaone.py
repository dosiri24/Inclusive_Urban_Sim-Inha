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

    def chat(self, prompt_data: dict, agent_id: str = None) -> str:
        """
        prompt_data: {"system": str, "timeline": str, "task": str}
        """
        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["timeline"] + "\n\n" + prompt_data["task"]}
        ]

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        u = response.usage
        cached = getattr(u.prompt_tokens_details, 'cached_tokens', 0) if u.prompt_tokens_details else 0
        logger.info(f"[{agent_id}] cached={cached}, prompt={u.prompt_tokens}, completion={u.completion_tokens}")
        return response.choices[0].message.content
