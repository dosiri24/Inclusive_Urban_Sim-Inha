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

    def chat(self, prompt_data: dict, agent_id: str = None) -> str:
        """
        prompt_data: {"system": str, "timeline": str, "task": str}
        """
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
        logger.info(f"[{agent_id}] created={u.cache_creation_input_tokens}, read={u.cache_read_input_tokens}, input={u.input_tokens}")
        return response.content[0].text
