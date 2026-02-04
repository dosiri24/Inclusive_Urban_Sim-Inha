"""Base LLM Class"""

import os
import logging
import time

logger = logging.getLogger("llm_api")


def is_enabled(provider: str) -> bool:
    """Check if provider is enabled via ENABLE_{PROVIDER}_API env var."""
    return os.getenv(f"ENABLE_{provider.upper()}_API", "").lower() in ("true", "1")


class BaseLLM:
    """Base class for all LLM providers. Child class must override chat()."""

    def chat(self, prompt_data: dict, agent_id: str = None) -> str:
        """prompt_data: {"system": str, "timeline": str, "task": str}"""
        raise NotImplementedError("Child class must implement chat()")

    def chat_with_retry(self, prompt_data: dict, agent_id: str = None, max_retries: int = 3) -> str:
        """Call chat() with retry on failure or empty response."""
        for attempt in range(max_retries):
            try:
                result = self.chat(prompt_data, agent_id)
                if result is not None:
                    return result
                logger.warning(f"[{agent_id}] Empty response, retry {attempt+1}/{max_retries}")
            except Exception as e:
                logger.warning(f"[{agent_id}] API error: {e}, retry {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        return None
