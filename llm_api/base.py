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

    def chat(self, messages: list) -> str:
        """messages: [{"role": "user/assistant/system", "content": "..."}]"""
        raise NotImplementedError("Child class must implement chat()")

    def chat_with_retry(self, messages: list, max_retries: int = 3) -> str:
        """Call chat() with retry on failure or empty response."""
        for attempt in range(max_retries):
            try:
                result = self.chat(messages)
                if result is not None:
                    return result
                logger.warning(f"Empty response, retry {attempt+1}/{max_retries}")
            except Exception as e:
                logger.warning(f"API error: {e}, retry {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1, 2, 4s
        return None
