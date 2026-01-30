"""Base LLM Class"""

import os
import logging

logger = logging.getLogger("llm_api")


def is_enabled(provider: str) -> bool:
    """Check if provider is enabled via ENABLE_{PROVIDER}_API env var."""
    return os.getenv(f"ENABLE_{provider.upper()}_API", "").lower() in ("true", "1")


class BaseLLM:
    """Base class for all LLM providers. Child class must override chat()."""

    def chat(self, messages: list) -> str:
        """messages: [{"role": "user/assistant/system", "content": "..."}]"""
        raise NotImplementedError("Child class must implement chat()")
