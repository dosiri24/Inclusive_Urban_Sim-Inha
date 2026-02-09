"""Base LLM Class"""

import os
import logging
import threading
import time

logger = logging.getLogger("llm_api")

LLM_TIMEOUT = 300  # seconds


def is_enabled(provider: str) -> bool:
    """Check if provider is enabled via ENABLE_{PROVIDER}_API env var."""
    return os.getenv(f"ENABLE_{provider.upper()}_API", "").lower() in ("true", "1")


class BaseLLM:
    """Base class for all LLM providers. Child class must override chat()."""

    model_name: str = "unknown"

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        """
        Call LLM and return response with usage.

        Returns:
            (response_text, {"cached": int, "prompt": int, "completion": int})
        """
        raise NotImplementedError("Child class must implement chat()")

    def _call_with_timeout(self, prompt_data: dict, timeout: int) -> tuple[str, dict]:
        """Call chat() in a daemon thread, raise TimeoutError if exceeded."""
        result_holder = {}
        error_holder = {}

        def target():
            try:
                r, u = self.chat(prompt_data)
                result_holder["result"] = r
                result_holder["usage"] = u
            except Exception as e:
                error_holder["error"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(f"LLM call exceeded {timeout}s")
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result"), result_holder.get("usage", {})

    def chat_with_retry(self, prompt_data: dict, max_retries: int = 3) -> tuple[str, dict]:
        """Call chat() with timeout and retry on failure or empty response."""
        for attempt in range(max_retries):
            try:
                result, usage = self._call_with_timeout(prompt_data, LLM_TIMEOUT)
                if result is not None:
                    return result, usage
                logger.warning(f"Empty response, retry {attempt+1}/{max_retries}")
            except TimeoutError:
                logger.warning(f"Timeout ({LLM_TIMEOUT}s), retry {attempt+1}/{max_retries}")
            except Exception as e:
                logger.warning(f"API error: {e}, retry {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        return None, {}
