"""Anthropic Claude LLM"""

import os
import logging
import threading
import time
from .base import BaseLLM, is_enabled, LLM_TIMEOUT

MODEL_NAME = "claude-haiku-4-5-20251001"
MIN_REQUEST_INTERVAL = 1.5  # seconds between requests (all instances share)
RETRY_WAITS = [30, 60]  # seconds to wait on 1st, 2nd retry

logger = logging.getLogger("llm_api.claude")


class ClaudeLLM(BaseLLM):

    model_name = MODEL_NAME
    _lock = threading.Lock()
    _last_request_time = 0.0

    def __init__(self):
        if not is_enabled("anthropic"):
            raise ValueError("Anthropic API is disabled")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests across all instances."""
        with ClaudeLLM._lock:
            now = time.time()
            elapsed = now - ClaudeLLM._last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            ClaudeLLM._last_request_time = time.time()

    def chat_with_retry(self, prompt_data: dict, max_retries: int = 3) -> tuple[str, dict]:
        """Claude-specific retry with longer waits for rate limits."""
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
                wait = RETRY_WAITS[min(attempt, len(RETRY_WAITS) - 1)]
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)
        return None, {}

    def refresh_cache(self, system_content: str, timeline_content: str) -> bool:
        """Signal cache refresh point for prefix caching."""
        return True

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        self._wait_for_rate_limit()

        # Build user content: cached portion (stable, with breakpoint) + new portion (changing)
        user_content = []

        # Agent path provides "cached_timeline"; planner path only has "timeline"
        if "cached_timeline" in prompt_data:
            cached_tl = prompt_data["cached_timeline"]
        else:
            cached_tl = prompt_data.get("timeline", "")
        new_tl = prompt_data.get("new_timeline", "")

        if cached_tl:
            # Stable prefix: gets cache_control breakpoint for prefix caching
            user_content.append({
                "type": "text",
                "text": cached_tl,
                "cache_control": {"type": "ephemeral"}
            })

        # Changing portion: new events + task, no breakpoint
        changing_parts = [p for p in [new_tl, prompt_data["task"]] if p]
        user_content.append({
            "type": "text",
            "text": "\n\n".join(changing_parts)
        })

        response = self.client.messages.create(
            model=MODEL_NAME,
            max_tokens=prompt_data.get("max_tokens", 4096),
            system=[
                {
                    "type": "text",
                    "text": prompt_data["system"],
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )

        u = response.usage
        cache_create = getattr(u, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(u, 'cache_read_input_tokens', 0) or 0
        usage = {
            "cached": cache_read,
            "prompt": u.input_tokens + cache_create + cache_read,
            "completion": u.output_tokens
        }
        return response.content[0].text, usage
