"""Moonshot Kimi LLM with explicit caching support."""

import os
import logging
import requests
from .base import BaseLLM, is_enabled

MODEL_NAME = "kimi-k2-0905-preview"
BASE_URL = "https://api.moonshot.ai/v1"
CACHE_TTL = 300

logger = logging.getLogger("llm_api.kimi")


class KimiLLM(BaseLLM):

    model_name = MODEL_NAME

    def __init__(self):
        if not is_enabled("moonshot"):
            raise ValueError("Moonshot API is disabled")

        self.api_key = os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY is not set")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)

        self.cache_id = None
        self.cached_content_hash = None

    def create_cache(self, system_content: str, timeline_content: str) -> bool:
        """Create explicit cache with system prompt and timeline."""
        content_to_cache = system_content + "\n\n" + timeline_content
        content_hash = hash(content_to_cache)

        if self.cached_content_hash == content_hash:
            return False

        self.delete_cache()

        try:
            res = requests.post(
                url=f"{BASE_URL}/caching",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": timeline_content},
                    ],
                    "ttl": CACHE_TTL,
                },
                timeout=30,
            )
            res.raise_for_status()
            data = res.json()
            self.cache_id = data["id"]
            self.cached_content_hash = content_hash
            logger.debug(f"Cache created: {self.cache_id}")
            return True
        except Exception as e:
            logger.warning(f"Cache creation failed: {e}")
            return False

    def delete_cache(self):
        """Reset cache state."""
        if self.cache_id:
            logger.debug(f"Cache discarded: {self.cache_id}")
        self.cache_id = None
        self.cached_content_hash = None

    def refresh_cache(self, system_content: str, timeline_content: str) -> bool:
        """Delete old cache and create new one with updated content."""
        return self.create_cache(system_content, timeline_content)

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        if self.cache_id:
            # Cached: cache ref + new_timeline + task only
            messages = [
                {"role": "cache", "content": f"cache_id={self.cache_id};reset_ttl={CACHE_TTL}"},
            ]
            if prompt_data.get("new_timeline"):
                messages.append({"role": "user", "content": prompt_data["new_timeline"]})
            messages.append({"role": "user", "content": prompt_data["task"]})
        else:
            # No cache: full content
            messages = [
                {"role": "system", "content": prompt_data["system"]},
                {"role": "user", "content": prompt_data["timeline"] + "\n\n" + prompt_data["task"]},
            ]

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        u = response.usage
        usage = {
            "cached": getattr(u.prompt_tokens_details, 'cached_tokens', 0) if u.prompt_tokens_details else 0,
            "prompt": u.prompt_tokens,
            "completion": u.completion_tokens,
        }
        return response.choices[0].message.content, usage
