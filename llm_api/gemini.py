"""Google Gemini LLM with explicit caching support."""

import os
import logging
from .base import BaseLLM, is_enabled

MODEL_NAME = "gemini-3-flash-preview"
CACHE_TTL = "3600s"  # 1 hour

logger = logging.getLogger("llm_api.gemini")


class GeminiLLM(BaseLLM):

    model_name = MODEL_NAME

    def __init__(self):
        if not is_enabled("google"):
            raise ValueError("Google API is disabled")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.base_config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        self.cache_name = None
        self.cached_content_hash = None

    def create_cache(self, system_content: str, timeline_content: str) -> bool:
        """Create explicit cache with system prompt and timeline."""
        content_to_cache = system_content + "\n\n" + timeline_content
        content_hash = hash(content_to_cache)

        if self.cached_content_hash == content_hash:
            return False  # no change

        self.delete_cache()

        try:
            cache = self.client.caches.create(
                model=MODEL_NAME,
                config=self.types.CreateCachedContentConfig(
                    system_instruction=content_to_cache,
                    ttl=CACHE_TTL
                )
            )
            self.cache_name = cache.name
            self.cached_content_hash = content_hash
            logger.debug(f"Cache created: {cache.name}")
            return True
        except Exception as e:
            logger.warning(f"Cache creation failed: {e}")
            return False

    def delete_cache(self):
        """Delete existing cache if any."""
        if self.cache_name:
            try:
                self.client.caches.delete(name=self.cache_name)
                logger.debug(f"Cache deleted: {self.cache_name}")
            except Exception as e:
                logger.warning(f"Cache deletion failed: {e}")
            self.cache_name = None
            self.cached_content_hash = None

    def refresh_cache(self, system_content: str, timeline_content: str) -> bool:
        """Delete old cache and create new one with updated content."""
        return self.create_cache(system_content, timeline_content)

    def chat(self, prompt_data: dict) -> tuple[str, dict]:
        if self.cache_name:
            # Use cached content: send new_timeline + task as new content
            new_parts = []
            if prompt_data.get("new_timeline"):
                new_parts.append(prompt_data["new_timeline"])
            new_parts.append(prompt_data["task"])
            contents = "\n\n".join(new_parts)
            config = self.types.GenerateContentConfig(
                cached_content=self.cache_name,
                automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True)
            )
        else:
            # No cache: send full content
            user_content = prompt_data["system"] + "\n\n" + prompt_data["timeline"] + "\n\n" + prompt_data["task"]
            contents = [{"role": "user", "parts": [{"text": user_content}]}]
            config = self.base_config

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config
        )

        # Extract text parts only, skip thought/thought_signature parts
        text = "".join(
            part.text for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text is not None
            and not getattr(part, "thought", False)
        )

        u = response.usage_metadata
        usage = {
            "cached": getattr(u, 'cached_content_token_count', 0) or 0,
            "prompt": u.prompt_token_count,
            "completion": u.candidates_token_count
        }
        return text, usage
