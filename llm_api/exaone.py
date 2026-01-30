"""
LG AI EXAONE API Module

Handles all interactions with LG AI Research's EXAONE API via Friendli AI.
Uses OpenAI-compatible endpoint.
"""

import os
from typing import List, Dict

from . import logger, is_model_enabled, to_openai_format

# Global client instance (lazy initialization)
_client = None


def _get_client():
    """
    Returns the EXAONE client instance via Friendli AI (OpenAI-compatible).
    Initializes on first call (lazy loading).

    Returns:
        openai.OpenAI: Initialized OpenAI-compatible client for EXAONE

    Raises:
        ValueError: If FRENDLI_TOKEN or EXAONE_BASE_URL is not set or API is disabled
    """
    global _client
    if _client is None:
        if not is_model_enabled("exaone"):
            raise ValueError("EXAONE API is disabled. Set ENABLE_EXAONE_API=True in .env")

        # Use FRENDLI_TOKEN (as specified by user)
        api_key = os.getenv("FRENDLI_TOKEN")
        base_url = os.getenv("EXAONE_BASE_URL")

        if not api_key:
            raise ValueError("FRENDLI_TOKEN environment variable is not set")
        if not base_url:
            raise ValueError("EXAONE_BASE_URL environment variable is not set")

        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.debug("EXAONE client initialized via Friendli AI")

    return _client


def call(model_id: str, memory: List[Dict[str, str]], question: str, temperature: float = 0.7) -> str:
    """
    Calls LG AI EXAONE API via Friendli AI with the given parameters.
    Uses OpenAI-compatible endpoint.

    Args:
        model_id: The EXAONE model identifier (from MODEL_NAME env var)
        memory: Conversation history in standard format
        question: Current question/prompt
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        Model response as plain text string
    """
    client = _get_client()

    # Build messages list: history + current question
    messages = to_openai_format(memory)
    messages.append({
        "role": "user",
        "content": question
    })

    # Call API
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content
