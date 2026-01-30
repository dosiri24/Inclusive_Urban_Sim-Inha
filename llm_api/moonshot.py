"""
Moonshot AI (Kimi) API Module

Handles all interactions with Moonshot AI's Kimi API.
Uses OpenAI-compatible endpoint.
"""

import os
from typing import List, Dict

from . import logger, is_model_enabled, to_openai_format

# Global client instance (lazy initialization)
_client = None


def _get_client():
    """
    Returns the Moonshot AI client instance (OpenAI-compatible).
    Initializes on first call (lazy loading).

    Returns:
        openai.OpenAI: Initialized OpenAI-compatible client for Moonshot

    Raises:
        ValueError: If MOONSHOT_API_KEY is not set or API is disabled
    """
    global _client
    if _client is None:
        if not is_model_enabled("moonshot"):
            raise ValueError("Moonshot API is disabled. Set ENABLE_MOONSHOT_API=True in .env")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY environment variable is not set")

        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1"
        )
        logger.debug("Moonshot client initialized")

    return _client


def call(model_id: str, memory: List[Dict[str, str]], question: str, temperature: float = 0.7) -> str:
    """
    Calls Moonshot AI (Kimi) API with the given parameters.
    Uses OpenAI-compatible endpoint.

    Args:
        model_id: The Moonshot model identifier
        memory: Conversation history in standard format
        question: Current question/prompt
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        Model response as plain text string
    """
    client = _get_client()

    # Build messages list: system + history + current question
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."}
    ]
    messages.extend(to_openai_format(memory))
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
