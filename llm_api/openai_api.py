"""
OpenAI API Module

Handles all interactions with OpenAI's API.
Note: File named openai_api.py to avoid conflict with openai package.
"""

import os
from typing import List, Dict

from . import logger, is_model_enabled, to_openai_format

# Global client instance (lazy initialization)
_client = None


def _get_client():
    """
    Returns the OpenAI client instance.
    Initializes on first call (lazy loading).

    Returns:
        openai.OpenAI: Initialized OpenAI client

    Raises:
        ValueError: If OPENAI_API_KEY is not set or API is disabled
    """
    global _client
    if _client is None:
        if not is_model_enabled("openai"):
            raise ValueError("OpenAI API is disabled. Set ENABLE_OPENAI_API=True in .env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        from openai import OpenAI
        _client = OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialized")

    return _client


def call(model_id: str, memory: List[Dict[str, str]], question: str, temperature: float = 0.7) -> str:
    """
    Calls OpenAI API with the given parameters.

    Args:
        model_id: The OpenAI model identifier
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
