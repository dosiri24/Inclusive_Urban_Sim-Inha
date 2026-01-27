"""
Anthropic API Module

Handles all interactions with Anthropic's Claude API.
Note: File named anthropic_api.py to avoid conflict with anthropic package.
"""

import os
from typing import List, Dict

from . import logger, is_model_enabled, to_openai_format

# Global client instance (lazy initialization)
_client = None


def _get_client():
    """
    Returns the Anthropic client instance.
    Initializes on first call (lazy loading).

    Returns:
        anthropic.Anthropic: Initialized Anthropic client

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set or API is disabled
    """
    global _client
    if _client is None:
        if not is_model_enabled("anthropic"):
            raise ValueError("Anthropic API is disabled. Set ENABLE_ANTHROPIC_API=True in .env")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        from anthropic import Anthropic
        _client = Anthropic(api_key=api_key)
        logger.debug("Anthropic client initialized")

    return _client


def call(model_id: str, memory: List[Dict[str, str]], question: str) -> str:
    """
    Calls Anthropic API with the given parameters.

    Args:
        model_id: The Anthropic model identifier
        memory: Conversation history in standard format
        question: Current question/prompt

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
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        messages=messages
    )

    # Extract text from content blocks
    return response.content[0].text
