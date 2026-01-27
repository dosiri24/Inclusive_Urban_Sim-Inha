"""
Google Gemini API Module

Handles all interactions with Google's Generative AI (Gemini) API.
"""

import os
from typing import List, Dict

from . import logger, is_model_enabled, to_google_format

# Global client instance (lazy initialization)
_client = None


def _get_client():
    """
    Returns the Google Generative AI client instance.
    Initializes on first call (lazy loading).

    Returns:
        google.genai.Client: Initialized Google GenAI client

    Raises:
        ValueError: If GOOGLE_API_KEY is not set or API is disabled
    """
    global _client
    if _client is None:
        if not is_model_enabled("google"):
            raise ValueError("Google API is disabled. Set ENABLE_GOOGLE_API=True in .env")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        from google import genai
        _client = genai.Client(api_key=api_key)
        logger.debug("Google GenAI client initialized")

    return _client


def call(model_id: str, memory: List[Dict[str, str]], question: str) -> str:
    """
    Calls Google Gemini API with the given parameters.

    Args:
        model_id: The Gemini model identifier
        memory: Conversation history in standard format
        question: Current question/prompt

    Returns:
        Model response as plain text string
    """
    client = _get_client()

    # Build contents list: history + current question
    contents = to_google_format(memory)
    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })

    # Call API
    response = client.models.generate_content(
        model=model_id,
        contents=contents
    )

    return response.text
