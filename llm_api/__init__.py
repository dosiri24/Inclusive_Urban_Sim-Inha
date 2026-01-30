"""
LLM API Package

Provides a unified interface for multiple LLM providers.

Usage:
    from llm_api import call_llm, get_enabled_models

    response = call_llm(
        model="gemini-3-flash",
        memory=[{"question": "Hi", "answer": "Hello!"}],
        question="How are you?"
    )

Supported Models:
    - gemini-3-flash: Google Gemini 3 Flash Preview
    - gpt-5-mini: OpenAI GPT-5 Mini
    - claude-haiku-4.5: Anthropic Claude Haiku 4.5
    - kimi-k2: Moonshot AI Kimi K2
    - exaone-4.0: LG AI Research EXAONE 4.0
"""

import os
import time
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("llm_api")


# =============================================================================
# Model Configuration
# =============================================================================

def is_model_enabled(provider: str) -> bool:
    """
    Checks if a model provider is enabled via environment variable.

    Args:
        provider: Provider name (google, openai, anthropic, moonshot, exaone)

    Returns:
        True if enabled, False otherwise
    """
    env_map = {
        "google": "ENABLE_GOOGLE_API",
        "openai": "ENABLE_OPENAI_API",
        "anthropic": "ENABLE_ANTHROPIC_API",
        "moonshot": "ENABLE_MOONSHOT_API",
        "exaone": "ENABLE_EXAONE_API",
    }
    env_var = env_map.get(provider, "")
    value = os.getenv(env_var, "False")
    return value.lower() in ("true", "1", "yes")


# Model key to provider and model ID mapping
MODEL_CONFIG = {
    "gemini-3-flash": {
        "provider": "google",
        "model_id": "gemini-3-flash-preview",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model_id": "gpt-5-mini",
    },
    "claude-haiku-4.5": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
    },
    "kimi-k2": {
        "provider": "moonshot",
        "model_id": "kimi-k2-0905-preview",
    },
    "exaone-4.0": {
        "provider": "exaone",
        "model_id": os.getenv("MODEL_NAME", "LGAI-EXAONE/K-EXAONE-236B-A23B").strip('"'),
    },
}

# Default temperature from environment variable
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))


# =============================================================================
# Memory Format Converters
# =============================================================================

def to_google_format(memory: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Converts memory format to Google Gemini format.

    Args:
        memory: List of {"question": str, "answer": str} dictionaries

    Returns:
        List of Gemini-compatible message dictionaries with role and parts
    """
    messages = []
    for item in memory:
        messages.append({
            "role": "user",
            "parts": [{"text": item["question"]}]
        })
        messages.append({
            "role": "model",
            "parts": [{"text": item["answer"]}]
        })
    return messages


def to_openai_format(memory: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Converts memory format to OpenAI/compatible format.
    Also used for Anthropic, Moonshot, and EXAONE.

    Args:
        memory: List of {"question": str, "answer": str} dictionaries

    Returns:
        List of OpenAI-compatible message dictionaries with role and content
    """
    messages = []
    for item in memory:
        messages.append({
            "role": "user",
            "content": item["question"]
        })
        messages.append({
            "role": "assistant",
            "content": item["answer"]
        })
    return messages


# =============================================================================
# Utility Functions
# =============================================================================

def get_enabled_models() -> List[str]:
    """
    Returns list of currently enabled model keys.

    Returns:
        List of model keys that are enabled in .env
    """
    enabled = []
    for model_key, config in MODEL_CONFIG.items():
        if is_model_enabled(config["provider"]):
            enabled.append(model_key)
    return enabled


def get_disabled_models() -> List[str]:
    """
    Returns list of currently disabled model keys.

    Returns:
        List of model keys that are disabled in .env
    """
    disabled = []
    for model_key, config in MODEL_CONFIG.items():
        if not is_model_enabled(config["provider"]):
            disabled.append(model_key)
    return disabled


def get_all_models() -> List[str]:
    """
    Returns list of all available model keys.

    Returns:
        List of all model keys in MODEL_CONFIG
    """
    return list(MODEL_CONFIG.keys())


# =============================================================================
# Provider Imports (Lazy)
# =============================================================================

from . import google
from . import openai_api
from . import anthropic_api
from . import moonshot
from . import exaone

# Provider call function mapping
_PROVIDER_CALL_MAP = {
    "google": google.call,
    "openai": openai_api.call,
    "anthropic": anthropic_api.call,
    "moonshot": moonshot.call,
    "exaone": exaone.call,
}


# =============================================================================
# Main Unified Interface
# =============================================================================

def call_llm(
    model: str,
    memory: List[Dict[str, str]],
    question: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Unified LLM interface for all supported providers.

    This function abstracts away provider-specific differences and provides
    a consistent interface for calling any supported LLM model.

    Temperature is always read from LLM_TEMPERATURE in .env file.

    Args:
        model: Model key (e.g., "gemini-3-flash", "gpt-5-mini")
        memory: Conversation history as list of {"question": str, "answer": str}
        question: Current question/prompt to send to the model
        max_retries: Maximum number of retry attempts for transient errors
        retry_delay: Initial delay between retries (exponential backoff applied)

    Returns:
        Model response as plain text string

    Raises:
        ValueError: If model key is not supported or model is disabled
        Exception: If all retry attempts fail

    Example:
        >>> response = call_llm(
        ...     model="gemini-3-flash",
        ...     memory=[{"question": "Hi", "answer": "Hello!"}],
        ...     question="How are you?"
        ... )
        >>> print(response)
    """
    # Temperature is always from environment variable
    temperature = DEFAULT_TEMPERATURE
    # Validate model key
    if model not in MODEL_CONFIG:
        supported = ", ".join(MODEL_CONFIG.keys())
        raise ValueError(f"Unsupported model: {model}. Supported models: {supported}")

    config = MODEL_CONFIG[model]
    provider = config["provider"]
    model_id = config["model_id"]

    # Check if model is enabled
    if not is_model_enabled(provider):
        raise ValueError(f"Model {model} is disabled. Enable it in .env file.")

    # Get provider-specific call function
    call_fn = _PROVIDER_CALL_MAP[provider]

    # Log request
    logger.info(f"Calling {model} (provider={provider}, model_id={model_id})")
    logger.debug(f"Memory length: {len(memory)}, Question length: {len(question)}")

    # Retry loop with exponential backoff
    last_exception = None
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = call_fn(model_id, memory, question, temperature)
            latency = time.time() - start_time

            # Log success
            logger.info(f"Response received in {latency:.2f}s, length={len(response)}")
            logger.debug(f"Response preview: {response[:100]}...")

            return response

        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")

            # Don't retry on authentication or validation errors
            error_str = str(e).lower()
            if any(x in error_str for x in ["auth", "invalid", "api key", "not found", "disabled"]):
                logger.error(f"Non-retryable error: {e}")
                raise

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

    # All retries exhausted
    logger.error(f"All {max_retries} attempts failed for {model}")
    raise last_exception


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "call_llm",
    "get_enabled_models",
    "get_disabled_models",
    "get_all_models",
    "MODEL_CONFIG",
    "DEFAULT_TEMPERATURE",
    "logger",
    "is_model_enabled",
    "to_google_format",
    "to_openai_format",
]
