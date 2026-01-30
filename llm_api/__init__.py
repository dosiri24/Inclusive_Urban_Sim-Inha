"""LLM API Package"""

from .gemini import GeminiLLM
from .chatgpt import ChatGPTLLM
from .claude import ClaudeLLM
from .kimi import KimiLLM
from .exaone import ExaoneLLM
from .base import is_enabled

LLM_MAP = {
    "gemini": GeminiLLM,
    "chatgpt": ChatGPTLLM,
    "claude": ClaudeLLM,
    "kimi": KimiLLM,
    "exaone": ExaoneLLM,
}

PROVIDER_MAP = {
    "gemini": "google",
    "chatgpt": "openai",
    "claude": "anthropic",
    "kimi": "moonshot",
    "exaone": "exaone",
}


def get_enabled_models() -> list:
    """Return list of model names that are enabled via env vars."""
    return [name for name, provider in PROVIDER_MAP.items() if is_enabled(provider)]


def get_llm(model_name: str):
    """Get LLM instance by name."""
    if model_name not in LLM_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LLM_MAP.keys())}")
    return LLM_MAP[model_name]()
