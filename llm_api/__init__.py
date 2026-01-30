"""LLM API Package"""

from .gemini import GeminiLLM
from .chatgpt import ChatGPTLLM
from .claude import ClaudeLLM
from .kimi import KimiLLM
from .exaone import ExaoneLLM

LLM_MAP = {
    "gemini": GeminiLLM,
    "chatgpt": ChatGPTLLM,
    "claude": ClaudeLLM,
    "kimi": KimiLLM,
    "exaone": ExaoneLLM,
}


def get_llm(model_name: str):
    """Get LLM instance by name."""
    if model_name not in LLM_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LLM_MAP.keys())}")
    return LLM_MAP[model_name]()
