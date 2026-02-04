"""Agent with cache support for Gemini."""

from .memory import Memory
from .prompt_builder import build_prompt
from llm_api import get_llm


class Agent:
    """Simple agent: receive task, build prompt, call LLM, return response."""

    def __init__(self, agent_id: str, model_name: str, memory: Memory):
        self.agent_id = agent_id
        self.model_name = model_name
        self.memory = memory
        self.llm = get_llm(model_name)

    def respond(self, task: str) -> tuple[str, dict]:
        """
        Process task and return LLM response with usage.

        Returns:
            (response_text, usage_dict)
        """
        self.memory.set_task(task)
        prompt_data = build_prompt(self.memory)
        response, usage = self.llm.chat_with_retry(prompt_data)
        return response, usage

    def refresh_cache(self) -> bool:
        """
        Refresh LLM cache with current memory content.
        Only works for Gemini model.

        Returns:
            True if cache was refreshed, False otherwise.
        """
        if not hasattr(self.llm, 'refresh_cache'):
            return False

        prompt_data = build_prompt(self.memory)
        return self.llm.refresh_cache(prompt_data["system"], prompt_data["timeline"])
