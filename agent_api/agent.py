"""Agent"""

from .memory import Memory
from .prompt_builder import build_prompt
from llm_api import get_llm


class Agent:
    """Simple agent: receive task, build prompt, call LLM, return response."""

    def __init__(self, agent_id: str, model_name: str, memory: Memory):
        self.agent_id = agent_id
        self.memory = memory
        self.llm = get_llm(model_name)

    def respond(self, task: str) -> str:
        """
        Process task and return LLM response.
        No parsing - returns raw response string.
        """
        self.memory.set_task(task)
        prompt_data = build_prompt(self.memory)
        response = self.llm.chat_with_retry(prompt_data, agent_id=self.agent_id)
        return response
