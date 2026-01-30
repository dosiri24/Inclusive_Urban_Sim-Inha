"""Agent Memory"""


class Memory:
    """
    7-slot memory structure for agent.
    1~4: static (string), 5~6: dynamic (list), 7: task (updated each call)
    """

    def __init__(
        self,
        system_context: str,
        debate_rule: str,
        local_context: str,
        persona: str
    ):
        # 1~4: static slots
        self.system_context = system_context  # 1: response format guide
        self.debate_rule = debate_rule        # 2: debate background & rules
        self.local_context = local_context    # 3: local area info
        self.persona = persona                # 4: agent persona

        # 5~6: dynamic slots
        self.conversation_history = []        # 5: conversation log
        self.think = []                       # 6: thoughts + reflections

        # 7: task slot
        self.task = ""                        # 7: current task instruction

    def add_conversation(self, speaker: str, content: str):
        """Add to conversation history (slot 5)"""
        self.conversation_history.append({"speaker": speaker, "content": content})

    def add_think(self, content: str):
        """Add thought or reflection (slot 6)"""
        self.think.append(content)

    def set_task(self, task: str):
        """Set current task (slot 7)"""
        self.task = task

    def get_all(self) -> dict:
        """Return all memory slots"""
        return {
            "system_context": self.system_context,
            "debate_rule": self.debate_rule,
            "local_context": self.local_context,
            "persona": self.persona,
            "conversation_history": self.conversation_history,
            "think": self.think,
            "task": self.task,
        }
