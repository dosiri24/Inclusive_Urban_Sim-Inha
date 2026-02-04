"""Agent Memory"""


class Memory:
    """
    7-slot memory structure for agent.
    1~4: static (string), 5: dynamic timeline (list), 6: task (updated each call)
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

        # 5: unified timeline (utterance, my_utterance, my_think in chronological order)
        self.timeline = []

        # 6: task slot
        self.task = ""

    def add_utterance(self, speaker: str, content: str):
        """Add other agent's utterance to timeline"""
        self.timeline.append({"type": "utterance", "speaker": speaker, "content": content})

    def add_my_utterance(self, content: str):
        """Add my own utterance to timeline"""
        self.timeline.append({"type": "my_utterance", "content": content})

    def add_my_think(self, content: str):
        """Add my thought to timeline"""
        self.timeline.append({"type": "my_think", "content": content})

    def set_task(self, task: str):
        """Set current task (slot 6)"""
        self.task = task

    def get_all(self) -> dict:
        """Return all memory slots"""
        return {
            "system_context": self.system_context,
            "debate_rule": self.debate_rule,
            "local_context": self.local_context,
            "persona": self.persona,
            "timeline": self.timeline,
            "task": self.task,
        }
