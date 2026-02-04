"""Prompt Builder"""

from .memory import Memory


def build_prompt(memory: Memory) -> dict:
    """
    Build structured prompt data from memory slots.

    Returns:
        {
            "system": str,    # slot 1~4 combined
            "timeline": str,  # slot 5
            "task": str       # slot 6
        }
    """
    m = memory.get_all()

    # === system (slot 1~4) ===
    system_parts = []

    if m["system_context"]:
        system_parts.append(f"[System Context]\n{m['system_context']}")

    if m["debate_rule"]:
        system_parts.append(f"[Debate Rule]\n{m['debate_rule']}")

    if m["local_context"]:
        system_parts.append(f"[Local Context]\n{m['local_context']}")

    if m["persona"]:
        system_parts.append(f"[Your Persona]\n{m['persona']}")

    system_content = "\n\n".join(system_parts)

    # === timeline (slot 5) ===
    timeline_lines = []
    for item in m["timeline"]:
        if item["type"] == "utterance":
            timeline_lines.append(f"[{item['speaker']}]: {item['content']}")
        elif item["type"] == "my_utterance":
            timeline_lines.append(f"[Me]: {item['content']}")
        elif item["type"] == "my_think":
            timeline_lines.append(f"[My Thought]: {item['content']}")

    if timeline_lines:
        timeline_content = "[Timeline]\n" + "\n".join(timeline_lines)
    else:
        timeline_content = "[Timeline]\n(empty)"

    # === task (slot 6) ===
    if m["task"]:
        task_content = f"[Task]\n{m['task']}"
    else:
        task_content = "[Task]\n(empty)"

    return {
        "system": system_content,
        "timeline": timeline_content,
        "task": task_content
    }
