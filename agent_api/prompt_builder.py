"""Prompt Builder"""

from .memory import Memory


def _format_timeline_items(items: list) -> list[str]:
    """Convert timeline items to formatted strings."""
    lines = []
    for item in items:
        if item["type"] == "utterance":
            lines.append(f"[{item['speaker']}]: {item['content']}")
        elif item["type"] == "my_utterance":
            lines.append(f"[Me]: {item['content']}")
        elif item["type"] == "my_think":
            lines.append(f"[My Thought]: {item['content']}")
    return lines


def build_prompt(memory: Memory) -> dict:
    """
    Build structured prompt data from memory slots.

    Returns:
        {
            "system": str,        # slot 1~4 combined
            "timeline": str,      # slot 5 (full timeline)
            "new_timeline": str,  # items added after last cache
            "task": str           # slot 6
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

    # === full timeline (slot 5) ===
    timeline_lines = _format_timeline_items(m["timeline"])
    if timeline_lines:
        timeline_content = "[Timeline]\n" + "\n".join(timeline_lines)
    else:
        timeline_content = "[Timeline]\n(empty)"

    # === new timeline (items after last cache) ===
    new_lines = _format_timeline_items(m["new_timeline"])
    if new_lines:
        new_timeline_content = "[New Events]\n" + "\n".join(new_lines)
    else:
        new_timeline_content = ""

    # === task (slot 6) ===
    if m["task"]:
        task_content = f"[Task]\n{m['task']}"
    else:
        task_content = "[Task]\n(empty)"

    return {
        "system": system_content,
        "timeline": timeline_content,
        "new_timeline": new_timeline_content,
        "task": task_content
    }
