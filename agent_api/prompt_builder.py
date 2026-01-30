"""Prompt Builder"""

from .memory import Memory


def build_prompt(memory: Memory) -> list:
    """
    Build LLM messages from memory slots.

    Returns:
        [
            {"role": "system", "content": "slot 1~4"},
            {"role": "user", "content": "slot 5~7"}
        ]
    """
    m = memory.get_all()

    # === system message (slot 1~4) ===
    system_parts = []

    # slot 1: system_context
    if m["system_context"]:
        system_parts.append(f"[System Context]\n{m['system_context']}")

    # slot 2: debate_rule
    if m["debate_rule"]:
        system_parts.append(f"[Debate Rule]\n{m['debate_rule']}")

    # slot 3: local_context
    if m["local_context"]:
        system_parts.append(f"[Local Context]\n{m['local_context']}")

    # slot 4: persona
    if m["persona"]:
        system_parts.append(f"[Your Persona]\n{m['persona']}")

    system_content = "\n\n".join(system_parts)

    # === user message (slot 5~7) ===
    user_parts = []

    # slot 5: conversation_history
    if m["conversation_history"]:
        conv_lines = []
        for c in m["conversation_history"]:
            conv_lines.append(f"{c['speaker']}: {c['content']}")
        user_parts.append(f"[Conversation History]\n" + "\n".join(conv_lines))

    # slot 6: think
    if m["think"]:
        think_lines = "\n".join(m["think"])
        user_parts.append(f"[Your Previous Thoughts]\n{think_lines}")

    # slot 7: task
    if m["task"]:
        user_parts.append(f"[Task]\n{m['task']}")

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
