"""Urban planner agent: synthesize debate and generate compromise."""

import logging

from llm_api import get_llm
from prompts.tasks import get_planner_task
from .parser import parse_planner_result

logger = logging.getLogger("debate.planner")


def compile_debate_text(debate_buffer: list[dict]) -> str:
    """Convert DebateLogger.debate_buffer into round-grouped text."""
    rounds = {}
    for entry in debate_buffer:
        r = entry["round"]
        if r not in rounds:
            rounds[r] = []
        rounds[r].append(f"{entry['resident_id']}: {entry['발화']}")

    lines = []
    for r in sorted(rounds.keys()):
        lines.append(f"=== {r}라운드 ===")
        lines.extend(rounds[r])
    return "\n".join(lines)


def compile_final_opinions(final_opinions: dict) -> str:
    """Convert resident_id -> speech text dict to text."""
    lines = []
    for rid, speech in final_opinions.items():
        lines.append(f"{rid}: {speech}")
    return "\n".join(lines)


def run_planner(
    model_name: str,
    system_prompt: str,
    debate_text: str,
    opinions_text: str,
    debate_rule: str = "",
    local_context: str = "",
) -> tuple[dict, dict]:
    """
    Run urban planner LLM to generate compromise.

    Returns:
        (parsed_result, usage_dict)
    """
    llm = get_llm(model_name)

    task = get_planner_task()

    # Build context block with debate rule and local context
    context_block = ""
    if debate_rule:
        context_block += f"[토의 의제 및 촉진계획 세부안]\n{debate_rule}\n\n"
    if local_context:
        context_block += f"[지역 맥락]\n{local_context}\n\n"

    prompt_data = {
        "system": system_prompt,
        "timeline": f"{context_block}[토의 전문]\n{debate_text}\n\n[주민 최종 의견]\n{opinions_text}",
        "new_timeline": "",
        "task": f"[Task]\n{task}"
    }

    response, usage = llm.chat_with_retry(prompt_data)
    result = parse_planner_result(response)

    logger.info(f"Planner generated {len(result['논쟁요소'])} issues, model={model_name}")
    return result, usage
