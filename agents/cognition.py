"""
Agent Cognition System for Inclusive Urban Simulation

This module implements the two-stage cognition system:
1. THINK - Generate private reactions to other agents' statements (not shared)
2. SPEAK - Generate public statements based on thinking (shared with all)

Usage:
    from agents import generate_thinking, generate_speaking

    thinking = generate_thinking(persona_prompt, other_statements, topic, local_context, round_num, model)
    speaking = generate_speaking(persona_prompt, thinking, model)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from llm_api import call_llm
from .models import ThinkingReaction, ThinkingOutput, SpeakingUnit, SpeakingOutput

logger = logging.getLogger("agents.cognition")


# =============================================================================
# Prompt Loading
# =============================================================================

# Path to prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    """
    Loads a prompt template from the prompts directory.

    Args:
        filename: Name of the prompt file (e.g., "thinking_prompt.md")

    Returns:
        Content of the prompt file as string
    """
    filepath = PROMPTS_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# JSON Parsing
# =============================================================================

def parse_json(text: str, model: str, max_retries: int = 1) -> Dict[str, Any]:
    """
    Parses JSON from LLM output with simple retry logic.

    Args:
        text: Raw LLM output text
        model: Model to use for retries
        max_retries: Number of retry attempts (default 1)

    Returns:
        Parsed dictionary

    Raises:
        ValueError: If parsing fails after all retries
    """
    # First attempt - direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed, requesting retry")

    # Retry with simple instruction
    for attempt in range(max_retries):
        logger.info(f"JSON retry attempt {attempt + 1}/{max_retries}")
        retry_prompt = f"{text}\n\nJSON parsing failed. Output only valid JSON without any explanation."

        try:
            response = call_llm(model, [], retry_prompt)
            return json.loads(response)
        except Exception:
            logger.warning(f"Retry {attempt + 1} failed")

    raise ValueError("JSON parsing failed after all retries")


# =============================================================================
# Data Conversion
# =============================================================================

def dict_to_thinking_output(data: Dict[str, Any]) -> ThinkingOutput:
    """
    Converts a dictionary to ThinkingOutput dataclass.

    Args:
        data: Parsed JSON dictionary from LLM

    Returns:
        ThinkingOutput instance
    """
    reactions = [
        ThinkingReaction(
            target_opinion_id=r["target_opinion_id"],
            reaction=r["reaction"],
            reason=r["reason"]
        )
        for r in data.get("reactions", [])
    ]

    return ThinkingOutput(
        reactions=reactions,
        overall_stance=data["overall_stance"],
        key_concerns=data.get("key_concerns", []),
        strategic_notes=data.get("strategic_notes", "")
    )


def dict_to_speaking_output(data: Dict[str, Any]) -> SpeakingOutput:
    """
    Converts a dictionary to SpeakingOutput dataclass.

    Args:
        data: Parsed JSON dictionary from LLM

    Returns:
        SpeakingOutput instance
    """
    units = [
        SpeakingUnit(
            reaction_type=unit["reaction_type"],
            target=unit.get("target"),
            content=unit["content"]
        )
        for unit in data.get("units", [])
    ]

    return SpeakingOutput(
        units=units,
        full_statement=data["full_statement"]
    )


# =============================================================================
# Format Helper Functions
# =============================================================================

def format_other_opinions(statements: List[Dict[str, Any]]) -> str:
    """
    Formats other agents' statements for the thinking prompt.

    Args:
        statements: List of {"agent_id": str, "round": int, "units": List[Dict]}

    Returns:
        Formatted string of all opinions with IDs
    """
    if not statements:
        return "(No other residents have spoken yet)"

    lines = []
    for stmt in statements:
        agent_id = stmt.get("agent_id", "Unknown")
        round_num = stmt.get("round", 1)
        units = stmt.get("units", [])

        for i, unit in enumerate(units, 1):
            opinion_id = f"{agent_id}_R{round_num}_U{i}"
            content = unit.get("content", "")
            lines.append(f"[{opinion_id}]: {content}")

    return "\n\n".join(lines)


def format_thinking_for_speaking(thinking: ThinkingOutput) -> str:
    """
    Formats thinking output for the speaking prompt.

    Args:
        thinking: ThinkingOutput from thinking stage

    Returns:
        Formatted string summarizing the thinking
    """
    lines = [f"[Overall stance]: {thinking.overall_stance}"]

    if thinking.key_concerns:
        lines.append(f"[Key concerns]: {', '.join(thinking.key_concerns)}")

    if thinking.strategic_notes:
        lines.append(f"[Strategy]: {thinking.strategic_notes}")

    if thinking.reactions:
        lines.append("\n[Reactions to other residents]:")
        for r in thinking.reactions:
            if r.reaction != "ignore":
                lines.append(f"  - {r.target_opinion_id}: {r.reaction} - {r.reason}")

    return "\n".join(lines)


# =============================================================================
# Core Generation Functions
# =============================================================================

def generate_thinking(
    persona_prompt: str,
    other_statements: List[Dict[str, Any]],
    topic: str,
    local_context: str,
    round_num: int,
    model: str,
    memory: Optional[List[Dict]] = None,
    max_retries: int = 1
) -> ThinkingOutput:
    """
    Generates the thinking stage output for an agent.

    Args:
        persona_prompt: Agent's persona description
        other_statements: List of {"agent_id": str, "round": int, "units": List[Dict]}
        topic: Discussion topic
        local_context: Local area context information
        round_num: Current round number
        model: LLM model key to use
        memory: Optional conversation memory
        max_retries: Maximum retry attempts for JSON parsing (default 1)

    Returns:
        ThinkingOutput with private reactions and stance
    """
    logger.info(f"Generating thinking: model={model}, round={round_num}")

    # Load prompts
    template = load_prompt("thinking_prompt.md")
    discussion_rules = load_prompt("discussion_rules.md")
    other_opinions = format_other_opinions(other_statements)

    prompt = template.format(
        persona_prompt=persona_prompt,
        topic=topic,
        round_num=round_num,
        discussion_rules=discussion_rules,
        local_context=local_context,
        other_opinions=other_opinions
    )

    # Call LLM and parse response
    response = call_llm(model, memory or [], prompt)
    parsed = parse_json(response, model, max_retries)
    output = dict_to_thinking_output(parsed)

    logger.info(f"Thinking complete: stance={output.overall_stance}, reactions={len(output.reactions)}")
    return output


def generate_speaking(
    persona_prompt: str,
    thinking: ThinkingOutput,
    model: str,
    memory: Optional[List[Dict]] = None,
    max_retries: int = 1
) -> SpeakingOutput:
    """
    Generates the speaking stage output for an agent.

    Args:
        persona_prompt: Agent's persona description
        thinking: ThinkingOutput from thinking stage
        model: LLM model key to use
        memory: Optional conversation memory
        max_retries: Maximum retry attempts for JSON parsing (default 1)

    Returns:
        SpeakingOutput with public statement
    """
    logger.info(f"Generating speaking: model={model}")

    # Load prompts
    template = load_prompt("speaking_prompt.md")
    discussion_rules = load_prompt("discussion_rules.md")
    thinking_summary = format_thinking_for_speaking(thinking)

    prompt = template.format(
        persona_prompt=persona_prompt,
        discussion_rules=discussion_rules,
        thinking_summary=thinking_summary
    )

    # Call LLM and parse response
    response = call_llm(model, memory or [], prompt)
    parsed = parse_json(response, model, max_retries)
    output = dict_to_speaking_output(parsed)

    logger.info(f"Speaking complete: units={len(output.units)}")
    return output
