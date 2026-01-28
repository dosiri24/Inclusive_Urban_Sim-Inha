"""
Agent Cognition System for Inclusive Urban Simulation

This module implements the two-stage cognition system:
1. THINK - Generate private reactions to other agents' statements (not shared)
2. SPEAK - Generate public statements based on thinking (shared with all)

Both stages output structured JSON for metrics extraction.

Usage:
    from agents import generate_thinking, generate_speaking

    thinking = generate_thinking(persona_prompt, other_statements, topic, local_context, model)
    speaking = generate_speaking(persona_prompt, thinking, discussion_rules, model)
"""

import json
import re
import logging
from typing import List, Dict, Optional, Any, Set

from llm_api import call_llm
from .models import (
    ThinkingReaction,
    ThinkingOutput,
    SpeakingUnit,
    SpeakingOutput,
)

logger = logging.getLogger("agents.cognition")


# =============================================================================
# Constants
# =============================================================================

# Valid values for constrained fields
VALID_FEELINGS: Set[str] = {"worried", "relieved", "angry", "hopeful", "indifferent", "empathetic"}
VALID_STANCES: Set[str] = {"strong_support", "support", "neutral", "oppose", "strong_oppose"}
VALID_PURPOSES: Set[str] = {"agree", "disagree", "partial_agree", "cite", "question", "new_point"}
VALID_AGREE_LEVELS: Set[int] = {1, 2, 3, 4, 5}

# JSON schema templates for LLM prompts
THINKING_JSON_SCHEMA = """{
  "reactions": [
    {
      "target_agent": "A03",
      "target_summary": "summary of their opinion",
      "my_feeling": "one of: worried|relieved|angry|hopeful|indifferent|empathetic",
      "agree_level": 1-5,
      "reason": "why I feel this way",
      "want_to_respond": true|false
    }
  ],
  "overall_stance": "one of: strong_support|support|neutral|oppose|strong_oppose",
  "key_concerns": ["concern 1", "concern 2"],
  "strategic_notes": "speaking strategy"
}"""

SPEAKING_JSON_SCHEMA = """{
  "units": [
    {
      "purpose": "one of: agree|disagree|partial_agree|cite|question|new_point",
      "target": "agent ID (e.g. A03), 'all', or null if not directed at anyone",
      "content": "speech content"
    }
  ],
  "full_statement": "complete statement in natural language"
}"""


# =============================================================================
# Prompt Templates
# =============================================================================

THINKING_PROMPT_TEMPLATE = """{persona_prompt}

A community discussion about {topic} is in progress.

[Local context]
{local_context}

[Other residents' opinions]
{other_opinions}

Read these opinions and output your internal reactions in JSON format.
This is your private thinking, not shared publicly. Be honest.

Follow this JSON format exactly:
{json_schema}

Output only JSON. Do not include any other text."""


THINKING_PROMPT_ROUND1 = """{persona_prompt}

A community discussion about {topic} has started. This is Round 1.

[Local context]
{local_context}

No other residents have spoken yet. First, organize your initial position on this topic.
Output in JSON format.

Follow this JSON format exactly:
{{
  "reactions": [],
  "overall_stance": "one of: strong_support|support|neutral|oppose|strong_oppose",
  "key_concerns": ["concern 1", "concern 2"],
  "strategic_notes": "first speaking strategy"
}}

Output only JSON. Do not include any other text."""


SPEAKING_PROMPT_TEMPLATE = """{persona_prompt}

[Discussion rules]
{discussion_rules}

[Your internal thoughts]
{thinking_summary}

Now speak publicly to other residents.
Speak naturally according to your persona characteristics.
Output in JSON format.

Follow this JSON format exactly:
{json_schema}

Output only JSON. Do not include any other text."""


JSON_FIX_PROMPT = """The following text should be JSON but failed to parse.
Please fix it and output valid JSON only. Output nothing but JSON.

Original:
{text}"""


# =============================================================================
# Exception
# =============================================================================

class CognitionError(Exception):
    """
    Exception for cognition module errors.
    Raised when LLM output cannot be parsed or validated.
    """
    pass


# =============================================================================
# JSON Parsing
# =============================================================================

def extract_json_from_text(text: str) -> str:
    """
    Extracts JSON content from text, handling markdown code blocks.

    Args:
        text: Raw text that may contain JSON wrapped in code blocks

    Returns:
        Extracted JSON string without code block markers
    """
    # Try to find ```json ... ``` or ``` ... ``` block
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()

    # Try to find raw JSON object (starts with { and ends with })
    json_obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_obj_match:
        return json_obj_match.group(0)

    # Return original if no JSON pattern found
    return text


def parse_json(text: str, max_retries: int, model: str, memory: Optional[List[Dict]]) -> Dict[str, Any]:
    """
    Parses JSON from LLM output with simple retry logic.

    Args:
        text: Raw LLM output text
        max_retries: Number of LLM retry attempts
        model: Model to use for retries
        memory: Memory context for retries

    Returns:
        Parsed dictionary

    Raises:
        CognitionError: If parsing fails after all retries
    """
    extracted = extract_json_from_text(text)

    # First attempt
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed")

    # Retry with LLM
    for attempt in range(max_retries):
        logger.info(f"Requesting LLM to fix JSON (attempt {attempt + 1}/{max_retries})")

        fix_prompt = JSON_FIX_PROMPT.format(text=text[:3000])

        try:
            response = call_llm(model, memory or [], fix_prompt)
            fixed = extract_json_from_text(response)
            return json.loads(fixed)
        except Exception:
            logger.warning(f"JSON fix attempt {attempt + 1} failed")

    # All attempts failed
    raise CognitionError("error")


# =============================================================================
# Validation
# =============================================================================

def validate_value(value: Any, valid_set: Set, convert_type: type = str) -> Any:
    """
    Validates a value against a set of valid values.

    Args:
        value: Value to validate
        valid_set: Set of valid values
        convert_type: Type to convert value to before validation

    Returns:
        Validated value

    Raises:
        CognitionError: If value is not valid
    """
    try:
        converted = convert_type(value)
    except (ValueError, TypeError):
        raise CognitionError("error")

    if converted not in valid_set:
        raise CognitionError("error")

    return converted


# =============================================================================
# Data Conversion Functions
# =============================================================================

def dict_to_thinking_output(data: Dict[str, Any]) -> ThinkingOutput:
    """
    Converts a dictionary to ThinkingOutput dataclass.
    Validates all constrained fields.

    Args:
        data: Parsed JSON dictionary

    Returns:
        ThinkingOutput instance

    Raises:
        CognitionError: If any field value is invalid or missing
    """
    try:
        reactions = []
        for r in data.get("reactions", []):
            reactions.append(ThinkingReaction(
                target_agent=str(r["target_agent"]),
                target_summary=str(r["target_summary"]),
                my_feeling=validate_value(r["my_feeling"], VALID_FEELINGS),
                agree_level=validate_value(r["agree_level"], VALID_AGREE_LEVELS, int),
                reason=str(r["reason"]),
                want_to_respond=bool(r["want_to_respond"])
            ))

        return ThinkingOutput(
            reactions=reactions,
            overall_stance=validate_value(data["overall_stance"], VALID_STANCES),
            key_concerns=list(data.get("key_concerns", [])),
            strategic_notes=str(data.get("strategic_notes", ""))
        )
    except KeyError:
        raise CognitionError("error")


def dict_to_speaking_output(data: Dict[str, Any]) -> SpeakingOutput:
    """
    Converts a dictionary to SpeakingOutput dataclass.
    Validates all constrained fields.

    Args:
        data: Parsed JSON dictionary

    Returns:
        SpeakingOutput instance

    Raises:
        CognitionError: If any field value is invalid or missing
    """
    try:
        units = []
        for unit in data.get("units", []):
            units.append(SpeakingUnit(
                purpose=validate_value(unit["purpose"], VALID_PURPOSES),
                target=unit.get("target"),  # Can be None
                content=str(unit["content"])
            ))

        return SpeakingOutput(
            units=units,
            full_statement=str(data["full_statement"])
        )
    except KeyError:
        raise CognitionError("error")


# =============================================================================
# Format Helper Functions
# =============================================================================

def format_other_opinions(statements: List[Dict[str, str]]) -> str:
    """
    Formats other agents' statements for the thinking prompt.

    Args:
        statements: List of {"agent_id": str, "statement": str} dictionaries

    Returns:
        Formatted string of all opinions
    """
    if not statements:
        return "(No other residents have spoken yet)"

    lines = []
    for i, stmt in enumerate(statements, 1):
        agent_id = stmt.get("agent_id", f"Resident{i}")
        statement = stmt.get("statement", "")
        lines.append(f"[{agent_id}]: {statement}")

    return "\n\n".join(lines)


def format_thinking_for_speaking(thinking: ThinkingOutput) -> str:
    """
    Formats thinking output for the speaking prompt.

    Args:
        thinking: ThinkingOutput from thinking stage

    Returns:
        Formatted string summarizing the thinking
    """
    lines = []

    lines.append(f"[Overall stance]: {thinking.overall_stance}")

    if thinking.key_concerns:
        lines.append(f"[Key concerns]: {', '.join(thinking.key_concerns)}")

    if thinking.strategic_notes:
        lines.append(f"[Strategy]: {thinking.strategic_notes}")

    if thinking.reactions:
        lines.append("\n[Reactions to other residents]:")
        for reaction in thinking.reactions:
            if reaction.want_to_respond:
                lines.append(f"  - Want to respond to {reaction.target_agent} (agree: {reaction.agree_level}/5, feeling: {reaction.my_feeling})")

    return "\n".join(lines)


# =============================================================================
# Core Generation Functions
# =============================================================================

def generate_thinking(
    persona_prompt: str,
    other_statements: List[Dict[str, str]],
    topic: str,
    local_context: str,
    model: str,
    is_round_1: bool = False,
    memory: Optional[List[Dict]] = None,
    max_retries: int = 2
) -> ThinkingOutput:
    """
    Generates the thinking stage output for an agent.
    The agent processes other agents' statements and forms private reactions.

    Args:
        persona_prompt: Agent's persona description (from persona_to_prompt())
        other_statements: List of {"agent_id": str, "statement": str}
        topic: Discussion topic
        local_context: Local area context information
        model: LLM model key to use
        is_round_1: Whether this is the first round (no prior statements)
        memory: Optional conversation memory
        max_retries: Maximum LLM retry attempts for JSON parsing (default 2)

    Returns:
        ThinkingOutput with private reactions and stance

    Raises:
        CognitionError: If JSON parsing or validation fails
    """
    logger.info(f"Generating thinking: model={model}, round_1={is_round_1}")

    # Build prompt
    if is_round_1 or not other_statements:
        prompt = THINKING_PROMPT_ROUND1.format(
            persona_prompt=persona_prompt,
            topic=topic,
            local_context=local_context[:2000]
        )
    else:
        other_opinions = format_other_opinions(other_statements)
        prompt = THINKING_PROMPT_TEMPLATE.format(
            persona_prompt=persona_prompt,
            topic=topic,
            local_context=local_context[:2000],
            other_opinions=other_opinions[:3000],
            json_schema=THINKING_JSON_SCHEMA
        )

    # Call LLM
    response = call_llm(model, memory or [], prompt)

    # Parse and validate response
    parsed = parse_json(response, max_retries, model, memory)
    output = dict_to_thinking_output(parsed)

    logger.info(f"Thinking complete: stance={output.overall_stance}, reactions={len(output.reactions)}")
    return output


def generate_speaking(
    persona_prompt: str,
    thinking: ThinkingOutput,
    discussion_rules: str,
    model: str,
    memory: Optional[List[Dict]] = None,
    max_retries: int = 2
) -> SpeakingOutput:
    """
    Generates the speaking stage output for an agent.
    Based on private thinking, formulates public statement.

    Args:
        persona_prompt: Agent's persona description
        thinking: ThinkingOutput from thinking stage
        discussion_rules: Discussion format and rules
        model: LLM model key to use
        memory: Optional conversation memory
        max_retries: Maximum LLM retry attempts for JSON parsing (default 2)

    Returns:
        SpeakingOutput with public statement and structured components

    Raises:
        CognitionError: If JSON parsing or validation fails
    """
    logger.info(f"Generating speaking: model={model}")

    # Format thinking for prompt
    thinking_summary = format_thinking_for_speaking(thinking)

    # Build prompt
    prompt = SPEAKING_PROMPT_TEMPLATE.format(
        persona_prompt=persona_prompt,
        discussion_rules=discussion_rules[:2000],
        thinking_summary=thinking_summary,
        json_schema=SPEAKING_JSON_SCHEMA
    )

    # Call LLM
    response = call_llm(model, memory or [], prompt)

    # Parse and validate response
    parsed = parse_json(response, max_retries, model, memory)
    output = dict_to_speaking_output(parsed)

    logger.info(f"Speaking complete: units={len(output.units)}")
    return output
