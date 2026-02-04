"""
Response Parser

Parse agent responses: JSON format for both debate and think.
"""

import json
import logging
import re

logger = logging.getLogger("debate.parser")


def _strip_markdown(text: str) -> str:
    """Strip markdown code blocks from LLM response."""
    if text is None:
        return ""
    text = text.strip()
    # Remove ```json or ``` wrapper
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_response(response: str) -> dict:
    """
    Parse agent debate response (JSON string).

    Expected format:
        {"발화": "...", "지목": [{"대상": "resident_05", "입장": "공감"}, ...]}

    Returns:
        dict with keys: 발화, 지목 (list of dicts)
        On parse failure, returns default values with raw response
    """
    if response is None:
        logger.warning("Response is None")
        return {"발화": "[응답 없음]", "지목": []}

    default = {
        "발화": response,
        "지목": []
    }

    try:
        cleaned = _strip_markdown(response)
        parsed = json.loads(cleaned)

        지목_raw = parsed.get("지목", [])

        # Normalize 지목 to list format
        if 지목_raw is None:
            지목 = []
        elif isinstance(지목_raw, str):
            # Legacy format: single string "resident_XX"
            입장 = parsed.get("입장", None)
            지목 = [{"대상": 지목_raw, "입장": 입장}] if 지목_raw else []
        elif isinstance(지목_raw, list):
            지목 = 지목_raw
        else:
            지목 = []

        return {
            "발화": parsed.get("발화", response),
            "지목": 지목
        }

    except json.JSONDecodeError as e:
        preview = response[:100] if len(response) > 100 else response
        logger.warning(f"JSON parse error: {e}, response: {preview}...")
        return default


def parse_think(response: str) -> dict:
    """
    Parse agent think response (JSON string).

    Expected format:
        {"상대의견": "resident_01_r1_03_r", "반응유형": "공감", "생각": "..."}

    Returns:
        dict with keys: 상대의견, 반응유형, 생각
        On parse failure, returns default values with raw response
    """
    if response is None:
        logger.warning("Response is None")
        return {"상대의견": None, "반응유형": None, "생각": "[응답 없음]"}

    default = {
        "상대의견": None,
        "반응유형": None,
        "생각": response
    }

    try:
        cleaned = _strip_markdown(response)
        parsed = json.loads(cleaned)

        return {
            "상대의견": parsed.get("상대의견", None),
            "반응유형": parsed.get("반응유형", None),
            "생각": parsed.get("생각", response)
        }

    except json.JSONDecodeError as e:
        preview = response[:100] if len(response) > 100 else response
        logger.warning(f"JSON parse error: {e}, response: {preview}...")
        return default


def parse_initial_opinion(response: str) -> dict:
    """
    Parse agent's initial opinion before debate.

    Expected format:
        {"입장": "찬성/반대", "생각": "..."}

    Returns:
        dict with keys: 입장, 생각
    """
    if response is None:
        logger.warning("Response is None")
        return {"입장": "무관심", "생각": "[응답 없음]"}

    default = {
        "입장": "무관심",
        "생각": response
    }

    try:
        cleaned = _strip_markdown(response)
        parsed = json.loads(cleaned)

        return {
            "입장": parsed.get("입장", "무관심"),
            "생각": parsed.get("생각", response)
        }

    except json.JSONDecodeError as e:
        preview = response[:100] if len(response) > 100 else response
        logger.warning(f"JSON parse error: {e}, response: {preview}...")
        return default
