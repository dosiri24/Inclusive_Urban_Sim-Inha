"""
Response Parser

Parse agent responses: JSON format for both debate and think.
"""

import json
import logging

logger = logging.getLogger("debate.parser")


def parse_response(response: str) -> dict:
    """
    Parse agent debate response (JSON string).

    Expected format:
        {"발화": "...", "지목": "agent_05" or null, "입장": "찬성"}

    Returns:
        dict with keys: 발화, 지목, 입장
        On parse failure, returns default values with raw response
    """
    default = {
        "발화": response,
        "지목": None,
        "입장": "무관심"
    }

    try:
        parsed = json.loads(response)

        return {
            "발화": parsed.get("발화", response),
            "지목": parsed.get("지목", None),
            "입장": parsed.get("입장", "무관심")
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}, response: {response[:100]}...")
        return default


def parse_think(response: str) -> dict:
    """
    Parse agent think response (JSON string).

    Expected format:
        {"상대의견": "agent_01_r1_03_r", "생각": "..."}

    Returns:
        dict with keys: 상대의견, 생각
        On parse failure, returns default values with raw response
    """
    default = {
        "상대의견": None,
        "생각": response
    }

    try:
        parsed = json.loads(response)

        return {
            "상대의견": parsed.get("상대의견", None),
            "생각": parsed.get("생각", response)
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}, response: {response[:100]}...")
        return default
