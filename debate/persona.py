"""
Persona Generator

Generate personas for debate agents based on demographic distribution.
"""

import json
import random
from pathlib import Path
from .config import PERSONA_CONFIG, BIGFIVE_TRAITS


def _weighted_choice(options: dict) -> str:
    """Select one option based on weighted probability."""
    choices = list(options.keys())
    weights = list(options.values())
    return random.choices(choices, weights=weights, k=1)[0]


def _generate_bigfive() -> dict:
    """Generate BigFive personality scores (1-5 for each trait)."""
    return {trait: random.randint(1, 5) for trait in BIGFIVE_TRAITS}


def generate_persona() -> dict:
    """
    Generate one random persona based on PERSONA_CONFIG distribution.

    Returns:
        dict with keys: 연령대, 성별, 직업, 주거유형, 자가여부, 소득수준, 거주기간, 가구구성, BigFive
    """
    persona = {}

    for key, options in PERSONA_CONFIG.items():
        if key == "거주기간":
            # Special handling: select period, then use its average value
            period_options = {k: v["비율"] for k, v in options.items()}
            selected_period = _weighted_choice(period_options)
            persona[key] = options[selected_period]["평균"]
        else:
            persona[key] = _weighted_choice(options)

    persona["BigFive"] = _generate_bigfive()

    return persona


def _load_vulnerable_json(json_path: str) -> dict:
    """
    Load vulnerable persona fields from JSON file.

    Expected JSON format:
        {
            "직업": "무직",
            "주거유형": "기타",
            "자가여부": "월세",
            "소득수준": "저",
            "거주기간": 3,
            "가구구성": "1인"
        }
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Vulnerable persona file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_vulnerable_persona(json_path: str) -> dict:
    """
    Generate vulnerable persona by combining random fields + JSON file fields.

    Random: 연령대, 성별, BigFive
    From JSON: 직업, 주거유형, 자가여부, 소득수준, 거주기간, 가구구성

    Args:
        json_path: Path to vulnerable persona JSON file

    Returns:
        dict with same structure as generate_persona()
    """
    # Random fields
    persona = {
        "연령대": _weighted_choice(PERSONA_CONFIG["연령대"]),
        "성별": _weighted_choice(PERSONA_CONFIG["성별"]),
        "BigFive": _generate_bigfive()
    }

    # Merge fields from JSON file
    json_fields = _load_vulnerable_json(json_path)
    persona.update(json_fields)

    return persona


def generate_all_personas(
    n_total: int = 20,
    n_vulnerable: int = 4,
    vulnerable_json_paths: list = None
) -> list:
    """
    Generate all personas for debate.

    Args:
        n_total: Total number of agents (default 20)
        n_vulnerable: Number of vulnerable agents (default 4)
        vulnerable_json_paths: List of JSON file paths for vulnerable personas.
                               If None, uses default paths in prompts/vulnerable/

    Returns:
        List of persona dicts, each with:
            - agent_id: "agent_01" ~ "agent_20"
            - is_vulnerable: bool
            - (all persona fields)
    """
    if vulnerable_json_paths is None:
        # Default vulnerable persona files
        base_dir = Path(__file__).parent.parent / "prompts" / "vulnerable"
        vulnerable_json_paths = [
            str(base_dir / "housing_01.json"),
            str(base_dir / "housing_02.json"),
            str(base_dir / "participation_01.json"),
            str(base_dir / "participation_02.json"),
        ]

    if len(vulnerable_json_paths) < n_vulnerable:
        raise ValueError(f"Need {n_vulnerable} vulnerable JSON files, got {len(vulnerable_json_paths)}")

    n_normal = n_total - n_vulnerable

    # Generate normal personas
    normal_personas = []
    for _ in range(n_normal):
        p = generate_persona()
        p["is_vulnerable"] = False
        normal_personas.append(p)

    # Generate vulnerable personas
    vulnerable_personas = []
    for i in range(n_vulnerable):
        p = generate_vulnerable_persona(vulnerable_json_paths[i])
        p["is_vulnerable"] = True
        vulnerable_personas.append(p)

    # Combine and shuffle to randomize vulnerable positions
    all_personas = normal_personas + vulnerable_personas
    random.shuffle(all_personas)

    # Assign agent_id after shuffle
    for i, persona in enumerate(all_personas):
        persona["agent_id"] = f"agent_{i+1:02d}"

    return all_personas
