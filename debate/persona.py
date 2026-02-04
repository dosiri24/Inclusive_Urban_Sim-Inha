"""
Persona Generator

Generate personas for debate agents based on demographic distribution.
"""

import json
import random
from pathlib import Path
from .config import PERSONA_CONFIG, BIGFIVE_TRAITS, VULNERABLE_FIELDS, N_VULNERABLE_POOL


def _weighted_choice(options: dict) -> str:
    """Select one option based on weighted probability."""
    choices = list(options.keys())
    weights = list(options.values())
    return random.choices(choices, weights=weights, k=1)[0]


def _generate_bigfive() -> dict:
    """Generate BigFive personality scores (1-5 for each trait)."""
    return {trait: random.randint(1, 5) for trait in BIGFIVE_TRAITS}


def _generate_knowledge_level() -> str:
    """Randomly select redevelopment knowledge level based on PERSONA_CONFIG."""
    return _weighted_choice(PERSONA_CONFIG["재개발지식"])


def generate_persona() -> dict:
    """
    Generate one random persona based on PERSONA_CONFIG distribution.

    Returns:
        dict with keys: 연령대, 성별, 직업, 주거유형, 자가여부, 소득수준, 거주기간, 가구구성, 재개발지식, 매수동기, BigFive
    """
    persona = {}

    for key, options in PERSONA_CONFIG.items():
        if key == "매수동기":
            # Skip here, will be set after 자가여부 is determined
            continue
        else:
            persona[key] = _weighted_choice(options)

    # Set 매수동기 based on 자가여부
    if persona["자가여부"] == "자가":
        persona["매수동기"] = _weighted_choice(PERSONA_CONFIG["매수동기"])
    else:
        persona["매수동기"] = "N/A"

    persona["BigFive"] = _generate_bigfive()

    return persona


def _load_vulnerable_json(json_path: str) -> dict:
    """Load vulnerable persona JSON file."""
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"Vulnerable persona file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_field(field_name: str, json_value, persona: dict = None) -> str:
    """
    Resolve field value: if "random" or None, pick from config distribution.
    Special handling for "비자가" (randomly pick 전세 or 월세).

    Args:
        field_name: Field name matching PERSONA_CONFIG key
        json_value: Value from JSON (can be str, None, or "random")
        persona: Current persona dict for constraint checking

    Returns:
        Resolved value (either from JSON or randomly selected)
    """
    # Special case: "비자가" means randomly pick 전세 or 월세
    if json_value == "비자가":
        return random.choice(["전세", "월세"])

    if json_value is None or json_value == "random":
        if field_name in PERSONA_CONFIG:
            return _weighted_choice(PERSONA_CONFIG[field_name])
        return None
    return json_value


def _apply_constraints(persona: dict) -> None:
    """
    Fix logically inconsistent field combinations.

    Constraints:
        - 70대 이상 cannot be 학생
        - 20대 cannot be 은퇴
    """
    age = persona.get("연령대")
    job = persona.get("직업")

    # 70대 이상 + 학생 -> change to 은퇴
    if age == "70대 이상" and job == "학생":
        persona["직업"] = "은퇴"

    # 20대 + 은퇴 -> change to random job (excluding 은퇴)
    if age == "20대" and job == "은퇴":
        jobs_without_retire = {k: v for k, v in PERSONA_CONFIG["직업"].items() if k != "은퇴"}
        persona["직업"] = _weighted_choice(jobs_without_retire)


def generate_vulnerable_persona(json_path: str) -> dict:
    """
    Generate vulnerable persona from JSON file.

    Any field set to "random" or None will be selected from config distribution.
    "비자가" will randomly select 전세 or 월세.

    Supported JSON fields:
        - 연령대, 성별, 직업, 주거유형, 자가여부, 연소득, 거주기간, 가구구성, 재개발지식
        - 스토리: Narrative description of the persona's situation
        - 취약유형: Type of vulnerability (for documentation)
        - 취약원인: Detailed reason for vulnerability

    Args:
        json_path: Path to vulnerable persona JSON file

    Returns:
        dict with same structure as generate_persona()
    """
    json_data = _load_vulnerable_json(json_path)

    persona = {}

    # Resolve each field from VULNERABLE_FIELDS
    for field in VULNERABLE_FIELDS:
        json_value = json_data.get(field)
        persona[field] = _resolve_field(field, json_value, persona)

    # Apply logical constraints (e.g., 70대 cannot be 학생)
    _apply_constraints(persona)

    # Set 매수동기 based on 자가여부
    json_motive = json_data.get("매수동기")
    if persona["자가여부"] == "자가":
        persona["매수동기"] = _resolve_field("매수동기", json_motive)
    else:
        persona["매수동기"] = "N/A"

    # Generate BigFive
    persona["BigFive"] = _generate_bigfive()

    # Preserve metadata fields from JSON
    if "스토리" in json_data:
        persona["스토리"] = json_data["스토리"]
    if "취약유형" in json_data:
        persona["취약유형"] = json_data["취약유형"]
    if "취약원인" in json_data:
        persona["취약원인"] = json_data["취약원인"]

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
        n_vulnerable: Number of vulnerable agents (default 4, selected from pool)
        vulnerable_json_paths: List of JSON file paths for vulnerable personas.
                               If None, uses default 8 files in prompts/vulnerable/

    Returns:
        List of persona dicts, each with:
            - resident_id: "resident_01" ~ "resident_20"
            - is_vulnerable: bool
            - 취약유형: (vulnerable only) type of vulnerability
            - 스토리: (vulnerable only) narrative description
            - (all persona fields)
    """
    if vulnerable_json_paths is None:
        # Default vulnerable persona files (8 types)
        base_dir = Path(__file__).parent.parent / "prompts" / "vulnerable"
        vulnerable_json_paths = [
            str(base_dir / f"vulnerable_{i:02d}.json")
            for i in range(1, N_VULNERABLE_POOL + 1)
        ]

    # Filter to only existing files
    existing_paths = [p for p in vulnerable_json_paths if Path(p).exists()]

    if len(existing_paths) < n_vulnerable:
        raise ValueError(f"Need {n_vulnerable} vulnerable JSON files, got {len(existing_paths)}")

    n_normal = n_total - n_vulnerable

    # Generate normal personas
    normal_personas = []
    for _ in range(n_normal):
        p = generate_persona()
        p["is_vulnerable"] = False
        normal_personas.append(p)

    # Generate vulnerable personas (randomly select from available JSON files)
    selected_paths = random.sample(existing_paths, n_vulnerable)
    vulnerable_personas = []
    for json_path in selected_paths:
        p = generate_vulnerable_persona(json_path)
        p["is_vulnerable"] = True
        vulnerable_personas.append(p)

    # Combine and shuffle to randomize vulnerable positions
    all_personas = normal_personas + vulnerable_personas
    random.shuffle(all_personas)

    # Assign resident_id after shuffle
    for i, persona in enumerate(all_personas):
        persona["resident_id"] = f"resident_{i+1:02d}"

    return all_personas
