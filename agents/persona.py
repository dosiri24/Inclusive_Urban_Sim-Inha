"""
Persona System for Inclusive Urban Simulation

This module provides persona generation, loading, and prompt conversion.
- Generates 16 general agents with randomized attributes
- Loads 4 vulnerable agents from markdown files
- Converts personas to English prompts for LLM

Usage:
    from agents import create_agent_group, persona_to_prompt

    agents = create_agent_group(seed=42)  # Returns 20 Persona objects
    prompt = persona_to_prompt(agents[0])  # Returns English prompt string
"""

import os
import re
import random
import logging
from typing import List, Dict, Optional

from .models import (
    Demographics,
    Personality,
    Economic,
    State,
    Context,
    Classification,
    Persona,
)

logger = logging.getLogger("agents.persona")


# =============================================================================
# Exception
# =============================================================================

class PersonaError(Exception):
    """
    Exception for persona module errors.
    """
    pass


# =============================================================================
# Sampling Distributions
# =============================================================================

# Categorical distributions for general agent generation
DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "age_group": {"30s": 0.15, "40s": 0.20, "50s": 0.25, "60s": 0.25, "70s+": 0.15},
    "gender": {"male": 0.48, "female": 0.52},
    "ownership": {"owner": 0.55, "tenant": 0.45},
    "income_level": {"low": 0.30, "middle": 0.50, "high": 0.20},
    "economic_pressure": {"comfortable": 0.25, "moderate": 0.45, "struggling": 0.30},
    "participation_tendency": {"active": 0.20, "moderate": 0.50, "passive": 0.30},
    "information_access": {"high": 0.30, "medium": 0.45, "low": 0.25},
    "community_engagement": {"active": 0.25, "moderate": 0.40, "minimal": 0.35},
}

# Occupation lists by age group
OCCUPATIONS: Dict[str, List[str]] = {
    "30s": ["office worker", "self-employed", "professional", "freelancer", "public servant"],
    "40s": ["office worker", "self-employed", "professional", "public servant", "homemaker"],
    "50s": ["office worker", "self-employed", "professional", "public servant", "homemaker", "pre-retirement"],
    "60s": ["self-employed", "retired", "homemaker", "security guard", "part-time worker"],
    "70s+": ["retired", "unemployed", "part-time worker"],
}

# Section configuration for markdown parsing
SECTION_FIELDS: Dict[str, List[str]] = {
    "Demographics": ["age_group", "gender", "residence_years", "ownership", "occupation"],
    "Personality": ["assertiveness", "openness", "risk_tolerance", "community_orientation"],
    "Economic": ["income_level", "can_afford_contribution"],
    "State": ["economic_pressure", "participation_tendency"],
    "Context": ["information_access", "community_engagement"],
}


# =============================================================================
# Sampling Functions
# =============================================================================

def sample_categorical(distribution: Dict[str, float], rng: random.Random) -> str:
    """
    Samples a value from a categorical distribution.

    Args:
        distribution: Dictionary mapping values to probabilities
        rng: Random number generator instance

    Returns:
        Sampled value from distribution
    """
    values = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(values, weights=weights, k=1)[0]


def sample_residence_years(rng: random.Random) -> int:
    """
    Samples residence years from normal distribution (mean=15, std=8, clipped to 1-40).

    Args:
        rng: Random number generator instance

    Returns:
        Integer years of residence (1-40)
    """
    years = rng.gauss(mu=15, sigma=8)
    return max(1, min(40, int(round(years))))


def determine_can_afford(income_level: str, economic_pressure: str, rng: random.Random) -> bool:
    """
    Determines if agent can afford contribution based on economic factors.

    Args:
        income_level: Income category (low, middle, high)
        economic_pressure: Pressure level (comfortable, moderate, struggling)
        rng: Random number generator instance

    Returns:
        Boolean indicating affordability
    """
    base_prob = {"low": 0.2, "middle": 0.6, "high": 0.9}[income_level]
    modifier = {"comfortable": 0.1, "moderate": 0.0, "struggling": -0.2}[economic_pressure]
    return rng.random() < max(0.0, min(1.0, base_prob + modifier))


# =============================================================================
# General Agent Generation
# =============================================================================

def generate_general_agent(agent_id: str, rng: random.Random) -> Persona:
    """
    Generates a single general (non-vulnerable) agent.

    Args:
        agent_id: Unique identifier for this agent (e.g., "A01")
        rng: Random number generator instance

    Returns:
        Complete Persona object with sampled attributes
    """
    # Sample demographics
    age_group = sample_categorical(DISTRIBUTIONS["age_group"], rng)

    demographics = Demographics(
        age_group=age_group,
        gender=sample_categorical(DISTRIBUTIONS["gender"], rng),
        residence_years=sample_residence_years(rng),
        ownership=sample_categorical(DISTRIBUTIONS["ownership"], rng),
        occupation=rng.choice(OCCUPATIONS[age_group]),
    )

    # Sample personality
    personality = Personality(
        assertiveness=rng.randint(1, 5),
        openness=rng.randint(1, 5),
        risk_tolerance=rng.randint(1, 5),
        community_orientation=rng.randint(1, 5),
    )

    # Sample economic
    income_level = sample_categorical(DISTRIBUTIONS["income_level"], rng)
    economic_pressure = sample_categorical(DISTRIBUTIONS["economic_pressure"], rng)

    economic = Economic(
        income_level=income_level,
        can_afford_contribution=determine_can_afford(income_level, economic_pressure, rng),
    )

    # Sample state
    state = State(
        economic_pressure=economic_pressure,
        participation_tendency=sample_categorical(DISTRIBUTIONS["participation_tendency"], rng),
    )

    # Sample context
    context = Context(
        information_access=sample_categorical(DISTRIBUTIONS["information_access"], rng),
        community_engagement=sample_categorical(DISTRIBUTIONS["community_engagement"], rng),
    )

    # Classification (not vulnerable)
    classification = Classification(
        is_vulnerable=False,
        vulnerable_type=None,
        agent_id=agent_id,
    )

    return Persona(
        demographics=demographics,
        personality=personality,
        economic=economic,
        state=state,
        context=context,
        classification=classification,
    )


def generate_general_agents(n: int, seed: int) -> List[Persona]:
    """
    Generates multiple general agents with reproducible randomness.

    Args:
        n: Number of agents to generate
        seed: Random seed for reproducibility

    Returns:
        List of Persona objects
    """
    rng = random.Random(seed)
    agents = [generate_general_agent(f"A{i+1:02d}", rng) for i in range(n)]
    logger.info(f"Generated {n} general agents with seed={seed}")
    return agents


# =============================================================================
# Vulnerable Agent Loading
# =============================================================================

def _parse_list_section(content: str, required_fields: List[str]) -> Dict[str, str]:
    """
    Parses a markdown list section into a dictionary.

    Args:
        content: Section content with "- key: value" lines
        required_fields: List of field names that must be present

    Returns:
        Dictionary of parsed key-value pairs

    Raises:
        PersonaError: If any required field is missing
    """
    result = {}
    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            key, value = line[2:].split(":", 1)
            result[key.strip()] = value.strip()

    for field in required_fields:
        if field not in result:
            raise PersonaError("error")

    return result


def parse_vulnerable_markdown(filepath: str) -> Persona:
    """
    Parses a vulnerable agent markdown file into a Persona object.

    Args:
        filepath: Path to markdown file

    Returns:
        Persona object with parsed attributes

    Raises:
        PersonaError: If file format is invalid or required fields are missing
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        raise PersonaError("error")

    # Parse YAML frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        raise PersonaError("error")

    frontmatter_dict = {}
    for line in frontmatter_match.group(1).strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter_dict[key.strip()] = value.strip()

    if "agent_id" not in frontmatter_dict or "vulnerable_type" not in frontmatter_dict:
        raise PersonaError("error")

    # Parse sections
    sections = {}
    current_section = None
    current_content = []
    body = content[frontmatter_match.end():].strip()

    for line in body.split("\n"):
        if line.startswith("# "):
            if current_section:
                sections[current_section] = "\n".join(current_content)
            current_section = line[2:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content)

    # Parse each required section
    parsed_sections = {}
    for section_name, fields in SECTION_FIELDS.items():
        if section_name not in sections:
            raise PersonaError("error")
        parsed_sections[section_name] = _parse_list_section(sections[section_name], fields)

    demo = parsed_sections["Demographics"]
    pers = parsed_sections["Personality"]
    econ = parsed_sections["Economic"]
    stat = parsed_sections["State"]
    ctx = parsed_sections["Context"]

    demographics = Demographics(
        age_group=demo["age_group"],
        gender=demo["gender"],
        residence_years=int(demo["residence_years"]),
        ownership=demo["ownership"],
        occupation=demo["occupation"],
    )

    personality = Personality(
        assertiveness=int(pers["assertiveness"]),
        openness=int(pers["openness"]),
        risk_tolerance=int(pers["risk_tolerance"]),
        community_orientation=int(pers["community_orientation"]),
    )

    economic = Economic(
        income_level=econ["income_level"],
        can_afford_contribution=econ["can_afford_contribution"].lower() == "true",
    )

    state = State(
        economic_pressure=stat["economic_pressure"],
        participation_tendency=stat["participation_tendency"],
    )

    context = Context(
        information_access=ctx["information_access"],
        community_engagement=ctx["community_engagement"],
    )

    classification = Classification(
        is_vulnerable=True,
        vulnerable_type=frontmatter_dict["vulnerable_type"],
        agent_id=frontmatter_dict["agent_id"],
    )

    background = sections.get("Background Story", "").strip()

    logger.debug(f"Loaded vulnerable agent from {filepath}")

    return Persona(
        demographics=demographics,
        personality=personality,
        economic=economic,
        state=state,
        context=context,
        classification=classification,
        background_story=background if background else None,
    )


def load_all_vulnerable_agents(prompts_dir: str = "prompts") -> List[Persona]:
    """
    Loads all 4 vulnerable agent profiles from markdown files.

    Args:
        prompts_dir: Directory containing vulnerable agent markdown files

    Returns:
        List of 4 Persona objects

    Raises:
        PersonaError: If any file is missing or invalid
    """
    agents = []
    for i in range(1, 5):
        filepath = os.path.join(prompts_dir, f"vulnerable_agent_{i}.md")
        agents.append(parse_vulnerable_markdown(filepath))

    logger.info(f"Loaded {len(agents)} vulnerable agents from {prompts_dir}")
    return agents


# =============================================================================
# Agent Group Creation
# =============================================================================

def create_agent_group(seed: int, prompts_dir: str = "prompts") -> List[Persona]:
    """
    Creates a complete agent group of 20 agents.
    Generates 16 general agents and loads 4 vulnerable agents.
    Reassigns agent IDs to A01-A20 (vulnerable agents get A17-A20).

    Args:
        seed: Random seed for general agent generation
        prompts_dir: Directory containing vulnerable agent files

    Returns:
        List of 20 Persona objects

    Raises:
        PersonaError: If any vulnerable agent file is missing or invalid
    """
    general_agents = generate_general_agents(16, seed)
    vulnerable_agents = load_all_vulnerable_agents(prompts_dir)

    # Reassign agent IDs for vulnerable agents (A17-A20)
    for i, agent in enumerate(vulnerable_agents):
        agent.classification.agent_id = f"A{17+i:02d}"

    all_agents = general_agents + vulnerable_agents
    logger.info(f"Created agent group: {len(general_agents)} general + {len(vulnerable_agents)} vulnerable = {len(all_agents)} total")

    return all_agents


# =============================================================================
# Prompt Generation
# =============================================================================

def persona_to_prompt(persona: Persona) -> str:
    """
    Converts a Persona object to an English prompt string for LLM.

    Args:
        persona: Persona object to convert

    Returns:
        English prompt string describing the persona
    """
    d = persona.demographics
    p = persona.personality
    e = persona.economic
    s = persona.state
    c = persona.context

    prompt = f"""You are a resident with the following characteristics:

[Demographics]
- Age group: {d.age_group}
- Gender: {d.gender}
- Years of residence: {d.residence_years}
- Housing status: {d.ownership}
- Occupation: {d.occupation}

[Personality] (1-5 scale, higher = stronger)
- Assertiveness: {p.assertiveness}
- Openness: {p.openness}
- Risk tolerance: {p.risk_tolerance}
- Community orientation: {p.community_orientation}

[Economic situation]
- Income level: {e.income_level}
- Can afford contribution: {"yes" if e.can_afford_contribution else "no"}

[Current state]
- Economic pressure: {s.economic_pressure}
- Participation tendency: {s.participation_tendency}

[Context]
- Information access: {c.information_access}
- Community engagement: {c.community_engagement}"""

    if persona.background_story:
        prompt += f"""

[Background story]
{persona.background_story}"""

    prompt += """

Act consistently with these characteristics. Express this resident's perspective and concerns authentically in the discussion."""

    return prompt


# =============================================================================
# Prompt File Loading
# =============================================================================

def load_prompt_file(filename: str, prompts_dir: str = "prompts") -> str:
    """
    Loads a prompt file from the prompts directory.

    Args:
        filename: Name of the file to load
        prompts_dir: Directory containing prompt files

    Returns:
        File content as string

    Raises:
        PersonaError: If file does not exist or cannot be read
    """
    filepath = os.path.join(prompts_dir, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Loaded {filename} from {prompts_dir}")
        return content
    except Exception:
        raise PersonaError("error")


def load_local_context(prompts_dir: str = "prompts") -> str:
    """
    Loads the local context prompt from markdown file.

    Args:
        prompts_dir: Directory containing prompt files

    Returns:
        Local context prompt as string

    Raises:
        PersonaError: If file does not exist
    """
    return load_prompt_file("local_context.md", prompts_dir)


def load_discussion_rules(prompts_dir: str = "prompts") -> str:
    """
    Loads the discussion rules prompt from markdown file.

    Args:
        prompts_dir: Directory containing prompt files

    Returns:
        Discussion rules prompt as string

    Raises:
        PersonaError: If file does not exist
    """
    return load_prompt_file("discussion_rules.md", prompts_dir)


def load_all_prompts(prompts_dir: str = "prompts") -> Dict[str, str]:
    """
    Loads all prompt files from the prompts directory.

    Args:
        prompts_dir: Directory containing prompt files

    Returns:
        Dictionary with 'local_context' and 'discussion_rules' keys

    Raises:
        PersonaError: If any prompt file is missing
    """
    return {
        "local_context": load_prompt_file("local_context.md", prompts_dir),
        "discussion_rules": load_prompt_file("discussion_rules.md", prompts_dir),
    }
