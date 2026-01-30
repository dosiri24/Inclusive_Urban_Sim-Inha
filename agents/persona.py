"""
Persona System for Inclusive Urban Simulation

This module provides persona generation, loading, and prompt conversion.
- Generates 16 general agents with randomized attributes
- Loads vulnerable agents from markdown files in a folder
- Converts personas to English prompts for LLM

Usage:
    from agents import create_agent_group, persona_to_prompt

    agents = create_agent_group()  # Returns 20 Persona objects
    prompt = persona_to_prompt(agents[0])  # Returns English prompt string
"""

import re
import random
import logging
from pathlib import Path
from typing import List, Dict

from .models import (
    Demographics,
    Economic,
    Context,
    Classification,
    Persona,
)

logger = logging.getLogger("agents.persona")


# =============================================================================
# Sampling Distributions
# =============================================================================

# Categorical distributions for general agent generation
DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    # Added 20s age group for inclusivity
    "age_group": {"20s": 0.10, "30s": 0.15, "40s": 0.20, "50s": 0.25, "60s": 0.20, "70s+": 0.10},
    "gender": {"male": 0.48, "female": 0.52},
    "ownership": {"owner": 0.55, "tenant": 0.45},
    "income_level": {"low": 0.30, "middle": 0.50, "high": 0.20},
    "information_access": {"high": 0.30, "medium": 0.45, "low": 0.25},
    "community_engagement": {"active": 0.25, "moderate": 0.40, "minimal": 0.35},
}

# Occupation lists by age group
OCCUPATIONS: Dict[str, List[str]] = {
    "20s": ["university student", "graduate student", "office worker", "freelancer", "job seeker"],
    "30s": ["office worker", "self-employed", "professional", "freelancer", "public servant"],
    "40s": ["office worker", "self-employed", "professional", "public servant", "homemaker"],
    "50s": ["office worker", "self-employed", "professional", "public servant", "homemaker", "pre-retirement"],
    "60s": ["self-employed", "retired", "homemaker", "security guard", "part-time worker"],
    "70s+": ["retired", "unemployed", "part-time worker"],
}


# =============================================================================
# Sampling Functions
# =============================================================================

def _sample_categorical(distribution: Dict[str, float]) -> str:
    """
    Samples a value from a categorical distribution using random.choices.

    Args:
        distribution: Dictionary mapping values to probabilities (e.g., {"male": 0.48, "female": 0.52})

    Returns:
        Sampled value from distribution
    """
    values = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(values, weights=weights, k=1)[0]


def _sample_residence_years(age_group: str) -> int:
    """
    Samples residence years from normal distribution, adjusted by age group.
    Younger age groups have shorter residence years on average.

    Args:
        age_group: Age category (20s, 30s, 40s, 50s, 60s, 70s+)

    Returns:
        Integer years of residence (1-40)
    """
    # Adjust mean based on age group
    age_means = {
        "20s": 5,
        "30s": 10,
        "40s": 15,
        "50s": 20,
        "60s": 25,
        "70s+": 30,
    }
    mean = age_means.get(age_group, 15)
    years = random.gauss(mu=mean, sigma=5)
    return max(1, min(40, int(round(years))))


# =============================================================================
# General Agent Generation
# =============================================================================

def generate_general_agents(n: int) -> List[Persona]:
    """
    Generates n general (non-vulnerable) agents with random attributes.

    Args:
        n: Number of agents to generate

    Returns:
        List of Persona objects with IDs A01, A02, ... A{n}
    """
    agents = []

    for i in range(n):
        agent_id = f"A{i+1:02d}"

        # Sample demographics
        age_group = _sample_categorical(DISTRIBUTIONS["age_group"])

        demographics = Demographics(
            age_group=age_group,
            gender=_sample_categorical(DISTRIBUTIONS["gender"]),
            residence_years=_sample_residence_years(age_group),
            ownership=_sample_categorical(DISTRIBUTIONS["ownership"]),
            occupation=random.choice(OCCUPATIONS[age_group]),
        )

        # Sample economic
        economic = Economic(
            income_level=_sample_categorical(DISTRIBUTIONS["income_level"]),
        )

        # Sample context
        context = Context(
            information_access=_sample_categorical(DISTRIBUTIONS["information_access"]),
            community_engagement=_sample_categorical(DISTRIBUTIONS["community_engagement"]),
        )

        # Classification (not vulnerable)
        classification = Classification(
            is_vulnerable=False,
            vulnerable_type=None,
            agent_id=agent_id,
        )

        agents.append(Persona(
            demographics=demographics,
            economic=economic,
            context=context,
            classification=classification,
        ))

    logger.info(f"Generated {n} general agents")
    return agents


# =============================================================================
# Vulnerable Agent Loading
# =============================================================================

def load_vulnerable_agents(folder: str = "prompts/vulnerable") -> List[Persona]:
    """
    Loads all vulnerable agent profiles from markdown files in the specified folder.
    Parses each .md file and converts it to a Persona object.

    Args:
        folder: Directory containing vulnerable agent markdown files

    Returns:
        List of Persona objects (sorted by filename)

    Raises:
        FileNotFoundError: If folder does not exist
        ValueError: If markdown format is invalid
    """
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Vulnerable agents folder not found: {folder}")

    agents = []
    md_files = sorted(folder_path.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No markdown files found in: {folder}")

    for filepath in md_files:
        content = filepath.read_text(encoding="utf-8")

        # Parse YAML frontmatter (between --- and ---)
        frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"Missing YAML frontmatter in: {filepath.name}")

        # Extract agent_id and vulnerable_type from frontmatter
        frontmatter = {}
        for line in frontmatter_match.group(1).strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip()

        if "agent_id" not in frontmatter or "vulnerable_type" not in frontmatter:
            raise ValueError(f"Missing agent_id or vulnerable_type in: {filepath.name}")

        # Parse sections (# SectionName followed by - key: value lines)
        body = content[frontmatter_match.end():].strip()
        sections = {}
        current_section = None
        current_lines = []

        for line in body.split("\n"):
            if line.startswith("# "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines)
                current_section = line[2:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines)

        # Helper to parse "- key: value" lines into dict
        def parse_list_section(section_content: str) -> Dict[str, str]:
            result = {}
            for line in section_content.strip().split("\n"):
                line = line.strip()
                if line.startswith("- ") and ":" in line:
                    key, value = line[2:].split(":", 1)
                    result[key.strip()] = value.strip()
            return result

        # Parse each section
        demo = parse_list_section(sections.get("Demographics", ""))
        econ = parse_list_section(sections.get("Economic", ""))
        ctx = parse_list_section(sections.get("Context", ""))

        # Build Persona object
        demographics = Demographics(
            age_group=demo.get("age_group", ""),
            gender=demo.get("gender", ""),
            residence_years=int(demo.get("residence_years", 0)),
            ownership=demo.get("ownership", ""),
            occupation=demo.get("occupation", ""),
        )

        economic = Economic(
            income_level=econ.get("income_level", "middle"),
        )

        context = Context(
            information_access=ctx.get("information_access", "medium"),
            community_engagement=ctx.get("community_engagement", "moderate"),
        )

        classification = Classification(
            is_vulnerable=True,
            vulnerable_type=frontmatter["vulnerable_type"],
            agent_id=frontmatter["agent_id"],
        )

        background = sections.get("Background Story", "").strip()

        agents.append(Persona(
            demographics=demographics,
            economic=economic,
            context=context,
            classification=classification,
            background_story=background if background else None,
        ))

        logger.debug(f"Loaded vulnerable agent from {filepath.name}")

    logger.info(f"Loaded {len(agents)} vulnerable agents from {folder}")
    return agents


# =============================================================================
# Agent Group Creation
# =============================================================================

def create_agent_group(
    n_general: int = 16,
    vulnerable_folder: str = "prompts/vulnerable"
) -> List[Persona]:
    """
    Creates a complete agent group for simulation.
    Generates general agents and loads vulnerable agents from folder.
    Reassigns IDs so all agents have sequential IDs (A01, A02, ... A20).

    Args:
        n_general: Number of general agents to generate (default: 16)
        vulnerable_folder: Folder containing vulnerable agent markdown files

    Returns:
        List of Persona objects with sequential IDs

    Raises:
        FileNotFoundError: If vulnerable agents folder is missing
    """
    general_agents = generate_general_agents(n_general)
    vulnerable_agents = load_vulnerable_agents(vulnerable_folder)

    # Combine and shuffle to randomize position of vulnerable agents
    all_agents = general_agents + vulnerable_agents
    random.shuffle(all_agents)

    # Reassign sequential IDs after shuffling
    for i, agent in enumerate(all_agents):
        agent.classification.agent_id = f"A{i + 1:02d}"
    logger.info(f"Created agent group: {len(general_agents)} general + {len(vulnerable_agents)} vulnerable = {len(all_agents)} total")

    return all_agents


# =============================================================================
# Prompt Generation
# =============================================================================

def persona_to_prompt(persona: Persona) -> str:
    """
    Converts a Persona object to an English prompt string for LLM.
    This prompt instructs the LLM to role-play as the specified resident.

    Args:
        persona: Persona object to convert

    Returns:
        English prompt string describing the persona
    """
    d = persona.demographics
    e = persona.economic
    c = persona.context

    prompt = f"""You are a resident with the following characteristics:

[Demographics]
- Age group: {d.age_group}
- Gender: {d.gender}
- Years of residence: {d.residence_years}
- Housing status: {d.ownership}
- Occupation: {d.occupation}

[Economic situation]
- Income level: {e.income_level}

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
    Use this for loading local_context.md, discussion_rules.md, etc.

    Args:
        filename: Name of the file to load (e.g., "local_context.md")
        prompts_dir: Directory containing prompt files

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file does not exist
    """
    filepath = Path(prompts_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")
    logger.debug(f"Loaded {filename} from {prompts_dir}")
    return content
