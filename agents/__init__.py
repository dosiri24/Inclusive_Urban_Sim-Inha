"""
Agents Package for Inclusive Urban Simulation

Provides agent generation, persona management, and cognition systems.

Usage:
    from agents import create_agent_group, persona_to_prompt
    from agents import generate_thinking, generate_speaking
"""

# Data models
from .models import (
    Persona,
    ThinkingReaction,
    ThinkingOutput,
    SpeakingUnit,
    SpeakingOutput,
    REACTION_TYPES,
)

# Persona functions
from .persona import (
    create_agent_group,
    persona_to_prompt,
    load_prompt_file,
)

# Cognition functions
from .cognition import (
    generate_thinking,
    generate_speaking,
)

__all__ = [
    # Models
    "Persona",
    "ThinkingReaction",
    "ThinkingOutput",
    "SpeakingUnit",
    "SpeakingOutput",
    "REACTION_TYPES",
    # Persona
    "create_agent_group",
    "persona_to_prompt",
    "load_prompt_file",
    # Cognition
    "generate_thinking",
    "generate_speaking",
]
