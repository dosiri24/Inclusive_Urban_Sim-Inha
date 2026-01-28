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
    ThinkingOutput,
    SpeakingUnit,
    SpeakingOutput,
)

# Persona functions and exceptions
from .persona import (
    PersonaError,
    create_agent_group,
    persona_to_prompt,
    load_local_context,
    load_discussion_rules,
)

# Cognition functions and exceptions
from .cognition import (
    CognitionError,
    generate_thinking,
    generate_speaking,
)

__all__ = [
    # Models
    "Persona",
    "ThinkingOutput",
    "SpeakingUnit",
    "SpeakingOutput",
    # Persona
    "PersonaError",
    "create_agent_group",
    "persona_to_prompt",
    "load_local_context",
    "load_discussion_rules",
    # Cognition
    "CognitionError",
    "generate_thinking",
    "generate_speaking",
]
