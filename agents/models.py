"""
Data Models for Agent System

This module contains all dataclass definitions used by the agent system.
Includes both Persona-related structures and Cognition-related structures.

Usage:
    from agents.models import Persona, ThinkingOutput, SpeakingOutput
    from dataclasses import asdict

    # Serialize any model to dictionary
    persona_dict = asdict(persona)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any


# =============================================================================
# Reaction Types (Unified across Thinking and Speaking stages)
# =============================================================================

# Five-level reaction spectrum:
# - agree: Explicit support or agreement (동의)
# - cite: Reference without explicit judgment (인용)
# - question: Request clarification or more info (질문)
# - refute: Explicit disagreement (반박)
# - ignore: No reaction / new independent point (무관심/새주장)
REACTION_TYPES = {"agree", "cite", "question", "refute", "ignore"}


# =============================================================================
# Persona Data Structures
# =============================================================================

@dataclass
class Demographics:
    """
    Demographic attributes of an agent.

    Attributes:
        age_group: Age range category (20s, 30s, 40s, 50s, 60s, 70s+)
        gender: Gender identity (male, female)
        residence_years: Years living in the area (1-40)
        ownership: Housing status (owner, tenant)
        occupation: Current job or status
    """
    age_group: str
    gender: str
    residence_years: int
    ownership: str
    occupation: str


@dataclass
class Economic:
    """
    Economic situation of an agent.

    Attributes:
        income_level: Relative income category (low, middle, high)
    """
    income_level: str


@dataclass
class Context:
    """
    Participation context factors.

    Attributes:
        information_access: Access to redevelopment information (high, medium, low)
        community_engagement: Level of community involvement (active, moderate, minimal)
    """
    information_access: str
    community_engagement: str


@dataclass
class Classification:
    """
    Agent classification for tracking.

    Attributes:
        is_vulnerable: Whether agent is in a vulnerable group
        vulnerable_type: Type of vulnerability (None, housing, participation, health, age)
        agent_id: Unique identifier (A01-A20)
    """
    is_vulnerable: bool
    vulnerable_type: Optional[str]
    agent_id: str


@dataclass
class Persona:
    """
    Complete persona for a simulation agent.

    Contains all attributes needed to generate consistent agent behavior
    throughout a discussion simulation.

    For serialization, use dataclasses.asdict(persona) instead of custom methods.
    """
    demographics: Demographics
    economic: Economic
    context: Context
    classification: Classification
    background_story: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts persona to dictionary for serialization.
        Uses dataclasses.asdict() for clean, maintainable conversion.

        Returns:
            Dictionary representation of all persona attributes
        """
        return asdict(self)


# =============================================================================
# Cognition Data Structures - Thinking Stage
# =============================================================================

@dataclass
class ThinkingReaction:
    """
    Private reaction to another agent's statement.

    Uses opinion_id pointer system for efficient reference.
    Opinion ID format: "{agent_id}_R{round}_U{unit}" (e.g., "A03_R2_U1")

    Attributes:
        target_opinion_id: Opinion ID being reacted to (e.g., "A03_R2_U1")
        reaction: Reaction type (agree, cite, question, refute, ignore)
        reason: Explanation for the reaction
    """
    target_opinion_id: str
    reaction: str
    reason: str


@dataclass
class ThinkingOutput:
    """
    Complete thinking stage output.

    Attributes:
        reactions: List of reactions to each other agent's statement
        overall_stance: Position on the issue (strong_support, support, neutral, oppose, strong_oppose)
        key_concerns: Main concerns about the topic
        strategic_notes: Tactical considerations for speaking
    """
    reactions: List[ThinkingReaction]
    overall_stance: str
    key_concerns: List[str]
    strategic_notes: str


# =============================================================================
# Cognition Data Structures - Speaking Stage
# =============================================================================

@dataclass
class SpeakingUnit:
    """
    Single unit of speech in public statement.

    Uses same reaction types as ThinkingReaction for consistency.
    - agree: Explicit support or agreement (동의)
    - cite: Reference without explicit judgment (인용)
    - question: Request clarification or more info (질문)
    - refute: Explicit disagreement (반박)
    - ignore: New independent point not responding to others (새주장)

    Attributes:
        reaction_type: Type of this speech unit (agree, cite, question, refute, ignore)
        target: Who this is directed at (opinion_id like "A03_R2_U1", or None for ignore)
        content: The actual speech content
    """
    reaction_type: str
    target: Optional[str]
    content: str


@dataclass
class SpeakingOutput:
    """
    Complete speaking stage output.

    Attributes:
        units: List of speech units with reaction_type, target, and content
        full_statement: Complete natural language public statement
    """
    units: List[SpeakingUnit]
    full_statement: str
