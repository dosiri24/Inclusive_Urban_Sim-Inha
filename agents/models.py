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
# Persona Data Structures
# =============================================================================

@dataclass
class Demographics:
    """
    Demographic attributes of an agent.

    Attributes:
        age_group: Age range category (30s, 40s, 50s, 60s, 70s+)
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
class Personality:
    """
    Personality traits on 1-5 scale.

    Attributes:
        assertiveness: Tendency to express opinions strongly (1=passive, 5=very assertive)
        openness: Willingness to consider new ideas (1=closed, 5=very open)
        risk_tolerance: Comfort with uncertainty (1=risk-averse, 5=risk-seeking)
        community_orientation: Focus on collective vs individual (1=individual, 5=community)
    """
    assertiveness: int
    openness: int
    risk_tolerance: int
    community_orientation: int


@dataclass
class Economic:
    """
    Economic situation of an agent.

    Attributes:
        income_level: Relative income category (low, middle, high)
        can_afford_contribution: Ability to pay redevelopment costs
    """
    income_level: str
    can_afford_contribution: bool


@dataclass
class State:
    """
    Current disposition and participation tendency.

    Attributes:
        economic_pressure: Financial stress level (comfortable, moderate, struggling)
        participation_tendency: Likelihood to speak up (active, moderate, passive)
    """
    economic_pressure: str
    participation_tendency: str


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
    personality: Personality
    economic: Economic
    state: State
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

    Attributes:
        target_agent: Agent ID being reacted to (e.g., "A03")
        target_summary: Brief summary of their opinion
        my_feeling: Emotional response (worried, relieved, angry, hopeful, indifferent, empathetic)
        agree_level: Agreement level 1-5 (1=strongly disagree, 5=strongly agree)
        reason: Explanation for the feeling/agreement level
        want_to_respond: Whether agent wants to respond publicly
    """
    target_agent: str
    target_summary: str
    my_feeling: str
    agree_level: int
    reason: str
    want_to_respond: bool


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

    Attributes:
        purpose: Intent of this speech unit
                 (agree, disagree, partial_agree, cite, question, new_point)
        target: Who this is directed at (agent ID like "A03", "all", or None)
        content: The actual speech content
    """
    purpose: str
    target: Optional[str]
    content: str


@dataclass
class SpeakingOutput:
    """
    Complete speaking stage output.

    Attributes:
        units: List of speech units with purpose, target, and content
        full_statement: Complete natural language public statement
    """
    units: List[SpeakingUnit]
    full_statement: str
