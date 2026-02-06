"""
Debate Package

Multi-agent debate simulation for inclusive urban participation research.
"""

from .config import (
    PERSONA_CONFIG,
    BIGFIVE_TRAITS,
    N_ROUNDS,
    N_AGENTS,
    N_VULNERABLE,
    PROMPTS_DIR,
    OUTPUT_DIR
)
from .persona import generate_persona, generate_all_personas
from .parser import (
    parse_response,
    parse_think,
    parse_batch_narrative,
    parse_batch_opinion,
    parse_batch_speech,
    parse_planner_result,
    parse_vote,
    parse_batch_vote
)
from .planner import run_planner, compile_debate_text, compile_final_opinions
from .simulation import DebateSimulation
from .simulation_lv1 import DebateSimulationLv1
