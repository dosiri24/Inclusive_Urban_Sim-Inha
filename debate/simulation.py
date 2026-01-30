"""
Debate Simulation

Main controller for running multi-agent debate simulation.
Handles agent creation, debate flow, and logging.
"""

import random
import logging
from pathlib import Path

from agent_api import Agent, Memory
from llm_api import LLM_MAP
from .config import N_ROUNDS, N_AGENTS, N_VULNERABLE, PROMPTS_DIR, OUTPUT_DIR
from .persona import generate_all_personas
from .parser import parse_response, parse_think
from .logger import DebateLogger

logger = logging.getLogger("debate.simulation")


def _load_prompt_file(path: str) -> str:
    """Load prompt content from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return p.read_text(encoding="utf-8")


def _persona_to_summary(persona: dict) -> str:
    """Convert persona dict to short summary string for logging."""
    return (
        f"{persona['연령대']} {persona['성별']} {persona['직업']} "
        f"{persona['자가여부']} {persona['거주기간']}년"
    )


def _persona_to_prompt(persona: dict) -> str:
    """Convert persona dict to prompt string for LLM."""
    bigfive = persona["BigFive"]

    base = (
        f"연령대: {persona['연령대']}\n"
        f"성별: {persona['성별']}\n"
        f"직업: {persona['직업']}\n"
        f"주거유형: {persona['주거유형']}\n"
        f"자가여부: {persona['자가여부']}\n"
        f"소득수준: {persona['소득수준']}\n"
        f"거주기간: {persona['거주기간']}년\n"
        f"가구구성: {persona['가구구성']}\n"
        f"성격(BigFive): 개방성={bigfive['개방성']}, 성실성={bigfive['성실성']}, "
        f"외향성={bigfive['외향성']}, 친화성={bigfive['친화성']}, 신경성={bigfive['신경성']}"
    )

    # Add story for vulnerable personas
    if "스토리" in persona:
        base += f"\n\n당신의 상황:\n{persona['스토리']}"

    return base


class DebateSimulation:
    """
    Run multi-agent debate simulation.

    For Lv2~4. (Lv1 uses shared context, to be implemented separately)
    """

    def __init__(
        self,
        set_id: int,
        level: int,
        n_rounds: int = N_ROUNDS,
        n_agents: int = N_AGENTS,
        n_vulnerable: int = N_VULNERABLE,
        prompts_dir: str = PROMPTS_DIR,
        output_dir: str = OUTPUT_DIR
    ):
        """
        Initialize debate simulation.

        Args:
            set_id: Experiment set number (1~20)
            level: Experiment level (2, 3, or 4)
            n_rounds: Number of debate rounds
            n_agents: Total number of agents
            n_vulnerable: Number of vulnerable agents
            prompts_dir: Directory containing prompt files
            output_dir: Directory for output CSV files
        """
        self.set_id = set_id
        self.level = level
        self.n_rounds = n_rounds
        self.output_dir = output_dir
        self.prompts_dir = Path(prompts_dir)

        # Load static prompts
        self.system_guide = _load_prompt_file(self.prompts_dir / "system_guide.md")
        self.debate_rule = _load_prompt_file(self.prompts_dir / "debate_rule.md")
        self.local_context = _load_prompt_file(self.prompts_dir / "local_context.md")

        # Generate personas
        self.personas = generate_all_personas(n_agents, n_vulnerable)

        # Assign random models to each agent
        available_models = list(LLM_MAP.keys())
        self.agent_models = {
            p["agent_id"]: random.choice(available_models)
            for p in self.personas
        }

        # Create agents with Memory
        self.agents = {}
        for persona in self.personas:
            agent_id = persona["agent_id"]
            model_name = self.agent_models[agent_id]

            memory = Memory(
                system_context=self.system_guide,
                debate_rule=self.debate_rule,
                local_context=self.local_context,
                persona=_persona_to_prompt(persona)
            )

            self.agents[agent_id] = {
                "agent": Agent(agent_id, model_name, memory),
                "persona": persona,
                "model": model_name
            }

        # Initialize logger
        self.logger = DebateLogger(set_id, level, output_dir)

        logger.info(f"Simulation initialized: set={set_id}, level={level}, agents={n_agents}")

    def run(self):
        """
        Run the debate simulation.

        Flow:
        1. For each round:
           a. Each agent speaks in turn
           b. After each speech, other agents react (think)
           c. After round ends, all agents reflect
        2. Save logs to CSV
        """
        agent_ids = [p["agent_id"] for p in self.personas]

        for round_num in range(1, self.n_rounds + 1):
            logger.info(f"=== Round {round_num} started ===")

            for turn, speaker_id in enumerate(agent_ids, start=1):
                # === 1. Speaker's turn ===
                speaker_data = self.agents[speaker_id]
                speaker_agent = speaker_data["agent"]
                speaker_persona = speaker_data["persona"]

                # Request speech
                task = "당신의 발화 차례입니다. JSON 형식으로 응답하세요."
                response = speaker_agent.respond(task)
                parsed = parse_response(response)

                # Log debate
                response_code = self.logger.log_debate(
                    round=round_num,
                    turn=turn,
                    agent_id=speaker_id,
                    model=speaker_data["model"],
                    is_vulnerable=speaker_persona["is_vulnerable"],
                    persona_summary=_persona_to_summary(speaker_persona),
                    발화=parsed["발화"],
                    지목=parsed["지목"],
                    입장=parsed["입장"]
                )

                logger.debug(f"{speaker_id} spoke: {parsed['발화'][:50]}...")

                # === 2. Update other agents' memory ===
                for other_id in agent_ids:
                    if other_id != speaker_id:
                        other_agent = self.agents[other_id]["agent"]
                        other_agent.memory.add_conversation(speaker_id, parsed["발화"])

                # === 3. Other agents react ===
                think_turn = 1
                for other_id in agent_ids:
                    if other_id != speaker_id:
                        other_agent = self.agents[other_id]["agent"]

                        think_task = (
                            f"{speaker_id}의 발화(코드: {response_code}): "
                            f"'{parsed['발화'][:100]}...'\n"
                            f"이 발화에 대한 당신의 생각을 JSON 형식으로 응답하세요."
                        )
                        think_response = other_agent.respond(think_task)
                        think_parsed = parse_think(think_response)

                        # Add to memory
                        other_agent.memory.add_think(think_parsed["생각"])

                        # Log think
                        self.logger.log_think(
                            round=round_num,
                            turn=think_turn,
                            agent_id=other_id,
                            think_type="reaction",
                            상대의견=think_parsed["상대의견"],
                            생각=think_parsed["생각"]
                        )

                        think_turn += 1

            # === 4. End of round reflection ===
            logger.info(f"=== Round {round_num} reflection ===")

            for reflect_turn, agent_id in enumerate(agent_ids, start=1):
                agent_data = self.agents[agent_id]
                agent = agent_data["agent"]

                reflection_task = (
                    f"{round_num}라운드가 끝났습니다. "
                    f"지금까지 논의를 당신의 페르소나 관점에서 정리하세요. "
                    f"JSON 형식으로 응답하세요."
                )
                reflection_response = agent.respond(reflection_task)
                reflection_parsed = parse_think(reflection_response)

                # Add to memory
                agent.memory.add_think(reflection_parsed["생각"])

                # Log reflection
                self.logger.log_think(
                    round=round_num,
                    turn=reflect_turn,
                    agent_id=agent_id,
                    think_type="reflection",
                    상대의견=None,
                    생각=reflection_parsed["생각"]
                )

        # === 5. Save all logs ===
        self.logger.save()
        logger.info(f"Simulation completed. Logs saved.")
