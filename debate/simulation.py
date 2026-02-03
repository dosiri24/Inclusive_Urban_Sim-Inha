"""
Debate Simulation

Main controller for running multi-agent debate simulation.
Handles agent creation, debate flow, and logging.
"""

import csv
import random
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_api import Agent, Memory
from llm_api import LLM_MAP, get_enabled_models
from .config import N_ROUNDS, N_AGENTS, N_VULNERABLE, PROMPTS_DIR, OUTPUT_DIR
from .persona import generate_all_personas
from .parser import parse_response, parse_think, parse_initial_opinion
from .logger import DebateLogger

logger = logging.getLogger("debate.simulation")


def setup_file_logger(output_dir: str, set_id: int, level: int):
    """Setup file handler for all logs."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"set{set_id}_lv{level}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return log_path


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
        f"매수동기: {persona['매수동기']}\n"
        f"소득수준: {persona['소득수준']}\n"
        f"거주기간: {persona['거주기간']}년\n"
        f"가구구성: {persona['가구구성']}\n"
        f"재개발지식: {persona['재개발지식']}\n"
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
        output_dir: str = OUTPUT_DIR,
        model: str = None
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
            model: If specified, all agents use this model (for testing)
        """
        self.fixed_model = model
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

        # Assign models to each agent (only enabled ones)
        available_models = get_enabled_models()
        if self.fixed_model:
            # Use specified model for all agents (testing)
            self.agent_models = {
                p["resident_id"]: self.fixed_model
                for p in self.personas
            }
        else:
            # Random assignment
            self.agent_models = {
                p["resident_id"]: random.choice(available_models)
                for p in self.personas
            }

        # Create agents with Memory
        self.agents = {}
        for persona in self.personas:
            resident_id = persona["resident_id"]
            model_name = self.agent_models[resident_id]

            memory = Memory(
                system_context=self.system_guide,
                debate_rule=self.debate_rule,
                local_context=self.local_context,
                persona=_persona_to_prompt(persona)
            )

            self.agents[resident_id] = {
                "agent": Agent(resident_id, model_name, memory),
                "persona": persona,
                "model": model_name
            }

        # Initialize logger
        self.logger = DebateLogger(set_id, level, output_dir)

        # Setup file logger for all logs
        self.log_path = setup_file_logger(output_dir, set_id, level)

        logger.info(f"Simulation initialized: set={set_id}, level={level}, agents={n_agents}")
        logger.info(f"Log file: {self.log_path}")

        # Save agent list (personas only)
        self._save_agent_list()

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
        resident_ids = [p["resident_id"] for p in self.personas]

        # === 0-1. Pre-debate: Generate narrative background (parallel) ===
        logger.info("=== Generating narrative backgrounds ===")

        narrative_task = (
            "당신은 주안2동에 사는 주민입니다. 주어진 페르소나를 바탕으로 다음 질문에 자연스럽게 답하세요.\n"
            "1. 이 동네에 어떻게 오게 되었나요?\n"
            "2. 재개발에 대해 어디서, 누구에게, 어떻게 들었나요?\n"
            "3. '비례율', '분담금', '조합', '감정평가' 같은 재개발 용어를 들어본 적 있나요? 어느 정도 이해하고 있나요?\n"
            "4. 재개발이 되면 당신의 생활은 어떻게 바뀔 것 같나요?\n"
            "5. 구체적으로 어떤 일을 하고, 대략적인 하루 일과가 어떻게 되나요?\n"
            "6. 이 동네에서 자주 만나는 이웃이 있나요? 혹은 동네 모임이나 반상회에 참여하나요?\n"
            "7. 여러 사람이 모여서 의견이 갈릴 때 당신은 보통 어떻게 행동하나요?\n"
            "8. 재개발 외에 요즘 가장 신경 쓰이는 생활 문제가 있나요? (예: 자녀교육, 건강, 경제적 문제 등)\n\n"
            "자연스러운 문장으로 답하세요. JSON이나 마크다운 형식 없이 일반 텍스트로만 작성하세요."
        )

        def generate_narrative(resident_id):
            agent = self.agents[resident_id]["agent"]
            response = agent.respond(narrative_task)
            return resident_id, response

        with ThreadPoolExecutor(max_workers=len(resident_ids)) as executor:
            futures = {executor.submit(generate_narrative, aid): aid for aid in resident_ids}
            narratives = {}
            for future in as_completed(futures):
                resident_id, response = future.result()
                narratives[resident_id] = response

        # Store narratives in memory (think slot)
        narrative_turn = 1
        for resident_id in resident_ids:
            narrative = narratives[resident_id]
            self.agents[resident_id]["agent"].memory.add_think(f"[서사 배경]\n{narrative}")

            self.logger.log_think(
                round=0,
                turn=narrative_turn,
                agent_id=resident_id,
                think_type="narrative",
                상대의견=None,
                생각=narrative
            )
            narrative_turn += 1
            logger.info(f"{resident_id} narrative generated")

        self.logger.save()

        # === 0-2. Pre-debate: Form initial opinions (parallel) ===
        logger.info("=== Forming initial opinions ===")

        initial_task = (
            "토론이 시작되기 전입니다. 아직 다른 주민의 의견을 듣지 않은 상태에서, "
            "당신의 페르소나와 상황을 바탕으로 주안2동 재개발에 대한 당신의 입장을 정하세요. "
            f'{{"입장": "매우찬성/찬성/반대/매우반대", "생각": "..."}} JSON만 출력. 마크다운 금지.'
        )

        def form_initial_opinion(resident_id):
            agent = self.agents[resident_id]["agent"]
            response = agent.respond(initial_task)
            return resident_id, parse_initial_opinion(response)

        with ThreadPoolExecutor(max_workers=len(resident_ids)) as executor:
            futures = {executor.submit(form_initial_opinion, aid): aid for aid in resident_ids}
            initial_opinions = {}
            for future in as_completed(futures):
                resident_id, parsed = future.result()
                initial_opinions[resident_id] = parsed

        # Store initial opinions in memory and log to CSV
        initial_turn = 1
        for resident_id in resident_ids:
            opinion = initial_opinions[resident_id]
            self.agents[resident_id]["initial_stance"] = opinion["입장"]
            self.agents[resident_id]["agent"].memory.add_think(
                f"[사전 의견] 입장: {opinion['입장']}, 이유: {opinion['생각']}"
            )
            # Log to think.csv with think_type="initial"
            self.logger.log_think(
                round=0,  # Round 0 = pre-debate
                turn=initial_turn,
                agent_id=resident_id,
                think_type="initial",
                상대의견=opinion["입장"],  # Store stance in 상대의견 field
                생각=opinion["생각"]
            )
            initial_turn += 1
            logger.info(f"{resident_id} initial stance: {opinion['입장']}")

        self.logger.save()

        # Save agent list (with initial opinions)
        self._save_agent_list(initial_opinions=initial_opinions)

        for round_num in range(1, self.n_rounds + 1):
            logger.info(f"=== Round {round_num} started ===")

            # Global think counter for unique codes within round
            think_turn = 1

            for turn, speaker_id in enumerate(resident_ids, start=1):
                # === 1. Speaker's turn ===
                speaker_data = self.agents[speaker_id]
                speaker_agent = speaker_data["agent"]
                speaker_persona = speaker_data["persona"]

                # Request speech
                task = '당신의 발화 차례입니다. 반드시 {"발화": "...", "지목": "..." or null, "입장": "..."} JSON만 출력하세요.'
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
                for other_id in resident_ids:
                    if other_id != speaker_id:
                        other_agent = self.agents[other_id]["agent"]
                        other_agent.memory.add_conversation(speaker_id, parsed["발화"])

                # === 3. Other agents react (parallel) ===
                other_ids = [aid for aid in resident_ids if aid != speaker_id]
                think_task = (
                    f"{speaker_id}의 발화(코드: {response_code}): "
                    f"'{parsed['발화'][:100]}'\n"
                    f'이 발화에 대한 생각을 {{"상대의견": "{response_code}", "생각": "..."}} JSON만 출력하세요.'
                )

                def do_think(other_id):
                    agent = self.agents[other_id]["agent"]
                    response = agent.respond(think_task)
                    return other_id, parse_think(response)

                with ThreadPoolExecutor(max_workers=len(other_ids)) as executor:
                    futures = {executor.submit(do_think, oid): oid for oid in other_ids}
                    results = {}
                    for future in as_completed(futures):
                        other_id, think_parsed = future.result()
                        results[other_id] = think_parsed

                # Process results in order
                for other_id in other_ids:
                    think_parsed = results[other_id]
                    self.agents[other_id]["agent"].memory.add_think(think_parsed["생각"])
                    self.logger.log_think(
                        round=round_num,
                        turn=think_turn,
                        agent_id=other_id,
                        think_type="reaction",
                        상대의견=think_parsed["상대의견"],
                        생각=think_parsed["생각"]
                    )
                    think_turn += 1

                # Auto-save after each speaker's turn
                self.logger.save()

            # === 4. End of round reflection (parallel) ===
            logger.info(f"=== Round {round_num} reflection ===")

            reflection_task = (
                f"{round_num}라운드가 끝났습니다. "
                f"지금까지 논의를 당신의 페르소나 관점에서 정리하세요. "
                f'반드시 {{"생각": "..."}} JSON만 출력하세요. 상대의견 필드는 포함하지 마세요.'
            )

            def do_reflect(resident_id):
                agent = self.agents[resident_id]["agent"]
                response = agent.respond(reflection_task)
                return resident_id, parse_think(response)

            with ThreadPoolExecutor(max_workers=len(resident_ids)) as executor:
                futures = {executor.submit(do_reflect, aid): aid for aid in resident_ids}
                results = {}
                for future in as_completed(futures):
                    resident_id, reflection_parsed = future.result()
                    results[resident_id] = reflection_parsed

            # Process results in order
            reflect_turn = 1
            for resident_id in resident_ids:
                reflection_parsed = results[resident_id]
                self.agents[resident_id]["agent"].memory.add_think(reflection_parsed["생각"])
                self.logger.log_think(
                    round=round_num,
                    turn=think_turn + reflect_turn,
                    agent_id=resident_id,
                    think_type="reflection",
                    상대의견=None,
                    생각=reflection_parsed["생각"]
                )
                reflect_turn += 1

            # Auto-save after reflections
            self.logger.save()

        # === 5. Post-debate: Form final opinions (parallel) ===
        logger.info("=== Forming final opinions ===")

        final_task = (
            "토론이 모두 끝났습니다. 다른 주민들의 의견을 모두 들은 지금, "
            "주안2동 재개발에 대한 당신의 최종 입장은 어떻습니까? "
            '{"입장": "매우찬성/찬성/반대/매우반대", "생각": "..."} JSON만 출력. 마크다운 금지.'
        )

        def form_final_opinion(resident_id):
            agent = self.agents[resident_id]["agent"]
            response = agent.respond(final_task)
            return resident_id, parse_initial_opinion(response)

        with ThreadPoolExecutor(max_workers=len(resident_ids)) as executor:
            futures = {executor.submit(form_final_opinion, aid): aid for aid in resident_ids}
            final_opinions = {}
            for future in as_completed(futures):
                resident_id, parsed = future.result()
                final_opinions[resident_id] = parsed

        # Log final opinions
        final_turn = 1
        for resident_id in resident_ids:
            opinion = final_opinions[resident_id]
            self.agents[resident_id]["final_stance"] = opinion["입장"]
            self.logger.log_think(
                round=self.n_rounds + 1,
                turn=final_turn,
                agent_id=resident_id,
                think_type="final",
                상대의견=opinion["입장"],
                생각=opinion["생각"]
            )
            final_turn += 1
            logger.info(f"{resident_id} final stance: {opinion['입장']}")

        self.logger.save()

        # Save agent list (with initial and final opinions)
        self._save_agent_list(initial_opinions=initial_opinions, final_opinions=final_opinions)

        logger.info(f"Simulation completed. Logs saved.")

    def _save_agent_list(
        self,
        initial_opinions: dict = None,
        final_opinions: dict = None
    ):
        """Save agent list with personas and opinions to CSV (incremental)."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        agent_list_path = output_path / f"set{self.set_id}_lv{self.level}_agents.csv"

        columns = [
            "resident_id", "model", "is_vulnerable",
            "연령대", "성별", "직업", "주거유형", "자가여부", "매수동기",
            "소득수준", "거주기간", "가구구성", "재개발지식",
            "개방성", "성실성", "외향성", "친화성", "신경성",
            "initial_stance", "final_stance"
        ]

        rows = []
        for persona in self.personas:
            resident_id = persona["resident_id"]
            bigfive = persona["BigFive"]

            row = {
                "resident_id": resident_id,
                "model": self.agents[resident_id]["model"],
                "is_vulnerable": persona["is_vulnerable"],
                "연령대": persona["연령대"],
                "성별": persona["성별"],
                "직업": persona["직업"],
                "주거유형": persona["주거유형"],
                "자가여부": persona["자가여부"],
                "매수동기": persona["매수동기"],
                "소득수준": persona["소득수준"],
                "거주기간": persona["거주기간"],
                "가구구성": persona["가구구성"],
                "재개발지식": persona["재개발지식"],
                "개방성": bigfive["개방성"],
                "성실성": bigfive["성실성"],
                "외향성": bigfive["외향성"],
                "친화성": bigfive["친화성"],
                "신경성": bigfive["신경성"],
                "initial_stance": "",
                "final_stance": ""
            }

            if initial_opinions and resident_id in initial_opinions:
                row["initial_stance"] = initial_opinions[resident_id]["입장"]

            if final_opinions and resident_id in final_opinions:
                row["final_stance"] = final_opinions[resident_id]["입장"]

            rows.append(row)

        with open(agent_list_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Agent list saved: {agent_list_path}")
