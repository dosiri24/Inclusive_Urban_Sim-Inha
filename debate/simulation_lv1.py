"""
Lv.1 Debate Simulation

Single session, single model simulation.
All agents are simulated within one LLM context using batch outputs.
"""

import csv
import json
import logging
from pathlib import Path

from llm_api import get_llm
from logger import setup_file_logger, DebateLogger, TokenLogger
from .config import N_ROUNDS, N_AGENTS, N_VULNERABLE, PROMPTS_DIR, OUTPUT_DIR
from .persona import generate_all_personas
from .parser import (
    parse_batch_narrative, parse_batch_opinion, parse_batch_speech,
    parse_planner_result, parse_batch_vote
)
from .planner import compile_debate_text, compile_final_opinions, run_planner
from prompts.tasks import (
    get_lv1_narrative_task,
    get_lv1_initial_task,
    get_lv1_speaking_task,
    get_lv1_final_task,
    get_lv1_vote_task,
)

logger = logging.getLogger("debate.simulation_lv1")


def _load_prompt_file(path: str) -> str:
    """Load prompt content from file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return p.read_text(encoding="utf-8")


def _persona_to_summary(persona: dict) -> str:
    """Convert persona dict to short summary string."""
    return (
        f"{persona['연령대']} {persona['성별']} {persona['직업']} "
        f"{persona['자가여부']} 거주{persona['거주기간']}"
    )


def _persona_to_line(persona: dict) -> str:
    """Convert persona to single-line description for batch prompt."""
    resident_id = persona["resident_id"]
    bigfive = persona["BigFive"]

    line = (
        f"{resident_id}: {persona['연령대']} {persona['성별']}, "
        f"직업={persona['직업']}, 주거={persona['주거유형']}, "
        f"소유={persona['자가여부']}, 소득={persona['연소득']}, "
        f"거주기간={persona['거주기간']}, 가구={persona['가구구성']}, "
        f"재개발지식={persona['재개발지식']}"
    )

    if persona["자가여부"] == "자가":
        line += f", 매수동기={persona['매수동기']}"

    if persona.get("스토리"):
        line += f" [상황: {persona['스토리'][:50]}...]"

    return line


class DebateSimulationLv1:
    """
    Lv.1: Single session, single model simulation.

    All agents are simulated in one LLM context.
    Each phase outputs batch JSON for all agents at once.
    """

    def __init__(
        self,
        set_id: int,
        n_rounds: int = N_ROUNDS,
        n_agents: int = N_AGENTS,
        n_vulnerable: int = N_VULNERABLE,
        prompts_dir: str = PROMPTS_DIR,
        output_dir: str = OUTPUT_DIR,
        model: str = "gemini"
    ):
        self.set_id = set_id
        self.level = 1
        self.n_rounds = n_rounds
        self.n_agents = n_agents
        self.output_dir = output_dir
        self.prompts_dir = Path(prompts_dir)
        self.model_name = model

        # Load static prompts
        self.system_guide = _load_prompt_file(self.prompts_dir / "system_guide.md")
        self.debate_rule = _load_prompt_file(self.prompts_dir / "debate_rule.md")
        self.local_context = _load_prompt_file(self.prompts_dir / "local_context.md")

        # Generate personas
        self.personas = generate_all_personas(n_agents, n_vulnerable)

        # Initialize single LLM instance
        self.llm = get_llm(model)

        # Initialize loggers
        self.logger = DebateLogger(set_id, self.level, output_dir)
        self.token_logger = TokenLogger(set_id, self.level, output_dir)

        # Setup file logger
        self.log_path = setup_file_logger(output_dir, set_id, self.level)

        # Cumulative conversation timeline (analogous to Lv2 Memory.timeline)
        self.timeline = ""

        logger.info(f"Lv.1 Simulation initialized: set={set_id}, agents={n_agents}, model={model}")
        logger.info(f"Log file: {self.log_path}")

    def _build_participants_prompt(self) -> str:
        """Build participants list for batch prompts."""
        lines = [_persona_to_line(p) for p in self.personas]
        return "\n".join(lines)

    def _build_context_prompt(self) -> str:
        """Build full context prompt (system + rules + local + participants)."""
        return f"""[System Context]
{self.system_guide}

[Debate Rule]
{self.debate_rule}

[Local Context]
{self.local_context}

[참여자 목록]
{self._build_participants_prompt()}"""

    def _append_timeline(self, section: str, content: str):
        """Append a section to the cumulative timeline."""
        self.timeline += f"\n\n[{section}]\n{content}"

    def _call_llm(self, task: str, task_type: str) -> tuple[str, dict]:
        """Call LLM with full context + accumulated timeline + task."""
        prompt_data = {
            "system": self._build_context_prompt(),
            "timeline": self.timeline,
            "new_timeline": "",
            "task": task
        }

        response, usage = self.llm.chat_with_retry(prompt_data)

        # Log token usage (batch call)
        self.token_logger.log(
            agent_id="batch",
            model=self.model_name,
            task_type=task_type,
            target=None,
            round=0,
            turn=0,
            usage=usage
        )

        return response, usage

    def run(self):
        """Run the Lv.1 debate simulation."""
        resident_ids = [p["resident_id"] for p in self.personas]

        # === Phase 0-1: Generate narratives ===
        logger.info("=== Generating narratives (batch) ===")

        narrative_task = get_lv1_narrative_task()
        response, _ = self._call_llm(narrative_task, "narrative")
        narratives = parse_batch_narrative(response)

        for rid in resident_ids:
            narrative = narratives.get(rid, "[서사 없음]")
            self.logger.log_think(
                round=0,
                turn=resident_ids.index(rid) + 1,
                agent_id=rid,
                think_type="narrative",
                상대의견=None,
                반응유형=None,
                생각=narrative
            )
            logger.debug(f"{rid} narrative: {narrative[:50]}...")

        self.logger.save()
        self.token_logger.save()

        # Append full narratives to timeline
        narrative_lines = [
            f"{rid}: {narratives.get(rid, '')}"
            for rid in resident_ids
        ]
        self._append_timeline("서사", "\n".join(narrative_lines))

        # === Phase 0-2: Generate initial opinions ===
        logger.info("=== Generating initial opinions (batch) ===")

        initial_task = get_lv1_initial_task()
        response, _ = self._call_llm(initial_task, "initial")
        initial_opinions = parse_batch_opinion(response)

        for rid in resident_ids:
            opinion = initial_opinions.get(rid, {"입장": "무응답", "생각": ""})
            self.logger.log_think(
                round=0,
                turn=resident_ids.index(rid) + 1,
                agent_id=rid,
                think_type="initial",
                상대의견=opinion["입장"],
                반응유형=None,
                생각=opinion["생각"]
            )
            logger.info(f"{rid} initial stance: {opinion['입장']}")

        self.logger.save()
        self.token_logger.save()

        # Save agent list with initial opinions
        self._save_agent_list(initial_opinions=initial_opinions)

        # Append full initial opinions to timeline
        opinion_lines = [
            f"{rid}: [{initial_opinions.get(rid, {}).get('입장', '무응답')}] "
            f"{initial_opinions.get(rid, {}).get('생각', '')}"
            for rid in resident_ids
        ]
        self._append_timeline("초기입장", "\n".join(opinion_lines))

        # === Phase 1-N: Debate rounds ===
        for round_num in range(1, self.n_rounds + 1):
            logger.info(f"=== Round {round_num} (batch) ===")

            speaking_task = get_lv1_speaking_task(round_num)
            max_retries = 3
            speeches = None
            for attempt in range(1, max_retries + 1):
                response, _ = self._call_llm(speaking_task, f"speak_r{round_num}")
                speeches = parse_batch_speech(response)
                if speeches is not None:
                    break
                logger.warning(f"Round {round_num} speech parse failed (attempt {attempt}/{max_retries}), retrying...")
            if speeches is None:
                logger.error(f"Round {round_num} speech parse failed after {max_retries} attempts, skipping round")
                speeches = []

            # Log each speech
            for turn, speech in enumerate(speeches, 1):
                rid = speech["resident_id"]
                persona = next((p for p in self.personas if p["resident_id"] == rid), None)

                self.logger.log_debate(
                    round=round_num,
                    turn=turn,
                    agent_id=rid,
                    model=self.model_name,
                    is_vulnerable=persona["is_vulnerable"] if persona else False,
                    취약유형=persona.get("취약유형", "N/A") if persona else "N/A",
                    persona_summary=_persona_to_summary(persona) if persona else "",
                    발화=speech["발화"],
                    지목=speech["지목"]
                )
                logger.debug(f"{rid} spoke: {speech['발화'][:50]}...")

            self.logger.save()
            self.token_logger.save()

            # Append full round speeches to timeline
            speech_lines = []
            for s in speeches:
                지목_str = ", ".join([
                    f"{j['대상']}({j.get('입장', '')})"
                    for j in s.get("지목", [])
                ])
                speech_lines.append(
                    f"{s['resident_id']}: {s['발화']} (지목: {지목_str or '없음'})"
                )
            self._append_timeline(f"{round_num}라운드", "\n".join(speech_lines))

        # === Phase 4: Generate final opinions ===
        logger.info("=== Generating final opinions (batch) ===")

        final_task = get_lv1_final_task()
        response, _ = self._call_llm(final_task, "final")
        final_opinions = parse_batch_opinion(response)

        for rid in resident_ids:
            opinion = final_opinions.get(rid, {"입장": "무관심", "생각": ""})
            self.logger.log_think(
                round=self.n_rounds + 1,
                turn=resident_ids.index(rid) + 1,
                agent_id=rid,
                think_type="final",
                상대의견=opinion["입장"],
                반응유형=None,
                생각=opinion["생각"]
            )
            logger.info(f"{rid} final stance: {opinion['입장']}")

        self.logger.save()
        self.token_logger.save()

        # Save agent list with initial and final opinions
        self._save_agent_list(initial_opinions=initial_opinions, final_opinions=final_opinions)

        # === Phase 5: Urban planner compromise ===
        logger.info("=== Urban planner synthesizing (Lv.1) ===")

        planner_guide = _load_prompt_file(self.prompts_dir / "planner_guide.md")
        debate_text = compile_debate_text(self.logger.debate_buffer)
        opinions_text = compile_final_opinions(final_opinions)

        planner_result, planner_usage = run_planner(
            "gemini", planner_guide, debate_text, opinions_text,
            debate_rule=self.debate_rule, local_context=self.local_context,
        )

        consensus_text = planner_result["최종합의문"]

        self.logger.log_think(
            round=self.n_rounds + 2,
            turn=1,
            agent_id="planner",
            think_type="planner",
            상대의견=None,
            반응유형=None,
            생각=consensus_text
        )
        self.token_logger.log(
            agent_id="planner",
            model="gemini",
            task_type="synthesize",
            target=None,
            round=self.n_rounds + 2,
            turn=1,
            usage=planner_usage
        )
        self.logger.save_consensus(planner_result)
        self.logger.save()
        self.token_logger.save()

        # Append consensus to timeline for resident vote
        self._append_timeline("도시계획가 합의문", consensus_text)

        # === Phase 6: Resident vote on compromise (batch) ===
        logger.info("=== Resident voting on compromise (Lv.1 batch) ===")

        vote_task = get_lv1_vote_task()
        response, vote_usage = self._call_llm(vote_task, "vote")
        vote_results = parse_batch_vote(response)

        for rid in resident_ids:
            vote = vote_results.get(rid, {"입장": "무응답", "이유": ""})
            self.logger.log_think(
                round=self.n_rounds + 2,
                turn=resident_ids.index(rid) + 2,
                agent_id=rid,
                think_type="vote",
                상대의견=vote["입장"],
                반응유형=None,
                생각=vote["이유"]
            )
            logger.info(f"{rid} vote: {vote['입장']}")

        self.logger.save()
        self.token_logger.save()

        logger.info("Lv.1 Simulation completed.")

    def _save_agent_list(
        self,
        initial_opinions: dict = None,
        final_opinions: dict = None
    ):
        """Save agent list with personas and opinions to CSV."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        agent_list_path = output_path / f"set{self.set_id}_lv{self.level}_agents.csv"

        columns = [
            "resident_id", "model", "is_vulnerable",
            "연령대", "성별", "직업", "주거유형", "자가여부", "매수동기",
            "연소득", "거주기간", "가구구성", "재개발지식",
            "개방성", "성실성", "외향성", "친화성", "신경성",
            "initial_stance", "final_stance"
        ]

        rows = []
        for persona in self.personas:
            resident_id = persona["resident_id"]
            bigfive = persona["BigFive"]

            row = {
                "resident_id": resident_id,
                "model": self.model_name,
                "is_vulnerable": persona["is_vulnerable"],
                "연령대": persona["연령대"],
                "성별": persona["성별"],
                "직업": persona["직업"],
                "주거유형": persona["주거유형"],
                "자가여부": persona["자가여부"],
                "매수동기": persona["매수동기"],
                "연소득": persona["연소득"],
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
