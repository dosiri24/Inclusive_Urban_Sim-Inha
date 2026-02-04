"""
Centralized logging for debate simulation.

Contains:
- setup_file_logger: Configure file handler for Python logging
- DebateLogger: Save debate/think results to CSV
- TokenLogger: Save LLM token usage to CSV
"""

import csv
import logging
from pathlib import Path


# =============================================================================
# File Logger Setup
# =============================================================================

def setup_file_logger(output_dir: str, set_id: int, level: int) -> Path:
    """
    Setup file handler for all Python logging.

    Returns:
        Path to the log file
    """
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


# =============================================================================
# Debate Logger (CSV)
# =============================================================================

def _generate_code(agent_id: str, round: int, turn: int, output_type: str) -> str:
    """Generate unique code for LLM output."""
    return f"{agent_id}_r{round}_{turn:02d}_{output_type}"


class DebateLogger:
    """Save debate and think logs to CSV files."""

    def __init__(self, set_id: int, level: int, output_dir: str = "outputs/"):
        self.set_id = set_id
        self.level = level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.debate_buffer = []
        self.think_buffer = []

        self.debate_columns = [
            "code", "set_id", "level", "round", "turn",
            "resident_id", "model", "is_vulnerable", "취약유형", "persona_summary",
            "발화", "지목", "입장"
        ]
        self.think_columns = [
            "code", "set_id", "level", "round", "turn",
            "resident_id", "think_type", "상대의견", "반응유형", "생각"
        ]

    def log_debate(
        self,
        round: int,
        turn: int,
        agent_id: str,
        model: str,
        is_vulnerable: bool,
        취약유형: str,
        persona_summary: str,
        발화: str,
        지목: str,
        입장: str
    ) -> str:
        """Log one debate response. Returns generated code."""
        code = _generate_code(agent_id, round, turn, "r")

        self.debate_buffer.append({
            "code": code,
            "set_id": self.set_id,
            "level": self.level,
            "round": round,
            "turn": turn,
            "resident_id": agent_id,
            "model": model,
            "is_vulnerable": is_vulnerable,
            "취약유형": 취약유형,
            "persona_summary": persona_summary,
            "발화": 발화,
            "지목": 지목,
            "입장": 입장
        })

        return code

    def log_think(
        self,
        round: int,
        turn: int,
        agent_id: str,
        think_type: str,
        상대의견: str,
        반응유형: str,
        생각: str
    ) -> str:
        """Log one think response. Returns generated code."""
        code = _generate_code(agent_id, round, turn, "t")

        self.think_buffer.append({
            "code": code,
            "set_id": self.set_id,
            "level": self.level,
            "round": round,
            "turn": turn,
            "resident_id": agent_id,
            "think_type": think_type,
            "상대의견": 상대의견,
            "반응유형": 반응유형,
            "생각": 생각
        })

        return code

    def save(self):
        """Save buffers to CSV files."""
        debate_path = self.output_dir / f"set{self.set_id}_lv{self.level}_debate.csv"
        think_path = self.output_dir / f"set{self.set_id}_lv{self.level}_think.csv"

        if self.debate_buffer:
            self._write_csv(debate_path, self.debate_columns, self.debate_buffer)

        if self.think_buffer:
            self._write_csv(think_path, self.think_columns, self.think_buffer)

    def _write_csv(self, path: Path, columns: list, data: list):
        """Write data to CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)


# =============================================================================
# Token Logger (CSV)
# =============================================================================

class TokenLogger:
    """Save LLM token usage to CSV file."""

    def __init__(self, set_id: int, level: int, output_dir: str = "outputs/"):
        self.set_id = set_id
        self.level = level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = []
        self.columns = [
            "timestamp", "agent_id", "model", "task_type", "target",
            "round", "turn", "cached", "prompt", "completion"
        ]

    def log(
        self,
        agent_id: str,
        model: str,
        task_type: str,
        target: str,
        round: int,
        turn: int,
        usage: dict
    ):
        """
        Log one LLM call's token usage.

        Args:
            agent_id: "resident_01" ~ "resident_20"
            model: LLM model name
            task_type: "narrative", "initial", "speak", "think", "reflect", "final"
            target: Target agent ID for think (e.g., "resident_03"), None otherwise
            round: Round number (0 for pre-debate)
            turn: Turn number
            usage: {"cached": int, "prompt": int, "completion": int}
        """
        from datetime import datetime

        self.buffer.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "model": model,
            "task_type": task_type,
            "target": target or "",
            "round": round,
            "turn": turn,
            "cached": usage.get("cached", 0),
            "prompt": usage.get("prompt", 0),
            "completion": usage.get("completion", 0)
        })

    def save(self):
        """Save buffer to CSV file."""
        path = self.output_dir / f"set{self.set_id}_lv{self.level}_tokens.csv"

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.buffer)
