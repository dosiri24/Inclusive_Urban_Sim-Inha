"""
Debate Logger

Save debate results to CSV files with unique codes for each output.

Code format: {agent_id}_r{round}_{turn:02d}_{type}
- type: 'r' for response, 't' for think
- Example: agent_01_r1_03_r (agent_01's 3rd response in round 1)
"""

import csv
from pathlib import Path


def generate_code(agent_id: str, round: int, turn: int, output_type: str) -> str:
    """
    Generate unique code for LLM output.

    Args:
        agent_id: "agent_01" ~ "agent_20"
        round: Round number (1, 2, 3)
        turn: Turn number within the round
        output_type: "r" for response, "t" for think

    Returns:
        Code string like "agent_01_r1_03_r"
    """
    return f"{agent_id}_r{round}_{turn:02d}_{output_type}"


class DebateLogger:
    """Save debate and think logs to CSV files."""

    def __init__(self, set_id: int, level: int, output_dir: str = "outputs/"):
        self.set_id = set_id
        self.level = level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for batch writing
        self.debate_buffer = []
        self.think_buffer = []

        # CSV column headers
        self.debate_columns = [
            "code", "set_id", "level", "round", "turn",
            "agent_id", "model", "is_vulnerable", "persona_summary",
            "발화", "지목", "입장"
        ]
        self.think_columns = [
            "code", "set_id", "level", "round", "turn",
            "agent_id", "think_type", "상대의견", "생각"
        ]

    def log_debate(
        self,
        round: int,
        turn: int,
        agent_id: str,
        model: str,
        is_vulnerable: bool,
        persona_summary: str,
        발화: str,
        지목: str,
        입장: str
    ) -> str:
        """
        Log one debate response.

        Returns:
            Generated code for this response
        """
        code = generate_code(agent_id, round, turn, "r")

        self.debate_buffer.append({
            "code": code,
            "set_id": self.set_id,
            "level": self.level,
            "round": round,
            "turn": turn,
            "agent_id": agent_id,
            "model": model,
            "is_vulnerable": is_vulnerable,
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
        생각: str
    ) -> str:
        """
        Log one think response.

        Args:
            think_type: "reaction" (response to specific utterance) or
                       "reflection" (end-of-round summary)
            상대의견: Code of the response being reacted to (can be None for reflection)

        Returns:
            Generated code for this think
        """
        code = generate_code(agent_id, round, turn, "t")

        self.think_buffer.append({
            "code": code,
            "set_id": self.set_id,
            "level": self.level,
            "round": round,
            "turn": turn,
            "agent_id": agent_id,
            "think_type": think_type,
            "상대의견": 상대의견,
            "생각": 생각
        })

        return code

    def save(self):
        """Save buffers to CSV files."""
        # File paths: outputs/set{set_id}_lv{level}_debate.csv
        debate_path = self.output_dir / f"set{self.set_id}_lv{self.level}_debate.csv"
        think_path = self.output_dir / f"set{self.set_id}_lv{self.level}_think.csv"

        # Save debate log
        if self.debate_buffer:
            self._write_csv(debate_path, self.debate_columns, self.debate_buffer)

        # Save think log
        if self.think_buffer:
            self._write_csv(think_path, self.think_columns, self.think_buffer)

    def _write_csv(self, path: Path, columns: list, data: list):
        """Write data to CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
