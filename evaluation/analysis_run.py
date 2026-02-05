"""Orchestrator for additional analysis metrics."""

import csv
import json
import logging
from pathlib import Path

from .loader import _find_csv, load_agents, load_debate
from .analysis import (
    stance_distribution,
    stance_shift_rate,
    shift_direction,
    response_type_ratio,
    response_type_by_round,
    response_type_by_group,
    speech_length_by_round,
    knowledge_effect,
    question_rate,
)

logger = logging.getLogger(__name__)


def load_agents_full(directory: str, prefix: str = "") -> list[dict]:
    """Load agents.csv and return full rows including demographics and stances."""
    path = _find_csv(Path(directory), "agents", prefix)
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize stance spaces from LLM output (e.g. "조건부 반대" -> "조건부반대")
            for key in ("initial_stance", "final_stance"):
                if key in row and row[key]:
                    row[key] = row[key].replace(" ", "")
            rows.append(row)
    logger.debug(f"Loaded {len(rows)} full agent rows")
    return rows


def analyze_set(output_dir: str, set_id: int, level: int) -> dict:
    """Run all additional analyses for one set and return flattened dict."""
    prefix = f"set{set_id}_lv{level}"

    agent_rows = load_agents_full(output_dir, prefix)
    vuln_ids, non_vuln_ids = load_agents(output_dir, prefix)
    debate_rows = load_debate(output_dir, prefix)

    result = {"level": level, "set_id": set_id}

    result.update(stance_distribution(agent_rows))
    result["shift_rate"] = stance_shift_rate(agent_rows)
    result["shift_direction"] = json.dumps(
        shift_direction(agent_rows), ensure_ascii=False
    )
    result.update(response_type_ratio(debate_rows))
    result.update(response_type_by_round(debate_rows))
    result.update(response_type_by_group(debate_rows, vuln_ids, non_vuln_ids))
    result.update(speech_length_by_round(debate_rows))
    result.update(knowledge_effect(agent_rows))
    result["question_rate"] = question_rate(debate_rows)

    logger.info(
        f"Set {set_id} Lv.{level} | shift_rate={result['shift_rate']:.4f}, "
        f"question_rate={result['question_rate']:.4f}"
    )
    return result
