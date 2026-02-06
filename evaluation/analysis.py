"""Additional analysis metrics for debate simulation."""

import logging
from collections import Counter

logger = logging.getLogger(__name__)

STANCES = ["매우만족", "만족", "불만족", "매우불만족"]
RESPONSE_TYPES = ["공감", "비판", "질문", "인용"]


def stance_distribution(agent_rows: list[dict]) -> dict:
    """Count agents per stance for initial and final."""
    init = Counter(r["initial_stance"] for r in agent_rows)
    final = Counter(r["final_stance"] for r in agent_rows)
    result = {}
    for s in STANCES:
        result[f"init_{s}"] = init.get(s, 0)
        result[f"final_{s}"] = final.get(s, 0)
    return result


def stance_shift_rate(agent_rows: list[dict]) -> float:
    """Ratio of agents whose stance changed."""
    if not agent_rows:
        return 0.0
    shifted = sum(1 for r in agent_rows if r["initial_stance"] != r["final_stance"])
    return shifted / len(agent_rows)


def shift_direction(agent_rows: list[dict]) -> dict:
    """Transition matrix: count per (initial, final) pair."""
    transitions = Counter()
    for r in agent_rows:
        init = r["initial_stance"]
        final = r["final_stance"]
        if init != final:
            transitions[f"{init}→{final}"] += 1
    return dict(transitions)


def _flatten_nominations(debate_rows: list[dict]) -> list[dict]:
    """Extract all individual nominations from debate rows."""
    noms = []
    for row in debate_rows:
        rd = row.get("round", "")
        for nom in row.get("지목", []):
            noms.append({
                "round": rd,
                "대상": nom.get("대상", ""),
                "입장": nom.get("입장", ""),
            })
    return noms


def response_type_ratio(debate_rows: list[dict]) -> dict:
    """Ratio of each response type across all nominations."""
    noms = _flatten_nominations(debate_rows)
    total = len(noms)
    if total == 0:
        return {f"rt_{rt}": 0.0 for rt in RESPONSE_TYPES}
    counts = Counter(n["입장"] for n in noms)
    return {f"rt_{rt}": counts.get(rt, 0) / total for rt in RESPONSE_TYPES}


def response_type_by_round(debate_rows: list[dict]) -> dict:
    """Count of each response type per round."""
    noms = _flatten_nominations(debate_rows)
    by_round = {}
    for n in noms:
        rd = n["round"]
        key = (rd, n["입장"])
        by_round[key] = by_round.get(key, 0) + 1

    result = {}
    rounds = sorted(set(n["round"] for n in noms)) if noms else []
    for rd in rounds:
        for rt in RESPONSE_TYPES:
            result[f"rt_r{rd}_{rt}"] = by_round.get((rd, rt), 0)
    return result


def response_type_by_group(
    debate_rows: list[dict],
    vuln_ids: list[str],
    non_vuln_ids: list[str],
) -> dict:
    """Response type counts split by whether the nomination target is vulnerable."""
    vuln_set = set(vuln_ids)
    non_vuln_set = set(non_vuln_ids)
    noms = _flatten_nominations(debate_rows)

    vuln_counts = Counter()
    non_vuln_counts = Counter()
    for n in noms:
        target = n["대상"]
        if target in vuln_set:
            vuln_counts[n["입장"]] += 1
        elif target in non_vuln_set:
            non_vuln_counts[n["입장"]] += 1

    result = {}
    for rt in RESPONSE_TYPES:
        result[f"rt_vuln_{rt}"] = vuln_counts.get(rt, 0)
        result[f"rt_nonvuln_{rt}"] = non_vuln_counts.get(rt, 0)
    return result


def speech_length_by_round(debate_rows: list[dict]) -> dict:
    """Average speech length (character count) per round."""
    by_round = {}
    for row in debate_rows:
        rd = row.get("round", "")
        text = row.get("발화", "")
        by_round.setdefault(rd, []).append(len(text))

    result = {}
    for rd in sorted(by_round.keys()):
        lengths = by_round[rd]
        result[f"speech_r{rd}"] = sum(lengths) / len(lengths) if lengths else 0.0
    return result


def knowledge_effect(agent_rows: list[dict]) -> dict:
    """Stance shift rate by knowledge level."""
    by_level = {}
    for r in agent_rows:
        level = r.get("재개발지식", "")
        by_level.setdefault(level, []).append(r["initial_stance"] != r["final_stance"])

    levels = ["낮음", "보통", "높음"]
    result = {}
    for lv in levels:
        shifts = by_level.get(lv, [])
        result[f"ke_{lv}"] = sum(shifts) / len(shifts) if shifts else 0.0
    return result


def question_rate(debate_rows: list[dict]) -> float:
    """Ratio of nominations with response type '질문'."""
    noms = _flatten_nominations(debate_rows)
    total = len(noms)
    if total == 0:
        return 0.0
    q_count = sum(1 for n in noms if n["입장"] == "질문")
    return q_count / total
