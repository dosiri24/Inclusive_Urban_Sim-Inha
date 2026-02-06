"""Inclusiveness metrics: AI, PNC, ODR."""

import os
import logging

from llm_api.gemini import MODEL_NAME as GEMINI_MODEL

logger = logging.getLogger("evaluation.metrics")


def attention_index(debate_rows: list[dict], vulnerable_ids: list[str], non_vulnerable_ids: list[str]) -> float:
    """
    Attention Index = mean(vulnerable nominated count) / mean(non-vulnerable nominated count).

    Counts how many times each agent is nominated (지목) regardless of stance.
    """
    counts = {}
    for row in debate_rows:
        for nom in row.get("지목", []):
            target = nom.get("대상", "")
            if target:
                counts[target] = counts.get(target, 0) + 1

    vuln_counts = [counts.get(rid, 0) for rid in vulnerable_ids]
    non_vuln_counts = [counts.get(rid, 0) for rid in non_vulnerable_ids]

    if not vuln_counts or not non_vuln_counts:
        logger.warning("Empty vulnerable or non-vulnerable group for AI calculation")
        return 0.0

    mean_vuln = sum(vuln_counts) / len(vuln_counts)
    mean_non_vuln = sum(non_vuln_counts) / len(non_vuln_counts)

    if mean_non_vuln == 0:
        logger.warning("Non-vulnerable mean nomination is 0, returning 0")
        return 0.0

    return mean_vuln / mean_non_vuln


def positive_nomination_count(debate_rows: list[dict], vulnerable_ids: list[str]) -> dict:
    """
    Positive nominations per vulnerable agent (공감/인용).

    Returns:
        {"pnc": float, "total": int, "by_round": {round_num: count}}
    """
    vuln_set = set(vulnerable_ids)
    total = 0
    by_round = {}

    for row in debate_rows:
        round_num = int(row.get("round", 0))
        for nom in row.get("지목", []):
            stance = nom.get("입장", "")
            target = nom.get("대상", "")
            if target in vuln_set and stance in ("공감", "인용"):
                total += 1
                by_round[round_num] = by_round.get(round_num, 0) + 1

    n_vuln = len(vulnerable_ids)
    pnc = total / n_vuln if n_vuln > 0 else 0.0

    return {"pnc": pnc, "total": total, "by_round": by_round}


def opinion_diffusion_rate(
    debate_rows: list[dict],
    vulnerable_ids: list[str],
    non_vulnerable_ids: list[str],
    keywords: list[str],
    rounds: list[int]
) -> dict:
    """
    ODR = keyword mentions by non-vulnerable / (num_rounds * num_non_vulnerable).

    Checks non-vulnerable agents' 발화 for keyword occurrences.

    Returns:
        {"odr": float, "mention_count": int, "by_round": {round_num: count}}
    """
    vuln_set = set(vulnerable_ids)
    non_vuln_set = set(non_vulnerable_ids)

    mention_total = 0
    by_round = {}

    for row in debate_rows:
        speaker = row.get("resident_id", "")
        if speaker not in non_vuln_set:
            continue

        round_num = int(row.get("round", 0))
        text = row.get("발화", "")

        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            mention_total += count
            by_round[round_num] = by_round.get(round_num, 0) + count

    denominator = len(rounds) * len(non_vulnerable_ids)
    odr = mention_total / denominator if denominator > 0 else 0.0

    return {"odr": odr, "mention_count": mention_total, "by_round": by_round}


def extract_keywords(
    think_rows: list[dict],
    vulnerable_ids: list[str],
    non_vulnerable_ids: list[str],
) -> list[str]:
    """Extract keywords unique to vulnerable agents by comparing both groups."""
    vuln_set = set(vulnerable_ids)
    non_vuln_set = set(non_vulnerable_ids)

    vuln_texts = []
    non_vuln_texts = []
    for row in think_rows:
        if row.get("think_type") != "initial":
            continue
        text = row.get("생각", "").strip()
        if not text:
            continue
        rid = row.get("resident_id")
        if rid in vuln_set:
            vuln_texts.append(text)
        elif rid in non_vuln_set:
            non_vuln_texts.append(text)

    if not vuln_texts:
        logger.warning("No initial texts found for vulnerable agents")
        return []

    vuln_combined = "\n---\n".join(vuln_texts)
    non_vuln_combined = "\n---\n".join(non_vuln_texts) if non_vuln_texts else "(없음)"

    prompt = (
        "재개발 토의에서 두 그룹의 초기 의견을 비교하여, "
        "취약계층에만 고유하게 나타나는 우려 키워드를 추출하세요.\n\n"
        "=== 취약계층 의견 ===\n"
        f"{vuln_combined}\n\n"
        "=== 일반주민 의견 ===\n"
        f"{non_vuln_combined}\n\n"
        "규칙:\n"
        "- 양쪽 모두 쓰는 공통 용어(분담금, 임대주택, 아파트, 비례율 등)는 제외\n"
        "- 취약계층의 처지/감정/상황을 반영하는 고유 표현만 추출\n"
        "- 토의 발화에서 직접 검색 가능한 구체적 단어 (2~4글자)\n"
        "- 정확히 5개, 쉼표로 구분, 다른 텍스트 없이\n"
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )

    raw = response.text.strip()
    keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords
