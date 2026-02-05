"""Run inclusiveness evaluation on simulation output."""

import logging

from dotenv import load_dotenv
load_dotenv()

from .loader import load_agents, load_debate, load_think
from .metrics import attention_index, positive_nomination_count, opinion_diffusion_rate, extract_keywords

logger = logging.getLogger("evaluation.run")


def evaluate_set(output_dir: str, set_id: int, level: int) -> dict:
    """Evaluate one set and return metrics dict."""
    prefix = f"set{set_id}_lv{level}"

    vulnerable_ids, non_vulnerable_ids = load_agents(output_dir, prefix)
    debate_rows = load_debate(output_dir, prefix)
    think_rows = load_think(output_dir, prefix=prefix)

    rounds = sorted(set(int(r["round"]) for r in debate_rows))

    ai = attention_index(debate_rows, vulnerable_ids, non_vulnerable_ids)
    pnc = positive_nomination_count(debate_rows, vulnerable_ids)
    keywords = extract_keywords(think_rows, vulnerable_ids, non_vulnerable_ids)
    odr = opinion_diffusion_rate(debate_rows, vulnerable_ids, non_vulnerable_ids, keywords, rounds)

    result = {"set_id": set_id, "level": level, "ai": ai, "pnc": pnc["pnc"], "odr": odr["odr"], "keywords": keywords}
    logger.info(f"Set {set_id} Lv.{level} | AI={ai:.4f}, PNC={pnc['pnc']:.4f}, ODR={odr['odr']:.4f}")
    return result
