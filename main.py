"""Research experiment runner for inclusive urban simulation."""

import csv
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from debate.simulation_lv1 import DebateSimulationLv1
from debate.simulation import DebateSimulation
from evaluation.run import evaluate_set

if __name__ == "__main__":
    # Lv.1
    for set_id in range(11, 11):
        sim = DebateSimulationLv1(
            set_id=set_id, n_rounds=3, n_agents=10, n_vulnerable=2,
            output_dir="outputs/lv1/", model="gemini",
        )
        sim.run()

    # Lv.2
    for set_id in range(1, 11):
        sim = DebateSimulation(
            set_id=set_id, level=2, n_rounds=3, n_agents=10, n_vulnerable=2,
            output_dir="outputs/lv2/", model="gemini",
        )
        sim.run()

    # Evaluation
    logger = logging.getLogger("main")
    logger.info("=== Evaluation ===")

    all_results = []

    for level, output_dir in [(1, "outputs/lv1/"), (2, "outputs/lv2/")]:
        results = []
        for set_id in range(1, 11):
            result = evaluate_set(output_dir, set_id, level)
            results.append(result)

        avg_ai = sum(r["ai"] for r in results) / len(results)
        avg_pnc = sum(r["pnc"] for r in results) / len(results)
        avg_odr = sum(r["odr"] for r in results) / len(results)

        logger.info(f"Lv.{level} Average | AI={avg_ai:.4f}, PNC={avg_pnc:.4f}, ODR={avg_odr:.4f}")

        all_results.extend(results)
        all_results.append({"set_id": "avg", "level": level, "ai": avg_ai, "pnc": avg_pnc, "odr": avg_odr, "keywords": ""})

    # Save evaluation results to CSV
    with open("outputs/evaluation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "set_id", "ai", "pnc", "odr", "keywords"])
        writer.writeheader()
        writer.writerows(all_results)

    logger.info("Evaluation saved: outputs/evaluation.csv")
