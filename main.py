"""Research experiment runner for inclusive urban simulation."""

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

if __name__ == "__main__":
    # Lv.1
    for set_id in range(31, 31):
        sim = DebateSimulationLv1(
            set_id=set_id, n_rounds=3, n_agents=10, n_vulnerable=2,
            output_dir="outputs/lv1/", model="gemini",
        )
        sim.run()

    # Lv.2
    for set_id in range(31, 31):
        sim = DebateSimulation(
            set_id=set_id, level=2, n_rounds=3, n_agents=10, n_vulnerable=2,
            output_dir="outputs/lv2/", model="gemini",
        )
        sim.run()

    # Lv.3
    for set_id in range(31, 31):
        sim = DebateSimulation(
            set_id=set_id, level=3, n_rounds=3, n_agents=10, n_vulnerable=2,
            output_dir="outputs/lv3/", 
            model=["gemini", "chatgpt", "kimi", "claudecode", "grok"],
        )
        sim.run()