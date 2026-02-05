"""CSV data loading for evaluation module."""

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger("evaluation.loader")


def _find_csv(directory: Path, suffix: str, prefix: str = "") -> Path:
    """Find CSV file matching the suffix pattern in directory."""
    pattern = f"{prefix}*_{suffix}.csv" if prefix else f"*_{suffix}.csv"
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    if len(matches) > 1:
        logger.warning(f"Multiple {suffix} CSVs found, using: {matches[0]}")
    return matches[0]


def load_agents(directory: str, prefix: str = "") -> tuple[list[str], list[str]]:
    """
    Load agents.csv and split into vulnerable/non-vulnerable ID lists.

    Returns:
        (vulnerable_ids, non_vulnerable_ids)
    """
    path = _find_csv(Path(directory), "agents", prefix)

    vulnerable = []
    non_vulnerable = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["resident_id"]
            if row["is_vulnerable"] == "True":
                vulnerable.append(rid)
            else:
                non_vulnerable.append(rid)

    logger.debug(f"Loaded agents: {len(vulnerable)} vulnerable, {len(non_vulnerable)} non-vulnerable")
    return vulnerable, non_vulnerable


def load_debate(directory: str, prefix: str = "") -> list[dict]:
    """
    Load debate.csv and parse 지목 JSON column.

    Returns:
        List of row dicts with 지목 parsed from JSON string to list.
    """
    path = _find_csv(Path(directory), "debate", prefix)
    rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("지목", "[]")
            try:
                row["지목"] = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse 지목 JSON: {raw[:80]}")
                row["지목"] = []
            rows.append(row)

    logger.debug(f"Loaded {len(rows)} debate rows")
    return rows


def load_think(directory: str, think_type: str = None, prefix: str = "") -> list[dict]:
    """
    Load think.csv, optionally filter by think_type.

    Args:
        directory: Output directory path
        think_type: Filter value (e.g. "initial", "narrative"). None = all rows.
        prefix: File name prefix filter (e.g. "set1_lv1").

    Returns:
        List of row dicts.
    """
    path = _find_csv(Path(directory), "think", prefix)
    rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if think_type and row.get("think_type") != think_type:
                continue
            rows.append(row)

    logger.debug(f"Loaded {len(rows)} think rows (filter={think_type})")
    return rows
