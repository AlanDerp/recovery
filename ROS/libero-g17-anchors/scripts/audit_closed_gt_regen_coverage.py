#!/usr/bin/env python3
"""Audit factual concept mentions in frozen M3 regeneration reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CONCEPT_PATTERNS = {
    "relative_or_topology": (
        r"\b(?:above|below|under|on top|underneath|left|right|side-by-side|"
        r"next to|adjacent|stacked|separate|apart)\b"
    ),
    "table_or_support": r"\b(?:table|surface|resting|support)\b",
    "gripper_configuration": r"\b(?:gripper|open|closed|partially closed)\b",
    "holding_or_contact": r"\b(?:hold|holding|held|grasp|grasped|contact|touch)\w*\b",
    "motion_or_change": (
        r"\b(?:move|moved|remain|remains|same|change|changed|retract|lifted|"
        r"dropped|placed)\w*\b"
    ),
    "orientation": r"\b(?:flat|tilt|tilted|upright|orientation|rotated)\w*\b",
    "visibility": r"\b(?:visible|not visible|occluded|frame|view)\b",
    "completion": r"\b(?:task|goal|completed|achieved)\b",
    "advice": r"\b(?:next step|should|need to|must|requires?|further action)\b",
}


def _native_regen_reports(path: Path) -> list[str]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("entries") if isinstance(value, dict) else None
    if not isinstance(entries, dict):
        raise ValueError(f"Replay has no entries object: {path}")
    reports: list[str] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            raise ValueError(f"Replay entry is not an object: {path}")
        images = entry.get("request_image_rgb_sha256")
        content = entry.get("content")
        if (
            isinstance(images, list)
            and len(images) == 2
            and isinstance(content, str)
            and "```python" not in content
            and not content.startswith("REGENERATE")
        ):
            reports.append(content)
    return reports


def audit(paths: list[Path]) -> dict[str, Any]:
    per_file: dict[str, int] = {}
    reports: list[str] = []
    for path in paths:
        selected = _native_regen_reports(path)
        per_file[str(path)] = len(selected)
        reports.extend(selected)
    mentions = {
        concept: sum(bool(re.search(pattern, report, re.IGNORECASE)) for report in reports)
        for concept, pattern in CONCEPT_PATTERNS.items()
    }
    return {
        "method": "two_image_non_code_audit_entries_v1",
        "per_file_report_count": per_file,
        "report_count": len(reports),
        "unique_report_count": len(set(reports)),
        "lexical_concept_mentions": mentions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(audit(args.replays), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
