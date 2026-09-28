#!/usr/bin/env python3
"""Export an exact request-hash replay package from a CaP-X Qwen run.

The Qwen proxy already records a canonical SHA-256 for every request and a
SHA-256 for the assistant content.  This tool reconstructs the content from
run artifacts and writes a replay file consumable by:

    qwen_proxy.py --request-hash-replay-file replay.json

It intentionally fails closed by default: if any audited successful response
cannot be matched to saved content, the replay package is not trustworthy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


NORMALIZATION_VERSION = "image_url_redacted_verifier_frame_removed_v2"
_VERIFIER_WRAPPER_FRAME_RE = re.compile(
    r'(?m)^[ \t]*File "(?:<CAPX_RUNTIME>|[^"\n]*)/capx/verification/'
    r'[^"\n]+\.py", line \d+, in wrapped\n(?:[ \t]+[^\n]*(?:\n|$))?'
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_request_for_replay(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if key == "image_url" and isinstance(child, dict) and "url" in child:
                image_child = dict(child)
                image_child["url"] = "<CAPX_IMAGE_URL_REDACTED_FOR_REPLAY>"
                normalized[key] = image_child
            else:
                normalized[key] = normalize_request_for_replay(child)
        return normalized
    if isinstance(value, list):
        return [normalize_request_for_replay(item) for item in value]
    if isinstance(value, str):
        return _VERIFIER_WRAPPER_FRAME_RE.sub("", value)
    return value


def normalized_request_hash_from_canonical(request_canonical: str | None) -> str | None:
    if not request_canonical:
        return None
    body = json.loads(request_canonical)
    normalized_canonical = json.dumps(
        normalize_request_for_replay(body),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(normalized_canonical.encode()).hexdigest()


def image_fingerprints_from_canonical(
    request_canonical: str | None,
) -> list[str] | None:
    if not request_canonical:
        return None
    body = json.loads(request_canonical)
    fingerprints: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            image = value.get("image_url")
            if isinstance(image, dict):
                url = image.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    _, separator, encoded = url.partition(",")
                    if not separator:
                        raise ValueError("malformed request image data URL")
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
                        rgb = source.convert("RGB")
                        payload = (
                            rgb.width.to_bytes(4, "big")
                            + rgb.height.to_bytes(4, "big")
                            + rgb.tobytes()
                        )
                    fingerprints.append(hashlib.sha256(payload).hexdigest())
            for key, child in value.items():
                if key != "image_url":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(body)
    return fingerprints


def student_content(entry: dict[str, Any]) -> str | None:
    blocks = entry.get("code_blocks") or []
    if not blocks:
        return None
    decision = str(entry.get("decision", "")).lower()
    if decision == "initial":
        return str(blocks[0])
    if decision == "regenerate":
        return "REGENERATE\n" + "```python\n" + str(blocks[-1]) + "\n```"
    return str(blocks[-1])


def trial_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.match(r"trial_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def collect_saved_contents(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    contents: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(run_root.glob("**/all_responses.json")):
        try:
            source = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(source, list):
            continue
        for index, entry in enumerate(source):
            if not isinstance(entry, dict):
                continue
            content = student_content(entry)
            if content is None:
                continue
            digest = sha256_text(content)
            contents.setdefault(digest, []).append(
                {
                    "role": "student",
                    "trial": trial_from_path(path),
                    "path": str(path.relative_to(run_root)),
                    "index": index,
                    "decision": entry.get("decision"),
                    "content": content,
                }
            )

    for path in sorted(run_root.glob("**/coach_traces/**/*raw_content*.txt")):
        content = path.read_text(errors="replace")
        digest = sha256_text(content)
        contents.setdefault(digest, []).append(
            {
                "role": "coach",
                "trial": trial_from_path(path),
                "path": str(path.relative_to(run_root)),
                "content": content,
            }
        )

    return contents


def load_audit(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_line_no"] = line_no
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial replay file even when some audited content is missing.",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    audit_path = run_root / "qwen_proxy_audit.jsonl"
    if not audit_path.exists():
        raise SystemExit(f"Missing audit log: {audit_path}")

    audit = load_audit(audit_path)
    saved = collect_saved_contents(run_root)
    entries: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    duplicate_requests: list[str] = []

    for index, record in enumerate(audit):
        if not record.get("ok", False):
            continue
        request_hash = str(record.get("request_sha256") or "")
        response_hash = str(record.get("response_content_sha256") or "")
        if not request_hash or not response_hash:
            missing.append(
                {
                    "audit_index": index,
                    "audit_line": record.get("_line_no"),
                    "reason": "missing request or response hash",
                }
            )
            continue

        if "response_content" in record:
            content = str(record.get("response_content") or "")
            source = {
                "role": "audit_raw_content",
                "trial": None,
                "path": str(audit_path.relative_to(run_root)),
                "content": content,
            }
            if sha256_text(content) != response_hash:
                missing.append(
                    {
                        "audit_index": index,
                        "audit_line": record.get("_line_no"),
                        "request_sha256": request_hash,
                        "response_content_sha256": response_hash,
                        "reason": "audit raw content hash mismatch",
                    }
                )
                continue
        else:
            candidates = saved.get(response_hash) or []
            if not candidates:
                missing.append(
                    {
                        "audit_index": index,
                        "audit_line": record.get("_line_no"),
                        "request_sha256": request_hash,
                        "response_content_sha256": response_hash,
                        "content_len": record.get("content_len"),
                        "reason": "response content not found in saved artifacts",
                    }
                )
                continue
            source = candidates[0]
            content = str(source["content"])

        if request_hash in entries and entries[request_hash]["response_sha256"] != response_hash:
            duplicate_requests.append(request_hash)
            continue

        recomputed_normalized_hash = normalized_request_hash_from_canonical(
            record.get("request_canonical")
        )
        entries[request_hash] = {
            "content": content,
            "response_sha256": response_hash,
            "role": source.get("role"),
            "trial": source.get("trial"),
            "source_path": source.get("path"),
            "audit_index": index,
            "audit_line": record.get("_line_no"),
            "model": record.get("model"),
            "effective_request_seed": record.get("effective_request_seed"),
            "upstream_seed": record.get("upstream_seed"),
            "request_normalized_sha256": (
                recomputed_normalized_hash
                or record.get("request_normalized_sha256")
            ),
            "request_normalization_version": (
                NORMALIZATION_VERSION
                if recomputed_normalized_hash is not None
                else record.get("request_normalization_version")
                or NORMALIZATION_VERSION
            ),
            "request_image_rgb_sha256": (
                record.get("request_image_rgb_sha256")
                if isinstance(record.get("request_image_rgb_sha256"), list)
                else image_fingerprints_from_canonical(record.get("request_canonical"))
            ),
        }

    result = {
        "schema": "capx_request_hash_replay_v1",
        "source_run_root": str(run_root),
        "source_audit": str(audit_path),
        "audit_records": len(audit),
        "ok_audit_records": sum(1 for record in audit if record.get("ok", False)),
        "entries_count": len(entries),
        "request_normalization_version": NORMALIZATION_VERSION,
        "missing_count": len(missing),
        "duplicate_request_conflicts": sorted(duplicate_requests),
        "missing": missing,
        "entries": entries,
    }

    if missing and not args.allow_missing:
        print(
            json.dumps(
                {
                    "source_run_root": str(run_root),
                    "audit_records": result["audit_records"],
                    "ok_audit_records": result["ok_audit_records"],
                    "entries_count": result["entries_count"],
                    "missing_count": result["missing_count"],
                    "missing": missing[:20],
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(
            "Replay export is incomplete. Re-run with QWEN_AUDIT_RAW_CONTENT=1 "
            "or pass --allow-missing for diagnostics only."
        )
    if duplicate_requests:
        raise SystemExit(
            f"Conflicting responses for identical request hash: {duplicate_requests[:5]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_run_root": str(run_root),
                "audit_records": result["audit_records"],
                "ok_audit_records": result["ok_audit_records"],
                "entries_count": result["entries_count"],
                "missing_count": result["missing_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
