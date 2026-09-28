"""Analyze a failed Agilex trial and produce hints for the next outer retry."""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from capx.llm.client import ModelQueryArgs, query_model

_ATTEMPT_FOCUS = {
    1: "Focus on the primary failure mode (grasp miss, collision, IK jump, or placement miss).",
    2: (
        "Previous retry likely reused a similar plan. Focus on perception refresh, grasp yaw/height, "
        "and Z-only lift before any large XY move."
    ),
    3: (
        "Two prior outer attempts failed. Propose a substantially different pick-place sequence: "
        "new approach direction, slower descent, re-segment ball/box, or conservative box placement."
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _append_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _load_history_excerpt(history_file: str | None, max_chars: int = 8000) -> str:
    if not history_file or not os.path.isfile(history_file):
        return ""
    text = Path(history_file).read_text(encoding="utf-8").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _split_analysis_sections(analysis_text: str) -> tuple[str, str]:
    failure = analysis_text.strip()
    fixes = ""
    if "FAILURE_ANALYSIS" in analysis_text:
        after = analysis_text.split("FAILURE_ANALYSIS", 1)[1]
        if "NEXT_ATTEMPT_FIXES" in after:
            failure, fixes = after.split("NEXT_ATTEMPT_FIXES", 1)
        else:
            failure = after
    elif "NEXT_ATTEMPT_FIXES" in analysis_text:
        failure, fixes = analysis_text.split("NEXT_ATTEMPT_FIXES", 1)
    failure = failure.strip().lstrip(":").strip()
    fixes = fixes.strip().lstrip(":").strip()
    return failure, fixes


def _format_log_entry(
    *,
    attempt: int,
    trial_dir: Path,
    num_images: int,
    failure_analysis: str,
    next_fixes: str,
    full_analysis: str,
) -> str:
    lines = [
        "",
        "=" * 72,
        f"OUTER FAILURE ANALYSIS | attempt {attempt} | {_utc_now()}",
        f"Trial: {trial_dir.name}",
        f"Images used: {num_images}",
        "-" * 72,
        "FAILURE_ANALYSIS:",
        failure_analysis or "(empty)",
        "-" * 72,
        "NEXT_ATTEMPT_FIXES:",
        next_fixes or "(empty)",
        "-" * 72,
        "FULL MODEL RESPONSE:",
        full_analysis or "(empty)",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


def _find_latest_failed_trial(output_dir: str) -> Path | None:
    patterns = [
        f"{output_dir}/**/trial_*_taskcompleted_0",
        f"{output_dir}/**/trial_*_sandboxrc_*_taskcompleted_0",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0])


def _load_stderr_excerpt(trial_dir: Path, max_chars: int = 6000) -> str:
    responses_path = trial_dir / "all_responses.json"
    if not responses_path.is_file():
        return ""
    try:
        responses = json.loads(responses_path.read_text())
    except json.JSONDecodeError:
        return ""
    stderr_parts: list[str] = []
    for entry in responses:
        if isinstance(entry, dict):
            stderr = entry.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                stderr_parts.append(stderr.strip())
    combined = "\n\n".join(stderr_parts)
    if len(combined) > max_chars:
        return combined[-max_chars:]
    return combined


def _encode_image_path(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _sample_video_frames(video_path: Path, max_frames: int = 4) -> list[str]:
    if not video_path.is_file():
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        count = min(max_frames, frame_count)
        if count == 1:
            indices = [max(frame_count - 1, 0)]
        else:
            indices = np.linspace(0, frame_count - 1, count, dtype=int).tolist()
        urls: list[str] = []
        for idx in dict.fromkeys(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            urls.append(
                f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
        return urls
    finally:
        cap.release()


def _collect_image_urls(trial_dir: Path, max_images: int = 4) -> list[str]:
    urls: list[str] = []
    for path in sorted(trial_dir.glob("visual_feedback_*.png")):
        urls.append(_encode_image_path(path))
    video_path = trial_dir / "video_combined.mp4"
    urls.extend(_sample_video_frames(video_path, max_frames=max(1, max_images - len(urls))))
    return urls[:max_images]


def analyze_failed_trial(
    *,
    output_dir: str,
    attempt: int,
    hints_file: str,
    analysis_file: str | None = None,
    log_file: str | None = None,
    history_file: str | None = None,
    model: str = "google/gemini-2.5-flash-lite",
    server_url: str = "http://127.0.0.1:8110/chat/completions",
    api_key: str | None = None,
    max_images: int = 4,
) -> dict[str, Any]:
    trial_dir = _find_latest_failed_trial(output_dir)
    if trial_dir is None:
        raise FileNotFoundError(f"No failed trial artifacts found under {output_dir}")

    summary = (trial_dir / "summary.txt").read_text() if (trial_dir / "summary.txt").is_file() else ""
    code = (trial_dir / "code.py").read_text() if (trial_dir / "code.py").is_file() else ""
    stderr_excerpt = _load_stderr_excerpt(trial_dir)
    image_urls = _collect_image_urls(trial_dir, max_images=max_images)
    history_excerpt = _load_history_excerpt(history_file)
    attempt_focus = _ATTEMPT_FOCUS.get(attempt, _ATTEMPT_FOCUS[3])

    history_block = ""
    if history_excerpt:
        history_block = (
            "\n\nPrevious outer-attempt analyses (these fixes did NOT succeed yet):\n"
            f"{history_excerpt}\n\n"
            "IMPORTANT: Do NOT repeat the same NEXT_ATTEMPT_FIXES bullets. "
            "Compare the new camera images against prior failures and propose NEW, "
            "substantially different corrections.\n"
        )

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Outer retry attempt {attempt} failed for an Agilex tennis-ball pick-and-place task.\n"
                f"{attempt_focus}\n"
                "Analyze the logs and camera images, then produce concise corrective guidance "
                "for the next full trial.\n\n"
                f"Trial directory: {trial_dir}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Final code:\n```python\n{code[:8000]}\n```\n\n"
                f"Stderr excerpt:\n```\n{stderr_excerpt}\n```\n"
                f"{history_block}\n"
                "Respond in two sections:\n"
                "1. FAILURE_ANALYSIS: what went wrong visually and in execution (bullet points; "
                "note what is DIFFERENT from prior attempts if history is provided)\n"
                "2. NEXT_ATTEMPT_FIXES: concrete code/strategy changes for the next attempt "
                "(bullet points; mention Z-only lift, safe_z, re-perception after failure, "
                "gripper timing, and box approach if relevant; must differ from prior fixes)\n"
            ),
        }
    ]
    for i, url in enumerate(image_urls, start=1):
        user_content.append({"type": "text", "text": f"Camera frame {i}/{len(image_urls)}:"})
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a robotics failure analyst for a single-arm Agilex/Piper pick-and-place task. "
                "Use the top-camera images and execution logs together. Be specific and actionable. "
                "Do not write code blocks in NEXT_ATTEMPT_FIXES; only strategy and corrections."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    query_args = ModelQueryArgs(
        model=model,
        server_url=server_url,
        api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )
    result = query_model(query_args, prompt)
    analysis_text = result.get("content") or ""
    failure_analysis, next_fixes = _split_analysis_sections(analysis_text)
    hints = next_fixes or analysis_text

    hints_path = Path(hints_file)
    hints_path.parent.mkdir(parents=True, exist_ok=True)
    hints_section = textwrap.dedent(
        f"""
        ## After outer attempt {attempt} (trial: {trial_dir.name})

        ### Failure summary
        {failure_analysis or "(no FAILURE_ANALYSIS section)"}

        ### Required fixes for next attempt
        {hints.strip()}
        """
    ).strip()
    if hints_path.is_file() and hints_path.stat().st_size > 0:
        _append_text_file(hints_path, "\n\n" + hints_section + "\n")
    else:
        hints_path.write_text(
            "# Outer retry guidance (cumulative)\n\n" + hints_section + "\n",
            encoding="utf-8",
        )

    analysis_path = Path(analysis_file) if analysis_file else hints_path.with_name(
        f"outer_failure_analysis_attempt_{attempt:02d}.txt"
    )
    analysis_path.write_text(analysis_text, encoding="utf-8")

    history_path = Path(history_file) if history_file else None
    if history_path is not None:
        history_entry = textwrap.dedent(
            f"""
            --- attempt {attempt} | trial {trial_dir.name} | {_utc_now()} ---
            FAILURE_ANALYSIS:
            {failure_analysis or "(empty)"}

            NEXT_ATTEMPT_FIXES:
            {hints.strip()}
            """
        ).strip()
        _append_text_file(history_path, "\n\n" + history_entry + "\n")

    log_entry = _format_log_entry(
        attempt=attempt,
        trial_dir=trial_dir,
        num_images=len(image_urls),
        failure_analysis=failure_analysis,
        next_fixes=hints,
        full_analysis=analysis_text,
    )
    if log_file:
        _append_text_file(Path(log_file), log_entry)
    print(log_entry, flush=True)

    print(f"[outer] Wrote failure analysis to {analysis_path}")
    print(f"[outer] Wrote cumulative retry hints to {hints_path}")
    if log_file:
        print(f"[outer] Appended failure analysis log to {log_file}")
    if history_path is not None:
        print(f"[outer] Appended failure history to {history_path}")

    return {
        "trial_dir": str(trial_dir),
        "analysis_path": str(analysis_path),
        "hints_path": str(hints_path),
        "log_file": log_file,
        "history_file": str(history_path) if history_path else None,
        "analysis_text": analysis_text,
        "failure_analysis": failure_analysis,
        "next_fixes": hints,
        "num_images": len(image_urls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--hints-file", required=True)
    parser.add_argument("--analysis-file", default=None)
    parser.add_argument(
        "--log-file",
        default=os.environ.get(
            "OUTER_FAILURE_LOG",
            "logs/outer_failure_analysis.log",
        ),
    )
    parser.add_argument(
        "--history-file",
        default=os.environ.get("OUTER_FAILURE_HISTORY", ""),
    )
    parser.add_argument("--model", default=os.environ.get("OUTER_ANALYSIS_MODEL", "google/gemini-2.5-flash-lite"))
    parser.add_argument(
        "--server-url",
        default=os.environ.get(
            "OUTER_ANALYSIS_SERVER_URL",
            "http://127.0.0.1:8110/chat/completions",
        ),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--max-images", type=int, default=4)
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key and os.path.isfile(".openrouterkey"):
        api_key = next(
            (
                line.strip()
                for line in Path(".openrouterkey").read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ),
            None,
        )

    analyze_failed_trial(
        output_dir=args.output_dir,
        attempt=args.attempt,
        hints_file=args.hints_file,
        analysis_file=args.analysis_file,
        log_file=args.log_file,
        history_file=args.history_file or None,
        model=args.model,
        server_url=args.server_url,
        api_key=api_key,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
