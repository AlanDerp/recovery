#!/usr/bin/env python3
"""Create, preview, and run a deterministic 48-alpha BRAIN research batch.

The default mode performs no write requests. Use --execute only after reviewing the
persisted manifest. This runner simulates candidates; it never calls /submit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests

API_ROOT = "https://api.worldquantbrain.com"
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "research" / "runs"
DEFAULT_SEED = "new-alpha-ideas-2026-08-16"
BATCH_SIZE = 48
FAMILY_SIZE = 8
MAX_WORKERS = 3
REQUEST_TIMEOUT = (10, 60)
MAX_POLL_SECONDS = 1_800

SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P0Y",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
}


class BrainError(RuntimeError):
    """A BRAIN request could not be completed safely."""


class PersonaRequired(BrainError):
    """The account requires interactive biometric authentication."""


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    family: str
    expression: str
    inputs: tuple[str, ...]
    fingerprint: str


class Ledger:
    """Small JSON ledger with atomic rewrites for deterministic recovery."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data
        self.lock = threading.Lock()

    @classmethod
    def create(cls, path: Path, manifest: list[Candidate], seed: str) -> "Ledger":
        settings_hash = digest(json.dumps(SETTINGS, sort_keys=True))
        data = {
            "schema_version": 1,
            "created_at": utc_now(),
            "seed": seed,
            "settings": SETTINGS,
            "settings_hash": settings_hash,
            "manifest_hash": digest("\n".join(candidate.fingerprint for candidate in manifest)),
            "candidates": [
                {**asdict(candidate), "inputs": list(candidate.inputs), "state": "planned", "events": []}
                for candidate in manifest
            ],
        }
        ledger = cls(path, data)
        ledger.save()
        return ledger

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        with path.open() as handle:
            return cls(path, json.load(handle))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.path)

    def transition(self, ordinal: int, state: str, **details: Any) -> None:
        with self.lock:
            row = self.data["candidates"][ordinal - 1]
            row["state"] = state
            event = {"at": utc_now(), "state": state, **details}
            row["events"].append(event)
            for key, value in details.items():
                if key not in {"message", "http_status"}:
                    row[key] = value
            self.save()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical(expression: str) -> str:
    return " ".join(expression.split())


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_credentials() -> tuple[str, str]:
    username = os.environ.get("API_USERNAME")
    password = os.environ.get("API_PASSWORD")
    if username and password:
        return username, password

    for credential_file in (ROOT / "brain.txt", ROOT / "2_QuantResearch" / "brain.txt"):
        if credential_file.is_file():
            with credential_file.open() as handle:
                credentials = json.load(handle)
            if isinstance(credentials, list) and len(credentials) == 2:
                return str(credentials[0]), str(credentials[1])

    raise BrainError(
        "Missing credentials. Set API_USERNAME/API_PASSWORD or place brain.txt in the project root or 2_QuantResearch/."
    )


class BrainClient:
    def __init__(self) -> None:
        self.username, self.password = load_credentials()
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.auth = (self.username, self.password)
            self.local.session = session
            self.authenticate(session)
        return session

    def authenticate(self, session: requests.Session) -> None:
        response = session.post(f"{API_ROOT}/authentication", timeout=REQUEST_TIMEOUT)
        if response.status_code == 401 and response.headers.get("WWW-Authenticate") == "persona":
            location = response.headers.get("Location")
            challenge = urljoin(response.url, location) if location else "(no challenge URL returned)"
            raise PersonaRequired(f"Interactive persona authentication is required: {challenge}")
        if not response.ok:
            raise BrainError(f"Authentication failed with HTTP {response.status_code}: {safe_message(response)}")

    def request(self, method: str, url: str, *, retry_auth: bool = True, **kwargs: Any) -> requests.Response:
        session = self.session()
        response = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        if response.status_code == 401 and retry_auth:
            self.authenticate(session)
            response = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        return response

    def datafields(self, dataset_id: str) -> list[dict[str, Any]]:
        params = {
            "instrumentType": SETTINGS["instrumentType"],
            "region": SETTINGS["region"],
            "delay": SETTINGS["delay"],
            "universe": SETTINGS["universe"],
            "dataset.id": dataset_id,
            "limit": 50,
            "offset": 0,
        }
        first = self.request("GET", f"{API_ROOT}/data-fields", params=params)
        if not first.ok:
            raise BrainError(f"Could not fetch datafields for {dataset_id}: HTTP {first.status_code}: {safe_message(first)}")
        body = first.json()
        count = body.get("count")
        if not isinstance(count, int):
            raise BrainError(f"Unexpected datafields response for {dataset_id}: missing count")
        fields = list(body.get("results", []))
        # A first page gives more than the eight fields required per family and avoids
        # triggering a rate limit by exhaustively downloading an entire dataset.
        return fields


def safe_message(response: requests.Response) -> str:
    try:
        body = response.json()
        message = body.get("message") or body.get("detail") or str(body)
    except ValueError:
        message = response.text
    return " ".join(str(message).split())[:500]


def process_field_records(fields: list[dict[str, Any]]) -> list[str]:
    """Mirror machine_lib.process_datafields without requiring pandas at runtime."""
    expressions: list[str] = []
    for field in fields:
        field_id = field.get("id")
        if not field_id:
            continue
        if field.get("type") == "MATRIX":
            expressions.append(str(field_id))
        elif field.get("type") == "VECTOR":
            expressions.extend((f"vec_avg({field_id})", f"vec_sum({field_id})"))
    return [f"winsorize(ts_backfill({expression}, 120), std=4)" for expression in expressions]


def select_fields(fields: list[dict[str, Any]], count: int, *, seed: str) -> list[str]:
    usable = [field for field in fields if field.get("type") in {"MATRIX", "VECTOR"} and field.get("id")]
    processed = sorted(set(process_field_records(usable)))
    if len(processed) < count:
        raise BrainError(f"Only {len(processed)} compatible fields available; need at least {count}.")
    chooser = random.Random(seed)
    return chooser.sample(processed, count)


def make_candidates(analyst_fields: list[str], fundamental_fields: list[str], seed: str) -> list[Candidate]:
    """Create eight bounded candidates per family using established FastExpr operators."""
    pool: list[tuple[str, str, tuple[str, ...]]] = []

    for field in analyst_fields[:8]:
        pool.append(("multi_timeframe", f"ts_rank({field}, 22) - ts_rank({field}, 120)", (field,)))

    # Avoid sigmoid/tanh: their availability is not proven in this account's saved operator catalog.
    for index, field in enumerate(analyst_fields[:8]):
        day = (22, 66)[index % 2]
        base = f"ts_zscore({field}, {day})"
        expression = f"log(1 + abs({base})) * sign({base})" if index % 2 == 0 else f"signed_power({base}, 0.5)"
        pool.append(("nonlinear_robust", expression, (field,)))

    for analyst, fundamental in zip(analyst_fields[:8], fundamental_fields[:8], strict=True):
        pool.append((
            "cross_dataset",
            f"ts_zscore({analyst}, 22) * ts_zscore({fundamental}, 66)",
            (analyst, fundamental),
        ))

    for index, field in enumerate(analyst_fields[:8]):
        expression = (
            f"rank(ts_delta({field}, 22)) - rank(ts_delta({field}, 120))"
            if index % 2 == 0
            else f"ts_delta({field}, 22) - ts_delta({field}, 44)"
        )
        pool.append(("momentum_reversal", expression, (field,)))

    for index, field in enumerate(fundamental_fields[:8]):
        volatility = f"rank(ts_std_dev({field}, 66))"
        expression = (
            f"ts_zscore({field}, 22) * (1 - {volatility})"
            if index % 2 == 0
            else f"{volatility} * (-ts_delta({field}, 120)) + (1 - {volatility}) * ts_delta({field}, 22)"
        )
        pool.append(("regime_adaptive", expression, (field,)))

    event_conditions = (
        "ts_mean(volume, 10) > ts_mean(volume, 60)",
        "ts_arg_max(volume, 5) == 0",
        "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
        "ts_arg_max(close, 20) == 0",
    )
    for index, field in enumerate(analyst_fields[:8]):
        entry = event_conditions[index % len(event_conditions)]
        exit_condition = "abs(returns) > 0.1" if index % 2 == 0 else "-1"
        pool.append(("event_driven", f"trade_when({entry}, rank(ts_delta({field}, 22)), {exit_condition})", (field,)))

    groups: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for family, expression, inputs in pool:
        groups.setdefault(family, []).append((canonical(expression), inputs))

    candidates: list[Candidate] = []
    ordinal = 1
    for family in (
        "multi_timeframe",
        "nonlinear_robust",
        "cross_dataset",
        "momentum_reversal",
        "regime_adaptive",
        "event_driven",
    ):
        choices = groups.get(family, [])
        if len(choices) != FAMILY_SIZE:
            raise BrainError(f"{family}: expected {FAMILY_SIZE} candidates, found {len(choices)}")
        for expression, inputs in choices:
            candidates.append(Candidate(ordinal, family, expression, inputs, digest(expression)))
            ordinal += 1

    fingerprints = {candidate.fingerprint for candidate in candidates}
    if len(candidates) != BATCH_SIZE or len(fingerprints) != BATCH_SIZE:
        raise BrainError("Manifest must contain exactly 48 distinct expressions.")
    return candidates


def poll_simulation(client: BrainClient, progress_url: str) -> tuple[str, str | None, str | None]:
    parsed = urlparse(progress_url)
    if not (parsed.scheme == "https" and parsed.netloc == urlparse(API_ROOT).netloc):
        raise BrainError("BRAIN returned an unsafe simulation progress URL.")

    deadline = time.monotonic() + MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        response = client.request("GET", progress_url)
        if response.status_code >= 400:
            return "poll_failed", None, f"HTTP {response.status_code}: {safe_message(response)}"
        body = response.json()
        status = str(body.get("status", "UNKNOWN"))
        if status in {"COMPLETE", "WARNING"}:
            alpha = body.get("alpha")
            alpha_id = alpha if isinstance(alpha, str) else (alpha or {}).get("id") if isinstance(alpha, dict) else None
            return status.lower(), alpha_id, None
        if status in {"ERROR", "FAILED", "CANCELLED"}:
            return status.lower(), None, safe_message(response)
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after is not None else 5.0
        except ValueError:
            delay = 5.0
        time.sleep(min(max(delay, 1.0), 60.0))
    return "poll_timeout", None, f"Polling exceeded {MAX_POLL_SECONDS} seconds."


def run_candidate(client: BrainClient, ledger: Ledger, candidate: Candidate) -> None:
    row = ledger.data["candidates"][candidate.ordinal - 1]
    if row["state"] in {"complete", "warning", "rejected", "failed"}:
        return
    if row["state"] == "posted" and row.get("progress_url"):
        progress_url = row["progress_url"]
    else:
        ledger.transition(candidate.ordinal, "posting")
        payload = {"type": "REGULAR", "settings": SETTINGS, "regular": candidate.expression}
        try:
            response = client.request("POST", f"{API_ROOT}/simulations", json=payload)
        except requests.RequestException as error:
            # Outcome is ambiguous; do not automatically POST again on resume.
            ledger.transition(candidate.ordinal, "post_unknown", message=str(error))
            return
        if response.status_code not in {200, 201}:
            ledger.transition(candidate.ordinal, "rejected", http_status=response.status_code, message=safe_message(response))
            return
        location = response.headers.get("Location")
        if not location:
            ledger.transition(candidate.ordinal, "post_unknown", http_status=response.status_code, message="Successful response did not include Location.")
            return
        progress_url = urljoin(API_ROOT, location)
        ledger.transition(candidate.ordinal, "posted", http_status=response.status_code, progress_url=progress_url)

    ledger.transition(candidate.ordinal, "polling", progress_url=progress_url)
    try:
        state, alpha_id, message = poll_simulation(client, progress_url)
        ledger.transition(candidate.ordinal, state, alpha_id=alpha_id, message=message)
    except (BrainError, requests.RequestException, ValueError) as error:
        ledger.transition(candidate.ordinal, "failed", message=str(error))


def preview(manifest: Iterable[Candidate]) -> None:
    print(f"Prepared {BATCH_SIZE} unique candidates: {FAMILY_SIZE} per strategy family.")
    print(f"Settings: region={SETTINGS['region']}, universe={SETTINGS['universe']}, decay={SETTINGS['decay']}, neutralization={SETTINGS['neutralization']}")
    for candidate in manifest:
        print(f"{candidate.ordinal:02d}  {candidate.family:18s}  {candidate.fingerprint[:12]}  {candidate.expression}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send real simulation POST requests after manifest generation.")
    parser.add_argument("--run-id", help="Use this identifier for the persistent ledger; defaults to a timestamp.")
    parser.add_argument("--resume", type=Path, help="Resume a ledger; post_unknown candidates are not reposted.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Deterministic source-field selection seed.")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, MAX_WORKERS + 1), metavar="1..3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = BrainClient()
    try:
        analyst = select_fields(client.datafields("analyst4"), 8, seed=f"{args.seed}:analyst")
        fundamental = select_fields(client.datafields("fundamental6"), 8, seed=f"{args.seed}:fundamental")
    except (BrainError, PersonaRequired, requests.RequestException, ValueError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 2

    manifest = make_candidates(analyst, fundamental, args.seed)
    preview(manifest)

    if args.resume:
        ledger = Ledger.load(args.resume)
        expected_hash = digest("\n".join(candidate.fingerprint for candidate in manifest))
        if ledger.data.get("manifest_hash") != expected_hash or ledger.data.get("settings") != SETTINGS:
            print("Resume refused: manifest or settings do not match the existing ledger.", file=sys.stderr)
            return 2
    else:
        run_id = args.run_id or datetime.now(UTC).strftime("new-alpha-sample-%Y%m%dT%H%M%SZ")
        ledger = Ledger.create(RUNS_DIR / f"{run_id}.json", manifest, args.seed)

    print(f"Ledger: {ledger.path}")
    if not args.execute:
        print("Preview only: no simulation POST requests were sent. Re-run with --execute to simulate this manifest.")
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_candidate, client, ledger, candidate) for candidate in manifest]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    counts: dict[str, int] = {}
    for row in ledger.data["candidates"]:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print("Final states: " + ", ".join(f"{state}={count}" for state, count in sorted(counts.items())))
    print(f"Ledger: {ledger.path}")
    return 1 if counts.get("post_unknown") or counts.get("poll_timeout") else 0


if __name__ == "__main__":
    raise SystemExit(main())
