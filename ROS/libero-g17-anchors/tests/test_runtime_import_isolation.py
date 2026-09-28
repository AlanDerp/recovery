from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).parents[1]
AUDITOR = (
    REPO_ROOT
    / "experiments"
    / "core_fg"
    / "runtime"
    / "tools"
    / "verify_runtime_imports.py"
)


def _run_audit(project: Path, import_root: Path, output: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(import_root)
    return subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--project",
            str(project),
            "--output",
            str(output),
            "samplepkg",
            "samplepkg.module",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_runtime_import_audit_accepts_frozen_tree(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    package = frozen / "samplepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "module.py").write_text("VALUE = 1\n")
    output = tmp_path / "audit.json"

    result = _run_audit(frozen, frozen, output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "valid"
    assert all(item["within_frozen_project"] for item in payload["modules"])


def test_runtime_import_audit_rejects_live_tree(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    live = tmp_path / "live"
    package = live / "samplepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "module.py").write_text("VALUE = 1\n")
    output = tmp_path / "audit.json"

    result = _run_audit(frozen, live, output)

    assert result.returncode != 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "invalid"
    assert not any(item["within_frozen_project"] for item in payload["modules"])
