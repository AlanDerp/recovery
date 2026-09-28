from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from capx.envs import trial as trial_module
from capx.utils.launch_utils import _parse_multi_turn_decision


def test_explicit_regenerate_is_preserved() -> None:
    decision, code = _parse_multi_turn_decision(
        "REGENERATE\n```python\ndo_work()\n```"
    )

    assert decision == "regenerate"
    assert code == "```python\ndo_work()\n```"


def test_exact_finish_is_preserved() -> None:
    assert _parse_multi_turn_decision("FINISH") == ("finish", None)


def test_single_unmarked_code_block_is_recovered() -> None:
    decision, code = _parse_multi_turn_decision(
        "The task is not complete. Continue now.\n```python\ndo_work()\n```"
    )

    assert decision == "regenerate"
    assert code == "```python\ndo_work()\n```"


def test_unmarked_plain_python_is_recovered() -> None:
    source = "# factual comment\npose = get_object_pose('cube')\ngoto_pose(*pose)"

    decision, code = _parse_multi_turn_decision(source)

    assert decision == "regenerate"
    assert code == f"```python\n{source}\n```"


@pytest.mark.parametrize(
    "content",
    (
        "",
        "The task is not complete, but no code follows.",
        "# comments alone are not executable",
        '"a string literal alone is not executable code"',
        "FINISH because it looks done.",
        "REGENERATE and later FINISH",
        "```python\nfirst()\n```\n```python\nsecond()\n```",
    ),
)
def test_ambiguous_or_malformed_response_fails_closed(content: str) -> None:
    with pytest.raises(ValueError):
        _parse_multi_turn_decision(content)


def test_v4_receives_executed_blocks_at_the_treatment_boundary() -> None:
    source = textwrap.dedent(inspect.getsource(trial_module._handle_multi_turn_step))
    tree = ast.parse(source)
    calls = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"run_takeover", "run_ideal_tk_v4"}
    }

    v4_keywords = {keyword.arg for keyword in calls["run_ideal_tk_v4"].keywords}
    closed_gt_keywords = {keyword.arg for keyword in calls["run_takeover"].keywords}
    assert "executed_code_blocks" in v4_keywords
    assert "executed_code_blocks" not in closed_gt_keywords
