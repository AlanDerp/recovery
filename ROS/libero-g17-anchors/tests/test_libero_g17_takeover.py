"""Unit tests for LIBERO G17 FG takeover helpers (no MuJoCo / LLM)."""

from __future__ import annotations

from capx.verification import libero_g17_takeover as takeover


def test_replace_report_once_changes_only_carrier_text() -> None:
    native = "NATIVE_VDM_REPORT_UNIQUE"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"prefix\n{native}\nsuffix"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        }
    ]
    replacement = "Verified ordered state changes over the executed interval:\n- no eye-checkable state changes."
    out = takeover._replace_report_once(prompt, original=native, replacement=replacement)
    assert takeover._non_text_structure(prompt) == takeover._non_text_structure(out)
    text = out[0]["content"][0]["text"]
    assert native not in text
    assert replacement in text
    assert out[0]["content"][1]["type"] == "image_url"


def test_native_task_language_uses_low_level_env_handle() -> None:
    class _Handle:
        task_language = (
            "Pick the akita black bowl between the plate and the ramekin "
            "and place it on the plate"
        )

    class _Low:
        handle = _Handle()

    class _Env:
        low_level_env = _Low()

    assert takeover._native_task_language(_Env()) == _Handle.task_language
    assert takeover._native_task_language(object()) is None


def test_goal_referent_takeover_expands_placeholder_and_refreshes_per_regen() -> None:
    """FG-domain Goal bind: placeholder → ordinal; re-resolve each call."""
    import os
    from types import SimpleNamespace

    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection
    from tests.test_libero_g17_executor_attachment import _spatial_problem

    os.environ["CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_TARGET_GRASP_OBJECT", None)

    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    native_np = projection.native_between_goal_referent(binding)
    native_goal = f"Pick the {native_np} and place it on the plate"
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(task_language=native_goal)
        )
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Goal: {libero_environment_goal}\n"
                        f"Note: {native_np} remains the starting referent."
                    ),
                }
            ],
        }
    ]

    att_a = adapter.LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    out_a, meta_a = takeover._apply_goal_referent_takeover(
        prompt, binding=binding, attachment=att_a, env=env
    )
    assert meta_a["applicable"] and meta_a["applied"]
    assert meta_a["placeholder_hits"] >= 1
    assert meta_a["native_goal_language"] == native_goal
    text_a = out_a[0]["content"][0]["text"]
    assert "{libero_environment_goal}" not in text_a
    assert native_np not in text_a
    assert "first akita black bowl from the robot's right" in text_a
    assert "sample_grasp_pose" not in text_a

    att_b = adapter.LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "POSITIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    out_b, meta_b = takeover._apply_goal_referent_takeover(
        prompt, binding=binding, attachment=att_b, env=env
    )
    assert meta_b["applicable"] and meta_b["applied"]
    assert meta_a["ordinal"] != meta_b["ordinal"]
    text_b = out_b[0]["content"][0]["text"]
    assert meta_b["ordinal"] in text_b
    assert meta_a["ordinal"] not in text_b
    assert "goal-identity-lock" in takeover.schema_version()



def test_goal_identity_lock_maps_sgp_and_goal_line_not_vdm_body() -> None:
    """Lock identity carriers only; leave wrong-grasp class wording intact."""
    import os
    from types import SimpleNamespace

    from capx.verification import libero_g17_adapter as adapter
    from capx.verification import libero_g17_fg_projection as projection
    from tests.test_libero_g17_executor_attachment import _spatial_problem

    os.environ["CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER"] = "1"
    os.environ.pop("CAPX_LIBERO_G17_TARGET_GRASP_OBJECT", None)

    binding = adapter.bind_libero_g17_goal(_spatial_problem())
    native_np = projection.native_between_goal_referent(binding)
    native_goal = f"Pick the {native_np} and place it on the plate"
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(
            handle=SimpleNamespace(task_language=native_goal)
        )
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Goal: Pick the akita black bowl and place it on the plate\n"
                        "Verified wrong-grasp same-name acquire:\n"
                        "- fact: grasping the akita black bowl (held_non_goal).\n"
                        'pos, quat = sample_grasp_pose("akita_black_bowl")\n'
                        f"Legacy: {native_np}\n"
                    ),
                }
            ],
        }
    ]
    att = adapter.LiberoAttachmentFact(
        "empty",
        (),
        same_name_other_raw="akita_black_bowl_2",
        goal_vs_same_name_other_rel_xyz=(
            "WITHIN_REFERENCE_BAND",
            "NEGATIVE_OF_REFERENCE_BAND",
            "WITHIN_REFERENCE_BAND",
        ),
    )
    out, meta = takeover._apply_goal_referent_takeover(
        prompt, binding=binding, attachment=att, env=env
    )
    assert meta["applicable"] and meta["applied"]
    assert meta["locked_identity"] == (
        "first akita black bowl from the robot's right"
    )
    text = out[0]["content"][0]["text"]
    locked = meta["locked_identity"]
    assert f"Goal: Pick {locked} and place it on the plate" in text or (
        f"Goal: Pick the {locked} and place it on the plate" in text
    )
    assert text.splitlines()[0].startswith("Goal:") and locked in text.splitlines()[0]
    assert "akita black bowl" not in text.splitlines()[0].replace(locked, "")
    assert f'sample_grasp_pose("{locked}")' in text
    assert native_np not in text
    # VDM wrong-grasp body keeps class wording (not globally rewritten).
    assert "grasping the akita black bowl (held_non_goal)" in text
    assert meta["sgp_arg_hits"] >= 1
    assert meta["goal_line_hits"] >= 1 or meta["between_hits"] >= 1



def test_verify_pending_acquire_code_and_rewrite() -> None:
    import os
    from capx.verification import libero_g17_takeover as takeover

    os.environ["CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER"] = "1"
    os.environ["CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY"] = "1"
    assert takeover.goal_identity_verify_enabled()
    assert "goal-identity-verify" in takeover.schema_version()

    out_dir = "/tmp/g17_identity_verify_unit"
    trial = 0
    takeover._PRIVATE_IDENTITY_LOCK.clear()
    takeover.remember_identity_lock(
        out_dir,
        trial,
        locked_identity="first akita black bowl from the robot's right",
        turn=1,
    )
    code = 'pos, quat = sample_grasp_pose("akita_black_bowl")\n'
    v = takeover.verify_pending_acquire_code(code, output_dir=out_dir, trial=trial)
    assert v["applicable"] and not v["ok"]
    assert v["mismatches"] == ["akita_black_bowl"]
    report = takeover.render_identity_verify_report(v)
    assert "locked goal identity" in report
    assert "akita_black_bowl" in report

    fixed, n = takeover.rewrite_sgp_args_to_locked_identity(
        code, v["locked_identity"]
    )
    assert n == 1
    assert "first akita black bowl from the robot's right" in fixed
    v2 = takeover.verify_pending_acquire_code(fixed, output_dir=out_dir, trial=trial)
    assert v2["applicable"] and v2["ok"]
