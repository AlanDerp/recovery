"""LIBERO G17 FG takeover — replaces the M3 visual-difference report once.

Isolated from Ideal-TK / cube takeover. Enabled only by CAPX_LIBERO_G17_FG=1.
Does not modify the frozen M3 baseline launch path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from capx.verification import closed_gt
from capx.verification import libero_g17_adapter as adapter
from capx.verification import libero_g17_fg_projection as projection
from capx.verification import libero_g17_xy_lock as xy_lock

def _schema_version() -> str:
    if os.environ.get("CAPX_LIBERO_G17_V354_FG_NATIVE_RETRY", "0") == "1":
        return "capx-libero-g17-fg-takeover-v3.54-native-retry-dual-projection"
    # Frozen fair-K winner (20260815): public On/In IFF only; no XY_LOCK /
    # Goal rewrite / v213–v223 report extensions.
    if os.environ.get("CAPX_LIBERO_G17_V24_PIN", "0") == "1":
        return "capx-libero-g17-fg-takeover-v2.4-public-in-iff"
    if os.environ.get("CAPX_LIBERO_G17_WORKGAP_ME", "0") == "1":
        return "capx-libero-g17-fg-takeover-v2.5-workgap-me"
    if os.environ.get("CAPX_LIBERO_G17_GOAL_REFERENT_TAKEOVER", "0") == "1":
        if os.environ.get("CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY", "0") == "1":
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.28-sparse-goal-identity-verify"
                )
            return "capx-libero-g17-fg-takeover-v2.28-goal-identity-verify"
        if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
            return (
                "capx-libero-g17-fg-takeover-v2.27-sparse-goal-identity-lock"
            )
        return "capx-libero-g17-fg-takeover-v2.27-goal-identity-lock"
    if os.environ.get("CAPX_LIBERO_G17_TARGET_GRASP_OBJECT", "0") == "1":
        if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
            return (
                "capx-libero-g17-fg-takeover-v2.24-sparse-target-grasp-object"
            )
        return "capx-libero-g17-fg-takeover-v2.24-target-grasp-object"
    # v223 wrong-grasp ordinal + silent SGP overload (no public which=).
    # v2.36: keep post-wrong-grasp ordinal; suppress empty-hand XY compare.
    # v2.37: ablation — SGP overload only; no same-name ordinal/XY acquire block.
    # v2.38/v2.39: + band obligations / mini predicate contract.
    # v2.40: task_gap aligned with acquire / endpoint_gap (On matches In).
    # v2.41: + post-miss clear-cache (drop thrash history; between→ordinal lock).
    # v2.42: + ReleaseReady XY nudge at open_gripper (no LLM / no lift/regrasp).
    # v2.43: ReleaseReady full-conjunct + EEF XY lock-Z (broken: panda_hand≠eef).
    # v2.44: full-conjunct + tip goto_pose nudge (still Z-drift via TCP mismatch).
    # v2.45: full-conjunct + panda_hand FK/IK XY-only (v247).
    # v2.46: full-conjunct + student tip↔goto_pose XY-only (v248).
    # v2.47: + CAPX_LIBERO_G17_GOTO_TIP_IDENTITY (obs TCP inverse in goto_pose).
    # v2.48: near-miss start/step oscillator on tip XY-only (v250).
    # v2.49: v246-style small-step ReleaseReady + tip XY-only (v251).
    # v2.50: near-miss-only tip_xy + fail-open (v252; never refuse open).
    # v2.51: fair place servo — get_object_pose + tip_xy (v253; no GT residual).
    # v2.52: RR tipxy band-step 3cm + slide window without contact refuse (v254).
    # v2.53: v254 + student get_object_pose residual (fair residual ablation; v255).
    # v2.54: near-miss osc 15−3t + tipxy window; sign-only residual (v256).
    # v2.55: v256 + student get_object_pose residual (fair osc ablation; v257).
    # v2.56: XY-decoupled 15−3t; student sign ablation (v258).
    # v2.57: XY-decoupled 15−3t; FG c2c sign; predicate within (v259 method).
    # v2.58: v256 wash — shared-t 15−3t; FG RelXY band sign+skip (v260).
    # v2.59: v260 + 圆中方 (/√2) + no Z-abort (v261).
    # v2.60: v261 + no-Z differential guards (tipxy budget / terminated refuse) (v262).
    # v2.61: v262 + v256 GraspNet freeze/replay (v263 fair wash check).
    # v2.62: v260 + v256 GraspNet freeze/replay overflow=repeat_last (v264 fair→9).
    # v2.63: v264 + ReleaseReady hold-guard (rolled back; over-strict).
    # v2.64: v264 + silent-allow close (soft-held promote; no tipxy contact gate).
    # v2.65: v266 + near-miss gate with plate-contact (over-strict; 5/10).
    # v2.66: v266 + XY-only near-miss gate (dist<=outer; no contact conjunct).
    # v2.74: v272 + 3cm/sqrt2 axis completion + acceptance-gated shared-t.
    # v2.75: v274 + live-seeded, preflighted transactional tip XY primitive.
    # v2.76: hard-local IK + absolute remaining-XY completion semantics.
    if os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") == "1":
        ordinal_on = os.environ.get("CAPX_LIBERO_G17_WRONG_GRASP_ORDINAL", "1") != "0"
        obl_on = os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS", "0") == "1"
        clear_on = os.environ.get("CAPX_LIBERO_G17_POST_MISS_CLEAR_CACHE", "0") == "1"
        rr_on = os.environ.get("CAPX_LIBERO_G17_RELEASE_READY", "0") == "1"
        rr_goto = os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_GOTO", "0") == "1"
        tip_id = os.environ.get("CAPX_LIBERO_G17_GOTO_TIP_IDENTITY", "0") == "1"
        osc_on = os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC", "0") == "1"
        tipxy_on = os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY", "0") == "1"
        nm_fo = os.environ.get("CAPX_LIBERO_G17_NEARMISS_FAILOPEN", "0") == "1"
        fair_ps = os.environ.get("CAPX_LIBERO_G17_FAIR_PLACE_SERVO", "0") == "1"
        tipxy_band = os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY_BANDSTEP", "0") == "1"
        tipxy_fair_r = (
            os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_TIPXY_FAIR_RESIDUAL", "0")
            == "1"
        )
        osc_bandwindow = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_BANDWINDOW", "0") == "1"
        )
        osc_fair_r = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_FAIR_RESIDUAL", "0") == "1"
        )
        osc_xy_decouple = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_XY_DECOUPLE", "0") == "1"
        )
        osc_student_sign = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_STUDENT_SIGN", "0") == "1"
        )
        osc_fg_wash = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_FG_PREDICATE_WASH", "0")
            == "1"
        )
        osc_v269_mech = (
            os.environ.get("CAPX_LIBERO_G17_V269_MECH", "0") == "1"
        )
        v274_mech_contract = (
            os.environ.get("CAPX_LIBERO_G17_V274_MECH_CONTRACT", "0") == "1"
        )
        v275_trusted_tipxy = (
            os.environ.get("CAPX_LIBERO_G17_V275_TRUSTED_TIPXY", "0") == "1"
        )
        v276_completion_ik = (
            os.environ.get("CAPX_LIBERO_G17_V276_COMPLETION_IK", "0") == "1"
        )
        v277_canonical_grasp = (
            os.environ.get("CAPX_LIBERO_G17_V277_CANONICAL_GRASP", "0") == "1"
        )
        v278_trusted_se3 = (
            os.environ.get("CAPX_LIBERO_G17_V278_TRUSTED_SE3", "0") == "1"
        )
        v279_reachable_se3 = (
            os.environ.get("CAPX_LIBERO_G17_V279_REACHABLE_SE3", "0") == "1"
        )
        v280_planned_se3 = (
            os.environ.get("CAPX_LIBERO_G17_V280_PLANNED_SE3", "0") == "1"
        )
        v281_joint_join = (
            os.environ.get("CAPX_LIBERO_G17_V281_JOINT_JOIN", "0") == "1"
        )
        v282_measured_seed_ik = (
            os.environ.get("CAPX_LIBERO_G17_V282_MEASURED_SEED_IK", "0") == "1"
        )
        v283_tool_pi_symmetry = (
            os.environ.get("CAPX_LIBERO_G17_V283_TOOL_PI_SYMMETRY", "0") == "1"
        )
        v284_release_commit_latch = (
            os.environ.get("CAPX_LIBERO_G17_V284_RELEASE_COMMIT_LATCH", "0") == "1"
        )
        v285_full_release_window = (
            os.environ.get("CAPX_LIBERO_G17_V285_FULL_RELEASE_WINDOW", "0") == "1"
        )
        v286_final_fg_check = (
            os.environ.get("CAPX_LIBERO_G17_V286_FINAL_FG_CHECK", "0") == "1"
        )
        v270_live_isolation = (
            os.environ.get("CAPX_LIBERO_G17_V270_LIVE_ISOLATION", "0") == "1"
        )
        v272_fair_live = (
            os.environ.get("CAPX_LIBERO_G17_V272_FAIR_LIVE", "0") == "1"
        )
        v271_pt_pair = (
            os.environ.get("CAPX_LIBERO_G17_V271_PT_PAIR", "0") == "1"
        )
        osc_inscribed = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_INSCRIBED_SQUARE", "0")
            == "1"
        )
        osc_no_z = (
            os.environ.get("CAPX_LIBERO_G17_NEARMISS_OSC_NO_Z_ABORT", "0") == "1"
        )
        grasp_replay = bool(
            os.environ.get("CAPX_LIBERO_G17_GRASP_REPLAY", "").strip()
        )
        hold_guard = (
            os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_HOLD_GUARD", "0") == "1"
        )
        silent_allow_close = (
            os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_NO_SILENT_ALLOW", "0")
            == "1"
        )
        nearmiss_gate = (
            os.environ.get("CAPX_LIBERO_G17_RELEASE_READY_NEARMISS_GATE", "0")
            == "1"
        )
        if not ordinal_on:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.37-sparse-sgp-overload-no-ordinal"
                )
            return "capx-libero-g17-fg-takeover-v2.37-sgp-overload-no-ordinal"
        if obl_on:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.39-sparse-ordinal-sgp-band-contract"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.39-ordinal-sgp-band-contract"
            )
        if clear_on:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.41-sparse-ordinal-sgp-post-miss-clear"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.41-ordinal-sgp-post-miss-clear"
            )
        if fair_ps:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.51-sparse-ordinal-sgp-fair-place-servo-tipxy"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.51-ordinal-sgp-fair-place-servo-tipxy"
            )
        if rr_on and tipxy_on and tipxy_band:
            if tipxy_fair_r:
                if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                    return (
                        "capx-libero-g17-fg-takeover-v2.53-sparse-ordinal-sgp-"
                        "release-ready-tipxy-bandstep-fair-residual"
                    )
                return (
                    "capx-libero-g17-fg-takeover-v2.53-ordinal-sgp-"
                    "release-ready-tipxy-bandstep-fair-residual"
                )
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.52-sparse-ordinal-sgp-release-ready-tipxy-bandstep"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.52-ordinal-sgp-release-ready-tipxy-bandstep"
            )
        if nm_fo:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.50-sparse-ordinal-sgp-nearmiss-failopen-tipxy"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.50-ordinal-sgp-nearmiss-failopen-tipxy"
            )
        if rr_on and tipxy_on:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.49-sparse-ordinal-sgp-release-ready-tipxy"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.49-ordinal-sgp-release-ready-tipxy"
            )
        if rr_on and osc_on:
            if osc_xy_decouple and osc_bandwindow:
                if osc_student_sign:
                    if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                        return (
                            "capx-libero-g17-fg-takeover-v2.56-sparse-ordinal-sgp-"
                            "nearmiss-osc-xy-decouple-student-sign"
                        )
                    return (
                        "capx-libero-g17-fg-takeover-v2.56-ordinal-sgp-"
                        "nearmiss-osc-xy-decouple-student-sign"
                    )
                if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                    return (
                        "capx-libero-g17-fg-takeover-v2.57-sparse-ordinal-sgp-"
                        "nearmiss-osc-xy-decouple-fg-sign"
                    )
                return (
                    "capx-libero-g17-fg-takeover-v2.57-ordinal-sgp-"
                    "nearmiss-osc-xy-decouple-fg-sign"
                )
            if osc_bandwindow:
                if v271_pt_pair:
                    source = os.environ.get(
                        "CAPX_LIBERO_G17_V271_SYMBOL_SOURCE", ""
                    ).strip().lower()
                    if source not in {"fg", "pt"}:
                        source = "invalid"
                    return (
                        "capx-libero-g17-fg-takeover-v2.71-ordinal-sgp-"
                        "nearmiss-osc-tipxy-bandwindow-v256-shared-mech-"
                        f"{source}-source-pt-paired"
                    )
                if osc_v269_mech:
                    source = os.environ.get(
                        "CAPX_LIBERO_G17_V269_SYMBOL_SOURCE", "fg"
                    ).strip().lower()
                    if source not in {"gt", "fg"}:
                        source = "invalid"
                    if (
                        v283_tool_pi_symmetry
                        and v285_full_release_window
                        and v286_final_fg_check
                    ):
                        version = "v2.86"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "measured-seed-tool-pi-native-once-full-release-window-"
                            "final-fg-axis-check"
                        )
                    elif v283_tool_pi_symmetry and v285_full_release_window:
                        version = "v2.85"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "measured-seed-tool-pi-native-once-full-release-window"
                        )
                    elif v283_tool_pi_symmetry and v284_release_commit_latch:
                        version = "v2.84"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "measured-seed-tool-pi-native-once-release-commit-latch"
                        )
                    elif v283_tool_pi_symmetry:
                        version = "v2.83"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "measured-seed-tool-pi-native-once"
                        )
                    elif v282_measured_seed_ik:
                        version = "v2.82"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "measured-seed-native-once"
                        )
                    elif v281_joint_join:
                        version = "v2.81"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "command-faithful-joint-join"
                        )
                    elif v280_planned_se3:
                        version = "v2.80"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "command-faithful-planned-se3"
                        )
                    elif v279_reachable_se3:
                        version = "v2.79"
                        suffix = (
                            "-fair-live-inscribed-completion-local-ik-"
                            "trusted-se3-fullpath-neutral-recovery"
                        )
                    elif v278_trusted_se3:
                        version = "v2.78"
                        suffix = "-fair-live-inscribed-completion-local-ik-trusted-se3"
                    elif v277_canonical_grasp:
                        version = "v2.77"
                        suffix = "-fair-live-single-seed-canonical-world-z-grasp"
                    elif v276_completion_ik:
                        version = "v2.76"
                        suffix = "-fair-live-inscribed-completion-local-ik"
                    elif v275_trusted_tipxy:
                        version = "v2.75"
                        suffix = "-fair-live-inscribed-transactional-tipxy"
                    elif v274_mech_contract:
                        version = "v2.74"
                        suffix = "-fair-live-inscribed-acceptance-gated"
                    elif v272_fair_live:
                        version = "v2.72"
                        suffix = "-fair-live-no-replay"
                    else:
                        version = "v2.70" if v270_live_isolation else "v2.69"
                        suffix = "-live-isolation" if v270_live_isolation else ""
                    return (
                        f"capx-libero-g17-fg-takeover-{version}-ordinal-sgp-"
                        "nearmiss-osc-tipxy-bandwindow-v256-shared-mech-"
                        f"{source}-source{suffix}"
                    )
                if osc_fg_wash and not osc_xy_decouple:
                    if osc_inscribed and osc_no_z:
                        if grasp_replay:
                            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                                return (
                                    "capx-libero-g17-fg-takeover-v2.61-sparse-ordinal-sgp-"
                                    "nearmiss-osc-tipxy-bandwindow-fg-wash-inscribed-no-z-"
                                    "grasp-replay"
                                )
                            return (
                                "capx-libero-g17-fg-takeover-v2.61-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-wash-inscribed-no-z-"
                                "grasp-replay"
                            )
                        if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                            return (
                                "capx-libero-g17-fg-takeover-v2.60-sparse-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-wash-inscribed-no-z-guard"
                            )
                        return (
                            "capx-libero-g17-fg-takeover-v2.60-ordinal-sgp-"
                            "nearmiss-osc-tipxy-bandwindow-fg-wash-inscribed-no-z-guard"
                        )
                    if grasp_replay:
                        if nearmiss_gate and silent_allow_close:
                            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                                return (
                                    "capx-libero-g17-fg-takeover-v2.66-sparse-ordinal-sgp-"
                                    "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                    "grasp-replay-silent-allow-close-nearmiss-gate-xy"
                                )
                            return (
                                "capx-libero-g17-fg-takeover-v2.66-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                "grasp-replay-silent-allow-close-nearmiss-gate-xy"
                            )
                        if silent_allow_close:
                            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                                return (
                                    "capx-libero-g17-fg-takeover-v2.64-sparse-ordinal-sgp-"
                                    "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                    "grasp-replay-silent-allow-close"
                                )
                            return (
                                "capx-libero-g17-fg-takeover-v2.64-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                "grasp-replay-silent-allow-close"
                            )
                        if hold_guard:
                            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                                return (
                                    "capx-libero-g17-fg-takeover-v2.63-sparse-ordinal-sgp-"
                                    "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                    "grasp-replay-hold-guard"
                                )
                            return (
                                "capx-libero-g17-fg-takeover-v2.63-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                "grasp-replay-hold-guard"
                            )
                        if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                            return (
                                "capx-libero-g17-fg-takeover-v2.62-sparse-ordinal-sgp-"
                                "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                                "grasp-replay"
                            )
                        return (
                            "capx-libero-g17-fg-takeover-v2.62-ordinal-sgp-"
                            "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash-"
                            "grasp-replay"
                        )
                    if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                        return (
                            "capx-libero-g17-fg-takeover-v2.58-sparse-ordinal-sgp-"
                            "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash"
                        )
                    return (
                        "capx-libero-g17-fg-takeover-v2.58-ordinal-sgp-"
                        "nearmiss-osc-tipxy-bandwindow-fg-predicate-wash"
                    )
                if osc_fair_r:
                    if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                        return (
                            "capx-libero-g17-fg-takeover-v2.55-sparse-ordinal-sgp-"
                            "nearmiss-osc-tipxy-bandwindow-fair-residual"
                        )
                    return (
                        "capx-libero-g17-fg-takeover-v2.55-ordinal-sgp-"
                        "nearmiss-osc-tipxy-bandwindow-fair-residual"
                    )
                if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                    return (
                        "capx-libero-g17-fg-takeover-v2.54-sparse-ordinal-sgp-"
                        "nearmiss-osc-tipxy-bandwindow"
                    )
                return (
                    "capx-libero-g17-fg-takeover-v2.54-ordinal-sgp-"
                    "nearmiss-osc-tipxy-bandwindow"
                )
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.48-sparse-ordinal-sgp-nearmiss-osc-tipxy"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.48-ordinal-sgp-nearmiss-osc-tipxy"
            )
        if rr_on:
            if rr_goto and tip_id:
                if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                    return (
                        "capx-libero-g17-fg-takeover-v2.47-sparse-ordinal-sgp-release-ready-goto-tip-identity"
                    )
                return (
                    "capx-libero-g17-fg-takeover-v2.47-ordinal-sgp-release-ready-goto-tip-identity"
                )
            if rr_goto:
                if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                    return (
                        "capx-libero-g17-fg-takeover-v2.46-sparse-ordinal-sgp-release-ready-goto"
                    )
                return (
                    "capx-libero-g17-fg-takeover-v2.46-ordinal-sgp-release-ready-goto"
                )
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.45-sparse-ordinal-sgp-release-ready-xyhand"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.45-ordinal-sgp-release-ready-xyhand"
            )
        if tip_id:
            if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
                return (
                    "capx-libero-g17-fg-takeover-v2.47-sparse-ordinal-sgp-goto-tip-identity"
                )
            return (
                "capx-libero-g17-fg-takeover-v2.47-ordinal-sgp-goto-tip-identity"
            )
        if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
            return (
                "capx-libero-g17-fg-takeover-v2.40-sparse-ordinal-sgp-task-gap-aligned"
            )
        return "capx-libero-g17-fg-takeover-v2.40-ordinal-sgp-task-gap-aligned"
    if os.environ.get("CAPX_LIBERO_G17_L1_SPARSE", "0") == "1":
        return "capx-libero-g17-fg-takeover-v2.29-sparse-same-name-xy-logic"
    return "capx-libero-g17-fg-takeover-v2.29-same-name-xy-logic"


def schema_version() -> str:
    """Resolve schema from env at call time (not import time)."""
    return _schema_version()


# Back-compat alias; prefer schema_version() so env flips in tests apply.
SCHEMA_VERSION = _schema_version()
TRACE_DIR = "libero_g17_fg_traces"

_PRIVATE_SNAPSHOTS: dict[tuple[str, int], list[closed_gt.CompiledSnapshot]] = {}
_PRIVATE_BINDINGS: dict[tuple[str, int], adapter.LiberoG17Binding] = {}
_PRIVATE_ATTACHMENTS: dict[tuple[str, int], list[adapter.LiberoAttachmentFact]] = {}
_PRIVATE_IDENTITY_LOCK: dict[tuple[str, int], dict[str, Any]] = {}
_INTERVAL_PEAK: dict[tuple[str, int], adapter.LiberoAttachmentFact] = {}
_INTERVAL_ACTIVE_KEY: tuple[str, int] | None = None
_INTERVAL_SAMPLE_EVERY = 5
# (output_dir, trial, turn) → clear-cache requested for the decision prompt.
_POST_MISS_CLEAR_FLAGS: dict[tuple[str, int, int], bool] = {}
_LAST_TAKEOVER_BUNDLE: dict[tuple[str, int, int], dict[str, Any]] = {}

_FORBIDDEN = (
    "reward",
    "task completed",
    "task success",
    "next move",
    "next action",
    "bddl",
    "obj01",
    "obj02",
    "exe01",
)
_NUMERIC = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")


def _cache_key(output_dir: str, trial: int) -> tuple[str, int]:
    return (str(Path(output_dir).resolve()), int(trial))


def consume_post_miss_clear_flag(
    output_dir: str, trial: int, turn: int
) -> bool:
    """Pop whether this takeover requested a post-miss clear-cache."""
    key = (str(Path(output_dir).resolve()), int(trial), int(turn))
    return bool(_POST_MISS_CLEAR_FLAGS.pop(key, False))


def consume_takeover_bundle(
    output_dir: str, trial: int, turn: int
) -> dict[str, Any] | None:
    """Pop takeover side-channel meta (clear-cache + replacement text)."""
    key = (str(Path(output_dir).resolve()), int(trial), int(turn))
    return _LAST_TAKEOVER_BUNDLE.pop(key, None)


def begin_turn_interval_sampling(output_dir: str, trial: int) -> None:
    """Reset interval peak latch at the start of a code-block turn."""
    global _INTERVAL_ACTIVE_KEY
    if os.environ.get("CAPX_LIBERO_G17_FG", "0") != "1":
        return
    key = _cache_key(output_dir, trial)
    if key not in _PRIVATE_BINDINGS:
        return
    _INTERVAL_ACTIVE_KEY = key
    _INTERVAL_PEAK[key] = adapter.LiberoAttachmentFact("empty", ())


def sample_interval_attachment(env: Any, *, force: bool = False) -> None:
    """Sample executor attachment into the active turn peak (cheap, throttled)."""
    global _INTERVAL_ACTIVE_KEY
    if os.environ.get("CAPX_LIBERO_G17_FG", "0") != "1":
        return
    key = _INTERVAL_ACTIVE_KEY
    if key is None:
        return
    binding = _PRIVATE_BINDINGS.get(key)
    if binding is None:
        return
    if not force:
        # Throttle via caller-provided sim step count when available.
        step_count = getattr(env, "_sim_step_count", None)
        if isinstance(step_count, int) and step_count % _INTERVAL_SAMPLE_EVERY != 0:
            return
    try:
        from dataclasses import replace as dc_replace

        domain = adapter._libero_domain(env)
        gripper = domain.robots[0].gripper
        sample = adapter.classify_executor_attachment(domain, gripper, binding)
        if sample.kind != "empty" and sample.held_movable_names:
            held_rel, vs_goal = adapter.compute_held_interaction_bands(
                domain,
                binding,
                sample.held_movable_names,
                wrapper=adapter._capx_wrapper(env),
            )
            sample = dc_replace(
                sample,
                held_rel_xyz=held_rel,
                held_vs_goal_rel_xyz=vs_goal,
            )
    except Exception as exc:  # noqa: BLE001 — never crash robot control on FG sample
        print(f"[libero-g17-fg] interval sample skipped: {exc}", flush=True)
        return
    peak = _INTERVAL_PEAK.get(key)
    _INTERVAL_PEAK[key] = adapter.merge_attachment_peak(peak, sample)


def _consume_interval_peak(key: tuple[str, int]) -> adapter.LiberoAttachmentFact:
    global _INTERVAL_ACTIVE_KEY
    peak = _INTERVAL_PEAK.pop(key, adapter.LiberoAttachmentFact("empty", ()))
    if _INTERVAL_ACTIVE_KEY == key:
        _INTERVAL_ACTIVE_KEY = None
    return peak


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _suite_task_from_env(env: Any) -> tuple[str | None, int | None]:
    low = getattr(env, "low_level_env", None) or env
    handle = getattr(low, "handle", None)
    suite = getattr(handle, "suite_name", None) or getattr(low, "suite_name", None)
    task_id = getattr(handle, "task_id", None)
    if task_id is None:
        task_id = getattr(low, "task_id", None)
    return (
        str(suite) if suite is not None else None,
        int(task_id) if task_id is not None else None,
    )


def _replace_report_once(
    prompt: list[dict[str, Any]], *, original: str, replacement: str
) -> list[dict[str, Any]]:
    if not original.strip() or original == replacement:
        raise ValueError("LIBERO G17 FG takeover requires a distinct non-empty report")
    updated = copy.deepcopy(prompt)
    matches = 0
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            count = content.count(original)
            if count:
                message["content"] = content.replace(original, replacement)
                matches += count
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                count = text.count(original)
                if count:
                    item["text"] = text.replace(original, replacement)
                    matches += count
    if matches != 1:
        raise ValueError(
            f"LIBERO G17 FG takeover expected one native report carrier, found {matches}"
        )
    return updated


def _non_text_structure(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structure: list[dict[str, Any]] = []
    for message in prompt:
        content = message.get("content")
        item = {
            "role": message.get("role"),
            "extra": sorted(set(message) - {"role", "content"}),
        }
        if isinstance(content, str):
            item["content_types"] = ["text"]
        elif isinstance(content, list):
            item["content_types"] = [
                value.get("type") if isinstance(value, dict) else type(value).__name__
                for value in content
            ]
            item["non_text"] = [
                value
                for value in content
                if not isinstance(value, dict) or value.get("type") != "text"
            ]
        else:
            item["content_types"] = [type(content).__name__]
        structure.append(item)
    return structure


def initialize_trial_trace(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    task: str,
) -> None:
    suite, task_id = _suite_task_from_env(env)
    binding = adapter.bind_libero_g17_env(env, suite_name=suite, task_id=task_id)
    binding, initial, attachment0 = adapter.extract_libero_snapshot(env, binding)
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    root.mkdir(parents=True, exist_ok=False)
    key = _cache_key(output_dir, trial)
    if key in _PRIVATE_SNAPSHOTS:
        raise RuntimeError("LIBERO G17 FG private trial cache already exists")
    _PRIVATE_SNAPSHOTS[key] = [initial]
    _PRIVATE_BINDINGS[key] = binding
    _PRIVATE_ATTACHMENTS[key] = [attachment0]
    xy_lock.publish_xy_lock(binding, attachment0, reset=True)
    _PRIVATE_IDENTITY_LOCK.pop(key, None)
    _INTERVAL_PEAK.pop(key, None)
    from capx.integrations.franka import fg_native_retry as native_retry

    if native_retry.enabled():
        native_retry.reset_trial(trial)
    engagement = {
        "schema_version": SCHEMA_VERSION,
        "fg_engaged": True,
        "task": task,
        "trial": trial,
        "suite_name": binding.suite_name,
        "task_id": binding.task_id,
        "relation": binding.relation,
        "takeover_module": __file__,
        "adapter_module": adapter.__file__,
    }
    _save_json(root / "engagement.json", engagement)
    _save_json(
        root / "state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "closed_gt_schema_version": closed_gt.SCHEMA_VERSION,
            "task": task,
            "trial": trial,
            "suite_name": binding.suite_name,
            "task_id": binding.task_id,
            "relation": binding.relation,
            "payload_public": binding.scene.objects[0].public_name,
            "receptacle_public": binding.scene.objects[1].public_name,
            "turns_completed": 0,
            "initial_public_state_sha256": _sha256_json(initial.values),
            "action_visible_initial_gt": False,
            "fg_engaged": True,
        },
    )
    print(
        f"[libero-g17-fg] engaged trial={trial} suite={binding.suite_name} "
        f"task_id={binding.task_id} relation={binding.relation}",
        flush=True,
    )


def _replace_text_in_prompt(
    prompt: list[dict[str, Any]], *, old: str, new: str
) -> tuple[list[dict[str, Any]], int]:
    """Replace ``old`` with ``new`` in all text parts; return (prompt, n_hits)."""
    if not old or old == new:
        return prompt, 0
    updated = copy.deepcopy(prompt)
    hits = 0
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            count = content.count(old)
            if count:
                message["content"] = content.replace(old, new)
                hits += count
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                count = text.count(old)
                if count:
                    item["text"] = text.replace(old, new)
                    hits += count
    return updated, hits


def _native_task_language(env: Any) -> str | None:
    """Read LIBERO Goal language the same way host ``_patch_libero_goal`` does.

    Takeover receives the code-exec env wrapper; task language lives on
    ``env.low_level_env.handle`` (not ``env.handle`` / ``env.low``).
    """
    low = getattr(env, "low_level_env", None) or getattr(env, "low", None) or env
    handle = getattr(low, "handle", None) or getattr(env, "handle", None)
    lang = getattr(handle, "task_language", None) if handle is not None else None
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return None


def _norm_identity_token(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


def _ambiguous_goal_identity_forms(binding: adapter.LiberoG17Binding) -> list[str]:
    """Ambiguous NL/snake forms of the goal class (not yet uniquely locked)."""
    from capx.verification.libero_g17_fg_lexicon import humanize_libero_name

    name = humanize_libero_name(binding.payload_name)
    snake = str(binding.payload_name).strip()
    # Longest first so "the akita black bowl" wins over "akita black bowl".
    forms = [f"the {name}", name]
    if snake and _norm_identity_token(snake) != _norm_identity_token(name):
        forms.append(snake)
    # De-dupe while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for form in forms:
        key = _norm_identity_token(form)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(form)
    return out


def _map_text_goal_lines(
    text: str, *, ambiguous: list[str], bound: str
) -> tuple[str, int]:
    """Replace goal identity only on Goal-carrier lines (not VDM body)."""
    if not ambiguous or not bound:
        return text, 0
    # Skip forms that are substrings of the lock string (avoids double rewrite).
    forms = [f for f in ambiguous if f and f not in bound]
    if not forms:
        return text, 0
    hits = 0
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("Goal:") or stripped.startswith("goal:"):
            if bound in line:
                lines.append(line)
                continue
            for form in forms:
                count = line.count(form)
                if count:
                    line = line.replace(form, bound)
                    hits += count
                    break
        lines.append(line)
    return "".join(lines), hits


def _map_text_sgp_args(text: str, *, ambiguous: list[str], bound: str) -> tuple[str, int]:
    """Replace ambiguous goal identity inside sample_grasp_pose quotes only."""
    if not ambiguous:
        return text, 0
    amb_norm = {_norm_identity_token(a) for a in ambiguous}
    hits = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal hits
        prefix, quote, arg = match.group(1), match.group(2), match.group(3)
        if _norm_identity_token(arg) in amb_norm:
            hits += 1
            return f"{prefix}{quote}{bound}{quote}"
        return match.group(0)

    updated = re.sub(
        r"(sample_grasp_pose\s*\(\s*)([\"'])([^\"']*)(\2)",
        repl,
        text,
    )
    return updated, hits


def _map_prompt_identity_carriers(
    prompt: list[dict[str, Any]],
    *,
    map_fn,
) -> tuple[list[dict[str, Any]], int]:
    """Apply a text mapper to all text parts of the decision prompt."""
    updated = copy.deepcopy(prompt)
    hits = 0
    for message in updated:
        content = message.get("content")
        if isinstance(content, str):
            new_text, n = map_fn(content)
            if n:
                message["content"] = new_text
                hits += n
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                new_text, n = map_fn(text)
                if n:
                    item["text"] = new_text
                    hits += n
    return updated, hits


def _apply_goal_referent_takeover(
    prompt: list[dict[str, Any]],
    *,
    binding: adapter.LiberoG17Binding,
    attachment: adapter.LiberoAttachmentFact,
    env: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-regen FG goal-identity lock: ambiguous carriers → unique NL.

    Purpose: eliminate same-name acquire ambiguity by locking the goal's NL
    identity for this regeneration only (refresh on the next regen).

    Carriers replaced inside the FG decision prompt (not API runtime):
    - ``{libero_environment_goal}`` / native Goal language
    - starting-state between-clause NP (primary native identity)
    - ``Goal:`` lines still using class-level goal name
    - ``sample_grasp_pose("…")`` args that are class-level goal identity

    VDM body class mentions are left intact so wrong-grasp held-object wording
    stays truthful. Unique string = geometry-resolved ordinal at regen start.
    """
    meta: dict[str, Any] = {
        "applicable": False,
        "applied": False,
        "ordinal": None,
        "native_between_referent": None,
        "placeholder_hits": 0,
        "between_hits": 0,
        "goal_line_hits": 0,
        "sgp_arg_hits": 0,
        "native_goal_language": None,
        "locked_identity": None,
        "ambiguous_forms": [],
    }
    ordinal = projection.resolve_same_name_goal_ordinal(binding, attachment)
    if ordinal is None:
        return prompt, meta
    native_np = projection.native_between_goal_referent(binding)
    ambiguous = _ambiguous_goal_identity_forms(binding)
    meta["applicable"] = True
    meta["ordinal"] = ordinal
    meta["locked_identity"] = ordinal
    meta["native_between_referent"] = native_np
    meta["ambiguous_forms"] = list(ambiguous)

    native_goal = _native_task_language(env)
    meta["native_goal_language"] = native_goal
    rewritten_goal = None
    if isinstance(native_goal, str) and native_goal.strip():
        rewritten_goal = native_goal
        if native_np in rewritten_goal:
            rewritten_goal = rewritten_goal.replace(native_np, ordinal)
        else:
            for form in ambiguous:
                if form in ordinal:
                    continue
                if form in rewritten_goal:
                    rewritten_goal = rewritten_goal.replace(form, ordinal)
                    break

    corrected = prompt
    if rewritten_goal is not None:
        corrected, n_ph = _replace_text_in_prompt(
            corrected,
            old="{libero_environment_goal}",
            new=rewritten_goal,
        )
        meta["placeholder_hits"] = n_ph

    corrected, n_bt = _replace_text_in_prompt(
        corrected, old=native_np, new=ordinal
    )
    meta["between_hits"] = n_bt

    corrected, goal_hits = _map_prompt_identity_carriers(
        corrected,
        map_fn=lambda t: _map_text_goal_lines(
            t, ambiguous=ambiguous, bound=ordinal
        ),
    )
    meta["goal_line_hits"] = goal_hits

    corrected, n_sgp = _map_prompt_identity_carriers(
        corrected,
        map_fn=lambda t: _map_text_sgp_args(
            t, ambiguous=ambiguous, bound=ordinal
        ),
    )
    meta["sgp_arg_hits"] = n_sgp

    meta["applied"] = bool(
        meta["placeholder_hits"]
        or meta["between_hits"]
        or meta["goal_line_hits"]
        or meta["sgp_arg_hits"]
    )
    return corrected, meta



def goal_identity_verify_enabled() -> bool:
    return (
        projection.goal_referent_takeover_enabled()
        and os.environ.get("CAPX_LIBERO_G17_GOAL_IDENTITY_VERIFY", "0") == "1"
    )


def _identity_equivalent(a: str, b: str) -> bool:
    return _norm_identity_token(a) == _norm_identity_token(b)


def _extract_sgp_args(code: str) -> list[str]:
    return [
        m.group(2)
        for m in re.finditer(
            r"sample_grasp_pose\s*\(\s*([\"'])(.*?)\1",
            code,
            flags=re.DOTALL,
        )
    ]


def remember_identity_lock(
    output_dir: str,
    trial: int,
    *,
    locked_identity: str,
    turn: int,
    ambiguous_forms: list[str] | None = None,
) -> None:
    key = _cache_key(output_dir, trial)
    _PRIVATE_IDENTITY_LOCK[key] = {
        "locked_identity": locked_identity,
        "turn": int(turn),
        "ambiguous_forms": list(ambiguous_forms or []),
    }


def get_identity_lock(output_dir: str, trial: int) -> dict[str, Any] | None:
    return _PRIVATE_IDENTITY_LOCK.get(_cache_key(output_dir, trial))


def verify_pending_acquire_code(
    code: str,
    *,
    output_dir: str,
    trial: int,
) -> dict[str, Any]:
    """Pre-exec check: pending SGP args must match this regen's locked identity."""
    meta: dict[str, Any] = {
        "enabled": goal_identity_verify_enabled(),
        "applicable": False,
        "ok": True,
        "locked_identity": None,
        "sgp_args": [],
        "mismatches": [],
    }
    if not meta["enabled"]:
        return meta
    lock = get_identity_lock(output_dir, trial)
    if not lock or not lock.get("locked_identity"):
        return meta
    locked = str(lock["locked_identity"])
    args = _extract_sgp_args(code)
    meta["locked_identity"] = locked
    meta["sgp_args"] = args
    if not args:
        return meta
    meta["applicable"] = True
    bad = [a for a in args if not _identity_equivalent(a, locked)]
    meta["mismatches"] = bad
    meta["ok"] = not bad
    return meta


def render_identity_verify_report(verdict: dict[str, Any]) -> str:
    locked = verdict.get("locked_identity")
    args = verdict.get("sgp_args") or []
    bad = verdict.get("mismatches") or []
    arg_txt = ", ".join(repr(a) for a in args) if args else "(none)"
    bad_txt = ", ".join(repr(a) for a in bad) if bad else "(none)"
    return (
        "Verified goal-acquire identity lock over the pending code:\n"
        f"- fact: locked goal identity: {locked}.\n"
        f"- fact: pending sample_grasp_pose argument(s): {arg_txt}.\n"
        f"- fact: mismatched acquire identity argument(s): {bad_txt}.\n"
        "- fact: pending acquire identity is not the locked goal identity.\n"
    )


def rewrite_sgp_args_to_locked_identity(
    code: str, locked_identity: str
) -> tuple[str, int]:
    """Last-resort: rewrite pending-code SGP identity carriers to the lock."""
    if not locked_identity:
        return code, 0
    hits = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal hits
        prefix, quote, arg = match.group(1), match.group(2), match.group(3)
        if _identity_equivalent(arg, locked_identity):
            return match.group(0)
        hits += 1
        return f"{prefix}{quote}{locked_identity}{quote}"

    updated = re.sub(
        r"(sample_grasp_pose\s*\(\s*)([\"'])([^\"']*)(\2)",
        repl,
        code,
    )
    return updated, hits


def apply_cached_goal_identity_lock(
    prompt: list[dict[str, Any]],
    *,
    output_dir: str,
    trial: int,
    env: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-apply goal identity lock on a verify-regen prompt without advancing turn."""
    key = _cache_key(output_dir, trial)
    binding = _PRIVATE_BINDINGS.get(key)
    attachments = _PRIVATE_ATTACHMENTS.get(key)
    if binding is None or not attachments:
        return prompt, {"applicable": False, "applied": False}
    attachment = attachments[-1]
    return _apply_goal_referent_takeover(
        prompt, binding=binding, attachment=attachment, env=env
    )



def run_takeover(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    native_report: str | None,
    native_decision_prompt: list[dict[str, Any]],
    private_task_completed: bool,
) -> list[dict[str, Any]]:
    if not isinstance(native_report, str) or not native_report.strip():
        raise ValueError("LIBERO G17 FG takeover requires a native M3 report")
    key = _cache_key(output_dir, trial)
    snapshots = _PRIVATE_SNAPSHOTS.get(key)
    binding = _PRIVATE_BINDINGS.get(key)
    if snapshots is None or binding is None or len(snapshots) != turn:
        raise ValueError("LIBERO G17 FG private state is not turn-synchronous")
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("trial") != trial
        or state.get("turns_completed") != turn - 1
    ):
        raise ValueError("LIBERO G17 FG trace provenance mismatch")

    before = snapshots[-1]
    attachments = _PRIVATE_ATTACHMENTS.get(key)
    if attachments is None or len(attachments) != turn:
        raise ValueError("LIBERO G17 FG attachment cache is not turn-synchronous")
    attachment_before = attachments[-1]
    sample_interval_attachment(env, force=True)
    peak = _consume_interval_peak(key)
    binding, after, attachment = adapter.extract_libero_snapshot(env, binding)
    attachment_peak = adapter.merge_attachment_peak(peak, attachment)
    attachment_peak = adapter.merge_attachment_peak(
        attachment_peak, attachment_before
    )
    _PRIVATE_BINDINGS[key] = binding
    proj = projection.project_libero_fg(
        binding,
        before,
        after,
        attachment=attachment,
        attachment_before=attachment_before,
        attachment_peak=attachment_peak,
    )
    from capx.verification import libero_g17_post_miss_clear as post_miss

    clear_cache = post_miss.should_clear_after_projection(proj)
    replacement = proj.combined_report
    native_retry_meta: dict[str, Any] | None = None
    from capx.integrations.franka import fg_native_retry as native_retry

    if native_retry.enabled():
        retry_projection = native_retry.consume_student_report(trial)
        if retry_projection is not None:
            retry_report, native_retry_meta = retry_projection
            replacement = f"{replacement.rstrip()}\n{retry_report}"
    if clear_cache:
        note = "\n".join(post_miss.context_reset_note_lines())
        replacement = f"{replacement.rstrip()}\n{note}\n"
        _POST_MISS_CLEAR_FLAGS[
            (str(Path(output_dir).resolve()), int(trial), int(turn))
        ] = True
    for term in _FORBIDDEN:
        if term in replacement.lower():
            raise ValueError(f"LIBERO G17 FG report contains prohibited term: {term}")
    if _NUMERIC.search(replacement):
        raise ValueError("LIBERO G17 FG report exposes a numeric value")

    dominate = projection.assert_fg_dominates_native_interaction(
        native_report=native_report,
        fg_report=replacement,
        attachment=attachment,
    )
    if dominate["applicable"] and not (
        dominate["correctness_ok"] and dominate["completeness_ok"]
    ):
        print(
            "[libero-g17-fg] WARN native-interaction-domination incomplete "
            f"correctness={int(dominate['correctness_ok'])} "
            f"completeness={int(dominate['completeness_ok'])}",
            flush=True,
        )
    place_dom = projection.assert_fg_dominates_native_placement(
        native_report=native_report,
        fg_report=replacement,
        binding=binding,
        after=after,
    )
    if place_dom["applicable"] and not (
        place_dom["correctness_ok"] and place_dom["completeness_ok"]
    ):
        print(
            "[libero-g17-fg] WARN native-placement-domination incomplete "
            f"correctness={int(place_dom['correctness_ok'])} "
            f"completeness={int(place_dom['completeness_ok'])}",
            flush=True,
        )

    corrected = _replace_report_once(
        native_decision_prompt,
        original=native_report,
        replacement=replacement,
    )
    goal_ref_meta: dict[str, Any] | None = None
    apply_goal_lock = (
        projection.goal_referent_takeover_enabled()
        or post_miss.goal_lock_between_to_ordinal_enabled()
    )
    if apply_goal_lock:
        corrected, goal_ref_meta = _apply_goal_referent_takeover(
            corrected,
            binding=binding,
            attachment=attachment,
            env=env,
        )
        if goal_ref_meta.get("locked_identity"):
            remember_identity_lock(
                output_dir,
                trial,
                locked_identity=str(goal_ref_meta["locked_identity"]),
                turn=turn,
                ambiguous_forms=list(goal_ref_meta.get("ambiguous_forms") or []),
            )
        if goal_ref_meta.get("applied"):
            print(
                "[libero-g17-fg] goal-identity-lock "
                f"locked={goal_ref_meta.get('locked_identity')!r} "
                f"placeholder_hits={goal_ref_meta.get('placeholder_hits')} "
                f"between_hits={goal_ref_meta.get('between_hits')} "
                f"goal_line_hits={goal_ref_meta.get('goal_line_hits')} "
                f"sgp_arg_hits={goal_ref_meta.get('sgp_arg_hits')}",
                flush=True,
            )
    if clear_cache:
        print(
            f"[libero-g17-fg] post-miss-clear turn={turn} trial={trial} "
            f"task_gap={proj.task_gap} released_goal="
            f"{int(proj.released_goal_after_transport)}",
            flush=True,
        )
    if _non_text_structure(native_decision_prompt) != _non_text_structure(corrected):
        raise RuntimeError("LIBERO G17 FG takeover changed request roles or media")
    if _sha256_json(native_decision_prompt) == _sha256_json(corrected):
        raise RuntimeError("LIBERO G17 FG takeover did not change the action request")

    audit = adapter.g17_versus_bddl_audit(after, env, binding)
    if audit["bddl_iff_task_completed"] is not True:
        raise RuntimeError("LIBERO G17 FG private BDDL IFF audit failed")
    # Completion label must stay private; only check consistency with the atom.
    if bool(private_task_completed) != bool(audit["private_task_completed"]):
        raise RuntimeError("LIBERO G17 FG task_completed disagrees with domain checker")

    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    (turn_root / "native_report.txt").write_text(native_report, encoding="utf-8")
    (turn_root / "takeover_report.txt").write_text(replacement, encoding="utf-8")
    (turn_root / "parent_report.txt").write_text(proj.parent_report, encoding="utf-8")
    (turn_root / "endpoint_report.txt").write_text(
        proj.endpoint_report, encoding="utf-8"
    )
    if proj.release_report:
        (turn_root / "release_report.txt").write_text(
            proj.release_report, encoding="utf-8"
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "carrier": "m3_visual_difference_report_libero_g17_fg",
        "gain_status": proj.gain.status,
        "bound_release": bool(proj.gain.bound_release),
        "release_package_applied": bool(proj.release_report),
        "world_fg_complete": bool(proj.world_fg_complete),
        "endpoint_gap": proj.endpoint_gap,
        "finish_authorized": bool(proj.finish_authorized),
        "executor_attachment": proj.executor_attachment,
        "interaction_grasp": proj.interaction_grasp,
        "executor_attachment_peak": proj.executor_attachment_peak,
        "interaction_grasp_peak": proj.interaction_grasp_peak,
        "task_gap": proj.task_gap,
        "released_goal_after_transport": bool(
            proj.released_goal_after_transport
        ),
        "native_interaction_domination": dominate,
        "native_placement_domination": place_dom,
        "goal_referent_takeover": goal_ref_meta,
        "fg_native_retry": native_retry_meta,
        "post_miss_clear": bool(clear_cache),
        "goal_evidence_key_count": len(proj.goal_evidence_keys),
        "acquire_evidence_key_count": len(
            projection.acquire_evidence_keys(binding)
        ),
        "place_evidence_key_count": len(projection.place_evidence_keys(binding)),
        "novel_fact_ids": list(proj.gain.novel_fact_ids),
        "novel_fact_count": len(proj.gain.novel_fact_ids),
        "parent_sha256": proj.parent_sha256,
        "combined_sha256": proj.combined_sha256,
        "parent_byte_identity_ok": projection.parent_byte_identity_preserved(proj),
        "native_report_sha256": _sha256_text(native_report),
        "takeover_report_sha256": _sha256_text(replacement),
        "native_prompt_sha256": _sha256_json(native_decision_prompt),
        "takeover_prompt_sha256": _sha256_json(corrected),
        "bddl_atom_true": audit["bddl_atom_true"],
        "private_task_completed": audit["private_task_completed"],
        "bddl_iff_task_completed": audit["bddl_iff_task_completed"],
        "geometric_proxy": audit["geometric_proxy"],
        "g17_iff_bddl": audit["g17_iff_bddl"],
        "action_visible_goal_label": False,
    }
    _save_json(turn_root / "record.json", record)
    snapshots.append(after)
    attachments.append(attachment)
    xy_lock.publish_xy_lock(binding, attachment)
    state["turns_completed"] = turn
    state["latest_gain_status"] = proj.gain.status
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    _save_json(state_path, state)
    print(
        f"[libero-g17-fg] takeover turn={turn} trial={trial} "
        f"gain={proj.gain.status} bound_release={int(proj.gain.bound_release)} "
        f"world_complete={int(proj.world_fg_complete)} "
        f"gap={proj.endpoint_gap} task_gap={proj.task_gap} "
        f"finish={int(proj.finish_authorized)} "
        f"attach={proj.executor_attachment} grasp={proj.interaction_grasp!r} "
        f"peak={proj.executor_attachment_peak} peak_grasp={proj.interaction_grasp_peak!r} "
        f"released_goal={int(proj.released_goal_after_transport)} "
        f"dom_c={int(dominate.get('correctness_ok', True))} "
        f"dom_p={int(dominate.get('completeness_ok', True))} "
        f"novel={len(proj.gain.novel_fact_ids)}",
        flush=True,
    )
    _LAST_TAKEOVER_BUNDLE[
        (str(Path(output_dir).resolve()), int(trial), int(turn))
    ] = {
        "post_miss_clear": bool(clear_cache),
        "replacement": replacement,
        "native_report": native_report,
        "goal_referent_takeover": goal_ref_meta,
        "fg_native_retry": native_retry_meta,
    }
    return corrected


def capture_terminal_witness(
    env: Any,
    *,
    output_dir: str,
    trial: int,
    turn: int,
    task: str,
    private_task_completed: bool,
    stop_reason: str,
) -> dict[str, Any]:
    """Layer-1 fix: project FG at episode end when multi-turn is skipped.

    When ``task_completed`` (or other stop) fires after ``env.step``, trial.py
    breaks before ``run_takeover``. Without this witness, success states never
    appear in ``libero_g17_fg_traces`` and ``finish_authorized`` stays false
    even on OK trials.
    """
    key = _cache_key(output_dir, trial)
    snapshots = _PRIVATE_SNAPSHOTS.get(key)
    binding = _PRIVATE_BINDINGS.get(key)
    if snapshots is None or binding is None:
        raise ValueError("LIBERO G17 FG terminal witness requires initialized private state")
    if len(snapshots) != turn:
        raise ValueError(
            "LIBERO G17 FG terminal witness is not turn-synchronous: "
            f"len(snapshots)={len(snapshots)} turn={turn}"
        )
    root = Path(output_dir) / TRACE_DIR / f"trial_{trial:02d}"
    state_path = root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("trial") != trial
        or state.get("turns_completed") != turn - 1
    ):
        raise ValueError("LIBERO G17 FG terminal witness provenance mismatch")

    before = snapshots[-1]
    attachments = _PRIVATE_ATTACHMENTS.get(key)
    if attachments is None or len(attachments) != turn:
        raise ValueError(
            "LIBERO G17 FG terminal witness attachment cache is not turn-synchronous"
        )
    attachment_before = attachments[-1]
    sample_interval_attachment(env, force=True)
    peak = _consume_interval_peak(key)
    binding, after, attachment = adapter.extract_libero_snapshot(env, binding)
    attachment_peak = adapter.merge_attachment_peak(peak, attachment)
    attachment_peak = adapter.merge_attachment_peak(
        attachment_peak, attachment_before
    )
    _PRIVATE_BINDINGS[key] = binding
    proj = projection.project_libero_fg(
        binding,
        before,
        after,
        attachment=attachment,
        attachment_before=attachment_before,
        attachment_peak=attachment_peak,
    )
    for term in _FORBIDDEN:
        if term in proj.combined_report.lower():
            raise ValueError(f"LIBERO G17 FG report contains prohibited term: {term}")
    if _NUMERIC.search(proj.combined_report):
        raise ValueError("LIBERO G17 FG report exposes a numeric value")

    audit = adapter.g17_versus_bddl_audit(after, env, binding)
    if audit["bddl_iff_task_completed"] is not True:
        raise RuntimeError("LIBERO G17 FG private BDDL IFF audit failed")
    if bool(private_task_completed) != bool(audit["private_task_completed"]):
        raise RuntimeError("LIBERO G17 FG task_completed disagrees with domain checker")

    # Layer-1 freeze gate on success stops: public finish must match private completion.
    if private_task_completed and not bool(proj.finish_authorized):
        raise RuntimeError(
            "LIBERO G17 FG Layer-1 witness gap: task_completed but finish_authorized=False"
        )
    if private_task_completed and not bool(audit["bddl_atom_true"]):
        raise RuntimeError(
            "LIBERO G17 FG Layer-1 witness gap: task_completed but bddl_atom_true=False"
        )

    turn_root = root / f"turn_{turn:02d}"
    turn_root.mkdir(parents=True, exist_ok=False)
    (turn_root / "takeover_report.txt").write_text(
        proj.combined_report, encoding="utf-8"
    )
    (turn_root / "parent_report.txt").write_text(proj.parent_report, encoding="utf-8")
    (turn_root / "endpoint_report.txt").write_text(
        proj.endpoint_report, encoding="utf-8"
    )
    if proj.release_report:
        (turn_root / "release_report.txt").write_text(
            proj.release_report, encoding="utf-8"
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial": trial,
        "turn": turn,
        "carrier": "terminal_success_witness_libero_g17_fg",
        "stop_reason": stop_reason,
        "gain_status": proj.gain.status,
        "bound_release": bool(proj.gain.bound_release),
        "release_package_applied": bool(proj.release_report),
        "world_fg_complete": bool(proj.world_fg_complete),
        "endpoint_gap": proj.endpoint_gap,
        "finish_authorized": bool(proj.finish_authorized),
        "executor_attachment": proj.executor_attachment,
        "interaction_grasp": proj.interaction_grasp,
        "executor_attachment_peak": proj.executor_attachment_peak,
        "interaction_grasp_peak": proj.interaction_grasp_peak,
        "task_gap": proj.task_gap,
        "released_goal_after_transport": bool(
            proj.released_goal_after_transport
        ),
        "goal_evidence_key_count": len(proj.goal_evidence_keys),
        "acquire_evidence_key_count": len(
            projection.acquire_evidence_keys(binding)
        ),
        "place_evidence_key_count": len(projection.place_evidence_keys(binding)),
        "novel_fact_ids": list(proj.gain.novel_fact_ids),
        "novel_fact_count": len(proj.gain.novel_fact_ids),
        "parent_sha256": proj.parent_sha256,
        "combined_sha256": proj.combined_sha256,
        "parent_byte_identity_ok": projection.parent_byte_identity_preserved(proj),
        "takeover_report_sha256": _sha256_text(proj.combined_report),
        "bddl_atom_true": audit["bddl_atom_true"],
        "private_task_completed": audit["private_task_completed"],
        "bddl_iff_task_completed": audit["bddl_iff_task_completed"],
        "geometric_proxy": audit["geometric_proxy"],
        "g17_iff_bddl": audit["g17_iff_bddl"],
        "action_visible_goal_label": False,
        "terminal_witness": True,
    }
    _save_json(turn_root / "record.json", record)
    snapshots.append(after)
    attachments.append(attachment)
    xy_lock.publish_xy_lock(binding, attachment)
    state["turns_completed"] = turn
    state["latest_gain_status"] = proj.gain.status
    state["latest_public_state_sha256"] = _sha256_json(after.values)
    state["terminal_witness"] = {
        "turn": turn,
        "stop_reason": stop_reason,
        "finish_authorized": bool(proj.finish_authorized),
        "bddl_atom_true": bool(audit["bddl_atom_true"]),
        "private_task_completed": bool(private_task_completed),
    }
    _save_json(state_path, state)
    print(
        f"[libero-g17-fg] terminal_witness turn={turn} trial={trial} "
        f"stop={stop_reason} finish={int(proj.finish_authorized)} "
        f"bddl={int(bool(audit['bddl_atom_true']))} "
        f"task_completed={int(private_task_completed)} gap={proj.endpoint_gap} "
        f"task_gap={proj.task_gap} released_goal={int(proj.released_goal_after_transport)}",
        flush=True,
    )
    return record
