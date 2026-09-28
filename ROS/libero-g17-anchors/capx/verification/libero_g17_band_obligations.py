"""Generic FG band obligations: mini predicate contract (current / required).

Not task-specific English hardcoding: open RelXY-style bands project to
FGPP-style ``current; required; satisfied`` rows via the shared lexicon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

Axis = Literal["x", "y", "z"]

_WITHIN = "WITHIN_REFERENCE_BAND"
_NEG = "NEGATIVE_OF_REFERENCE_BAND"
_POS = "POSITIVE_OF_REFERENCE_BAND"

# Robot-frame axis CNL (same as libero_g17_fg_lexicon._AXIS_VALUE_CNL).
_AXIS_CNL = {
    "x": {
        _NEG: "closer to the robot than {ref}",
        _WITHIN: "at the same depth as {ref}",
        _POS: "farther from the robot than {ref}",
    },
    "y": {
        _NEG: "to the robot's right of {ref}",
        _WITHIN: "laterally aligned with {ref}",
        _POS: "to the robot's left of {ref}",
    },
    "z": {
        _NEG: "below {ref}",
        _WITHIN: "level with {ref}",
        _POS: "above {ref}",
    },
}

_AXIS_LABEL = {
    "x": "robot-x depth",
    "y": "robot-y lateral",
    "z": "robot-z height",
}

# Match public On XY band (libero_g17_public_iff.LIBERO_OBJECT_ONTOP_XY_M).
_DEFAULT_AXIS_BAND_M = 0.03


def band_obligations_enabled() -> bool:
    return os.environ.get("CAPX_LIBERO_G17_BAND_OBLIGATIONS", "0") == "1"


def invert_band(band: str) -> str:
    if band == _NEG:
        return _POS
    if band == _POS:
        return _NEG
    return band


def sign_to_band(*, axis: Axis, sign: str) -> str:
    """Map qualitative offset signs (closer/farther/left/right) to RelXY bands."""
    if axis == "x":
        if sign == "closer":
            return _NEG
        if sign == "farther":
            return _POS
    if axis == "y":
        if sign == "right":
            return _NEG
        if sign == "left":
            return _POS
    if axis == "z":
        if sign == "below":
            return _NEG
        if sign == "above":
            return _POS
    raise ValueError(f"unsupported axis/sign: {axis!r}/{sign!r}")


def residual_to_band(*, axis: Axis, residual: float, band_m: float) -> str:
    """Per-axis WITHIN when |residual| < band; else signed side."""
    if abs(float(residual)) < float(band_m):
        return _WITHIN
    if axis == "x":
        return _POS if residual > 0.0 else _NEG
    if axis == "y":
        return _POS if residual > 0.0 else _NEG
    if axis == "z":
        return _POS if residual > 0.0 else _NEG
    raise ValueError(f"unsupported axis: {axis!r}")


def cnl_for_band(axis: Axis, band: str, *, ref: str) -> str:
    return _AXIS_CNL[axis][band].format(ref=ref)


@dataclass(frozen=True)
class BandObligation:
    """One axis row of the mini align contract."""

    axis: Axis
    current_band: str
    goal_band: str
    predicate: str
    ref_label: str

    @property
    def current_cnl(self) -> str:
        return cnl_for_band(self.axis, self.current_band, ref=self.ref_label)

    @property
    def goal_cnl(self) -> str:
        return cnl_for_band(self.axis, self.goal_band, ref=self.ref_label)

    @property
    def axis_label(self) -> str:
        return _AXIS_LABEL[self.axis]

    @property
    def satisfied(self) -> bool:
        return self.current_band == self.goal_band


def pick_primary_obligation(
    obligations: list[BandObligation],
) -> BandObligation | None:
    """Chess order among *open* rows: lateral (y) then depth (x) then height (z)."""
    open_rows = [o for o in obligations if not o.satisfied]
    if not open_rows:
        return None
    rank = {"y": 0, "x": 1, "z": 2}
    return sorted(open_rows, key=lambda o: rank.get(o.axis, 9))[0]


def open_c2c_band_obligations(
    *,
    payload_label: str,
    receptacle_label: str,
    depth_sign: str,
    lateral_sign: str,
    center_within: bool,
) -> list[BandObligation]:
    """Legacy sign-only open axes (open rows only; no satisfied siblings)."""
    del payload_label
    if center_within:
        return []
    ref = f"{receptacle_label} center"
    out: list[BandObligation] = []
    for axis, sign in (("x", depth_sign), ("y", lateral_sign)):
        band = sign_to_band(axis=axis, sign=sign)  # type: ignore[arg-type]
        if band == _WITHIN:
            continue
        out.append(
            BandObligation(
                axis=axis,  # type: ignore[arg-type]
                current_band=band,
                goal_band=_WITHIN,
                predicate="center_to_center",
                ref_label=ref,
            )
        )
    return out


def open_c2c_band_contract(
    *,
    payload_label: str,
    receptacle_label: str,
    dx: float,
    dy: float,
    center_within: bool,
    band_m: float = _DEFAULT_AXIS_BAND_M,
) -> list[BandObligation]:
    """Both XY axes as contract rows (satisfied + open) when center is outside."""
    del payload_label
    if center_within:
        return []
    ref = f"{receptacle_label} center"
    return [
        BandObligation(
            axis="x",
            current_band=residual_to_band(axis="x", residual=dx, band_m=band_m),
            goal_band=_WITHIN,
            predicate="center_to_center",
            ref_label=ref,
        ),
        BandObligation(
            axis="y",
            current_band=residual_to_band(axis="y", residual=dy, band_m=band_m),
            goal_band=_WITHIN,
            predicate="center_to_center",
            ref_label=ref,
        ),
    ]


def render_band_obligation_lines(
    obligations: list[BandObligation],
    *,
    payload_label: str,
    primary_only: bool = False,
    holding_payload: bool | None = None,
) -> list[str]:
    """Student-facing mini predicate contract (no API / numeric coaching)."""
    if not obligations:
        return []
    if primary_only:
        primary = pick_primary_obligation(obligations)
        chosen = [primary] if primary is not None else []
    else:
        # Chess order for reading; keep satisfied siblings visible.
        rank = {"y": 0, "x": 1, "z": 2}
        chosen = sorted(obligations, key=lambda o: rank.get(o.axis, 9))
    if not chosen:
        return []

    lines = ["Instantiated predicate contract:"]
    for obl in chosen:
        status = "satisfied." if obl.satisfied else "not satisfied."
        required = (
            "within band"
            if obl.goal_band == _WITHIN
            else obl.goal_cnl
        )
        # Side language for current when outside; "within band" when already in.
        current = (
            "within band" if obl.current_band == _WITHIN else obl.current_cnl
        )
        lines.append(
            f"- {payload_label} aligned with {obl.ref_label} along "
            f"{obl.axis_label}:"
        )
        lines.append(
            f"  current {current}; required {required}; {status}"
        )

    if holding_payload is True:
        lines.append("Terminal factual conditions:")
        lines.append(
            f"- gripper holding {payload_label}:"
        )
        lines.append(
            "  current true; target true; satisfied."
        )
    return lines


def render_placement_xy_band_obligations(
    binding: Any,
    after: Any,
    basis: Any,
    *,
    depth_sign: str,
    lateral_sign: str,
    dx: float | None = None,
    dy: float | None = None,
    holding_payload: bool | None = None,
) -> list[str]:
    """Emit c2c mini-contract when band-obligation env is on and XY is open."""
    if not band_obligations_enabled():
        return []
    if getattr(binding, "relation", None) != "on":
        return []
    center_within = getattr(basis, "xy_band", None) == "WITHIN_ONTOP_XY_BAND"
    del after
    payload = binding.scene.objects[0].public_name
    receptacle = binding.scene.objects[1].public_name
    if dx is not None and dy is not None:
        obls = open_c2c_band_contract(
            payload_label=payload,
            receptacle_label=receptacle,
            dx=float(dx),
            dy=float(dy),
            center_within=center_within,
        )
    else:
        obls = open_c2c_band_obligations(
            payload_label=payload,
            receptacle_label=receptacle,
            depth_sign=depth_sign,
            lateral_sign=lateral_sign,
            center_within=center_within,
        )
    return render_band_obligation_lines(
        obls,
        payload_label=payload,
        primary_only=False,
        holding_payload=holding_payload,
    )
