"""FG RelXY token → same-name instance resolve without breaking stock grasp.

Policy (net fix for type-level lottery):
1. Rank sim bodies by robot-frame RelXY (``right``/``left``/``closer``/``farther``).
2. Run stock Molmo+SAM; **keep it** when the cloud matches the chosen body.
3. Only then point-correct with a projected body pixel.

``object_name`` stays a type string. Omitted ``which`` reads the live FG token.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.verification.libero_g17_fg_lexicon import humanize_libero_name
from capx.verification.libero_g17_fg_projection import _wrong_grasp_api_xy_token

_LIVE: dict[str, Any] | None = None
_TOKENS = ("right", "left", "closer", "farther")


def xy_lock_enabled() -> bool:
    import os

    return os.environ.get("CAPX_LIBERO_G17_XY_LOCK", "0") == "1"


def publish_xy_lock(binding: Any, attachment: Any, *, reset: bool = False) -> None:
    """Publish FG identity while preserving the bound instance across regen.

    The RelXY token is evidence used to resolve a task payload instance.  Once
    resolved, empty-hand or wrong-grasp feedback must not erase that identity:
    those states say the acquisition failed, not that the task target changed.
    ``reset=True`` is used only at a new trial boundary.
    """
    global _LIVE
    class_name = humanize_libero_name(binding.payload_name)
    body_name = str(binding.payload_name)
    token = None
    bands = getattr(attachment, "goal_vs_same_name_other_rel_xyz", None)
    sibling = getattr(attachment, "same_name_other_raw", None) is not None
    if sibling and bands is not None:
        token = _wrong_grasp_api_xy_token(bands)
    prior = None if reset else _LIVE
    same_binding = bool(
        prior is not None and prior.get("body_name") == body_name
    )
    if token is None and same_binding:
        token = prior.get("token")
    if same_binding:
        sibling = bool(sibling or prior.get("same_name"))
    _LIVE = {
        "token": token,
        "class_name": class_name,
        "same_name": bool(sibling),
        "body_name": body_name,
    }


def clear_xy_lock() -> None:
    global _LIVE
    _LIVE = None


def live_xy_lock() -> dict[str, Any] | None:
    if not _LIVE or not _LIVE.get("token") or not _LIVE.get("class_name"):
        return None
    return dict(_LIVE)


def query_matches_class(query: str, class_name: str) -> bool:
    q = str(query).strip().lower().replace("_", " ")
    c = str(class_name).strip().lower().replace("_", " ")
    if not q or not c:
        return False
    return c in q


def normalize_which(which: str | None) -> str | None:
    if which is None:
        return None
    w = str(which).strip().lower()
    return w if w in _TOKENS else None


def resolve_lock(text_prompt: str, which: str | None = None) -> dict[str, Any] | None:
    """Token for this call: explicit ``which``, else live FG token on the same class."""
    if not xy_lock_enabled():
        return None
    token = normalize_which(which)
    lock = live_xy_lock()
    if _LIVE and not query_matches_class(text_prompt, str(_LIVE.get("class_name") or "")):
        return None
    if token is not None:
        return {
            "token": token,
            "class_name": (lock or _LIVE or {}).get("class_name") or str(text_prompt),
            "body_name": (lock or _LIVE or {}).get("body_name"),
        }
    return lock


def same_name_unresolved(text_prompt: str, which: str | None = None) -> bool:
    """True when FG saw a sibling but this skill call has no ``which`` and no token."""
    if not xy_lock_enabled() or _LIVE is None:
        return False
    if not _LIVE.get("same_name"):
        return False
    if not query_matches_class(text_prompt, str(_LIVE.get("class_name") or "")):
        return False
    if normalize_which(which) or _LIVE.get("token"):
        return False
    return True


def world_to_base(points_xyz: np.ndarray, base_wxyz_xyz: np.ndarray) -> np.ndarray:
    """Robot-base frame used by RelXY (+X forward, +Y robot-left)."""
    from scipy.spatial.transform import Rotation as SciRotation

    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(base_wxyz_xyz, dtype=np.float64).reshape(7)
    wxyz, origin = pose[:4], pose[4:]
    rot = SciRotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return rot.inv().apply(pts - origin)


def pick_index_by_token(centroids_base: np.ndarray, token: str) -> int:
    xy = np.asarray(centroids_base, dtype=np.float64).reshape(-1, 3)[:, :2]
    if token == "right":
        return int(np.argmin(xy[:, 1]))
    if token == "left":
        return int(np.argmax(xy[:, 1]))
    if token == "closer":
        return int(np.argmin(xy[:, 0]))
    if token == "farther":
        return int(np.argmax(xy[:, 0]))
    raise ValueError(f"unsupported XY lock token: {token!r}")


def choose_complete_index(
    centroids_world: np.ndarray,
    token: str,
    base_wxyz_xyz: np.ndarray,
) -> int:
    return pick_index_by_token(world_to_base(centroids_world, base_wxyz_xyz), token)


def robot_base_or_none(env: Any) -> np.ndarray | None:
    for obj in (
        env,
        getattr(env, "low_level_env", None),
        getattr(env, "_env", None),
    ):
        if obj is None:
            continue
        pose = getattr(obj, "base_link_wxyz_xyz", None)
        if pose is None:
            continue
        arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if arr.size == 7:
            return arr
    try:
        from capx.verification import libero_g17_adapter as adapter

        domain = adapter._libero_domain(env)
        wrapper = adapter._capx_wrapper(env)
        pos, ori = adapter._robot_base_frame(domain, wrapper)
        return np.concatenate(
            [np.asarray(ori, dtype=np.float64).reshape(4), np.asarray(pos, dtype=np.float64).reshape(3)]
        )
    except Exception:
        return None


def project_frame_to_pixel(
    xyz_frame: np.ndarray,
    t_frame_from_cam: np.ndarray,
    intrinsics: np.ndarray,
    hw: tuple[int, int],
) -> tuple[int, int] | None:
    """Project a point already in the parent frame of ``pose_mat``.

    LIBERO ``pose_mat`` is ``T_robot_from_cam`` (OpenCV optical), not world.
    Pass robot-base XYZ here (not MuJoCo ``body_xpos``).
    """
    T = np.asarray(t_frame_from_cam, dtype=np.float64).reshape(4, 4)
    T_cam = np.linalg.inv(T)
    p = np.append(np.asarray(xyz_frame, dtype=np.float64).reshape(3), 1.0)
    cam = T_cam @ p
    if cam[2] <= 1e-6:
        return None
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    u = k[0, 0] * cam[0] / cam[2] + k[0, 2]
    v = k[1, 1] * cam[1] / cam[2] + k[1, 2]
    h, w = int(hw[0]), int(hw[1])
    if not (0 <= u < w and 0 <= v < h):
        return None
    return int(round(u)), int(round(v))


# Back-compat alias used by older tests / call sites.
project_world_to_pixel = project_frame_to_pixel


def body_world_to_robot_xyz(env: Any, xyz_world: np.ndarray) -> np.ndarray | None:
    """MuJoCo world XYZ → robot-base XYZ (same frame as LIBERO ``pose_mat``)."""
    base = robot_base_or_none(env)
    if base is None:
        return None
    return world_to_base(xyz_world, base)[0]


def mask_centroid_matches_body(
    centroid_robot: np.ndarray,
    body_robot: np.ndarray,
    *,
    thresh_xy_m: float = 0.08,
    thresh_z_m: float | None = None,
) -> bool:
    """True when a segmented cloud is consistent with the chosen body.

    ``thresh_z_m=None`` preserves the frozen v286 XY-only contract.  v291
    explicitly supplies a Z tolerance so a cabinet-top target cannot accept a
    tabletop/cabinet mask merely because their projected XY overlaps.
    """
    c = np.asarray(centroid_robot, dtype=np.float64).reshape(3)
    b = np.asarray(body_robot, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(c[:2] - b[:2])) > float(thresh_xy_m):
        return False
    if thresh_z_m is not None and abs(float(c[2] - b[2])) > float(thresh_z_m):
        return False
    return True


def same_name_body_world_centroids(
    env: Any, text_prompt: str
) -> list[tuple[str, np.ndarray]]:
    """Graspable bodies whose public name matches ``text_prompt``, with world XYZ."""
    from capx.verification import libero_g17_adapter as adapter

    domain = adapter._libero_domain(env)
    rows: list[tuple[str, np.ndarray]] = []
    for name in adapter._graspable_movable_names(domain):
        if not query_matches_class(text_prompt, humanize_libero_name(name)):
            continue
        body_id = adapter._body_id(domain, name)
        pos = np.asarray(domain.sim.data.body_xpos[body_id], dtype=np.float64).reshape(3)
        rows.append((str(name), pos.copy()))
    return rows


def pick_same_name_body(
    env: Any, text_prompt: str, token: str
) -> tuple[str, np.ndarray] | None:
    """Select one same-name body by robot-frame RelXY token (privileged centroids)."""
    rows = same_name_body_world_centroids(env, text_prompt)
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    base = robot_base_or_none(env)
    if base is None:
        return None
    names = [name for name, _ in rows]
    cents = np.stack([pos for _, pos in rows], axis=0)
    idx = choose_complete_index(cents, token, base)
    return names[idx], cents[idx]


def pick_bound_body(
    env: Any, text_prompt: str, body_name: str
) -> tuple[str, np.ndarray] | None:
    """Resolve a previously bound task instance without rerunning the lottery."""
    wanted = str(body_name)
    for name, position in same_name_body_world_centroids(env, text_prompt):
        if name == wanted:
            return name, position
    return None
