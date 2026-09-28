"""Validate Agilex tennis task code in a MuJoCo robosuite proxy before/after real runs."""

from __future__ import annotations

import contextlib
import io
import os
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import viser.transforms as vtf

from capx.envs.simulators.agilex_mujoco_pickplace_proxy import AgilexMujocoPickPlaceCanProxy
from capx.integrations.base_api import ApiBase
from capx.integrations.motion.pyroki import init_pyroki


@dataclass
class MujocoSimValidationResult:
    exec_ok: bool
    sim_success: bool
    stderr: str
    stdout: str
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "exec_ok": self.exec_ok,
            "sim_success": self.sim_success,
            "stderr": self.stderr,
            "stdout": self.stdout,
            "reason": self.reason,
            "details": self.details,
        }


class _TennisMujocoProxyApi(ApiBase):
    """Agilex-like API surface backed by MuJoCo privileged poses + Panda IK."""

    _DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, low_level: AgilexMujocoPickPlaceCanProxy) -> None:
        super().__init__(low_level)
        self._ik = init_pyroki()
        self._cfg = None

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_object_pose(
        self, object_name: str, camera: str = "main"
    ) -> tuple[np.ndarray, np.ndarray]:
        del camera
        return self._env.get_alias_object_pose(object_name)

    def sample_grasp_pose(
        self, object_name: str, camera: str = "main"
    ) -> tuple[np.ndarray, np.ndarray]:
        del camera
        pos, quat = self.get_object_pose(object_name)
        grasp = pos.copy()
        grasp[2] += 0.04
        return grasp, quat

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        gripper: float | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = pos + (
            vtf.SO3(wxyz=quat_wxyz).apply(self._TCP_OFFSET)
        )
        target = np.concatenate([quat_wxyz, offset_pos])
        if self._cfg is None:
            self._cfg = self._ik(target_pose_wxyz_xyz=target)
        else:
            self._cfg = self._ik(target_pose_wxyz_xyz=target, prev_cfg=self._cfg)
        joints = np.asarray(self._cfg, dtype=np.float64).reshape(-1)[:7]
        self._env.move_to_joints_blocking(joints)
        if gripper is not None:
            self._env._set_gripper(float(gripper))
            self._env._step_once()

    def open_gripper(self) -> None:
        self._env._set_gripper(1.0)
        self._env._step_once()

    def close_gripper(self) -> None:
        self._env._set_gripper(0.0)
        for _ in range(15):
            self._env._step_once()


def validate_agilex_tennis_code_in_mujoco(
    code: str,
    *,
    seed: int | None = 42,
    max_steps: int = 2000,
) -> MujocoSimValidationResult:
    """Execute generated Agilex tennis code inside a MuJoCo pick-place proxy."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    low_level = AgilexMujocoPickPlaceCanProxy(max_steps=max_steps, seed=seed)
    proxy_api = _TennisMujocoProxyApi(low_level)
    low_level.reset(seed=seed)

    exec_globals: dict[str, Any] = {
        "__name__": "__main__",
        "env": low_level,
        "APIS": {"TennisMujocoProxyApi": proxy_api},
        "np": np,
    }
    for fn_name, fn in proxy_api.functions().items():
        exec_globals[fn_name] = fn

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exec_ok = True
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, exec_globals, exec_globals)
    except BaseException:
        exec_ok = False
        traceback.print_exc(file=stderr_buffer)

    sim_success = bool(low_level.task_completed()) if exec_ok else False
    if exec_ok and sim_success:
        reason = "mujoco_sim_success"
    elif exec_ok:
        reason = "mujoco_exec_ok_but_not_in_bin"
    else:
        reason = "mujoco_exec_failed"

    return MujocoSimValidationResult(
        exec_ok=exec_ok,
        sim_success=sim_success,
        stderr=stderr_buffer.getvalue(),
        stdout=stdout_buffer.getvalue(),
        reason=reason,
        details={
            "method": "robosuite_pickplace_can_proxy",
            "reward": float(low_level.compute_reward()),
            "sim_task_completed": sim_success,
        },
    )
