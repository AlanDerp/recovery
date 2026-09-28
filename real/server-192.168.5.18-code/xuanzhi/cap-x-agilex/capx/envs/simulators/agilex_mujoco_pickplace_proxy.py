"""MuJoCo robosuite PickPlaceCan proxy for Agilex tennis pick-place code validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class AgilexMujocoPickPlaceCanProxy(RobosuiteBaseEnv):
    """Robosuite can-in-bin task used as a MuJoCo proxy for tennis-ball-in-box."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 2000,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = True,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=viser_debug,
            privileged=privileged,
            enable_render=enable_render,
        )
        self.robosuite_env = suite.environments.manipulation.pick_place.PickPlaceCan(
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(
                controller=self.controller_cfg
            ),
            horizon=max_steps,
            reward_shaping=False,
        )
        self._init_robot_links()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.robosuite_env.reset()
        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(40):
            self.robosuite_env.sim.forward()
            self._set_gripper(1.0)
            self._do_robosuite_step(
                np.concatenate(
                    [
                        self._current_joints,
                        [self._gripper_fraction, self._gripper_fraction],
                    ]
                )
            )

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        obs = self.get_observation()
        info = {"task_prompt": "MuJoCo proxy: place can into bin (tennis-ball analog)."}
        return obs, info

    def compute_reward(self) -> float:
        return float(self.robosuite_env.reward())

    def task_completed(self) -> bool:
        return bool(self.robosuite_env._check_success())

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs

    def _base_transform(self) -> vtf.SE3:
        return vtf.SE3(
            wxyz_xyz=np.concatenate(
                [
                    self.robosuite_env.sim.data.xquat[self.base_link_idx],
                    self.robosuite_env.sim.data.xpos[self.base_link_idx],
                ]
            )
        ).inverse()

    def get_alias_object_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Map Agilex tennis/box names to privileged MuJoCo poses in robot base frame."""
        name = object_name.strip().lower()
        sim = self.robosuite_env.sim
        rs_env = self.robosuite_env
        down_quat = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

        if "ball" in name or "tennis" in name:
            obj = rs_env.objects[rs_env.object_id]
            body_id = rs_env.obj_body_id[obj.name]
            world = vtf.SE3(
                wxyz_xyz=np.concatenate(
                    [sim.data.xquat[body_id], sim.data.xpos[body_id]]
                )
            )
        elif "box" in name or "cardboard" in name:
            bin_id = min(max(int(rs_env.object_id), 0), 1)
            bin_name = "bin1" if bin_id == 0 else "bin2"
            body_id = sim.model.body_name2id(bin_name)
            world = vtf.SE3.from_translation(sim.data.body_xpos[body_id].copy())
        else:
            raise ValueError(f"Unsupported proxy object name: {object_name}")

        robot = self._base_transform() @ world
        return robot.translation().astype(np.float64), down_quat.copy()


__all__ = ["AgilexMujocoPickPlaceCanProxy"]
