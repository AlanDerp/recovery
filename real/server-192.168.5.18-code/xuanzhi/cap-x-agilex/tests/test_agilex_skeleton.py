import numpy as np
import pytest


def test_agilex_observation_adapter_repackages_single_arm_and_main_camera():
    from capx.envs.simulators.agilex_real import AgilexObsAdapter

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.ones((4, 5), dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float64)
    pose = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pose_mat = np.eye(4, dtype=np.float64)

    msg = {
        b"joint_pos": np.arange(6),
        b"gripper": 0.25,
        b"camera_main": {
            b"images": {b"rgb": rgb},
            b"depth": depth,
            b"intrinsics": intrinsics,
            b"pose": pose,
            b"pose_mat": pose_mat,
        },
    }

    obs = AgilexObsAdapter().convert(msg)

    assert obs["robot_joint_pos"].shape == (7,)
    assert obs["robot_joint_pos"][-1] == 0.25
    assert obs["robot0_robotview"]["images"]["rgb"].shape == (4, 5, 3)
    assert obs["robot0_robotview"]["images"]["depth"].shape == (4, 5, 1)
    assert set(obs) == {"robot_joint_pos", "robot0_robotview"}


def test_agilex_api_and_low_level_are_registered():
    import capx.envs.simulators  # noqa: F401
    import capx.integrations  # noqa: F401
    from capx.envs.base import list_envs
    from capx.integrations.base_api import list_apis

    assert "agilex_real_low_level" in list_envs()
    assert "AgilexControlApi" in list_apis()


def test_agilex_control_api_exposes_single_arm_functions():
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel
    from capx.integrations.agilex.control import AgilexControlApi

    env = AgilexRealLowLevel(start_server=False, wait_for_observation=False)
    api = AgilexControlApi(env, initialize_perception=False, initialize_motion=False)

    names = set(api.functions())

    assert {
        "get_observation",
        "get_camera_observation",
        "move_to_joints",
        "open_gripper",
        "close_gripper",
        "solve_ik",
        "goto_pose",
    }.issubset(names)
    assert not any("arm0" in name or "arm1" in name or "both" in name for name in names)


def test_agilex_low_level_records_expected_single_arm_command():
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel

    env = AgilexRealLowLevel(start_server=False, wait_for_observation=False)
    env.move_to_joints_blocking(
        np.arange(6, dtype=np.float64),
        gripper=0.1,
        max_steps=0,
    )

    np.testing.assert_allclose(
        env.latest_action["joint_pos"],
        [0.0, 1.0, -0.005, 1.7403292, 1.2167304, 2.0893951],
        atol=1e-6,
    )
    assert env.latest_action["gripper"] == 0.1
    assert "right" not in env.latest_action


def test_agilex_reset_interpolates_to_configured_home():
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel

    class FakeHardware:
        def __init__(self):
            self.joints = np.array([0.5, 0.6, -0.7, 0.8, -0.9, 1.0], dtype=np.float64)
            self.width = 0.01
            self.moves = []

        def get_state(self):
            return self.joints.copy(), self.width

        def move(self, *, joints=None, gripper_width=None):
            if joints is not None:
                self.joints = np.asarray(joints, dtype=np.float64).reshape(6)
            if gripper_width is not None:
                self.width = float(gripper_width)
            self.moves.append((self.joints.copy(), self.width))

    env = AgilexRealLowLevel(
        control_backend="direct",
        camera_backend="none",
        connect_on_init=False,
        start_server=False,
        wait_for_observation=False,
        reset_interpolation_steps=50,
    )
    fake_hardware = FakeHardware()
    env.hardware = fake_hardware

    env.reset()

    assert len(fake_hardware.moves) == 50
    np.testing.assert_allclose(fake_hardware.moves[-1][0], np.zeros(6), atol=1e-12)
    assert not np.allclose(fake_hardware.moves[0][0], fake_hardware.moves[-1][0])
    assert fake_hardware.moves[-1][1] == 0.05
    assert env.latest_action["gripper"] == 0.5


def test_agilex_reset_skips_when_clipped_joints_already_near_home():
    from capx.envs.simulators.agilex_real import AgilexRealLowLevel

    class FakeHardware:
        def __init__(self):
            # Slightly out-of-range feedback near software home (common on real Piper).
            self.joints = np.array(
                [0.0, -0.02, 0.02, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.width = 0.05
            self.moves = []

        def get_state(self):
            return self.joints.copy(), self.width

        def move(self, *, joints=None, gripper_width=None):
            if joints is not None:
                self.joints = np.asarray(joints, dtype=np.float64).reshape(6)
            if gripper_width is not None:
                self.width = float(gripper_width)
            self.moves.append((self.joints.copy(), self.width))

    env = AgilexRealLowLevel(
        control_backend="direct",
        camera_backend="none",
        connect_on_init=False,
        start_server=False,
        wait_for_observation=False,
        reset_interpolation_steps=50,
        reset_joint_tolerance=0.03,
    )
    fake_hardware = FakeHardware()
    env.hardware = fake_hardware

    env.reset()

    assert fake_hardware.moves == []
    np.testing.assert_allclose(env._current_joints, np.zeros(6), atol=1e-12)


def test_agilex_ros_arm_hardware_publishes_seven_joint_positions():
    import threading

    import numpy as np
    from capx.envs.simulators.agilex_real import AgilexRosArmHardware

    class FakeHeader:
        def __init__(self):
            self.stamp = None

    class FakeJointState:
        def __init__(self):
            self.header = FakeHeader()
            self.position = None

    published = []

    class FakePublisher:
        def publish(self, msg):
            published.append(msg)

    puppet_msg = type("JointState", (), {"position": [0.1, 0.2, -0.3, 0.0, 0.0, 0.0, 0.04]})()

    hardware = AgilexRosArmHardware.__new__(AgilexRosArmHardware)
    hardware._rospy = type("rospy", (), {"Time": type("FakeTime", (), {"now": staticmethod(lambda: 0)})})()
    hardware._JointState = FakeJointState
    hardware.gripper_max_width = 0.1
    hardware.command_sleep_s = 0.0
    hardware.joint_limit_margin = 0.01
    hardware._lock = threading.Lock()
    hardware._latest_puppet = puppet_msg
    hardware._cmd_joints = np.zeros(6)
    hardware._cmd_width = 0.1
    hardware._pub = FakePublisher()

    hardware.move(joints=np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]), gripper_width=0.05)
    assert len(published) == 1
    assert len(published[0].position) == 7
    np.testing.assert_allclose(published[0].position[:6], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert published[0].position[6] == 0.05


def test_clip_piper_joint_targets_clamps_near_zero_violations():
    from capx.envs.simulators.agilex_real import clip_piper_joint_targets

    clipped = clip_piper_joint_targets(
        [0.0, -0.02, 0.02, 0.0, 0.0, 0.0], margin=0.01
    )
    np.testing.assert_allclose(clipped, [0.0, 0.01, -0.01, 0.0, 0.0, 0.0], atol=1e-12)


def test_reset_agilex_from_config_path_skips_non_agilex_config(tmp_path):
    from capx.envs.simulators.agilex_reset import reset_agilex_from_config_path

    config_path = tmp_path / "not_agilex.yaml"
    config_path.write_text(
        "env:\n  cfg:\n    low_level:\n      _target_: capx.envs.simulators.franka_real.FrankaRealLowLevel\n"
    )
    assert reset_agilex_from_config_path(str(config_path)) is False


def test_staged_tennis_plan_requires_the_safe_fixed_order():
    from capx.envs.trial_agilex_ensemble import _TENNIS_STAGE_IDS, _parse_staged_plan

    content = '{"stages": [{"id": "perceive"}, {"id": "approach"}, {"id": "grasp_gate"}, {"id": "transfer"}, {"id": "place"}, {"id": "verify"}]}'
    assert _parse_staged_plan(content) == _TENNIS_STAGE_IDS

    with pytest.raises(ValueError, match="unsafe staged plan"):
        _parse_staged_plan('{"stages": ["perceive", "transfer", "grasp_gate"]}')


def test_staged_steer_prompt_limits_code_to_current_stage():
    from capx.envs.trial_agilex_ensemble import _build_steer_stage_prompt

    text = _build_steer_stage_prompt("grasp_gate")[0]["content"][0]["text"]
    assert "Subtask id: grasp_gate" in text
    assert "Only gripper commands" in text
    assert "false, missing, or non-boolean" in text
    assert "future actions" in text


def test_failed_grasp_gate_cannot_authorize_transfer():
    from capx.envs.trial_agilex_ensemble import _stage_succeeded

    env = type("FakeEnv", (), {"_exec_globals": {"RESULT": {"grasp_verified": False}}})()
    clean_info = {"sandbox_rc": 0, "stderr": ""}
    assert not _stage_succeeded("grasp_gate", clean_info, env)

    env._exec_globals["RESULT"] = {"grasp_verified": True}
    assert _stage_succeeded("grasp_gate", clean_info, env)


def test_final_stage_requires_explicit_visual_success():
    from capx.envs.trial_agilex_ensemble import _stage_succeeded

    env = type("FakeEnv", (), {"_exec_globals": {"RESULT": {"success": False}}})()
    clean_info = {"sandbox_rc": 0, "stderr": ""}
    assert not _stage_succeeded("verify", clean_info, env)
    env._exec_globals["RESULT"] = {"success": True}
    assert _stage_succeeded("verify", clean_info, env)


def test_r1_staged_plan_and_grasp_gate_are_not_skippable():
    from capx.envs.trial_r1lite_staged import R1_STAGES, _stage_ok, parse_r1_stage_plan

    plan = {"stages": [{"id": stage} for stage in R1_STAGES]}
    assert parse_r1_stage_plan(__import__("json").dumps(plan)) == R1_STAGES

    with pytest.raises(ValueError, match="expected fixed R1 stage order"):
        parse_r1_stage_plan('{"stages": ["perceive", "transfer", "verify"]}')

    env = type("FakeEnv", (), {"_exec_globals": {"RESULT": {"grasp_verified": False}}})()
    assert not _stage_ok("grasp_gate", {"sandbox_rc": 0, "stderr": ""}, env)
    env._exec_globals["RESULT"] = {"grasp_verified": True}
    assert _stage_ok("grasp_gate", {"sandbox_rc": 0, "stderr": ""}, env)


def test_r1_deterministic_harness_keeps_motion_out_of_perception_retry():
    from capx.envs.trial_r1lite_staged import R1_STEER, _validate_stage_code

    assert "no motion" in R1_STEER["perceive"]
    with pytest.raises(ValueError, match="crossed safety boundary"):
        _validate_stage_code("perceive", "goto_pose(grasp_pos, grasp_quat)")
    with pytest.raises(ValueError, match="crossed safety boundary"):
        _validate_stage_code("verify", "close_gripper()")


def _feedback_only_r1():
    from capx.envs.simulators.r1lite_real import R1LiteRealLowLevel

    r1 = R1LiteRealLowLevel.__new__(R1LiteRealLowLevel)
    r1.publish_rate_hz = 20.0
    r1.command_duration_s = 0.0
    r1.feedback_timeout_s = 0.1
    r1.gripper_name = "right_gripper"
    r1.gripper_feedback_tolerance = 5.0
    r1.gripper_min_response = 2.0
    r1.arm_min_motion_rad = 0.005
    r1.arm_motion_timeout_s = 0.0
    r1._last_action = {}
    r1._gripper_pub = type(
        "Pub",
        (),
        {
            "publish": lambda self, msg: None,
            "get_subscription_count": lambda self: 1,
        },
    )()
    r1._make_joint_state = lambda _name, _position: object()
    r1.ensure_ee_control_enabled = lambda: None
    return r1


def test_r1_gripper_feedback_accepts_target_and_rejects_no_response():
    r1 = _feedback_only_r1()
    samples = iter([0.0, 98.0])
    r1.get_gripper_position = lambda: next(samples)
    assert r1.command_gripper(100.0) == pytest.approx(98.0)

    samples = iter([50.0, 50.0])
    r1.get_gripper_position = lambda: next(samples)
    with pytest.raises(RuntimeError, match="no valid feedback response"):
        r1.command_gripper(0.0, allow_contact=True)


def test_r1_arm_feedback_requires_measurable_joint_change():
    r1 = _feedback_only_r1()
    r1.get_arm_joints = lambda: np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    details = r1.verify_arm_motion(np.zeros(6), context="test")
    assert details["max_joint_delta"] == pytest.approx(0.01)

    r1.get_arm_joints = lambda: np.zeros(6)
    with pytest.raises(RuntimeError, match="ignored or overridden"):
        r1.verify_arm_motion(np.zeros(6), context="test")


def test_r1_wrist_gate_rejects_missing_rgb():
    from capx.integrations.r1lite.control import R1LiteControlApi
    from capx.envs.trial_r1lite_staged import _require_wrist_tennis

    api = R1LiteControlApi.__new__(R1LiteControlApi)
    api._r1 = type(
        "FakeR1",
        (),
        {"get_wrist_rgbd": lambda self, require_depth=False: {"rgb": None}},
    )()
    ok, details = api.check_wrist_tennis()
    assert not ok
    assert details["reason"] == "missing_wrist_rgb"

    rejecting_api = type(
        "RejectingApi",
        (),
        {"check_wrist_tennis": lambda self, **kwargs: (False, {"reason": "missing_wrist_rgb"})},
    )()
    with pytest.raises(RuntimeError, match="wrist gate rejected"):
        _require_wrist_tennis(rejecting_api, "pre_grasp", {})


def test_r1_grasp_gate_uses_measured_empty_close_baseline():
    from capx.integrations.r1lite.control import R1LiteControlApi

    api = R1LiteControlApi.__new__(R1LiteControlApi)
    api._r1 = type(
        "FakeR1",
        (),
        {
            "gripper_close_position": 0.0,
            "gripper_empty_close_feedback": 5.5,
            "get_gripper_position": lambda self: 5.4,
        },
    )()
    ok, details = api.verify_grasp(min_gap_above_close=2.0, save_debug=False)
    assert not ok
    assert details["threshold"] == pytest.approx(7.5)

    api._r1.get_gripper_position = lambda: 10.0
    ok, _ = api.verify_grasp(min_gap_above_close=2.0, save_debug=False)
    assert ok


def test_r1_config_uses_vendor_gripper_scale():
    from pathlib import Path

    config = Path("env_configs/r1lite/r1lite_tennis_ball_in_box.yaml").read_text()
    assert "gripper_open_position: 100.0" in config
    assert "gripper_close_position: 0.0" in config
    assert "arm_min_motion_rad: 0.005" in config
