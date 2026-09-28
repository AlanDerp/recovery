from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import perceptual_swept_path as swept


def test_voxel_sample_is_deterministic_and_fixed_shape() -> None:
    points = np.array(
        [[0.10, 0.10, 0.10], [0.101, 0.101, 0.101], [0.30, 0.20, 0.10]]
    )
    first, n_first = swept._voxel_sample(points, voxel_m=0.01, count=5)
    second, n_second = swept._voxel_sample(points, voxel_m=0.01, count=5)
    assert first.shape == (5, 3)
    assert n_first == n_second == 2
    np.testing.assert_array_equal(first, second)


def test_candidate_audit_reports_link_and_never_changes_behavior(monkeypatch) -> None:
    names = tuple(f"panda_link{i}" for i in range(9)) + (
        "panda_hand",
        "panda_hand_tcp",
        "panda_leftfinger",
        "panda_rightfinger",
    )
    api = SimpleNamespace(_robot_coll=SimpleNamespace(link_names=names))
    scene = swept.PerceptualScene(
        scene_points=np.zeros((swept._SCENE_POINTS, 3)),
        scene_valid=2,
        target_points=np.zeros((swept._TARGET_POINTS, 3)),
        target_valid=2,
        input_point_count=10,
        rejected_robot_points=3,
        rejected_target_points=2,
    )
    calls = 0

    def distances(_api, configs, points, *, radius_m):
        nonlocal calls
        calls += 1
        out = np.ones((len(configs), len(names), len(points)), dtype=np.float64)
        if calls == 1:
            out[7, names.index("panda_link6"), 0] = -0.01
        return out

    monkeypatch.setattr(swept, "_distances", distances)
    report = swept.audit_candidate_path(api, scene, np.zeros(8), np.ones(8))
    assert report["behavior_effect"] == "audit_only"
    assert report["scene_first_predicted_collision_sample"] == 7
    assert report["scene_closest_link"] == "panda_link6"
    assert report["scene_arm_closest_link"] == "panda_link6"
    assert report["scene_arm_first_predicted_collision_sample"] == 7
    assert report["predicted_arm_obstacle_path_clear"] is False
    assert report["predicted_native_path_clear"] is False
