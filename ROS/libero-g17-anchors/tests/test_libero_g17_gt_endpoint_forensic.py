from types import SimpleNamespace

import numpy as np

from capx.integrations.franka import gt_endpoint_forensic as forensic


class _FakeSim:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            qpos=np.zeros(10, dtype=np.float64),
            xquat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            xpos=np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
            ncon=0,
            contact=[],
        )
        self.model = SimpleNamespace(
            geom_id2name=lambda index: {
                1: "robot0_link6_collision",
                2: "wooden_cabinet_1_g15",
            }[index]
        )

    def get_state(self):
        return self.data.qpos.copy()

    def set_state(self, state) -> None:
        self.data.qpos[:] = state

    def forward(self) -> None:
        if np.max(np.abs(self.data.qpos[:7])) > 0.5:
            self.data.ncon = 1
            self.data.contact = [
                SimpleNamespace(
                    geom1=1,
                    geom2=2,
                    dist=-0.012,
                    pos=np.array([0.7, -0.1, 0.2]),
                )
            ]
        else:
            self.data.ncon = 0
            self.data.contact = []


def test_endpoint_forensic_records_contact_and_restores_state(monkeypatch) -> None:
    sim = _FakeSim()
    env = SimpleNamespace(
        handle=SimpleNamespace(env=SimpleNamespace(sim=sim)),
        _panda_joint_qpos_addrs=list(range(7)),
        base_link_idx=0,
        gripper_link_idx=1,
    )
    api = SimpleNamespace(_env=env, _capx_v293_active_object="akita_black_bowl_1")
    monkeypatch.setattr(
        forensic,
        "_pyroki_hand_pose",
        lambda _api, _cfg: (
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    report = forensic.audit_endpoint(api, np.ones(8))
    assert report["cabinet_contact_count"] == 1
    assert report["target_contact_count"] == 0
    assert report["contacts"][0]["distance_m"] == -0.012
    assert report["endpoint_joint_error_max_rad"] == 0.0
    assert report["restored_qpos_error_max"] == 0.0
    assert report["fk_crosscheck"]["position_error_m"] == 0.0
    assert report["fk_crosscheck"]["orientation_error_rad"] == 0.0
    np.testing.assert_array_equal(sim.data.qpos, np.zeros(10))


def test_forensic_is_explicitly_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CAPX_LIBERO_G17_V300A_GT_ENDPOINT_FORENSIC", raising=False)
    assert not forensic.enabled()
    monkeypatch.setenv("CAPX_LIBERO_G17_V300A_GT_ENDPOINT_FORENSIC", "1")
    assert forensic.enabled()
