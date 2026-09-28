#!/usr/bin/env python3
"""Return a stopped pipeline-eight grasp approach to its bound initial joints."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capx.integrations.r1lite.bimanual_runtime import (  # noqa: E402
    BimanualRuntime,
    BimanualRuntimeConfig,
)
from capx.integrations.r1lite.execution_bundle import (  # noqa: E402
    load_eighth_execution_bundle,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--open-grippers-first", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_sha = sha256_file(args.plan)
    bundle = load_eighth_execution_bundle(args.bundle, plan_sha256=plan_sha)
    initial = {
        side: np.asarray(
            bundle["initial_state"][f"{side}_joints_rad"],
            dtype=float,
        )
        for side in ("left", "right")
    }
    route = [
        {
            "id": "bundle_initial",
            "left": initial["left"],
            "right": initial["right"],
        }
    ]
    for stage_id in ("grasp", "lift", "translate_center"):
        stage = next(
            stage for stage in bundle["stages"] if stage["id"] == stage_id
        )
        route.extend(
            {
                "id": f"{stage_id}_{index:02d}",
                "left": np.asarray(waypoint["left"]["joints_rad"], dtype=float),
                "right": np.asarray(waypoint["right"]["joints_rad"], dtype=float),
            }
            for index, waypoint in enumerate(stage["waypoints"], start=1)
        )
    audit = {
        "schema": "capx.r1lite.eighth_return_to_bundle_initial.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "plan_sha256": plan_sha,
        "bundle_sha256": sha256_file(args.bundle),
        "arrival_soft_threshold_rad": 0.02,
        "passed": False,
        "events": [],
    }
    controller = BimanualRuntime(
        config=BimanualRuntimeConfig(
            arrival_timeout_s=5.0,
            arrival_soft_threshold_rad=0.02,
            arrival_velocity_threshold_rad_s=0.10,
            arrival_settle_time_s=0.15,
        )
    )
    try:
        controller.start()
        controller.wait_until_ready(require_tool_pose=True)
        if args.open_grippers_first:
            controller.command_grippers(controller.config.gripper_open_position)
            audit["events"].append({"grippers_opened_before_return": True})
        current = {
            side: np.asarray(value, dtype=float)
            for side, value in controller.get_feedback_pair().items()
        }
        distances = [
            max(
                float(np.max(np.abs(current["left"] - item["left"]))),
                float(np.max(np.abs(current["right"] - item["right"]))),
            )
            for item in route
        ]
        nearest = int(np.argmin(distances))
        audit["nearest_route_point"] = route[nearest]["id"]
        audit["nearest_route_error_rad"] = distances[nearest]
        if distances[nearest] > 0.25:
            raise RuntimeError(
                "current joints are not near the certified initial/grasp route: "
                f"{distances[nearest]:.6f}rad"
            )
        return_targets = (
            list(reversed(route[:nearest]))
            if nearest > 0
            else [route[0]]
        )
        for item in return_targets:
            controller.execute_joint_motion(
                item["left"],
                item["right"],
                progress_steps=20,
            )
            audit["events"].append({"returned_to": item["id"]})
        final = {
            side: np.asarray(value, dtype=float)
            for side, value in controller.get_feedback_pair().items()
        }
        final_error = max(
            float(np.max(np.abs(final["left"] - initial["left"]))),
            float(np.max(np.abs(final["right"] - initial["right"]))),
        )
        audit["final_initial_joint_error_rad"] = final_error
        if final_error > 0.02:
            raise RuntimeError(
                f"bundle initial return error exceeds 0.02rad: {final_error}"
            )
        audit["passed"] = True
    except Exception as exc:
        audit["error"] = repr(exc)
        try:
            controller.hold()
        except Exception:
            pass
        raise
    finally:
        controller.close()
        audit["joint_arrival_trace"] = controller.joint_arrival_trace
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
