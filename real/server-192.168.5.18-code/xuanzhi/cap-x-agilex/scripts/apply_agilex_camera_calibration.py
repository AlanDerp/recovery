#!/usr/bin/env python3
"""Apply a saved Agilex camera calibration to CaP-X env config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agilex_camera_calib_io import (
    CALIB_TYPE_EYE_TO_HAND,
    CALIB_TYPE_POINT_CORRESPONDENCE,
    apply_calibration_to_config,
    print_yaml_matrix,
    read_T_base_camera,
    solve_point_correspondence_from_file,
    validate_sample_format,
)


def _detect_source_type(path: Path) -> str:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return "yaml"
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        calib_type = payload.get("calibration_type")
        if calib_type == CALIB_TYPE_EYE_TO_HAND:
            if "T_base_camera" in payload or "homogeneous_matrix" in payload:
                return "json_result"
            return "eye_to_hand_samples"
        if calib_type == CALIB_TYPE_POINT_CORRESPONDENCE or "samples" in payload:
            return "point_samples"
        if "T_base_camera" in payload or "homogeneous_matrix" in payload:
            return "json_result"
    if isinstance(payload, list):
        return "point_samples_legacy"
    raise ValueError(f"Unrecognized calibration file: {path}")


def resolve_transform(source_path: Path) -> tuple[str, object]:
    source_type = _detect_source_type(source_path)
    if source_type in {"yaml", "json_result"}:
        return source_type, read_T_base_camera(source_path)
    if source_type == "point_samples_legacy":
        validate_sample_format(json.loads(source_path.read_text()), CALIB_TYPE_POINT_CORRESPONDENCE, source_path)
        transform, rms, errors = solve_point_correspondence_from_file(source_path)
        return source_type, (transform, rms, errors)
    if source_type == "point_samples":
        transform, rms, errors = solve_point_correspondence_from_file(source_path)
        return source_type, (transform, rms, errors)
    if source_type == "eye_to_hand_samples":
        raise RuntimeError(
            f"{source_path} contains eye-to-hand samples only. "
            "Run eye_to_hand_calibration.py and press c, or pass T_base_camera.yaml / calibration_result.json."
        )
    raise RuntimeError(f"Unsupported source type: {source_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a saved T_base_camera into camera_pose_mats.main.")
    parser.add_argument("--from", dest="source", required=True, help="Calibration yaml/json or point samples json.")
    parser.add_argument("--write-config", required=True, help="Target CaP-X env yaml.")
    parser.add_argument(
        "--reset-graspnet-offsets",
        action="store_true",
        help="Zero graspnet_world_offset and graspnet_rotation_correction_rpy after writing the pose.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    config_path = Path(args.write_config)
    source_type, result = resolve_transform(source_path)

    if source_type in {"point_samples", "point_samples_legacy"}:
        transform, rms, errors = result  # type: ignore[misc]
        print(f"[apply] solved point correspondence from {source_path}")
        print(f"[apply] rms_error_m={rms:.5f}")
        print(f"[apply] per_sample_error_m={errors.round(5).tolist()}")
    else:
        transform = result  # type: ignore[assignment]
        print(f"[apply] loaded T_base_camera from {source_path} ({source_type})")

    print("[apply] T_base_camera / camera_pose_mats.main:")
    print(transform)
    print_yaml_matrix(transform)

    apply_calibration_to_config(
        config_path,
        transform,
        reset_graspnet_offsets=bool(args.reset_graspnet_offsets),
    )
    print(f"[apply] updated {config_path}")


if __name__ == "__main__":
    main()
