# R1 Lite head-camera distortion plan

## Safety and scope

- Do not execute chassis, torso, arm, gripper, teleoperation, odometry, or pose-control commands.
- Do not modify the installed Galaxea package code.
- Keep experimental launchers and processing code under `/home/nvidia/jimmy`.
- Preserve the original data-collection topic set and message content; replace only head image pixels and rectified head `CameraInfo`.
- Keep a rollback snapshot before testing.

## Company information used

- Official stereo calibration: `/opt/galaxea/body/stereo_ros.yaml`
- Whole-robot calibration: `/opt/galaxea/body/robot_calibration.json`
- Head calibration resolution: 1920x1536
- Official stereo RMS: 0.771 pixels
- The motion-heavy bias/extrinsic collection instructions are reference only and are not required because the calibrated output files already exist.

## Candidate solutions

1. **Native V4L2 + NVIDIA VPI CUDA + JPEG (selected)**
   - Captures `/dev/video0` and `/dev/video1` before JPEG compression.
   - Uses official K/D/R/P maps.
   - Avoids JPEG decoding.
   - Keeps original compressed ROS topic types and 1920x1536 resolution.
   - Measured prototype: about 26.7 FPS per camera.

2. **Lower-resolution native VPI**
   - Same pipeline with 1280x1024 or 960x768 output.
   - Expected to approach the camera limit with less storage and compute.
   - Use only if full-resolution performance under the complete robot session is insufficient.

3. **Compressed-topic VPI**
   - Uses GPU remap but still requires JPEG decode and encode.
   - Safer integration but retains avoidable latency.

4. **Offline MCAP correction**
   - Proven to preserve all 30 FPS timestamps and messages.
   - Remains the rollback path if native live publication is unstable.

5. **Vendor `signal_camera_node` modification**
   - Best theoretical integration: insert VPI before `NvJPEGEncoder`.
   - Not selected because vendor source is absent and installed package code must remain unchanged.

## Acceptance criteria

- No robot motion.
- Both corrected head streams are visually rectified.
- Full resolution preferred: 1920x1536.
- Target at least 20 FPS per head camera under the full stationary recording session.
- Both wrist RGB streams remain present near their original rate.
- All original non-camera topics and counts remain structurally unchanged.
- Corrected packet has a distortion-fixed marker and rectified head CameraInfo.
- A four-camera MP4 and verified MCAP are stored under `jimmy` and uploaded to the requested Hugging Face dataset.

## Rollback

1. Stop `native_head_rectifier.py`.
2. Restart the unchanged vendor camera/session.
3. If needed, consult `camera_distortion_backup_20260726_before_company_calibration/README.md`.
