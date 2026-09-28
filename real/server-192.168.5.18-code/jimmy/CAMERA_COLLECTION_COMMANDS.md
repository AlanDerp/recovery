# Distortion-corrected head-camera collection

This workflow leaves all Galaxea package files unchanged. It replaces only the
running head-camera process. Wrist cameras and the original recorder topic
profile remain unchanged.

After a machine reboot, the normal vendor camera with distortion is restored
automatically. Run the switch below before each corrected collection session.

## Recommended: one-command interactive collection

After the normal robot session has started, run:

```bash
/home/nvidia/jimmy/collect_with_reset_guide.sh
```

The script starts the corrected head camera and vendor data-collection launcher
in named tmux sessions only when they are missing. It refuses to record unless
both corrected head cameras and both wrist cameras have publishers.

The local reset-guide window opens first:

- `Space` or `C`: capture a reset-reference photo.
- `Y` or `Enter`: accept the reference.
- `R`: retake the reference.
- `Q` or `Esc`: close/cancel the window.

After acceptance, a magenta-biased semi-transparent reference and yellow
reference edges are overlaid on the live corrected head-left image. Match the
colored ghost to reset objects before the next episode. Press `R` in the window
at any time to replace the reference. Full-resolution reference PNG files are
saved under `/home/nvidia/jimmy/reset_references/`.

Terminal flow:

1. `Start episode? [y/n]`: `y` starts; `n` ends the overall loop.
2. Perform the task and press Enter when finished.
3. `Accept and save? [y/n]`: `y` stops and saves; `n` calls the delete service
   while the episode is still active.

The script contains no chassis, torso, arm, gripper, IK, odometry, pose-control,
or teleoperation command.

## 1. Start the normal robot session

Use the team's normal supervised startup procedure. Wait until the Fast DDS
discovery server, wrist cameras, HDAS feedback, and data-collection service are
ready. Do not run calibration pose, odometry, IK, or teleoperation commands
unless a person is supervising the robot and those actions are intended.

## 2. Switch only the head camera

In a dedicated tmux window:

```bash
tmux new -s rectified_head
/home/nvidia/jimmy/switch_to_rectified_head.sh
```

Detach with `Ctrl-B`, then `D`. Wait about 15 seconds for the measured rates to
settle.

## 3. Verify all four cameras

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
export ROS_DOMAIN_ID=01
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/galaxea/find_server/super_client_configuration_file.xml

ros2 topic hz /hdas/camera_head/left_raw/image_raw_color/compressed
ros2 topic hz /hdas/camera_head/right_raw/image_raw_color/compressed
ros2 topic hz /hdas/camera_wrist_left/color/image_raw/compressed
ros2 topic hz /hdas/camera_wrist_right/color/image_raw/compressed
```

Run the `hz` checks one at a time and stop each with `Ctrl-C`. Expected rates
under recording load are approximately 26 FPS for each 1920 x 1536 corrected
head camera and 30 FPS for each 1280 x 720 wrist camera.

Check that the corrected head CameraInfo has zero distortion coefficients:

```bash
ros2 topic echo --once /calib/head_left/camera_info
ros2 topic echo --once /calib/head_right/camera_info
```

## 4. Record one episode

The normal data-collection launch remains unchanged. Start exactly once:

```bash
ros2 service call /start_data_collection std_srvs/srv/Trigger "{}"
```

Save normally:

```bash
ros2 service call /stop_data_collection std_srvs/srv/Trigger "{}"
```

Delete a failed active episode:

```bash
ros2 service call /delete_data_collection std_srvs/srv/Trigger "{}"
```

Do not call `/start_data_collection` twice for one episode.

## 5. Add the distortion-fixed packet marker without re-encoding

Replace `EPISODE_PATH` with the saved episode directory:

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
python3 /home/nvidia/jimmy/finalize_live_rectified_episode.py EPISODE_PATH
```

The source episode is not modified. A dated `_DISTORTION_FIXED` copy is created
under `/home/nvidia/jimmy/YYYYMMDD_DISTORTION_FIXED` with one extra
`/dataset/distortion_fix` message and a `distortion_fix.json` sidecar. All
original serialized messages are copied byte-for-byte.

## Roll back to the vendor head camera

Stop the `rectified_head` tmux process, then run in a dedicated tmux window:

```bash
/home/nvidia/jimmy/restore_vendor_head_camera.sh
```

A machine reboot also restores the normal vendor behavior because no Galaxea
package, launch, session, or configuration file was modified by this solution.

The file snapshot for manual rollback inspection is:

```text
/home/nvidia/jimmy/camera_distortion_backup_20260726_before_company_calibration
```
