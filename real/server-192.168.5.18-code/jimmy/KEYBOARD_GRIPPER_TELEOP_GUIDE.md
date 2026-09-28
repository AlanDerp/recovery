# Galaxea R1 Lite Keyboard Gripper Guide

This guide starts the custom SSH keyboard gripper controller while retaining
Gello arm teleoperation. It supports either a dedicated foreground terminal or
direct keys routed to a background tmux window while the collection window
stays focused. It replaces Bluetooth gripper input for the session; it does not
replace arm, torso, chassis, or data-collection control.

The deployed files are:

```text
/home/nvidia/jimmy/keyboard_gripper_teleop.py
/home/nvidia/jimmy/start_keyboard_gripper_teleop.sh
/home/nvidia/jimmy/r1litet_homogeneous_keyboard_gripper.launch.py
/home/nvidia/jimmy/start_remapped_homogeneous_teleop.sh
/home/nvidia/jimmy/keyboard_gripper_mode.sh
/home/nvidia/jimmy/tmux_gripper_hotkeys.sh
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh
/home/nvidia/jimmy/KEYBOARD_GRIPPER_TELEOP_GUIDE.md
```

## Key mapping

| Key | Action |
| --- | --- |
| `Q` | Open left gripper |
| `E` | Close left gripper |
| `I` | Open right gripper |
| `P` | Close right gripper |
| `0` | Cancel active publish bursts without changing position |
| `Esc` or `Ctrl-C` | Exit without sending a new position |
| `=` | Enable/disable direct keys in the isolated tmux workspace |

Uppercase and lowercase letters are accepted. In the ordinary foreground
workflow, the controller terminal must have focus. In the isolated tmux
workflow, the `collect` window may stay focused: when direct keys are enabled,
tmux intercepts `Q/E/I/P/0` and forwards them to the background `gripper`
window.

## Safety and control ownership

Opening or closing a gripper is physical robot motion.

Before live use:

1. Assign one operator.
2. Clear both grippers and the surrounding workspace.
3. Keep the emergency stop accessible.
4. Confirm normal R1 startup is complete.
5. Confirm no other person is using Gello, Bluetooth control, or the R1
   `r1lite_teleop` tmux session.
6. Stop or omit the separate Bluetooth `joy_pub` session.
7. Switch the R1 homogeneous controller into the documented gripper-remapped
   mode before starting the keyboard publisher.

Do not run Bluetooth and keyboard gripper publishers together. ROS 2 does not
provide deterministic priority between two publishers commanding the same
topic. The keyboard controller refuses to arm when it discovers an existing
gripper publisher and stops publishing if a competing publisher later appears.

Stopping `joy_pub` may also remove other Bluetooth functions, including any
Bluetooth chassis command. If those controls are needed simultaneously, do not
use this mode until a gripper-specific input filter is implemented.

## Step 1 — Start the robot and Gello arm teleoperation

Follow the normal startup procedure in
`GALAXEA_R1_LITE_DATA_COLLECTION_GUIDE.md`.

On the LubanCat, run the established `gello_teleop` session. Do not start the
`joy_pub` session for keyboard-gripper mode.

If `joy_pub` is already running, first confirm no one else is using it. Then
stop only that tmux session on the LubanCat:

```bash
tmux has-session -t joy_pub 2>/dev/null && tmux kill-session -t joy_pub
```

The custom keyboard launcher never kills that shared session automatically.

## Step 2 — Switch the R1 controller to keyboard-gripper mode

Live inspection on 2026-08-14 established that the R1
`homogeneous_teleoperation` node publishes both gripper target topics even when
the separate LubanCat `joy_pub` session is absent. Therefore stopping
`joy_pub` alone is insufficient.

In a separate R1 SSH terminal, inspect the current mode:

```bash
/home/nvidia/jimmy/keyboard_gripper_mode.sh status
```

Expected before switching:

```text
tmux session: present (r1lite_teleop), attached=0
teleoperation mode: vendor/Bluetooth gripper
```

When the workspace and control-ownership checks are complete, switch modes:

```bash
/home/nvidia/jimmy/keyboard_gripper_mode.sh start
```

This command interrupts only the known homogeneous teleoperation process in
the existing `r1lite_teleop` tmux pane and starts the same vendor controller
through a custom launch wrapper. Only these two outputs are remapped:

```text
/motion_target/target_position_gripper_left
  -> /jimmy/disabled_homogeneous/gripper_left
/motion_target/target_position_gripper_right
  -> /jimmy/disabled_homogeneous/gripper_right
```

Arm, torso, chassis, parameter, and collection interfaces keep their vendor
names. The switch script refuses an attached/unknown session, verifies the
replacement process, and automatically restores the vendor launch if the real
gripper topics do not reach zero publishers. It sends no gripper position.

Expected success:

```text
Keyboard-gripper remap active; real gripper topics have zero publishers.
```

## Step 3 — Verify gripper command ownership on the R1

Open a separate R1 SSH terminal and load the usual environment:

```bash
export ROS_DOMAIN_ID=01
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/galaxea/find_server/super_client_configuration_file.xml

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash
```

Inspect both topics without publishing:

```bash
ros2 topic info --no-daemon --spin-time 5 -v \
  /motion_target/target_position_gripper_left
```

```bash
ros2 topic info --no-daemon --spin-time 5 -v \
  /motion_target/target_position_gripper_right
```

After the mode switch and before starting the keyboard node, each topic should
have:

- message type `sensor_msgs/msg/JointState`;
- at least one subscriber from the robot-side control stack; and
- zero publishers.

If another publisher remains, stop. Do not try to override it by repeatedly
pressing keyboard keys.

Check the required parameter service:

```bash
ros2 service type /moca_adapter/set_parameters
```

Expected type:

```text
rcl_interfaces/srv/SetParameters
```

## Step 4 — Test keyboard handling without ROS

Run this first after deployment or after changing terminal software:

```bash
/home/nvidia/jimmy/start_keyboard_gripper_teleop.sh --dry-run
```

Press `Q`, `E`, `I`, `P`, and `0`, then press `Esc`. The screen should show the
intended actions. Dry-run mode imports no ROS modules, creates no publishers or
parameter clients, changes no parameters, and cannot move the robot.

Test the background-window routing without ROS:

```bash
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh dry-run
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh attach
```

The attached `collect` window starts with a green
`GRIPPER KEYS OFF — = TO ENABLE` status. Press `=`; the status must turn red
and say `GRIPPER KEYS ACTIVE`. Press `Q/E/I/P/0`. Tmux must report each key was
forwarded, while the background controller shows the corresponding
`(display only)` action.

Press `=` again and confirm the status turns green. A plain `q` should now be
typed only at the collection shell. Clear it with `Ctrl-C`, detach with
`Ctrl-B`, then `D`, and clean up:

```bash
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh shutdown
```

## Step 5 — Start live keyboard gripper control

Only continue when the robot and operator safety checks are complete.

For keys typed into a dedicated foreground R1 SSH terminal, run:

```bash
/home/nvidia/jimmy/start_keyboard_gripper_teleop.sh
```

The launcher:

1. verifies an interactive terminal;
2. refuses to start a duplicate keyboard controller;
3. loads the R1 ROS 2 and Fast DDS environment;
4. checks `/moca_adapter/set_parameters`;
5. checks topic type, subscribers, and existing publishers;
6. enables the required end-effector control parameters; and
7. arms its two publishers without sending a gripper position.

The screen must say:

```text
Mode:        LIVE ROS — GRIPPER ONLY
ROS status:  armed
```

Each accepted key publishes only the selected gripper's absolute target at
20 Hz for one second:

- open = `100.0`;
- close = `0.0`.

The publish interval is bounded; the node does not continuously resend a
position after the burst expires.

## Step 6 — Use during data collection

### Recommended background tmux workflow

Start only after normal vendor teleoperation is running and all operator safety
checks are complete:

```bash
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh live
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh attach
```

`live` switches only the homogeneous gripper outputs to their isolated names,
creates a separate tmux server named `jimmy_gripper`, opens `collect` and
`gripper` windows, starts the controller in `gripper`, waits for `armed`, and
installs only the `=` gate. It sends no startup gripper position. Direct keys
always start off.

Run the collection workflow in the visible `collect` window:

```bash
/home/nvidia/jimmy/collect_with_reset_guide.sh
```

After the reset-reference GUI is complete and the tmux terminal is focused:

1. press `=`; confirm the status bar turns red;
2. use plain `Q/E/I/P/0` during the episode without changing windows;
3. press `=` again; confirm the status bar turns green before answering
   collection prompts; and
4. save or reject the episode normally.

When direct routing is on, those five keys do not reach the collection window.
All other keys, including `Y`, `N`, and `Enter`, continue to go there. To inspect
the controller, press `Ctrl-B`, then `1`; return with `Ctrl-B`, then `0`.

The tmux shortcuts work only while the tmux client has focus. They do not
capture keys while the reset-reference GUI or another macOS application is
focused.

### Collection lifecycle

The normal collection lifecycle remains unchanged:

- call `/start_data_collection` once;
- finish with either `/stop_data_collection` or
  `/delete_data_collection`, never both; and
- do not mix manual collection services with the interactive collector for the
  same episode.

The keyboard controller does not call any collection service and does not send
arm, torso, chassis, brake, or generic controller commands.

## Step 7 — Stop and return to Bluetooth control

For the dedicated foreground workflow, press `Esc` or `Ctrl-C` in that
terminal. The controller cancels active publish bursts, restores the terminal,
destroys its publishers, and sends no new shutdown position.

For the tmux workflow, finish/save/delete any active episode, ensure the
`collect` window is back at an idle shell, press `=` if the status is red, and
detach with `Ctrl-B`, then `D`. From the ordinary R1 SSH shell, run:

```bash
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh shutdown
```

`shutdown` refuses to proceed while the collection window has an active child
process. It disables direct keys, sends `Esc` to the controller, waits for it to
exit, restores vendor homogeneous teleoperation for a live workspace, and
removes only the isolated `jimmy_gripper` tmux session.

Verify it ended:

```bash
pgrep -af 'keyboard_gripper_[t]eleop.py' || echo "keyboard gripper stopped"
```

Recheck the two gripper topics and confirm the custom publisher is gone. Then
restore the original R1 homogeneous teleoperation launch:

```bash
/home/nvidia/jimmy/keyboard_gripper_mode.sh restore
```

Expected result:

```text
Vendor/Bluetooth gripper authority restored.
```

Only after that should the normal `joy_pub` procedure be restarted on the
LubanCat if Bluetooth control is needed again.

## Troubleshooting

### `keyboard control requires an interactive terminal`

Run the launcher directly in a foreground SSH terminal. Do not start it as a
detached process because it must read keys from the active terminal.

### `gripper controller is not healthy/armed; direct keys remain disabled`

The tmux helper could not prove that the exact controller process and its
`armed`/dry-run status are present in the background pane. It automatically
removes the direct `Q/E/I/P/0` bindings. Inspect:

```bash
/home/nvidia/jimmy/start_tmux_gripper_workspace.sh status
tmux -L jimmy_gripper capture-pane -p \
  -t jimmy_collection:gripper.0 -S -100
```

Do not manually bind or forward keys around this check.

### `a keyboard gripper controller is already running`

Inspect the existing process:

```bash
pgrep -af 'keyboard_gripper_[t]eleop.py'
```

Do not start another copy. Attach to or safely end the existing operator
session first.

### `existing publisher(s)` or `competing gripper publisher detected`

Another process is commanding one or both grippers. Keep the keyboard
controller stopped. First inspect the R1 mode:

```bash
/home/nvidia/jimmy/keyboard_gripper_mode.sh status
```

If it reports `vendor/Bluetooth gripper`, run the Step 2 mode switch after
confirming control ownership. If it already reports `keyboard-gripper remap`,
inspect the endpoints:

```bash
ros2 topic info --no-daemon --spin-time 5 -v \
  /motion_target/target_position_gripper_left
```

```bash
ros2 topic info --no-daemon --spin-time 5 -v \
  /motion_target/target_position_gripper_right
```

Do not bypass this safety check.

### `no subscriber`

The robot-side controller is not ready. Confirm normal robot startup and the
R1 control sessions. The keyboard node deliberately refuses to publish into an
unverified control path.

### `missing service /moca_adapter/set_parameters`

The adapter is not ready or DDS discovery is incomplete. Verify the standard
ROS environment and normal robot startup. Do not replace the missing service
with hard-coded assumptions.

## Validation status

As of 2026-08-18:

- local Python compile and shell syntax checks passed;
- fourteen hardware-free unit tests passed;
- the interactive dry-run key sequence passed locally;
- R1 Python compile and shell syntax checks passed;
- the deployed interactive R1 dry-run passed for `Q`, `E`, `I`, `P`, `0`, and
  `Esc` without creating ROS entities; and
- active-stack inspection found one `homogeneous_teleoperation` publisher on
  each real gripper topic, two compatible subscribers, and the required
  `/moca_adapter/set_parameters` service;
- the custom homogeneous launch, mode switch, and rollback scripts passed
  Python/shell syntax, launch-argument, checksum, and non-mutating status
  validation on the R1;
- with the robot stack off, normal mode correctly failed closed on the missing
  `/moca_adapter/set_parameters` service and missing gripper subscribers before
  creating gripper publishers or changing parameters;
- no vendor file was modified; and
- the guarded live mode switch passed with an operator beside the robot: both
  real gripper topics reached zero publishers before the keyboard node armed;
- one lowercase `q` command published the bounded left-open request, left
  gripper feedback reached `97.526`, and feedback velocity was effectively
  stationary after the burst;
- `Esc` ended the keyboard controller without a shutdown target, the vendor
  homogeneous launch was restored, and both real gripper topics returned to
  one publisher and two subscribers; and
- the isolated tmux dry-run passed on the R1 without importing ROS: real client
  `=`, lowercase and uppercase direct routing, `Q/E/I/P/0` mapping, green/red
  status, collection-pane key swallowing, disabled passthrough, controller-loss
  refusal, and cleanup were verified; and
- the isolated tmux live workflow passed in guarded operator windows: live
  startup armed without a target, `=` enabled/disabled direct routing, `E`
  moved left feedback from `97.659` to `5.612`, `Q` returned it to `97.793`,
  `P` moved right feedback from `98.258` to `5.068`, and `I` returned it to
  `98.362`; all reported post-motion velocity magnitude `0.012` or less;
- `0` was sent during an active left-open burst and changed the controller to
  `CANCEL — no new target sent` with `Active: none`; shutdown then restored one
  vendor publisher on each real gripper topic; and
- concurrent collection, MCAP gripper-topic evidence, and direct-key latency
  under camera/recorder load remain unverified. The
  auxiliary best-effort target observer timed out in the first live trial, so
  the live tmux evidence is the controller's bounded-action status plus
  independent gripper feedback rather than a captured target message.

Continue physical validation only with an operator beside the robot, one
gripper and one key at a time.
