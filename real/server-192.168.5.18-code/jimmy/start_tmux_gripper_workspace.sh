#!/usr/bin/env bash

# Create an isolated two-window tmux workspace:
#   collect  - normal shell for the collection workflow
#   gripper  - foreground keyboard controller in a background tmux window

set -euo pipefail

ACTION="${1:-status}"
TMUX_BIN="${TMUX_BIN:-tmux}"
SOCKET="jimmy_gripper"
SESSION="jimmy_collection"
COLLECT_TARGET="${SESSION}:collect.0"
GRIPPER_TARGET="${SESSION}:gripper.0"
HOTKEYS="/home/nvidia/jimmy/tmux_gripper_hotkeys.sh"
LAUNCHER="/home/nvidia/jimmy/start_keyboard_gripper_teleop.sh"
MODE_SWITCH="/home/nvidia/jimmy/keyboard_gripper_mode.sh"
WORKSPACE_MODE_OPTION="@jimmy_gripper_workspace_mode"

tmux_cmd() {
  "${TMUX_BIN}" -L "${SOCKET}" "$@"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

workspace_exists() {
  tmux_cmd has-session -t "${SESSION}" 2>/dev/null
}

wait_for_controller() {
  local expected="$1"
  local deadline=$((SECONDS + 15))
  local screen
  while (( SECONDS < deadline )); do
    screen="$(tmux_cmd capture-pane -p -t "${GRIPPER_TARGET}" -S -80 2>/dev/null || true)"
    if [[ "${expected}" == "dry-run" ]] \
      && grep -Eq 'Mode:[[:space:]]+DRY RUN.*NO ROS ENTITIES' <<<"${screen}"; then
      return 0
    fi
    if [[ "${expected}" == "live" ]] \
      && grep -Eq 'ROS status:[[:space:]]+armed' <<<"${screen}"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

create_windows() {
  local mode="$1"
  local command="$2"
  workspace_exists && die "isolated workspace ${SESSION} already exists"

  tmux_cmd new-session -d -s "${SESSION}" -n collect -c /home/nvidia
  tmux_cmd new-window -d -t "${SESSION}" -n gripper -c /home/nvidia "${command}"
  tmux_cmd set-option -w -t "${SESSION}:gripper" remain-on-exit on
  tmux_cmd set-option -g "${WORKSPACE_MODE_OPTION}" "${mode}"
  tmux_cmd select-window -t "${SESSION}:collect"
}

cleanup_failed_start() {
  local restore_vendor="$1"
  tmux_cmd kill-session -t "${SESSION}" 2>/dev/null || true
  if [[ "${restore_vendor}" == "yes" ]]; then
    "${MODE_SWITCH}" restore 2>/dev/null || true
  fi
}

start_dry_run() {
  if ! create_windows "dry-run" "${LAUNCHER} --dry-run"; then
    cleanup_failed_start no
    die "could not create isolated dry-run workspace"
  fi
  if ! wait_for_controller dry-run; then
    echo "ERROR: dry-run controller did not become ready" >&2
    tmux_cmd capture-pane -p -t "${GRIPPER_TARGET}" -S -100 2>/dev/null || true
    cleanup_failed_start no
    exit 2
  fi
  "${HOTKEYS}" install
  echo "Dry-run workspace ready; no ROS entities were created."
  echo "Attach with: tmux -L ${SOCKET} attach -t ${SESSION}"
  echo "Direct keys start OFF. Press = inside tmux to enable/disable them."
}

start_live() {
  workspace_exists && die "isolated workspace ${SESSION} already exists"
  local current_mode
  current_mode="$(${MODE_SWITCH} status 2>&1)"
  if ! grep -q 'teleoperation mode: vendor/Bluetooth gripper' <<<"${current_mode}"; then
    current_mode="${current_mode//$'\n'/; }"
    die "live start requires normal vendor teleoperation first; status: ${current_mode}"
  fi

  "${MODE_SWITCH}" start
  if ! create_windows "live" "${LAUNCHER}"; then
    cleanup_failed_start yes
    die "could not create isolated live workspace; vendor restore attempted"
  fi
  if ! wait_for_controller live; then
    echo "ERROR: live controller did not become armed; restoring vendor mode" >&2
    tmux_cmd capture-pane -p -t "${GRIPPER_TARGET}" -S -100 2>/dev/null || true
    cleanup_failed_start yes
    exit 2
  fi
  "${HOTKEYS}" install
  echo "Live workspace ready; controller armed without sending a position."
  echo "Attach with: tmux -L ${SOCKET} attach -t ${SESSION}"
  echo "Direct keys start OFF. Press = only when ready to operate grippers."
}

attach_workspace() {
  workspace_exists || die "isolated workspace ${SESSION} is not running"
  exec "${TMUX_BIN}" -L "${SOCKET}" attach-session -t "${SESSION}"
}

show_status() {
  if ! workspace_exists; then
    echo "workspace=stopped"
    exit 0
  fi
  echo "workspace=running"
  echo "mode=$(tmux_cmd show-options -gv "${WORKSPACE_MODE_OPTION}")"
  "${HOTKEYS}" status
  tmux_cmd list-windows -t "${SESSION}" \
    -F 'window=#{window_name} active=#{window_active} panes=#{window_panes}'
}

shutdown_workspace() {
  workspace_exists || die "isolated workspace ${SESSION} is not running"
  local collect_pid workspace_mode deadline
  collect_pid="$(tmux_cmd display-message -p -t "${COLLECT_TARGET}" '#{pane_pid}')"
  if pgrep -P "${collect_pid}" >/dev/null 2>&1; then
    die "collection window has an active child process; finish the episode first"
  fi

  workspace_mode="$(tmux_cmd show-options -gv "${WORKSPACE_MODE_OPTION}")"
  "${HOTKEYS}" disable
  if tmux_cmd display-message -p -t "${GRIPPER_TARGET}" '#{pane_dead}' 2>/dev/null \
      | grep -qx '0'; then
    tmux_cmd send-keys -t "${GRIPPER_TARGET}" Escape
  fi

  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    tmux_cmd display-message -p -t "${GRIPPER_TARGET}" '#{pane_dead}' 2>/dev/null \
      | grep -qx '1' && break
    sleep 0.1
  done

  if ! tmux_cmd display-message -p -t "${GRIPPER_TARGET}" '#{pane_dead}' 2>/dev/null \
      | grep -qx '1'; then
    die "gripper controller did not exit; workspace remains with direct keys disabled"
  fi

  if [[ "${workspace_mode}" == "live" ]]; then
    "${MODE_SWITCH}" restore
  fi
  tmux_cmd kill-session -t "${SESSION}"
  echo "Workspace stopped; direct keys disabled."
}

case "${ACTION}" in
  dry-run) start_dry_run ;;
  live) start_live ;;
  attach) attach_workspace ;;
  status) show_status ;;
  shutdown) shutdown_workspace ;;
  *)
    echo "Usage: $0 {dry-run|live|attach|status|shutdown}" >&2
    exit 64
    ;;
esac
