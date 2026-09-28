#!/usr/bin/env bash

# Direct-key routing for Jimmy's isolated data-collection tmux server.
# This script does not publish ROS messages. It forwards a whitelisted literal
# key only when the target pane contains a healthy dry-run or armed controller.

set -euo pipefail

ACTION="${1:-status}"
KEY="${2:-}"
TMUX_BIN="${TMUX_BIN:-tmux}"
SOCKET="jimmy_gripper"
SESSION="jimmy_collection"
TARGET="${SESSION}:gripper.0"
STATE_OPTION="@jimmy_gripper_hotkeys"
REMOTE_SELF="/home/nvidia/jimmy/tmux_gripper_hotkeys.sh"
CONTROLLER="python3 /home/nvidia/jimmy/keyboard_gripper_teleop.py"

tmux_cmd() {
  "${TMUX_BIN}" -L "${SOCKET}" "$@"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_workspace() {
  tmux_cmd has-session -t "${SESSION}" 2>/dev/null \
    || die "isolated tmux workspace ${SESSION} is not running"
  tmux_cmd display-message -p -t "${TARGET}" '#{pane_id}' >/dev/null 2>&1 \
    || die "gripper pane ${TARGET} is missing"
}

state() {
  tmux_cmd show-options -gv "${STATE_OPTION}" 2>/dev/null || echo "off"
}

notify() {
  tmux_cmd display-message -d 3000 "$1" 2>/dev/null || true
}

controller_process_present() {
  local pane_tty tty_name
  pane_tty="$(tmux_cmd display-message -p -t "${TARGET}" '#{pane_tty}')"
  tty_name="${pane_tty#/dev/}"
  ps -t "${tty_name}" -o args= 2>/dev/null \
    | grep -Fq "${CONTROLLER}"
}

controller_ready() {
  local screen
  controller_process_present || return 1
  screen="$(tmux_cmd capture-pane -p -t "${TARGET}" -S -80)"
  grep -Eq 'Mode:[[:space:]]+DRY RUN.*NO ROS ENTITIES|ROS status:[[:space:]]+armed' \
    <<<"${screen}"
}

set_status_off() {
  tmux_cmd set-option -g status on
  tmux_cmd set-option -g status-right-length 80
  tmux_cmd set-option -g status-right \
    '#[fg=black,bg=green,bold] GRIPPER KEYS OFF — = TO ENABLE #[default] %H:%M'
}

set_status_on() {
  tmux_cmd set-option -g status on
  tmux_cmd set-option -g status-right-length 80
  tmux_cmd set-option -g status-right \
    '#[fg=white,bg=red,bold] GRIPPER KEYS ACTIVE — = TO DISABLE #[default] %H:%M'
}

bind_direct_key() {
  local physical_key="$1"
  local forwarded_key="$2"
  tmux_cmd bind-key -n "${physical_key}" run-shell \
    "${REMOTE_SELF} send ${forwarded_key}"
}

disable() {
  require_workspace
  local key
  for key in q Q e E i I p P 0; do
    tmux_cmd unbind-key -n "${key}" 2>/dev/null || true
  done
  tmux_cmd set-option -g "${STATE_OPTION}" off
  set_status_off
  notify "Direct gripper keys disabled"
}

enable() {
  require_workspace
  if ! controller_ready; then
    disable
    die "gripper controller is not healthy/armed; direct keys remain disabled"
  fi

  bind_direct_key q q
  bind_direct_key Q q
  bind_direct_key e e
  bind_direct_key E e
  bind_direct_key i i
  bind_direct_key I i
  bind_direct_key p p
  bind_direct_key P p
  bind_direct_key 0 0
  tmux_cmd set-option -g "${STATE_OPTION}" on
  set_status_on
  notify "DIRECT GRIPPER KEYS ACTIVE — = disables"
}

install() {
  require_workspace
  tmux_cmd bind-key -n = run-shell "${REMOTE_SELF} toggle"
  disable
  notify "Gripper hotkeys installed; press = to enable"
}

uninstall() {
  require_workspace
  disable
  tmux_cmd unbind-key -n = 2>/dev/null || true
  notify "Gripper hotkeys removed"
}

toggle() {
  require_workspace
  if [[ "$(state)" == "on" ]]; then
    disable
  else
    enable
  fi
}

send_key() {
  require_workspace
  case "${KEY}" in
    q|e|i|p|0) ;;
    *) die "refusing non-whitelisted forwarded key: ${KEY:-<empty>}" ;;
  esac
  [[ "$(state)" == "on" ]] \
    || die "direct gripper keys are disabled"
  if ! controller_ready; then
    disable
    die "gripper controller lost readiness; direct keys were disabled"
  fi
  tmux_cmd send-keys -l -t "${TARGET}" "${KEY}"
  notify "Forwarded ${KEY^^} to background gripper controller"
}

show_status() {
  require_workspace
  echo "socket=${SOCKET}"
  echo "session=${SESSION}"
  echo "target=${TARGET}"
  echo "direct_keys=$(state)"
  if controller_ready; then
    echo "controller=ready"
  elif controller_process_present; then
    echo "controller=running_not_ready"
  else
    echo "controller=stopped"
  fi
}

case "${ACTION}" in
  install) install ;;
  enable) enable ;;
  disable) disable ;;
  toggle) toggle ;;
  send) send_key ;;
  status) show_status ;;
  uninstall) uninstall ;;
  *)
    echo "Usage: $0 {install|enable|disable|toggle|send KEY|status|uninstall}" >&2
    exit 64
    ;;
esac
