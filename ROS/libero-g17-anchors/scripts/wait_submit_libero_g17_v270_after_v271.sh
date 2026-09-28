#!/usr/bin/env bash
# Wait outside Slurm until the v271 batch releases enough submit quota, then
# submit exactly one v270 K=5 batch.  A lock prevents duplicate waiters.
set -u

G17=/home/qshou/capx_g17_observation_integrity
SUBMIT="$G17/scripts/submit_libero_g17_fair_fg_v270_live_isolation_kseed.sh"
STATE="$G17/outputs/v270_live_isolation_waiter"
LOCK="$STATE/waiter.lock"
LOG="$STATE/waiter.log"
PIDFILE="$STATE/waiter.pid"

mkdir -p "$STATE"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another v270 waiter already owns $LOCK" >&2
  exit 2
fi

echo "$$" >"$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
exec >>"$LOG" 2>&1

echo "[$(date -Is)] waiter_start pid=$$"
while true; do
  queue=$(squeue -u qshou -h -o '%j|%T' 2>&1)
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[$(date -Is)] squeue_error rc=$rc output=$queue"
    sleep 60
    continue
  fi

  total=$(printf '%s\n' "$queue" | awk 'NF {n++} END {print n+0}')
  v271=$(printf '%s\n' "$queue" | awk -F'|' '$1 ~ /^libero-g17-v271-/ {n++} END {print n+0}')
  echo "[$(date -Is)] waiting total_jobs=$total v271_jobs=$v271"

  # User submit cap is ten. Require the complete v271 batch to leave and five
  # free cells so the K=5 loop cannot be partially accepted under normal use.
  if [[ "$v271" -eq 0 && "$total" -le 5 ]]; then
    echo "[$(date -Is)] quota_ready; submitting v270 K=5"
    output=$(K=5 "$SUBMIT" 2>&1)
    submit_rc=$?
    printf '%s\n' "$output"
    if [[ "$submit_rc" -eq 0 ]]; then
      echo "[$(date -Is)] submit_complete"
      exit 0
    fi

    # Never retry blindly after a partial acceptance: duplicate replicate jobs
    # would invalidate the registered K=5 cell.
    active_v270=$(squeue -u qshou -h -o '%j' | awk '$1 ~ /^libero-g17-v270-live-/ {n++} END {print n+0}')
    if [[ "$active_v270" -gt 0 ]]; then
      echo "[$(date -Is)] partial_submit_detected active_v270=$active_v270; stopping"
      exit 3
    fi
    echo "[$(date -Is)] submit_rejected_without_partial_jobs rc=$submit_rc; retrying"
  fi
  sleep 60
done
