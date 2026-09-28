#!/usr/bin/env python3
"""Freeze the v283 focused clean-IK anchor."""
import hashlib, json, re, time
from pathlib import Path

ROOT=Path('/project/peilab/qshou/cap-x-agilex/outputs/libero_g17_fair_fg_v283_tool_pi_20260819')
JOB='523797'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'manifests/SOURCE_SHA256.json'
summary=ROOT/f'arms/v283/task-00/rep-01/SUMMARY_job-{JOB}.json'
log=ROOT/f'logs/libero-g17-v283-tool-pi-t00-r01-{JOB}.out'
audit=ROOT/'evidence/v272_trial2_tool_pi_ik.json'
for p in (source,summary,log,audit):
    if not p.is_file(): raise SystemExit(f'missing {p}')
sdata=json.loads(source.read_text())
for raw,want in sdata['files'].items():
    p=Path(raw); got=sha(p)
    if got!=want: raise SystemExit(f'source drift {p}: {got} != {want}')
d=json.loads(summary.read_text()); outcomes=d.get('treatment_outcomes',{})
if d.get('validity')!='valid_prefix_arm' or sum(int(x['completed']) for x in outcomes.values())!=2:
    raise SystemExit('v283 focused is not valid 2/2')
text=log.read_text(errors='replace')
rows=[float(x) for x in re.findall(r'candidate_dq=([0-9.]+)',text)]
times=[float(x) for x in re.findall(r'Trial [12] took ([0-9.]+) seconds',text)]
if not rows or max(rows)>1.25: raise SystemExit(f'v283 branch bound failed: {max(rows)}')
for forbidden in ('goto_pose stopped','rollback=True','measured_joint_jump'):
    if forbidden in text: raise SystemExit(f'forbidden live behavior: {forbidden}')
data={
 'schema':'capx_libero_g17_v283_focused_frozen_anchor_v1','status':'frozen_read_only',
 'score':{'successes':2,'trials':2},'job_id':JOB,'task_id':0,'replicate':1,
 'mechanical':{'max_candidate_arm_delta_rad':max(rows),'trial_seconds':times,
   'cartesian_waypoints':0,'joint_delta_gate':0,'terminal_residual_gate':0,
   'executed_neutral_route':0,'rollback':0,'native_commands_per_stage':1},
 'artifacts':{str(p):sha(p) for p in (source,summary,log,audit)},
 'boundary':'focused task0/rep1 clean-motion anchor; K=5 and held-out remain unvalidated',
 'frozen_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
out=ROOT/'manifests/FROZEN_FOCUSED_ANCHOR.json'; out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
print(json.dumps(data,indent=2,sort_keys=True))
