# LIBERO G17 Exact-Rerun Fair Contrast: v2.5 vs v2.6

## Goal

Align Layer-1 **completeness** with **effectiveness** using **exact replay**
as the primary method (not live K-rate).

Unique delta under test:

| Arm | Env | Schema |
|-----|-----|--------|
| `v25` | `CAPX_LIBERO_G17_FG=1`, `CAPX_LIBERO_G17_WORKGAP_ME=1` | `…-v2.5-workgap-me` |
| `v26` | `CAPX_LIBERO_G17_FG=1`, `WORKGAP_ME=0` | `…-v2.6-success-witness` |

Shared freeze: M3 fingerprints, `code_prefix_replay_v1` π0, same two inits,
`GLOBAL_SEED=20260718`, temp=0, workers=1.

## Why not export from 519248 / 519358

Those audits store `response_content_sha256` only (no `response_content`).
Saved `all_responses` student reconstruction does **not** match audited hashes
(0/20 match on 519358). Fail-closed export is impossible without a new
**capture** pass using `--audit-raw-content`.

## Pipeline

1. **Capture (live, once per arm)**  
   `sbatch_smoke_libero_g17_exact_25_26_capture.sh`  
   Proxy flags: `--audit-raw-content` (required), optional `--audit-raw-request`.

2. **Export**  
   `export_request_hash_replay.py --run-root …` → `capx_request_hash_replay_v1`  
   Fail-closed on any missing content.

3. **Exact-rerun**  
   Same code + schema + π0; proxy:
   `--request-hash-replay-file … --require-all-request-hash-replay`
   `--enable-normalized-replay` (image URL redaction only).  
   Zero live Qwen hits for audited chat calls.

4. **Compare**  
   Registered capture outcomes must equal exact-rerun outcomes per arm.  
   Then contrast v25 vs v26 on the **re-proven** trajectories (t1/t2, witness,
   workgap notes). Live K is not the claim.

## Claim rule

A v2.5↔v2.6 gain/completeness claim is valid only if both arms pass exact-rerun
gates. Cross-arm outcome differences are interpreted as schema effects under
shared π0, not as remote sampling noise.
