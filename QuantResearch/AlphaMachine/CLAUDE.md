# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Alpha-mining research for the WorldQuant BRAIN platform. Everything revolves around the BRAIN REST API (`api.worldquantbrain.com`): fetching datafields, assembling alpha expressions in the FastExpr language, submitting simulations, and filtering results by Sharpe/fitness until they pass submission checks. Comments and notebook markdown are largely in Chinese; the project is bilingual.

There is no build system, linter, test suite, or requirements file. The interpreter is a miniforge Python 3.13; `requests`, `pandas`, `jupyter`, `openai`, and `python-dotenv` are expected to be installed.

## Layout and the duplicated library

- Root (`machine_lib.py`, `Alpha Machine.ipynb`, `operators.ipynb`) — active working area.
- `2_QuantResearch/` — earlier copy of the same library/notebook, plus LLM-agent experiments in `src/` and tutorial scripts (`world1/2/3.py`, `weight.py` — scratch/learning code).

`machine_lib.py` exists in BOTH places and the copies have diverged:

- `2_QuantResearch/machine_lib.py` has `_load_credentials()` — reads `API_USERNAME`/`API_PASSWORD` env vars, falling back to a `brain.txt` JSON file (`["username", "password"]`).
- Root `machine_lib.py` instead has hardcoded empty username/password inside `login()`, yet root `operators.ipynb` calls `mlb._load_credentials()`, which doesn't exist in the root copy (this is why root notebook runs show `INVALID_CREDENTIALS`). For credential logic, treat the `2_QuantResearch` version as canonical and port `_load_credentials()` forward rather than hardcoding.

## Credentials (sensitive)

- `2_QuantResearch/brain.txt` contains real BRAIN credentials; `2_QuantResearch/.env` (created from `.env.example`) holds `API_USERNAME`, `API_PASSWORD`, and `API_KEY` (DeepSeek/OpenAI-compatible key for the LLM agents). Don't print these or copy them into other files.
- `brain.txt` and `.env` lookups are CWD-relative — run scripts from the directory containing them (usually `2_QuantResearch/`).

## Running things

- Primary workflow: Jupyter (`jupyter lab`), running `Alpha Machine.ipynb` top to bottom. The notebook does `from machine_lib import *` from its own directory — after editing `machine_lib.py`, restart the kernel and re-import.
- LLM agents (run from `2_QuantResearch/`):
  - `python src/pro.py` — interactive: type a factor idea in natural language; DeepSeek turns it into a FastExpr expression and simulates it.
  - `python src/service.py` — long-running worker; polls a `tasks.json` file for ideas, logs to `agent_service.log`, clean shutdown on SIGINT/SIGTERM.
  - `python src/wqAgentPro.py` — batch miner: LLM produces a template with `{MATRIX}`/`{DAYS}` placeholders, assembled across all datafields and simulated concurrently via ThreadPoolExecutor.

## The alpha-mining pipeline (machine_lib.py + Alpha Machine.ipynb)

The notebook drives a staged refinement loop; each stage's survivors are the next stage's input:

1. **Fetch & preprocess** — `get_datafields()` pulls a dataset's fields; `process_datafields()` wraps each in `winsorize(ts_backfill(field, 120), std=4)` and expands VECTOR-type fields via `vec_avg`/`vec_sum`.
2. **First order** — `first_order_factory(fields, ts_ops)` crosses fields with time-series operators over day windows `[5, 22, 66, 120, 240]`.
3. **Simulate** — pair each expression with an initial decay, shuffle, chunk with `load_task_pool_single()`, run `single_simulate()` (POST to `/simulations`, poll honoring `Retry-After`, re-login + 600s sleep on failure).
4. **Filter** — `get_alphas()` pulls results above Sharpe/fitness thresholds. Quirks: it flips the expression sign when Sharpe is below the negative threshold, appends a follow-up decay scaled by turnover, and the year **2025 is hardcoded** into its date-range URLs (dates are passed as "MM-DD" strings).
5. **Prune** — `prune(recs, datafield_prefix, keep_num)` keeps the top-N alphas per underlying datafield (prefix like `anl4`, `fnd6`).
6. **Second order** — `get_group_second_order_factory()` wraps survivors in `group_neutralize`/`group_rank`/`group_zscore` across groupings (sector/industry/subindustry, cap/asset/volatility/liquidity buckets, `pv13` groups).
7. **Third order** — `trade_when_factory()` wraps in `trade_when(entry_event, alpha, exit_event)` using a library of region-specific event conditions (USA/ASI/EUR/GLB/CHN/KOR/TWN lists).
8. **Submission check** — `check_submission()` runs the platform's checks per alpha id, accumulating passers into a `gold_bag` with their self-correlation values; `view_alphas()` sorts them by Sharpe for manual submission on the website.

Sharpe/fitness thresholds rise per stage (≈1.0/0.7 → 1.3/0.8 → 1.25/1.0 at submission). `multi_simulate()` / `load_task_pool()` / `generate_sim_data()` are the multi-simulation variants (10 alphas per request) for consultant-tier accounts; single simulation is the default path. `login_hk()` is the consultant login handling biometric (persona) auth.

## API interaction conventions

- Sessions expire; the established pattern is catch → `sleep` → `login()` again, not token refresh.
- All simulation polling must respect the `Retry-After` response header.
- Simulation settings dicts (region, universe, neutralization, truncation, decay, ...) are duplicated at every call site rather than centralized — when changing a setting, check all of them (`single_simulate`, `generate_sim_data`, plus each script in `src/`).
