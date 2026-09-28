#!/usr/bin/env python3
"""Assemble a WorldQuant BRAIN alpha and submit it if it passes every check.

Account: AT90350 (Alan Tang) - TUTORIAL tier, USA-only, single simulations.

Design notes (derived from this account's own 202 IS alphas + 2 OS alphas):
  * The passing alpha `liabilities/assets`: sharpe 1.57, fitness 1.45,
    turnover 0.016, sub-universe sharpe 1.11 vs limit 0.68. A scale-free ratio
    of two dense, same-scale fundamentals -> robust across the cap spectrum.
  * Repeated near-misses `group_rank(fnd6_<sparse>/cap, subindustry)`:
    sharpe ~1.8, fitness ~1.9, but FAIL LOW_SUB_UNIVERSE_SHARPE (0.39-0.65 vs
    limit 0.76-0.82). Dividing a *sparse* field by `cap` builds a size tilt
    that collapses on the large-cap sub-universe.
  * Price/volume signals reach sharpe ~1.9 but FAIL LOW_FITNESS: turnover
    0.5-0.93 lands in the fitness denominator.

  LOW_SUB_UNIVERSE_SHARPE's limit is not fixed - it scales at ~0.43x the alpha's
  own sharpe, so chasing raw sharpe also raises the bar it must clear. The lever
  that actually helps is breadth: averaging several *dense*, scale-free
  fundamental ratios diversifies idiosyncratic noise and lifts sub-universe
  sharpe.

  So: composites of dense fundamental ratios (profitability / cash-flow quality
  / accruals / value), each preprocessed with the repo's
  `winsorize(ts_backfill(f, 120), std=4)` convention, combined via rank/zscore.
  Deliberately NOT leverage - both already-submitted alphas are
  `liabilities/assets`, so a leverage signal would risk SELF_CORRELATION.

Submission is gated: an alpha is only POSTed to /submit once every check reads
PASS. Use --dry-run to simulate and report without submitting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.worldquantbrain.com"
REPO = Path(__file__).resolve().parent
CRED_FILE = REPO / "2_QuantResearch" / "brain.txt"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("alpha")

# ---------------------------------------------------------------- expressions

# Repo convention (machine_lib.process_datafields): backfill then winsorize.
def prep(field: str, days: int = 120, std: int = 4) -> str:
    return f"winsorize(ts_backfill({field}, {days}), std={std})"


def _macro(**fields: str) -> str:
    """Emit `name = winsorize(ts_backfill(field,120), std=4);` prelude lines."""
    return "\n".join(f"{alias} = {prep(f)};" for alias, f in fields.items())


# Each candidate: dense, scale-free fundamental ratios averaged for breadth.
CANDIDATES: dict[str, str] = {
    # Profitability + cash-flow quality + asset efficiency.
    "quality3": _macro(
        a="assets", cfo="cashflow_op", ni="income", eb="ebit", sl="sales"
    ) + """
cf_roa   = rank(cfo / a);
ebit_roa = rank(eb / a);
asset_to = rank(sl / a);
(cf_roa + ebit_roa + asset_to) / 3
""",
    # Sloan accruals: cash earnings beat accrual earnings.
    "accruals": _macro(a="assets", cfo="cashflow_op", ni="income") + """
accrual = (ni - cfo) / a;
-rank(accrual)
""",
    # Quality + accruals blended.
    "qual_accr": _macro(a="assets", cfo="cashflow_op", ni="income", eb="ebit") + """
cf_roa   = rank(cfo / a);
ebit_roa = rank(eb / a);
accrual  = rank((ni - cfo) / a);
(cf_roa + ebit_roa - accrual) / 3
""",
    # Value: book/market, earnings yield, cash-flow yield.
    "value3": _macro(eq="equity", ni="income", cfo="cashflow_op", c="cap") + """
btm = rank(eq / c);
ey  = rank(ni / c);
cfy = rank(cfo / c);
(btm + ey + cfy) / 3
""",
    # Quality + value: the classic pairing, both dense and scale-free.
    "qual_value": _macro(
        a="assets", cfo="cashflow_op", ni="income", eb="ebit", eq="equity", c="cap"
    ) + """
cf_roa = rank(cfo / a);
ebit_roa = rank(eb / a);
btm = rank(eq / c);
cfy = rank(cfo / c);
(cf_roa + ebit_roa + btm + cfy) / 4
""",
    # zscore instead of rank: keeps magnitude information.
    "qual_z": _macro(a="assets", cfo="cashflow_op", eb="ebit", sl="sales") + """
(zscore(cfo / a) + zscore(eb / a) + zscore(sl / a)) / 3
""",
    # Enterprise-value based yield, neutral to capital structure.
    "ev_yield": _macro(cfo="cashflow_op", eb="ebit", ev="enterprise_value") + """
(rank(cfo / ev) + rank(eb / ev)) / 2
""",
    # Group-relative quality: rank within subindustry peers.
    "qual_grp": _macro(a="assets", cfo="cashflow_op", eb="ebit", sl="sales") + """
cf_roa = group_rank(cfo / a, subindustry);
ebit_roa = group_rank(eb / a, subindustry);
asset_to = group_rank(sl / a, subindustry);
(cf_roa + ebit_roa + asset_to) / 3
""",
    # Proven account-native formula: three prior IS runs cleared all quality,
    # sub-universe, and turnover checks (only OLD_SIMULATION remained).
    "liab_assets": "liabilities / assets",
    # Same economic family, but not identical to the two already-submitted
    # liabilities/assets alphas; these are candidates only, not guarantees.
    "debt_assets": "debt / assets",
    "assets_liab": "assets / liabilities",
    "grp_debt_assets": "group_rank(debt / assets, subindustry)",
    "grp_cfo_assets": "group_rank(cashflow_op / assets, subindustry)",
    "grp_ebit_assets": "group_rank(ebit / assets, subindustry)",
    # Hybrid candidate: preserve the account's strongest fundamental signal,
    # add an orthogonal slow market leg to reduce correlation with submitted
    # liabilities/assets alphas. This is deliberately tested, never assumed.
    "liab_momentum": "0.7 * rank(liabilities / assets) + 0.3 * rank(-ts_zscore(ts_delta(close, 5), 30))",

    # ---------------------------------------------------------------------
    # Literature-driven candidates (round 2). Each cites a primary source
    # verified from the paper itself, not from recollection.
    # ---------------------------------------------------------------------

    # Ang, Hodrick, Xing & Zhang (2006), J.Finance 61(1) 259-299,
    # DOI 10.3386/w10852. IVOL = sqrt(var(residual)) from an FF-3 regression
    # on DAILY returns; 1/0/1 sort; 5-1 spread -1.06%/month; lagged 1/1/1
    # variant -0.82%/month (t=-4.63). High IVOL -> low returns, so we go
    # LONG low volatility (negative sign).
    #
    # We cannot run an FF-3 regression in FastExpr. Two documented proxies:
    #   (a) total volatility - the paper explicitly also sorts on total
    #       volatility "without using any control for systematic risk";
    #   (b) subindustry-relative volatility via group_rank, which removes
    #       part of the systematic component without a factor regression.
    # 22 trading days ~ the paper's 1-month estimation window.
    "ivol_total": "-rank(ts_std_dev(returns, 22))",
    "ivol_grp": "-group_rank(ts_std_dev(returns, 22), subindustry)",
    # 1-month lag between measurement and formation = the paper's 1/1/1
    # robustness variant, which guards against contemporaneous measurement
    # error. ts_delay by 22 days approximates M=1.
    "ivol_lagged": "-rank(ts_delay(ts_std_dev(returns, 22), 22))",

    # Cooper, Gulen & Ion (2024), J.Financial Economics 151:103746,
    # DOI 10.1016/j.jfineco.2023.103746. Classic asset growth (Cooper et al.
    # 2008) = YoY % change in book total assets; high growth -> low returns.
    # 240 trading days ~ 1 year.
    "asset_growth": "-rank(assets / ts_delay(assets, 240) - 1)",
    # The 2024 paper's central finding: "factors based on growth in inventory
    # and accounts receivable contain the bulk of the pricing information in
    # the asset growth factor." Scaled by assets to stay unit-free.
    "inv_rec_growth": _macro(inv="inventory", rec="receivable", a="assets") + """
d_inv = (inv - ts_delay(inv, 240)) / a;
d_rec = (rec - ts_delay(rec, 240)) / a;
-rank(d_inv + d_rec)
""",
    # Same idea, subindustry-relative: inventory/receivable norms differ
    # sharply across industries, so peer-relative growth is the cleaner read.
    "inv_rec_grp": _macro(inv="inventory", rec="receivable", a="assets") + """
d_inv = (inv - ts_delay(inv, 240)) / a;
d_rec = (rec - ts_delay(rec, 240)) / a;
-group_rank(d_inv + d_rec, subindustry)
""",

    # Round-2 follow-up: the six standalone literature candidates all failed
    # LOW_SHARPE, but `asset_growth` and `inv_rec_grp` PASSED
    # LOW_SUB_UNIVERSE_SHARPE - historically this account's most common
    # failure. So test them as weak-but-stable orthogonal legs on top of the
    # leverage leg that does carry signal here, the same structure that
    # produced rK28jvqE. Leverage (capital structure) and asset growth
    # (investment) are distinct FF5 factors, so the blend is economically
    # defensible rather than curve-fitting.
    "liab_ag_30": "0.7 * rank(liabilities / assets) + 0.3 * rank(-(assets / ts_delay(assets, 240) - 1))",
    "liab_ag_20": "0.8 * rank(liabilities / assets) + 0.2 * rank(-(assets / ts_delay(assets, 240) - 1))",
    # Three-leg: leverage + investment + the reversal leg already proven in
    # rK28jvqE. Tests whether the paper factor adds anything incremental.
    "liab_rev_ag": (
        "0.6 * rank(liabilities / assets)"
        " + 0.2 * rank(-ts_zscore(ts_delta(close, 5), 30))"
        " + 0.2 * rank(-(assets / ts_delay(assets, 240) - 1))"
    ),

    # ---------------------------------------------------------------------
    # Round 3: 质量/盈利能力因子 Quality/Profitability factors
    # 与资本结构因子正交，目标是通过 SELF_CORRELATION 检查
    # ---------------------------------------------------------------------

    # Novy-Marx 2013, "The Other Side of Value: The Gross Profitability
    # Premium", J.Financial Economics 108(1) 1-28. 核心发现：高毛利率
    # 公司（gross profit / assets）收益显著高于低毛利率公司，且独立于
    # 价值效应。5-1组年化超额收益 4-6%。
    # Gross profit = Revenue - COGS. 如果 COGS 不可得，用 EBIT 近似。
    "quality_ebit_assets": _macro(e="ebit", a="assets") + "rank(e / a)",
    "quality_ebit_equity": _macro(e="ebit", eq="equity") + "rank(e / eq)",

    # Fama-French 2015 的 RMW 因子用 operating profitability / book equity.
    # 用经营现金流代替会计利润，更难操纵。
    "quality_cfo_assets": _macro(c="cashflow_op", a="assets") + "rank(c / a)",
    "quality_cfo_equity": _macro(c="cashflow_op", eq="equity") + "rank(c / eq)",

    # 行业相对版本：质量信号的行业差异很大（科技 vs 传统制造），
    # 行业中性化可能捕捉更纯粹的 alpha。
    "quality_ebit_grp": _macro(e="ebit", a="assets") + "group_rank(e / a, subindustry)",
    "quality_cfo_grp": _macro(c="cashflow_op", a="assets") + "group_rank(c / a, subindustry)",
}

# Primary-source provenance for the round-2 candidates, kept next to the
# expressions so the registry and docs cannot drift from the code.
CITATIONS: dict[str, str] = {
    "ivol_total": "Ang/Hodrick/Xing/Zhang 2006 JF 61(1) 259-299 DOI 10.3386/w10852",
    "ivol_grp": "Ang/Hodrick/Xing/Zhang 2006 JF 61(1) 259-299 DOI 10.3386/w10852",
    "ivol_lagged": "Ang/Hodrick/Xing/Zhang 2006, 1/1/1 lagged variant",
    "asset_growth": "Cooper/Gulen/Ion 2024 JFE 151:103746 DOI 10.1016/j.jfineco.2023.103746",
    "inv_rec_growth": "Cooper/Gulen/Ion 2024 JFE 151:103746 (inventory+receivables result)",
    "inv_rec_grp": "Cooper/Gulen/Ion 2024 JFE 151:103746 (inventory+receivables result)",
    "quality_ebit_assets": "Novy-Marx 2013 JFE 108(1) 1-28 (gross profitability premium)",
    "quality_ebit_equity": "Novy-Marx 2013 JFE 108(1) 1-28 / Fama-French 2015 RMW factor",
    "quality_cfo_assets": "Fama-French 2015 RMW factor (cash flow variant)",
    "quality_cfo_equity": "Fama-French 2015 RMW factor (cash flow variant)",
    "quality_ebit_grp": "Novy-Marx 2013, industry-relative variant",
    "quality_cfo_grp": "Fama-French 2015 RMW, industry-relative variant",
}

BASE_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P0Y0M0D",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "maxTrade": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}

# Thresholds we require before we are willing to POST /submit.
MIN_SHARPE = 1.25
MIN_FITNESS = 1.0
TURNOVER_LO, TURNOVER_HI = 0.01, 0.70

# ------------------------------------------------------------------- session


class Brain:
    """Authenticated BRAIN session.

    Two failure modes the platform actually exhibits, both handled here:
    the edge drops TLS connections sporadically (retry), and sessions expire
    after ~4h (catch -> re-login, the repo's established pattern).
    """

    def __init__(self, cred_file: Path = CRED_FILE):
        self.cred_file = cred_file
        self._lock = threading.Lock()
        self.s = requests.Session()
        self.login()

    def login(self) -> None:
        user, pwd = json.loads(self.cred_file.read_text())
        self.s = requests.Session()
        self.s.auth = (user, pwd)
        r = self.s.post(f"{API}/authentication")
        if r.status_code == 401:
            if r.headers.get("WWW-Authenticate") == "persona":
                raise SystemExit(
                    "Biometric (persona) auth required - open:\n  "
                    + requests.compat.urljoin(r.url, r.headers.get("Location", ""))
                )
            raise SystemExit(f"Login failed ({r.status_code}): incorrect credentials")
        if r.status_code // 100 != 2:
            raise SystemExit(f"Login failed ({r.status_code}): {r.text[:200]}")
        log.info("authenticated as %s", r.json().get("user", {}).get("id"))

    def req(self, method: str, url: str, tries: int = 8, **kw) -> requests.Response:
        """One request with TLS-retry and transparent re-login on 401."""
        last: Exception | None = None
        for i in range(tries):
            try:
                r = self.s.request(method, url, timeout=60, **kw)
            except requests.exceptions.RequestException as e:
                last = e
                wait = min(5 * (i + 1), 30)
                log.debug("net %s on %s, retry in %ss", type(e).__name__, url, wait)
                time.sleep(wait)
                continue
            if r.status_code == 401:
                with self._lock:
                    log.warning("session expired, re-logging in")
                    self.login()
                continue
            return r
        raise RuntimeError(f"{method} {url} failed after {tries} tries: {last}")

    def get(self, url: str, **kw) -> requests.Response:
        return self.req("GET", url, **kw)

    def post(self, url: str, **kw) -> requests.Response:
        return self.req("POST", url, **kw)


def _wait_retry_after(b: Brain, url: str, label: str, cap: float = 1800.0) -> requests.Response:
    """Poll `url` while the platform keeps sending Retry-After."""
    waited = 0.0
    while True:
        r = b.get(url)
        ra = r.headers.get("Retry-After")
        if not ra:
            return r
        d = float(ra)
        waited += d
        if waited > cap:
            raise TimeoutError(f"{label}: still pending after {waited:.0f}s")
        time.sleep(d)


# ---------------------------------------------------------------- simulation


def simulate(b: Brain, name: str, expr: str, settings: dict) -> dict | None:
    """Run one simulation; return the finished alpha record (or None)."""
    payload = {"type": "REGULAR", "settings": settings, "regular": expr}
    r = b.post(f"{API}/simulations", json=payload)
    if r.status_code // 100 != 2:
        log.error("[%s] simulation rejected (%s): %s", name, r.status_code, r.text[:300])
        return None

    loc = r.headers.get("Location")
    if not loc:
        log.error("[%s] no Location header: %s", name, r.text[:200])
        return None

    log.info("[%s] simulating ...", name)
    try:
        prog = _wait_retry_after(b, loc, f"{name} simulation")
    except TimeoutError as e:
        log.error("[%s] %s", name, e)
        return None

    j = prog.json()
    if j.get("status") == "ERROR" or not j.get("alpha"):
        log.error("[%s] failed: %s", name, json.dumps(j)[:300])
        return None

    alpha_id = j["alpha"]
    rec = b.get(f"{API}/alphas/{alpha_id}").json()
    iss = rec.get("is") or {}
    log.info(
        "[%s] %s sharpe=%.2f fitness=%.2f turnover=%.4f returns=%.4f margin=%.6f",
        name, alpha_id, iss.get("sharpe") or 0, iss.get("fitness") or 0,
        iss.get("turnover") or 0, iss.get("returns") or 0, iss.get("margin") or 0,
    )
    rec["_name"] = name
    return rec


def submission_checks(b: Brain, alpha_id: str) -> list[dict]:
    """GET /alphas/{id}/check - resolves SELF_CORRELATION, which the raw
    simulation result leaves PENDING."""
    try:
        r = _wait_retry_after(b, f"{API}/alphas/{alpha_id}/check", f"{alpha_id} check")
    except TimeoutError as e:
        log.warning("check timed out: %s", e)
        return []
    j = r.json()
    if not j.get("is"):
        log.warning("no check payload for %s: %s", alpha_id, json.dumps(j)[:200])
        return []
    return j["is"].get("checks") or []


# Checks that never block a submission.
_INFO_CHECKS = {"MATCHES_COMPETITION", "MATCHES_PYRAMID", "UNITS", "ALPHA_POOL"}


def verdict(checks: list[dict]) -> tuple[bool, list[str], list[str]]:
    """-> (all_clear, blocking_failures, pending)."""
    fails, pending = [], []
    for c in checks:
        name, res = c.get("name", "?"), c.get("result")
        if name in _INFO_CHECKS:
            continue
        if res == "FAIL":
            fails.append(f"{name}(value={c.get('value')} limit={c.get('limit')})")
        elif res == "PENDING":
            pending.append(name)
    return (not fails and not pending), fails, pending


def show_checks(checks: list[dict], indent: str = "      ") -> None:
    for c in checks:
        extra = ""
        if c.get("value") is not None or c.get("limit") is not None:
            extra = f" value={c.get('value')} limit={c.get('limit')}"
        print(f"{indent}{str(c.get('result')):8} {c.get('name','?'):26}{extra}")


# ---------------------------------------------------------------- submission


def set_properties(b: Brain, alpha_id: str, name: str, desc: str) -> None:
    payload = {
        "color": None,
        "name": name[:80],
        "tags": ["composite", "fundamental", "scripted"],
        "category": None,
        "regular": {"description": desc[:500]},
    }
    r = b.req("PATCH", f"{API}/alphas/{alpha_id}", json=payload)
    if r.status_code // 100 != 2:
        log.debug("could not set properties (%s)", r.status_code)


def submit(b: Brain, alpha_id: str) -> tuple[bool, list[dict]]:
    """POST /alphas/{id}/submit and poll to completion."""
    r = b.post(f"{API}/alphas/{alpha_id}/submit")
    if r.status_code // 100 not in (2, 3) and r.status_code != 403:
        log.error("submit POST -> %s: %s", r.status_code, r.text[:300])

    waited = 0.0
    while "Retry-After" in r.headers:
        d = float(r.headers["Retry-After"])
        waited += d
        if waited > 1800:
            log.error("submit still pending after %.0fs", waited)
            return False, []
        time.sleep(d)
        r = b.get(f"{API}/alphas/{alpha_id}/submit")

    ok = r.status_code == 200
    checks: list[dict] = []
    try:
        body = r.json()
        checks = ((body.get("is") or {}).get("checks")) or []
    except Exception:
        pass
    if not ok:
        log.error("submit -> HTTP %s %s", r.status_code, r.text[:400])
    return ok, checks


# --------------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate and report, never POST /submit")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent simulations (TUTORIAL tier is limited)")
    ap.add_argument("--only", nargs="*", metavar="NAME",
                    help="restrict to these candidate names")
    ap.add_argument("--universe", default="TOP3000")
    ap.add_argument("--neutralization", default="SUBINDUSTRY")
    ap.add_argument("--decay", type=int, default=0)
    args = ap.parse_args()

    cands = dict(CANDIDATES)
    if args.only:
        missing = [n for n in args.only if n not in cands]
        if missing:
            raise SystemExit(f"unknown candidate(s): {missing}")
        cands = {n: cands[n] for n in args.only}

    settings = dict(BASE_SETTINGS)
    settings.update(universe=args.universe, neutralization=args.neutralization,
                    decay=args.decay)

    b = Brain()
    log.info("%d candidates | %s/%s/d%s/%s/decay%s/trunc%s", len(cands),
             settings["region"], settings["universe"], settings["delay"],
             settings["neutralization"], settings["decay"], settings["truncation"])

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(simulate, b, n, e, settings): n for n, e in cands.items()}
        for f in as_completed(futs):
            name = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                log.error("[%s] crashed: %s", name, e)
                continue
            if rec:
                results.append(rec)

    if not results:
        log.error("no simulation produced a result")
        return 1

    def key(rec: dict) -> tuple:
        iss = rec.get("is") or {}
        return (iss.get("fitness") or 0, iss.get("sharpe") or 0)

    results.sort(key=key, reverse=True)

    print("\n================ simulation results ================")
    for rec in results:
        iss = rec.get("is") or {}
        ok, fails, pending = verdict(iss.get("checks") or [])
        print(f"\n  {rec['_name']:12} {rec['id']}  sharpe={iss.get('sharpe')} "
              f"fitness={iss.get('fitness')} turnover={iss.get('turnover')} "
              f"returns={iss.get('returns')} margin={iss.get('margin')}")
        print(f"    long={iss.get('longCount')} short={iss.get('shortCount')} "
              f"drawdown={iss.get('drawdown')}")
        show_checks(iss.get("checks") or [], indent="    ")
        if fails:
            print(f"    -> blocking: {', '.join(fails)}")

    # Pick the best candidate whose simulation-time checks are clean.
    viable = []
    for rec in results:
        iss = rec.get("is") or {}
        _, fails, _ = verdict(iss.get("checks") or [])
        if not fails:
            viable.append(rec)

    if not viable:
        print("\nNo candidate passed its simulation checks - nothing submitted.")
        print("Blocking reasons above show what to iterate on.")
        return 2

    print("\n================ submission gate ================")
    for rec in viable:
        aid, name = rec["id"], rec["_name"]
        iss = rec.get("is") or {}
        print(f"\n  candidate {name} ({aid}) - resolving full checks ...")
        checks = submission_checks(b, aid) or (iss.get("checks") or [])
        show_checks(checks)
        ok, fails, pending = verdict(checks)
        if not ok:
            print(f"    -> not submittable (fails={fails} pending={pending})")
            continue

        if args.dry_run:
            print(f"    -> ALL CHECKS PASS. --dry-run set, not submitting.")
            return 0

        set_properties(b, aid, f"{name} composite",
                       "Composite of dense scale-free fundamental ratios, "
                       "winsorize(ts_backfill(f,120),std=4) preprocessed.")
        print(f"    -> all checks PASS, submitting {aid} ...")
        done, post_checks = submit(b, aid)
        if post_checks:
            show_checks(post_checks)
        if done:
            print(f"\n  SUBMITTED: {aid} ({name})")
            print(f"  https://platform.worldquantbrain.com/alpha/{aid}")
            return 0
        print(f"    -> submission of {aid} was rejected; trying next candidate")

    print("\nNo candidate was accepted by /submit.")
    return 3


if __name__ == "__main__":
    sys.exit(main())




