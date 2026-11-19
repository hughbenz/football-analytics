#!/usr/bin/env python3
"""
referee_ou_backtest.py - walk-forward backtest of betting Over/Under 2.5 on referees
====================================================================================

Idea: some referees' matches may go over (or under) 2.5 goals more often than Pinnacle prices.
If that is real and lasting, betting the referee's direction should pay.

HOW EACH MATCH IS SCORED (no look-ahead)
  * Pinnacle's last pre-kickoff Over/Under 2.5 prices, margin removed -> P(over).
  * Residual = (3+ goals in 90 minutes) - P(over).
  * League adjustment: the league's own average residual from EARLIER days is subtracted (shrunk), so a
    referee is not credited with his league's pricing bias.
  * Before each match day, every referee's record is computed from EARLIER days only:
      matches, adjusted residual sum, z-score, p-value, Benjamini-Hochberg q-value across all referees
      tested that day, shrunk effect size (pp of probability), and whether his lean held in both halves
      of his history ("steady").
  * The day's results are added only after the day's rows are written.
  * Time-travel test: the whole computation is re-run with every match from a cut-off day onwards
    deleted; that day's rows must come out identical.

THE STRATEGY (chosen live in the HTML page, nothing tuned in Python)
  Bet the referee's direction when his p (or q) is under the threshold. Stake = flat, or scaled by
  his shrunk effect size (1 unit per X pp of estimated edge, capped). Prices: Pinnacle, average or best.
  The page also splits the period into discovery and hold-out so a setting picked by looking at the
  discovery results can be checked on later, untouched matches.

USAGE (same folder as backtest_models.py, for the database connection)
    python3 referee_ou_backtest.py
    python3 referee_ou_backtest.py --out referee_backtest.html --min-matches 15 --shrink 30
"""
import argparse
import html as htmlmod
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import backtest_models as bt  # noqa: E402
from backtest_models import log  # noqa: E402

MATCHES = """
WITH sc AS (SELECT fixture_id,
       max(goals) FILTER (WHERE location='home' AND description='CURRENT')  AS h_cur, max(goals) FILTER (WHERE location='away' AND description='CURRENT')  AS a_cur,
       max(goals) FILTER (WHERE location='home' AND description='2ND_HALF') AS h_90,  max(goals) FILTER (WHERE location='away' AND description='2ND_HALF') AS a_90
       FROM football.fixture_scores WHERE description IN ('CURRENT','2ND_HALF') GROUP BY fixture_id),
ref AS (SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
        FROM football.fixture_officials fo LEFT JOIN core.types t ON t.type_id = fo.type_id
        WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
        ORDER BY fo.fixture_id, fo.fixture_official_id)
SELECT f.fixture_id, f.starting_at, f.league_id, ref.referee_id,
       COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
       CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.h_90, sc.h_cur) ELSE sc.h_cur END
     + CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.a_90, sc.a_cur) ELSE sc.a_cur END AS goals
FROM football.fixtures f
JOIN core.states st ON st.state_id = f.state_id AND st.developer_name IN ('FT','AET','FT_PEN')
JOIN ref ON ref.fixture_id = f.fixture_id
JOIN sc ON sc.fixture_id = f.fixture_id
LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'"""


def bh(p):
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        ps = p[ok]
        o = np.argsort(ps)
        r = ps[o] * len(ps) / np.arange(1, len(ps) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
        out = np.empty(len(ps))
        out[o] = r
        q[ok] = out
    return q


def pval(z):
    from scipy.special import erfc
    return erfc(np.abs(z) / math.sqrt(2))


def walk_forward(d, args):
    """d sorted by day. Returns pre-match referee features for every row, using earlier days only."""
    n_rows = len(d)
    ri = d.ref_i.to_numpy()
    li = d.lg_i.to_numpy()
    p_pin = d.p.to_numpy()
    y = d.y.to_numpy()
    days = d.day.to_numpy()
    R, L = int(ri.max()) + 1, int(li.max()) + 1
    n, S, V = np.zeros(R), np.zeros(R), np.zeros(R)
    hist = [[] for _ in range(R)]
    Ls, Ln = np.zeros(L), np.zeros(L)
    out = {k: np.full(n_rows, np.nan) for k in ("n", "eff", "mean", "z", "p", "q", "steady", "ladj")}
    bounds = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = np.arange(a, b)
        # ---- snapshot: everything below uses results from earlier days only
        with np.errstate(invalid="ignore", divide="ignore"):
            z_all = np.where(n >= args.min_matches, S / np.sqrt(np.maximum(V, 1e-12)), np.nan)
        p_all = pval(z_all)
        q_all = bh(np.where(n >= args.bh_min, p_all, np.nan))
        for i in rows:
            j = ri[i]
            out["n"][i] = n[j]
            out["ladj"][i] = Ls[li[i]] / (Ln[li[i]] + args.league_k)
            if n[j] >= args.min_matches:
                out["eff"][i] = S[j] / (n[j] + args.shrink)
                out["mean"][i] = S[j] / n[j]
                out["z"][i], out["p"][i], out["q"][i] = z_all[j], p_all[j], q_all[j]
                h = hist[j]
                m1, m2 = np.mean(h[: len(h) // 2]), np.mean(h[len(h) // 2:])
                out["steady"][i] = float(np.sign(m1) == np.sign(m2) == np.sign(S[j]) != 0)
        # ---- then the day's results enter the state
        resid = y[rows] - p_pin[rows]
        adj = resid - out["ladj"][rows]
        np.add.at(Ls, li[rows], resid)
        np.add.at(Ln, li[rows], 1)
        np.add.at(n, ri[rows], 1)
        np.add.at(S, ri[rows], adj)
        np.add.at(V, ri[rows], p_pin[rows] * (1 - p_pin[rows]))
        for i, r_ in zip(rows, adj):
            hist[ri[i]].append(r_)
    out["adj"] = y - p_pin - out["ladj"]
    return out


def leak_test(d, full, args, k=3):
    rng = np.random.default_rng(5)
    days = np.unique(d.day)
    days = days[len(days) // 3:]
    tests, ok = [], True
    for c in sorted(rng.choice(days, size=min(k, len(days)), replace=False)):
        cut = d[d.day <= c].copy()
        cut.loc[cut.day == c, ["goals", "y"]] = np.nan          # that day's results are unknown
        cut["y"] = cut.y.fillna(0.0)                              # placeholder: never read before the day is emitted
        part = walk_forward(cut, args)
        sel = (d.day == c).to_numpy()
        selc = (cut.day == c).to_numpy()
        bad = 0
        for key in ("n", "eff", "z", "p", "q", "steady", "ladj"):
            bad += int((~np.isclose(full[key][sel], part[key][selc], equal_nan=True, rtol=1e-9, atol=1e-12)).sum())
        tests.append({"day": str(pd.Timestamp(c).date()), "rows": int(sel.sum()), "mismatches": bad})
        ok &= bad == 0
        log(f"  time-travel test {pd.Timestamp(c).date()}: {sel.sum()} matches, mismatches = {bad}")
    return {"tests": tests, "passed": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--out", default="referee_ou_backtest.html")
    ap.add_argument("--market-ou", type=int, default=80, help="Sportmonks market id for Over/Under")
    ap.add_argument("--bookmaker", default="pinnacle", help="bookmaker whose prices define 'expected' (case-insensitive name match)")
    ap.add_argument("--min-matches", type=int, default=15, help="earliest point a referee can be tested (the page can raise it)")
    ap.add_argument("--bh-min", type=int, default=30, help="referees with this many matches enter the daily multiple-testing correction")
    ap.add_argument("--shrink", type=float, default=30, help="effect size = residual sum / (matches + shrink)")
    ap.add_argument("--league-k", type=float, default=100, help="shrinkage of the league adjustment")
    ap.add_argument("--max-p", type=float, default=0.25, help="rows with a larger p are not exported (the page's threshold tops out here)")
    ap.add_argument("--holdout-share", type=float, default=0.4, help="last share of the period kept as hold-out")
    ap.add_argument("--leak-tests", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    db = bt.DB(args.dsn)
    m = db.query(MATCHES).dropna(subset=["goals", "referee_id", "starting_at"])
    log(f"{len(m):,} finished matches with a referee and a score")
    o = db.query(bt.odds_sql(args, "ou", False)).dropna(subset=["o_o", "o_u"])
    pin = o[o.bookmaker.fillna("").str.lower().str.contains(args.bookmaker.lower())].groupby("fixture_id")[["o_o", "o_u"]].mean()
    if pin.empty:
        sys.exit(f"No '{args.bookmaker}' Over/Under 2.5 prices found (check --bookmaker / --market-ou).")
    agg = o.groupby("fixture_id").agg(avg_o=("o_o", "mean"), avg_u=("o_u", "mean"), best_o=("o_o", "max"), best_u=("o_u", "max"), books=("o_o", "size"))
    d = m.merge(pin.rename(columns={"o_o": "pin_o", "o_u": "pin_u"}), left_on="fixture_id", right_index=True).merge(agg, left_on="fixture_id", right_index=True, how="left")
    d["day"] = pd.to_datetime(d.starting_at, utc=True).dt.tz_convert(None).dt.normalize()
    d["p"] = (1 / d.pin_o) / (1 / d.pin_o + 1 / d.pin_u)
    d["y"] = (d.goals >= 3).astype(float)
    d = d.sort_values(["day", "fixture_id"]).reset_index(drop=True)
    d["ref_i"], refs = pd.factorize(d.referee_id)
    d["lg_i"], lgs = pd.factorize(d.league_id)
    log(f"{len(d):,} matches with {args.bookmaker.title()} prices · {len(refs):,} referees · {len(lgs)} competitions")

    def names(sql, key):
        try:
            x = db.query(sql)
            return dict(zip(x[key].astype("int64"), x.name))
        except Exception:  # noqa: BLE001
            return {}
    rname = names("SELECT referee_id, COALESCE(display_name, common_name, name) AS name FROM football.referees", "referee_id")
    tname = names("SELECT team_id, name FROM football.teams", "team_id")
    lgq = None
    for sql in ("SELECT league_id, name, type FROM football.leagues", "SELECT league_id, name, NULL AS type FROM football.leagues"):
        try:
            lgq = db.query(sql)
            break
        except Exception:  # noqa: BLE001
            continue
    lname = dict(zip(lgq.league_id.astype("int64"), lgq.name)) if lgq is not None else {}
    ltype = dict(zip(lgq.league_id.astype("int64"), lgq["type"])) if lgq is not None else {}

    log("walking forward day by day ...")
    W = walk_forward(d, args)
    leak = {"tests": [], "passed": True, "skipped": True}
    if args.leak_tests:
        log("time-travel test: re-running with the future deleted ...")
        leak = {**leak_test(d, W, args, args.leak_tests), "skipped": False}

    # ---- rows the page can bet on
    keep = np.isfinite(W["p"]) & (W["p"] <= args.max_p)
    hold_from = d.day[np.isfinite(W["p"])].quantile(1 - args.holdout_share).normalize() if np.isfinite(W["p"]).any() else d.day.max()
    teams = sorted({int(x) for x in pd.concat([d.home_id, d.away_id]).dropna()})
    tix = {t: i for i, t in enumerate(teams)}
    ix = np.where(keep)[0]
    r4 = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(a, float)]  # noqa: E731
    rows = {"d": d.day.dt.strftime("%Y-%m-%d").to_numpy()[ix].tolist(), "r": d.ref_i.to_numpy()[ix].tolist(), "l": d.lg_i.to_numpy()[ix].tolist(),
            "h": [tix.get(int(x), -1) if pd.notna(x) else -1 for x in d.home_id.to_numpy()[ix]],
            "a": [tix.get(int(x), -1) if pd.notna(x) else -1 for x in d.away_id.to_numpy()[ix]],
            "n": W["n"][ix].astype(int).tolist(), "e": r4(W["eff"][ix]), "p": r4(W["p"][ix]), "q": r4(W["q"][ix]), "s": W["steady"][ix].astype(int).tolist(),
            "y": d.y.to_numpy()[ix].astype(int).tolist(), "g": d.goals.to_numpy()[ix].astype(int).tolist(), "pp": r4(d.p.to_numpy()[ix]), "la": r4(W["ladj"][ix]),
            "po": r4(d.pin_o.to_numpy()[ix]), "pu": r4(d.pin_u.to_numpy()[ix]), "ao": r4(d.avg_o.to_numpy()[ix]), "au": r4(d.avg_u.to_numpy()[ix]),
            "bo": r4(d.best_o.to_numpy()[ix]), "bu": r4(d.best_u.to_numpy()[ix])}
    # ---- referee summary (whole period, descriptive) + seasons for the detail view
    d["adj"] = W["adj"]
    d["year"] = d.day.dt.year
    summ = []
    for j, g in d.groupby("ref_i"):
        nn, var = len(g), float((g.p * (1 - g.p)).sum())
        z = g.adj.sum() / math.sqrt(var) if var > 0 else np.nan
        h1, h2 = g.adj.iloc[: nn // 2].mean(), g.adj.iloc[nn // 2:].mean()
        lg = int(g.lg_i.mode().iloc[0])
        yrs = g.groupby("year").agg(n=("y", "size"), over=("y", "mean"), exp=("p", "mean"), adj=("adj", "mean")).reset_index()
        summ.append({"name": rname.get(int(refs[j])) or f"Referee {int(refs[j])}", "lg": lg, "n": nn, "over": g.y.mean(), "exp": g.p.mean(),
                     "eff": g.adj.sum() / (nn + args.shrink), "z": z, "p": float(pval(np.array([z]))[0]) if np.isfinite(z) else None,
                     "steady": bool(nn >= 10 and np.sign(h1) == np.sign(h2) != 0), "leagues": int(g.lg_i.nunique()), "last": str(g.day.max().date()),
                     "years": yrs[["year", "n", "over", "exp", "adj"]].round(4).values.tolist()})
    pz = np.array([s["p"] if s["p"] is not None and s["n"] >= args.bh_min else np.nan for s in summ], float)
    qz = bh(pz)
    for s, q in zip(summ, qz):
        s["q"] = None if not np.isfinite(q) else float(q)
    payload = bt.clean({
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "db": db.name, "book": args.bookmaker.title(),
        "params": {"min_matches": args.min_matches, "bh_min": args.bh_min, "shrink": args.shrink, "league_k": args.league_k, "max_p": args.max_p},
        "hold_from": str(hold_from.date()), "from": str(d.day.min().date()), "to": str(d.day.max().date()),
        "n_matches": len(d), "over_rate": float(d.y.mean()), "exp_rate": float(d.p.mean()),
        "margin": float((1 / d.pin_o + 1 / d.pin_u - 1).mean()), "with_avg": float(d.avg_o.notna().mean()),
        "leagues": [lname.get(int(x)) or f"Competition {int(x)}" for x in lgs],
        "ltypes": [str(ltype.get(int(x)) or "").strip().lower() or "unknown" for x in lgs],
        "teams": [tname.get(t) or str(t) for t in teams], "refs": summ, "rows": rows, "leak": leak, "runtime": round(time.time() - t0)})
    page = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__TITLE__", htmlmod.escape("Referee Over/Under backtest"))
    Path(args.out).write_text(page, encoding="utf-8")
    log(f"{len(ix):,} bettable match rows exported · page written: {args.out} ({len(page) / 1e6:.1f} MB)")
    if not leak["skipped"] and not leak["passed"]:
        log("!!! TIME-TRAVEL TEST FAILED - see the page")


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root{color-scheme:light;--page:#f6f6f3;--surface:#fcfcfb;--raise:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e6e5df;--axis:#c3c2b7;--ring:rgba(11,11,11,.09);
  --over:#eb6834;--under:#2a78d6;--ctrl:#b4b2aa;--good:#006300;--bad:#d03b3b;--goodbg:rgba(12,163,12,.10);--badbg:rgba(208,59,59,.09);--wash:rgba(42,120,214,.07);--shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#171716;--raise:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2a2a28;--axis:#3c3c39;--ring:rgba(255,255,255,.09);
  --over:#d95926;--under:#3987e5;--ctrl:#5f5e59;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.11);--shadow:none}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#171716;--raise:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2a2a28;--axis:#3c3c39;--ring:rgba(255,255,255,.09);
  --over:#d95926;--under:#3987e5;--ctrl:#5f5e59;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.11);--shadow:none}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;min-height:100vh;padding:26px clamp(14px,3vw,40px) 70px}
.wrap{max-width:1240px;margin:0 auto}
h1{font-size:25px;font-weight:680;margin:0 0 4px;letter-spacing:-.015em}h2{font-size:16px;font-weight:640;margin:0 0 6px}h3{font-size:13.5px;font-weight:620;margin:0 0 4px;color:var(--ink2)}
.sub{color:var(--ink2);margin:0 0 14px;max-width:900px}.note{color:var(--muted);font-size:12.5px;margin:4px 0 10px}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--ring);margin:6px 0 20px}
.tabs button{border:0;background:none;font:inherit;font-size:14px;padding:9px 14px;color:var(--ink2);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tabs button.on{color:var(--ink);border-bottom-color:var(--ink);font-weight:620}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow);min-width:0}
.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}.side{grid-template-columns:300px minmax(0,1fr)}@media(max-width:900px){.side{grid-template-columns:1fr}}
.ctl{display:flex;flex-direction:column;gap:14px}.ctl label{font-size:12.5px;color:var(--ink2);display:flex;flex-direction:column;gap:5px}
.ctl .row{display:flex;justify-content:space-between;align-items:baseline}.ctl .val{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600}
input[type=range]{width:100%;accent-color:var(--under)}
select,input[type=search]{font:inherit;font-size:13px;color:var(--ink);background:var(--page);border:1px solid var(--ring);border-radius:8px;padding:6px 9px}
.seg{display:inline-flex;background:var(--page);border:1px solid var(--ring);border-radius:9px;padding:2px;flex-wrap:wrap}
.seg button{border:0;background:transparent;padding:5px 10px;border-radius:7px;font:inherit;font-size:12.5px;color:var(--ink2);cursor:pointer}
.seg button.on{background:var(--raise);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12);font-weight:600}
.tile .lbl{color:var(--ink2);font-size:12.5px}.tile .v{font-size:24px;font-weight:660;font-variant-numeric:tabular-nums;letter-spacing:-.01em;margin:2px 0}.tile .ci{color:var(--muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.verdict{border-radius:12px;padding:11px 14px;margin:0 0 14px;font-size:14px;background:var(--page);border:1px solid var(--ring)}
.verdict.good{background:var(--goodbg);border-color:transparent}.verdict.bad{background:var(--badbg);border-color:transparent}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:2px 0 10px}.chip{border:1px solid var(--ring);background:var(--page);border-radius:99px;padding:4px 11px;font-size:12.5px;cursor:pointer;color:var(--ink2)}
.chip.on{background:var(--ink);color:var(--page);border-color:var(--ink)}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{font-family:inherit;font-size:11px;fill:var(--muted)}svg .lab{fill:var(--ink2);font-size:11.5px}svg .val{fill:var(--ink);font-size:11px}
.hit{fill:transparent}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;font-size:12.5px;cursor:pointer;user-select:none;background:var(--surface);position:sticky;top:0}
th .u{display:block;font-weight:400;color:var(--muted);font-size:11px}
td:first-child,th:first-child,.l{text-align:left}td{font-variant-numeric:tabular-nums}tbody tr{cursor:pointer}tbody tr:hover td{background:var(--wash)}tr.sel td{background:var(--wash)}
.pill{display:inline-block;font-size:11.5px;font-weight:620;padding:2px 9px;border-radius:99px;color:#fff}
.pill.o{background:var(--over)}.pill.u{background:var(--under)}.pill.n{background:transparent;color:var(--muted);border:1px solid var(--ring)}
.sig{display:inline-flex;align-items:center;gap:6px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.wide{overflow-x:auto}.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 12px}
.pager{display:flex;gap:10px;align-items:center;justify-content:flex-end;margin-top:10px;font-size:12.5px;color:var(--ink2)}
.pager button{font:inherit;font-size:13px;border:1px solid var(--ring);background:var(--page);color:var(--ink);border-radius:8px;padding:4px 10px;cursor:pointer}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:2px 0 10px}.key{display:inline-flex;align-items:center;gap:6px}
.tip{position:fixed;pointer-events:none;background:var(--raise);color:var(--ink);border:1px solid var(--ring);box-shadow:0 6px 24px rgba(0,0,0,.16);border-radius:9px;padding:9px 11px;font-size:12px;opacity:0;transition:opacity .08s;z-index:40;max-width:320px}
details{border-top:1px solid var(--grid);padding:10px 0}details summary{cursor:pointer;font-weight:600;font-size:13.5px}details p,details li{color:var(--ink2);font-size:13px;max-width:860px}
.badge{display:inline-flex;gap:6px;font-size:12.5px;padding:4px 11px;border-radius:99px;background:var(--goodbg);color:var(--good)}.badge.bad{background:var(--badbg);color:var(--bad)}
.drawer{margin-top:14px}.ctl label.toggle,.toggle{flex-direction:row;display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink2);cursor:pointer}
button.plain{font:inherit;font-size:13px;border:1px solid var(--ring);background:var(--page);color:var(--ink);border-radius:8px;padding:5px 11px;cursor:pointer}
</style></head><body><div class="viz-root"><div class="wrap">
<div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start"><div><h1>__TITLE__</h1><p class="sub" id="sub"></p></div><button class="plain" id="theme" type="button">◐</button></div>
<div class="tabs" id="tabs"><button data-t="strategy" class="on">Strategy</button><button data-t="refs">Referees</button><button data-t="how">How it works</button></div>

<div id="t-strategy">
 <div class="grid side">
  <div class="card ctl" id="controls">
   <div><h3>When to bet</h3>
    <label style="margin-top:8px"><span class="row">Test<span></span></span><div class="seg" id="c-test"></div></label></div>
   <label><span class="row">Significance threshold<span class="val" id="v-thr"></span></span><input type="range" id="c-thr" min="0" max="11" step="1" value="4"></label>
   <label><span class="row">Referee needs at least<span class="val" id="v-min"></span></span><input type="range" id="c-min" min="0" max="100" step="5" value="30"></label>
   <label><span class="row">Smallest expected profit, after margin<span class="val" id="v-edge"></span></span><input type="range" id="c-edge" min="-5" max="15" step="0.5" value="1"></label>
   <label class="toggle"><input type="checkbox" id="c-steady"> Only referees whose lean held in both halves of their history</label>
   <div><h3>How much</h3><div class="seg" id="c-stake" style="margin-top:6px"></div></div>
   <label id="w-unit"><span class="row">1 unit per<span class="val" id="v-unit"></span></span><input type="range" id="c-unit" min="1" max="10" step="0.5" value="4"></label>
   <label id="w-cap"><span class="row">Largest stake<span class="val" id="v-cap"></span></span><input type="range" id="c-cap" min="1" max="5" step="0.5" value="3"></label>
   <div><h3>Where</h3><div class="seg" id="c-price" style="margin-top:6px"></div></div>
   <div class="seg" id="c-dir"></div>
   <label>Competition<select id="c-lg"></select></label>
  </div>
  <div>
   <div class="chips" id="c-period"></div>
   <div class="verdict" id="verdict"></div>
   <div class="grid g4" id="tiles"></div>
   <div class="card" style="margin-top:14px"><div class="chips" id="c-view"></div><div id="view"></div></div>
  </div>
 </div>
</div>

<div id="t-refs" hidden>
 <div class="card">
  <div class="bar"><input type="search" id="r-q" placeholder="Search referee" style="min-width:220px">
   <select id="r-lg"></select><select id="r-type"></select>
   <div class="seg" id="r-dir"></div>
   <label class="toggle"><input type="checkbox" id="r-sig"> Significant only</label></div>
  <div class="legend"><span class="key"><span class="pill o">Over</span> more goals than priced</span><span class="key"><span class="pill u">Under</span> fewer goals than priced</span>
   <span class="key"><span class="dot" style="background:var(--good)"></span>q &lt; 0.05</span><span class="key"><span class="dot" style="background:#c99a2e"></span>q &lt; 0.20</span><span class="key"><span class="dot" style="background:var(--ctrl)"></span>not significant</span>
   <span class="key">Bets and profit use the settings on the Strategy tab</span></div>
  <div class="wide"><table id="r-table"></table></div>
  <div class="pager"><button id="pg-prev">‹</button><span id="pg-info"></span><button id="pg-next">›</button></div>
 </div>
 <div class="card drawer" id="r-detail" hidden></div>
</div>

<div id="t-how" hidden><div class="card" id="how"></div></div>
</div><div class="tip" id="tip"></div></div>
<script>
const R = __DATA__;
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const int = v => v == null ? '–' : Math.round(v).toLocaleString();
const pctu = (v, d=1) => v == null || !isFinite(v) ? '–' : (v*100).toFixed(d) + '%';
const pct = (v, d=1) => v == null || !isFinite(v) ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(d) + '%';
const sgn = (v, d=1) => v == null || !isFinite(v) ? '–' : (v >= 0 ? '+' : '') + (+v).toFixed(d);
const pp = v => v == null || !isFinite(v) ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(1) + ' pp';
const cls = v => v == null ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : '';
const tip = $('#tip');
function showTip(e, h){ tip.innerHTML = h; tip.style.opacity = 1; const w = tip.offsetWidth; tip.style.left = (e.clientX + 16 + w > innerWidth ? e.clientX - w - 16 : e.clientX + 16) + 'px'; tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8) + 'px'; }
const hideTip = () => tip.style.opacity = 0;
const NS = 'http://www.w3.org/2000/svg';
function el(t, a, p){ const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
function svg(h, w, ht){ h.innerHTML = ''; return el('svg', {viewBox:`0 0 ${w} ${ht}`}, h); }
function text(p, x, y, s, a={}){ const t = el('text', Object.assign({x, y}, a), p); t.textContent = s; return t; }
function ticks(lo, hi, n=5){ const sp = hi - lo || 1, s0 = sp/n, mg = Math.pow(10, Math.floor(Math.log10(s0))); const st = [1,2,2.5,5,10].map(m=>m*mg).find(s => sp/s <= n) || mg*10; const o=[]; for (let v = Math.ceil(lo/st)*st; v <= hi+1e-9; v += st) o.push(+v.toFixed(10)); return o; }
function ext(v){ v = v.filter(x => x != null && isFinite(x)).concat([0]); let lo = Math.min(...v), hi = Math.max(...v); if (lo === hi){ lo -= 1; hi += 1; } const p = (hi-lo)*.08; return [lo-p, hi+p]; }
function empty(h, m){ h.innerHTML = `<p class="note" style="padding:40px 0;text-align:center">${m}</p>`; }
function vbarPath(x, w, y0, y1){ const r = Math.min(4, Math.abs(y1-y0)/2, w/2); return y1 <= y0 ? `M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z` : `M${x},${y0}V${y1-r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1-r}V${y0}Z`; }

// ------------------------------------------------------------------ data
const D = R.rows, N = D.d.length, THR = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25];
const S = {test:'q', thr:4, min:30, edge:1, steady:false, stake:'scaled', unit:4, cap:3, price:'pin', dir:'both', lg:'', period:'all', view:'curve',
           rq:'', rlg:'', rtype:'', rdir:'all', rsig:false, sortK:'q', sortD:1, page:0, selRef:null};
const minN = () => Math.max(S.min, R.params.min_matches);
function bets(){
  const thr = THR[S.thr], out = [];
  for (let i = 0; i < N; i++){
    const pv = S.test === 'q' ? D.q[i] : D.p[i]; if (pv == null || pv >= thr) continue;
    if (D.n[i] < minN()) continue;
    const e = D.e[i]; if (e == null) continue;
    if (S.steady && !D.s[i]) continue;
    const over = e > 0; if (S.dir === 'over' && !over || S.dir === 'under' && over) continue;
    if (S.lg !== '' && D.l[i] !== +S.lg) continue;
    if (S.period === 'disc' && D.d[i] >= R.hold_from || S.period === 'hold' && D.d[i] < R.hold_from) continue;
    const price = over ? D[S.price[0] + 'o'][i] : D[S.price[0] + 'u'][i]; if (!price || price <= 1) continue;
    const pOver = Math.min(0.99, Math.max(0.01, D.pp[i] + (D.la[i] || 0) + e)), pm = over ? pOver : 1 - pOver, ev = pm * price - 1;   // model chance x the price actually paid
    if (ev * 100 < S.edge) continue;
    let stake = 1; if (S.stake === 'scaled'){ stake = Math.min(S.cap, Math.max(0.25, Math.round(Math.abs(e)*100 / S.unit * 4) / 4)); }
    const won = over ? D.y[i] === 1 : D.y[i] === 0;
    out.push({i, over, price, stake, won, ev, pm, pl: won ? stake*(price-1) : -stake});
  }
  return out;
}
function summary(B){
  const st = B.reduce((a, b) => a + b.stake, 0), pl = B.reduce((a, b) => a + b.pl, 0), roi = st ? pl/st : null;
  const se = st ? Math.sqrt(B.reduce((a, b) => a + (b.pl - roi*b.stake) ** 2, 0)) / st : null;
  let cum = 0, peak = 0, dd = 0; for (const b of B){ cum += b.pl; peak = Math.max(peak, cum); dd = Math.min(dd, cum - peak); }
  return {n:B.length, st, pl, roi, lo: roi != null ? roi - 1.96*se : null, hi: roi != null ? roi + 1.96*se : null, win: B.length ? B.filter(b => b.won).length/B.length : null,
          odds: B.length ? B.reduce((a, b) => a + b.price, 0)/B.length : null, dd, over: B.filter(b => b.over).length,
          ev: st ? B.reduce((a, b) => a + b.ev*b.stake, 0)/st : null};
}
// ------------------------------------------------------------------ header & tabs
$('#sub').innerHTML = `${int(R.n_matches)} matches with ${esc(R.book)} Over/Under 2.5 prices, ${R.from} → ${R.to}, ${R.leagues.length} competitions, ${R.refs.length} referees, average margin ${pctu(R.margin, 1)}. Overall ${pctu(R.over_rate)} went over vs ${pctu(R.exp_rate)} priced.
  ${R.leak.skipped ? '' : `<span class="badge ${R.leak.passed ? '' : 'bad'}">${R.leak.passed ? '✓' : '✕'} Time-travel test ${R.leak.passed ? 'passed' : 'FAILED'}</span>`}`;
$$('#tabs button').forEach(b => b.onclick = () => { $$('#tabs button').forEach(x => x.classList.toggle('on', x === b)); for (const t of ['strategy','refs','how']) $('#t-' + t).hidden = t !== b.dataset.t; if (b.dataset.t === 'refs') drawRefs(); });
function seg(id, opts, key, cb){ const h = $(id); h.innerHTML = opts.map(([v, l]) => `<button data-v="${v}" class="${S[key] === v ? 'on' : ''}">${l}</button>`).join('');
  h.querySelectorAll('button').forEach(b => b.onclick = () => { S[key] = b.dataset.v; h.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); cb(); }); }
function chips(id, opts, key, cb){ const h = $(id); h.innerHTML = opts.map(([v, l]) => `<span class="chip ${S[key] === v ? 'on' : ''}" data-v="${v}">${l}</span>`).join('');
  h.querySelectorAll('.chip').forEach(c => c.onclick = () => { S[key] = c.dataset.v; chips(id, opts, key, cb); cb(); }); }
const lgOpts = (all) => `<option value="">${all}</option>` + R.leagues.map((n, i) => [n, i]).sort((a, b) => a[0].localeCompare(b[0])).map(([n, i]) => `<option value="${i}">${esc(n)}</option>`).join('');

// ------------------------------------------------------------------ strategy controls
seg('#c-test', [['q', 'Corrected (q)'], ['p', 'Raw p']], 'test', update);
seg('#c-stake', [['scaled', 'By effect size'], ['flat', 'Flat 1 unit']], 'stake', update);
seg('#c-price', [['pin', R.book], ['avg', 'Average'], ['best', 'Best']], 'price', update);
seg('#c-dir', [['both', 'Overs & unders'], ['over', 'Overs'], ['under', 'Unders']], 'dir', update);
chips('#c-period', [['all', 'Whole period'], ['disc', `Discovery (to ${R.hold_from})`], ['hold', `Hold-out (from ${R.hold_from})`]], 'period', update);
chips('#c-view', [['curve', 'Profit over time'], ['years', 'By year'], ['leagues', 'By competition'], ['stakes', 'By stake'], ['list', 'Bet list']], 'view', drawView);
$('#c-lg').innerHTML = lgOpts('All competitions');
$('#c-min').min = R.params.min_matches;
for (const [id, k] of [['#c-thr', 'thr'], ['#c-min', 'min'], ['#c-edge', 'edge'], ['#c-unit', 'unit'], ['#c-cap', 'cap']]) $(id).oninput = e => { S[k] = +e.target.value; update(); };
$('#c-steady').onchange = e => { S.steady = e.target.checked; update(); }; $('#c-lg').onchange = e => { S.lg = e.target.value; update(); };
let B = [], SUM = null;
function update(){
  $('#v-thr').textContent = `${S.test} < ${THR[S.thr]}`; $('#v-min').textContent = `${minN()} matches`; $('#v-edge').textContent = `${S.edge >= 0 ? '+' : ''}${S.edge.toFixed(1)}% per unit`;
  $('#v-unit').textContent = `${S.unit} pp of edge`; $('#v-cap').textContent = `${S.cap} units`;
  $('#w-unit').hidden = $('#w-cap').hidden = S.stake !== 'scaled';
  B = bets(); SUM = summary(B); tiles(); drawView(); if (!$('#t-refs').hidden) drawRefs();
}
function tiles(){
  const s = SUM, priceNote = S.price === 'best' ? ' Best prices often include stale quotes nobody would honour - trust them least.' : '';
  const v = $('#verdict');
  if (!s.n){ v.className = 'verdict'; v.innerHTML = 'No bets with these settings. Loosen the threshold or lower the minimum matches.'; $('#tiles').innerHTML = ''; return; }
  const sure = s.lo > 0, bad = s.hi < 0;
  v.className = 'verdict ' + (sure ? 'good' : bad ? 'bad' : '');
  v.innerHTML = `${int(s.n)} bets, ${s.st.toFixed(1)} units staked: <b class="${cls(s.pl)}">${sgn(s.pl, 1)} units</b> (${pct(s.roi)} per unit, 95% range ${pct(s.lo)} to ${pct(s.hi)}). ` +
    (sure ? 'The range is above zero: unlikely to be luck alone.' : bad ? 'The range is below zero: this loses.' : 'The range includes zero: could be luck either way.') +
    (S.period !== 'hold' ? ` <span class="note">Choosing settings by looking here is itself fitting: confirm them on the hold-out.</span>` : '') + `<span class="note">${priceNote}</span>`;
  const t = [['Profit', `<span class="${cls(s.pl)}">${sgn(s.pl, 1)}u</span>`, `${sgn(s.st, 1).replace('+', '')} units staked`], ['Return per unit', `<span class="${cls(s.roi)}">${pct(s.roi)}</span>`, `${pct(s.lo)} to ${pct(s.hi)}`],
    ['Bets won', pctu(s.win), `average odds ${s.odds.toFixed(2)} · model expected ${pct(s.ev)}`], ['Worst drawdown', `${s.dd.toFixed(1)}u`, `${int(s.over)} overs · ${int(s.n - s.over)} unders`]];
  $('#tiles').innerHTML = t.map(x => `<div class="card tile"><div class="lbl">${x[0]}</div><div class="v">${x[1]}</div><div class="ci">${x[2]}</div></div>`).join('');
}
// ------------------------------------------------------------------ views
function drawView(){ const h = $('#view'); if (!B.length) return empty(h, 'No bets with these settings');
  ({curve, years:() => groups(b => D.d[b.i].slice(0, 4), 'Year'), leagues:() => groups(b => R.leagues[D.l[b.i]], 'Competition'),
    stakes:() => groups(b => b.stake.toFixed(2).replace(/\.?0+$/, '') + 'u', 'Stake'), list})[S.view](h); }
function curve(h){
  const W = 860, H = 300, top = 14, bottom = 30, left = 54, right = 70, s = svg(h, W, H);
  const pts = []; let c = 0; const byDay = new Map(); for (const b of B){ c += b.pl; byDay.set(D.d[b.i], c); }
  for (const [d, v] of byDay) pts.push([Date.parse(d), v, d]);
  pts.unshift([pts[0][0] - 864e5, 0, '']);
  const x0 = pts[0][0], x1 = pts[pts.length-1][0] || x0 + 1, [lo, hi] = ext(pts.map(p => p[1]));
  const x = v => left + (v-x0)/(x1-x0 || 1)*(W-left-right), y = v => top + (hi-v)/(hi-lo)*(H-top-bottom);
  for (const t of ticks(lo, hi, 5)){ el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-8, y(t)+4, sgn(t, 0), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis'), 'stroke-width':1.3}, s);
  const y0 = new Date(x0).getUTCFullYear(), y1 = new Date(x1).getUTCFullYear();
  for (let yy = y0; yy <= y1 + 1; yy++){ const t = Date.UTC(yy, 0, 1); if (t > x0 && t < x1) text(s, x(t), H-9, yy, {'text-anchor':'middle'}); }
  const hf = Date.parse(R.hold_from); if (S.period === 'all' && hf > x0 && hf < x1){ el('rect', {x:x(hf), y:top, width:W-right-x(hf), height:H-top-bottom, fill:css('--wash')}, s); text(s, x(hf)+6, top+12, 'hold-out'); }
  const col = SUM.pl >= 0 ? css('--good') : css('--bad');
  el('path', {d: pts.map((p, j) => (j ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1)).join(''), fill:'none', stroke:css('--under'), 'stroke-width':2.2, 'stroke-linejoin':'round'}, s);
  const l = pts[pts.length-1]; el('circle', {cx:x(l[0]), cy:y(l[1]), r:4, fill:css('--under'), stroke:css('--surface'), 'stroke-width':2}, s);
  text(s, x(l[0]) + 8, y(l[1]) + 4, sgn(l[1], 1) + 'u', {class:'val', fill:col});
  const cross = el('line', {y1:top, y2:H-bottom, stroke:css('--axis'), opacity:0}, s), dot = el('circle', {r:4, fill:css('--under'), opacity:0}, s);
  const hr = el('rect', {x:left, y:top, width:W-left-right, height:H-top-bottom, class:'hit'}, s);
  hr.addEventListener('mousemove', e => { const bb = s.getBoundingClientRect(), xv = x0 + ((e.clientX - bb.left)/bb.width*W - left)/(W-left-right)*(x1-x0);
    let best = pts[0]; for (const p of pts) if (Math.abs(p[0]-xv) < Math.abs(best[0]-xv)) best = p;
    cross.setAttribute('x1', x(best[0])); cross.setAttribute('x2', x(best[0])); cross.setAttribute('opacity', 1); dot.setAttribute('cx', x(best[0])); dot.setAttribute('cy', y(best[1])); dot.setAttribute('opacity', 1);
    showTip(e, `<b>${best[2] || 'start'}</b><br>Running profit ${sgn(best[1], 2)} units`); });
  hr.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); });
}
function groups(key, label){
  const h = $('#view'), m = new Map(); for (const b of B){ const k = key(b); if (!m.has(k)) m.set(k, []); m.get(k).push(b); }
  let rows = [...m].map(([k, bs]) => ({k, ...summary(bs)}));
  rows.sort(label === 'Competition' ? (a, b) => b.n - a.n : (a, b) => String(a.k).localeCompare(String(b.k), undefined, {numeric:true}));
  if (label === 'Competition') rows = rows.slice(0, 20);
  const W = 860, rowH = 30, left = 190, right = 150, H = rows.length*rowH + 28, s = svg(h, W, H);
  const [lo, hi] = ext(rows.map(r => r.roi)); const x = v => left + (v-lo)/(hi-lo)*(W-left-right);
  for (const t of ticks(lo, hi, 5)){ el('line', {x1:x(t), x2:x(t), y1:0, y2:H-22, stroke:css('--grid')}, s); text(s, x(t), H-6, pct(t, 0), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:0, y2:H-22, stroke:css('--axis'), 'stroke-width':1.3}, s);
  rows.forEach((r, i) => { const yy = i*rowH + 7, hh = 16;
    text(s, left-10, yy+12, String(r.k).length > 26 ? String(r.k).slice(0, 25) + '…' : r.k, {'text-anchor':'end', class:'lab'});
    const a = x(0), b = x(r.roi), rr = Math.min(4, Math.abs(b-a)/2);
    el('path', {d: b >= a ? `M${a},${yy}H${b-rr}Q${b},${yy} ${b},${yy+rr}V${yy+hh-rr}Q${b},${yy+hh} ${b-rr},${yy+hh}H${a}Z` : `M${a},${yy}H${b+rr}Q${b},${yy} ${b},${yy+rr}V${yy+hh-rr}Q${b},${yy+hh} ${b+rr},${yy+hh}H${a}Z`,
      fill: r.lo > 0 ? css('--good') : r.hi < 0 ? css('--bad') : css('--ctrl')}, s);
    text(s, W-right+10, yy+12, `${pct(r.roi, 0)} · ${int(r.n)} bets · ${sgn(r.pl, 1)}u`, {class:'val'});
    const hr = el('rect', {x:0, y:i*rowH, width:W, height:rowH, class:'hit'}, s);
    hr.addEventListener('mousemove', e => showTip(e, `<b>${esc(r.k)}</b><br>${int(r.n)} bets, ${sgn(r.st, 1).replace('+', '')} units staked<br>Profit ${sgn(r.pl, 2)}u · return ${pct(r.roi)}<br>95% range ${pct(r.lo)} to ${pct(r.hi)}<br>won ${pctu(r.win)}`)); hr.addEventListener('mouseleave', hideTip); });
  h.insertAdjacentHTML('beforeend', `<p class="note">Return per unit staked. Green / red = the 95% range is entirely above / below zero; grey = could be chance.${label === 'Competition' && m.size > 20 ? ' Top 20 competitions by number of bets.' : ''}</p>`);
}
let LP = 0;
function list(h){
  const rows = B.slice().reverse(), s0 = LP*25, sl = rows.slice(s0, s0+25);
  h.innerHTML = `<div class="wide"><table><thead><tr><th class="l">Date</th><th class="l">Match</th><th class="l">Referee</th><th>Bet</th><th>Odds</th><th>Stake</th><th>Goals</th><th>Result</th></tr></thead><tbody>` +
    sl.map(b => { const i = b.i; return `<tr><td class="l">${D.d[i]}</td><td class="l">${esc(R.teams[D.h[i]] ?? '?')} – ${esc(R.teams[D.a[i]] ?? '?')}<div class="note" style="margin:0">${esc(R.leagues[D.l[i]])}</div></td>
      <td class="l">${esc(R.refs[D.r[i]].name)}</td><td><span class="pill ${b.over ? 'o' : 'u'}">${b.over ? 'Over' : 'Under'}</span></td><td>${b.price.toFixed(2)}</td><td>${b.stake}u</td><td>${D.g[i]}</td>
      <td class="${cls(b.pl)}">${sgn(b.pl, 2)}u</td></tr>`; }).join('') + '</tbody></table></div>' +
    `<div class="pager"><button id="lp-prev">‹</button><span>${s0+1}–${Math.min(s0+25, rows.length)} of ${rows.length.toLocaleString()}</span><button id="lp-next">›</button></div>`;
  $('#lp-prev').onclick = () => { if (LP > 0){ LP--; list(h); } }; $('#lp-next').onclick = () => { if ((LP+1)*25 < rows.length){ LP++; list(h); } };
}
// ------------------------------------------------------------------ referees tab
$('#r-lg').innerHTML = lgOpts('All competitions');
const TYPES = [...new Set(R.ltypes)].sort();
$('#r-type').innerHTML = '<option value="">All competition types</option>' + TYPES.map(t => `<option value="${t}">${t[0].toUpperCase() + t.slice(1)}</option>`).join('');
seg('#r-dir', [['all', 'Both'], ['over', 'Over'], ['under', 'Under']], 'rdir', () => { S.page = 0; drawRefs(); });
$('#r-q').oninput = e => { S.rq = e.target.value.toLowerCase(); S.page = 0; drawRefs(); };
$('#r-lg').onchange = e => { S.rlg = e.target.value; S.page = 0; drawRefs(); }; $('#r-type').onchange = e => { S.rtype = e.target.value; S.page = 0; drawRefs(); };
$('#r-sig').onchange = e => { S.rsig = e.target.checked; S.page = 0; drawRefs(); };
$('#pg-prev').onclick = () => { if (S.page > 0){ S.page--; drawRefs(); } }; $('#pg-next').onclick = () => { S.page++; drawRefs(); };
const COLS = [
  ['name', 'Referee', '', 'l'], ['lg', 'Main competition', 'and type', 'l'], ['n', 'Matches', 'with prices'], ['dir', 'Lean', 'vs the price'],
  ['over', 'Went over', 'vs ' + R.book + ' price'], ['eff', 'Effect', 'league-adjusted, shrunk'],
  ['q', 'Significance', 'q (corrected)'], ['steady', 'Steady', 'halves agree'], ['bets', 'Bets', 'with current settings'], ['pl', 'Profit', 'units'], ['roi', 'Return', 'per unit']];
function refRows(){
  const per = new Map(); for (const b of B){ const r = D.r[b.i]; if (!per.has(r)) per.set(r, []); per.get(r).push(b); }
  return R.refs.map((f, j) => { const s = per.has(j) ? summary(per.get(j)) : null;
    return {j, f, name:f.name, lg:R.leagues[f.lg], type:R.ltypes[f.lg], n:f.n, dir:f.eff, over:f.over, exp:f.exp, eff:f.eff, q:f.q ?? (f.p != null ? 2 : 3), steady:f.steady ? 1 : 0,
            bets:s ? s.n : 0, pl:s ? s.pl : null, roi:s ? s.roi : null, s}; });
}
function drawRefs(){
  let rows = refRows().filter(r => (!S.rq || r.name.toLowerCase().includes(S.rq)) && (S.rlg === '' || r.f.lg === +S.rlg) && (!S.rtype || r.type === S.rtype)
    && (S.rdir === 'all' || (S.rdir === 'over') === (r.eff > 0)) && (!S.rsig || (r.f.q != null && r.f.q < 0.05)));
  const k = S.sortK; rows.sort((a, b) => { const x = a[k], y = b[k]; if (x == null) return 1; if (y == null) return -1; return (typeof x === 'string' ? x.localeCompare(y) : x - y) * S.sortD; });
  const pages = Math.max(1, Math.ceil(rows.length/50)); S.page = Math.min(S.page, pages-1); const sl = rows.slice(S.page*50, S.page*50+50);
  const sigCell = f => { const q = f.q, c = q == null ? css('--ctrl') : q < 0.05 ? css('--good') : q < 0.2 ? '#c99a2e' : css('--ctrl');
    return `<span class="sig"><span class="dot" style="background:${c}"></span>${q == null ? (f.n < R.params.bh_min ? `<span class="note" style="margin:0">needs ${R.params.bh_min}</span>` : '–') : q < 0.001 ? '<0.001' : q.toFixed(3)}</span>`; };
  $('#r-table').innerHTML = '<thead><tr>' + COLS.map(([key, h, u, c]) => `<th data-k="${key}" class="${c || ''}">${h}${S.sortK === key ? (S.sortD > 0 ? ' ▲' : ' ▼') : ''}<span class="u">${u}</span></th>`).join('') + '</tr></thead><tbody>' +
    sl.map(r => `<tr data-j="${r.j}" class="${S.selRef === r.j ? 'sel' : ''}"><td class="l"><b>${esc(r.name)}</b></td><td class="l">${esc(r.lg)}${r.f.leagues > 1 ? `<span class="note" style="margin:0"> +${r.f.leagues - 1}</span>` : ''}<div class="note" style="margin:0">${esc(r.type)}</div></td><td>${int(r.n)}</td><td><span class="pill ${Math.abs(r.eff) < 0.005 ? 'n' : r.eff > 0 ? 'o' : 'u'}">${Math.abs(r.eff) < 0.005 ? 'None' : r.eff > 0 ? 'Over' : 'Under'}</span></td>
      <td>${pctu(r.over)} <span class="note">vs ${pctu(r.exp)}</span></td><td>${pp(r.eff)}</td><td>${sigCell(r.f)}</td><td>${r.f.steady ? '✓' : ''}</td>
      <td>${r.bets ? int(r.bets) : '<span class="note">–</span>'}</td><td class="${cls(r.pl)}">${r.pl == null ? '' : sgn(r.pl, 1)}</td><td class="${cls(r.roi)}">${r.roi == null ? '' : pct(r.roi, 0)}</td></tr>`).join('') + '</tbody>';
  $$('#r-table th').forEach(th => th.onclick = () => { const kk = th.dataset.k; S.sortD = S.sortK === kk ? -S.sortD : (['name', 'lg', 'type', 'q'].includes(kk) ? 1 : -1); S.sortK = kk; drawRefs(); });
  $$('#r-table tbody tr').forEach(tr => tr.onclick = () => { S.selRef = +tr.dataset.j; drawRefs(); refDetail(); });
  $('#pg-info').textContent = rows.length ? `${S.page*50+1}–${Math.min(S.page*50+50, rows.length)} of ${rows.length.toLocaleString()} referees` : 'No referees match';
}
function refDetail(){
  const j = S.selRef, f = R.refs[j], box = $('#r-detail'); box.hidden = false;
  const mine = B.filter(b => D.r[b.i] === j), s = summary(mine);
  box.innerHTML = `<div style="display:flex;justify-content:space-between;gap:10px"><h2>${esc(f.name)} <span class="note">· ${esc(R.leagues[f.lg])} · last match ${f.last}</span></h2><button class="plain" id="rd-x">Close</button></div>
    <p class="note">Whole period: ${int(f.n)} matches, ${pctu(f.over)} over vs ${pctu(f.exp)} priced, league-adjusted effect ${pp(f.eff)}, ${f.q != null ? 'q = ' + f.q.toFixed(3) : 'not in the corrected test'}.
    With the current strategy settings: ${mine.length ? `${int(s.n)} bets, <b class="${cls(s.pl)}">${sgn(s.pl, 1)}u</b> (${pct(s.roi)})` : 'no bets'}.</p>
    <h3>Each year: went over vs priced</h3><div class="legend"><span class="key"><span class="dot" style="background:var(--over)"></span>Went over</span><span class="key"><span class="dot" style="background:var(--ctrl)"></span>${esc(R.book)} priced</span></div><div id="rd-chart"></div>`;
  $('#rd-x').onclick = () => { box.hidden = true; S.selRef = null; drawRefs(); };
  const Y = f.years, h = $('#rd-chart'), W = 860, H = 200, top = 10, bottom = 34, left = 48, right = 10, s2 = svg(h, W, H);
  const mx = Math.max(.6, ...Y.map(r => Math.max(r[2], r[3]))) * 1.1, y = v => top + (1 - v/mx)*(H-top-bottom), bw = (W-left-right)/Y.length, w = Math.min(22, bw*.3);
  for (const t of ticks(0, mx, 4)){ el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s2); text(s2, left-7, y(t)+4, Math.round(t*100) + '%', {'text-anchor':'end'}); }
  Y.forEach((r, i) => { const cx = left + bw*i + bw/2;
    el('path', {d: vbarPath(cx - w - 1, w, y(0), y(r[2])), fill:css('--over')}, s2); el('path', {d: vbarPath(cx + 1, w, y(0), y(r[3])), fill:css('--ctrl')}, s2);
    text(s2, cx, H-bottom+15, r[0], {'text-anchor':'middle', class:'lab'}); text(s2, cx, H-bottom+28, int(r[1]) + ' m', {'text-anchor':'middle'});
    const hr = el('rect', {x:cx-bw/2, y:top, width:bw, height:H-top-bottom, class:'hit'}, s2);
    hr.addEventListener('mousemove', e => showTip(e, `<b>${r[0]}</b> · ${r[1]} matches<br>Went over ${pctu(r[2])}<br>Priced ${pctu(r[3])}<br>League-adjusted ${pp(r[4])}`)); hr.addEventListener('mouseleave', hideTip); });
  box.scrollIntoView({behavior:'smooth', block:'nearest'});
}
// ------------------------------------------------------------------ how it works
$('#how').innerHTML = `<h2>How it works</h2>
<details open><summary>What is being tested</summary><p>For each match, ${esc(R.book)}'s last price before kickoff gives the chance of 3+ goals (margin removed: ${pctu(R.margin, 1)} on average). A referee "leans over" when his matches go over more often than those prices said, after allowing for his league's own tendency.</p></details>
<details><summary>Why there is no look-ahead</summary><ul>
<li>Every number a bet uses - the referee's record, his significance, effect size and the league adjustment - comes only from matches on <b>earlier days</b>. A day's results are added after its bets are placed.</li>
<li>The multiple-testing correction is redone each day across the referees known that day (those with ${R.params.bh_min}+ matches).</li>
<li>Prices are the last ones before kickoff. Bets settle on 90 minutes, like the market.</li>
<li>A time-travel test re-ran everything with all later matches deleted: ${R.leak.skipped ? 'skipped' : R.leak.tests.map(t => `${t.day}: ${t.mismatches} values changed`).join(' · ')}.</li>
<li>Nothing is tuned in the script. You pick settings here - which <i>is</i> tuning if you pick by looking at results, so pick on the discovery period and judge on the hold-out (from ${R.hold_from}).</li></ul></details>
<details><summary>The columns</summary><ul>
<li><b>Effect</b>: how many percentage points more (or fewer) of his matches went over than priced, adjusted for the league and shrunk towards zero (as if he had ${R.params.shrink} extra average matches) so small samples don't look extreme.</li>
<li><b>Significance (q)</b>: the chance of a lean this strong by luck, corrected for testing every referee at once (Benjamini-Hochberg). Raw p ignores that correction: with hundreds of referees, some will look significant by luck.</li>
<li><b>Steady</b>: the first and second half of his matches lean the same way.</li>
<li><b>Bets / Profit / Return</b>: what the strategy on the Strategy tab did on his matches, betting only with what was known at the time.</li></ul></details>
<details><summary>Margin</summary><p>Every price includes the bookmaker's margin (${pctu(R.margin, 1)} on average here${R.devig ? `; the fair chance is taken out with the ${esc(R.devig)} method` : ''}). A bet is only placed when the model's chance - the fair price, plus the league's earlier tendency, plus the referee's shrunk effect - times the price you would actually get leaves at least the chosen expected profit. So the margin is paid for before any bet counts as worth taking, and every profit shown is after margin.</p></details>
<details><summary>Staking</summary><p>"By effect size" stakes 1 unit per chosen number of percentage points of estimated edge (rounded to quarter units, from 0.25 up to the cap). Flat stakes 1 unit on every bet.</p></details>
<details><summary>Reading the result</summary><p>The 95% range on return comes from the spread of bet results. If it includes zero the profit could be luck. Best prices often include stale quotes that would not be honoured; ${esc(R.book)} prices are the realistic test.</p></details>`;
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; update(); if (S.selRef != null) refDetail(); };
update();
</script></body></html>
'''

if __name__ == "__main__":
    main()
