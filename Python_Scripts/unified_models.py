#!/usr/bin/env python3
"""
unified_models.py - base, tactical and player-form models, alone and combined
==============================================================================

Put this file next to backtest_models.py, tactical_model.py and form_downturn.py
(it imports all three, so every feature and every leak safeguard is the same code).

WHAT IT RUNS
------------
1. Three feature families, each from its own leak-proof engine:
     B  Base      backtest_models.Engine: form, Elo, Poisson/xG ratings, rest, travel,
                  table, line-up strength, referee, weather, market microstructure
     T  Tactical  tactical_model.TacticalEngine: team style & execution, starting-XI
                  technique, matchups, in-game habits, formation, style memory
     F  Form      form_downturn.py: every starter's context-adjusted confidence
                  signals and walk-forward downturn risk (next game / sustained /
                  losing his place), rolled up to the team: averages, the most at-risk
                  players, importance-weighted risk and HOW MANY starters are in a
                  downturn.
2. All 7 coalitions - B, T, F, B+T, B+F, T+F, B+T+F - each trained three ways:
     1X2 vs market  gradient boosting of the closing market's errors (main test)
     O/U 2.5        the same for the Over/Under 2.5 market
     No odds        gradient boosting from the family alone (raw football content)
   Walk-forward (retrained every few months on earlier matches only), shrinkage of
   each correction chosen inside the training window, scored on discovery and on a
   later hold-out.
3. Attribution: Shapley values split the combined gain fairly between B, T and F
   (exact, from the 8 coalition values) with week-block bootstrap intervals; a
   marginal-value matrix (what adding X to S is worth); Holm-adjusted p-values.
4. Betting check at sharp / average / best prices with thresholds frozen on the
   discovery period, and a sanity check on best prices (if the market itself
   "profits" there, the best prices are stale or unbettable).
5. Players in a downturn: results vs the market by how many starters on each side
   are in a downturn (observed and model-based), and where form adds most.
6. Player stats: next-match shots, shots on target, goals, assists, key passes,
   fouls, tackles+interceptions, touches, passes, dribbles and rating, predicted
   for every starter by 8 feature mixes (player history P alone, then + B / T / F
   and every combination).
7. Matchups: does a player struggle against particular teams, opponent STYLES
   (pressing, possession, direct play, high line, aerial, aggression), or the
   direct opponent he faces (winger vs full-back, striker vs centre-backs)? Each
   is tested for persistence: the history must predict LATER games, or it is noise.

NO LOOK-AHEAD
-------------
* B and T: the day-by-day engines of backtest_models.py / tactical_model.py,
  re-checked by their time-travel tests. F: form_downturn.py's baselines from
  earlier days only, purged training, and its own time-travel test.
* The form risk used in a match comes from a form model trained only on labels
  that ENDED before that match's prediction block; an audit asserts it for
  every prediction.
* Match models: walk-forward, training rows strictly before the predicted month;
  audited per block. Player-stat and matchup histories use earlier days only.
* Thresholds (downturn cut-offs, bet edges) are frozen before they are used.

USAGE
-----
  python3 unified_models.py --cache .cache
  python3 unified_models.py --cache .cache --quick          # fewer refits, smaller samples
  python3 unified_models.py --leagues 8,82 --export-csv unified.csv
"""
from __future__ import annotations

import argparse
import copy
import html as htmlmod
import json
import math
import os
import signal
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEDED = ("backtest_models.py", "tactical_model.py", "form_downturn.py")


def _bootstrap():
    missing = [f for f in NEEDED if not (HERE / f).exists()]
    if missing:
        sys.exit(f"unified_models.py needs {', '.join(missing)} in the same folder ({HERE}).")
    import importlib.util
    if any(importlib.util.find_spec(m) is None for m in ("numpy", "pandas", "scipy", "sklearn")):
        subprocess.run([sys.executable, str(HERE / "backtest_models.py"), "--install-only"])


if __name__ == "__main__":
    _bootstrap()

sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest_models as bt  # noqa: E402
import tactical_model as tm  # noqa: E402
import form_downturn as fd  # noqa: E402
from backtest_models import log  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
FAMS = ("B", "T", "F")
FAM_NAME = {"B": "Base", "T": "Tactical", "F": "Form"}
COALS = ["B", "T", "F", "BT", "BF", "TF", "BTF"]
DAY0 = pd.Timestamp("2000-01-01")


def cname(c):
    return " + ".join(FAM_NAME[x] for x in c)


class UProgress(bt.Progress):
    WEIGHTS = {"reading database": 3, "reading tactical data": 3, "building features": 3, "tactical features": 6, "form features": 6,
               "time-travel test": 8, "form models": 12, "coalition models": 35, "player stats": 16, "scoring": 8}


# =============================================================================
# Form: walk-forward player models, then roll up to (fixture, team)
# =============================================================================
def form_args(args):
    f = copy.copy(args)
    f.retrain_months, f.train_window_days, f.min_train = args.form_retrain_months, args.form_window_days, args.form_min_train
    f.max_train_rows, f.burn_in_days = args.form_max_rows, args.burn_in_days
    return f


def run_form(fargs, db_dsn, raw_override=None):
    """form_downturn pipeline: features, labels, frozen thresholds, walk-forward risk for 3 targets (full feature set)."""
    if raw_override is None:
        fdb = fd.DB(db_dsn)
        raw = fd.extract(fdb, fargs)
    else:
        raw = raw_override
    raw = fd.prepare_raw(raw)
    if bt.PROGRESS:
        bt.PROGRESS.start("form features", 1)
    d, side, feat, avail, cov = fd.build_all(raw, fargs)
    feat = feat.loc[:, (feat[d.ghost.to_numpy() == 0].notna().mean() > 0.01) & (feat.nunique() > 1)]
    fx = side["fx"]
    tmq = pd.concat([fx[["fixture_id", "home_id", "t"]].rename(columns={"home_id": "team_id"}), fx[["fixture_id", "away_id", "t"]].rename(columns={"away_id": "team_id"})])
    tmq["team_id"] = tmq.team_id.astype("int64")
    tmq = tmq.sort_values(["team_id", "t", "fixture_id"], kind="stable")
    tmq["tidx"] = tmq.groupby("team_id").cumcount()
    side["team_seq"] = tmq
    ptc = pd.factorize(pd.MultiIndex.from_arrays([d.player_id, d.team_id]))[0]
    k2 = ptc.astype(np.int64) * 10_000_000 + np.nan_to_num(d.tidx.to_numpy(float), nan=-1).astype(np.int64)
    o2 = np.argsort(k2, kind="stable")
    side["k2"], side["ks"] = k2, k2[o2]
    side["cst"] = np.concatenate([[0], np.cumsum(np.nan_to_num(d.started.to_numpy(float))[o2])])
    fcols = list(feat.columns)
    df = pd.concat([d, feat.add_prefix("f_")], axis=1)
    perf_name = fd.labels(df, side, fargs)
    feats = ["f_" + c for c in fcols]
    real = df.ghost.to_numpy() == 0
    tmin, tmax = df.t[real].min(), df.t[real].max()
    blocks = fd.months_blocks(tmin, tmax, fargs)
    first = blocks[0][0] if blocks else tmax
    taus = {}
    for tg in ("next", "sustained"):
        dv, le = df[f"d_{tg}"].to_numpy(float), df[f"lend_{tg}"].to_numpy(float)
        early = np.isfinite(dv) & (le < first)
        if early.sum() < 300:
            early = np.isfinite(dv) & (df.t.to_numpy() <= np.nanquantile(df.t[np.isfinite(dv)], 0.25))
        taus[tg] = float(np.quantile(dv[early], fargs.downturn_quantile))
        df[f"y_{tg}"] = np.where(np.isfinite(dv), (dv <= taus[tg]).astype(float), np.nan)
    if bt.PROGRESS:
        bt.PROGRESS.start("form models", max(1, 3 * len(blocks)))
    t = df.t.to_numpy(float)
    X = df[feats].to_numpy(np.float32)
    risk, audit, quality = {}, [], {}
    rng = np.random.default_rng(3)
    for tg in ("next", "sustained", "dropped"):
        y, le = df[f"y_{tg}"].to_numpy(float), df[f"lend_{tg}"].to_numpy(float)
        ok = np.isfinite(y) & real
        P = np.full(len(df), np.nan)
        cut = np.full(len(df), np.nan)
        # predictions are needed for EVERY line-up row in the block (labelled or not): they are pre-match features
        for b0, b1 in blocks:
            tr = ok & (le < b0) & (t >= b0 - fargs.train_window_days)
            te = (t >= b0) & (t < b1) & real
            if tr.sum() < fargs.min_train or te.sum() == 0:
                if bt.PROGRESS:
                    bt.progress(1, f"form {tg}")
                continue
            idx = np.where(tr)[0]
            if len(idx) > fargs.max_train_rows:
                idx = np.sort(rng.choice(idx, fargs.max_train_rows, replace=False))
            m = fd.fit_tuned(X[idx], y[idx], t[idx], fargs)
            P[te] = fd.proba(m, X[te])
            cut[te] = b0
            audit.append({"target": tg, "block": str((DAY0 + pd.Timedelta(days=b0)).date()), "train_rows": int(len(idx)),
                          "max_label_end": str((DAY0 + pd.Timedelta(days=float(le[idx].max()))).date()), "ok": bool(le[idx].max() < b0)})
            if bt.PROGRESS:
                bt.progress(1, f"form {tg} {(DAY0 + pd.Timedelta(days=b0)):%Y-%m}")
        # forward rows (ghosts) are not needed here: match rows always exist
        risk[tg] = P
        ev = ok & np.isfinite(P)
        quality[tg] = {"auc": fd.auc(y[ev], P[ev]), "n": int(ev.sum()), "rate": float(y[ev].mean()) if ev.any() else None}
        log(f"  form model {tg}: out-of-sample AUC {quality[tg]['auc'] or float('nan'):.3f} on {ev.sum():,} labelled rows")
        df[f"risk_{tg}"] = P
        df[f"riskcut_{tg}"] = cut
    return {"raw": raw, "d": df, "side": side, "feat": feat, "fcols": fcols, "avail": avail, "cov": cov, "taus": taus,
            "audit": audit, "quality": quality, "perf_name": perf_name, "first_block": first}


FO_COLS = ["rating_d", "rating_s", "rating_l", "rating_vol", "pass_acc_d", "passes_d", "touches_d", "drib_att_d", "drib_succ_d", "shot_acc_d",
           "shots_d", "turnover_rate_d", "fouls_d", "cards_s", "duel_win_d", "keypass_rate_d", "backward_share_d", "played_to_s", "hooked_s",
           "finishing_l", "drought_xg", "mins_14d", "mins_28d", "rest_days", "start_drop", "role_familiarity", "rumours_30d", "days_since_injury"]


def form_team(F, args):
    """(fixture, team) features from the form pipeline. Only pre-match facts: who starts (line-up) and each starter's
    history-based features / walk-forward risk. Counts use thresholds frozen before the first predicted block."""
    d = F["d"]
    st = (d.ghost == 0) & (d.started == 1)
    S = d.loc[st, ["fixture_id", "team_id", "player_id", "t", "role"]].copy()
    imp = d.loc[st, "f_starts_20"].to_numpy(float) if "f_starts_20" in d else np.ones(st.sum())
    S["imp"] = np.nan_to_num(imp, nan=0.3) + 0.05
    for tg in ("next", "sustained", "dropped"):
        S[f"r_{tg}"] = d.loc[st, f"risk_{tg}"].to_numpy(float)
    for c in FO_COLS:
        if "f_" + c in d:
            S[c] = d.loc[st, "f_" + c].to_numpy(float)
    # frozen thresholds: observed downturn = recent-minus-usual adjusted rating in the bottom 20% of starters seen before the
    # first form block; model downturn = risk in the top 20% of the first 180 days of out-of-sample predictions
    first = F["first_block"]
    early = S.t < first
    thr = {}
    if "rating_d" in S and S.rating_d[early].notna().sum() > 200:
        thr["obs"] = float(S.rating_d[early].quantile(0.2))
    else:
        thr["obs"] = float(S.rating_d.quantile(0.2)) if "rating_d" in S else np.nan
    for tg in ("next", "sustained"):
        r = S[f"r_{tg}"]
        t0 = S.t[r.notna()].min() if r.notna().any() else np.nan
        e = r[(S.t < t0 + 180) & r.notna()]
        thr[tg] = float(e.quantile(0.8)) if len(e) > 100 else (float(r.quantile(0.8)) if r.notna().any() else np.nan)
    S["down_obs"] = (S.rating_d <= thr["obs"]).astype(float).where(S.rating_d.notna()) if "rating_d" in S else np.nan
    S["down_next"] = (S.r_next >= thr["next"]).astype(float).where(S.r_next.notna())
    S["down_sus"] = (S.r_sustained >= thr["sustained"]).astype(float).where(S.r_sustained.notna())
    g = S.groupby(["fixture_id", "team_id"])
    out = pd.DataFrame(index=g.size().index)
    out["n_xi"] = g.size()
    for tg in ("next", "sustained", "dropped"):
        c = f"r_{tg}"
        out[f"{tg}_mean"] = g[c].mean()
        out[f"{tg}_max"] = g[c].max()
        w = S[c] * S.imp
        out[f"{tg}_wmean"] = w.groupby([S.fixture_id, S.team_id]).sum(min_count=1) / S.imp.where(S[c].notna()).groupby([S.fixture_id, S.team_id]).sum()
    Ss = S.sort_values(["fixture_id", "team_id", "r_next"], ascending=[True, True, False], kind="stable")
    top = Ss[Ss.groupby(["fixture_id", "team_id"]).cumcount() < 3]
    out["next_top3"] = top.groupby(["fixture_id", "team_id"]).r_next.mean()
    Ss = S.sort_values(["fixture_id", "team_id", "imp"], ascending=[True, True, False], kind="stable")
    key = Ss[Ss.groupby(["fixture_id", "team_id"]).cumcount() < 4]
    out["key4_sustained"] = key.groupby(["fixture_id", "team_id"]).r_sustained.mean()
    out["key4_next"] = key.groupby(["fixture_id", "team_id"]).r_next.mean()
    for c in ("down_obs", "down_next", "down_sus"):
        out[f"n_{c}"] = g[c].sum(min_count=1)
    for c in FO_COLS:
        if c in S:
            out[f"obs_{c}"] = g[c].mean()
    out["obs_rating_d_min"] = g["rating_d"].min() if "rating_d" in S else np.nan
    out = out.reset_index()
    # players at risk (for the match explorer)
    names = dict(zip(F["raw"]["players"].player_id.astype("int64"), F["raw"]["players"].name)) if len(F["raw"]["players"]) else {}
    Ss = S.dropna(subset=["r_next"]).sort_values(["fixture_id", "team_id", "r_next"], ascending=[True, True, False], kind="stable")
    top = Ss[Ss.groupby(["fixture_id", "team_id"]).cumcount() < 3]
    atrisk = defaultdict(list)
    for r in top.itertuples():
        atrisk[(int(r.fixture_id), int(r.team_id))].append([names.get(int(r.player_id), str(int(r.player_id))), r.role, round(float(r.r_next), 3),
                                                              None if not np.isfinite(r.r_sustained) else round(float(r.r_sustained), 3)])
    return out, thr, atrisk, S


def to_match(team_rows, base_ids, prefix="fm_"):
    """(fixture, team) rows -> one row per fixture with h_/a_/d_ columns."""
    cols = [c for c in team_rows.columns if c not in ("fixture_id", "team_id")]
    h = base_ids[["fixture_id", "home_id"]].merge(team_rows, left_on=["fixture_id", "home_id"], right_on=["fixture_id", "team_id"], how="left")
    a = base_ids[["fixture_id", "away_id"]].merge(team_rows, left_on=["fixture_id", "away_id"], right_on=["fixture_id", "team_id"], how="left")
    out = pd.DataFrame({"fixture_id": base_ids.fixture_id.to_numpy()})
    for c in cols:
        out[f"{prefix}h_{c}"] = h[c].to_numpy(float)
        out[f"{prefix}a_{c}"] = a[c].to_numpy(float)
        out[f"{prefix}d_{c}"] = out[f"{prefix}h_{c}"] - out[f"{prefix}a_{c}"]
    return out


# =============================================================================
# Coalition models (walk-forward chunks, run in parallel through tactical_model.run_chunks)
# =============================================================================
def _uchunk(blocks):
    from sklearn.ensemble import HistGradientBoostingClassifier
    W = tm._TW
    df, G, args = W["df"], W["feats"], W["args"]
    t0 = time.time()
    out = {"P": [], "Q": [], "S": [], "N": [], "shrink": [], "audit": []}
    shrink = {}
    y = df.y.to_numpy()
    yo = df.y_over.to_numpy(float)
    has_mk = df.mk_lr_h.notna().to_numpy()
    has_ou = df.ou_p_o.notna().to_numpy() if "ou_p_o" in df else np.zeros(len(df), bool)
    for k, block in enumerate(blocks):
        m0 = block[0].start_time
        train = ((df.day < m0) & (df.day >= m0 - pd.Timedelta(days=args.train_window_days))).to_numpy() & has_mk
        test = df.month.isin(block).to_numpy() & has_mk
        if train.sum() < args.min_train or test.sum() == 0:
            tm._treport(float(train.sum()), str(block[0]))
            continue
        out["audit"].append({"block": str(block[0]), "train_last_day": str(df.day[train].max().date()), "test_first_day": str(df.day[test].min().date()),
                             "ok": bool(df.day[train].max() < df.day[test].min()), "train": int(train.sum()), "test": int(test.sum())})
        tr, te = df[train], df[test]
        idx = np.where(test)[0]
        frq = np.bincount(y[train], minlength=3) / train.sum()
        out["N"].append((idx, np.tile(frq, (len(idx), 1))))
        Mtr, Mte = tr[["p_h", "p_d", "p_a"]].to_numpy(float), te[["p_h", "p_d", "p_a"]].to_numpy(float)
        Ytr = np.eye(3)[y[train]]
        for ci, c in enumerate(COALS):
            cols = G["mk"][c]
            Xtr, Xte = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
            if c not in shrink or k == 0:
                shrink[c] = tm.tune_shrink(Xtr, Ytr, Mtr)
                out["shrink"].append(("1x2", c, str(block[0]), shrink[c]))
            Gp, _ = tm.boost_resid(Xtr, Ytr, Mtr, Xte, seed=ci)
            out["P"].append((c, idx, tm.combine(Mte, Gp, shrink[c])))
            if not args.no_standalone:
                sc = G["raw"][c]
                Xa, Xb = tm.nonempty(tr[sc].to_numpy(np.float32), te[sc].to_numpy(np.float32))
                if Xa.shape[1]:
                    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=120, max_bins=63, l2_regularization=2.0,
                                                         **tm._hgb_extra(0.4), early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=ci)
                    clf.fit(Xa, y[train])
                    full = np.zeros((len(Xb), 3))
                    full[:, clf.classes_.astype(int)] = clf.predict_proba(Xb)
                    out["S"].append((c, idx, full))
            if not args.no_ou:
                otr, ote = train & has_ou, test & has_ou
                if otr.sum() >= args.min_train and ote.sum():
                    oc = G["ou"][c]
                    Xo, Xoe = df.loc[otr, oc].to_numpy(np.float32), df.loc[ote, oc].to_numpy(np.float32)
                    Mo, Moe = df.loc[otr, ["ou_p_o"]].to_numpy(float), df.loc[ote, ["ou_p_o"]].to_numpy(float)
                    Yo = yo[otr][:, None]
                    key = "ou_" + c
                    if key not in shrink or k == 0:
                        shrink[key] = tm.tune_shrink(Xo, Yo, Mo)
                        out["shrink"].append(("ou", c, str(block[0]), shrink[key]))
                    Go, _ = tm.boost_resid(Xo, Yo, Mo, Xoe, seed=10 + ci)
                    out["Q"].append((c, np.where(ote)[0], tm.combine(Moe, Go, shrink[key])))
        tm._treport(float(train.sum()), str(block[0]))
    out["seconds"], out["span"] = time.time() - t0, f"{blocks[0][0]}..{blocks[-1][-1]}"
    return out


def coalition_walk_forward(df, G, args):
    n = len(df)
    P = {c: np.full((n, 3), np.nan) for c in COALS}
    S = {c: np.full((n, 3), np.nan) for c in COALS}
    Q = {c: np.full(n, np.nan) for c in COALS}
    NULL = np.full((n, 3), np.nan)
    blocks, chunks = tm.blocks_for(df, args)
    plan = bt.compute_plan(args, len(chunks), float(df.memory_usage(deep=True).sum()))
    log(f"coalition models: 7 coalitions x (1X2{'' if args.no_ou else ' + O/U'}{'' if args.no_standalone else ' + no-odds'}), "
        f"{len(blocks)} retrains in {len(chunks)} chunks, {plan['workers']} worker(s) x {plan['threads']} thread(s)")
    res = tm.run_chunks(_uchunk, (df, G, args), chunks, plan, "coalition models", tm.work_of(df, blocks, args))
    shrink, audit = defaultdict(list), []
    for r in sorted(res, key=lambda r: r["span"]):
        for c, idx, p in r["P"]:
            P[c][idx] = p
        for c, idx, p in r["S"]:
            S[c][idx] = p
        for c, idx, p in r["Q"]:
            Q[c][idx] = p
        for idx, p in r["N"]:
            NULL[idx] = p
        for mk, c, b, s in r["shrink"]:
            shrink[f"{mk}:{c}"].append([b, s])
        audit += r["audit"]
    return P, S, Q, NULL, dict(shrink), audit, plan


# =============================================================================
# Scoring helpers
# =============================================================================
def ll3(P3, y):
    return -np.log(np.clip(P3[np.arange(len(y)), y], 1e-9, 1))


def ll2(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_mean(x, week, B=1000, seed=4):
    """Mean of x with a 95% interval and two-sided p (vs 0) from resampling whole weeks."""
    ok = np.isfinite(x)
    if ok.sum() < 30:
        return None
    if np.abs(x[ok]).max() < 1e-5:      # identical predictions (e.g. a model that fell back to the market): exactly no difference
        return {"v": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "n": int(ok.sum()), "identical": True}
    g = pd.DataFrame({"x": x[ok], "w": week[ok]}).groupby("w").x.agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(s), (B, len(s)))
    bs = s[ix].sum(1) / c[ix].sum(1)
    return {"v": float(s.sum() / c.sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B)), "n": int(c.sum())}


def holm(pvals):
    keys = [k for k, p in pvals.items() if p is not None]
    order = sorted(keys, key=lambda k: pvals[k])
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def shapley_weights(n=3):
    """weight of each (S, i) term: |S|!(n-|S|-1)!/n!"""
    return {s: math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)}


def coal_key(members):
    s = "".join(x for x in FAMS if x in members)
    return s if s else "M"


def shapley_rows(L):
    """L: coalition key ('M' = market / null) -> per-row loss. Returns per-row Shapley contributions of B, T, F
    (contribution = loss saved). Their sum equals L['M'] - L['BTF'] row by row."""
    w = shapley_weights()
    out = {}
    for i in FAMS:
        others = [x for x in FAMS if x != i]
        tot = 0
        for r in range(3):
            for S in combinations(others, r):
                tot = tot + w[r] * (L[coal_key(S)] - L[coal_key(S + (i,))])
        out[i] = tot
    return out


def calib3(P3, y, bins=20):
    pp, yy = P3.ravel(), np.eye(3)[y].ravel()
    b = np.clip((pp * bins).astype(int), 0, bins - 1)
    c = pd.DataFrame({"b": b, "p": pp, "y": yy}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return [[round(float(r.p), 4), round(float(r.y), 4), int(r.n)] for r in c.itertuples() if r.n >= 30]


def calib2(p, y, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    c = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return [[round(float(r.p), 4), round(float(r.y), 4), int(r.n)] for r in c.itertuples() if r.n >= 30]


def downsample(xs, ys, n=360):
    if len(xs) == 0:
        return []
    step = max(1, len(xs) // n)
    ix = list(range(0, len(xs), step)) + [len(xs) - 1]
    return [[xs[i], round(float(ys[i]), 3)] for i in dict.fromkeys(ix)]


# =============================================================================
# Coalition evaluation: accuracy, gains, Shapley, marginal values, betting
# =============================================================================
def eval_market(df, PRED, mk, y, loss, E, hold, week, kind):
    """PRED: coalition -> predictions; mk: market (or null) predictions; loss(pred, rows)->per-row loss."""
    n = len(df)
    have = E.copy()
    for c in COALS:
        have &= np.isfinite(PRED[c]).all(axis=1) if PRED[c].ndim == 2 else np.isfinite(PRED[c])
    have &= np.isfinite(mk).all(axis=1) if mk.ndim == 2 else np.isfinite(mk)
    L = {}
    for key, p in [("M", mk)] + [(c, PRED[c]) for c in COALS]:
        a = np.full(n, np.nan)
        a[have] = loss(p[have], have)
        L[key] = a
    periods = {"discovery": have & ~hold, "holdout": have & hold, "all": have}
    res = {"periods": {}, "n_rows": int(have.sum())}
    for pn, pm in periods.items():
        if pm.sum() < 100:
            continue
        R = {"models": {}, "shapley": {}, "marginal": {}}
        for key in ["M"] + COALS:
            m = {"loss": float(np.mean(L[key][pm])), "n": int(pm.sum())}
            if key != "M":
                m["gain"] = boot_mean((L["M"] - L[key])[pm], week[pm])
                pk = PRED[key][pm]
                m["same_as_market"] = float((np.abs(pk - mk[pm]).reshape(len(pk), -1).max(axis=1) < 1e-6).mean())
                if key != "B" and "B" in key:
                    m["vs_base"] = boot_mean((L["B"] - L[key])[pm], week[pm])
            if kind == "1x2":
                p = mk if key == "M" else PRED[key]
                pb = np.clip(p[pm], 1e-9, 1)
                Y = np.eye(3)[y[pm]]
                m["acc"] = float((pb.argmax(1) == y[pm]).mean())
                m["brier"] = float(((pb - Y) ** 2).sum(1).mean())
                cp, cy = np.cumsum(pb, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
                m["rps"] = float(((cp - cy) ** 2).sum(1).mean() / 2)
            else:
                p = mk if key == "M" else PRED[key]
                m["brier"] = float(((p[pm] - y[pm]) ** 2).mean())
                m["acc"] = float(((p[pm] > .5) == (y[pm] == 1)).mean())
            R["models"][key] = m
        sh = shapley_rows({k: L[k][pm] for k in L})
        for f in FAMS:
            R["shapley"][f] = boot_mean(sh[f], week[pm])
        for r in range(3):
            for S in combinations(FAMS, r):
                for i in FAMS:
                    if i in S:
                        continue
                    a, b = coal_key(S), coal_key(S + (i,))
                    R["marginal"][f"{a}+{i}"] = boot_mean((L[a] - L[b])[pm], week[pm])
        res["periods"][pn] = R
    hp = {c: (res["periods"].get("holdout", {}).get("models", {}).get(c, {}).get("gain") or {}).get("p") for c in COALS}
    res["holm"] = holm(hp)
    # cumulative loss saved vs market (all evaluated rows, in date order)
    order = np.argsort(df.day.to_numpy()[have], kind="stable")
    dts = df.day.dt.strftime("%Y-%m-%d").to_numpy()[have][order]
    res["cum"] = {c: downsample(list(dts), np.cumsum((L["M"] - L[c])[have][order])) for c in COALS}
    # by league (hold-out), the full coalition and base alone
    lg = df.league.fillna("").to_numpy()
    rows = []
    hm = have & hold
    for name in pd.unique(lg[hm]):
        mm = hm & (lg == name)
        if mm.sum() < 150:
            continue
        rows.append({"league": name, "n": int(mm.sum()), **{c: float(np.mean((L["M"] - L[c])[mm])) for c in COALS}})
    res["leagues"] = sorted(rows, key=lambda r: -r["n"])
    res["_L"] = L
    res["_have"] = have
    return res


def betting(df, P, y, E, hold, week, args):
    """Back the outcome with the largest edge when edge >= tau; tau chosen on discovery only."""
    sets = {"sharp": ["pin_h", "pin_d", "pin_a"], "avg": ["avg_h", "avg_d", "avg_a"], "best": ["best_h", "best_d", "best_a"]}
    sets = {k: v for k, v in sets.items() if all(c in df for c in v) and df[v[0]].notna().mean() > 0.2}
    mk = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    out = {}
    for pr, cols in sets.items():
        prices = df[cols].to_numpy(float)
        rec = {}
        for c in ["M"] + COALS:
            P3 = mk if c == "M" else P[c]
            okp = E & np.isfinite(P3).all(1) & np.isfinite(prices).all(1)
            best_tau, best_roi = 0.05, -9
            for tau in bt.TAUS:
                bet, sel, e, price, pl = bt.bet_table(P3, prices, y, args.max_odds, tau)
                b = bet & okp & ~hold
                if b.sum() >= args.min_bets and pl[b].mean() > best_roi:
                    best_tau, best_roi = tau, float(pl[b].mean())
            bet, sel, e, price, pl = bt.bet_table(P3, prices, y, args.max_odds, best_tau)
            r = {"tau": best_tau}
            for pn, pm in (("discovery", ~hold), ("holdout", hold)):
                b = bet & okp & pm
                lo, hi, pv = bt.boot_roi(pl[b], week[b]) if b.sum() >= 5 else (None, None, None)
                r[pn] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv, "profit": float(pl[b].sum())}
            b = bet & okp
            order = np.argsort(df.day.to_numpy()[b], kind="stable")
            r["cum"] = downsample(list(df.day.dt.strftime("%Y-%m-%d").to_numpy()[b][order]), np.cumsum(pl[b][order]), 300)
            rec[c] = r
        out[pr] = rec
    return out


def base_subfamily(c):
    s = c[2:] if c[:2] in ("h_", "a_", "d_") else c
    if c in ("mk_lr_h", "mk_lr_a") or c.startswith(("move_", "pin_diff", "disp", "overround", "n_books", "sm_")):
        return "B: market signals"
    if s.startswith(("xi_", "absent_", "n_former", "club_", "tenure", "coach_new")):
        return "B: line-ups, absences & coach"
    if s.startswith(("elo", "pois", "xpois", "xgf", "xga", "ppg", "pos", "promoted", "s_", "gf_l", "ga_l")) or c.startswith(("elo_", "pois", "xpois", "s_")):
        return "B: strength ratings"
    if s.startswith(("rest", "cong", "travel", "prev_cup")):
        return "B: schedule & travel"
    if s.startswith(("ref_", "lg_", "wind", "temp", "humid", "rain", "derby", "h2h")) or c.startswith(("ref_", "lg_", "wind", "temp", "humidity", "rain", "derby", "h2h")):
        return "B: referee, league & weather"
    if s.startswith(("safe_", "europe", "releg", "nothing", "progress", "remaining")):
        return "B: table & motivation"
    return "B: recent results & stats"


def form_subfamily(c):
    if any(k in c for k in ("_n_down", "n_down")):
        return "F: number of players in a downturn"
    if any(k in c for k in ("next_", "sustained_", "dropped_", "key4_")):
        return "F: model downturn risk"
    if any(k in c for k in ("mins_", "rest_days", "days_since_injury", "start_drop", "role_familiarity", "rumours")):
        return "F: workload, selection & availability"
    return "F: observed confidence signals"


def family_map(G):
    fams = {}
    for c in G["base"]:
        fams.setdefault(base_subfamily(c), []).append(c)
    for k, v in tm.families(G["tac"]).items():
        fams.setdefault("T: " + k.lower().replace("base football data", "other"), []).extend(v)
    for c in G["form"]:
        fams.setdefault(form_subfamily(c), []).append(c)
    return fams


# =============================================================================
# How many players in a downturn
# =============================================================================
def downturn_breakdown(df, L, have, hold, week):
    y = df.y.to_numpy()
    ph, pd_, pa = (df[c].to_numpy(float) for c in ("p_h", "p_d", "p_a"))
    pts_h = np.select([y == 0, y == 1], [3.0, 1.0], 0.0)
    pts_a = np.select([y == 2, y == 1], [3.0, 1.0], 0.0)
    res_h, res_a = pts_h - (3 * ph + pd_), pts_a - (3 * pa + pd_)
    gl_h = df.hg.to_numpy(float) - df.xpois_lh.to_numpy(float) if "xpois_lh" in df else np.full(len(df), np.nan)
    gl_a = df.ag.to_numpy(float) - df.xpois_la.to_numpy(float) if "xpois_la" in df else np.full(len(df), np.nan)
    out = {}
    for dfn, lab in (("down_obs", "Observed: recent adjusted rating well below his norm"), ("down_next", "Model: high next-game downturn risk"),
                     ("down_sus", "Model: high sustained-downturn risk")):
        hc, ac = df.get(f"fm_h_n_{dfn}"), df.get(f"fm_a_n_{dfn}")
        if hc is None or hc.notna().mean() < 0.1:
            continue
        hc, ac = hc.to_numpy(float), ac.to_numpy(float)
        ok = have & np.isfinite(hc) & np.isfinite(ac)
        rec = {"label": lab, "periods": {}}
        for pn, pm in (("discovery", ok & ~hold), ("holdout", ok & hold), ("all", ok)):
            if pm.sum() < 100:
                continue
            R = {}
            diff = np.clip(hc - ac, -3, 3)
            R["diff"] = [{"k": int(k), **(boot_mean(res_h[pm & (diff == k)], week[pm & (diff == k)]) or {"n": int((pm & (diff == k)).sum())})} for k in range(-3, 4)]
            hcc, acc = np.clip(hc, 0, 4), np.clip(ac, 0, 4)
            R["grid"] = [[{"n": int((pm & (hcc == i) & (acc == j)).sum()), "v": float(res_h[pm & (hcc == i) & (acc == j)].mean()) if (pm & (hcc == i) & (acc == j)).sum() >= 20 else None}
                          for j in range(5)] for i in range(5)]
            # team view: own count vs own points and goals vs expectation (home and away rows pooled)
            cnt = np.concatenate([np.clip(hc, 0, 5), np.clip(ac, 0, 5)])
            rp, rg = np.concatenate([res_h, res_a]), np.concatenate([gl_h, gl_a])
            wk = np.concatenate([week, week])
            mm = np.concatenate([pm, pm])
            R["team"] = [{"k": int(k), "pts": boot_mean(rp[mm & (cnt == k)], wk[mm & (cnt == k)]), "goals": boot_mean(rg[mm & (cnt == k)], wk[mm & (cnt == k)]),
                          "share": float((mm & (cnt == k)).sum() / mm.sum())} for k in range(6)]
            # where form adds most: base+form vs base, by total players in a downturn
            tot = np.clip(hc + ac, 0, 5)
            R["gain_by_total"] = [{"k": int(k), "BF_vs_B": boot_mean((L["B"] - L["BF"])[pm & (tot == k)], week[pm & (tot == k)]),
                                   "BTF_vs_M": boot_mean((L["M"] - L["BTF"])[pm & (tot == k)], week[pm & (tot == k)])} for k in range(6)]
            rec["periods"][pn] = R
        out[dfn] = rec
    return out


# =============================================================================
# Player stats: next-match counts for starters under 8 feature mixes
# =============================================================================
STAT_T = [("shots", "Shots", "count"), ("sot", "Shots on target", "count"), ("goals", "Goals", "count"), ("assists", "Assists", "count"),
          ("key_passes", "Key passes", "count"), ("fouls", "Fouls committed", "count"), ("cards", "Cards", "count"),
          ("def_actions", "Tackles + interceptions", "count"), ("touches", "Touches", "count"), ("passes", "Passes", "count"),
          ("drib_succ", "Successful dribbles", "count"), ("rating", "Match rating", "level")]
SG = ["P", "PB", "PT", "PF", "PBT", "PBF", "PTF", "PBTF"]
BASE_PAIRS = ["elo", "xgf", "xga", "gf", "ga", "sotf", "sota", "shf", "sha", "daf", "daa", "poss", "corf", "cora", "pts", "expts", "luck", "dluck",
              "xgf_l", "xga_l", "rest", "cong14", "travel", "ppg", "pos", "xi_rating", "xi_rating_delta", "xi_regular", "absent_share", "absent_top11", "tenure"]


def orient(dfm, fi, home, hcols, acols):
    H = dfm[hcols].to_numpy(np.float32)[fi]
    A = dfm[acols].to_numpy(np.float32)[fi]
    hm = (home == 1)[:, None]
    return np.where(hm, H, A), np.where(hm, A, H)


def make_reg(kind, args, seed=0):
    if fd.HAVE_XGB:
        return fd.xgb.XGBRegressor(objective="count:poisson" if kind == "count" else "reg:squarederror", n_estimators=args.stats_trees, learning_rate=0.06,
                                   max_depth=5, min_child_weight=30, subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", max_bin=128,
                                   n_jobs=args.threads, random_state=seed)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="poisson" if kind == "count" else "squared_error", max_iter=args.stats_trees, learning_rate=0.06,
                                         max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=2.0, max_bins=63, early_stopping=False, random_state=seed)


def pois_dev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def player_stats(F, dfm, tac_cols, hold_t, args, eval_start_t):
    """Starters' next-match stats. P = his own history; B/T/F add base, tactical and form information (all pre-match)."""
    d = F["d"]
    if bt.PROGRESS:
        bt.PROGRESS.start("player stats", 1)
    pcode = pd.factorize(d.player_id)[0]
    pos = d.groupby("player_id").cumcount().to_numpy()
    played = (d.minutes.fillna(0) > 0).to_numpy()
    if "yel_ev" in d:
        d["cards"] = (d.yel_ev.fillna(0) + 2 * d.red.fillna(0)).where(d.minutes.notna())
    stats = [s for s in STAT_T if s[0] in d and d.loc[played, s[0]].notna().mean() > 0.3 and d.loc[played, s[0]].std() > 0 and d.loc[played, s[0]].min() >= (0 if s[2] == "count" else -1e9)]
    mins = d.minutes.to_numpy(float)
    hist = []
    for k, _, _ in stats:
        v = np.where(played, d[k].to_numpy(float), np.nan)
        hist += [v, np.where(played & (mins >= 10), v / np.maximum(mins, 1) * 90, np.nan)]
    hist += [np.where(played, mins, np.nan), np.where(played, d.started.to_numpy(float), np.nan)]
    Hm = np.column_stack(hist)
    aS, aL = 1 - 0.5 ** (1 / 3), 1 - 0.5 ** (1 / 15)
    E = fd.ew_prior(pcode, pos, Hm, [aS, aL])
    pcols, P = [], []
    for i, (k, _, _) in enumerate(stats):
        P += [E[aS][:, 2 * i], E[aL][:, 2 * i], E[aL][:, 2 * i + 1]]
        pcols += [f"{k}_app_s", f"{k}_app_l", f"{k}_p90_l"]
    P += [E[aS][:, -2], E[aL][:, -2], E[aL][:, -1]]
    pcols += ["mins_s", "mins_l", "start_share"]
    for c in ("f_age", "f_apps_prior", "f_rest_days", "f_mins_14d", "f_starts_10", "f_role_familiarity", "f_is_GK", "f_is_DEF", "f_is_MID", "f_is_ATT", "home"):
        if c in d:
            P.append(d[c].to_numpy(float))
            pcols.append(c)
    P = np.column_stack(P).astype(np.float32)
    fid_set = pd.Index(dfm.fixture_id.astype("int64"))
    fi_all = fid_set.get_indexer(d.fixture_id.astype("int64"))
    rows = np.where((d.ghost.to_numpy() == 0) & (d.started.to_numpy() == 1) & played & (fi_all >= 0) & (d.t.to_numpy() >= eval_start_t - args.stats_window_days))[0]
    if len(rows) > args.stats_max_total:
        rows = np.sort(np.random.default_rng(1).choice(rows, args.stats_max_total, replace=False))
    fi, home = fi_all[rows], d.home.to_numpy(float)[rows]
    blocks = {"P": P[rows]}
    names = {"P": pcols}
    pairs = [x for x in BASE_PAIRS if f"h_{x}" in dfm and f"a_{x}" in dfm]
    own, opp = orient(dfm, fi, home, [f"h_{x}" for x in pairs], [f"a_{x}" for x in pairs])
    extra = []
    sgn = np.where(home == 1, 1.0, -1.0)
    for hc, ac, nm in (("p_h", "p_a", "p_win"), ("xpois_lh", "xpois_la", "lam_xg"), ("pois_lh", "pois_la", "lam_goals")):
        if hc in dfm:
            o, p = orient(dfm, fi, home, [hc], [ac])
            extra += [(f"own_{nm}", o[:, 0]), (f"opp_{nm}", p[:, 0])]
    for c in ("elo_diff",):
        if c in dfm:
            extra.append((c, dfm[c].to_numpy(np.float32)[fi] * sgn))
    for c in ("p_d", "ou_p_o", "lg_goals", "ref_cards", "wind", "derby"):
        if c in dfm:
            extra.append((c, dfm[c].to_numpy(np.float32)[fi]))
    blocks["B"] = np.column_stack([own, opp] + [e[1] for e in extra]).astype(np.float32)
    names["B"] = [f"own_{x}" for x in pairs] + [f"opp_{x}" for x in pairs] + [e[0] for e in extra]
    tp = sorted({c[2:] for c in tac_cols if c.startswith("h_") and "a_" + c[2:] in dfm})
    mp = sorted({c[3:] for c in tac_cols if c.startswith("mh_") and "ma_" + c[3:] in dfm})
    to, tpp = orient(dfm, fi, home, [f"h_{x}" for x in tp], [f"a_{x}" for x in tp])
    mo, mpp = orient(dfm, fi, home, [f"mh_{x}" for x in mp], [f"ma_{x}" for x in mp])
    tx = [(c, dfm[c].to_numpy(np.float32)[fi] * (sgn if c == "poss_clash" else 1)) for c in ("poss_clash", "style_distance") if c in dfm]
    blocks["T"] = np.column_stack([to, tpp, mo, mpp] + [x[1] for x in tx]).astype(np.float32)
    names["T"] = [f"own_{x}" for x in tp] + [f"opp_{x}" for x in tp] + [f"att_{x}" for x in mp] + [f"def_{x}" for x in mp] + [x[0] for x in tx]
    fc = ["f_" + c for c in F["fcols"] if "f_" + c in d and ("f_" + c) not in pcols]
    blocks["F"] = d[fc].to_numpy(np.float32)[rows]
    names["F"] = fc
    t = d.t.to_numpy(float)[rows]
    week = (d.day.dt.isocalendar().week.astype(int) + 100 * d.day.dt.year).to_numpy()[rows]
    role = d.role.map(fd.ROLE_GROUP).to_numpy()[rows]
    groups = {g: np.hstack([blocks[x] for x in g]) for g in SG} if args.stats_groups == "all" else {g: np.hstack([blocks[x] for x in g]) for g in args.stats_groups.split(",")}
    log(f"player stats: {len(rows):,} starter-matches, {len(stats)} stats, feature mixes {', '.join(groups)} "
        f"(P {len(names['P'])}, B {len(names['B'])}, T {len(names['T'])}, F {len(names['F'])} columns)")
    edges = pd.date_range(DAY0 + pd.Timedelta(days=float(eval_start_t)), DAY0 + pd.Timedelta(days=float(t.max()) + 2), freq=f"{args.stats_retrain_months}MS")
    if len(edges) < 2 or edges[0] > DAY0 + pd.Timedelta(days=float(eval_start_t)):
        edges = edges.insert(0, DAY0 + pd.Timedelta(days=float(eval_start_t)))
    edges = edges.append(pd.DatetimeIndex([DAY0 + pd.Timedelta(days=float(t.max()) + 2)]))
    bl = [((a - DAY0).days, (b - DAY0).days) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if bt.PROGRESS:
        bt.PROGRESS.start("player stats", max(1, len(bl) * len(stats) * len(groups)))
    rng = np.random.default_rng(11)
    Y = {k: d[k].to_numpy(float)[rows] for k, _, _ in stats}
    pred = {k: {g: np.full(len(rows), np.nan) for g in groups} for k, _, _ in stats}
    audit = []
    for b0, b1 in bl:
        te = (t >= b0) & (t < b1)
        base_tr = (t < b0) & (t >= b0 - args.stats_window_days)
        if not te.any():
            continue
        for k, _, kind in stats:
            tr = base_tr & np.isfinite(Y[k])
            if tr.sum() < args.min_train:
                if bt.PROGRESS:
                    bt.progress(len(groups), k)
                continue
            idx = np.where(tr)[0]
            if len(idx) > args.stats_max_rows:
                idx = np.sort(rng.choice(idx, args.stats_max_rows, replace=False))
            if kind == "count" and Y[k][idx].sum() <= 0:
                continue
            for g, X in groups.items():
                keep = np.isfinite(X[idx]).any(axis=0)
                m = make_reg(kind, args).fit(X[idx][:, keep], Y[k][idx])
                pred[k][g][te] = np.clip(m.predict(X[te][:, keep]), 1e-4 if kind == "count" else -10, None)
                if bt.PROGRESS:
                    bt.progress(1, f"{k} {g}")
        audit.append({"block": str((DAY0 + pd.Timedelta(days=b0)).date()), "train_last_day": str((DAY0 + pd.Timedelta(days=float(t[base_tr].max()))).date()) if base_tr.any() else None,
                      "ok": bool(not base_tr.any() or t[base_tr].max() < b0)})
        log(f"  player stats block {(DAY0 + pd.Timedelta(days=b0)):%Y-%m}: {te.sum():,} starter-matches predicted")
    # ---- scoring
    hold = t >= hold_t
    res = {"stats": [], "groups": list(groups), "n_cols": {k: len(v) for k, v in names.items()}, "audit": audit}
    for k, lab, kind in stats:
        yv = Y[k]
        have = np.isfinite(yv)
        for g in groups:
            have &= np.isfinite(pred[k][g])
        if have.sum() < 300:
            continue
        naive_raw = blocks["P"][:, pcols.index(f"{k}_app_l")].astype(float)
        fill = np.nanmean(yv[have])
        naive = np.where(np.isfinite(naive_raw), naive_raw, fill)
        rec = {"key": k, "label": lab, "kind": kind, "mean": float(np.mean(yv[have])), "periods": {}}
        lossf = (lambda yy, mu: pois_dev(yy, mu)) if kind == "count" else (lambda yy, mu: (yy - mu) ** 2)
        for pn, pm in (("discovery", have & ~hold), ("holdout", have & hold)):
            if pm.sum() < 200:
                continue
            ln = lossf(yv[pm], np.clip(naive[pm], 1e-4, None) if kind == "count" else naive[pm])
            lp = lossf(yv[pm], pred[k]["P"][pm])
            R = {"n": int(pm.sum()), "naive_loss": float(ln.mean()), "groups": {}}
            for g in groups:
                lg = lossf(yv[pm], pred[k][g][pm])
                G_ = {"loss": float(lg.mean()), "vs_naive": float(1 - lg.mean() / ln.mean()),
                      "vs_P": boot_mean((lp - lg) / max(lp.mean(), 1e-9), week[pm]) if g != "P" else None}
                if kind == "count":
                    a = (yv[pm] >= 1).astype(float)
                    if 0.03 < a.mean() < 0.97:
                        p1 = 1 - np.exp(-pred[k][g][pm])
                        G_["auc1"] = fd.auc(a, p1)
                        G_["ll1"] = float(ll2(p1, a).mean())
                        G_["rate1"] = float(a.mean())
                else:
                    G_["r2"] = float(1 - lg.mean() / np.var(yv[pm]))
                    G_["rmse"] = float(np.sqrt(lg.mean()))
                R["groups"][g] = G_
            # by position group and by form state (best mix vs P)
            gb = list(groups)[-1]
            R["roles"] = {}
            for rg in ("GK", "DEF", "MID", "ATT"):
                mm = pm & (role == rg)
                if mm.sum() >= 200 and (kind != "count" or yv[mm].mean() >= 0.1):
                    R["roles"][rg] = {"n": int(mm.sum()), "gain": float(1 - lossf(yv[mm], pred[k][gb][mm]).mean() / lossf(yv[mm], pred[k]["P"][mm]).mean())}
            if "PF" in groups and "f_rating_d" in fc:
                rd = blocks["F"][:, fc.index("f_rating_d")]
                thr = F.get("_thr_obs")
                if thr is not None and np.isfinite(thr):
                    for st_, cond in (("in downturn", rd <= thr), ("normal form", rd > thr)):
                        mm = pm & cond
                        if mm.sum() >= 200:
                            R.setdefault("form_state", {})[st_] = {"n": int(mm.sum()), "gain": float(1 - lossf(yv[mm], pred[k]["PF"][mm]).mean() / lossf(yv[mm], pred[k]["P"][mm]).mean())}
            # calibration of the richest mix
            mu = pred[k][gb][pm]
            cuts = np.unique(np.quantile(mu, np.linspace(0.1, 0.9, 9)))
            b = np.searchsorted(cuts, mu, side="right")
            R["calib"] = [[float(mu[b == j].mean()), float(yv[pm][b == j].mean()), int((b == j).sum())] for j in range(len(cuts) + 1) if (b == j).sum() >= 20]
            rec["periods"][pn] = R
        res["stats"].append(rec)
    return res


# =============================================================================
# Matchups: player vs team, vs opponent style, vs direct opponent - with persistence tests
# =============================================================================
def cluster_corr(x, y, cl, B=400, seed=2):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 200:
        return None
    df = pd.DataFrame({"c": cl[ok], "x": x[ok], "y": y[ok]})
    df["xy"], df["xx"], df["yy"] = df.x * df.y, df.x ** 2, df.y ** 2
    g = df.groupby("c")[["x", "y", "xy", "xx", "yy"]].sum()
    g["n"] = df.groupby("c").size()
    A = g.to_numpy()

    def corr(s):
        n = s[5]
        cov = s[2] - s[0] * s[1] / n
        return cov / math.sqrt(max((s[3] - s[0] ** 2 / n) * (s[4] - s[1] ** 2 / n), 1e-12))
    est = corr(A.sum(0))
    rng = np.random.default_rng(seed)
    bs = np.array([corr(A[rng.integers(0, len(A), len(A))].sum(0)) for _ in range(B)])
    return {"r": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B)), "n": int(ok.sum()), "players": int(len(A))}


def bucket_means(x, y, disc, hold, q=5):
    xd = x[disc & np.isfinite(x) & np.isfinite(y)]
    if len(xd) < 200:
        return None
    cuts = np.unique(np.quantile(xd, np.linspace(0, 1, q + 1)[1:-1]))
    out = []
    for k in range(len(cuts) + 1):
        row = {"q": k + 1}
        for pn, pm in (("discovery", disc), ("holdout", hold)):
            m = pm & np.isfinite(x) & np.isfinite(y) & (np.searchsorted(cuts, np.nan_to_num(x), side="right") == k)
            row[pn] = {"n": int(m.sum()), "v": float(y[m].mean()) if m.sum() >= 30 else None,
                       "se": float(y[m].std() / math.sqrt(m.sum())) if m.sum() >= 30 else None}
        out.append(row)
    return out


def prior_pair_sums(keys_sorted_frame, cols):
    """keys_sorted_frame: sorted by (key, t). Exclusive cumulative sums of cols within each key (earlier rows only)."""
    g = keys_sorted_frame.groupby("key", sort=False)
    out = {}
    for c in cols:
        cs = g[c].cumsum()
        out[c] = (cs - keys_sorted_frame[c]).to_numpy()
    return out


def lag_adjacent(d, r):
    """For each appearance: mean of the player's residuals in his previous two rated games (strictly earlier)."""
    ok = np.isfinite(r) & (d.ghost.to_numpy() == 0)
    sub = pd.DataFrame({"p": d.player_id.to_numpy()[ok], "r": r[ok], "i": np.where(ok)[0]})   # d is sorted by player, then date
    g = sub.groupby("p").r
    out = np.full(len(d), np.nan)
    out[sub.i.to_numpy()] = pd.concat([g.shift(1), g.shift(2)], axis=1).mean(axis=1).to_numpy()
    return out


def matchups(F, tac, hold_t, args):
    d = F["d"]
    real = d.ghost.to_numpy() == 0
    r = d["d_next"].to_numpy(float)
    t = d.t.to_numpy(float)
    names = dict(zip(F["raw"]["players"].player_id.astype("int64"), F["raw"]["players"].name)) if len(F["raw"]["players"]) else {}
    tn = dict(zip(F["raw"]["teams"].team_id.astype("int64"), F["raw"]["teams"].name)) if len(F["raw"]["teams"]) else {}
    first = F["first_block"]
    disc, hold = real & (t >= first) & (t < hold_t), real & (t >= hold_t)
    pid = d.player_id.to_numpy()
    out = {}
    # ---------- (a) player vs a particular opponent team
    fr = pd.DataFrame({"key": pd.factorize(pd.MultiIndex.from_arrays([d.player_id, d.opp_id]))[0], "t": t, "r": np.nan_to_num(r), "n": np.isfinite(r).astype(float),
                       "i": np.arange(len(d))})
    fr = fr.sort_values(["key", "t"], kind="stable")
    ps = prior_pair_sums(fr, ["r", "n"])
    S, N = np.empty(len(d)), np.empty(len(d))
    S[fr.i.to_numpy()], N[fr.i.to_numpy()] = ps["r"], ps["n"]
    k = 3.0
    h2h = np.where(N >= 1, S / (N + k), np.nan)
    # lag-matched placebo. Past games feed the player's own baseline, so ANY past residual correlates with
    # performance-vs-baseline by construction, and how much depends on how long ago it was. For every past meeting we
    # take his residuals in the two games he played just before it (same lag, always earlier, different opponent):
    # only a genuine team-specific effect survives in (record vs team) - (record in the games just before those meetings).
    adj = lag_adjacent(d, r)
    fr2 = pd.DataFrame({"key": fr.key.to_numpy(), "t": fr.t.to_numpy(), "i": fr.i.to_numpy()})
    both = np.isfinite(r) & np.isfinite(adj)
    fr2["a"] = np.where(both, adj, 0)[fr2.i]
    fr2["rb"] = np.where(both, r, 0)[fr2.i]
    fr2["nb"] = both.astype(float)[fr2.i]
    pb = prior_pair_sums(fr2, ["a", "rb", "nb"])
    SB, AB, NB = np.empty(len(d)), np.empty(len(d)), np.empty(len(d))
    SB[fr2.i.to_numpy()], AB[fr2.i.to_numpy()], NB[fr2.i.to_numpy()] = pb["rb"], pb["a"], pb["nb"]
    placebo = np.where(NB >= 1, AB / (NB + k), np.nan)
    excess = np.where(NB >= 1, (SB - AB) / (NB + k), np.nan)
    corr = lambda x: {pn: cluster_corr(x[pm], r[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))}  # noqa: E731
    rec = {"k": k, "corr": corr(excess), "raw": corr(h2h), "placebo": corr(placebo),
           "buckets": bucket_means(excess, r, disc, hold), "coverage": float(np.isfinite(h2h[hold & np.isfinite(r)]).mean()) if (hold & np.isfinite(r)).any() else None}
    # split-half persistence for pairs with 4+ meetings
    pr = pd.DataFrame({"p": d.player_id, "o": d.opp_id, "t": t, "r": r})[real & np.isfinite(r)].sort_values("t")
    pr["k"] = pr.groupby(["p", "o"]).cumcount()
    pr["n"] = pr.groupby(["p", "o"]).r.transform("size")
    pr = pr[pr.n >= 4]
    if len(pr):
        half = (pr.k < pr.n / 2)
        a = pr[half].groupby(["p", "o"]).r.mean()
        b = pr[~half].groupby(["p", "o"]).r.mean()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        rec["split_half"] = {"pairs": int(len(j)), "r": float(j.a.corr(j.b)) if len(j) > 30 else None}
    # bogey table: all meetings so far, shrunk
    allp = pd.DataFrame({"p": d.player_id, "o": d.opp_id, "r": r, "a": adj, "t": t})[real & np.isfinite(r) & np.isfinite(adj)]
    gp = allp.groupby(["p", "o"]).agg(n=("r", "size"), s=("r", "sum"), sa=("a", "sum"), last=("t", "max")).reset_index()
    gp = gp[gp.n >= 4]
    gp["shr"] = (gp.s - gp.sa) / (gp.n + k)      # vs the games just before each meeting, shrunk
    tab = pd.concat([gp.nsmallest(40, "shr"), gp.nlargest(25, "shr")])
    rec["table"] = [{"player": names.get(int(x.p), str(int(x.p))), "opp": tn.get(int(x.o), str(int(x.o))), "n": int(x.n), "mean": float(x.s / x.n), "shrunk": float(x.shr),
                     "last": str((DAY0 + pd.Timedelta(days=float(x.last))).date())} for x in tab.itertuples()]
    out["teams"] = rec
    # ---------- (b) player vs opponent STYLE (tactical profile of the opponent, pre-match)
    attrs = [a for a in ("press", "poss", "direct", "line", "aerial", "aggression", "width", "dribble") if f"h_st_{a}" in tac and f"a_st_{a}" in tac]
    styl = {}
    if attrs:
        tx = tac.set_index("fixture_id")
        tx = tx[~tx.index.duplicated()]
        fi = tx.index.get_indexer(d.fixture_id)
        home = d.home.to_numpy(float)
        # point-in-time standardisation: mean / sd over all team-matches on EARLIER days
        tday = d.t.to_numpy(float)
        tt = tac.merge(pd.DataFrame({"fixture_id": d.fixture_id, "t": tday}).drop_duplicates("fixture_id"), on="fixture_id", how="left")
        for a in attrs:
            raw_x = np.where(home == 1, tx[f"a_st_{a}"].to_numpy(float)[np.clip(fi, 0, None)], tx[f"h_st_{a}"].to_numpy(float)[np.clip(fi, 0, None)])
            raw_x[fi < 0] = np.nan
            vals = pd.concat([tt[["t", f"h_st_{a}"]].rename(columns={f"h_st_{a}": "x"}), tt[["t", f"a_st_{a}"]].rename(columns={f"a_st_{a}": "x"})]).dropna()
            dd = vals.groupby("t").x.agg(["sum", "count"]).join(vals.assign(x2=vals.x ** 2).groupby("t").x2.sum()).sort_index()
            cs = dd.cumsum().shift(1).fillna(0)
            mu = (cs["sum"] / cs["count"]).reindex(np.unique(tday)).ffill() if len(cs) else None
            if mu is None:
                continue
            m2 = (cs["x2"] / cs["count"])
            idxr = pd.Index(cs.index)
            pos_ = idxr.searchsorted(tday, side="right") - 1    # last day <= t: its 'cs' already excludes that day
            pos_ = np.clip(pos_, 0, len(idxr) - 1)
            mean_t = (cs["sum"] / cs["count"]).to_numpy()[pos_]
            sd_t = np.sqrt(np.clip(m2.to_numpy()[pos_] - mean_t ** 2, 1e-12, None))
            same_day = idxr.to_numpy()[pos_] == tday
            mean_t = np.where(same_day, mean_t, np.nan)   # every row's own day exists in the index, so this keeps the strict-prior value
            x = (raw_x - mean_t) / sd_t
            ok = real & np.isfinite(x) & np.isfinite(r)
            fr = pd.DataFrame({"key": pd.factorize(d.player_id)[0], "t": t, "x": np.where(ok, x, 0), "y": np.where(ok, r, 0), "n": ok.astype(float), "i": np.arange(len(d))})
            fr["xy"], fr["xx"] = fr.x * fr.y, fr.x ** 2
            fr = fr.sort_values(["key", "t"], kind="stable")
            ps = prior_pair_sums(fr, ["x", "y", "xy", "xx", "n"])
            Z = {c: np.empty(len(d)) for c in ps}
            for c in ps:
                Z[c][fr.i.to_numpy()] = ps[c]
            n_ = Z["n"]
            with np.errstate(all="ignore"):
                slope = (Z["xy"] - Z["x"] * Z["y"] / n_) / (Z["xx"] - Z["x"] ** 2 / n_ + 8.0)
            slope = np.where(n_ >= 8, slope, np.nan)
            pred = slope * x
            styl[a] = {"corr": {pn: cluster_corr(pred[pm], r[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))},
                       "buckets": bucket_means(pred, r, disc, hold)}
            # latest sensitivities (all games so far, same shrinkage), for the table
            last = pd.DataFrame({"p": d.player_id, "x": x, "y": r})[ok]
            gg = last.assign(xy=last.x * last.y, xx=last.x ** 2).groupby("p").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum"))
            gg = gg[gg.n >= 20]
            gg["slope"] = (gg.sxy - gg.sx * gg.sy / gg.n) / (gg.sxx - gg.sx ** 2 / gg.n + 8.0)
            styl[a]["table"] = [{"player": names.get(int(p), str(int(p))), "n": int(row.n), "slope": float(row.slope)}
                                for p, row in pd.concat([gg.nsmallest(12, "slope"), gg.nlargest(8, "slope")]).iterrows()]
            d[f"_style_{a}"] = pred
        if styl:
            comb = np.nansum(np.column_stack([d[f"_style_{a}"].to_numpy(float) for a in styl]), axis=1)
            comb[~np.isfinite(np.column_stack([d[f"_style_{a}"].to_numpy(float) for a in styl])).any(axis=1)] = np.nan
            out["style_combined"] = {"corr": {pn: cluster_corr(comb[pm], r[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))},
                                     "buckets": bucket_means(comb, r, disc, hold)}
    out["style"] = styl
    # ---------- (c) direct opponent: wide players vs the full-back on their flank, strikers vs the centre-backs
    dpos = d.dpos.fillna("").astype(str).str.upper().to_numpy()
    sidec = np.where(np.char.find(dpos.astype(str), "LEFT") >= 0, "L", np.where(np.char.find(dpos.astype(str), "RIGHT") >= 0, "R", ""))
    st = real & (d.started.to_numpy() == 1)
    role = d.role.to_numpy()
    q = {}
    for c in ("f_rating_l", "f_duel_win_l", "f_def_actions_l"):
        if c in d:
            q[c] = d[c].to_numpy(float)
    if q and (sidec != "").any():
        D = pd.DataFrame({"fixture_id": d.fixture_id, "team_id": d.team_id, "player_id": d.player_id, "role": role, "side": sidec, **q})[st]
        fbs = D[D.role == "FB"]
        cbs = D[D.role == "CB"].groupby(["fixture_id", "team_id"])[list(q)].mean().reset_index()
        A = pd.DataFrame({"i": np.arange(len(d)), "fixture_id": d.fixture_id, "opp_id": d.opp_id, "role": role, "side": sidec})[st & np.isin(role, ["W", "ST"])]
        w = A[A.role == "W"].copy()
        w["opp_side"] = np.where(w.side == "L", "R", np.where(w.side == "R", "L", ""))
        w = w.merge(fbs.rename(columns={"team_id": "opp_id", "side": "opp_side", "player_id": "def_id"}).drop(columns="role"), on=["fixture_id", "opp_id", "opp_side"], how="inner")
        w = w.drop_duplicates("i")
        s_ = A[A.role == "ST"].merge(cbs.rename(columns={"team_id": "opp_id"}), on=["fixture_id", "opp_id"], how="inner")
        s_["def_id"] = -1
        M = pd.concat([w, s_], ignore_index=True)
        if len(M) > 200:
            qc = "f_rating_l" if "f_rating_l" in q else list(q)[0]
            dq = np.full(len(d), np.nan)
            dq[M.i.to_numpy()] = M[qc].to_numpy(float)
            dw = np.full(len(d), np.nan)
            if "f_duel_win_l" in M:
                dw[M.i.to_numpy()] = M["f_duel_win_l"].to_numpy(float)
            rd = {"defender_form": {"corr": {pn: cluster_corr(dq[pm], r[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))},
                                    "buckets": bucket_means(dq, r, disc, hold), "n_pairs": int(len(M)), "wingers": int((M.role == "W").sum()), "strikers": int((M.role == "ST").sum())}}
            if np.isfinite(dw).any():
                for zc, lab in (("z_drib_succ", "dribble success"), ("z_duel_win", "duel win rate")):
                    if zc in d:
                        zz = d[zc].to_numpy(float)
                        rd[f"duel_vs_{zc[2:]}"] = {"label": lab, "corr": {pn: cluster_corr(dw[pm], zz[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))},
                                                   "buckets": bucket_means(dw, zz, disc, hold)}
            # pair history (winger vs the same full-back)
            wp = w[["i", "def_id"]].copy()
            wp["p"], wp["t"], wp["r"] = d.player_id.to_numpy()[wp.i], t[wp.i], r[wp.i]
            wp["key"] = pd.factorize(pd.MultiIndex.from_arrays([wp.p, wp.def_id]))[0]
            wp = wp.sort_values(["key", "t"], kind="stable")
            wp["rr"], wp["nn"] = np.nan_to_num(wp.r), np.isfinite(wp.r).astype(float)
            g = wp.groupby("key", sort=False)
            wp["S"], wp["N"] = g.rr.cumsum() - wp.rr, g.nn.cumsum() - wp.nn
            ph = np.full(len(d), np.nan)
            sel = wp.N >= 1
            ii = wp.i.to_numpy()
            wp["aa"] = np.where(np.isfinite(wp.r) & np.isfinite(adj[ii]), adj[ii], 0)
            wp["rr2"] = np.where(np.isfinite(wp.r) & np.isfinite(adj[ii]), wp.r, 0)
            wp["nn2"] = (np.isfinite(wp.r) & np.isfinite(adj[ii])).astype(float)
            g2 = wp.groupby("key", sort=False)
            S2, A2, N2 = (g2.rr2.cumsum() - wp.rr2).to_numpy(), (g2.aa.cumsum() - wp.aa).to_numpy(), (g2.nn2.cumsum() - wp.nn2).to_numpy()
            sel = pd.Series(N2 >= 1, index=wp.index)
            val = (S2 - A2) / (N2 + 3)
            ph[ii[sel.to_numpy()]] = val[sel.to_numpy()]
            rd["pair_history"] = {"corr": {pn: cluster_corr(ph[pm], r[pm], pid[pm]) for pn, pm in (("discovery", disc), ("holdout", hold))},
                                  "repeat_share": float(sel.mean())}
            pt = wp[np.isfinite(wp.r)].groupby(["p", "def_id"]).agg(n=("r", "size"), mean=("r", "mean")).reset_index()
            pt = pt[pt.n >= 3]
            pt["shr"] = pt["mean"] * pt.n / (pt.n + 3)
            rd["pairs"] = [{"attacker": names.get(int(x.p), str(int(x.p))), "defender": names.get(int(x.def_id), str(int(x.def_id))), "n": int(x.n), "mean": float(x.mean),
                            "shrunk": float(x.shr)} for x in pd.concat([pt.nsmallest(20, "shr"), pt.nlargest(15, "shr")]).itertuples()]
            out["direct"] = rd
    return out


# =============================================================================
# Main
# =============================================================================
MARKETISH = ("p_", "avg_", "best_", "pin_", "pinp_", "bestsoft_", "open_", "ou_", "n_books", "overround", "disp", "mk_", "op_", "move_", "pin_diff", "sm_")


def r3(a):
    return [None if (x is None or not np.isfinite(x)) else round(float(x), 3) for x in np.asarray(a, float)]


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data")
    g.add_argument("--dsn")
    g.add_argument("--cache", help="folder for query results (re-used on later runs; add --refresh after loading new data)")
    g.add_argument("--refresh", action="store_true")
    g.add_argument("--leagues", help="comma-separated league_ids to evaluate (history always uses everything)")
    g.add_argument("--db-jobs", type=int, default=6)
    g.add_argument("--work-mem", default="256MB")
    g.add_argument("--market-1x2", type=int, default=1)
    g.add_argument("--market-ou", type=int, default=80)
    g.add_argument("--sharp-bookmaker", default="Pinnacle")
    g = ap.add_argument_group("match models")
    g.add_argument("--holdout-from")
    g.add_argument("--burn-in-days", type=int, default=365)
    g.add_argument("--train-window-days", type=int, default=1460)
    g.add_argument("--retrain-months", type=int, default=4)
    g.add_argument("--retune-every", type=int, default=3)
    g.add_argument("--min-train", type=int, default=1500)
    g.add_argument("--half-life-days", type=float, default=180)
    g.add_argument("--no-standalone", action="store_true", help="skip the no-odds versions of the coalitions")
    g.add_argument("--no-ou", action="store_true", help="skip the Over/Under 2.5 coalitions")
    g.add_argument("--max-odds", type=float, default=10.0)
    g.add_argument("--min-bets", type=int, default=150)
    g = ap.add_argument_group("form models")
    g.add_argument("--form-retrain-months", type=int, default=6)
    g.add_argument("--form-window-days", type=int, default=1095)
    g.add_argument("--form-min-train", type=int, default=3000)
    g.add_argument("--form-max-rows", type=int, default=300_000)
    g.add_argument("--max-trees", type=int, default=600)
    g.add_argument("--min-minutes", type=int, default=30)
    g.add_argument("--min-history", type=int, default=8)
    g.add_argument("--k-games", type=int, default=6)
    g.add_argument("--horizon-days", type=int, default=150)
    g.add_argument("--downturn-quantile", type=float, default=0.25)
    g = ap.add_argument_group("player stats")
    g.add_argument("--stats-groups", default="all", help="'all' or e.g. P,PB,PF,PBTF")
    g.add_argument("--stats-retrain-months", type=int, default=12)
    g.add_argument("--stats-window-days", type=int, default=730)
    g.add_argument("--stats-max-rows", type=int, default=150_000)
    g.add_argument("--stats-max-total", type=int, default=1_500_000)
    g.add_argument("--stats-trees", type=int, default=250)
    g.add_argument("--no-player-stats", action="store_true")
    g = ap.add_argument_group("run")
    g.add_argument("--out", default="unified_dashboard.html")
    g.add_argument("--export-csv")
    g.add_argument("--title", default="Base, tactical and form models - alone and together")
    g.add_argument("--quick", action="store_true", help="fewer refits and smaller samples (for a first look)")
    g.add_argument("--leak-tests", type=int, default=3)
    g.add_argument("--skip-leak-test", action="store_true")
    g.add_argument("--jobs", type=int, default=0)
    g.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    g.add_argument("--progress-every", type=float, default=30)
    g.add_argument("--synthetic-label", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.quick:
        args.retrain_months, args.retune_every = max(args.retrain_months, 6), 2
        args.form_retrain_months = max(args.form_retrain_months, 12)
        args.stats_retrain_months, args.stats_max_rows, args.stats_trees = max(args.stats_retrain_months, 12), 60_000, 150
        args.form_max_rows, args.leak_tests = 150_000, 2
        if args.stats_groups == "all":
            args.stats_groups = "P,PB,PT,PF,PBTF"
    # names other modules expect
    args.no_gbm, args.no_inplay, args.alarm_gain = False, True, 0.006
    args.perm_repeats, args.explain_rows, args.watchlist = 2, 60_000, 0
    return args


def main(argv=None):
    args = build_args(argv)
    signal.signal(signal.SIGINT, fd._on_sigint)
    if args.work_mem:
        os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + f" -c work_mem={args.work_mem}").strip()
    skip = tuple(s for s, on in (("time-travel test", args.skip_leak_test), ("player stats", args.no_player_stats)) if on)
    bt.PROGRESS = UProgress(args.progress_every, skip=skip)
    log(f"learner for form & player stats: {'XGBoost' if fd.HAVE_XGB else 'scikit-learn HistGradientBoosting (XGBoost unavailable: ' + fd.XGB_ERR + ')'}")
    db = bt.DB(args.dsn)
    args.db_name = db.name
    # ---------------------------------------------------------------- B and T
    d = bt.extract(db, args)
    td = tm.tactical_extract(db, args)
    mk = bt.market_table(d, args)
    if len(d["predictions"]):
        mk = mk.join(d["predictions"].set_index("fixture_id"), how="left")
    log("base features ...")
    beng = bt.Engine(d, mk, args)
    base = beng.run()
    log("tactical features ...")
    teng = tm.TacticalEngine(d, td, mk, args)
    tac = teng.run()
    # ---------------------------------------------------------------- F
    log("form features and walk-forward downturn models ...")
    fargs = form_args(args)
    F = run_form(fargs, db.dsn)
    # ---------------------------------------------------------------- leak tests (all three engines)
    leak = {"skipped": args.skip_leak_test, "tests": [], "passed": True}
    if not args.skip_leak_test:
        log("time-travel tests: base, tactical and form engines re-run with the future deleted ...")
        for name, eng, full in (("Base", beng, base), ("Tactical", teng, tac)):
            r = bt.leak_test(eng, full, args)
            leak["tests"] += [dict(x, engine=name) for x in r["tests"]]
            leak["passed"] &= r["passed"]
        fdays = np.sort(F["d"].t[(F["d"].ghost == 0) & (F["d"].t >= F["first_block"])].unique())
        picks = sorted(np.random.default_rng(5).choice(fdays, size=min(args.leak_tests, len(fdays)), replace=False)) if len(fdays) else []
        fl = fd.leak_test(F["raw"], fargs, F["d"], F["feat"], [DAY0 + pd.Timedelta(days=float(x)) for x in picks], F["avail"])
        leak["tests"] += [{"engine": "Form", "day": x["day"], "matches": x["rows"], "features": x["features"], "mismatches": x["mismatches"]} for x in fl["tests"]]
        leak["passed"] &= fl["passed"]
    mt = tm.minute_travel_test(teng) if not args.skip_leak_test else {"skipped": True}
    # ---------------------------------------------------------------- one table per match
    team_rows, thr, atrisk, S = form_team(F, args)
    F["_thr_obs"] = thr.get("obs")
    fm = to_match(team_rows, base[["fixture_id", "home_id", "away_id"]])
    df = base.merge(tac.drop(columns=["day"]), on="fixture_id", how="left").merge(fm, on="fixture_id", how="left")
    df = df.merge(d["fixtures"][["fixture_id", "hg", "ag"]], on="fixture_id")
    df = bt.add_derived(df, mk).sort_values(["day", "fixture_id"]).reset_index(drop=True)
    tac_all = [c for c in tac.columns if c not in ("fixture_id", "day")]
    form_all = [c for c in fm.columns if c != "fixture_id"]
    bfs = bt.feature_sets(df.drop(columns=[c for c in tac_all + form_all if c in df]))
    anchors = ["mk_lr_h", "mk_lr_a"]
    ok_col = lambda c: c in df and df[c].notna().mean() > 0.05 and df[c].nunique() > 1  # noqa: E731
    base_cols = [c for c in dict.fromkeys(bfs["gbm"]) if c not in anchors and ok_col(c)]
    tac_cols = [c for c in tac_all if ok_col(c)]
    form_cols = [c for c in form_all if ok_col(c)]
    fam_cols = {"B": base_cols, "T": tac_cols, "F": form_cols}
    G = {"mk": {}, "raw": {}, "ou": {}, "base": base_cols, "tac": tac_cols, "form": form_cols}
    for c in COALS:
        cols = list(dict.fromkeys(x for f in c for x in fam_cols[f]))
        G["mk"][c] = anchors + cols
        G["raw"][c] = [x for x in cols if not x.startswith(MARKETISH)]
        G["ou"][c] = (["mk_lo"] if "mk_lo" in df and df.mk_lo.notna().any() else []) + cols
    log(f"features: base {len(base_cols)}, tactical {len(tac_cols)}, form {len(form_cols)} (one row per match: {len(df):,})")
    # ---------------------------------------------------------------- coalitions
    df["day_t"] = (df.day - DAY0).dt.days.astype(float)
    P, Sx, Q, NULL, shrink, audit, plan = coalition_walk_forward(df, G, args)
    E = df[["p_h", "p_d", "p_a"]].notna().all(axis=1).to_numpy().copy()
    for c in COALS:
        E &= np.isfinite(P[c]).all(axis=1)
    if args.leagues:
        E &= df.league_id.isin([int(x) for x in args.leagues.split(",")]).to_numpy()
    if E.sum() == 0:
        sys.exit("No matches could be evaluated - not enough history.")
    hf = pd.Timestamp(args.holdout_from) if args.holdout_from else df.day[E].quantile(0.6).normalize()
    hold = (df.day >= hf).to_numpy()
    hold_t = float((hf - DAY0).days)
    df["eval_row"], df["holdout_from"] = E, hf
    week = (df.day.dt.isocalendar().week.astype(int) + 100 * df.day.dt.year).to_numpy()
    y = df.y.to_numpy()
    bt.PROGRESS.start("scoring", 8)
    mkp = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    res = {"1x2": eval_market(df, P, mkp, y, lambda p, m: ll3(p, y[m]), E, hold, week, "1x2")}
    bt.progress(1, "1X2 scored")
    if not args.no_ou and "ou_p_o" in df and df.ou_p_o.notna().mean() > 0.3:
        yo = df.y_over.to_numpy(float)
        res["ou"] = eval_market(df, Q, df.ou_p_o.to_numpy(float), yo, lambda p, m: ll2(p, yo[m]), E & df.ou_p_o.notna().to_numpy(), hold, week, "ou")
    bt.progress(1, "O/U scored")
    if not args.no_standalone:
        res["raw"] = eval_market(df, Sx, NULL, y, lambda p, m: ll3(p, y[m]), E, hold, week, "1x2")
    bt.progress(1, "no-odds scored")
    have = res["1x2"]["_have"]
    calib = {c: calib3(P[c][have & hold], y[have & hold]) for c in COALS}
    calib["M"] = calib3(mkp[have & hold], y[have & hold])
    bets = betting(df, P, y, have, hold, week, args)
    bt.progress(1, "betting")
    log("feature-family importance (discovery-trained model, scored on the hold-out) ...")
    fams = family_map(G)
    imp = tm.permutation_families(df, {"combined": G["mk"]["BTF"]}, args, fams)
    bt.progress(1, "importance")
    down = downturn_breakdown(df, res["1x2"]["_L"], have, hold, week)
    bt.progress(1, "downturn breakdown")
    eval_start_t = float(df.day_t[E].min())
    pstats = None
    if not args.no_player_stats:
        pstats = player_stats(F, df, tac_cols, hold_t, args, eval_start_t)
    log("matchups: player vs team, vs opponent style, vs direct opponent ...")
    mu = matchups(F, tac, hold_t, args)
    alarms = [c for c in COALS if ((res["1x2"]["periods"].get("holdout", {}).get("models", {}).get(c, {}).get("gain") or {}).get("v") or 0) > args.alarm_gain]
    for c in alarms:
        log(f"!!! WARNING: {cname(c)} beats the closing market by an implausible margin - check for leakage before trusting it")
    payload = build_payload(df, P, Sx, Q, res, calib, bets, imp, fams, down, pstats, mu, F, thr, atrisk, leak, mt, audit, shrink, plan, args, E, hold, hf, alarms, G)
    if args.export_csv:
        out = df.drop(columns=["month"]).copy()
        for c in COALS:
            for i, s in enumerate("hda"):
                out[f"pred_{c}_{s}"] = P[c][:, i]
            out[f"pred_{c}_over"] = Q[c]
        out.to_csv(args.export_csv, index=False)
        log(f"CSV written: {args.export_csv}")
    page = TEMPLATE.replace("__TITLE__", htmlmod.escape(args.title)).replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    Path(args.out).write_text(page, encoding="utf-8")
    bt.PROGRESS.close()
    log(f"dashboard written: {args.out} ({len(page) / 1e6:.1f} MB)")
    H = res["1x2"]["periods"].get("holdout", {})
    for c in COALS:
        g = (H.get("models", {}).get(c, {}) or {}).get("gain")
        if g:
            log(f"  {cname(c):28s} hold-out gain vs market {g['v'] * 1000:+.2f}e-3 [{g['lo'] * 1000:+.2f}, {g['hi'] * 1000:+.2f}]  Holm p={res['1x2']['holm'].get(c, float('nan')):.3f}")
    for f in FAMS:
        s = H.get("shapley", {}).get(f)
        if s:
            log(f"  Shapley share {FAM_NAME[f]:9s} {s['v'] * 1000:+.2f}e-3 [{s['lo'] * 1000:+.2f}, {s['hi'] * 1000:+.2f}]")
    if not leak["skipped"] and not leak["passed"]:
        log("!!! TIME-TRAVEL TEST FAILED - see the dashboard")


def build_payload(df, P, Sx, Q, res, calib, bets, imp, fams, down, pstats, mu, F, thr, atrisk, leak, mt, audit, shrink, plan, args, E, hold, hf, alarms, G):
    for k in res:
        res[k].pop("_L", None)
        res[k].pop("_have", None)
    feat_fam = {c: f for f, cs in fams.items() for c in cs}
    impo = {}
    if imp:
        impo = {"base_ll": imp.get("base_ll"), "market_ll": imp.get("market_ll"),
                "families": [{"name": k, "origin": k[0], "inc": v["increase"], "n": v["n_features"]} for k, v in sorted(imp["families"].items(), key=lambda kv: -kv[1]["increase"])],
                "features": [{"name": k, "family": feat_fam.get(k, "?"), "origin": feat_fam.get(k, "?")[0], "inc": v} for k, v in imp["features"].items()]}
    # match explorer: every evaluated match (probabilities from each model), at-risk starters for hold-out matches
    ix = np.where(E)[0]
    D = df.iloc[ix]
    rows = {"id": D.fixture_id.astype(int).tolist(), "date": D.day.dt.strftime("%Y-%m-%d").tolist(), "league": D.league.fillna("").tolist(),
            "home": D.home.fillna("").tolist(), "away": D.away.fillna("").tolist(), "hg": D.hg.astype(int).tolist(), "ag": D.ag.astype(int).tolist(),
            "hold": hold[ix].astype(int).tolist(), "M": [r3(D[c]) for c in ("p_h", "p_d", "p_a")],
            "P": {c: [r3(P[c][ix, i]) for i in range(3)] for c in COALS},
            "ou_M": r3(D.ou_p_o) if "ou_p_o" in D else None, "ou_BTF": r3(Q["BTF"][ix])}
    for c in ("fm_h_n_down_obs", "fm_a_n_down_obs", "fm_h_n_down_next", "fm_a_n_down_next", "fm_h_next_mean", "fm_a_next_mean"):
        if c in D:
            rows[c[3:]] = r3(D[c])
    risk = {}
    for i in ix[hold[ix]]:
        f, h, a = int(df.fixture_id.iat[i]), int(df.home_id.iat[i]), int(df.away_id.iat[i])
        hh, aa = atrisk.get((f, h)), atrisk.get((f, a))
        if hh or aa:
            risk[f] = [hh or [], aa or []]
    rows["risk"] = risk
    real = F["d"].ghost == 0
    cov = {"eval": int(E.sum()), "from": str(df.day[E].min().date()), "to": str(df.day[E].max().date()), "leagues": int(df.league_id[E].nunique()),
           "n_disc": int((E & ~hold).sum()), "n_hold": int((E & hold).sum()), "holdout_from": str(hf.date()),
           "n_base": len(G["base"]), "n_tac": len(G["tac"]), "n_form": len(G["form"]),
           "form_rows": int(real.sum()), "form_players": int(F["d"].player_id[real].nunique()), "form_quality": F["quality"], "form_taus": F["taus"],
           "thresholds": thr, "perf": F["perf_name"], "compute": plan, "runtime_s": round(time.time() - bt.T0),
           "learner": "XGBoost" if fd.HAVE_XGB else "scikit-learn HistGradientBoosting",
           "with_ou": float(df.ou_p_o[E].notna().mean()) if "ou_p_o" in df else 0.0, "with_sharp": float(df.pin_h[E].notna().mean()) if "pin_h" in df else 0.0,
           "overround": float(df.overround[E].mean()) if "overround" in df else None}
    aud = {"form": {"n": len(F["audit"]), "ok": all(a["ok"] for a in F["audit"]), "rows": F["audit"]},
           "coalitions": {"n": len(audit), "ok": all(a["ok"] for a in audit), "rows": audit},
           "stats": {"n": len((pstats or {}).get("audit", [])), "ok": all(a["ok"] for a in (pstats or {}).get("audit", []))}}
    return fd.clean({"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "db": args.db_name, "synthetic": bool(args.synthetic_label),
                     "params": {k: v for k, v in vars(args).items() if k not in ("dsn",)}, "coals": COALS, "res": res, "calib": calib, "bets": bets,
                     "imp": impo, "down": down, "stats": pstats, "match": mu, "rows": rows, "cov": cov, "leak": leak, "minute": mt, "audit": aud,
                     "shrink": shrink, "alarms": alarms})


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root{color-scheme:light;--page:#f6f6f3;--surface:#fcfcfb;--raise:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e6e5df;--axis:#c3c2b7;--ring:rgba(11,11,11,.09);
  --s1:#2a78d6;--fB:#2a78d6;--fT:#eb6834;--fF:#199e70;--ctrl:#b4b2aa;--good:#006300;--bad:#d03b3b;--goodbg:rgba(12,163,12,.10);--badbg:rgba(208,59,59,.09);--wash:rgba(42,120,214,.07);--mid:#f0efec;--shadow:0 1px 2px rgba(0,0,0,.04),0 8px 24px rgba(0,0,0,.05)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#171716;--raise:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2a2a28;--axis:#3c3c39;--ring:rgba(255,255,255,.09);
  --s1:#3987e5;--fB:#3987e5;--fT:#d95926;--fF:#22a87a;--ctrl:#5f5e59;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.11);--mid:#383835;--shadow:none}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#171716;--raise:#1d1d1c;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2a2a28;--axis:#3c3c39;--ring:rgba(255,255,255,.09);
  --s1:#3987e5;--fB:#3987e5;--fT:#d95926;--fF:#22a87a;--ctrl:#5f5e59;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.11);--mid:#383835;--shadow:none}
*{box-sizing:border-box}body{margin:0}html{scroll-behavior:smooth}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;min-height:100vh}
.shell{display:grid;grid-template-columns:220px minmax(0,1fr);max-width:1480px;margin:0 auto}
nav.side{position:sticky;top:0;height:100vh;padding:28px 10px 20px 22px;border-right:1px solid var(--ring);overflow:auto}
nav.side .brand{font-weight:650;font-size:13px;margin:0 0 14px;color:var(--ink2);letter-spacing:.02em;text-transform:uppercase}
nav.side a{display:flex;gap:9px;align-items:center;color:var(--ink2);text-decoration:none;padding:6px 9px;border-radius:7px;font-size:13px;margin:1px 0}
nav.side a .n{font-variant-numeric:tabular-nums;color:var(--muted);font-size:11px;width:16px}
nav.side a:hover{background:var(--wash)}nav.side a.on{background:var(--wash);color:var(--ink);font-weight:600}
nav.side .foot{margin-top:18px;font-size:11.5px;color:var(--muted)}
main{padding:28px clamp(16px,3vw,40px) 80px;min-width:0}
@media (max-width:980px){.shell{grid-template-columns:1fr}nav.side{display:none}}
h1{font-size:27px;font-weight:680;margin:0 0 6px;letter-spacing:-.015em}h2{font-size:18px;font-weight:640;margin:0 0 4px;letter-spacing:-.005em}h3{font-size:14px;font-weight:620;margin:0 0 3px}
.sub{color:var(--ink2);margin:0 0 12px;max-width:1000px}.note{color:var(--muted);font-size:12.5px;margin:3px 0 10px;max-width:980px}
.verdict{font-size:15.5px;line-height:1.6;max-width:1000px;margin:6px 0 16px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;padding:4px 11px;border-radius:99px;border:1px solid var(--ring);background:var(--surface);color:var(--ink2)}
.badge.ok{background:var(--goodbg);color:var(--good);border-color:transparent}.badge.bad{background:var(--badbg);color:var(--bad);border-color:transparent}
.bar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:9px 14px;margin:0 0 22px;background:color-mix(in srgb,var(--surface) 88%,transparent);backdrop-filter:blur(8px);border:1px solid var(--ring);border-radius:12px;position:sticky;top:10px;z-index:8;box-shadow:var(--shadow)}
.seg{display:inline-flex;background:var(--page);border:1px solid var(--ring);border-radius:9px;padding:2px}
.seg button{border:0;background:transparent;padding:5px 11px;border-radius:7px;font:inherit;font-size:12.5px;color:var(--ink2);cursor:pointer}
.seg button.on{background:var(--raise);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12);font-weight:600}
.bar .lab{font-size:11.5px;color:var(--muted);margin-right:-8px}
select,input,button.plain{font:inherit;font-size:13px;color:var(--ink);background:var(--page);border:1px solid var(--ring);border-radius:8px;padding:5px 9px}
button.plain{cursor:pointer}button.plain:hover{background:var(--wash)}
section{padding-top:18px;margin-top:22px;scroll-margin-top:70px}
.sec-h{display:flex;align-items:baseline;gap:12px;margin-bottom:4px}.sec-h .k{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.grid{display:grid;gap:16px}.g2{grid-template-columns:repeat(auto-fit,minmax(480px,1fr))}.g3{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}.g4{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.lab-grid{display:grid;grid-template-columns:minmax(300px,380px) minmax(0,1fr);gap:16px}@media (max-width:1100px){.lab-grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:14px;padding:16px 18px;min-width:0;box-shadow:var(--shadow)}
.tile .lbl{color:var(--ink2);font-size:12.5px}.tile .val{font-size:25px;font-weight:650;margin:2px 0;font-variant-numeric:tabular-nums;letter-spacing:-.01em}.tile .ci{color:var(--muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:4px 0 8px;flex-wrap:wrap}.key{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}.ln{width:16px;height:2px;display:inline-block;border-radius:1px}
.dots{display:inline-flex;gap:3px;vertical-align:middle;margin-right:6px}.dots i{width:9px;height:9px;border-radius:50%;display:inline-block;border:1.5px solid var(--ctrl)}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{font-family:inherit;font-size:11px;fill:var(--muted)}svg .lab{fill:var(--ink2);font-size:11.5px}svg .val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none;z-index:1}
td:first-child,th:first-child,.l{text-align:left}td{font-variant-numeric:tabular-nums}tbody tr:hover td{background:var(--wash)}tr.sel td{background:var(--wash)}
.scroll{max-height:460px;overflow:auto}.wide{overflow-x:auto}
td.cell{text-align:center;cursor:pointer;color:var(--ink);transition:outline .1s}td.cell:hover{outline:2px solid var(--ink2);outline-offset:-2px}td.cell.on{outline:2px solid var(--ink);outline-offset:-2px}
.tip{position:fixed;pointer-events:none;background:var(--raise);color:var(--ink);border:1px solid var(--ring);box-shadow:0 6px 24px rgba(0,0,0,.16);border-radius:9px;padding:9px 11px;font-size:12px;opacity:0;transition:opacity .08s;z-index:40;max-width:340px;line-height:1.45}
.hit{fill:transparent}.why{color:var(--muted);font-size:11.5px;white-space:normal;max-width:520px}
.venn circle.fam{cursor:pointer;transition:fill-opacity .25s,stroke-opacity .25s}.venn text.fl{font-size:13px;font-weight:650;pointer-events:none}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px}.chip{border:1px solid var(--ring);background:var(--page);border-radius:99px;padding:4px 11px;font-size:12.5px;cursor:pointer;color:var(--ink2)}
.chip.on{background:var(--ink);color:var(--page);border-color:var(--ink)}
.callout{border-radius:10px;padding:10px 13px;font-size:13px;margin:8px 0;border:1px solid var(--ring);background:var(--page)}.callout.warn{background:var(--badbg);border-color:transparent}
.synthetic{background:var(--badbg);border-radius:12px;padding:11px 15px;margin:0 0 16px;font-size:13.5px}
.drawer{margin-top:12px;border-top:1px solid var(--grid);padding-top:12px}
.pbar{display:flex;height:16px;border-radius:5px;overflow:hidden;background:var(--grid)}.pbar span{display:block;height:100%}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:4px 18px;font-size:12.5px}.kv span{color:var(--muted)}
.pager{display:flex;gap:8px;align-items:center;margin-top:8px;font-size:12.5px;color:var(--ink2)}
.big{font-size:44px;font-weight:680;letter-spacing:-.02em;line-height:1;font-variant-numeric:tabular-nums}
ul.m{padding-left:18px}ul.m li{margin:4px 0;color:var(--ink2)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--ring);border-bottom-width:2px;border-radius:4px;padding:0 4px;color:var(--ink2)}
</style></head><body><div class="viz-root"><div class="shell">
<nav class="side" id="nav"><div class="brand">Model report</div></nav>
<main>
<h1>__TITLE__</h1><p class="sub" id="sub"></p><div id="synthetic"></div><p class="verdict" id="verdict"></p><div class="badges" id="badges"></div>
<div class="bar">
  <span class="lab">Market</span><div class="seg" id="seg-mkt"></div>
  <span class="lab">Period</span><div class="seg" id="seg-per"></div>
  <span class="lab">Price</span><div class="seg" id="seg-price"></div>
  <button class="plain" id="theme" type="button" style="margin-left:auto">◐ Theme</button>
</div>

<section id="lab"><div class="sec-h"><h2>Model lab</h2><span class="k">01</span></div>
<p class="note">Click the circles to switch each family on or off; every combination was trained and tested separately. Keys <kbd>1</kbd>–<kbd>7</kbd> jump through the seven combinations. Gains are in thousandths of log loss per match (×10⁻³): above zero means better forecasts than the reference.</p>
<div class="lab-grid">
  <div class="card"><svg class="venn" id="venn" viewBox="0 0 360 335" style="max-width:340px;margin:0 auto"></svg><div id="venn-cap" style="text-align:center;margin-top:4px"></div></div>
  <div><div class="grid g4" id="tiles"></div><div class="card" style="margin-top:16px"><h3>All seven combinations</h3><p class="note" id="coal-note"></p><div id="c-coal"></div></div></div>
</div></section>

<section id="shap"><div class="sec-h"><h2>Who brings what</h2><span class="k">02</span></div>
<p class="note">The Shapley value splits the full model's gain fairly between the three families: each family is credited with its average contribution over every order in which it could be added. The grid shows what adding one family to a given starting point is worth.</p>
<div class="grid g2">
  <div class="card"><h3>Fair share of the combined gain</h3><div id="c-shap"></div><p class="note" id="shap-note"></p></div>
  <div class="card"><h3>What adding a family is worth</h3><div class="wide"><table id="t-marg"></table></div><p class="note">Blue = adding it helps, red = it hurts. A dot marks a 95% interval that excludes zero.</p></div>
</div></section>

<section id="time"><div class="sec-h"><h2>Over time</h2><span class="k">03</span></div>
<p class="note">Cumulative log loss saved against the reference, match by match. A real edge climbs steadily; luck climbs in jumps. The highlighted line is the combination selected above; click a chip to highlight another.</p>
<div class="card"><div class="chips" id="cum-chips"></div><div id="c-cum"></div></div></section>

<section id="cal"><div class="sec-h"><h2>Calibration and betting</h2><span class="k">04</span></div>
<div class="grid g2">
  <div class="card"><h3>Calibration (hold-out)</h3><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span><span id="cal-name"></span></span><span class="key"><span class="sw" style="background:var(--ctrl);border-radius:99px"></span>Closing market</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Perfect</span></div><div id="c-cal"></div></div>
  <div class="card"><h3>Return per bet</h3><p class="note" id="bet-note"></p><div id="bet-warn"></div><div id="c-bet"></div></div>
</div>
<div class="card" style="margin-top:16px"><h3>Cumulative profit · <span id="pl-name"></span></h3><p class="note">1-unit stakes on the outcome with the biggest edge when the edge clears the threshold chosen on the discovery period.</p><div id="c-pl"></div></div></section>

<section id="drivers"><div class="sec-h"><h2>What drives the full model</h2><span class="k">05</span></div>
<p class="note" id="imp-note"></p>
<div class="grid g2">
  <div class="card"><h3>By feature family</h3><div class="legend"><span class="key"><span class="sw" style="background:var(--fB)"></span>Base</span><span class="key"><span class="sw" style="background:var(--fT)"></span>Tactical</span><span class="key"><span class="sw" style="background:var(--fF)"></span>Form</span></div><div id="c-fam"></div></div>
  <div class="card"><h3>Top individual features</h3><div class="scroll" style="max-height:520px"><table id="t-feat"></table></div></div>
</div></section>

<section id="down"><div class="sec-h"><h2>Players in a downturn</h2><span class="k">06</span></div>
<p class="note">Before kick-off, how many of each side's starters are in a downturn, and what happened against the market's expectation. Points are from the home side's view unless stated; positive = did better than the price implied.</p>
<div class="chips" id="down-chips"></div>
<div class="grid g2">
  <div class="card"><h3>Home starters × away starters in a downturn</h3><p class="note">Home points minus market-expected points. Click a cell for detail.</p><div class="wide"><table id="t-grid"></table></div><div id="grid-detail" class="note"></div></div>
  <div class="card"><h3>By the difference (home minus away)</h3><div id="c-diff"></div></div>
  <div class="card"><h3>A team's own downturns</h3><div class="chips" id="team-chips"></div><div id="c-team"></div></div>
  <div class="card"><h3>Where form adds most</h3><p class="note">Log-loss gain (×10⁻³) of Base + Form over Base alone, by how many starters (both sides) are in a downturn.</p><div id="c-addf"></div></div>
</div></section>

<section id="stats"><div class="sec-h"><h2>Predicting player stats</h2><span class="k">07</span></div>
<p class="note">Every starter's next-match numbers, predicted before kick-off. P = his own history; B, T and F add base, tactical and form information. Cells show deviance explained (counts) or variance explained (rating) beyond the comparison. Click a cell.</p>
<div class="chips" id="stat-chips"></div>
<div class="card"><div class="wide"><table id="t-stats"></table></div></div>
<div class="card" id="stat-detail" style="margin-top:16px"></div></section>

<section id="match"><div class="sec-h"><h2>Matchups</h2><span class="k">08</span></div>
<p class="note">Do some players struggle against particular teams, styles or opponents? Every pattern is tested for persistence: a player's history up to a match must predict how that match goes for him (performance vs his own norm). Correlations are clustered by player, with 95% intervals.</p>
<div class="chips" id="mu-tabs"></div><div id="mu-body"></div></section>

<section id="explore"><div class="sec-h"><h2>Match explorer</h2><span class="k">09</span></div>
<div class="card">
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">
    <input id="mx-q" placeholder="Search team" style="min-width:220px"><select id="mx-league"></select>
    <select id="mx-per"><option value="hold">Hold-out</option><option value="disc">Discovery</option><option value="all">All</option></select>
    <select id="mx-sort"><option value="date">Newest first</option><option value="dis">Biggest disagreement with market</option><option value="down">Most players in a downturn</option></select>
  </div>
  <div class="wide"><table id="t-mx"></table></div>
  <div class="pager"><button class="plain" id="pg-prev">‹</button><span id="pg-info"></span><button class="plain" id="pg-next">›</button></div>
  <div class="drawer" id="mx-detail"><p class="note">Click a match to compare every model's forecast.</p></div>
</div></section>

<section id="rigour"><div class="sec-h"><h2>Rigour, data and method</h2><span class="k">10</span></div>
<div class="grid g2">
  <div class="card"><h3>How look-ahead is ruled out</h3><ul class="m" id="safe"></ul><h3 style="margin-top:12px">Time-travel tests</h3><div class="scroll" style="max-height:260px"><table id="t-leak"></table></div></div>
  <div class="card"><h3>Coverage and settings</h3><table id="t-cov"></table><h3 style="margin-top:12px">Multiple testing (hold-out, vs market)</h3><table id="t-holm"></table></div>
</div></section>
</main></div><div class="tip" id="tip"></div></div>
<script>
const R = __DATA__;
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = (v, d=2) => v == null || !isFinite(v) ? '–' : (+v).toFixed(d);
const int = v => v == null ? '–' : Math.round(v).toLocaleString();
const pctu = (v, d=1) => v == null ? '–' : (v*100).toFixed(d) + '%';
const pct = (v, d=1) => v == null ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(d) + '%';
const sgn = (v, d=2) => v == null || !isFinite(v) ? '–' : (v >= 0 ? '+' : '') + (+v).toFixed(d);
const k3 = v => v == null ? '–' : sgn(v*1000, 2);
const cls = v => v == null ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : '';
const FN = {B:'Base', T:'Tactical', F:'Form', M:'Market only'};
const FC = f => css('--f' + f);
const cname = c => c === 'M' ? (S.mkt === 'raw' ? 'No information' : 'Closing market') : c.split('').map(x => FN[x]).join(' + ');
const tip = $('#tip');
function showTip(e, h){ tip.innerHTML = h; tip.style.opacity = 1; const w = tip.offsetWidth, x = e.clientX + 16 + w > innerWidth ? e.clientX - w - 16 : e.clientX + 16; tip.style.left = x + 'px'; tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8) + 'px'; }
function hideTip(){ tip.style.opacity = 0; }
const NS = 'http://www.w3.org/2000/svg';
function el(t, a, p){ const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
function svg(h, w, ht){ h.innerHTML = ''; return el('svg', {viewBox:`0 0 ${w} ${ht}`, role:'img'}, h); }
function text(p, x, y, s, a={}){ const t = el('text', Object.assign({x, y}, a), p); t.textContent = s; return t; }
function ticks(lo, hi, n=5){ const sp = hi - lo || 1, s0 = sp/n, mg = Math.pow(10, Math.floor(Math.log10(s0))); const st = [1,2,2.5,5,10].map(m=>m*mg).find(s => sp/s <= n) || mg*10; const o=[]; for (let v = Math.ceil(lo/st)*st; v <= hi+1e-9; v += st) o.push(+v.toFixed(10)); return o; }
function ext(v, zero=true){ v = v.filter(x => x != null && isFinite(x)); if (zero) v.push(0); if (!v.length) v = [0,1]; let lo = Math.min(...v), hi = Math.max(...v); if (lo === hi){ lo -= .5; hi += .5; } const p = (hi-lo)*.08; return [lo-p, hi+p]; }
function hit(n, h){ n.addEventListener('mousemove', e => showTip(e, typeof h === 'function' ? h() : h)); n.addEventListener('mouseleave', hideTip); }
function empty(h, m){ h.innerHTML = `<p class="note" style="padding:36px 0;text-align:center">${m}</p>`; }
function hbarPath(x0, x1, y, h){ const r = Math.min(4, Math.abs(x1-x0)/2, h/2); if (x1 >= x0) return `M${x0},${y}H${x1-r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1-r},${y+h}H${x0}Z`; return `M${x0},${y}H${x1+r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1+r},${y+h}H${x0}Z`; }
function vbarPath(x, w, y0, y1){ const r = Math.min(4, Math.abs(y1-y0)/2, w/2); if (y1 <= y0) return `M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z`; return `M${x},${y0}V${y1-r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1-r}V${y0}Z`; }
// horizontal bars: rows {label, v, lo, hi, v2, color, dots, sel, tip, onclick}
function hbars(host, rows, fmt, o={}){
  if (!rows.length) return empty(host, 'No data for this selection');
  const W = 640, rowH = o.rowH || 30, top = 4, left = o.left || 190, right = 70, H = top + rows.length*rowH + 24, s = svg(host, W, H);
  const [lo, hi] = ext(rows.flatMap(r => [r.v, r.lo, r.hi, r.v2])); const x = v => left + (v-lo)/(hi-lo)*(W-left-right);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-20, stroke:css('--grid')}, s); text(s, x(t), H-5, fmt(t), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:top, y2:H-20, stroke:css('--axis'), 'stroke-width':1.2}, s);
  rows.forEach((r, i) => { const y = top + i*rowH + (rowH-16)/2, h = 16;
    if (r.sel) el('rect', {x:2, y:y-(rowH-h)/2+1, width:W-4, height:rowH-2, rx:6, fill:css('--wash')}, s);
    let lx = left - 8; const lb = r.label.length > 34 ? r.label.slice(0, 33) + '…' : r.label;
    text(s, lx, y+h/2+4, lb, {'text-anchor':'end', class:'lab', 'font-weight': r.sel ? 650 : 400});
    if (r.dots){ const dw = 12; r.dots.forEach((d, j) => el('circle', {cx: 10 + j*dw, cy:y+h/2, r:4.2, fill: d.on ? FC(d.f) : 'none', stroke: d.on ? FC(d.f) : css('--ctrl'), 'stroke-width':1.4}, s)); }
    if (r.v != null) el('path', {d: hbarPath(x(0), x(r.v), y, h), fill: r.color || css('--s1')}, s);
    if (r.lo != null && r.hi != null){ el('line', {x1:x(r.lo), x2:x(r.hi), y1:y+h/2, y2:y+h/2, stroke:css('--ink2'), 'stroke-width':1.4}, s); for (const e of [r.lo, r.hi]) el('line', {x1:x(e), x2:x(e), y1:y+h/2-4, y2:y+h/2+4, stroke:css('--ink2'), 'stroke-width':1.4}, s); }
    if (r.v2 != null) el('circle', {cx:x(r.v2), cy:y+h/2, r:4.5, fill:css('--surface'), stroke:css('--ink2'), 'stroke-width':1.6}, s);
    if (r.v != null) text(s, Math.max(x(r.v), x(0), r.hi != null ? x(r.hi) : 0, r.v2 != null ? x(r.v2) + 5 : 0) + 7, y+h/2+4, fmt(r.v), {class:'val'});
    const hr = el('rect', {x:0, y:y-(rowH-h)/2, width:W, height:rowH, class:'hit', style: r.onclick ? 'cursor:pointer' : ''}, s);
    if (r.tip) hit(hr, r.tip); if (r.onclick) hr.addEventListener('click', r.onclick); });
}
function vbars(host, rows, fmt, o={}){
  if (!rows.length) return empty(host, 'No data for this selection');
  const W = 620, H = o.H || 250, top = 12, bottom = 44, left = 54, right = 10, s = svg(host, W, H);
  const [lo, hi] = ext(rows.flatMap(r => [r.v, r.lo, r.hi]).concat(o.ref != null ? [o.ref] : []));
  const y = v => top + (hi-v)/(hi-lo)*(H-top-bottom), bw = (W-left-right)/rows.length, w = Math.min(30, bw*.55);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, fmt(t), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis')}, s);
  if (o.ref != null) el('line', {x1:left, x2:W-right, y1:y(o.ref), y2:y(o.ref), stroke:css('--axis'), 'stroke-dasharray':'4 3', 'stroke-width':1.5}, s);
  rows.forEach((r, i) => { const cx = left + bw*i + bw/2;
    text(s, cx, H-bottom+16, r.label, {'text-anchor':'middle', class:'lab'}); if (r.n != null) text(s, cx, H-bottom+30, 'n ' + int(r.n), {'text-anchor':'middle'});
    if (r.v != null) el('path', {d: vbarPath(cx-w/2, w, y(0), y(r.v)), fill: r.color || css('--s1'), opacity: r.faint ? .45 : 1}, s);
    if (r.lo != null && r.hi != null){ el('line', {x1:cx, x2:cx, y1:y(r.lo), y2:y(r.hi), stroke:css('--ink2'), 'stroke-width':1.4}, s); for (const e of [r.lo, r.hi]) el('line', {x1:cx-4, x2:cx+4, y1:y(e), y2:y(e), stroke:css('--ink2'), 'stroke-width':1.4}, s); }
    if (r.v2 != null) el('circle', {cx, cy:y(r.v2), r:5, fill:css('--surface'), stroke:css('--ink2'), 'stroke-width':1.8}, s);
    if (r.tip) hit(el('rect', {x:cx-bw/2, y:top, width:bw, height:H-top-bottom, class:'hit'}, s), r.tip); });
  if (o.xtitle) text(s, left + (W-left-right)/2, H-2, o.xtitle, {'text-anchor':'middle'});
}
function lines(host, series, o={}){
  const pts = series.flatMap(s => s.pts); if (!pts.length) return empty(host, o.empty || 'No data');
  const isDate = typeof pts[0][0] === 'string', X = v => isDate ? Date.parse(v) : v;
  const W = 900, H = o.H || 300, top = 14, bottom = 30, left = 60, right = o.right ?? 120, s = svg(host, W, H);
  const xs = pts.map(p => X(p[0])), x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const [lo, hi] = o.domain || ext(pts.map(p => p[1]).concat(o.ref != null ? [o.ref] : []), o.zero !== false);
  const x = v => left + (X(v)-x0)/(x1-x0)*(W-left-right), y = v => top + (hi-v)/(hi-lo)*(H-top-bottom);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-8, y(t)+4, o.fmt(t), {'text-anchor':'end'}); }
  if (o.ref != null) el('line', {x1:left, x2:W-right, y1:y(o.ref), y2:y(o.ref), stroke:css('--axis'), 'stroke-width':1.4, 'stroke-dasharray': o.refDash || ''}, s);
  if (isDate){ const y0 = new Date(x0).getUTCFullYear(), y1 = new Date(x1).getUTCFullYear(); for (let yy = y0; yy <= y1+1; yy++){ const t = Date.UTC(yy,0,1); if (t >= x0 && t <= x1){ text(s, left+(t-x0)/(x1-x0)*(W-left-right), H-9, yy, {'text-anchor':'middle'}); el('line', {x1:left+(t-x0)/(x1-x0)*(W-left-right), x2:left+(t-x0)/(x1-x0)*(W-left-right), y1:H-bottom, y2:H-bottom+4, stroke:css('--axis')}, s); } } }
  else (o.xticks || ticks(x0, x1, 8)).forEach(t => text(s, x(t), H-10, o.xfmt ? o.xfmt(t) : t, {'text-anchor':'middle'}));
  if (o.vline){ const vx = x(o.vline); if (vx >= left && vx <= W-right){ el('rect', {x:vx, y:top, width:W-right-vx, height:H-top-bottom, fill:css('--wash')}, s); el('line', {x1:vx, x2:vx, y1:top, y2:H-bottom, stroke:css('--axis'), 'stroke-width':1.4}, s); text(s, vx+6, top+12, o.vlabel || '', {}); } }
  const ordered = series.slice().sort((a, b) => (a.hl ? 1 : 0) - (b.hl ? 1 : 0));
  for (const se of ordered){ if (!se.pts.length) continue;
    el('path', {d: se.pts.map((p, j) => (j?'L':'M') + x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1)).join(''), fill:'none', stroke:se.color, 'stroke-width': se.w || 2, 'stroke-linejoin':'round', 'stroke-linecap':'round', opacity: se.op ?? 1, 'stroke-dasharray': se.dash || ''}, s);
    if (se.dots) se.pts.forEach(p => el('circle', {cx:x(p[0]), cy:y(p[1]), r:4, fill:se.color, stroke:css('--surface'), 'stroke-width':2}, s));
    if (se.end){ const l = se.pts[se.pts.length-1]; el('circle', {cx:x(l[0]), cy:y(l[1]), r: se.hl ? 4.5 : 3, fill:se.color, stroke:css('--surface'), 'stroke-width':2}, s); text(s, x(l[0]) + 8, y(l[1]) + 4, se.end, {class: se.hl ? 'val' : '', 'font-weight': se.hl ? 650 : 400}); } }
  const cross = el('line', {y1:top, y2:H-bottom, stroke:css('--axis'), opacity:0}, s);
  const hr = el('rect', {x:left, y:top, width:W-left-right, height:H-top-bottom, class:'hit'}, s);
  hr.addEventListener('mousemove', e => { const b = s.getBoundingClientRect(), mx = (e.clientX - b.left)/b.width*W, xv = x0 + (mx-left)/(W-left-right)*(x1-x0);
    const rows = []; let cx = null, lab = '';
    for (const se of series){ if (!se.pts.length || se.notip) continue; let best = se.pts[0]; for (const p of se.pts) if (Math.abs(X(p[0])-xv) < Math.abs(X(best[0])-xv)) best = p; cx = x(best[0]); lab = o.xlabel ? o.xlabel(best[0]) : best[0]; rows.push([se, best]); }
    rows.sort((a, b) => b[1][1] - a[1][1]);
    if (cx != null){ cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('opacity', 1); }
    showTip(e, `<b>${lab}</b><br>` + rows.map(([se, p]) => `<span style="color:${se.color}">●</span> ${se.hl ? '<b>' : ''}${esc(se.name)}: ${o.fmt(p[1])}${se.hl ? '</b>' : ''}${p[2] != null && o.extra ? ' · ' + o.extra(p) : ''}`).join('<br>')); });
  hr.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); });
  if (o.onclick) { hr.style.cursor = 'pointer'; hr.addEventListener('click', o.onclick); }
}
function calChart(host, pts, ref){
  if (!pts || !pts.length) return empty(host, 'No data');
  const W = 620, H = 300, top = 10, bottom = 34, left = 48, right = 12, s = svg(host, W, H);
  const mx = Math.min(1, Math.max(...pts.concat(ref || []).map(p => Math.max(p[0], p[1]))) * 1.08);
  const x = v => left + v/mx*(W-left-right), y = v => top + (1-v/mx)*(H-top-bottom);
  for (const t of ticks(0, mx, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-bottom, stroke:css('--grid')}, s); el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, x(t), H-bottom+15, Math.round(t*100) + '%', {'text-anchor':'middle'}); text(s, left-7, y(t)+4, Math.round(t*100) + '%', {'text-anchor':'end'}); }
  text(s, left + (W-left-right)/2, H-2, 'Forecast probability', {'text-anchor':'middle'});
  el('line', {x1:x(0), y1:y(0), x2:x(mx), y2:y(mx), stroke:css('--axis'), 'stroke-width':1.5}, s);
  for (const [P_, col, main_] of [[ref, css('--ctrl'), false], [pts, css('--s1'), true]]){ if (!P_) continue;
    el('path', {d: P_.map((p, j) => (j?'L':'M') + x(p[0]) + ',' + y(p[1])).join(''), fill:'none', stroke:col, 'stroke-width': main_ ? 2 : 1.5, 'stroke-dasharray': main_ ? '' : '4 3'}, s);
    P_.forEach(p => { const r = main_ ? Math.max(4, Math.min(8, Math.sqrt(p[2])/6)) : 3.5; el('circle', {cx:x(p[0]), cy:y(p[1]), r, fill:col, stroke:css('--surface'), 'stroke-width':2}, s);
      if (main_) hit(el('circle', {cx:x(p[0]), cy:y(p[1]), r:r+8, class:'hit'}, s), `Forecast ${pctu(p[0])}<br>Happened ${pctu(p[1])}<br>${int(p[2])} outcomes`); }); }
}
function sortable(t, head, rows, o={}){
  let key = o.sortKey ?? null, dir = o.sortDir ?? -1;
  function draw(){ const r = rows.slice(); if (key != null) r.sort((a, b) => { const x = a.v[key], y = b.v[key]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    t.innerHTML = '<thead><tr>' + head.map((h, i) => `<th data-i="${i}" class="${h.c||''}" title="${esc(h.t||'')}">${h.h}${key === i ? (dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr></thead><tbody>' +
      r.map(x => `<tr data-k="${esc(x.k ?? '')}" class="${x.sel ? 'sel' : ''}" ${o.onRow ? 'style="cursor:pointer"' : ''}>` + x.c.map((c, i) => `<td class="${head[i].c||''}" ${x.st && x.st[i] ? `style="${x.st[i]}"` : ''}>${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
    t.querySelectorAll('th').forEach(th => th.onclick = () => { const i = +th.dataset.i; dir = key === i ? -dir : -1; key = i; draw(); });
    if (o.onRow) t.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => o.onRow(tr.dataset.k)); }
  draw();
}
function heat(v, scale){ if (v == null || !isFinite(v)) return ''; const a = Math.min(1, Math.abs(v)/scale); return `background:color-mix(in srgb, ${v >= 0 ? 'var(--s1)' : 'var(--bad)'} ${Math.round(a*62)}%, var(--surface))`; }
const dots = c => `<span class="dots">${['B','T','F'].map(f => `<i style="${c.includes(f) ? `background:${FC(f)};border-color:${FC(f)}` : ''}"></i>`).join('')}</span>`;
const sig = g => g && g.lo != null && (g.lo > 0 || g.hi < 0);
// ---------------------------------------------------------------- state
const S = {mkt:'1x2', per:'holdout', price: Object.keys(R.bets || {})[0] || 'sharp', coal:'BTF', hl:null, downDef: Object.keys(R.down || {})[0], teamMetric:'pts', statMode:'naive', statCell:null, mutab:'teams', styleAttr:null, mxPage:0, mxSel:null};
const MK = () => R.res[S.mkt] || R.res['1x2'];
const PER = () => MK().periods[S.per] || MK().periods.holdout || Object.values(MK().periods)[0];
function seg(id, opts, key, cb){ const h = $(id); h.innerHTML = opts.map(([v, l]) => `<button data-v="${v}" class="${S[key] === v ? 'on' : ''}">${l}</button>`).join(''); h.querySelectorAll('button').forEach(b => b.onclick = () => { S[key] = b.dataset.v; h.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); cb(); }); }
// ---------------------------------------------------------------- header
function header(){
  const C = R.cov;
  $('#sub').innerHTML = `${int(C.eval)} matches evaluated across ${C.leagues} competitions, ${C.from} → ${C.to} (discovery ${int(C.n_disc)} · hold-out from <b>${C.holdout_from}</b>: ${int(C.n_hold)}). Features: Base ${C.n_base}, Tactical ${C.n_tac}, Form ${C.n_form}, from ${int(C.form_rows)} player appearances of ${int(C.form_players)} players. Database <b>${esc(R.db)}</b> · ${R.generated}.`;
  if (R.synthetic) $('#synthetic').innerHTML = '<div class="synthetic"><b>Demo on a synthetic database.</b> The leagues, players and every number on this page are invented, generated to exercise the pipeline. Run the script on your database for real findings.</div>';
  const L = R.leak, b = [];
  if (L.skipped) b.push('<span class="badge">Time-travel tests skipped</span>');
  else { const by = {}; L.tests.forEach(t => { by[t.engine] = (by[t.engine] || 0) + t.mismatches; });
    b.push(`<span class="badge ${L.passed ? 'ok' : 'bad'}">${L.passed ? '✓' : '✕'} Time-travel tests ${L.passed ? 'passed' : 'FAILED'}: ${Object.entries(by).map(([k, v]) => `${k} ${v} changed`).join(' · ')}</span>`); }
  if (R.minute && !R.minute.skipped) b.push(`<span class="badge ${R.minute.passed ? 'ok' : 'bad'}">${R.minute.passed ? '✓' : '✕'} Minute-travel test (tactical)</span>`);
  const A = R.audit, aok = A.form.ok && A.coalitions.ok && A.stats.ok;
  b.push(`<span class="badge ${aok ? 'ok' : 'bad'}">${aok ? '✓' : '✕'} Training audits: ${A.form.n + A.coalitions.n + A.stats.n} model fits, every one trained strictly before what it predicts</span>`);
  b.push('<span class="badge">Holm-corrected p-values</span>', '<span class="badge">Thresholds frozen before use</span>');
  if (R.alarms.length) b.push(`<span class="badge bad">⚠ Implausibly strong: ${R.alarms.map(cname).join(', ')}. Treat as a leak until proven otherwise.</span>`);
  $('#badges').innerHTML = b.join('');
  // plain-language verdict (1X2, hold-out)
  const H = R.res['1x2'].periods.holdout || {}; const M = H.models || {};
  const best = R.coals.map(c => [c, M[c]?.gain]).filter(x => x[1]).sort((a, b) => b[1].v - a[1].v)[0];
  const sh = H.shapley || {};
  if (best){ const g = best[1], holm = R.res['1x2'].holm[best[0]];
    const shs = ['B','T','F'].map(f => sh[f] ? `${FN[f]} ${k3(sh[f].v)}` : null).filter(Boolean).join(', ');
    $('#verdict').innerHTML = `On the hold-out, the strongest combination was <b>${cname(best[0])}</b>, ${g.v > 0 ? 'improving on' : 'falling short of'} the closing market by <b class="${cls(g.v)}">${k3(g.v)}</b> thousandths of log loss per match (95% interval ${k3(g.lo)} to ${k3(g.hi)}; Holm-adjusted p = ${num(holm, 3)}${holm != null && holm < .05 ? ', <b>unlikely to be chance</b>' : ', could be chance'}). Fair shares of the full model's gain: ${shs}.`; }
}
// ---------------------------------------------------------------- 01 model lab
function venn(){
  const s = $('#venn'); s.innerHTML = '';
  const C = {B:[130, 125], T:[230, 125], F:[180, 212]}, r = 92, on = f => S.coal.includes(f);
  for (const f of ['B','T','F']){ const [cx, cy] = C[f];
    const c = el('circle', {cx, cy, r, class:'fam', fill:FC(f), 'fill-opacity': on(f) ? .2 : .03, stroke:FC(f), 'stroke-opacity': on(f) ? 1 : .35, 'stroke-width': on(f) ? 2.2 : 1.4, 'stroke-dasharray': on(f) ? '' : '5 4'}, s);
    c.addEventListener('click', () => { const set = new Set(S.coal === 'M' ? [] : S.coal.split('')); set.has(f) ? set.delete(f) : set.add(f); S.coal = ['B','T','F'].filter(x => set.has(x)).join('') || 'M'; S.hl = null; drawLab(); });
    hit(c, () => `<b>${FN[f]}</b> · click to ${on(f) ? 'remove' : 'add'}`); }
  const lp = {B:[46, 40], T:[314, 40], F:[180, 322]};
  for (const f of ['B','T','F']) text(s, lp[f][0], lp[f][1], FN[f], {'text-anchor':'middle', class:'fl', fill: on(f) ? FC(f) : css('--muted')});
  const P_ = PER(), m = S.coal === 'M' ? null : P_.models[S.coal], g = m?.gain;
  $('#venn-cap').innerHTML = `<div class="note" style="margin:0">${S.mkt === '1x2' ? 'Boosted corrections to the closing 1X2 market' : S.mkt === 'ou' ? 'Boosted corrections to the closing Over/Under 2.5 market' : 'Trained without any odds'}</div>
    <div style="font-size:16px;font-weight:650;margin:4px 0 2px">${S.coal === 'M' ? (S.mkt === 'raw' ? 'No information (base rates)' : 'Closing market only') : cname(S.coal)}</div>
    <div class="big ${cls(g?.v)}">${S.coal === 'M' ? '—' : k3(g?.v)}</div><div class="note" style="margin:4px 0 0">${S.coal === 'M' ? 'the reference every model is compared with' : `×10⁻³ log loss per match vs ${S.mkt === 'raw' ? 'no information' : 'the market'} · 95% interval ${k3(g?.lo)} to ${k3(g?.hi)}`}</div>`;
}
function tiles(){
  const P_ = PER(), c = S.coal, m = P_.models[c], mm = P_.models.M, holm = MK().holm?.[c];
  if (c === 'M' || !m){ $('#tiles').innerHTML = `<div class="card tile"><div class="lbl">Reference log loss</div><div class="val">${num(mm.loss, 4)}</div><div class="ci">${int(mm.n)} matches</div></div><div class="card tile"><div class="lbl">Accuracy</div><div class="val">${pctu(mm.acc)}</div><div class="ci">${S.mkt === 'ou' ? 'over/under called right' : 'favourite won'}</div></div>`; return; }
  const g = m.gain, vb = m.vs_base, bet = R.bets?.[S.price]?.[c]?.[S.per === 'all' ? 'holdout' : S.per];
  const t = [['Gain vs ' + (S.mkt === 'raw' ? 'no information' : 'market'), `<span class="${cls(g?.v)}">${k3(g?.v)}</span>`, g ? `${k3(g.lo)} to ${k3(g.hi)} · ${S.per === 'holdout' && holm != null ? 'Holm p ' + num(holm, 3) : 'p ' + num(g.p, 3)}${g.identical ? ' · identical to reference' : ''}` : ''],
    ['Gain vs Base alone', vb ? `<span class="${cls(vb.v)}">${k3(vb.v)}</span>` : '–', vb ? `${k3(vb.lo)} to ${k3(vb.hi)}` : 'needs Base in the mix'],
    ['Log loss', num(m.loss, 4), `reference ${num(mm.loss, 4)}`],
    ['Accuracy', pctu(m.acc), `reference ${pctu(mm.acc)}${m.same_as_market != null ? ` · ${pctu(m.same_as_market, 0)} of forecasts equal the market` : ''}`],
    S.mkt === '1x2' ? ['Return per bet', bet && bet.n ? `<span class="${cls(bet.roi)}">${pct(bet.roi)}</span>` : '–', bet && bet.n ? `${int(bet.n)} bets at ${S.price} prices${bet.lo != null ? ` · ${pct(bet.lo)} to ${pct(bet.hi)}` : ''}` : 'no bets cleared the threshold'] : ['Brier', num(m.brier, 4), `reference ${num(mm.brier, 4)}`]];
  $('#tiles').innerHTML = t.map(x => `<div class="card tile"><div class="lbl">${x[0]}</div><div class="val">${x[1]}</div><div class="ci">${x[2]}</div></div>`).join('');
}
function coalBars(){
  const P_ = PER();
  $('#coal-note').textContent = `${S.per === 'holdout' ? 'Hold-out' : S.per === 'discovery' ? 'Discovery' : 'All'} period. Bars: gain with 95% interval (resampling whole weeks). Blue = interval excludes zero. Click a bar to select it.`;
  hbars($('#c-coal'), R.coals.map(c => { const g = P_.models[c]?.gain; return {label: cname(c), v: g?.v*1000, lo: g?.lo*1000, hi: g?.hi*1000, sel: c === S.coal, dots: ['B','T','F'].map(f => ({f, on: c.includes(f)})),
    color: sig(g) ? (g.v > 0 ? css('--s1') : css('--bad')) : css('--ctrl'), onclick: () => { S.coal = c; S.hl = null; drawLab(); },
    tip: `<b>${cname(c)}</b><br>Gain ${k3(g?.v)} (${k3(g?.lo)} to ${k3(g?.hi)})<br>${int(g?.n)} matches${S.per === 'holdout' ? '<br>Holm p ' + num(MK().holm?.[c], 3) : ''}`}; }), v => sgn(v, 1), {left:205});
}
// ---------------------------------------------------------------- 02 shapley + marginal
function shapView(){
  const sh = PER().shapley || {};
  hbars($('#c-shap'), ['B','T','F'].map(f => { const x = sh[f]; return {label: FN[f], v: x?.v*1000, lo: x?.lo*1000, hi: x?.hi*1000, color: FC(f),
    tip: `<b>${FN[f]}</b><br>Fair share ${k3(x?.v)} (${k3(x?.lo)} to ${k3(x?.hi)})`}; }), v => sgn(v, 2), {left:110, rowH:38});
  const tot = ['B','T','F'].reduce((a, f) => a + (sh[f]?.v || 0), 0), full = PER().models.BTF?.gain?.v;
  $('#shap-note').innerHTML = `The three shares add up to the full model's gain: ${k3(tot)} ≈ ${k3(full)}. A share near zero means that family adds nothing on average, whatever it is combined with.`;
  const starts = ['M','B','T','F','BT','BF','TF'], M_ = PER().marginal || {};
  const head = [{h:'Starting from', c:'l'}, ...['B','T','F'].map(f => ({h:`+ ${FN[f]}`}))];
  const rows = starts.map(sx => ({v:[sx], c:[`${dots(sx === 'M' ? '' : sx)}${sx === 'M' ? (S.mkt === 'raw' ? 'Nothing' : 'Market only') : cname(sx)}`,
    ...['B','T','F'].map(f => { if (sx.includes(f)) return '<span class="note">—</span>'; const g = M_[`${sx}+${f}`]; return g ? `${k3(g.v)}${sig(g) ? ' •' : ''}` : '–'; })],
    st:[null, ...['B','T','F'].map(f => sx.includes(f) ? '' : heat((M_[`${sx}+${f}`] || {}).v*1000, 3) + ';text-align:center')]}));
  sortable($('#t-marg'), head, rows);
  $('#t-marg').querySelectorAll('tbody tr').forEach((tr, i) => tr.querySelectorAll('td').forEach((td, j) => { if (!j) return; const f = 'BTF'[j-1], sx = starts[i], g = M_[`${sx}+${f}`];
    if (g) hit(td, `<b>${sx === 'M' ? 'Market' : cname(sx)} + ${FN[f]}</b><br>${k3(g.v)} (${k3(g.lo)} to ${k3(g.hi)})<br>p ${num(g.p, 3)} · ${int(g.n)} matches`); }));
}
// ---------------------------------------------------------------- 03 cumulative
function cumView(){
  const C = MK().cum || {}, hl = S.hl || (S.coal === 'M' ? 'BTF' : S.coal);
  $('#cum-chips').innerHTML = R.coals.map(c => `<span class="chip ${c === hl ? 'on' : ''}" data-c="${c}">${cname(c)}</span>`).join('');
  $$('#cum-chips .chip').forEach(ch => ch.onclick = () => { S.hl = ch.dataset.c; cumView(); });
  lines($('#c-cum'), R.coals.map(c => ({name: cname(c), color: c === hl ? css('--s1') : css('--ctrl'), w: c === hl ? 2.6 : 1.2, op: c === hl ? 1 : .55, hl: c === hl,
    pts: C[c] || [], end: c === hl ? cname(c) : ''})), {fmt: v => sgn(v, 1), ref:0, vline: R.cov.holdout_from, vlabel:'hold-out', xlabel: v => v});
}
// ---------------------------------------------------------------- 04 calibration + betting
function calView(){
  const c = S.coal === 'M' ? 'BTF' : S.coal; $('#cal-name').textContent = cname(c);
  if (S.mkt !== '1x2'){ empty($('#c-cal'), 'Calibration is shown for the 1X2 models.'); }
  else calChart($('#c-cal'), R.calib[c], R.calib.M);
  const B = R.bets?.[S.price]; const per = S.per === 'all' ? 'holdout' : S.per;
  if (!B){ empty($('#c-bet'), 'No prices'); return; }
  $('#bet-note').textContent = `${per === 'holdout' ? 'Hold-out' : 'Discovery'} bets at ${S.price} closing prices; thresholds were chosen on the discovery period. Whiskers: 95% interval.`;
  const mroi = B.M?.[per]?.roi;
  $('#bet-warn').innerHTML = mroi != null && mroi > .03 && B.M[per].n >= 30 ? `<div class="callout warn">⚠ The closing market <i>itself</i> shows ${pct(mroi)} at these prices. A model cannot beat itself, so these prices include stale or unbettable quotes: treat every return on this price set as unreliable.</div>` : '';
  hbars($('#c-bet'), ['M', ...R.coals].map(k => { const x = B[k]?.[per]; return {label: k === 'M' ? 'Closing market (control)' : cname(k), v: x?.roi, lo: x?.lo, hi: x?.hi, sel: k === S.coal,
    dots: k === 'M' ? null : ['B','T','F'].map(f => ({f, on: k.includes(f)})), color: x && x.lo != null && x.lo > 0 ? css('--s1') : css('--ctrl'),
    tip: `<b>${k === 'M' ? 'Market' : cname(k)}</b><br>${int(x?.n)} bets · ${pct(x?.roi)}${x?.lo != null ? ` (${pct(x.lo)} to ${pct(x.hi)})` : ''}<br>edge threshold ${pctu(B[k]?.tau, 0)}`}; }), v => pct(v, 0), {left:205});
  const pl = B[c]; $('#pl-name').textContent = cname(c) + ' · ' + S.price + ' prices';
  lines($('#c-pl'), [{name:'Profit', color:css('--s1'), pts: pl?.cum || [], end: pl?.cum?.length ? sgn(pl.cum[pl.cum.length-1][1], 1) + 'u' : '', hl:true}], {fmt: v => sgn(v, 0), ref:0, vline: R.cov.holdout_from, vlabel:'hold-out', empty:'No bets', H:240});
}
// ---------------------------------------------------------------- 05 drivers
function driverView(){
  const I = R.imp; if (!I || !I.families){ empty($('#c-fam'), 'Not enough data'); return; }
  $('#imp-note').textContent = `The full model (Base + Tactical + Form, 1X2) was trained on the discovery period and scored on the hold-out (log loss ${num(I.base_ll, 4)} vs market ${num(I.market_ll, 4)}). Each family of features is scrambled in turn: the rise in log loss (×10⁻³) is how much the model relies on it for forecasts that hold up.`;
  hbars($('#c-fam'), I.families.map(f => ({label: f.name.slice(3), v: f.inc*1000, color: FC(f.origin), dots: [{f: f.origin, on:true}], tip: `<b>${esc(f.name.slice(3))}</b> (${FN[f.origin]})<br>${f.n} features<br>log loss +${num(f.inc*1000, 3)}×10⁻³ when scrambled`})), v => sgn(v, 2), {left:230, rowH:26});
  sortable($('#t-feat'), [{h:'Feature', c:'l'}, {h:'Family', c:'l'}, {h:'Importance ×10⁻³'}], I.features.map(f => ({v:[f.name, f.family, f.inc], c:[`<span class="dots"><i style="background:${FC(f.origin)};border-color:${FC(f.origin)}"></i></span><span class="mono">${esc(f.name)}</span>`, esc(f.family.slice(3)), num(f.inc*1000, 3)]})), {sortKey:2});
}
// ---------------------------------------------------------------- 06 downturns
function downView(){
  const D = R.down || {}; const keys = Object.keys(D);
  if (!keys.length){ ['#c-diff','#c-team','#c-addf'].forEach(s => empty($(s), 'Not available')); return; }
  if (!D[S.downDef]) S.downDef = keys[0];
  $('#down-chips').innerHTML = keys.map(k => `<span class="chip ${k === S.downDef ? 'on' : ''}" data-k="${k}">${esc(D[k].label)}</span>`).join('');
  $$('#down-chips .chip').forEach(c => c.onclick = () => { S.downDef = c.dataset.k; downView(); });
  const P_ = D[S.downDef].periods[S.per] || D[S.downDef].periods.holdout || Object.values(D[S.downDef].periods)[0];
  if (!P_){ empty($('#c-diff'), 'No rows'); return; }
  const G = P_.grid; let h = '<thead><tr><th class="l">Home ↓ · Away →</th>' + [0,1,2,3,'4+'].map(j => `<th style="text-align:center">${j}</th>`).join('') + '</tr></thead><tbody>';
  G.forEach((row, i) => { h += `<tr><td class="l"><b>${i === 4 ? '4+' : i}</b></td>` + row.map((c, j) => `<td class="cell" data-i="${i}" data-j="${j}" style="${heat(c.v, .4)}">${c.v == null ? `<span class="note">n ${c.n}</span>` : `${sgn(c.v, 2)}<div class="note" style="margin:0">n ${c.n}</div>`}</td>`).join('') + '</tr>'; });
  $('#t-grid').innerHTML = h + '</tbody>';
  $$('#t-grid td.cell').forEach(td => { const c = G[+td.dataset.i][+td.dataset.j];
    hit(td, `<b>${td.dataset.i}${td.dataset.i == 4 ? '+' : ''} home vs ${td.dataset.j}${td.dataset.j == 4 ? '+' : ''} away starters in a downturn</b><br>${int(c.n)} matches<br>${c.v == null ? 'too few matches' : `home did ${sgn(c.v, 2)} points vs the market's expectation`}`);
    td.onclick = () => { $$('#t-grid td.cell').forEach(x => x.classList.remove('on')); td.classList.add('on'); const v = c.v;
      $('#grid-detail').innerHTML = v == null ? `Only ${c.n} matches - too few to read.` : `${c.n} matches with ${td.dataset.i} home and ${td.dataset.j} away starters in a downturn: the home side took <b class="${cls(v)}">${sgn(v, 2)}</b> points per match more than the closing prices implied (${sgn(v*38, 1)} over a 38-match season).`; }; });
  vbars($('#c-diff'), P_.diff.map(r => ({label: (r.k > 0 ? '+' : '') + r.k + (Math.abs(r.k) === 3 ? (r.k > 0 ? '+' : '−') : ''), v: r.v, lo: r.lo, hi: r.hi, n: r.n, color: r.lo != null && (r.lo > 0 || r.hi < 0) ? css('--s1') : css('--ctrl'),
    tip: `<b>Home has ${r.k > 0 ? r.k + ' more' : r.k < 0 ? -r.k + ' fewer' : 'the same number of'} players in a downturn</b><br>${int(r.n)} matches<br>home points vs market ${sgn(r.v, 3)}${r.lo != null ? ` (${sgn(r.lo, 3)} to ${sgn(r.hi, 3)})` : ''}`})), v => sgn(v, 2), {xtitle:'Home minus away starters in a downturn'});
  $('#team-chips').innerHTML = [['pts', 'Points vs market'], ['goals', 'Goals vs expected goals']].map(([k, l]) => `<span class="chip ${S.teamMetric === k ? 'on' : ''}" data-k="${k}">${l}</span>`).join('');
  $$('#team-chips .chip').forEach(c => c.onclick = () => { S.teamMetric = c.dataset.k; downView(); });
  vbars($('#c-team'), P_.team.map(r => { const x = r[S.teamMetric] || {}; return {label: r.k === 5 ? '5+' : String(r.k), v: x.v, lo: x.lo, hi: x.hi, n: x.n, color: x.lo != null && (x.lo > 0 || x.hi < 0) ? css('--s1') : css('--ctrl'),
    tip: `<b>${r.k}${r.k === 5 ? '+' : ''} of the team's starters in a downturn</b><br>${pctu(r.share)} of team-matches<br>${S.teamMetric === 'pts' ? 'points vs market' : 'goals vs expected'} ${sgn(x.v, 3)}${x.lo != null ? ` (${sgn(x.lo, 3)} to ${sgn(x.hi, 3)})` : ''}`}; }), v => sgn(v, 2), {xtitle:'Own starters in a downturn'});
  vbars($('#c-addf'), P_.gain_by_total.map(r => { const x = r.BF_vs_B || {}; return {label: r.k === 5 ? '5+' : String(r.k), v: x.v != null ? x.v*1000 : null, lo: x.lo != null ? x.lo*1000 : null, hi: x.hi != null ? x.hi*1000 : null, n: x.n,
    color: sig(x) ? css('--fF') : css('--ctrl'), tip: `<b>${r.k}${r.k === 5 ? '+' : ''} starters in a downturn (both sides)</b><br>Base + Form vs Base: ${k3(x.v)}${x.lo != null ? ` (${k3(x.lo)} to ${k3(x.hi)})` : ''}<br>${int(x.n)} matches`}; }), v => sgn(v, 1), {xtitle:'Starters in a downturn, both sides'});
}
// ---------------------------------------------------------------- 07 player stats
function statView(){
  const ST = R.stats; if (!ST || !ST.stats.length){ $('#t-stats').innerHTML = '<tr><td class="note">Player stats were not run (--no-player-stats).</td></tr>'; $('#stat-detail').style.display = 'none'; return; }
  $('#stat-chips').innerHTML = [['naive', 'Improvement over his plain average'], ['P', 'Improvement over his history model (P)']].map(([k, l]) => `<span class="chip ${S.statMode === k ? 'on' : ''}" data-k="${k}">${l}</span>`).join('');
  $$('#stat-chips .chip').forEach(c => c.onclick = () => { S.statMode = c.dataset.k; statView(); });
  const per = S.per === 'all' ? 'holdout' : S.per, G = ST.groups;
  const gname = g => g.split('').map(x => x === 'P' ? 'History' : FN[x]).join(' + ');
  let h = '<thead><tr><th class="l">Stat</th><th>Avg</th>' + G.map(g => `<th style="text-align:center" title="${gname(g)}">${g === 'P' ? 'P' : 'P+' + g.slice(1)}</th>`).join('') + '</tr></thead><tbody>';
  ST.stats.forEach((s, i) => { const P_ = s.periods[per]; if (!P_) return;
    h += `<tr><td class="l">${esc(s.label)}</td><td>${num(s.mean, s.kind === 'level' ? 2 : 2)}</td>` + G.map(g => { const x = P_.groups[g];
      const v = S.statMode === 'naive' ? x?.vs_naive : (g === 'P' ? 0 : x?.vs_P?.v); const on = S.statCell && S.statCell[0] === i && S.statCell[1] === g;
      return `<td class="cell ${on ? 'on' : ''}" data-i="${i}" data-g="${g}" style="${heat(v, S.statMode === 'naive' ? .3 : .06)}">${v == null ? '–' : pct(v, 1)}${S.statMode === 'P' && x?.vs_P && sig(x.vs_P) ? ' •' : ''}</td>`; }).join('') + '</tr>'; });
  $('#t-stats').innerHTML = h + '</tbody>';
  $$('#t-stats td.cell').forEach(td => { const s = ST.stats[+td.dataset.i], x = s.periods[per].groups[td.dataset.g];
    hit(td, `<b>${esc(s.label)} · ${gname(td.dataset.g)}</b><br>vs his plain average: ${pct(x.vs_naive)}${x.vs_P ? `<br>vs history model: ${pct(x.vs_P.v)} (${pct(x.vs_P.lo)} to ${pct(x.vs_P.hi)})` : ''}${x.auc1 != null ? `<br>P(at least one) AUC ${num(x.auc1, 3)}` : ''}${x.r2 != null ? `<br>R² ${num(x.r2, 3)} · RMSE ${num(x.rmse, 3)}` : ''}`);
    td.onclick = () => { S.statCell = [+td.dataset.i, td.dataset.g]; statView(); }; });
  if (!S.statCell) S.statCell = [0, G[G.length-1]];
  const s = ST.stats[S.statCell[0]], P_ = s.periods[per]; if (!P_) return;
  const box = $('#stat-detail');
  box.innerHTML = `<h3>${esc(s.label)} <span class="note">· ${int(P_.n)} starter-matches (${per})</span></h3><div class="grid g2" style="margin-top:6px"><div><p class="note">Calibration of the richest mix: predicted vs actual average, by decile of the prediction.</p><div id="c-scal"></div></div>
    <div><p class="note">${s.kind === 'count' ? 'How well each mix ranks who records at least one (AUC).' : 'Variance explained (R²) by each mix.'}</p><div id="c-sauc"></div></div></div><div id="s-extra" class="note"></div>`;
  lines($('#c-scal'), [{name:'Actual', color:css('--s1'), pts: P_.calib.map(r => [r[0], r[1], r[2]]), dots:true, hl:true}, {name:'Perfect', color:css('--axis'), pts: P_.calib.map(r => [r[0], r[0]]), dash:'4 3', notip:true}],
    {fmt: v => num(v, 2), xfmt: v => num(v, 1), xlabel: v => 'predicted ' + num(v, 2), extra: p => int(p[2]) + ' starter-matches', H:230, right:30, zero:false});
  const key = s.kind === 'count' ? 'auc1' : 'r2';
  hbars($('#c-sauc'), G.map(g => ({label: gname(g), v: P_.groups[g][key], color: g === S.statCell[1] ? css('--s1') : css('--ctrl'), tip: `<b>${gname(g)}</b><br>${key === 'auc1' ? 'AUC' : 'R²'} ${num(P_.groups[g][key], 3)}`})), v => num(v, 3), {left:200, rowH:24});
  const roles = Object.entries(P_.roles || {}).map(([k, v]) => `${k} ${pct(v.gain)}`).join(' · '), fs = Object.entries(P_.form_state || {}).map(([k, v]) => `${k}: ${pct(v.gain)} (${int(v.n)})`).join(' · ');
  $('#s-extra').innerHTML = `${roles ? `<b>Richest mix vs history alone, by position:</b> ${roles}<br>` : ''}${fs ? `<b>Form signals vs history alone, by the player's current form:</b> ${fs}` : ''}`;
}
// ---------------------------------------------------------------- 08 matchups
const corrTxt = c => c ? `r = ${sgn(c.r, 3)} (${sgn(c.lo, 3)} to ${sgn(c.hi, 3)}), ${int(c.n)} games, ${int(c.players)} players` : 'too few games';
function verdictOf(cc){ const d = cc?.discovery, h = cc?.holdout; if (!h) return ['Not enough data', '']; const persists = h.lo > 0 || h.hi < 0; const same = d && Math.sign(d.r) === Math.sign(h.r);
  return persists && same ? ['Persists out of sample', 'ok'] : persists ? ['Significant only in the hold-out', ''] : ['No reliable pattern', 'bad']; }
function forest(host, rows){ hbars(host, rows.map(r => ({label: r.label, v: r.c?.holdout?.r, lo: r.c?.holdout?.lo, hi: r.c?.holdout?.hi, v2: r.c?.discovery?.r, sel: r.sel, onclick: r.onclick,
  color: r.c?.holdout && (r.c.holdout.lo > 0 || r.c.holdout.hi < 0) ? css('--s1') : css('--ctrl'), tip: `<b>${esc(r.label)}</b><br>Hold-out: ${corrTxt(r.c?.holdout)}<br>Discovery: ${corrTxt(r.c?.discovery)}`})), v => sgn(v, 2), {left:220}); }
function bucketChart(host, B, xl){ if (!B) return empty(host, 'Too few games');
  lines(host, [{name:'Discovery', color:css('--ctrl'), pts: B.filter(r => r.discovery.v != null).map(r => [r.q, r.discovery.v, r.discovery.n]), dots:true},
               {name:'Hold-out', color:css('--s1'), pts: B.filter(r => r.holdout.v != null).map(r => [r.q, r.holdout.v, r.holdout.n]), dots:true, hl:true}],
    {fmt: v => sgn(v, 2), ref:0, xticks: B.map(r => r.q), xfmt: v => 'Q' + v, xlabel: v => xl + ' quintile ' + v, extra: p => int(p[2]) + ' games', H:250, right:30, zero:false}); }
function muView(){
  const M = R.match || {}; const tabs = [['teams', 'Against particular teams'], ['style', 'Against opponent styles'], ['direct', 'Against direct opponents']].filter(t => M[t[0]] && (t[0] !== 'style' || Object.keys(M.style).length));
  if (!tabs.length){ $('#mu-body').innerHTML = '<div class="card note">Not enough data.</div>'; return; }
  if (!tabs.find(t => t[0] === S.mutab)) S.mutab = tabs[0][0];
  $('#mu-tabs').innerHTML = tabs.map(([k, l]) => `<span class="chip ${S.mutab === k ? 'on' : ''}" data-k="${k}">${l}</span>`).join('');
  $$('#mu-tabs .chip').forEach(c => c.onclick = () => { S.mutab = c.dataset.k; muView(); });
  const body = $('#mu-body');
  if (S.mutab === 'teams'){ const T = M.teams, [vt, vc] = verdictOf(T.corr);
    body.innerHTML = `<div class="grid g2"><div class="card"><h3>Does a player's record against a team predict the next meeting? <span class="badge ${vc}">${vt}</span></h3><p class="note">His past performance-vs-norm against this opponent, minus the same measure in the games he played just before each of those meetings (a lag-matched placebo: past games feed his own baseline, so any raw record correlates with the next game by construction). Shrunk with ${T.k} pseudo-games. Raw record: hold-out r ${sgn(T.raw?.holdout?.r, 3)}; placebo alone: ${sgn(T.placebo?.holdout?.r, 3)}. ${T.split_half ? `Split-half check on ${int(T.split_half.pairs)} pairs with 4+ meetings: r = ${sgn(T.split_half.r, 3)}.` : ''}</p><div id="c-mt1"></div><div id="c-mt2"></div></div>
      <div class="card"><h3>Bogey teams and favourite opponents</h3><p class="note">Players with 4+ meetings: average vs his norm against this team, minus the games just before those meetings, shrunk (descriptive - only trust it if the controlled test on the left persists).</p><input id="mt-q" placeholder="Search player or team" style="margin-bottom:8px;min-width:220px"><div class="scroll"><table id="t-mt"></table></div></div></div>`;
    forest($('#c-mt1'), [{label:'Record vs team (placebo-controlled)', c:T.corr}, {label:'Raw record vs team', c:T.raw}, {label:'Placebo: the games just before', c:T.placebo}]);
    bucketChart($('#c-mt2'), T.buckets, 'Record vs team (placebo-controlled)');
    const draw = () => { const q = ($('#mt-q').value || '').toLowerCase();
      sortable($('#t-mt'), [{h:'Player', c:'l'}, {h:'Opponent', c:'l'}, {h:'Meetings'}, {h:'Avg vs norm'}, {h:'Controlled (shrunk)'}, {h:'Last'}], T.table.filter(r => !q || (r.player + ' ' + r.opp).toLowerCase().includes(q)).map(r => ({v:[r.player, r.opp, r.n, r.mean, r.shrunk, r.last], c:[esc(r.player), esc(r.opp), r.n, `<span class="${cls(r.mean)}">${sgn(r.mean, 2)}</span>`, `<b class="${cls(r.shrunk)}">${sgn(r.shrunk, 2)}</b>`, r.last]})), {sortKey:4, sortDir:1}); };
    $('#mt-q').oninput = draw; draw(); }
  else if (S.mutab === 'style'){ const ST = M.style, attrs = Object.keys(ST); if (!S.styleAttr || !ST[S.styleAttr]) S.styleAttr = attrs[0];
    const AN = {press:'pressing', poss:'possession', direct:'direct play', line:'a high line', aerial:'aerial play', aggression:'aggression (fouls)', width:'width (crosses)', dribble:'dribbling'};
    const [vt, vc] = verdictOf(ST[S.styleAttr].corr);
    body.innerHTML = `<div class="grid g2"><div class="card"><h3>Is a player's sensitivity to an opponent style persistent?</h3><p class="note">For each player, his earlier games give a slope: how his performance-vs-norm moved with the opponent's style (standardised against earlier matches, shrunk). Slope × this opponent's style = predicted effect. Bars: hold-out correlation with what then happened (circle: discovery). Click a style.</p><div id="c-ms1"></div>${M.style_combined ? `<p class="note">All styles combined: hold-out ${corrTxt(M.style_combined.corr.holdout)}.</p>` : ''}</div>
      <div class="card"><h3>Opponents strong in ${AN[S.styleAttr] || S.styleAttr} <span class="badge ${vc}">${vt}</span></h3><div id="c-ms2"></div><h3 style="margin-top:6px">Most style-sensitive players</h3><p class="note">Negative slope = struggles when the opponent is strong in ${AN[S.styleAttr] || S.styleAttr}. 20+ games.</p><div class="scroll" style="max-height:280px"><table id="t-ms"></table></div></div></div>`;
    forest($('#c-ms1'), attrs.map(a => ({label: 'vs ' + (AN[a] || a), c: ST[a].corr, sel: a === S.styleAttr, onclick: () => { S.styleAttr = a; muView(); }})));
    bucketChart($('#c-ms2'), ST[S.styleAttr].buckets, 'Predicted style effect');
    sortable($('#t-ms'), [{h:'Player', c:'l'}, {h:'Games'}, {h:'Slope per 1 sd'}], (ST[S.styleAttr].table || []).map(r => ({v:[r.player, r.n, r.slope], c:[esc(r.player), r.n, `<b class="${cls(r.slope)}">${sgn(r.slope, 3)}</b>`]})), {sortKey:2, sortDir:1}); }
  else { const D = M.direct, [vt, vc] = verdictOf(D.defender_form.corr);
    const rows = [{label:"Defender's usual form", c: D.defender_form.corr}, ...Object.entries(D).filter(([k]) => k.startsWith('duel_vs_')).map(([k, v]) => ({label:`Defender duels → attacker ${v.label}`, c: v.corr})), {label:'Same full-back before (controlled)', c: D.pair_history.corr}];
    body.innerHTML = `<div class="grid g2"><div class="card"><h3>Does the man marking him matter? <span class="badge ${vc}">${vt}</span></h3><p class="note">${int(D.defender_form.wingers)} winger-vs-full-back (same flank) and ${int(D.defender_form.strikers)} striker-vs-centre-backs match-ups, defenders' form measured before the match. ${pctu(D.pair_history.repeat_share, 0)} of winger match-ups had met before.</p><div id="c-md1"></div><div id="c-md2"></div></div>
      <div class="card"><h3>Repeat duels</h3><p class="note">Winger vs the same full-back, 3+ meetings: his performance vs norm (shrunk).</p><div class="scroll"><table id="t-md"></table></div></div></div>`;
    forest($('#c-md1'), rows);
    bucketChart($('#c-md2'), D.defender_form.buckets, "Defender's form");
    sortable($('#t-md'), [{h:'Attacker', c:'l'}, {h:'Defender', c:'l'}, {h:'Meetings'}, {h:'Avg vs norm'}, {h:'Shrunk'}], (D.pairs || []).map(r => ({v:[r.attacker, r.defender, r.n, r.mean, r.shrunk], c:[esc(r.attacker), esc(r.defender), r.n, sgn(r.mean, 2), `<b class="${cls(r.shrunk)}">${sgn(r.shrunk, 2)}</b>`]})), {sortKey:4, sortDir:1}); }
}
// ---------------------------------------------------------------- 09 explorer
const RW = R.rows, NR = RW.id.length;
$('#mx-league').innerHTML = '<option value="">All competitions</option>' + [...new Set(RW.league)].sort().map(l => `<option>${esc(l)}</option>`).join('');
let MX = [];
function mxFilter(){ const q = $('#mx-q').value.trim().toLowerCase(), lg = $('#mx-league').value, per = $('#mx-per').value, so = $('#mx-sort').value;
  MX = []; for (let i = 0; i < NR; i++){ if (lg && RW.league[i] !== lg) continue; if (per === 'hold' && !RW.hold[i]) continue; if (per === 'disc' && RW.hold[i]) continue;
    if (q && !(RW.home[i] + ' ' + RW.away[i]).toLowerCase().includes(q)) continue; MX.push(i); }
  const dis = i => Math.max(...[0,1,2].map(k => Math.abs((RW.P.BTF[k][i] ?? 0) - (RW.M[k][i] ?? 0))));
  const dn = i => (RW.h_n_down_obs?.[i] || 0) + (RW.a_n_down_obs?.[i] || 0);
  if (so === 'date') MX.sort((a, b) => RW.date[b] < RW.date[a] ? -1 : 1); else if (so === 'dis') MX.sort((a, b) => dis(b) - dis(a)); else MX.sort((a, b) => dn(b) - dn(a));
  S.mxPage = 0; mxDraw(); }
function mxDraw(){ const s0 = S.mxPage*25, sl = MX.slice(s0, s0+25);
  const head = '<thead><tr><th class="l">Date</th><th class="l">Match</th><th>Score</th><th>Market H·D·A</th><th>Full model H·D·A</th><th>In a downturn (home·away)</th></tr></thead>';
  $('#t-mx').innerHTML = head + '<tbody>' + sl.map(i => `<tr data-i="${i}" style="cursor:pointer" class="${S.mxSel === i ? 'sel' : ''}"><td class="l">${RW.date[i]}</td><td class="l">${esc(RW.home[i])} – ${esc(RW.away[i])}<div class="why">${esc(RW.league[i])}</div></td><td><b>${RW.hg[i]}–${RW.ag[i]}</b></td>
    <td>${[0,1,2].map(k => pctu(RW.M[k][i], 0)).join(' · ')}</td><td>${[0,1,2].map(k => pctu(RW.P.BTF[k][i], 0)).join(' · ')}</td><td>${RW.h_n_down_obs ? `${RW.h_n_down_obs[i] ?? '–'} · ${RW.a_n_down_obs[i] ?? '–'}` : '–'}</td></tr>`).join('') + '</tbody>';
  $$('#t-mx tbody tr').forEach(tr => tr.onclick = () => { S.mxSel = +tr.dataset.i; mxDraw(); mxDetail(S.mxSel); });
  $('#pg-info').textContent = MX.length ? `${s0+1}–${Math.min(s0+25, MX.length)} of ${MX.length.toLocaleString()}` : 'No matches'; }
function mxDetail(i){ const res = RW.hg[i] > RW.ag[i] ? 0 : RW.hg[i] === RW.ag[i] ? 1 : 2;
  const bar = p => `<div class="pbar">${[0,1,2].map(k => `<span style="width:${(p[k]*100).toFixed(1)}%;background:${k === 0 ? css('--fB') : k === 1 ? css('--ctrl') : css('--fT')};opacity:${k === res ? 1 : .45}" title="${['Home','Draw','Away'][k]} ${pctu(p[k], 1)}"></span>`).join('')}</div>`;
  const models = [['M', 'Closing market'], ...R.coals.map(c => [c, cname(c)])];
  const risk = RW.risk[RW.id[i]];
  const side = (L, team) => L && L.length ? L.map(r => `<div>${esc(r[0])} <span class="note">${esc(r[1])}</span> · next-game risk <b>${pctu(r[2], 0)}</b>${r[3] != null ? ` · sustained ${pctu(r[3], 0)}` : ''}</div>`).join('') : '<span class="note">–</span>';
  $('#mx-detail').innerHTML = `<h3>${esc(RW.home[i])} ${RW.hg[i]}–${RW.ag[i]} ${esc(RW.away[i])} <span class="note">${RW.date[i]} · ${esc(RW.league[i])}</span></h3>
    <div class="legend"><span class="key"><span class="sw" style="background:var(--fB)"></span>Home</span><span class="key"><span class="sw" style="background:var(--ctrl)"></span>Draw</span><span class="key"><span class="sw" style="background:var(--fT)"></span>Away</span><span class="key">full opacity = what happened</span></div>
    <table><tbody>${models.map(([c, n]) => { const p = c === 'M' ? [0,1,2].map(k => RW.M[k][i]) : [0,1,2].map(k => RW.P[c][k][i]); return `<tr><td class="l" style="width:230px">${c === 'M' ? '' : dots(c)}${n}</td><td style="width:55%">${p[0] == null ? '–' : bar(p)}</td><td>${p[0] == null ? '' : p.map(x => pctu(x, 0)).join(' · ')}</td></tr>`; }).join('')}</tbody></table>
    <div class="grid g2" style="margin-top:10px"><div><h3>${esc(RW.home[i])}: most at-risk starters</h3>${side(risk?.[0])}</div><div><h3>${esc(RW.away[i])}: most at-risk starters</h3>${side(risk?.[1])}</div></div>`; }
$('#pg-prev').onclick = () => { if (S.mxPage > 0){ S.mxPage--; mxDraw(); } }; $('#pg-next').onclick = () => { if ((S.mxPage+1)*25 < MX.length){ S.mxPage++; mxDraw(); } };
['#mx-q'].forEach(s => $(s).oninput = mxFilter); ['#mx-league', '#mx-per', '#mx-sort'].forEach(s => $(s).onchange = mxFilter);
// ---------------------------------------------------------------- 10 rigour
function rigour(){
  const P = R.params, C = R.cov;
  $('#safe').innerHTML = [
    'Base and tactical features come from day-by-day engines: each day\'s matches are described from the state before that day, and results enter only afterwards.',
    'Form features use each player\'s earlier appearances only, with every baseline (league × position, team, opponent, venue, game state) built from earlier days.',
    `Form risk used in a match comes from a form model whose training labels all ended before the block being predicted (audited: ${R.audit.form.n} fits).`,
    `Match models retrain every ${P.retrain_months} months on earlier matches only (audited: ${R.audit.coalitions.n} blocks); player-stat models every ${P.stats_retrain_months} months (${R.audit.stats.n} blocks).`,
    'Shrinkage of each model\'s correction is chosen inside its training window. Downturn cut-offs and bet thresholds are frozen before they are used.',
    'Time-travel tests re-run each engine with the future deleted: features for the test day must not change. Importance comes from a model trained on discovery and scored on the hold-out.',
    `Seven combinations are tested against the market, so hold-out p-values are Holm-adjusted; an alarm fires if any model beats the closing market by more than ${(P.alarm_gain*1000).toFixed(0)}×10⁻³.`].map(x => `<li>${x}</li>`).join('');
  const L = R.leak;
  $('#t-leak').innerHTML = L.skipped ? '<tr><td class="note">Skipped</td></tr>' : '<thead><tr><th class="l">Engine</th><th class="l">Cut-off</th><th>Rows</th><th>Features</th><th>Changed</th></tr></thead><tbody>' + L.tests.map(t => `<tr><td class="l">${t.engine}</td><td class="l">${t.day}</td><td>${int(t.matches)}</td><td>${t.features}</td><td class="${t.mismatches ? 'neg' : 'pos'}">${t.mismatches}</td></tr>`).join('') + '</tbody>';
  const fq = C.form_quality || {};
  const cv = [['Matches evaluated', `${int(C.eval)} (${int(C.n_disc)} discovery · ${int(C.n_hold)} hold-out)`], ['Competitions', C.leagues], ['Period', `${C.from} → ${C.to}`],
    ['Features', `Base ${C.n_base} · Tactical ${C.n_tac} · Form ${C.n_form}`], ['Form models (out-of-sample AUC)', Object.entries(fq).map(([k, v]) => `${k} ${num(v.auc, 3)}`).join(' · ')],
    ['Performance measure', esc(C.perf)], ['Downturn cut-offs', `observed ≤ ${num(C.thresholds?.obs, 3)} · next-game risk ≥ ${num(C.thresholds?.next, 3)} · sustained ≥ ${num(C.thresholds?.sustained, 3)}`],
    ['With Over/Under prices', pctu(C.with_ou)], ['With sharp prices', pctu(C.with_sharp)], ['Bookmaker margin', pctu(C.overround, 2)], ['Learner (form, stats)', esc(C.learner)],
    ['Compute', C.compute ? `${C.compute.workers} worker(s) × ${C.compute.threads} thread(s)` : '–'], ['Run time', `${Math.floor(C.runtime_s/3600)} h ${Math.floor(C.runtime_s%3600/60)} min`]];
  $('#t-cov').innerHTML = '<tbody>' + cv.map(r => `<tr><td class="l">${r[0]}</td><td style="white-space:normal">${r[1]}</td></tr>`).join('') + '</tbody>';
  const H = R.res['1x2'].periods.holdout?.models || {};
  $('#t-holm').innerHTML = '<thead><tr><th class="l">Combination</th><th>Gain ×10⁻³</th><th>Raw p</th><th>Holm p</th></tr></thead><tbody>' + R.coals.map(c => `<tr><td class="l">${dots(c)}${cname(c)}</td><td class="${cls(H[c]?.gain?.v)}">${k3(H[c]?.gain?.v)}</td><td>${num(H[c]?.gain?.p, 3)}</td><td><b>${num(R.res['1x2'].holm[c], 3)}</b></td></tr>`).join('') + '</tbody>';
}
// ---------------------------------------------------------------- nav, controls, keys
function nav(){ const secs = $$('main section'); $('#nav').innerHTML = '<div class="brand">Model report</div>' + secs.map((s, i) => `<a href="#${s.id}" data-id="${s.id}"><span class="n">${String(i+1).padStart(2, '0')}</span>${s.querySelector('h2').textContent}</a>`).join('') + `<div class="foot">${esc(R.db)}<br>${R.generated}</div>`;
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) $$('#nav a').forEach(a => a.classList.toggle('on', a.dataset.id === e.target.id)); }), {rootMargin:'-30% 0px -60% 0px'}); secs.forEach(s => io.observe(s)); }
function drawLab(){ venn(); tiles(); coalBars(); cumView(); calView(); }
function drawAll(){ drawLab(); shapView(); driverView(); downView(); statView(); muView(); mxFilter(); }
seg('#seg-mkt', [['1x2', '1X2 vs market'], ...(R.res.ou ? [['ou', 'Over/Under 2.5']] : []), ...(R.res.raw ? [['raw', 'No odds']] : [])], 'mkt', () => { drawLab(); shapView(); });
seg('#seg-per', [['holdout', 'Hold-out'], ['discovery', 'Discovery'], ['all', 'All']], 'per', drawAll);
seg('#seg-price', Object.keys(R.bets || {}).map(k => [k, k[0].toUpperCase() + k.slice(1)]), 'price', () => { tiles(); calView(); });
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; drawAll(); };
document.addEventListener('keydown', e => { if (e.target.matches('input,select')) return; const k = +e.key; if (k >= 1 && k <= 7){ S.coal = R.coals[k-1]; S.hl = null; drawLab(); } });
header(); nav(); rigour(); drawAll();
</script></body></html>
'''

if __name__ == "__main__":
    main()
