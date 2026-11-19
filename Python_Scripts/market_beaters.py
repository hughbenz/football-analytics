#!/usr/bin/env python3
"""
MARKET-BEATING PLAYERS - when he plays, does his team beat the betting market?
And which of his attributes explain it?

MODEL (market-adjusted RAPM)
  * Every match is split into STINTS: spells with the same 22 players on the pitch
    (cut at substitutions and red cards).
  * Market expectation per stint: each side's scoring rate from the closing 1X2
    odds (decision_impact.solve_rates), scaled to the stint length and to
    players sent off.  Target = goal difference above the market, per 90.
  * Weighted ridge regression (Bayesian RAPM): +1 home players, -1 away players,
    with team-season effects (lightly penalised - they absorb anything collective,
    so a player is measured against his own teammates) and score-state effects
    (unpenalised).  Penalty chosen by match-grouped cross-validation; posterior
    standard errors from the inverse normal matrix.
  * A second fit with separate attack / defence columns splits the impact into
    goals created above market and goals prevented above market.

SIGNAL CHECKS (shown first in the dashboard)
  * Cross-validated skill: does adding players improve out-of-sample predictions
    beyond team effects?   * Season-forward: do earlier seasons' player ratings
    predict the latest season?   * Split-half reliability (odd vs even matches).

DRIVERS
  Per-90 stats (from lineup details), standardised within position, regressed on
  the impact estimates (precision-weighted ridge, cross-fitted R^2, bootstrap
  intervals): which attributes predict market-beating, and per player which
  attributes push his estimate up or down.

RUN (same folder as load_league.py and decision_impact.py; no API calls)
    pip install numpy
    python3 market_beaters.py                  # top 5 European leagues
    python3 market_beaters.py --leagues 8 82   # any leagues (fitted jointly)
    python3 market_beaters.py --min-minutes 1500
Then open market_beaters.html.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

try:
    import numpy as np
except ImportError:
    sys.exit('This script needs numpy:  pip install numpy')
try:
    import load_league as L
    from decision_impact import implied_1x2, solve_rates, RED_OWN, RED_OPP, MATCH_MINUTES
except ImportError:
    sys.exit("load_league.py and decision_impact.py must be in the same folder as this script.")

TOP5 = [8, 564, 82, 384, 301]
REPORT = "market_beaters.html"
LAMBDAS = [10, 30, 100, 300, 1000, 3000]   # player penalty, in 90-minute units of evidence
TEAM_SHRINK = 0.1                           # team-season effects penalised 10x less
FOLDS = 5
DENSE_LIMIT = 6000                          # exact posterior SEs up to this many columns
T_DF = 3                                    # Student-t prior on player effects (heavy tails); 0 = plain ridge
GOALS = {"GOAL": 1, "PENALTY": 1, "OWNGOAL": -1}
REDS = ("REDCARD", "YELLOWREDCARD")
rng = np.random.default_rng(7)


# ================================================================ SPARSE RIDGE ===
class Design:
    """Sparse design matrix as (row, col, value) triplets."""
    def __init__(self, r, c, v, n, p):
        self.r, self.c, self.v, self.n, self.p = np.array(r), np.array(c), np.array(v, float), n, p

    def mv(self, b):
        return np.bincount(self.r, self.v * b[self.c], minlength=self.n)

    def rmv(self, z):
        return np.bincount(self.c, self.v * z[self.r], minlength=self.p)


def ridge(X, y, w, pen, mask=None, tol=1e-8, iters=800):
    """Solve (X'WX + diag(pen)) b = X'Wy by preconditioned conjugate gradient."""
    ww = w if mask is None else w * mask
    diag = np.bincount(X.c, X.v ** 2 * ww[X.r], minlength=X.p) + pen
    A = lambda b: X.rmv(ww * X.mv(b)) + pen * b
    rhs = X.rmv(ww * y)
    b = np.zeros(X.p); r = rhs.copy(); z = r / diag; p = z.copy(); rz = r @ z
    stop = tol * (np.linalg.norm(rhs) or 1)
    for _ in range(iters):
        Ap = A(p); a = rz / (p @ Ap); b += a * p; r -= a * Ap
        if np.linalg.norm(r) < stop:
            break
        z = r / diag; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    return b


def robust_ridge(X, y, w, pen, k, mask=None, rounds=4):
    """Ridge with a Student-t prior on the first k columns (players), fitted by EM:
    each player's penalty is rescaled by (df+1)/(df + b^2/s0^2), so players with
    strong, consistent evidence are shrunk less while noise stays shrunk."""
    b = ridge(X, y, w, pen, mask)
    if not T_DF:
        return b, pen
    ww = w if mask is None else w * mask
    for _ in range(rounds):
        s2 = float(np.sum(ww * (y - X.mv(b)) ** 2) / max(1, np.count_nonzero(ww)))
        s0 = s2 / np.maximum(pen[:k], 1e-9)                     # prior variance implied by the penalty
        p2 = pen.copy()
        p2[:k] = pen[:k] * (T_DF + 1) / (T_DF + b[:k] ** 2 / s0)
        b = ridge(X, y, w, p2, mask)
    return b, p2


def posterior_se(X, y, w, pen, b):
    """Bayesian ridge: Cov = sigma^2 (X'WX + Pen)^-1 (exact if small enough)."""
    resid = y - X.mv(b)
    s2 = float(np.sum(w * resid ** 2) / max(1, X.n - 1))
    if X.p > DENSE_LIMIT:
        diag = np.bincount(X.c, X.v ** 2 * w[X.r], minlength=X.p) + pen
        return np.sqrt(s2 / diag), s2
    M = np.zeros((X.p, X.p))
    order = np.argsort(X.r, kind="stable")
    r, c, v = X.r[order], X.c[order], X.v[order]
    bounds = np.searchsorted(r, np.arange(X.n + 1))
    for i in range(X.n):
        cs, vs = c[bounds[i]:bounds[i + 1]], v[bounds[i]:bounds[i + 1]]
        M[np.ix_(cs, cs)] += w[i] * np.outer(vs, vs)
    M[np.diag_indices(X.p)] += pen
    return np.sqrt(s2 * np.diag(np.linalg.inv(M))), s2


def wloss(X, y, w, b, mask):
    return float(np.sum(mask * w * (y - X.mv(b)) ** 2))


# ===================================================================== DATA ===
def load(db, leagues):
    q = db.query
    lim = " AND f.league_id = ANY(%(lg)s)" if leagues else ""
    P = {"lg": leagues}
    matches = {r[0]: dict(zip(("lg", "se", "date", "h", "a"), r[1:])) for r in q(
        "SELECT f.fixture_id, f.league_id, f.season_id, f.starting_at, f.home_team_id, f.away_team_id "
        "FROM football.fixtures f WHERE f.state_id IN (SELECT state_id FROM core.states WHERE developer_name "
        "IN ('FT','AET','FT_PEN')) AND f.home_team_id IS NOT NULL" + lim, P)}
    odds = implied_1x2(db, list(matches))
    matches = {k: v for k, v in matches.items() if k in odds}
    ids = list(matches)
    print(f"Finished matches with closing odds: {len(ids):,}")
    lineups = defaultdict(lambda: defaultdict(list))
    positions = defaultdict(Counter)
    for fid, tid, pid, ltype, pos in q(
            "SELECT l.fixture_id, l.team_id, l.player_id, upper(coalesce(t.developer_name, '')), l.position_id "
            "FROM football.fixture_lineups l LEFT JOIN core.types t ON t.type_id = l.type_id "
            "WHERE l.fixture_id = ANY(%(ids)s) AND l.player_id IS NOT NULL", {"ids": ids}):
        if "BENCH" not in ltype and "SUB" not in ltype:
            lineups[fid][tid].append(pid)
        if pos:
            positions[pid][pos] += 1
    events = defaultdict(list)
    for fid, minute, extra, order, eid, tid, pid, rel, dev in q(
            "SELECT e.fixture_id, e.minute, e.extra_minute, e.sort_order, e.event_id, e.team_id, e.player_id, "
            "e.related_player_id, t.developer_name FROM football.fixture_events e JOIN core.types t "
            "ON t.type_id = e.type_id WHERE e.fixture_id = ANY(%(ids)s) AND e.rescinded IS NOT TRUE "
            "AND t.developer_name = ANY(%(d)s) AND coalesce(e.minute, 0) <= 125",
            {"ids": ids, "d": list(GOALS) + list(REDS) + ["SUBSTITUTION"]}):
        events[fid].append(((minute or 0) + (extra or 0), order or 0, eid, tid, pid, rel, dev))
    return matches, odds, lineups, positions, events


def build_stints(matches, odds, lineups, events):
    """-> list of (fid, home_on, away_on, minutes, home_goals, away_goals, exp_home, exp_away, state)."""
    out, skipped = [], 0
    for fid, m in matches.items():
        h, a = m["h"], m["a"]
        on = {h: set(lineups[fid].get(h, [])), a: set(lineups[fid].get(a, []))}
        if min(len(on[h]), len(on[a])) < 10:
            skipped += 1
            continue
        lh, la = solve_rates(*odds[fid])
        evs = sorted(events[fid])
        end = max(MATCH_MINUTES - 1, (evs[-1][0] + 1) if evs else 0)
        cur = {"t": 0, "gh": 0, "ga": 0, "state": 0}
        diff = 0

        def close(t1):
            dur = t1 - cur["t"]
            if dur > 0:
                men = len(on[h]) - len(on[a])
                fh, fa = (RED_OWN ** -men, RED_OPP ** -men) if men < 0 else (RED_OPP ** men, RED_OWN ** men)
                out.append((fid, tuple(on[h]), tuple(on[a]), dur, cur["gh"], cur["ga"],
                            lh * dur / MATCH_MINUTES * fh, la * dur / MATCH_MINUTES * fa, cur["state"]))
            cur.update(t=t1, gh=0, ga=0, state=max(-2, min(2, diff)))

        for t, _, _, tid, pid, rel, dev in evs:
            side = h if tid == h else a if tid == a else None
            if side is None:
                continue
            if dev in GOALS:
                scorer = side if GOALS[dev] == 1 else (a if side == h else h)
                cur["gh" if scorer == h else "ga"] += 1
                diff += 1 if scorer == h else -1
            elif dev == "SUBSTITUTION":
                out_p, in_p = (pid, rel) if pid in on[side] else (rel, pid)
                if out_p in on[side] and in_p and in_p not in on[side]:
                    close(min(t, end)); on[side].discard(out_p); on[side].add(in_p)
            elif dev in REDS and pid in on[side]:
                close(min(t, end)); on[side].discard(pid)
        close(end)
    print(f"Stints: {len(out):,} (skipped {skipped:,} matches with incomplete lineups)")
    return out


# ================================================================ MODEL FITS ===
def designs(stints, matches):
    players = sorted({p for s in stints for p in s[1] + s[2]})
    teamseasons = sorted({(matches[s[0]][k], matches[s[0]]["se"]) for s in stints for k in ("h", "a")})
    pi, ti = {p: i for i, p in enumerate(players)}, {t: i for i, t in enumerate(teamseasons)}
    n, t = len(players), len(teamseasons)
    S = {-2: 0, -1: 1, 1: 2, 2: 3}
    # total model
    r, c, v, y, w = [], [], [], [], []
    for i, (fid, hon, aon, dur, gh, ga, eh, ea, st) in enumerate(stints):
        m = matches[fid]
        for p in hon: r.append(i); c.append(pi[p]); v.append(1)
        for p in aon: r.append(i); c.append(pi[p]); v.append(-1)
        r += [i, i]; c += [n + ti[(m["h"], m["se"])], n + ti[(m["a"], m["se"])]]; v += [1, -1]
        if st: r.append(i); c.append(n + t + S[st]); v.append(1)
        y.append(((gh - eh) - (ga - ea)) * 90 / dur); w.append(dur / 90)
    tot = (Design(r, c, v, len(stints), n + t + 4), np.array(y), np.array(w))
    pen_tot = lambda lam: np.r_[np.full(n, lam), np.full(t, lam * TEAM_SHRINK), np.full(4, 1e-3)]
    # attack/defence model: one row per side
    r, c, v, y, w = [], [], [], [], []
    for i, (fid, hon, aon, dur, gh, ga, eh, ea, st) in enumerate(stints):
        m = matches[fid]
        for j, (att, dfn, g, e, tA, tD, s) in enumerate(((hon, aon, gh, eh, m["h"], m["a"], st),
                                                          (aon, hon, ga, ea, m["a"], m["h"], -st))):
            row = 2 * i + j
            for p in att: r.append(row); c.append(pi[p]); v.append(1)
            for p in dfn: r.append(row); c.append(n + pi[p]); v.append(1)
            r += [row, row]; c += [2 * n + ti[(tA, m["se"])], 2 * n + t + ti[(tD, m["se"])]]; v += [1, 1]
            if s: r.append(row); c.append(2 * n + 2 * t + S[s]); v.append(1)
            y.append((g - e) * 90 / dur); w.append(dur / 90)
    od = (Design(r, c, v, 2 * len(stints), 2 * n + 2 * t + 4), np.array(y), np.array(w))
    pen_od = lambda lam: np.r_[np.full(2 * n, lam), np.full(2 * t, lam * TEAM_SHRINK), np.full(4, 1e-3)]
    return players, n, tot, pen_tot, od, pen_od


def fit(stints, matches):
    players, n, (X, y, w), pen, (Xo, yo, wo), pen_o = designs(stints, matches)
    fids = np.array([s[0] for s in stints])
    uniq = np.unique(fids)
    fold = dict(zip(uniq, rng.integers(0, FOLDS, len(uniq))))
    folds = np.array([fold[f] for f in fids])
    no_players = lambda lam: np.r_[np.full(n, 1e12), pen(lam)[n:]]

    def cv(penalty):
        loss = 0.0
        for k in range(FOLDS):
            b = robust_ridge(X, y, w, penalty, n, mask=(folds != k).astype(float))[0]
            loss += wloss(X, y, w, b, (folds == k).astype(float))
        return loss
    base = float(np.sum(w * y ** 2))
    scores = {lam: cv(pen(lam)) for lam in LAMBDAS}
    lam = min(scores, key=scores.get)
    team_only = cv(no_players(lam))
    print(f"Cross-validated penalty: {lam}  (skill vs market: team-only {1 - team_only / base:+.4%}, "
          f"with players {1 - scores[lam] / base:+.4%})")
    b, pen_fit = robust_ridge(X, y, w, pen(lam), n)
    se, s2 = posterior_se(X, y, w, pen_fit, b)
    bo = robust_ridge(Xo, yo, wo, pen_o(lam), 2 * n)[0]

    # season-forward: earlier seasons predict the latest one
    season_start = defaultdict(lambda: "9999")
    for s in stints:
        m = matches[s[0]]
        season_start[m["se"]] = min(season_start[m["se"]], str(m["date"]))
    last = max(season_start, key=season_start.get)
    test = np.array([matches[s[0]]["se"] == last for s in stints], float)
    forward = None
    if 0 < test.sum() < len(test):
        bf = robust_ridge(X, y, w, pen(lam), n, mask=1 - test)[0]
        base_t = float(np.sum(test * w * y ** 2))
        forward = {"skill": 1 - wloss(X, y, w, bf, test) / base_t, "season": last}

    # split-half reliability (odd vs even matches)
    half = np.array([fold[f] % 2 for f in fids], float)
    b1 = robust_ridge(X, y, w, pen(lam), n, mask=half)[0]
    b2 = robust_ridge(X, y, w, pen(lam), n, mask=1 - half)[0]
    mins1 = np.bincount(X.c, np.abs(X.v) * w[X.r] * half[X.r] * 90, minlength=X.p)[:n]
    mins2 = np.bincount(X.c, np.abs(X.v) * w[X.r] * (1 - half[X.r]) * 90, minlength=X.p)[:n]
    ok = (mins1 >= 600) & (mins2 >= 600)
    r_half = float(np.corrcoef(b1[:n][ok], b2[:n][ok])[0, 1]) if ok.sum() > 10 else float("nan")
    minutes = np.bincount(X.c, np.abs(X.v) * w[X.r] * 90, minlength=X.p)[:n]
    signal = {"lambda": lam, "cvTeam": 1 - team_only / base, "cvFull": 1 - scores[lam] / base,
              "cvAdded": (team_only - scores[lam]) / base, "forward": forward,
              "splitHalf": r_half, "splitHalfFull": 2 * r_half / (1 + r_half) if r_half == r_half else None,
              "splitHalfPlayers": int(ok.sum()), "stints": len(stints), "matches": len(uniq),
              "players": n, "sigma90": s2 ** 0.5, "tDf": T_DF, "lambdas": {str(k): 1 - v / base for k, v in scores.items()}}
    return players, b[:n], se[:n], bo[:n], -bo[n:2 * n], minutes, signal


# ================================================================== DRIVERS ===
def drivers(db, players, est, se, minutes, pos_of, match_ids, min_minutes):
    """Which per-90 attributes (within position) predict a player's market-adjusted impact?"""
    keep = [i for i, p in enumerate(players) if minutes[i] >= min_minutes and pos_of.get(p)]
    if len(keep) < 40:
        return None
    ids = [players[i] for i in keep]
    rows = db.query(
        "SELECT ld.player_id, t.developer_name, max(t.name), sum(ld.value), avg(ld.value), count(*) "
        "FROM football.fixture_lineup_details ld JOIN core.types t ON t.type_id = ld.type_id "
        "WHERE ld.fixture_id = ANY(%(f)s) AND ld.player_id = ANY(%(p)s) AND ld.value IS NOT NULL "
        "GROUP BY 1, 2", {"f": list(match_ids), "p": ids})
    stat, names, peak = defaultdict(dict), {}, defaultdict(float)
    mins = dict(zip(ids, (minutes[i] for i in keep)))
    for pid, key, name, total, mean, cnt in rows:
        names[key] = name or key
        peak[key] = max(peak[key], float(mean or 0))
        rate_like = any(s in (key or "").upper() for s in ("RATING", "ACCURA", "PERCENT", "PCT")) or "%" in (name or "")
        stat[key][pid] = float(mean) if rate_like else float(total) * 90 / max(mins[pid], 1)
    skip = {"MINUTES_PLAYED", "CAPTAIN"}
    feats = [k for k in stat if k not in skip and peak[k] > 1 and len(stat[k]) >= 0.5 * len(ids)]
    if not feats:
        return None
    y = np.array([est[i] for i in keep]); s = np.array([se[i] for i in keep])
    tau2 = max(float(np.var(y) - np.mean(s ** 2)), 1e-4)
    wt = 1 / (s ** 2 + tau2)
    pos = [pos_of[p] for p in ids]
    Z = np.zeros((len(ids), len(feats)))
    for j, k in enumerate(feats):
        col = np.array([stat[k].get(p, np.nan) for p in ids])
        for g in set(pos):                                   # z-score within position
            idx = np.array([q == g for q in pos]) & ~np.isnan(col)
            if idx.sum() > 2:
                mu = np.average(col[idx], weights=wt[idx]); sd = np.sqrt(np.average((col[idx] - mu) ** 2, weights=wt[idx])) or 1
                Z[idx, j] = (col[idx] - mu) / sd
    G = np.array([[q == g for g in sorted(set(pos))] for q in pos], float)   # position intercepts

    def solve(rows_, lam):
        A = np.hstack([Z[rows_], G[rows_]]); W = wt[rows_]
        pen = np.r_[np.full(len(feats), lam), np.full(G.shape[1], 1e-6)]
        return np.linalg.solve(A.T @ (A * W[:, None]) + np.diag(pen), A.T @ (W * y[rows_]))
    fold = rng.integers(0, FOLDS, len(ids)); allr = np.arange(len(ids))
    def oof(lam):
        pred = np.zeros(len(ids))
        for k in range(FOLDS):
            beta = solve(allr[fold != k], lam)
            pred[fold == k] = np.hstack([Z, G])[fold == k] @ beta
        return pred
    best = min((1, 3, 10, 30, 100, 300), key=lambda lam: np.sum(wt * (y - oof(lam)) ** 2))
    pred = oof(best)
    ybar = np.sum(wt * (G @ np.linalg.lstsq(G * np.sqrt(wt)[:, None], y * np.sqrt(wt), rcond=None)[0] - y) ** 2)
    r2 = 1 - np.sum(wt * (y - pred) ** 2) / ybar
    beta = solve(allr, best)[:len(feats)]
    boots = np.array([solve(rng.integers(0, len(ids), len(ids)), best)[:len(feats)] for _ in range(200)])
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    contrib = {p: sorted(((names[k], float(beta[j] * Z[i, j]), float(Z[i, j])) for j, k in enumerate(feats)),
                         key=lambda x: -abs(x[1]))[:6] for i, p in enumerate(ids)}
    return {"features": [{"key": k, "name": names[k], "coef": float(beta[j]), "lo": float(lo[j]), "hi": float(hi[j])}
                         for j, k in enumerate(feats)], "r2": float(r2), "n": len(ids), "lambda": best,
            "contrib": contrib, "explained": dict(zip(ids, (float(Z[i] @ beta) for i in range(len(ids)))))}


# ===================================================================== MAIN ===
def main():
    ap = argparse.ArgumentParser(description="Players whose teams beat the market when they play")
    ap.add_argument("--leagues", nargs="+", type=int, default=TOP5)
    ap.add_argument("--all-leagues", action="store_true")
    ap.add_argument("--min-minutes", type=int, default=900, help="minutes to appear in the drivers model")
    args = ap.parse_args()
    db = L.Db(L.DATABASE)
    db.columns("football.fixtures")
    leagues = None if args.all_leagues else args.leagues
    matches, odds, lineups, positions, events = load(db, leagues)
    stints = build_stints(matches, odds, lineups, events)
    if len(stints) < 500:
        sys.exit("Too few stints to fit the model - check lineups, substitutions and odds are loaded.")
    players, est, se, att, dfn, minutes, signal = fit(stints, matches)

    q = db.query
    pos_names = dict(q("SELECT type_id, name FROM core.types WHERE type_id = ANY(%(i)s)",
                       {"i": list({k for c in positions.values() for k in c})})) if positions else {}
    pos_of = {p: pos_names.get(positions[p].most_common(1)[0][0]) for p in players if positions.get(p)}
    used = {s[0] for s in stints}
    D = drivers(db, players, est, se, minutes, pos_of, used, args.min_minutes)
    if D:
        print(f"Drivers model: {D['n']} players, cross-fitted R^2 {D['r2']:.1%}")

    team_min = defaultdict(Counter)
    for fid, hon, aon, dur, *_ in stints:
        m = matches[fid]
        for p in hon: team_min[p][(m["h"], m["lg"])] += dur
        for p in aon: team_min[p][(m["a"], m["lg"])] += dur
    names = dict(q("SELECT player_id, coalesce(display_name, common_name, name) FROM football.players "
                   "WHERE player_id = ANY(%(i)s)", {"i": players}))
    for pid, nm in q("SELECT DISTINCT ON (player_id) player_id, player_name FROM football.fixture_lineups "
                     "WHERE player_id = ANY(%(i)s)", {"i": [p for p in players if p not in names]}):
        names[pid] = nm
    teams = dict(q("SELECT team_id, name FROM football.teams WHERE team_id = ANY(%(i)s)",
                   {"i": list({k[0] for c in team_min.values() for k in c})}))
    lgs = {r[0]: ((r[2] + " · ") if r[2] else "") + r[1] for r in q(
        "SELECT l.league_id, l.name, c.name FROM football.leagues l LEFT JOIN core.countries c "
        "ON c.country_id = l.country_id WHERE l.league_id = ANY(%(i)s)", {"i": list({m['lg'] for m in matches.values()})})}

    from math import erf, sqrt
    rows = []
    for i, p in enumerate(players):
        (tid, lid), _ = team_min[p].most_common(1)[0]
        rows.append({"id": p, "name": names.get(p, f"player {p}"), "team": teams.get(tid, ""), "league": lgs.get(lid, ""),
                     "pos": pos_of.get(p) or "Unknown", "min": round(float(minutes[i])), "est": round(float(est[i]), 4),
                     "se": round(float(se[i]), 4), "att": round(float(att[i]), 4), "def": round(float(dfn[i]), 4),
                     "p": round(0.5 * (1 + erf(est[i] / (se[i] * sqrt(2)))), 3),
                     "drivers": D["contrib"].get(p) if D else None,
                     "explained": round(D["explained"][p], 4) if D and p in D["explained"] else None})
    data = {"meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "minMinutes": args.min_minutes},
            "signal": signal, "players": rows,
            "drivers": {k: v for k, v in D.items() if k not in ("contrib", "explained")} if D else None}
    path = os.path.abspath(REPORT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"\nDashboard written: {path}  ({len(rows):,} players)\nOpen it in your browser (on a Mac:  open {REPORT})")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Beaters</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#677084;--line:#d9dde5;--grid:#eceff3;--accent:#2d6cdf;--pos:#1f8a5b;--neg:#c0392b;--neutral:#9aa3b2;--att:#d4892a;--def:#2d6cdf}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e6e9ef;--muted:#99a2b3;--line:#2c3340;--grid:#242a35;--accent:#6aa0ff;--pos:#4cc38a;--neg:#e0675a;--neutral:#6f7888;--att:#e6a650;--def:#6aa0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 4px}h1{font-size:20px;margin:0 0 4px}header p{margin:0;color:var(--muted);max-width:940px}
.tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid var(--line);overflow-x:auto}.tabs button{font:inherit;background:none;border:0;border-bottom:2px solid transparent;padding:10px 12px;color:var(--muted);cursor:pointer;white-space:nowrap}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin-bottom:12px}label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
select,input{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
main{padding:16px 20px 30px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:16px}
.card h2{font-size:15px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin-bottom:10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:16px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:500;cursor:pointer}
.l{text-align:left}tr.row{cursor:pointer}tr.row:hover{background:var(--grid)}.scroll{overflow:auto;max-height:680px}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;margin:1px 2px;border:1px solid var(--line)}.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}
.banner{padding:10px 12px;border-radius:10px;margin-bottom:14px;border:1px solid var(--line)}
svg{display:block;width:100%;height:auto}svg text{font-size:11px;fill:var(--ink)}.note{color:var(--muted);font-size:12px}
.method p{max-width:880px;margin:0 0 10px}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;display:none;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.15);max-width:320px}
</style></head><body>
<header><h1>Market Beaters</h1><p>When a player is on the pitch, does his team score and concede better than the closing odds expected? Estimated against everyone else on the pitch at the time and his own teammates, in <b>goal difference per 90 above the market</b>.</p></header>
<nav class="tabs" id="tabs"><button data-tab="board" class="on">Leaderboard</button><button data-tab="player">Player</button><button data-tab="drivers">What drives it</button><button data-tab="signal">Signal check</button><button data-tab="method">How it works</button></nav>
<main id="main"></main><div id="tip"></div>
<script>
const D=__DATA__, $=s=>document.querySelector(s), css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt=(n,d=2)=>isFinite(n)?(+n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}):"–", sg=(n,d=2)=>isFinite(n)?(n>0?"+":"")+fmt(n,d):"–";
let tab="board", sel=null, sortK="est", sortD=-1, F={lg:"all",pos:"all",min:1500,q:"",conf:false};
const S=D.signal, weak=!(S.cvAdded>0) || (S.forward&&!(S.forward.skill>0)) || !(S.splitHalf>0.1);
function banner(){return weak?`<div class="banner" style="border-color:var(--neg)"><b class="neg">Weak signal.</b> The checks on the Signal tab don't show that player ratings predict anything beyond the market and team effects. Treat individual numbers as noise, even where the interval excludes zero.</div>`
 :`<div class="banner" style="border-color:var(--pos)"><b class="pos">Signal present.</b> Player ratings improve out-of-sample predictions beyond the market and team effects (see Signal check). Intervals still matter: most players are indistinguishable from zero.</div>`}
function ci(p,lo,hi){const W=150,X=v=>W/2+Math.max(-1,Math.min(1,v/0.6))*W/2*0.95, c=p.est-1.645*p.se>0?css("--pos"):p.est+1.645*p.se<0?css("--neg"):css("--neutral");
 return `<svg viewBox="0 0 ${W} 14" style="width:${W}px;display:inline-block;vertical-align:middle"><line x1="${W/2}" x2="${W/2}" y1="0" y2="14" stroke="${css("--line")}"/>
 <line x1="${X(p.est-1.645*p.se)}" x2="${X(p.est+1.645*p.se)}" y1="7" y2="7" stroke="${c}" stroke-width="2.5" stroke-linecap="round"/><circle cx="${X(p.est)}" cy="7" r="3.5" fill="${c}"/></svg>`}
function list(){return D.players.filter(p=>(F.lg=="all"||p.league==F.lg)&&(F.pos=="all"||p.pos==F.pos)&&p.min>=F.min&&(!F.q||p.name.toLowerCase().includes(F.q)||p.team.toLowerCase().includes(F.q))
 &&(!F.conf||p.est-1.645*p.se>0||p.est+1.645*p.se<0))}
function board(){const opts=(k)=>[...new Set(D.players.map(p=>p[k]).filter(Boolean))].sort();
 const rows=list().sort((a,b)=>sortD*((typeof a[sortK]=="string")?a[sortK].localeCompare(b[sortK]):a[sortK]-b[sortK]));
 const cols=[["name","Player","l"],["team","Team","l"],["pos","Pos","l"],["min","Minutes"],["est","Impact /90"],["ci","90% interval",""],["p","P(>0)"],["att","Attack"],["def","Defence"],["drv","Top drivers","l"]];
 $("#main").innerHTML=banner()+`<div class="bar">
  <label>Competition<select id="f-lg"><option value="all">All</option>${opts("league").map(x=>`<option>${x}</option>`).join("")}</select></label>
  <label>Position<select id="f-pos"><option value="all">All</option>${opts("pos").map(x=>`<option>${x}</option>`).join("")}</select></label>
  <label>Min. minutes<input id="f-min" type="number" step="100" value="${F.min}" style="width:100px"></label>
  <label>Find<input id="f-q" type="search" value="${F.q}" placeholder="player or team"></label>
  <label><span><input id="f-conf" type="checkbox" ${F.conf?"checked":""}> only clear effects</span></label></div>
 <div class="card"><h2>${rows.length} players</h2><div class="sub">Impact = goal difference per 90 above the closing-odds expectation while he's on the pitch, relative to teammates. Attack = goals created above market, Defence = goals prevented. Click a row for details.</div>
 <div class="scroll"><table><tr>${cols.map(([k,l,c])=>`<th class="${c||""}" data-k="${k}">${l}${sortK==k?(sortD<0?" ▾":" ▴"):""}</th>`).join("")}</tr>
 ${rows.slice(0,600).map(p=>`<tr class="row" data-id="${p.id}"><td class="l">${p.name}</td><td class="l">${p.team}</td><td class="l muted">${p.pos}</td><td>${fmt(p.min,0)}</td>
  <td class="${p.est>0?"pos":"neg"}"><b>${sg(p.est)}</b></td><td>${ci(p)}</td><td>${fmt(p.p*100,0)}%</td><td>${sg(p.att)}</td><td>${sg(p.def)}</td>
  <td class="l">${(p.drivers||[]).slice(0,3).map(d=>`<span class="chip ${d[1]>0?"pos":"neg"}">${d[1]>0?"▲":"▼"} ${d[0]}</span>`).join("")}</td></tr>`).join("")}</table></div>
  ${rows.length>600?`<p class="note">Showing the first 600 - narrow the filters to see more.</p>`:""}</div>`;
 const on=(id,ev,f)=>$(id).addEventListener(ev,f);
 on("#f-lg","change",e=>{F.lg=e.target.value;board()}); on("#f-pos","change",e=>{F.pos=e.target.value;board()});
 on("#f-min","change",e=>{F.min=+e.target.value;board()}); on("#f-q","change",e=>{F.q=e.target.value.toLowerCase();board()}); on("#f-conf","change",e=>{F.conf=e.target.checked;board()});
 $("#f-lg").value=F.lg; $("#f-pos").value=F.pos;
 document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{const k=th.dataset.k; if(k=="ci"||k=="drv")return; sortD=sortK==k?-sortD:(["name","team","pos"].includes(k)?1:-1); sortK=k; board()});
 document.querySelectorAll("tr.row").forEach(tr=>tr.onclick=()=>{sel=+tr.dataset.id; go("player")})}
function hbars(items,title){ // items [{label,value,color}]
 const W=520,rowH=24,H=items.length*rowH+20,mx=Math.max(0.05,...items.map(i=>Math.abs(i.value))),X=v=>260+v/mx*170;
 return `<svg viewBox="0 0 ${W} ${H}" style="max-width:${W*1.15}px"><line x1="260" x2="260" y1="0" y2="${H-14}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`+items.map((it,i)=>{const y=i*rowH+4;
  return `<g data-t="${it.label}: ${sg(it.value,3)}"><text x="250" y="${y+14}" text-anchor="end">${it.label}</text><rect x="${Math.min(260,X(it.value))}" y="${y+3}" width="${Math.abs(X(it.value)-260)||1}" height="15" rx="3" fill="${it.color}"/>
  <text x="${it.value>=0||260-X(it.value)<40?Math.max(260,X(it.value))+5:X(it.value)-5}" y="${y+14}" text-anchor="${it.value>=0||260-X(it.value)<40?"start":"end"}" fill="${css("--muted")}">${sg(it.value,3)}</text></g>`}).join("")+`</svg>`}
function player(){const p=D.players.find(x=>x.id===sel)||list().sort((a,b)=>b.est-a.est)[0]; if(!p){$("#main").innerHTML="<p class='note'>No player selected.</p>";return}
 const verdict=p.est-1.645*p.se>0?`<strong class="pos">Clearly beats the market</strong>`:p.est+1.645*p.se<0?`<strong class="neg">Clearly below the market</strong>`:`<span class="muted">Not distinguishable from zero</span>`;
 const drv=p.drivers||[], rest=p.explained==null?null:p.est-p.explained;
 $("#main").innerHTML=banner()+`<div class="kpis"><div class="kpi"><b>${p.name}</b><span>${p.team} · ${p.pos} · ${p.league} · ${fmt(p.min,0)} minutes</span></div>
  <div class="kpi"><b class="${p.est>0?"pos":"neg"}">${sg(p.est)} /90</b><span>${verdict} · 90%: ${sg(p.est-1.645*p.se)} to ${sg(p.est+1.645*p.se)} · P(&gt;0) ${fmt(p.p*100,0)}%</span></div>
  <div class="kpi"><b>${sg(p.att)} · ${sg(p.def)}</b><span>attack · defence (goals per 90 above market)</span></div></div>
  <div class="card"><h2>Attack vs defence</h2><div class="sub">From a separate fit with attack and defence columns for every player.</div>
  ${hbars([{label:"Goals created above market",value:p.att,color:css("--att")},{label:"Goals prevented above market",value:p.def,color:css("--def")}])}</div>
  <div class="card"><h2>What drives his estimate</h2><div class="sub">${drv.length?`Each bar = attribute effect × how far he is from the average ${p.pos.toLowerCase()} (in SDs). "Not explained" = the part of his impact his stats don't account for.`:"Not in the drivers model (needs "+D.meta.minMinutes+"+ minutes and a known position)."}</div>
  ${drv.length?hbars(drv.map(d=>({label:`${d[0]} (${d[2]>0?"+":""}${fmt(d[2],1)} SD)`,value:d[1],color:d[1]>0?css("--pos"):css("--neg")})).concat(rest==null?[]:[{label:"Not explained by stats",value:rest,color:css("--neutral")}])):""}</div>`}
function drivers(){const R=D.drivers; if(!R){$("#main").innerHTML="<p class='note'>The drivers model needs player statistics (lineup details) for at least 40 players.</p>";return}
 const fs=[...R.features].sort((a,b)=>b.coef-a.coef), W=640,rowH=22,left=230,H=fs.length*rowH+30, lo=Math.min(-0.02,...fs.map(f=>f.lo)), hi=Math.max(0.02,...fs.map(f=>f.hi)), X=v=>left+(v-lo)/(hi-lo)*(W-left-60);
 let s=`<line x1="${X(0)}" x2="${X(0)}" y1="0" y2="${fs.length*rowH}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`;
 fs.forEach((f,i)=>{const y=i*rowH+11, c=f.lo>0?css("--pos"):f.hi<0?css("--neg"):css("--neutral");
  s+=`<g data-t="${f.name}: ${sg(f.coef,3)} per +1 SD (90%: ${sg(f.lo,3)} to ${sg(f.hi,3)})"><text x="${left-8}" y="${y+4}" text-anchor="end">${f.name}</text>
  <line x1="${X(f.lo)}" x2="${X(f.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2.5" stroke-linecap="round"/><circle cx="${X(f.coef)}" cy="${y}" r="4" fill="${c}"/>
  <text x="${W-52}" y="${y+4}" fill="${css("--muted")}">${sg(f.coef,3)}</text></g>`});
 $("#main").innerHTML=banner()+`<div class="kpis"><div class="kpi"><b>${fmt(R.r2*100,1)}%</b><span>of the real (non-noise) differences between players' impacts explained by their stats, out of sample</span></div>
  <div class="kpi"><b>${R.n}</b><span>players in the model (${D.meta.minMinutes}+ minutes, known position)</span></div></div>
  <div class="card"><h2>Which attributes predict beating the market?</h2><div class="sub">Change in impact (goal difference per 90 above market) for +1 standard deviation in each per-90 stat, compared within position, all stats together. Green/red = 90% bootstrap interval clear of zero.</div>
  <svg viewBox="0 0 ${W} ${H}" style="max-width:${W*1.15}px">${s}</svg></div>`}
function signal(){const f=S.forward, row=(v,lab,good)=>`<div class="kpi"><b class="${good===undefined?"":good?"pos":"neg"}">${v}</b><span>${lab}</span></div>`;
 $("#main").innerHTML=banner()+`<div class="kpis">
  ${row(sg(S.cvAdded*100,3)+"%","extra out-of-sample accuracy from adding players, beyond team effects (cross-validated, share of market-residual variance)",S.cvAdded>0)}
  ${row(f?sg(f.skill*100,3)+"%":"–",f?"season-forward: earlier seasons' ratings predicting the latest season (players only; new team effects unknown)":"needs 2+ seasons",f?f.skill>0:undefined)}
  ${row(fmt(S.splitHalf,2),"split-half reliability: ratings from odd vs even matches agree (r, "+S.splitHalfPlayers+" players with 600+ minutes in each half). Full-sample ≈ "+fmt(S.splitHalfFull,2),S.splitHalf>0.1)}
  ${row(fmt(S.sigma90,2),"noise: SD of goal difference vs market per 90 minutes - why effects need thousands of minutes")}
  ${row(fmt(S.matches,0)+" · "+fmt(S.stints,0),"matches · stints")}${row(S.lambda,"cross-validated penalty (90-minute units of evidence each player starts with at zero)")}</div>
  <div class="card"><h2>Reading these</h2><p class="note">Football results are dominated by noise: even a genuinely strong player shifts goal difference by tenths of a goal per 90, against a standard deviation of about ${fmt(S.sigma90,1)}. Positive out-of-sample skill and split-half reliability above about 0.2 mean the leaderboard ranks something real; numbers near zero mean it mostly ranks luck. Effects are relative to teammates, so a great player in a great squad can look modest.</p></div>`}
function method(){$("#main").innerHTML=`<div class="card method"><h2>How it works</h2>
 <p><b>Stints.</b> Each match is split wherever the players on the pitch change (substitutions, red cards). For each stint the market's expected goals come from the closing 1X2 odds turned into scoring rates, scaled by stint length and players sent off.</p>
 <p><b>Model.</b> Goal difference above the market (per 90, weighted by stint length) is regressed on who was on the pitch: +1 home players, −1 away players. Team-season terms absorb anything collective (a squad the market under-rated all year, a manager's system), so each player is measured against his teammates; score-state terms absorb game-state effects. Ridge regularisation (a Bayesian prior centred on zero) with the penalty chosen by cross-validation on whole matches; intervals are Bayesian posterior 90% intervals. A second fit gives every player separate attack and defence terms.</p>
 <p><b>Drivers.</b> Per-90 stats are standardised within position and regressed on players' impacts (precision-weighted ridge, players with more evidence count more). R² is cross-fitted (out of sample) and the attribute intervals come from 200 bootstrap resamples. Per-player bars = attribute effect × the player's standardised value.</p>
 <p class="note">Limits: needs complete starting lineups and substitution events; can't see injuries or tactics that the market knew about; goalkeepers rarely rotate so their estimates are weak; within-team collinearity (players who always play together) is shared out by the prior, not resolved.</p></div>`}
function go(t){tab=t;document.querySelectorAll("#tabs button").forEach(b=>b.classList.toggle("on",b.dataset.tab==t));({board,player,drivers,signal,method})[t]()}
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>go(b.dataset.tab));
document.addEventListener("mousemove",ev=>{const t=ev.target.closest&&ev.target.closest("[data-t]"),tip=$("#tip");if(t){tip.textContent=t.getAttribute("data-t");tip.style.display="block";tip.style.left=Math.min(ev.clientX+12,innerWidth-330)+"px";tip.style.top=(ev.clientY+12)+"px"}else tip.style.display="none"});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>go(tab));
go("board");
</script></body></html>"""

if __name__ == "__main__":
    main()
