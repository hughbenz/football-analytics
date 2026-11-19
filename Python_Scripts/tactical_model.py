#!/usr/bin/env python3
"""
tactical_model.py - tactical & technical profiles, matchups and in-game tweaks
===============================================================================

Builds on backtest_models.py (same folder): same database access, same
day-by-day leak-proof engine pattern, time-travel test, walk-forward training,
compute sizing, progress reporting and dashboard style.

WHAT IT MODELS
--------------
1. Team tactical style (what a team does), 11 ratios:
     possession, directness (long balls / passes), patience (passes / attack),
     width (crosses per 10 minutes on the ball), penetration (dangerous attacks
     / attacks), shot selection (inside-box share), pressing (defensive actions
     per 10 minutes WITHOUT the ball), line height (opponent offsides), aerial
     reliance (aerial share of duels), dribbling (per 10 minutes on the ball),
     aggression (fouls). Rates are per minute on / off the ball so style is not
     confused with possession. Matchups use each attribute as standard
     deviations from its league's running average (past matches only).
2. Team execution quality (how well), 9 ratios:
     xG for / against, xG per shot, finishing (goals - xG), pass accuracy, duel
     and aerial win %, ball security (possession lost per touch), keeper save %.
3. Starting-XI technique, from each starter's per-90 history shrunk towards
   his positional average (so 3 good games do not make a star), grouped by
   line (keeper / defence / midfield / attack): threat, creativity, dribbling,
   crossing, aerial attack, build-up security, ball-winning, aerial defence,
   duel defence, ball security, keeper shot-stopping, XI rating vs usual,
   bench strength.
4. Matchups - how the two profiles interact (both directions):
     press vs build-up, crossing vs aerial defence, direct play vs a high line,
     dribblers vs duel defence, set pieces vs aerial defence, chance quality vs
     keeper, counter-attack vs a possession side, possession clash, style
     distance, and "style memory" (how each team has fared, against the market,
     versus opponents of this style before).
5. In-game tactical habits (team and coach): scoring when behind, conceding
   when ahead, comeback rate, late-game goal difference, first-substitution
   minute, attacking substitutions when chasing, and goal difference after
   the first substitution compared with before.

MODELS
------
Pre-match (gradient boosting on the market's errors, as in backtest_models.py):
  base_gbm            base football features only (form, Elo, xG ratings ...)
  tactical_gbm        tactical features only
  combined_gbm        base + tactical: the difference to base_gbm is exactly
                      what tactics add
  tactical_standalone gradient boosting WITHOUT odds (tactics + strength)
  + Over/Under 2.5 equivalents
In-play (at 15', 30', half-time, 60', 75'), for in-game tweaks:
  ip_state            scoreboard model: pre-match odds, score, minute, red cards
  ip_state_gbm        the same inputs, gradient boosting
  ip_tactical         the boosted scoreboard model plus corrections learned from
                      substitutions so far and how attacking they were, red
                      cards, momentum (last 15'), pressure, both teams' and
                      coaches' in-game habits, bench strength and the pre-match
                      tactical forecast. How much correction to keep is chosen on
                      later dates, so it cannot do worse than the scoreboard model
                      on its own validation data.
  The gap ip_tactical vs ip_state_gbm = value of in-game tactical information.

NO LOOK-AHEAD
-------------
Everything from backtest_models.py applies (day-by-day state, results only read
after the day's rows are written, monthly walk-forward, frozen thresholds, the
time-travel test with future results AND fixtures deleted). In addition:
  - In-play rows use only events, momentum and pressure from BEFORE the
    checkpoint minute; a "minute-travel" test recomputes a sample with all
    later events deleted and requires identical features.
  - The in-play models use the pre-match models' out-of-sample predictions.
  - Feature-family importance comes from a model trained on the discovery
    period and tested on the hold-out.

USAGE
-----
  Same folder as backtest_models.py and load_league.py:
    python3 tactical_model.py
    python3 tactical_model.py --leagues 8,82 --cache .cache --export-csv tactical.csv
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bootstrap():
    """Install missing packages through backtest_models.py (which knows how)."""
    import importlib
    import importlib.util
    mods = ("numpy", "pandas", "scipy", "sklearn")
    need = [m for m in mods if importlib.util.find_spec(m) is None]
    if not (HERE / "backtest_models.py").exists():
        sys.exit("tactical_model.py needs backtest_models.py in the same folder.")
    if not need:
        return
    subprocess.run([sys.executable, str(HERE / "backtest_models.py"), "--install-only"])
    importlib.invalidate_caches()
    if [m for m in mods if importlib.util.find_spec(m) is None]:
        vpy = HERE / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if vpy.exists() and Path(sys.executable).resolve() != vpy.resolve():
            os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])
        sys.exit("Missing packages: " + ", ".join(need))


if __name__ == "__main__":
    _bootstrap()

sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtest_models as bt  # noqa: E402
from backtest_models import log  # noqa: E402

# =============================================================================
# Attribute dictionary: Sportmonks developer names (first alias found is used)
# =============================================================================
PLAYER_ATTR = {
    "minutes": ["MINUTES_PLAYED"], "rating": ["RATING"],
    "passes": ["PASSES", "TOTAL_PASSES"], "acc_passes": ["ACCURATE_PASSES", "SUCCESSFUL_PASSES"],
    "key_passes": ["KEY_PASSES"], "crosses": ["TOTAL_CROSSES", "CROSSES"], "acc_crosses": ["ACCURATE_CROSSES"],
    "long_balls": ["LONG_BALLS", "TOTAL_LONG_BALLS"],
    "drib_att": ["DRIBBLE_ATTEMPTS", "DRIBBLED_ATTEMPTS", "TOTAL_DRIBBLES"], "drib_succ": ["SUCCESSFUL_DRIBBLES"],
    "duels": ["TOTAL_DUELS"], "duels_won": ["DUELS_WON"],
    "aer_won": ["AERIALS_WON", "AERIAL_DUELS_WON"], "aer_lost": ["AERIALS_LOST", "AERIAL_DUELS_LOST"],
    "tackles": ["TACKLES", "TOTAL_TACKLES"], "intercepts": ["INTERCEPTIONS"],
    "fouls": ["FOULS", "FOULS_COMMITTED"], "dispossessed": ["DISPOSSESSED", "POSSESSION_LOST", "BALL_LOSSES"],
    "touches": ["TOUCHES"], "sot": ["SHOTS_ON_TARGET"], "goals": ["GOALS"], "assists": ["ASSISTS"],
    "saves": ["SAVES"], "conceded": ["GOALS_CONCEDED"],
}
PA = list(PLAYER_ATTR)                      # column order of player arrays
PI = {k: i for i, k in enumerate(PA)}
TEAM_ATTR = {
    "poss": ["BALL_POSSESSION"], "passes": ["PASSES", "TOTAL_PASSES"], "acc_passes": ["SUCCESSFUL_PASSES", "ACCURATE_PASSES"],
    "attacks": ["ATTACKS"], "da": ["DANGEROUS_ATTACKS"], "shots": ["SHOTS_TOTAL", "TOTAL_SHOTS"], "sot": ["SHOTS_ON_TARGET"],
    "sib": ["SHOTS_INSIDEBOX"], "corners": ["CORNERS"], "fouls": ["FOULS"], "offsides": ["OFFSIDES"],
    "tackles": ["TACKLES"], "intercepts": ["INTERCEPTIONS"], "crosses": ["TOTAL_CROSSES", "CROSSES"],
    "long_balls": ["LONG_BALLS"], "saves": ["SAVES"],
}
TREND_TYPES = {"da": ["DANGEROUS_ATTACKS"], "sot": ["SHOTS_ON_TARGET"], "shots": ["SHOTS_TOTAL"], "corners": ["CORNERS"]}
CHECKPOINTS = [15, 30, 46, 60, 75]          # 46 = half-time (all first-half events incl. stoppage)

STYLE = ["poss", "direct", "patience", "width", "penetration", "boxshots", "press", "line", "aerial", "dribble", "aggression"]
QUALITY = ["xgf", "xga", "xg_shot", "finish", "pass_acc", "duel", "aerial_win", "security", "gk_save", "corners"]
TEAM_M = STYLE + QUALITY
TI = {k: i for i, k in enumerate(TEAM_M)}
# in-game accumulators (decayed sums): ratios are built from them with shrinkage
GAME = ["trail_min", "trail_gf", "lead_min", "lead_ga", "trailed", "trail_pts", "late_gd", "games", "first_sub", "subs_n",
        "chase_subs", "chase_aggr", "post_min", "post_gd", "pre_min", "pre_gd", "reds"]
GI = {k: i for i, k in enumerate(GAME)}
ROLE_RANK = {"G": 0, "D": 1, "M": 2, "A": 3}
XI_KEYS = ["threat", "creative", "dribble", "cross", "aer_att", "buildup", "ballwin", "aer_def", "duel_def", "security", "gk", "rating"]


def zscore(x, ref):
    """Standard deviations from the league's running average (past matches only)."""
    m, sq = ref[0].get(), ref[1].get()
    sd = np.sqrt(np.maximum(sq - m * m, 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(m) & (sd > 1e-9), (x - m) / sd, np.nan)

# =============================================================================
# Extraction
# =============================================================================
TYPES_SQL = "SELECT type_id, developer_name, lower(coalesce(name, '')) AS name FROM core.types"


def resolve(types: pd.DataFrame, spec: dict, present: set | None = None) -> dict:
    """attribute -> list of type_ids (first alias that exists, optionally restricted to ids present in a table)."""
    by = defaultdict(list)
    for r in types.itertuples():
        if r.developer_name:
            by[str(r.developer_name).upper()].append(int(r.type_id))
    out = {}
    for k, aliases in spec.items():
        for al in aliases:
            ids = [i for i in by.get(al, []) if present is None or i in present]
            if ids:
                out[k] = (al, ids)
                break
    return out


def pivot_sql(table, key_cols, value_expr, found):
    cols = ", ".join(f"max({value_expr}) FILTER (WHERE t.type_id IN ({','.join(map(str, ids))})) AS {k}" for k, (_, ids) in found.items())
    allids = ",".join(str(i) for _, ids in found.values() for i in ids)
    keys = ", ".join(f"t.{c}" for c in key_cols)
    return f"SELECT {keys}, {cols} FROM {table} t WHERE t.type_id IN ({allids}) GROUP BY {keys}"


def tactical_extract(db, args) -> dict:
    """Detect which tactical stats exist, then pull them (pivoted in SQL to keep transfers small)."""
    types = db.query(TYPES_SQL)
    sample = lambda tbl: set(db.query(f"SELECT DISTINCT type_id FROM {tbl} TABLESAMPLE SYSTEM (2)").type_id.dropna().astype(int)) if db.exists(tbl) else set()  # noqa: E731
    p_present, t_present = sample("football.fixture_lineup_details"), sample("football.fixture_statistics")
    pf, tf = resolve(types, PLAYER_ATTR, p_present), resolve(types, TEAM_ATTR, t_present)
    tr_present = sample("football.fixture_trends") if db.exists("football.fixture_trends") else set()
    trf = resolve(types, TREND_TYPES, tr_present)
    log(f"tactical stats found: players {len(pf)}/{len(PLAYER_ATTR)}, team {len(tf)}/{len(TEAM_ATTR)}, momentum {len(trf)}/{len(TREND_TYPES)}")
    queries = {}
    if pf:
        queries["t_players"] = pivot_sql("football.fixture_lineup_details", ["fixture_id", "team_id", "player_id"], "t.value", pf)
    if tf:
        queries["t_team"] = pivot_sql("football.fixture_statistics", ["fixture_id", "team_id"], "t.value", tf)
    queries["t_lineups"] = """
        SELECT l.fixture_id, l.team_id, l.player_id, (lt.developer_name = 'LINEUP')::int AS started,
               upper(coalesce(pt.developer_name, pt.name, '')) AS pos
        FROM football.fixture_lineups l
        LEFT JOIN core.types lt ON lt.type_id = l.type_id
        LEFT JOIN core.types pt ON pt.type_id = l.position_id
        WHERE l.player_id IS NOT NULL AND l.team_id IS NOT NULL"""
    queries["t_events"] = """
        SELECT e.fixture_id, e.team_id, e.player_id, e.related_player_id, e.minute, coalesce(e.extra_minute, 0) AS extra,
               e.result, upper(t.developer_name) AS kind
        FROM football.fixture_events e JOIN core.types t ON t.type_id = e.type_id
        WHERE upper(t.developer_name) IN ('GOAL', 'OWNGOAL', 'PENALTY', 'SUBSTITUTION', 'REDCARD', 'YELLOWREDCARD')
          AND e.rescinded IS NOT TRUE AND e.minute IS NOT NULL"""
    queries["t_formations"] = "SELECT fixture_id, team_id, formation FROM football.fixture_formations WHERE formation IS NOT NULL"
    queries["t_coach_names"] = "SELECT coach_id, COALESCE(display_name, common_name, name) AS name FROM football.coaches"
    queries["t_player_names"] = "SELECT player_id, COALESCE(display_name, common_name, name) AS name FROM football.players"
    if trf:
        # cumulative counts (max) and per-minute counts (sum) before each checkpoint and 15' earlier
        parts = []
        for k, (_, ids) in trf.items():
            for c in CHECKPOINTS:
                for lag in (0, 15):
                    cond = f"t.type_id IN ({','.join(map(str, ids))}) AND t.minute < {c - lag}"
                    parts.append(f"max(t.value) FILTER (WHERE {cond}) AS {k}_max_{c}_{lag}")
                    parts.append(f"sum(t.value) FILTER (WHERE {cond}) AS {k}_sum_{c}_{lag}")
        allids = ",".join(str(i) for _, ids in trf.values() for i in ids)
        queries["t_trends"] = f"SELECT t.fixture_id, t.team_id, {', '.join(parts)} FROM football.fixture_trends t WHERE t.type_id IN ({allids}) GROUP BY 1, 2"
        queries["t_trend_mode"] = f"""
            SELECT type_id, avg((value <= nxt)::int) AS monotone FROM (
              SELECT type_id, value, lead(value) OVER (PARTITION BY fixture_id, team_id, type_id ORDER BY minute) AS nxt
              FROM football.fixture_trends TABLESAMPLE SYSTEM (3) WHERE type_id IN ({allids})) x
            WHERE nxt IS NOT NULL GROUP BY type_id"""
    if db.exists("football.fixture_pressure"):
        queries["t_pressure"] = "SELECT fixture_id, team_id, " + ", ".join(
            f"avg(pressure) FILTER (WHERE minute >= {c - 10} AND minute < {c}) AS pr_{c}" for c in CHECKPOINTS) + \
            " FROM football.fixture_pressure GROUP BY 1, 2"
    d = run_queries(db, queries, args, "tac")
    d["found"] = {"players": {k: v[0] for k, v in pf.items()}, "team": {k: v[0] for k, v in tf.items()},
                  "momentum": {k: v[0] for k, v in trf.items()}}
    mode = d.get("t_trend_mode", pd.DataFrame())
    d["trend_cum"] = {}
    for k, (_, ids) in trf.items():
        m = mode[mode.type_id.isin(ids)].monotone.mean() if len(mode) else np.nan
        d["trend_cum"][k] = bool(m >= 0.95) if pd.notna(m) else True
    return d


def run_queries(db, queries, args, prefix):
    import concurrent.futures as cf
    cache = Path(args.cache) if args.cache else None
    out, todo = {}, []
    for name, sql in queries.items():
        path = cache / f"{prefix}_{name}.pkl" if cache else None
        if path and path.exists() and not args.refresh:
            out[name] = pd.read_pickle(path)
            log(f"{name:14s} {len(out[name]):>11,} rows (cache)")
        else:
            todo.append((name, sql, path))
    if not todo:
        return out
    if bt.PROGRESS:
        bt.PROGRESS.start("reading tactical data", len(todo))

    def run(sql):
        c = db.clone()
        try:
            return c.query(sql), None
        except Exception as e:  # noqa: BLE001
            return None, e
        finally:
            c.close()
    jobs = max(1, min(len(todo), args.db_jobs or max(2, min(6, bt.usable_cpus()))))
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run, sql): (name, path) for name, sql, path in todo}
        for f in cf.as_completed(futs):
            name, path = futs[f]
            res, err = f.result()
            if err is not None:
                log(f"{name}: not available ({str(err).splitlines()[0][:110]})")
                res = pd.DataFrame()
            out[name] = res
            bt.progress(1, name)
            log(f"{name:14s} {len(res):>11,} rows")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                res.to_pickle(path)
    return out


# =============================================================================
# Tactical engine (same pattern as backtest_models.Engine: emit, then update)
# =============================================================================
def role_of(pos: str) -> str | None:
    p = (pos or "").upper()
    if "GOAL" in p or p in ("GK", "G"):
        return "G"
    if "DEF" in p or "BACK" in p:
        return "D"
    if "MID" in p:
        return "M"
    if "ATT" in p or "FORW" in p or "STRIK" in p or "WING" in p:
        return "A"
    return None


def parse_formation(s):
    parts = [int(x) for x in re.findall(r"\d", str(s or ""))]
    if len(parts) < 2 or not 9 <= sum(parts) <= 10:
        return (np.nan, np.nan, np.nan)
    return (float(parts[0]), float(sum(parts[1:-1])), float(parts[-1]))


def safe_div(a, b):
    return a / b if b and np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


class TEW:
    """Vector exponentially weighted mean that ignores missing entries."""
    __slots__ = ("s", "w")

    def __init__(self, n):
        self.s, self.w = np.zeros(n), np.zeros(n)

    def add(self, x, a):
        ok = np.isfinite(x)
        self.s[ok] = (1 - a) * self.s[ok] + a * x[ok]
        self.w[ok] = (1 - a) * self.w[ok] + a

    def get(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.w > 1e-9, self.s / np.where(self.w > 1e-9, self.w, 1), np.nan)


class TacticalEngine:
    A_TEAM = 1 - 0.5 ** (1 / 10)          # team style/quality: half-life 10 matches
    A_GAME = 0.97                         # in-game sums decay per match (half-life ~23 matches)
    P_DECAY = 0.93                        # player sums decay per appearance (half-life ~10)
    K_SHRINK = 3.0                        # player prior strength, in 90-minute matches

    def __init__(self, d: dict, td: dict, mk: pd.DataFrame, args):
        self.args = args
        fx = d["fixtures"].dropna(subset=["home_id", "away_id", "starting_at"]).copy()
        fx[["home_id", "away_id"]] = fx[["home_id", "away_id"]].astype("int64")
        fx["day"] = fx.starting_at.dt.tz_convert(None).dt.normalize()
        self.fx = fx.sort_values("starting_at").reset_index(drop=True)
        self.mkp = {}
        if len(mk) and "p_h" in mk:
            ok = mk[["p_h", "p_d", "p_a"]].dropna()
            self.mkp = {int(k): (a, b, c) for k, a, b, c in zip(ok.index, ok.p_h, ok.p_d, ok.p_a)}
        self.coach = {(int(r.fixture_id), int(r.team_id)): int(r.coach_id) for r in d["coaches"].itertuples()} if len(d["coaches"]) else {}
        # player stats: one sorted matrix, sliced per (fixture, team)
        P = td.get("t_players", pd.DataFrame())
        self.pmat, self.pidx = np.zeros((0, len(PA))), {}
        if len(P):
            P = P.dropna(subset=["fixture_id", "team_id", "player_id"]).sort_values(["fixture_id", "team_id"])
            for k in PA:
                if k not in P:
                    P[k] = np.nan
            self.pmat = P[PA].to_numpy(float)
            self.ppid = P.player_id.astype("int64").to_numpy()
            keys = list(zip(P.fixture_id.astype("int64"), P.team_id.astype("int64")))
            start = 0
            for i in range(1, len(keys) + 1):
                if i == len(keys) or keys[i] != keys[start]:
                    self.pidx[keys[start]] = (start, i)
                    start = i
        T = td.get("t_team", pd.DataFrame())
        self.tstat = {}
        if len(T):
            for r in T.dropna(subset=["fixture_id", "team_id"]).to_dict("records"):
                self.tstat[(int(r.pop("fixture_id")), int(r.pop("team_id")))] = r
        self.xg = {(int(r.fixture_id), int(r.team_id)): r.xg for r in d["xg"].itertuples()} if len(d["xg"]) else {}
        L = td.get("t_lineups", pd.DataFrame())
        self.xi, self.bench, self.role = defaultdict(list), defaultdict(list), {}
        for fid, tid, pid, st, pos in zip(L.fixture_id.astype("int64"), L.team_id.astype("int64"), L.player_id.astype("int64"),
                                          L.started.fillna(0), L.pos.fillna("")) if len(L) else []:
            (self.xi if st == 1 else self.bench)[(fid, tid)].append(pid)
            r = role_of(pos)
            if r:
                self.role[(fid, pid)] = r
        E = td.get("t_events", pd.DataFrame())
        self.events = defaultdict(list)
        if len(E):
            E = E.assign(tm=E.minute.astype(float) + E.extra.fillna(0).astype(float) / 100).sort_values(["fixture_id", "tm"])
            for r in E.itertuples():
                self.events[int(r.fixture_id)].append((r.tm, r.kind, None if pd.isna(r.team_id) else int(r.team_id),
                                                       None if pd.isna(r.player_id) else int(r.player_id),
                                                       None if pd.isna(r.related_player_id) else int(r.related_player_id), r.result))
        Fm = td.get("t_formations", pd.DataFrame())
        self.form = {(int(r.fixture_id), int(r.team_id)): parse_formation(r.formation) for r in Fm.itertuples()} if len(Fm) else {}
        Tr = td.get("t_trends", pd.DataFrame())
        self.trend = {(int(r["fixture_id"]), int(r["team_id"])): r for r in Tr.to_dict("records")} if len(Tr) else {}
        self.trend_cum = td.get("trend_cum", {})
        Pr = td.get("t_pressure", pd.DataFrame())
        self.press = {(int(r["fixture_id"]), int(r["team_id"])): r for r in Pr.to_dict("records")} if len(Pr) else {}
        self.results = {int(r.fixture_id): (r.hg, r.ag) for r in self.fx.itertuples()}
        self.reset_state()

    # ---------------------------------------------------------------- state
    def reset_state(self):
        self.team = defaultdict(lambda: TEW(len(TEAM_M)))
        self.game_team = defaultdict(lambda: np.zeros(len(GAME)))
        self.game_coach = defaultdict(lambda: np.zeros(len(GAME)))
        self.game_prior = np.zeros(len(GAME))
        self.ps = {}                                           # player -> decayed sums (len(PA))
        self.prole = defaultdict(lambda: defaultdict(float))   # player -> role votes
        self.rprior = {r: np.zeros(len(PA)) for r in "GDMA"}   # role -> decayed sums (league-wide)
        self.team_xi_rating = defaultdict(lambda: TEW(1))
        self.team_ndef = defaultdict(lambda: TEW(1))
        self.memory = defaultdict(lambda: np.zeros((4, 2)))   # team -> style bucket -> [resid sum, weight]
        self.league_ref = defaultdict(lambda: TEW(2))          # league -> mean (possession, press)
        self.lz_team = defaultdict(lambda: (TEW(len(TEAM_M)), TEW(len(TEAM_M))))   # league -> (mean, mean of squares)
        self.lz_xi = defaultdict(lambda: (TEW(len(XI_KEYS)), TEW(len(XI_KEYS))))
        self.emitted_xi = {}
        self.max_result_day = None
        self.inplay = []

    # ---------------------------------------------------------------- players
    def player_rates(self, pid, role):
        """Per-90 rates and success ratios, shrunk towards the role average."""
        s = self.ps.get(pid)
        pr = self.rprior[role]
        k = self.K_SHRINK
        prior90 = pr / (pr[PI["minutes"]] / 90) if pr[PI["minutes"]] > 0 else np.full(len(PA), np.nan)
        if s is None:
            s = np.zeros(len(PA))
        m90 = s[PI["minutes"]] / 90
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = (s + k * prior90) / (m90 + k)
        return rate

    def player_role(self, fid, pid):
        r = self.role.get((fid, pid))
        if r:
            return r
        votes = self.prole.get(pid)
        return max(votes, key=votes.get) if votes else "M"

    def xi_profile(self, fid, team):
        xi = self.xi.get((fid, team), [])
        if not xi or not self.ps:
            return {}
        R = {p: self.player_role(fid, p) for p in xi}
        rates = {p: self.player_rates(p, R[p]) for p in xi}
        g = lambda key, roles: np.array([rates[p][PI[key]] for p in xi if R[p] in roles])  # noqa: E731

        def ssum(key, roles):
            v = g(key, roles)
            return float(np.nansum(v)) if v.size and np.isfinite(v).any() else np.nan

        def ratio(num, den, roles):
            a, b = ssum(num, roles), ssum(den, roles)
            return safe_div(a, b)
        out = {
            "threat": ssum("sot", "MA") * 0.3 + ssum("key_passes", "MA") * 0.1 + ssum("goals", "MA") * 0.4 + ssum("assists", "MA") * 0.2,
            "creative": ssum("key_passes", "DMA"),
            "dribble": ssum("drib_succ", "MA"),
            "cross": ssum("acc_crosses", "DMA"),
            "aer_att": safe_div(ssum("aer_won", "A"), ssum("aer_won", "A") + ssum("aer_lost", "A")),
            "buildup": ratio("acc_passes", "passes", "GDM"),
            "ballwin": ssum("tackles", "DM") + ssum("intercepts", "DM"),
            "aer_def": safe_div(ssum("aer_won", "D"), ssum("aer_won", "D") + ssum("aer_lost", "D")),
            "duel_def": ratio("duels_won", "duels", "DM"),
            "security": ratio("dispossessed", "touches", "DM"),
            "gk": safe_div(ssum("saves", "G"), ssum("saves", "G") + ssum("conceded", "G")),
            "rating": float(np.nanmean(g("rating", "GDMA"))) if np.isfinite(g("rating", "GDMA")).any() else np.nan,
        }
        bench = self.bench.get((fid, team), [])
        br = [self.player_rates(p, self.player_role(fid, p))[PI["rating"]] for p in bench if p in self.ps]
        out["bench"] = float(np.nanmean(br)) - out["rating"] if br and np.isfinite(br).any() and np.isfinite(out["rating"]) else np.nan
        out["rating_vs_usual"] = out["rating"] - self.team_xi_rating[team].get()[0] if np.isfinite(out["rating"]) else np.nan
        return out

    # ---------------------------------------------------------------- in-game profile
    def game_profile(self, sums):
        s, p = sums, self.game_prior
        sh = lambda num, den, k=5: (s[GI[num]] + k * safe_div(p[GI[num]], p[GI["games"]] or 1)) / (s[GI[den]] + k * safe_div(p[GI[den]], p[GI["games"]] or 1)) if (s[GI[den]] + p[GI[den]]) > 0 else np.nan  # noqa: E731
        return {
            "trail_attack": sh("trail_gf", "trail_min") * 90 if s[GI["games"]] + p[GI["games"]] > 0 else np.nan,
            "lead_defence": sh("lead_ga", "lead_min") * 90 if s[GI["games"]] + p[GI["games"]] > 0 else np.nan,
            "comeback": sh("trail_pts", "trailed"),
            "late_gd": sh("late_gd", "games"),
            "first_sub": sh("first_sub", "games"),
            "chase_aggr": sh("chase_aggr", "chase_subs"),
            "chase_rate": sh("chase_subs", "trailed"),
            "sub_impact": (sh("post_gd", "post_min") - sh("pre_gd", "pre_min")) * 90,
            "reds": sh("reds", "games"),
        }

    # ---------------------------------------------------------------- emit (pre-match)
    def emit(self, f):
        if self.max_result_day is not None:
            assert self.max_result_day < f.day, "look-ahead: tactical state contains a result from this day or later"
        fid, h, a = int(f.fixture_id), int(f.home_id), int(f.away_id)
        R = {"fixture_id": fid, "day": f.day}
        prof, xi = {}, {}
        for side, t, o in (("h", h, a), ("a", a, h)):
            v = self.team[t].get()
            prof[side] = dict(zip(TEAM_M, v))
            for k, x in prof[side].items():
                R[f"{side}_{'st' if k in STYLE else 'q'}_{k}"] = x
            xi[side] = self.xi_profile(fid, t)
            for k, x in xi[side].items():
                R[f"{side}_tx_{k}"] = x
            gp = self.game_profile(self.game_team[t])
            for k, x in gp.items():
                R[f"{side}_g_{k}"] = x
            c = self.coach.get((fid, t))
            if c is not None and self.game_coach[c][GI["games"]] >= 3:
                for k, x in self.game_profile(self.game_coach[c]).items():
                    R[f"{side}_c_{k}"] = x
            fm = self.form.get((fid, t), (np.nan,) * 3)
            R[f"{side}_f_def"], R[f"{side}_f_mid"], R[f"{side}_f_fwd"] = fm
            R[f"{side}_f_shape_change"] = fm[0] - self.team_ndef[t].get()[0] if np.isfinite(fm[0]) else np.nan
        # league-relative z-scores, then matchups (attacker side x defender side, both directions)
        zt, zx = {}, {}
        for side, t in (("h", h), ("a", a)):
            zt[side] = dict(zip(TEAM_M, zscore(np.array([prof[side][k] for k in TEAM_M]), self.lz_team[f.league_id])))
            xv = np.array([xi[side].get(k, np.nan) for k in XI_KEYS])
            zx[side] = dict(zip(XI_KEYS, zscore(xv, self.lz_xi[f.league_id])))
            self.emitted_xi[(fid, t)] = xv
        for att, de, tag in (("h", "a", "mh"), ("a", "h", "ma")):
            A, D, XA, XD = zt[att], zt[de], zx[att], zx[de]
            R[f"{tag}_press_vs_build"] = A["press"] * -XD["buildup"]          # our press x their shaky build-up
            R[f"{tag}_cross_vs_air"] = A["width"] * -XD["aer_def"]            # our crossing x their aerial weakness
            R[f"{tag}_direct_vs_line"] = A["direct"] * D["line"]              # our long balls x their high line
            R[f"{tag}_drib_vs_duel"] = XA["dribble"] * -XD["duel_def"]        # our dribblers x their duel losers
            R[f"{tag}_setpiece"] = A["corners"] + XA["aer_att"] - XD["aer_def"]   # set-piece volume and aerial edge
            R[f"{tag}_chance_vs_gk"] = A["xg_shot"] * -XD["gk"]               # our chance quality x their keeper
            R[f"{tag}_counter"] = A["penetration"] * D["poss"]                # our incisiveness x their possession
            R[f"{tag}_threat_vs_win"] = XA["threat"] - XD["ballwin"]          # our attackers vs their ball-winners
        for k in [c[3:] for c in list(R) if c.startswith("mh_")]:
            R[f"md_{k}"] = R[f"mh_{k}"] - R[f"ma_{k}"]
        R["poss_clash"] = prof["h"]["poss"] - prof["a"]["poss"]
        sv = np.array([[prof[s][k] for k in STYLE] for s in ("h", "a")])
        R["style_distance"] = float(np.sqrt(np.nansum((sv[0] - sv[1]) ** 2))) if np.isfinite(sv).all() else np.nan
        # style memory: how each team has done vs the market against opponents of this style
        ref = self.league_ref[f.league_id].get()
        for side, t, o in (("h", h, a), ("a", a, h)):
            b = self.bucket(prof["a" if side == "h" else "h"], ref)
            m = self.memory[t]
            R[f"{side}_memory"] = m[b, 0] / (m[b, 1] + 3) if b is not None else np.nan
        self.emit_inplay(f, R)
        return R

    def bucket(self, p, ref):
        if not np.isfinite(p["poss"]) or not np.isfinite(p["press"]) or not np.isfinite(ref).all():
            return None
        return int(p["poss"] > ref[0]) * 2 + int(p["press"] > ref[1])

    # ---------------------------------------------------------------- in-play rows
    def inplay_state(self, fid, h, a, c, events=None):
        """Everything known at minute c of fixture fid: only events, momentum and pressure before c."""
        ev = [e for e in (self.events.get(fid, []) if events is None else events) if e[0] < c]
        S = {"minute": c, "frac": c / 90}
        gh = ga = 0
        for e in ev:
            if e[1] in ("GOAL", "OWNGOAL", "PENALTY"):
                if e[5] and re.match(r"^\d+\s*-\s*\d+$", str(e[5])):
                    gh, ga = (int(x) for x in str(e[5]).split("-"))
                elif e[2] is not None:
                    scorer_home = (e[2] == h) != (e[1] == "OWNGOAL")
                    gh, ga = gh + scorer_home, ga + (not scorer_home)
        S.update(gh=gh, ga=ga, gd=gh - ga, tot=gh + ga)
        last_goal = max([e[0] for e in ev if e[1] in ("GOAL", "OWNGOAL", "PENALTY")], default=0)
        S["since_goal"] = c - last_goal
        sh = sa = 0
        for side, t in (("h", h), ("a", a)):
            S[f"{side}_red"] = sum(1 for e in ev if e[1] in ("REDCARD", "YELLOWREDCARD") and e[2] == t)
            subs = [e for e in ev if e[1] == "SUBSTITUTION" and e[2] == t]
            S[f"{side}_subs"] = len(subs)
            S[f"{side}_last_sub"] = c - max([e[0] for e in subs], default=c)
            aggr = 0.0
            for e in subs:
                rin, rout = self.player_role(fid, e[3]) if e[3] else "M", self.player_role(fid, e[4]) if e[4] else "M"
                aggr += ROLE_RANK.get(rin, 2) - ROLE_RANK.get(rout, 2)
            S[f"{side}_sub_aggr"] = aggr
        for side, t in (("h", h), ("a", a)):
            tr = self.trend.get((fid, t))
            for k, cum in self.trend_cum.items():
                if tr is None:
                    S[f"{side}_mo_{k}"] = S[f"{side}_mo15_{k}"] = np.nan
                    continue
                if cum:
                    now, then = tr.get(f"{k}_max_{c}_0"), tr.get(f"{k}_max_{c}_15")
                else:
                    now, then = tr.get(f"{k}_sum_{c}_0"), tr.get(f"{k}_sum_{c}_15")
                now = np.nan if now is None or pd.isna(now) else float(now)
                then = 0.0 if then is None or pd.isna(then) else float(then)
                S[f"{side}_mo_{k}"], S[f"{side}_mo15_{k}"] = now, now - then if np.isfinite(now) else np.nan
            pr = self.press.get((fid, t))
            S[f"{side}_pressure"] = float(pr.get(f"pr_{c}")) if pr is not None and pd.notna(pr.get(f"pr_{c}")) else np.nan
        for k in self.trend_cum:
            S[f"d_mo15_{k}"] = S[f"h_mo15_{k}"] - S[f"a_mo15_{k}"]
        # who is chasing, and have they changed things?
        S["h_chasing_aggr"] = S["h_sub_aggr"] * (S["gd"] < 0)
        S["a_chasing_aggr"] = S["a_sub_aggr"] * (S["gd"] > 0)
        return S

    def emit_inplay(self, f, R):
        fid, h, a = int(f.fixture_id), int(f.home_id), int(f.away_id)
        if fid not in self.events:
            return
        for c in CHECKPOINTS:
            S = self.inplay_state(fid, h, a, c)
            S["fixture_id"] = fid
            self.inplay.append(S)

    # ---------------------------------------------------------------- update (results)
    def team_obs(self, fid, t, o, gf, ga):
        T, O = self.tstat.get((fid, t), {}), self.tstat.get((fid, o), {})
        sl = self.pidx.get((fid, t))
        so = self.pidx.get((fid, o))
        Ps = np.nansum(self.pmat[sl[0]:sl[1]], axis=0) if sl else np.full(len(PA), np.nan)
        Po = np.nansum(self.pmat[so[0]:so[1]], axis=0) if so else np.full(len(PA), np.nan)
        has = lambda arr, k: sl is not None and np.isfinite(self.pmat[sl[0]:sl[1], PI[k]]).any()  # noqa: E731

        def val(k, team_d, psum, pk=None, side_sl=sl):
            v = team_d.get(k)
            if v is not None and pd.notna(v):
                return float(v)
            if pk and side_sl is not None and np.isfinite(self.pmat[side_sl[0]:side_sl[1], PI[pk]]).any():
                return float(psum[PI[pk]])
            return np.nan
        passes, acc = val("passes", T, Ps, "passes"), val("acc_passes", T, Ps, "acc_passes")
        opp_passes = val("passes", O, Po, "passes", so)
        attacks, da = val("attacks", T, Ps), val("da", T, Ps)
        shots, sib = val("shots", T, Ps), val("sib", T, Ps)
        tackles, inter, fouls = val("tackles", T, Ps, "tackles"), val("intercepts", T, Ps, "intercepts"), val("fouls", T, Ps, "fouls")
        long_b, crosses = val("long_balls", T, Ps, "long_balls"), val("crosses", T, Ps, "crosses")
        duels, dwon = (float(Ps[PI["duels"]]), float(Ps[PI["duels_won"]])) if has(Ps, "duels") else (np.nan, np.nan)
        aw, al = (float(Ps[PI["aer_won"]]), float(Ps[PI["aer_lost"]])) if has(Ps, "aer_won") and has(Ps, "aer_lost") else (np.nan, np.nan)
        dri = float(Ps[PI["drib_att"]]) if has(Ps, "drib_att") else np.nan
        disp, tch = (float(Ps[PI["dispossessed"]]), float(Ps[PI["touches"]])) if has(Ps, "dispossessed") and has(Ps, "touches") else (np.nan, np.nan)
        saves = val("saves", T, Ps, "saves")
        opp_sot = val("sot", O, Po, "sot", so)
        xgf, xga = self.xg.get((fid, t), np.nan), self.xg.get((fid, o), np.nan)
        poss = T.get("poss")
        x = np.full(len(TEAM_M), np.nan)
        x[TI["poss"]] = float(poss) / 100 if poss is not None and pd.notna(poss) else np.nan
        x[TI["direct"]] = safe_div(long_b, passes)
        x[TI["patience"]] = safe_div(passes, attacks)
        on_ball = x[TI["poss"]] * 90 if np.isfinite(x[TI["poss"]]) else np.nan          # minutes with the ball
        off_ball = (1 - x[TI["poss"]]) * 90 if np.isfinite(x[TI["poss"]]) else np.nan   # minutes without it
        x[TI["width"]] = safe_div(crosses, on_ball) * 10 if np.isfinite(on_ball) else safe_div(crosses, attacks)
        x[TI["penetration"]] = safe_div(da, attacks)
        x[TI["boxshots"]] = safe_div(sib, shots)
        da_ = np.nansum([tackles, inter, fouls]) if np.isfinite([tackles, inter, fouls]).any() else np.nan
        # pressing = defensive actions per 10 minutes WITHOUT the ball (falls back to inverse PPDA)
        x[TI["press"]] = safe_div(da_, off_ball) * 10 if np.isfinite(off_ball) else (safe_div(da_, opp_passes) * 100 if np.isfinite(safe_div(da_, opp_passes)) else np.nan)
        x[TI["line"]] = val("offsides", O, Po)
        x[TI["aerial"]] = safe_div(aw + al, duels) if np.isfinite(aw) else np.nan
        x[TI["dribble"]] = safe_div(dri, on_ball) * 10 if np.isfinite(on_ball) else (safe_div(dri, passes) * 100 if np.isfinite(safe_div(dri, passes)) else np.nan)
        x[TI["aggression"]] = fouls
        x[TI["xgf"]], x[TI["xga"]] = xgf, xga
        x[TI["xg_shot"]] = safe_div(xgf, shots)
        x[TI["finish"]] = gf - xgf if np.isfinite(xgf) else np.nan
        x[TI["pass_acc"]] = safe_div(acc, passes)
        x[TI["duel"]] = safe_div(dwon, duels)
        x[TI["aerial_win"]] = safe_div(aw, aw + al) if np.isfinite(aw) else np.nan
        x[TI["security"]] = safe_div(disp, tch)
        x[TI["gk_save"]] = safe_div(saves, opp_sot)
        x[TI["corners"]] = val("corners", T, Ps)
        return x

    def game_obs(self, fid, t, o, h):
        """In-game behaviour of team t in a finished match, from its events."""
        ev = self.events.get(fid, [])
        g = np.zeros(len(GAME))
        if not ev:
            return None
        gt = go = 0
        prev, first_sub = 0.0, None
        trailed, subs = False, 0
        state_min = {"trail": 0.0, "lead": 0.0}
        post_gd = pre_gd = 0
        for e in ev + [(90.0, "END", None, None, None, None)]:
            tm = min(e[0], 90.0)
            span = max(0.0, tm - prev)
            if gt < go:
                state_min["trail"] += span
            elif gt > go:
                state_min["lead"] += span
            prev = max(prev, tm)
            if e[1] in ("GOAL", "OWNGOAL", "PENALTY"):
                if e[5] and re.match(r"^\d+\s*-\s*\d+$", str(e[5])):
                    hs, as_ = (int(x) for x in str(e[5]).split("-"))
                    ngt, ngo = (hs, as_) if t == h else (as_, hs)
                else:
                    ours = (e[2] == t) != (e[1] == "OWNGOAL")
                    ngt, ngo = gt + ours, go + (not ours)
                if ngt > gt:
                    if gt < go:
                        g[GI["trail_gf"]] += 1
                    if first_sub is not None:
                        post_gd += 1
                    else:
                        pre_gd += 1
                    if tm > 70:
                        g[GI["late_gd"]] += 1
                if ngo > go:
                    if gt > go:
                        g[GI["lead_ga"]] += 1
                    if first_sub is not None:
                        post_gd -= 1
                    else:
                        pre_gd -= 1
                    if tm > 70:
                        g[GI["late_gd"]] -= 1
                gt, go = ngt, ngo
                trailed = trailed or gt < go
            elif e[1] == "SUBSTITUTION" and e[2] == t:
                subs += 1
                if first_sub is None:
                    first_sub = tm
                if gt < go:
                    g[GI["chase_subs"]] += 1
                    rin = self.player_role(fid, e[3]) if e[3] else "M"
                    rout = self.player_role(fid, e[4]) if e[4] else "M"
                    g[GI["chase_aggr"]] += ROLE_RANK.get(rin, 2) - ROLE_RANK.get(rout, 2)
            elif e[1] in ("REDCARD", "YELLOWREDCARD") and e[2] == t:
                g[GI["reds"]] += 1
        g[GI["trail_min"]], g[GI["lead_min"]] = state_min["trail"], state_min["lead"]
        g[GI["trailed"]] = float(trailed)
        g[GI["trail_pts"]] = (3 if gt > go else (1 if gt == go else 0)) if trailed else 0
        g[GI["games"]] = 1
        g[GI["first_sub"]] = first_sub if first_sub is not None else 90
        g[GI["subs_n"]] = subs
        fs = first_sub if first_sub is not None else 90
        g[GI["pre_min"]], g[GI["post_min"]] = fs, 90 - fs
        g[GI["pre_gd"]], g[GI["post_gd"]] = pre_gd, post_gd
        return g

    def update(self, f):
        fid, h, a = int(f.fixture_id), int(f.home_id), int(f.away_id)
        hg, ag = self.results.get(fid, (np.nan, np.nan))
        self.max_result_day = f.day if self.max_result_day is None else max(self.max_result_day, f.day)
        if pd.isna(hg) or pd.isna(ag):
            return
        hg, ag = float(hg), float(ag)
        m = self.mkp.get(fid)
        obs = {}
        for t, o, gf, ga in ((h, a, hg, ag), (a, h, ag, hg)):
            x = self.team_obs(fid, t, o, gf, ga)
            obs[t] = x
            self.team[t].add(x, self.A_TEAM)
            gobs = self.game_obs(fid, t, o, h)
            if gobs is not None:
                self.game_team[t] = self.game_team[t] * self.A_GAME + gobs
                self.game_prior = self.game_prior * 0.999 + gobs
                c = self.coach.get((fid, t))
                if c is not None:
                    self.game_coach[c] = self.game_coach[c] * self.A_GAME + gobs
            sl = self.pidx.get((fid, t))
            if sl:
                rows, pids = self.pmat[sl[0]:sl[1]], self.ppid[sl[0]:sl[1]]
                rs = []
                for pid, row in zip(pids, rows):
                    mins = row[PI["minutes"]]
                    if not np.isfinite(mins) or mins <= 0:
                        continue
                    role = self.player_role(fid, pid)
                    self.prole[pid][role] += 1
                    r = np.nan_to_num(row)
                    if not np.isfinite(row[PI["rating"]]):
                        r[PI["rating"]] = 0
                    # rating is a level, not a count: store rating x minutes/90 so per-90 maths returns the average
                    r[PI["rating"]] = (row[PI["rating"]] if np.isfinite(row[PI["rating"]]) else 0) * mins / 90
                    self.ps[pid] = self.ps.get(pid, np.zeros(len(PA))) * self.P_DECAY + r
                    self.rprior[role] = self.rprior[role] * 0.9995 + r
                    if pid in self.xi.get((fid, t), []) and np.isfinite(row[PI["rating"]]):
                        rs.append(row[PI["rating"]])
                if rs:
                    self.team_xi_rating[t].add(np.array([np.mean(rs)]), self.A_TEAM)
            fm = self.form.get((fid, t))
            if fm and np.isfinite(fm[0]):
                self.team_ndef[t].add(np.array([fm[0]]), self.A_TEAM)
        if m:
            ref = self.league_ref[f.league_id].get()
            for t, o, exp, pts in ((h, a, 3 * m[0] + m[1], 3 if hg > ag else (1 if hg == ag else 0)),
                                   (a, h, 3 * m[2] + m[1], 3 if ag > hg else (1 if hg == ag else 0))):
                opp = dict(zip(TEAM_M, self.team[o].get()))
                b = self.bucket(opp, ref)
                if b is not None:
                    self.memory[t][b] = self.memory[t][b] * 0.97 + np.array([pts - exp, 1.0])
        for t in (h, a):
            v = self.team[t].get()
            self.league_ref[f.league_id].add(np.array([v[TI["poss"]], v[TI["press"]]]), 0.02)
            m_, s_ = self.lz_team[f.league_id]
            m_.add(v, 0.01); s_.add(v * v, 0.01)
            xv = self.emitted_xi.pop((fid, t), None)
            if xv is not None:
                m_, s_ = self.lz_xi[f.league_id]
                m_.add(xv, 0.01); s_.add(xv * xv, 0.01)

    def run(self, cutoff=None, only_day=None):
        """Same contract as backtest_models.Engine.run (used by the time-travel test)."""
        self.reset_state()
        saved = None
        if cutoff is not None:
            saved = (self.results, self.tstat, self.xg, self.pidx, self.events)
            late = set(self.fx.fixture_id[self.fx.day >= cutoff].astype(int))
            self.results = {k: (v if k not in late else (np.nan, np.nan)) for k, v in self.results.items()}
            self.tstat = {k: v for k, v in self.tstat.items() if k[0] not in late}
            self.xg = {k: v for k, v in self.xg.items() if k[0] not in late}
            self.pidx = {k: v for k, v in self.pidx.items() if k[0] not in late}
            self.events = defaultdict(list, {k: v for k, v in self.events.items() if k not in late})
        out = []
        days = self.fx.groupby("day", sort=True)
        track = bt.PROGRESS is not None and only_day is None and cutoff is None
        if track:
            bt.PROGRESS.start("tactical features", len(days))
        for day, g in days:
            if only_day is not None and day > only_day:
                break
            rows = [self.emit(f) for f in g.itertuples()]
            if only_day is None or day == only_day:
                out.extend(rows)
            for f in g.itertuples():
                self.update(f)
            if track:
                bt.progress(1, str(day.date()))
        if saved:
            self.results, self.tstat, self.xg, self.pidx, self.events = saved
        return pd.DataFrame(out)


def minute_travel_test(eng: TacticalEngine, n=300, seed=7) -> dict:
    """In-play features at minute c must not change when every event at or after c is deleted."""
    rng = np.random.default_rng(seed)
    fids = [f for f in eng.events if eng.events[f]]
    if not fids:
        return {"skipped": True, "passed": True, "checked": 0, "mismatches": 0}
    fx = eng.fx.set_index("fixture_id")
    bad = checked = 0
    for fid in rng.choice(fids, size=min(n, len(fids)), replace=False):
        if fid not in fx.index:
            continue
        h, a = int(fx.loc[fid, "home_id"]), int(fx.loc[fid, "away_id"])
        for c in CHECKPOINTS:
            full = eng.inplay_state(fid, h, a, c)
            cut = eng.inplay_state(fid, h, a, c, events=[e for e in eng.events[fid] if e[0] < c])
            for k, v in full.items():
                w = cut[k]
                checked += 1
                if not ((pd.isna(v) and pd.isna(w)) or v == w):
                    bad += 1
    log(f"  minute-travel test: {checked:,} in-play values checked, mismatches = {bad}")
    return {"skipped": False, "passed": bad == 0, "checked": checked, "mismatches": bad}


# =============================================================================
# Feature families
# =============================================================================
def families(cols) -> dict:
    fam = defaultdict(list)
    for c in cols:
        if re.match(r"^[ha]_st_", c):
            fam["Team style"].append(c)
        elif re.match(r"^[ha]_q_", c):
            fam["Team execution"].append(c)
        elif re.match(r"^[ha]_tx_", c):
            fam["Starting XI technique"].append(c)
        elif re.match(r"^(mh|ma|md)_", c) or c in ("poss_clash", "style_distance"):
            fam["Matchups"].append(c)
        elif re.match(r"^[ha]_(g|c)_", c):
            fam["In-game habits"].append(c)
        elif re.match(r"^[ha]_f_", c):
            fam["Formation"].append(c)
        elif re.match(r"^[ha]_memory$", c):
            fam["Style memory"].append(c)
        else:
            fam["Base football data"].append(c)
    return dict(fam)


# =============================================================================
# Walk-forward training (chunks run in parallel, like backtest_models.py)
# =============================================================================
_TW: dict = {}


def _tinit(df, feats, args, threads, queue=None):
    _TW.update(df=df, feats=feats, args=args, limiter=bt.set_thread_limit(threads), queue=queue)


def _treport(work, label):
    q = _TW.get("queue")
    if q is not None:
        q.put((work, label))
    else:
        bt.progress(work, label)


def _hgb_extra(frac):
    """Feature subsampling per split (scikit-learn >= 1.4); silently skipped on older versions."""
    import inspect
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {"max_features": frac} if "max_features" in inspect.signature(HistGradientBoostingRegressor).parameters else {}


def nonempty(Xa, *others):
    """Drop features with no data in this training window (scikit-learn cannot bin an all-missing column)."""
    keep = np.isfinite(Xa).any(axis=0)
    return (Xa[:, keep],) + tuple(o[:, keep] for o in others)


def gbr(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # early stopping on an internal slice of the TRAINING rows only; coarse bins for speed
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=150, max_bins=63,
                                         l2_regularization=5.0, **_hgb_extra(0.4), early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=20, random_state=seed)


def boost_resid(Xa, Ya, Ma, Xb, seed=0):
    """Model the market's errors. For 1X2 the home and away errors are modelled and the draw
    error is their negative sum (the three always add up to zero)."""
    keep = np.isfinite(Xa).any(axis=0)
    Xa, Xb = Xa[:, keep], Xb[:, keep]
    out = np.zeros((len(Xb), Ma.shape[1]))
    cols = [0, 2] if Ma.shape[1] == 3 else [0]
    models = {}
    for j in cols:
        reg = gbr(seed + j)
        reg.fit(Xa, Ya[:, j] - Ma[:, j])
        out[:, j] = reg.predict(Xb)
        models[j] = reg
    if Ma.shape[1] == 3:
        out[:, 1] = -(out[:, 0] + out[:, 2])
    return out, (keep, models)


def combine(M, G, s):
    if M.shape[1] == 1:
        return np.clip(M[:, 0] + s * G[:, 0], 0.01, 0.99)
    Pn = np.clip(M + s * G, 0.005, 1)
    return Pn / Pn.sum(axis=1, keepdims=True)


def tune_shrink(X, Y, M):
    cut = int(len(X) * 0.8)
    Gv, _ = boost_resid(X[:cut], Y[:cut], M[:cut], X[cut:])
    yv = Y[cut:]
    best, best_ll = 0.0, np.inf
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        pv = combine(M[cut:], Gv, s)
        ll = -(yv[:, 0] * np.log(pv) + (1 - yv[:, 0]) * np.log(1 - pv)).mean() if M.shape[1] == 1 else -np.log((pv * yv).sum(1)).mean()
        if ll < best_ll - 1e-6:
            best, best_ll = s, ll
    return best


PRE_MODELS = {   # name -> (feature set, kind, target)
    "base_gbm": ("base", "resid", "1x2"), "tactical_gbm": ("tactical", "resid", "1x2"), "combined_gbm": ("combined", "resid", "1x2"),
    "tactical_standalone": ("standalone", "clf", "1x2"),
    "base_gbm_ou": ("base", "resid", "ou"), "combined_gbm_ou": ("combined", "resid", "ou"),
}


def _tchunk(blocks):
    from sklearn.ensemble import HistGradientBoostingClassifier
    df, feats, args = _TW["df"], _TW["feats"], _TW["args"]
    t0 = time.time()
    out = {"P": [], "Q": [], "shrink": []}
    shrink = {}
    y_all = df.y.to_numpy()
    has_mk = df.mk_lr_h.notna().to_numpy()
    for k, block in enumerate(blocks):
        m0 = block[0].start_time
        train = ((df.day < m0) & (df.day >= m0 - pd.Timedelta(days=args.train_window_days))).to_numpy() & has_mk
        test = df.month.isin(block).to_numpy() & has_mk
        if train.sum() < args.min_train or test.sum() == 0:
            _treport(float(train.sum()), str(block[0]))
            continue
        tr, te = df[train], df[test]
        for name, (fset, kind, target) in PRE_MODELS.items():
            cols = feats[fset]
            if target == "ou" and ("ou_p_o" not in df or df.ou_p_o.notna().mean() < 0.3):
                continue
            Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
            if kind == "clf":
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=120, max_bins=63,
                                                     l2_regularization=2.0, **_hgb_extra(0.4), early_stopping=True, validation_fraction=0.15,
                                                     n_iter_no_change=20, random_state=0)
                Xa, Xb = nonempty(Xtr, Xte)
                clf.fit(Xa, y_all[train])
                pr = clf.predict_proba(Xb)
                full = np.zeros((len(Xte), 3))
                full[:, clf.classes_.astype(int)] = pr
                out["P"].append((name, np.where(test)[0], full))
                continue
            pcols = ["p_h", "p_d", "p_a"] if target == "1x2" else ["ou_p_o"]
            ok_tr, ok_te = tr[pcols].notna().all(axis=1).to_numpy(), te[pcols].notna().all(axis=1).to_numpy()
            if ok_tr.sum() < args.min_train or ok_te.sum() == 0:
                continue
            Mtr, Mte = tr.loc[ok_tr, pcols].to_numpy(float), te.loc[ok_te, pcols].to_numpy(float)
            Ytr = np.eye(3)[y_all[train][ok_tr]] if target == "1x2" else tr.y_over.to_numpy(float)[ok_tr][:, None]
            if name not in shrink or k == 0:
                shrink[name] = tune_shrink(Xtr[ok_tr], Ytr, Mtr)
                out["shrink"].append((name, str(block[0]), shrink[name]))
            G, _ = boost_resid(Xtr[ok_tr], Ytr, Mtr, Xte[ok_te])
            pred = combine(Mte, G, shrink[name])
            out["P" if target == "1x2" else "Q"].append((name, np.where(test)[0][ok_te], pred))
        _treport(float(train.sum()), str(block[0]))
    out["seconds"], out["span"] = time.time() - t0, f"{blocks[0][0]}..{blocks[-1][-1]}"
    return out


def run_chunks(fn, init_args, chunks, plan, stage, work_total):
    import concurrent.futures as cf
    import multiprocessing as mp
    import queue as qm
    if bt.PROGRESS:
        bt.PROGRESS.start(stage, work_total)
    results = []
    if plan["workers"] <= 1:
        _tinit(*init_args, plan["threads"])
        for ch in chunks:
            results.append(fn(ch))
        _TW.clear()
        return results
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    q = ctx.Queue()
    with cf.ProcessPoolExecutor(max_workers=plan["workers"], mp_context=ctx, initializer=_tinit,
                                initargs=(*init_args, plan["threads"], q)) as ex:
        pending = {ex.submit(fn, chunks[i]) for i in sorted(range(len(chunks)), key=lambda i: -i)}
        while pending:
            done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            while True:
                try:
                    w, lab = q.get_nowait()
                except qm.Empty:
                    break
                bt.progress(w, lab)
            for f in done:
                r = f.result()
                results.append(r)
                log(f"  {stage}: chunk {r['span']} in {r['seconds']:.0f}s")
    return results


def blocks_for(df, args):
    months = sorted(df.month.unique())
    start = df.day.min() + pd.Timedelta(days=args.burn_in_days)
    em = [m for m in months if m.start_time >= start]
    blocks = [em[i: i + args.retrain_months] for i in range(0, len(em), args.retrain_months)]
    return blocks, [blocks[i: i + args.retune_every] for i in range(0, len(blocks), args.retune_every)]


def work_of(df, blocks, args, mask=None):
    days = df.day.to_numpy()
    ok = df.mk_lr_h.notna().to_numpy() if mask is None else mask
    tw = np.timedelta64(args.train_window_days, "D")
    return sum(float(((days < np.datetime64(b[0].start_time)) & (days >= np.datetime64(b[0].start_time) - tw) & ok).sum()) for b in blocks)


def pre_match_walk_forward(df, feats, args):
    n = len(df)
    P = {m: np.full((n, 3), np.nan) for m, v in PRE_MODELS.items() if v[2] == "1x2"}
    Q = {m: np.full(n, np.nan) for m, v in PRE_MODELS.items() if v[2] == "ou"}
    blocks, chunks = blocks_for(df, args)
    plan = bt.compute_plan(args, len(chunks), float(df.memory_usage(deep=True).sum()))
    log(f"compute: {plan['logical']} CPUs, {plan['mem_gb']} GB free -> {plan['workers']} worker(s) x {plan['threads']} thread(s), "
        f"{len(blocks)} retrains in {len(chunks)} chunks")
    t0 = time.time()
    res = run_chunks(_tchunk, (df, feats, args), chunks, plan, "training models", work_of(df, blocks, args))
    shrink = defaultdict(list)
    for r in sorted(res, key=lambda r: r["span"]):
        for name, idx, pred in r["P"]:
            P[name][idx] = pred
        for name, idx, pred in r["Q"]:
            Q[name][idx] = pred
        for name, mth, s in r["shrink"]:
            shrink[name].append((mth, s))
    log(f"pre-match models trained in {time.time() - t0:.0f}s")
    return P, Q, dict(shrink), plan


# ---------------------------------------------------------------- in-play
IP_STATE = ["mk_lr_h", "mk_lr_a", "gd", "frac", "gd_frac", "gd_frac2", "tot", "red_diff", "minute"]


def _ipchunk(blocks):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ip, feats, args = _TW["df"], _TW["feats"], _TW["args"]
    t0 = time.time()
    out = {"P": []}
    y = ip.y.to_numpy()
    ok_all = ip.ip_ok.to_numpy()
    for block in blocks:
        m0 = block[0].start_time
        train = ((ip.day < m0) & (ip.day >= m0 - pd.Timedelta(days=args.train_window_days))).to_numpy() & ok_all
        test = ip.month.isin(block).to_numpy() & ok_all
        if train.sum() < args.min_train * 2 or test.sum() == 0:
            _treport(float(train.sum()), str(block[0]))
            continue
        tr, te = ip[train], ip[test]
        idx = np.where(test)[0]
        prep = bt.Prep(tr[IP_STATE])
        wf = bt.fit_mnl(prep(tr[IP_STATE]), y[train], 3, 1e-4, np.ones(len(IP_STATE)))
        Ptr, Pte = bt.mnl_predict(wf, prep(tr[IP_STATE]), 3), bt.mnl_predict(wf, prep(te[IP_STATE]), 3)
        out["P"].append(("ip_state", idx, Pte))
        days_tr = tr.day.to_numpy()
        Xa, Xb = nonempty(tr[feats["ip_state"]].to_numpy(float), te[feats["ip_state"]].to_numpy(float))
        Gte, Gtr = fit_grouped_clf(Xa, y[train], days_tr, Xb, return_train=True)
        out["P"].append(("ip_state_gbm", idx, Gte))
        # tactical in-play = boosted scoreboard model + boosted corrections from what the teams have done and changed
        Xa, Xb = nonempty(tr[feats["ip_tactical"]].to_numpy(float), te[feats["ip_tactical"]].to_numpy(float))
        out["P"].append(("ip_tactical", idx, fit_grouped_resid(Xa, np.eye(3)[y[train]], Gtr, days_tr, Xb, Gte)))
        _treport(float(train.sum()), str(block[0]))
    out["seconds"], out["span"] = time.time() - t0, f"{blocks[0][0]}..{blocks[-1][-1]}"
    return out


def fit_grouped_clf(X, y, days, Xtest, return_train=False):
    """In-play rows come 5 per match with the same final result, so a random validation slice would
    contain other checkpoints of the same matches and never stop the model memorising them. Instead the
    number of trees is chosen on the LATEST 15% of training dates (whole matches), then the model is
    refit on everything with that many trees."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=3, min_samples_leaf=300, max_bins=63,  # noqa: E731
                                                  l2_regularization=5.0, early_stopping=False, random_state=0)
    cut = np.quantile(days, 0.85)
    fit_m, val_m = days < cut, days >= cut
    best_n = 60
    if fit_m.sum() > 1000 and val_m.sum() > 200 and len(np.unique(y[fit_m])) == 3:
        clf = mk(400).fit(X[fit_m], y[fit_m])
        yv = y[val_m]
        losses = [-np.log(np.clip(p[np.arange(len(yv)), yv], 1e-9, 1)).mean() for p in clf.staged_predict_proba(X[val_m])]
        best_n = int(np.argmin(losses)) + 1
    clf = mk(max(best_n, 10)).fit(X, y)
    out = []
    for Z in ((Xtest, X) if return_train else (Xtest,)):
        full = np.zeros((len(Z), 3))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(Z)
        out.append(full)
    return tuple(out) if return_train else out[0]


def fit_grouped_resid(X, Y, M, days, Xt, Mt):
    """Corrections to a base in-play model (M), learned from tactical inputs. Trees and the share of the
    correction to keep are both chosen on the LATEST 15% of training dates (whole matches)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    mk = lambda n, seed: HistGradientBoostingRegressor(max_iter=n, learning_rate=0.05, max_depth=3, min_samples_leaf=300,  # noqa: E731
                                                       max_bins=63, l2_regularization=5.0, early_stopping=False, random_state=seed)
    cut = np.quantile(days, 0.85)
    fm, vm = days < cut, days >= cut
    best_n, best_s = {0: 50, 2: 50}, 0.0
    if fm.sum() > 1000 and vm.sum() > 200:
        Gv = np.zeros((vm.sum(), 3))
        for j in (0, 2):
            reg = mk(300, j).fit(X[fm], Y[fm, j] - M[fm, j])
            stages = list(reg.staged_predict(X[vm]))
            err = [((Y[vm, j] - M[vm, j] - p) ** 2).mean() for p in stages]
            best_n[j] = int(np.argmin(err)) + 1
            Gv[:, j] = stages[best_n[j] - 1]
        Gv[:, 1] = -(Gv[:, 0] + Gv[:, 2])
        yv = Y[vm]
        best_ll = -np.log((M[vm] * yv).sum(1)).mean()
        for s in (0.25, 0.5, 0.75, 1.0):
            ll = -np.log((combine(M[vm], Gv, s) * yv).sum(1)).mean()
            if ll < best_ll - 1e-6:
                best_s, best_ll = s, ll
    if best_s == 0:
        return Mt
    G = np.zeros((len(Xt), 3))
    for j in (0, 2):
        G[:, j] = mk(max(best_n[j], 5), j).fit(X, Y[:, j] - M[:, j]).predict(Xt)
    G[:, 1] = -(G[:, 0] + G[:, 2])
    return combine(Mt, G, best_s)


def inplay_walk_forward(ip, feats, args):
    n = len(ip)
    P = {m: np.full((n, 3), np.nan) for m in ("ip_state", "ip_state_gbm", "ip_tactical")}
    blocks, chunks = blocks_for(ip, args)
    plan = bt.compute_plan(args, len(chunks), float(ip.memory_usage(deep=True).sum()))
    t0 = time.time()
    res = run_chunks(_ipchunk, (ip, feats, args), chunks, plan, "in-play models", work_of(ip, blocks, args, ip.ip_ok.to_numpy()))
    for r in res:
        for name, idx, pred in r["P"]:
            P[name][idx] = pred
    log(f"in-play models trained in {time.time() - t0:.0f}s")
    return P


# =============================================================================
# Analysis
# =============================================================================
def ll_rows(P3, y):
    return -np.log(np.clip(P3[np.arange(len(y)), y], 1e-9, 1))


def paired_gain(a_ll, b_ll, week, B=1000, seed=4):
    """Mean of (b - a) log loss per match (positive = a better), weekly block bootstrap."""
    d = pd.DataFrame({"x": b_ll - a_ll, "w": week}).dropna().groupby("w").x.agg(["sum", "size"])
    if len(d) < 5:
        return {"gain": None}
    s, c = d["sum"].to_numpy(), d["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    est = s.sum() / c.sum()
    return {"gain": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B)), "n": int(c.sum())}


def value_added(df, P, models):
    E, hold = df.eval_row.to_numpy(), (df.day >= df.holdout_from).to_numpy()
    y = df.y.to_numpy()
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    cup = (df.league_type != "league").to_numpy()
    mk = df[["p_h", "p_d", "p_a"]].to_numpy(float)
    pairs = [("combined_gbm", "base_gbm", "What tactics add on top of the base model"),
             ("tactical_gbm", "market", "Tactics alone vs the closing market"),
             ("combined_gbm", "market", "Base + tactics vs the closing market"),
             ("base_gbm", "market", "Base model vs the closing market")]
    out = []
    for a, b, label in pairs:
        if a not in models or b not in models:
            continue
        pa = P[a] if a != "market" else mk
        pb = P[b] if b != "market" else mk
        ok = E & ~np.isnan(pa).any(1) & ~np.isnan(pb).any(1)
        rec = {"a": a, "b": b, "label": label, "periods": {}}
        for pn, pm in (("discovery", ~hold), ("holdout", hold), ("all", np.ones(len(df), bool))):
            for seg, sm in (("all", np.ones(len(df), bool)), ("league", ~cup), ("cup", cup)):
                m = ok & pm & sm
                if m.sum() < 50:
                    continue
                rec["periods"].setdefault(pn, {})[seg] = paired_gain(ll_rows(pa[m], y[m]), ll_rows(pb[m], y[m]), week[m])
        out.append(rec)
    return out


def permutation_families(df, feats, args, fams):
    """Train a combined residual model on discovery, measure hold-out log-loss increase when each
    feature family (and each top feature) is shuffled. Trained on the past, scored on the future."""
    E = df.eval_row.to_numpy()
    hold = E & (df.day >= df.holdout_from).to_numpy()
    disc = df.mk_lr_h.notna().to_numpy() & (df.day < df.holdout_from).to_numpy() & (df.day >= df.holdout_from - pd.Timedelta(days=args.train_window_days)).to_numpy()
    if disc.sum() < args.min_train or hold.sum() < 200:
        return {}
    cols = feats["combined"]
    Xd, Xh = df.loc[disc, cols].to_numpy(float), df.loc[hold, cols].to_numpy(float)
    Md, Mh = df.loc[disc, ["p_h", "p_d", "p_a"]].to_numpy(float), df.loc[hold, ["p_h", "p_d", "p_a"]].to_numpy(float)
    yd, yh = df.y.to_numpy()[disc], df.y.to_numpy()[hold]
    Yd = np.eye(3)[yd]
    s = max(tune_shrink(Xd, Yd, Md), 0.5)
    G, (keep, models) = boost_resid(Xd, Yd, Md, Xh)

    def score(X):
        Xk = X[:, keep]
        Gx = np.zeros((len(X), 3))
        for j, m in models.items():
            Gx[:, j] = m.predict(Xk)
        Gx[:, 1] = -(Gx[:, 0] + Gx[:, 2])
        return ll_rows(combine(Mh, Gx, s), yh).mean()
    base = score(Xh)
    rng = np.random.default_rng(1)
    res = {"base_ll": float(base), "market_ll": float(ll_rows(Mh, yh).mean()), "shrink": s, "families": {}, "features": {}}
    ix = {c: i for i, c in enumerate(cols)}
    for fam, fc in fams.items():
        fc = [c for c in fc if c in ix]
        if not fc:
            continue
        X = Xh.copy()
        perm = rng.permutation(len(X))
        for c in fc:
            X[:, ix[c]] = X[perm, ix[c]]
        res["families"][fam] = {"increase": float(score(X) - base), "n_features": len(fc)}
    for c in cols:
        X = Xh.copy()
        X[:, ix[c]] = X[rng.permutation(len(X)), ix[c]]
        res["features"][c] = float(score(X) - base)
    res["features"] = dict(sorted(res["features"].items(), key=lambda kv: -kv[1])[:30])
    return res


def matchup_quintiles(df, cols):
    """Market residual (home points minus market-expected points) by quintile of each matchup,
    with cut points from the discovery period only."""
    E = df.eval_row.to_numpy()
    hold = (df.day >= df.holdout_from).to_numpy()
    pts = np.select([df.hg > df.ag, df.hg == df.ag], [3, 1], 0)
    resid = pts - (3 * df.p_h + df.p_d).to_numpy()
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    out = {}
    for c in cols:
        x = df[c].to_numpy(float)
        ok = E & np.isfinite(x) & np.isfinite(resid)
        dmask = ok & ~hold
        if dmask.sum() < 200:
            continue
        cuts = np.quantile(x[dmask], [0.2, 0.4, 0.6, 0.8])
        q = np.searchsorted(cuts, x)
        rec = []
        for k in range(5):
            row = {"q": k + 1}
            for pn, pm in (("discovery", ~hold), ("holdout", hold)):
                m = ok & pm & (q == k)
                lo, hi, p = bt.boot_roi(resid[m], week[m]) if m.sum() >= 20 else (None, None, None)
                row[pn] = {"n": int(m.sum()), "resid": float(resid[m].mean()) if m.sum() else None, "lo": lo, "hi": hi}
            rec.append(row)
        top, bot = rec[4]["holdout"], rec[0]["holdout"]
        out[c] = {"quintiles": rec, "cuts": cuts.tolist(),
                  "spread_hold": (top["resid"] - bot["resid"]) if top["resid"] is not None and bot["resid"] is not None else None,
                  "spread_disc": (rec[4]["discovery"]["resid"] or 0) - (rec[0]["discovery"]["resid"] or 0)}
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["spread_hold"] or 0)))


def inplay_analysis(ip, P):
    E = ip.ip_eval.to_numpy()
    hold = (ip.day >= ip.holdout_from).to_numpy()
    y = ip.y.to_numpy()
    week = ip.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * ip.day.dt.year.to_numpy()
    res = {"checkpoints": {}, "gain": {}, "tweaks": []}
    have = E & ~np.isnan(P["ip_state"]).any(1) & ~np.isnan(P["ip_tactical"]).any(1) & ~np.isnan(P["ip_state_gbm"]).any(1)
    for c in CHECKPOINTS:
        m = have & (ip.minute == c).to_numpy()
        row = {}
        for pn, pm in (("discovery", ~hold), ("holdout", hold)):
            mm = m & pm
            if mm.sum() < 50:
                continue
            row[pn] = {name: float(ll_rows(P[name][mm], y[mm]).mean()) for name in P}
            row[pn]["n"] = int(mm.sum())
            row[pn]["gain"] = paired_gain(ll_rows(P["ip_tactical"][mm], y[mm]), ll_rows(P["ip_state_gbm"][mm], y[mm]), week[mm])
        res["checkpoints"][str(c)] = row
    for pn, pm in (("discovery", ~hold), ("holdout", hold)):
        mm = have & pm
        if mm.sum() > 100:
            res["gain"][pn] = paired_gain(ll_rows(P["ip_tactical"][mm], y[mm]), ll_rows(P["ip_state_gbm"][mm], y[mm]), week[mm])
    # tweak effects: points of the chasing side vs what the scoreboard model expected at that minute
    base = P["ip_state"]
    for c in (60, 75):
        for side in ("h", "a"):
            chasing = (ip.gd < 0) if side == "h" else (ip.gd > 0)
            m = have & (ip.minute == c).to_numpy() & chasing.to_numpy()
            if m.sum() < 50:
                continue
            pts = np.where(y == (0 if side == "h" else 2), 3, np.where(y == 1, 1, 0))
            exp = 3 * base[:, 0 if side == "h" else 2] + base[:, 1]
            resid = pts - exp
            for label, cond in (("made an attacking change", ip[f"{side}_chasing_aggr"].to_numpy() > 0),
                                ("no attacking change yet", ip[f"{side}_chasing_aggr"].to_numpy() <= 0),
                                ("coach habitually attacking (top third)", ip[f"{side}_c_chase_aggr"].to_numpy() >= np.nanquantile(ip[f"{side}_c_chase_aggr"], 0.67) if f"{side}_c_chase_aggr" in ip and ip[f"{side}_c_chase_aggr"].notna().any() else np.zeros(len(ip), bool))):
                mm = m & cond
                if mm.sum() < 30:
                    continue
                lo, hi, p = bt.boot_roi(resid[mm], week[mm])
                res["tweaks"].append({"minute": c, "side": "home" if side == "h" else "away", "group": label, "n": int(mm.sum()),
                                      "resid": float(resid[mm].mean()), "lo": lo, "hi": hi, "p": p})
    return res


# =============================================================================
# Main
# =============================================================================
class TProgress(bt.Progress):
    WEIGHTS = {"reading database": 3, "reading tactical data": 4, "building features": 3, "tactical features": 10,
               "time-travel test": 6, "training models": 52, "in-play models": 17, "scoring": 5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--out", default="tactical_dashboard.html")
    ap.add_argument("--export-csv")
    ap.add_argument("--title", default="Tactical & technical model vs the market")
    ap.add_argument("--leagues")
    ap.add_argument("--holdout-from")
    ap.add_argument("--burn-in-days", type=int, default=365)
    ap.add_argument("--train-window-days", type=int, default=1460)
    ap.add_argument("--retrain-months", type=int, default=3, help="months between retrains (tactical models are heavier: default 3)")
    ap.add_argument("--retune-every", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=1500)
    ap.add_argument("--half-life-days", type=float, default=180)
    ap.add_argument("--market-1x2", type=int, default=1)
    ap.add_argument("--market-ou", type=int, default=80)
    ap.add_argument("--sharp-bookmaker", default="Pinnacle")
    ap.add_argument("--max-odds", type=float, default=10.0)
    ap.add_argument("--min-bets", type=int, default=150)
    ap.add_argument("--leak-tests", type=int, default=3)
    ap.add_argument("--skip-leak-test", action="store_true")
    ap.add_argument("--alarm-gain", type=float, default=0.006)
    ap.add_argument("--no-inplay", action="store_true", help="skip the in-play models")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--db-jobs", type=int, default=0)
    ap.add_argument("--progress-every", type=float, default=30)
    ap.add_argument("--cache")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    args.no_gbm = False

    bt.PROGRESS = TProgress(args.progress_every, skip=(("time-travel test",) if args.skip_leak_test else ()) + (("in-play models",) if args.no_inplay else ()))
    db = bt.DB(args.dsn)
    args.db_name = db.name
    d = bt.extract(db, args)
    td = tactical_extract(db, args)
    mk = bt.market_table(d, args)
    if len(d["predictions"]):
        mk = mk.join(d["predictions"].set_index("fixture_id"), how="left")
    log("base features ...")
    beng = bt.Engine(d, mk, args)
    base = beng.run()
    log("tactical features ...")
    teng = TacticalEngine(d, td, mk, args)
    tac = teng.run()
    inplay_raw = pd.DataFrame(teng.inplay)
    log(f"tactical rows: {len(tac):,} x {tac.shape[1]} · in-play rows: {len(inplay_raw):,}")
    leak = {"skipped": True}
    if not args.skip_leak_test:
        log("time-travel test (base features) ...")
        leak = bt.leak_test(beng, base, args)
        log("time-travel test (tactical features) ...")
        lt = bt.leak_test(teng, tac, args)
        leak["tests"] += [dict(t, engine="tactical") for t in lt["tests"]]
        leak["passed"] = leak["passed"] and lt["passed"]
        leak["skipped"] = False
    mt = minute_travel_test(teng)
    df = base.merge(tac.drop(columns=["day"]), on="fixture_id", how="left").merge(d["fixtures"][["fixture_id", "hg", "ag"]], on="fixture_id")
    df = bt.add_derived(df, mk).sort_values(["day", "fixture_id"]).reset_index(drop=True)
    tac_all = [c for c in tac.columns if c not in ("fixture_id", "day")]
    bfs = bt.feature_sets(df.drop(columns=tac_all))          # base feature list must not include tactical columns
    tac_cols = [c for c in tac_all if df[c].notna().mean() > 0.05 and df[c].nunique() > 1]
    strength = [c for c in ("elo_diff", "pois_p_h", "pois_p_a", "xpois_p_h", "xpois_p_a", "h_xgf", "a_xgf", "h_xga", "a_xga", "lg_home") if c in df]
    mkt = [c for c in ("mk_lr_h", "mk_lr_a") if c in df]
    feats = {"base": list(dict.fromkeys(bfs["gbm"])), "tactical": mkt + tac_cols, "combined": list(dict.fromkeys(bfs["gbm"] + tac_cols)),
             "standalone": strength + tac_cols}
    fams = families([c for c in feats["combined"] if c not in mkt])
    log(f"features: base {len(feats['base'])}, tactical {len(tac_cols)}, combined {len(feats['combined'])}")
    P, Q, shrink, plan = pre_match_walk_forward(df, feats, args)
    models_1x2 = ["market"] + [m for m in P if not np.isnan(P[m]).all()]
    models_ou = (["market"] + [m for m in Q if not np.isnan(Q[m]).all()]) if "ou_p_o" in df and df.ou_p_o.notna().mean() > .3 else []
    have = ~np.isnan(P["base_gbm"]).any(1) & ~np.isnan(P["combined_gbm"]).any(1) & df[["p_h", "p_d", "p_a"]].notna().all(axis=1).to_numpy()
    if args.leagues:
        have &= df.league_id.isin([int(x) for x in args.leagues.split(",")]).to_numpy()
    df["eval_row"] = have
    if have.sum() == 0:
        sys.exit("No rows could be evaluated - not enough history or odds.")
    hf = pd.Timestamp(args.holdout_from) if args.holdout_from else df.day[have].quantile(0.6).normalize()
    df["holdout_from"] = hf
    bt.PROGRESS.start("scoring", 6)
    evals, per_row = bt.evaluate(df, P, Q, args, models_1x2, models_ou)
    bt.progress(1, "models scored")
    alarms = []
    for m, r in list(evals["models"].items()) + [("ou:" + k, v) for k, v in evals.get("ou", {}).items()]:
        mtx = r["metrics"].get("holdout", {})
        if mtx.get("logloss") is not None and mtx.get("mk_logloss") is not None and mtx["mk_logloss"] - mtx["logloss"] > args.alarm_gain:
            alarms.append({"model": m, "gain": mtx["mk_logloss"] - mtx["logloss"]})
            log(f"!!! WARNING: {m} beats the closing market by {(mtx['mk_logloss'] - mtx['logloss']) * 1000:.1f}e-3 log loss - implausibly large, check for leakage")
    leak["alarms"] = alarms
    va = value_added(df, P, models_1x2)
    bt.progress(1, "value added")
    imp = permutation_families(df, feats, args, fams)
    bt.progress(1, "importance")
    mcols = [c for c in df.columns if c.startswith("md_")] + [c for c in ("poss_clash", "style_distance", "h_memory", "a_memory") if c in df]
    mq = matchup_quintiles(df, mcols)
    bt.progress(1, "matchups")
    # ---- in-play
    ipres, ip_cov = {}, {}
    if not args.no_inplay and len(inplay_raw):
        pre = df[["fixture_id", "day", "month", "league", "league_type", "y", "mk_lr_h", "mk_lr_a", "eval_row", "holdout_from"]].copy()
        for i, s in enumerate("hda"):
            pre[f"pre_combined_{s}"] = P["combined_gbm"][:, i]
        prof_cols = [c for c in df.columns if re.match(r"^[ha]_(g|c)_", c)] + [c for c in df.columns if c.startswith("md_")] + \
                    [c for c in ("h_tx_bench", "a_tx_bench") if c in df]
        ip = inplay_raw.merge(pre.join(df[prof_cols]), on="fixture_id", how="inner")
        ip["gd_frac"], ip["gd_frac2"] = ip.gd * ip.frac, ip.gd * ip.frac ** 2
        ip["red_diff"] = ip.h_red - ip.a_red
        ip["ip_ok"] = ip.mk_lr_h.notna()
        ip["ip_eval"] = ip.eval_row & ip.ip_ok
        ip = ip.sort_values(["day", "fixture_id", "minute"]).reset_index(drop=True)
        # compact in-game set: changes so far, momentum, habits of both teams and coaches, bench, pre-match tactical forecast
        tweak_cols = [c for c in inplay_raw.columns if re.match(r"^(h|a|d)_(subs|last_sub|sub_aggr|chasing_aggr|red|mo15_\w+|pressure)$", c)] + ["since_goal"]
        habit_cols = [c for c in prof_cols if re.match(r"^[ha]_(c|g)_(chase_aggr|chase_rate|trail_attack|lead_defence|sub_impact|late_gd|comeback)$", c)] + \
                     [c for c in ("h_tx_bench", "a_tx_bench") if c in ip]
        feats_ip = {"ip_state": IP_STATE,
                    "ip_tactical": list(dict.fromkeys(IP_STATE + tweak_cols + habit_cols + [f"pre_combined_{s}" for s in "hda"]))}
        feats_ip["ip_tactical"] = [c for c in feats_ip["ip_tactical"] if c in ip and ip[c].notna().mean() > 0.02]
        Pip = inplay_walk_forward(ip, feats_ip, args)
        ipres = inplay_analysis(ip, Pip)
        ip_cov = {"rows": int(len(ip)), "fixtures": int(ip.fixture_id.nunique()), "features": len(feats_ip["ip_tactical"]),
                  "with_momentum": float(ip[[c for c in ip.columns if "_mo15_" in c]].notna().any(axis=1).mean()) if any("_mo15_" in c for c in ip.columns) else 0.0,
                  "with_pressure": float(ip.h_pressure.notna().mean()) if "h_pressure" in ip else 0.0}
    bt.progress(1, "in-play analysis")
    payload = build_payload(df, P, Q, evals, per_row, va, imp, mq, fams, ipres, ip_cov, teng, td, leak, mt, shrink, plan, args,
                            models_1x2, models_ou, feats, tac_cols)
    if args.export_csv:
        out = df.drop(columns=["month"]).copy()
        for m in P:
            for i, s in enumerate("hda"):
                out[f"pred_{m}_{s}"] = P[m][:, i]
        for m in Q:
            out[f"pred_{m}_over"] = Q[m]
        out.to_csv(args.export_csv, index=False)
        log(f"CSV written: {args.export_csv}")
    page = render(payload, args.title)
    Path(args.out).write_text(page, encoding="utf-8")
    bt.PROGRESS.close()
    log(f"dashboard written: {args.out} ({len(page) / 1e6:.1f} MB)")
    for v in va:
        g = v["periods"].get("holdout", {}).get("all", {})
        if g.get("gain") is not None:
            log(f"  {v['label']:46s} hold-out gain {g['gain'] * 1000:+.2f}e-3 [{g['lo'] * 1000:+.2f}, {g['hi'] * 1000:+.2f}] p={g['p']:.3f}")
    if ipres.get("gain", {}).get("holdout"):
        g = ipres["gain"]["holdout"]
        log(f"  in-play: tactical vs scoreboard model, hold-out gain {g['gain'] * 1000:+.2f}e-3 [{g['lo'] * 1000:+.2f}, {g['hi'] * 1000:+.2f}]")


# =============================================================================
# Dashboard payload & page
# =============================================================================
def build_payload(df, P, Q, evals, per_row, va, imp, mq, fams, ipres, ip_cov, teng, td, leak, mt, shrink, plan, args,
                  models_1x2, models_ou, feats, tac_cols):
    E = df.eval_row.to_numpy()
    d = df[E]
    cols = {"id": d.fixture_id.astype(int).tolist(), "date": d.day.dt.strftime("%Y-%m-%d").tolist(), "league": d.league.fillna("").tolist(),
            "home": d.home.fillna("").tolist(), "away": d.away.fillna("").tolist(), "hg": d.hg.astype(int).tolist(), "ag": d.ag.astype(int).tolist()}
    keep = ["avg_h", "avg_d", "avg_a", "best_h", "best_d", "best_a", "pin_h", "pin_d", "pin_a", "ou_p_o", "ou_avg_o", "ou_avg_u"]
    keep += [c for c in tac_cols if re.match(r"^(md_|poss_clash|style_distance|[ha]_memory|[ha]_st_|[ha]_tx_(rating_vs_usual|buildup|aer_def|threat|gk)|[ha]_g_(trail_attack|chase_aggr|sub_impact))", c)]
    for c in dict.fromkeys(keep):
        if c in d:
            cols[c] = bt.r3(d[c].to_numpy(float))
    probs = {m: [bt.r3(bt.model_probs(df, P, m)[E][:, i]) for i in range(3)] for m in models_1x2}
    ou = {m: bt.r3(bt.model_ou(df, Q, m)[E]) for m in models_ou}
    bets = {m: [int(x) for x in v[0][E]] for m, v in per_row.items()}
    edges = {m: bt.r3(v[1][E]) for m, v in per_row.items()}
    # latest team profiles (state at the end of the run) and players
    cn = td.get("t_coach_names", pd.DataFrame())
    cnames = dict(zip(cn.coach_id.astype("int64"), cn.name)) if len(cn) else {}
    pn = td.get("t_player_names", pd.DataFrame())
    pnames = dict(zip(pn.player_id.astype("int64"), pn.name)) if len(pn) else {}
    names = {}
    for r in df[["home_id", "home", "away_id", "away", "league"]].itertuples():
        names[int(r.home_id)] = (r.home, r.league)
        names[int(r.away_id)] = (r.away, r.league)
    teams = []
    for t, ew in teng.team.items():
        v = ew.get()
        gp = teng.game_profile(teng.game_team[t])
        rec = {"team": names.get(t, (str(t), ""))[0], "league": names.get(t, ("", ""))[1]}
        rec.update({k: (None if not np.isfinite(x) else round(float(x), 4)) for k, x in zip(TEAM_M, v)})
        rec.update({f"g_{k}": (None if not np.isfinite(x) else round(float(x), 4)) for k, x in gp.items()})
        teams.append(rec)
    players = []
    if teng.ps:
        mins = {p: s[PI["minutes"]] for p, s in teng.ps.items()}
        for p in sorted(mins, key=mins.get, reverse=True)[:600]:
            votes = teng.prole.get(p, {})
            role = max(votes, key=votes.get) if votes else "M"
            r = teng.player_rates(p, role)
            rec = {"id": int(p), "name": pnames.get(int(p)) or f"player {p}", "role": role, "minutes_w": round(float(mins[p]), 0)}
            for k in ("rating", "passes", "key_passes", "acc_crosses", "drib_succ", "tackles", "intercepts", "aer_won", "duels_won", "sot", "goals", "assists", "saves"):
                rec[k] = None if not np.isfinite(r[PI[k]]) else round(float(r[PI[k]]), 3)
            pa, pp = r[PI["acc_passes"]], r[PI["passes"]]
            rec["pass_acc"] = round(float(pa / pp), 3) if np.isfinite(pa) and np.isfinite(pp) and pp > 0 else None
            aw, al = r[PI["aer_won"]], r[PI["aer_lost"]]
            rec["aer_pct"] = round(float(aw / (aw + al)), 3) if np.isfinite(aw) and np.isfinite(al) and aw + al > 0 else None
            players.append(rec)
    coaches = []
    for c, sums in teng.game_coach.items():
        if sums[GI["games"]] >= 5:
            gp = teng.game_profile(sums)
            coaches.append({"coach": cnames.get(c) or f"coach {c}", "games_w": round(float(sums[GI["games"]]), 1), **{k: (None if not np.isfinite(x) else round(float(x), 3)) for k, x in gp.items()}})
    cov = {"eval": int(E.sum()), "from": str(d.day.min().date()), "to": str(d.day.max().date()), "leagues": int(d.league_id.nunique()),
           "n_discovery": int((E & (df.day < df.holdout_from)).sum()), "n_holdout": int((E & (df.day >= df.holdout_from)).sum()),
           "n_tactical": len(tac_cols), "n_combined": len(feats["combined"]), "found": td.get("found", {}),
           "tactical_coverage": {fam: float(np.nanmean([df.loc[E, c].notna().mean() for c in cs])) for fam, cs in fams.items() if cs},
           "inplay": ip_cov, "compute": plan, "runtime_s": round(time.time() - bt.T0), "avg_overround": float(df.overround[E].mean())}
    return bt.clean({"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "db": args.db_name,
                     "params": {k: v for k, v in vars(args).items() if k != "dsn"}, "holdout_from": str(pd.Timestamp(df.holdout_from.iloc[0]).date()),
                     "models_1x2": models_1x2, "models_ou": models_ou, "eval": evals, "value_added": va, "importance": imp,
                     "matchups": mq, "families": {k: v for k, v in fams.items()}, "inplay": ipres, "leak": leak, "minute_test": mt,
                     "shrink": shrink, "coverage": cov, "teams": teams, "players": players, "coaches": coaches,
                     "rows": {"cols": cols, "probs": probs, "ou": ou, "bets": bets, "edges": edges}})


def render(payload, title):
    css = re.search(r"<style>(.*?)</style>", bt.DASHBOARD, re.S).group(1)
    js = bt.DASHBOARD[bt.DASHBOARD.index("const $ = s =>"): bt.DASHBOARD.index("// ---------------- state & controls")]
    page = TEMPLATE.replace("__CSS__", css).replace("__HELPERS__", js).replace("__TITLE__", htmlmod.escape(title))
    return page.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__
.fam{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;background:var(--s1)}
.kpi{font-size:13px;color:var(--ink2);max-width:760px}
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub" id="sub"></p>
<div class="badges" id="badges"></div>

<div class="controls">
  <label>Market<select id="mkt"><option value="1x2">Match result (1X2)</option><option value="ou">Over/Under 2.5 goals</option></select></label>
  <label>Price taken<select id="price"></select></label>
  <label>Period<select id="period"><option value="holdout">Hold-out (honest)</option><option value="discovery">Discovery</option><option value="all">All</option></select></label>
  <label>Model in focus<select id="focus"></select></label>
  <button id="theme" type="button">Theme</button>
</div>

<div class="card hero" id="hero"></div>
<div class="grid g5" id="tiles" style="margin-top:14px"></div>

<div class="section"><h2>Do tactics add anything?</h2><p class="note">Log-loss gain per match (×10⁻³) of the first model over the second on the same matches. Above zero = better forecasts. Whiskers: 95% interval from resampling whole weeks. This is the cleanest test: it does not depend on prices or betting luck.</p></div>
<div class="grid g2">
  <div class="card"><h2>Forecast gain by comparison</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>All competitions</span><span class="key"><span class="sw" style="background:var(--s2);border-radius:99px"></span>Leagues only</span></div><div id="c-va"></div></div>
  <div class="card"><h2>Comparison detail</h2><div class="scroll"><table id="t-va"></table></div></div>
</div>

<div class="section"><h2>Which tactical families carry signal?</h2><p class="note" id="imp-note"></p></div>
<div class="grid g2">
  <div class="card"><h2>By attribute family</h2><p class="note">Hold-out log-loss increase when the family is scrambled (×10⁻³). Bigger = the model relies on it and it helps out of sample.</p><div id="c-fam"></div></div>
  <div class="card"><h2>Top individual attributes</h2><div id="c-feat"></div></div>
</div>

<div class="section"><h2>Matchups: where styles collide</h2><p class="note">Home points won minus the market's expected points, by quintile of each matchup (home advantage in the matchup minus away's). Cut points are set on the discovery period. A real effect slopes the same way in both periods.</p></div>
<div class="grid g2">
  <div class="card"><label class="lbl">Matchup <select id="mq-sel"></select></label><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Hold-out</span><span class="key"><span class="ln" style="background:var(--s2)"></span>Discovery</span></div><div id="c-mq"></div><p class="note" id="mq-why"></p></div>
  <div class="card"><h2>All matchups</h2><div class="scroll"><table id="t-mq"></table></div></div>
</div>

<div class="section"><h2>Model leaderboard</h2><p class="note">Bets back the outcome with the largest edge when edge ≥ τ; τ was chosen on the discovery period.</p></div>
<div class="card wide"><table id="t-lead"></table></div>
<div class="grid g2" style="margin-top:14px">
  <div class="card"><h2>Cumulative profit · <span id="fname"></span></h2><div id="c-cum"></div></div>
  <div class="card"><h2>Calibration · <span id="fname2"></span></h2><div id="c-cal"></div></div>
</div>

<div class="section"><h2>In-game tactical tweaks</h2><p class="note" id="ip-note"></p></div>
<div class="grid g2">
  <div class="card"><h2>In-play log loss by minute (hold-out)</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--ctrl)"></span>Scoreboard (logit)</span><span class="key"><span class="ln" style="background:var(--s2)"></span>Scoreboard (boosting)</span><span class="key"><span class="ln" style="background:var(--s1)"></span>Tactical in-play</span></div><div id="c-ip"></div></div>
  <div class="card"><h2>Do changes from the bench work?</h2><p class="note">Points won by a side that is behind, minus what the scoreboard model expected at that minute.</p><div id="c-tw"></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>Coaches' in-game habits</h2><div class="scroll"><table id="t-coach"></table></div></div>

<div class="section"><h2>Team style map</h2><p class="note">Latest profile of every team (weighted towards recent matches).</p></div>
<div class="grid g2">
  <div class="card"><div class="controls" style="position:static;margin:0 0 8px;padding:6px 8px"><label>X axis<select id="sx"></select></label><label>Y axis<select id="sy"></select></label><label>League<select id="sl"></select></label></div><div id="c-map"></div></div>
  <div class="card"><h2>Team profiles</h2><div class="scroll" style="max-height:560px"><table id="t-teams"></table></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>Player technical profiles (per 90, shrunk to positional average)</h2><div class="controls" style="position:static;margin:0 0 8px;padding:6px 8px"><label>Role<select id="prole"><option value="">All</option><option value="G">Goalkeepers</option><option value="D">Defenders</option><option value="M">Midfielders</option><option value="A">Attackers</option></select></label></div><div class="scroll"><table id="t-players"></table></div></div>

<div class="section"><h2>Match explorer</h2><p class="note" id="mx-note"></p></div>
<div class="card">
  <div class="controls" style="position:static;margin-bottom:10px">
    <label>League<select id="mx-league"></select></label>
    <label>Team<input id="mx-team" placeholder="search"></label>
    <label>Show<select id="mx-show"><option value="all">All matches</option><option value="bets">Focus model bets</option></select></label>
    <button id="mx-csv" type="button">Download all rows (CSV)</button>
  </div>
  <div class="wide"><table id="t-mx"></table></div>
  <div class="pager"><button id="pg-prev" type="button">‹ Prev</button><span id="pg-info"></span><button id="pg-next" type="button">Next ›</button></div>
</div>

<div class="section"><h2>Data, safeguards and method</h2></div>
<div class="grid g2">
  <div class="card"><h2>Tactical data found in your database</h2><table id="t-found"></table><h3>Coverage of each attribute family (evaluated matches)</h3><table id="t-famcov"></table></div>
  <div class="card"><h2>How look-ahead is prevented</h2><ul class="m" id="safeguards"></ul><h3>Leak tests</h3><table id="t-leak"></table></div>
</div>
</div></div>
<div class="tip" id="tip"></div>
<script>
const R = __DATA__;
__HELPERS__

// ---------------- tactical dashboard ----------------
const FAM_DESC = {
  'Team style':'What a team does: possession, directness, patience, width, penetration, shot selection, pressing, line height, aerial reliance, dribbling, aggression.',
  'Team execution':'How well: xG for and against, xG per shot, finishing, pass accuracy, duel and aerial win %, ball security, keeper save %, corners.',
  'Starting XI technique':'Today\'s starters\' per-90 history by line: threat, creativity, dribbling, crossing, aerial strength, build-up security, ball-winning, keeper.',
  'Matchups':'How the two profiles interact: press vs build-up, crossing vs aerial defence, direct vs high line, dribbling vs duels, set pieces, chance quality vs keeper, counters vs possession.',
  'In-game habits':'Team and coach behaviour in matches: scoring when behind, protecting leads, comebacks, late goals, substitution timing, attacking changes when chasing, sub impact.',
  'Formation':'Starting shape (defenders, midfielders, forwards) and change from the usual.',
  'Style memory':'How each team has done against the market versus opponents of this style before.',
  'Base football data':'Form, Elo, xG ratings, schedule, table, referee, weather (as in backtest_models.py).'};
const MQ_WHY = {
  md_press_vs_build:'High press against a side whose defenders and midfielders are weak on the ball.', md_cross_vs_air:'Crossing volume against a defence that loses aerial duels.',
  md_direct_vs_line:'Long balls against a high defensive line (one that catches many offsides).', md_drib_vs_duel:'Dribblers against a defence that loses duels.',
  md_setpiece:'Corner volume times the aerial edge of our forwards over their defenders.', md_chance_vs_gk:'Chance quality (xG per shot) against a keeper who saves fewer.',
  md_counter:'Penetrating attacks against a team that keeps the ball (space behind).', md_threat_vs_win:'Starting attackers\' threat against their ball-winners.',
  poss_clash:'Possession style of home minus away.', style_distance:'How different the two teams\' styles are.', h_memory:'Home team\'s record vs the market against this opponent style.', a_memory:'Away team\'s record vs the market against this opponent style.'};
const DESC = {market:'De-vigged closing consensus.', base_gbm:'Boosted market errors using base football features only.', tactical_gbm:'Boosted market errors using tactical features only.',
  combined_gbm:'Boosted market errors using base + tactical features.', tactical_standalone:'Gradient boosting on tactics and strength, no odds.',
  base_gbm_ou:'Over/Under, base features.', tactical_gbm_ou:'Over/Under, tactical features.', combined_gbm_ou:'Over/Under, base + tactical.'};
const PRICE_NAME = {avg:'Average closing price', best:'Best closing price', sharp:'Sharp closing price'};
const mktSel = $('#mkt'), priceSel = $('#price'), perSel = $('#period'), focusSel = $('#focus');
const P = () => perSel.value, isOU = () => mktSel.value === 'ou';
const models = () => isOU() ? R.models_ou : R.models_1x2, rec = m => isOU() ? R.eval.ou[m] : R.eval.models[m];
const k3 = v => v == null ? '–' : sgn(v*1000, 2);

function fillControls(){
  const pr = isOU() ? Object.keys(R.eval.ou[R.models_ou[0]]?.bets || {}) : R.eval.prices.filter(p => p !== 'open');
  const cur = priceSel.value; priceSel.innerHTML = pr.map(p => `<option value="${p}">${PRICE_NAME[p]||p}</option>`).join(''); if (pr.includes(cur)) priceSel.value = cur;
  const cf = focusSel.value; focusSel.innerHTML = models().map(m => `<option>${m}</option>`).join('');
  focusSel.value = models().includes(cf) ? cf : (isOU() ? 'combined_gbm_ou' : 'combined_gbm');
}

function header(){
  const C = R.coverage;
  $('#sub').innerHTML = `${int(C.eval)} matches evaluated across ${C.leagues} competitions, ${C.from} → ${C.to}. ${C.n_tactical} tactical attributes (${C.n_combined} features with the base set). Hold-out from <b>${R.holdout_from}</b>. Database <b>${esc(R.db)}</b> · ${R.generated}.`;
  const L = R.leak, M = R.minute_test, b = [];
  if (L.skipped) b.push('<span class="badge">Time-travel test skipped</span>');
  else b.push(`<span class="badge ${L.passed?'ok':'bad'}">${L.passed?'✓':'✕'} Time-travel test ${L.passed?'passed':'FAILED'} (base + tactical engines, ${L.tests.length} cutoffs, ${L.tests.reduce((a,t)=>a+t.mismatches,0)} changed)</span>`);
  if (M && !M.skipped) b.push(`<span class="badge ${M.passed?'ok':'bad'}">${M.passed?'✓':'✕'} Minute-travel test: ${int(M.checked)} in-play values, ${M.mismatches} changed</span>`);
  if (L.alarms && L.alarms.length) b.push(`<span class="badge bad">⚠ Implausibly strong: ${L.alarms.map(a => esc(a.model) + ' (' + sgn(a.gain*1000,1) + '×10⁻³)').join(', ')}. Treat as a leak until proven otherwise.</span>`);
  b.push(`<span class="badge">Thresholds fixed before ${R.holdout_from}</span>`);
  $('#badges').innerHTML = b.join('');
}

function vaRow(label){ return R.value_added.find(v => v.label === label); }
function hero(){
  const per = P(), v = vaRow('What tactics add on top of the base model'), x = v?.periods?.[per]?.all, t = vaRow('Tactics alone vs the closing market')?.periods?.[per]?.all;
  const ip = R.inplay?.gain?.[per];
  if (!x || x.gain == null){ $('#hero').innerHTML = '<div class="lbl">Not enough matches in this period.</div>'; }
  else { const sig = x.p < 0.05;
    $('#hero').innerHTML = `<div><div class="big ${cls(x.gain)}">${k3(x.gain)}</div><div class="lbl">Log-loss gain ×10⁻³ from adding tactics<br><span class="note">${per} · ${int(x.n)} matches</span></div></div>
      <div class="kpi">Adding the tactical profiles to the base model ${x.gain > 0 ? 'improved' : 'did not improve'} forecasts by ${k3(x.gain)} thousandths of log loss per match
      (95% interval ${k3(x.lo)} to ${k3(x.hi)}; ${sig ? '<b>unlikely to be chance</b>' : 'could be chance'}, p = ${num(x.p,3)}).
      ${t && t.gain != null ? `Tactics alone vs the closing market: ${k3(t.gain)}.` : ''}
      ${ip && ip.gain != null ? ` In play, tactical information beat the scoreboard model by ${k3(ip.gain)} (${k3(ip.lo)} to ${k3(ip.hi)}).` : ''}
      <div class="note" style="margin-top:6px">For scale: beating a closing market by 1–3 thousandths is a serious edge; much more than ${(R.params.alarm_gain*1000).toFixed(0)} usually means a leak.</div></div>`; }
  const fam = Object.entries(R.importance?.families || {}).sort((a,b) => b[1].increase - a[1].increase)[0];
  const mk = rec('market')?.metrics?.[per];
  const best = models().filter(m => m !== 'market').map(m => { const r = rec(m), pr = r.bets[priceSel.value]; const xx = pr?.periods?.[per]; return {m, x:xx}; }).filter(o => o.x && o.x.n >= 100).sort((a,b) => (b.x.lo ?? -9) - (a.x.lo ?? -9))[0];
  const tiles = [['Matches in period', int(mk?.n), isOU() ? 'with Over/Under prices' : 'with 1X2 prices'],
    ['Most useful family', fam ? esc(fam[0]) : '–', fam ? `scrambling it costs ${k3(fam[1].increase)} ×10⁻³` : ''],
    ['Most reliable bettor', best ? esc(best.m) : '–', best ? `${pct(best.x.roi)} on ${int(best.x.n)} bets (${pct(best.x.lo)} to ${pct(best.x.hi)})` : ''],
    ['In-play tactical gain', ip?.gain != null ? `<span class="${cls(ip.gain)}">${k3(ip.gain)}</span>` : '–', '×10⁻³ log loss vs scoreboard model'],
    ['Bookmaker margin', pctu(R.coverage.avg_overround, 2), 'the hurdle every bet must clear']];
  $('#tiles').innerHTML = tiles.map(t => `<div class="card tile"><div class="lbl">${t[0]}</div><div class="val">${t[1]}</div><div class="ci">${t[2]}</div></div>`).join('');
}

function valueAdded(){
  const per = P(), rows = R.value_added.filter(v => v.periods?.[per]?.all);
  hbars($('#c-va'), rows.map(v => { const a = v.periods[per].all, l = v.periods[per].league;
    return {label: v.label.length > 30 ? v.label.slice(0, 29) + '…' : v.label, v: a.gain*1000, lo: a.lo*1000, hi: a.hi*1000, v2: l?.gain != null ? l.gain*1000 : null,
      color: a.lo > 0 ? css('--s1') : css('--ctrl'), tip:`<b>${esc(v.label)}</b><br>${esc(v.a)} vs ${esc(v.b)}<br>All: ${k3(a.gain)} (${k3(a.lo)} to ${k3(a.hi)}), ${int(a.n)} matches<br>Leagues: ${k3(l?.gain)}`}; }), v => sgn(v,1), {left:220});
  const tr = [];
  for (const v of R.value_added) for (const pn of ['discovery','holdout']) for (const seg of ['all','league','cup']) { const x = v.periods?.[pn]?.[seg]; if (!x || x.gain == null) continue;
    tr.push({v:[v.label, pn, seg, x.n, x.gain, x.p], c:[esc(v.label), pn, seg, int(x.n), `<b class="${cls(x.gain)}">${k3(x.gain)}</b>`, `${k3(x.lo)} to ${k3(x.hi)}`, num(x.p,3)]}); }
  sortable($('#t-va'), [{h:'Comparison', c:'l'}, {h:'Period', c:'l'}, {h:'Matches in', c:'l'}, {h:'n'}, {h:'Gain ×10⁻³'}, {h:'95% CI'}, {h:'p'}], tr);
}

function importanceView(){
  const I = R.importance;
  if (!I || !I.families){ empty($('#c-fam'), 'Not enough data'); empty($('#c-feat'), 'Not enough data'); return; }
  $('#imp-note').textContent = `A model of the market's errors with every feature was trained on the discovery period and scored on the hold-out (log loss ${num(I.base_ll,4)} vs market ${num(I.market_ll,4)}). Each family is then scrambled in turn.`;
  const F = Object.entries(I.families).sort((a,b) => b[1].increase - a[1].increase);
  hbars($('#c-fam'), F.map(([k, v]) => ({label:k, v:v.increase*1000, color: v.increase > 0 ? css('--s1') : css('--ctrl'), tip:`<b>${esc(k)}</b> (${v.n_features} features)<br>${esc(FAM_DESC[k]||'')}<br>Hold-out log loss +${k3(v.increase)} ×10⁻³ when scrambled`})), v => sgn(v,2), {left:180});
  const T = Object.entries(I.features).slice(0, 20);
  hbars($('#c-feat'), T.map(([k, v]) => ({label:k.length > 24 ? k.slice(0,23)+'…' : k, v:v*1000, color: v > 0 ? css('--s1') : css('--ctrl'), tip:`<b>${esc(k)}</b><br>+${k3(v)} ×10⁻³ when scrambled`})), v => sgn(v,2), {left:180, rowH:24});
}

function matchups(){
  const M = R.matchups || {}, keys = Object.keys(M);
  const sel = $('#mq-sel'); if (!sel.options.length) { sel.innerHTML = keys.map(k => `<option>${k}</option>`).join(''); sel.onchange = matchups; }
  const k = sel.value || keys[0];
  if (!k){ empty($('#c-mq'), 'No matchup data'); return; }
  const q = M[k].quintiles;
  const ser = pn => q.filter(r => r[pn].resid != null).map(r => [r.q, r[pn].resid, r[pn].n]);
  lines($('#c-mq'), [{name:'Discovery', color:css('--s2'), pts:ser('discovery'), dots:true}, {name:'Hold-out', color:css('--s1'), pts:ser('holdout'), dots:true}],
    {fmt:v => sgn(v,2), xfmt:v => 'Q' + v, xlabel:v => 'Quintile ' + v + (v === 1 ? ' (away edge)' : v === 5 ? ' (home edge)' : ''), yref:0, extra:p => p[2] + ' matches'});
  $('#mq-why').textContent = (MQ_WHY[k] || '') + ' Points per match above market expectation.';
  sortable($('#t-mq'), [{h:'Matchup', c:'l'}, {h:'Discovery spread'}, {h:'Hold-out spread'}, {h:'Same direction'}],
    keys.map(kk => { const x = M[kk]; const same = x.spread_hold != null && Math.sign(x.spread_hold) === Math.sign(x.spread_disc);
      return {k:kk, v:[kk, x.spread_disc, x.spread_hold, same], c:[esc(kk), sgn(x.spread_disc,3), `<b class="${cls(x.spread_hold)}">${sgn(x.spread_hold,3)}</b>`, same ? '✓' : '']}; }),
    {onRow: kk => { sel.value = kk; matchups(); }});
}

function leaderboard(){
  const per = P(), focus = focusSel.value;
  const head = [{h:'Model', c:'l'}, {h:'Log loss'}, {h:'Gain ×10⁻³'}, {h:'τ'}, {h:'Bets'}, {h:'Won'}, {h:'Avg price'}, {h:'Return'}, {h:'95% CI'}, {h:'p'}, {h:'Profit'}];
  sortable($('#t-lead'), head, models().map(m => { const r = rec(m), mt = r.metrics[per] || {}, b = r.bets[priceSel.value], x = b?.periods?.[per];
    const g = mt.logloss != null && mt.mk_logloss != null ? (mt.mk_logloss - mt.logloss) : null;
    return {k:m, sel:m === focus, v:[m, mt.logloss, g, b?.tau, x?.n, x?.win, x?.avg_price, x?.roi, x?.lo, x?.p, x?.profit],
      c:[`${esc(m)}<div class="why">${esc(DESC[m]||'')}</div>`, num(mt.logloss,4), `<span class="${cls(g)}">${k3(g)}</span>`, b ? (b.tau*100).toFixed(0)+'%' : '–', int(x?.n), pctu(x?.win), num(x?.avg_price),
         `<b class="${cls(x?.roi)}">${pct(x?.roi)}</b>`, x?.lo != null ? `${pct(x.lo)} to ${pct(x.hi)}` : '–', num(x?.p,3), `<span class="${cls(x?.profit)}">${sgn(x?.profit)}</span>`]}; }),
    {onRow: k => { focusSel.value = k; draw(); }});
  const m = focusSel.value, r = rec(m), b = r.bets[priceSel.value];
  $('#fname').textContent = m; $('#fname2').textContent = m;
  if (b && b.cum) lines($('#c-cum'), [{name:'Profit', color:css('--s1'), pts:b.cum, endLabel: b.cum.length ? sgn(b.cum[b.cum.length-1][1]) + 'u' : ''}], {vline:R.holdout_from, vlabel:'hold-out', yref:0, fmt:v => sgn(v,0), tfmt:v => sgn(v,1)+'u'});
  else empty($('#c-cum'), 'This model does not place bets.');
  calChart($('#c-cal'), r.calibration);
}

function inplayView(){
  const I = R.inplay, per = P() === 'all' ? 'holdout' : P();
  if (!I || !I.checkpoints || !Object.keys(I.checkpoints).length){ empty($('#c-ip'), 'In-play models not run (no events, or --no-inplay).'); empty($('#c-tw'), ''); $('#ip-note').textContent=''; return; }
  const C = R.coverage.inplay || {};
  $('#ip-note').textContent = `At 15', 30', half-time, 60' and 75' each model sees only what has happened so far. The tactical model adds substitutions and how attacking they were, momentum${C.with_momentum ? '' : ' (no momentum data found)'}, pressure${C.with_pressure ? '' : ' (no pressure data found)'}, both teams' and coaches' in-game habits and the pre-match tactical forecast. ${int(C.rows)} in-play rows from ${int(C.fixtures)} matches.`;
  const pts = name => Object.entries(I.checkpoints).filter(([c, v]) => v[per]).map(([c, v]) => [+c === 46 ? 45 : +c, v[per][name], v[per].n]);
  lines($('#c-ip'), [{name:'Scoreboard (logit)', color:css('--ctrl'), pts:pts('ip_state'), dots:true}, {name:'Scoreboard (boosting)', color:css('--s2'), pts:pts('ip_state_gbm'), dots:true},
    {name:'Tactical in-play', color:css('--s1'), pts:pts('ip_tactical'), dots:true}], {fmt:v => v.toFixed(3), xfmt:v => v + "'", xlabel:v => 'Minute ' + v, zero:false, extra:p => p[2] + ' matches'});
  const T = (I.tweaks || []).slice().sort((a, b) => a.minute - b.minute || (a.side > b.side ? 1 : -1));
  hbars($('#c-tw'), T.map(t => ({label:`${t.minute}' ${t.side}: ${t.group}`.slice(0, 40), v:t.resid, lo:t.lo, hi:t.hi, color: t.lo > 0 || t.hi < 0 ? css('--s1') : css('--ctrl'),
    tip:`<b>${t.side} side behind at ${t.minute}'</b><br>${esc(t.group)}<br>${int(t.n)} matches<br>Points vs expected: ${sgn(t.resid,3)} (${sgn(t.lo,3)} to ${sgn(t.hi,3)})`})), v => sgn(v,2), {left:250, rowH:26});
  sortable($('#t-coach'), [{h:'Coach', c:'l'}, {h:'Matches (weighted)'}, {h:'First sub (min)'}, {h:'Subs per match when behind'}, {h:'Change aggressiveness'}, {h:'Goal diff/90 after first sub vs before'}, {h:'Goals/90 when behind'}, {h:'Conceded/90 when ahead'}, {h:'Comeback pts'}],
    (R.coaches || []).map(c => ({v:[c.coach, c.games_w, c.first_sub, c.chase_rate, c.chase_aggr, c.sub_impact, c.trail_attack, c.lead_defence, c.comeback],
      c:[esc(c.coach), num(c.games_w,1), num(c.first_sub,1), num(c.chase_rate,2), num(c.chase_aggr,2), `<span class="${cls(c.sub_impact)}">${sgn(c.sub_impact,2)}</span>`, num(c.trail_attack,2), num(c.lead_defence,2), num(c.comeback,2)]})), {sortKey:1});
}

// ---- team style map (scatter) ----
const AX = ['poss','direct','patience','width','penetration','boxshots','press','line','aerial','dribble','aggression','xgf','xga','xg_shot','finish','pass_acc','duel','aerial_win','security','gk_save','corners'];
const AXN = {poss:'Possession', direct:'Directness (long balls / passes)', patience:'Patience (passes / attack)', width:'Width (crosses per 10 min on the ball)', penetration:'Penetration (dangerous / all attacks)',
  boxshots:'Inside-box shot share', press:'Pressing (def. actions per 10 min without the ball)', line:'Line height (opp offsides)', aerial:'Aerial share of duels', dribble:'Dribbles per 10 min on the ball', aggression:'Fouls',
  xgf:'xG for', xga:'xG against', xg_shot:'xG per shot', finish:'Finishing (goals − xG)', pass_acc:'Pass accuracy', duel:'Duel win %', aerial_win:'Aerial win %', security:'Possession lost per touch', gk_save:'Keeper save %', corners:'Corners'};
function teamMap(){
  const sx = $('#sx'), sy = $('#sy'), sl = $('#sl');
  if (!sx.options.length){ sx.innerHTML = sy.innerHTML = AX.map(a => `<option value="${a}">${AXN[a]}</option>`).join(''); sx.value = 'poss'; sy.value = 'press';
    sl.innerHTML = '<option value="">All</option>' + [...new Set(R.teams.map(t => t.league))].sort().map(l => `<option>${esc(l)}</option>`).join('');
    for (const s of [sx, sy, sl]) s.onchange = teamMap; }
  const T = R.teams.filter(t => (!sl.value || t.league === sl.value) && t[sx.value] != null && t[sy.value] != null);
  const host = $('#c-map');
  if (!T.length) return empty(host, 'No team profiles');
  const W = 600, H = 420, top = 12, bottom = 40, left = 56, right = 14, s = svg(host, W, H);
  const xs = T.map(t => t[sx.value]), ys = T.map(t => t[sy.value]);
  const [x0, x1] = ext(xs, false), [y0, y1] = ext(ys, false);
  const x = v => left + (v-x0)/(x1-x0)*(W-left-right), y = v => top + (y1-v)/(y1-y0)*(H-top-bottom);
  for (const t of ticks(x0, x1, 6)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-bottom, stroke:css('--grid')}, s); text(s, x(t), H-bottom+15, +t.toFixed(3), {'text-anchor':'middle'}); }
  for (const t of ticks(y0, y1, 6)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, +t.toFixed(3), {'text-anchor':'end'}); }
  text(s, left+(W-left-right)/2, H-4, AXN[sx.value], {'text-anchor':'middle'});
  text(s, 12, top + (H-top-bottom)/2, AXN[sy.value], {'text-anchor':'middle', transform:`rotate(-90 12 ${top + (H-top-bottom)/2})`});
  for (const t of T){ const c = el('circle', {cx:x(t[sx.value]), cy:y(t[sy.value]), r:5, fill:css('--s1'), stroke:css('--surface'), 'stroke-width':2, opacity:.85}, s);
    hit(el('circle', {cx:x(t[sx.value]), cy:y(t[sy.value]), r:10, class:'hit'}, s), `<b>${esc(t.team)}</b><br>${esc(t.league)}<br>${AXN[sx.value]}: ${num(t[sx.value],3)}<br>${AXN[sy.value]}: ${num(t[sy.value],3)}<br>xG ${num(t.xgf,2)} for / ${num(t.xga,2)} against`); }
  const cols = ['poss','direct','width','penetration','press','line','aerial','xgf','xga','pass_acc','aerial_win','gk_save','g_trail_attack','g_lead_defence','g_chase_aggr','g_sub_impact'];
  sortable($('#t-teams'), [{h:'Team', c:'l'}, {h:'League', c:'l'}, ...cols.map(c => ({h:(AXN[c] || c.replace('g_','in-game ').replace('_',' ')).split(' (')[0]}))],
    T.map(t => ({v:[t.team, t.league, ...cols.map(c => t[c])], c:[esc(t.team), esc(t.league), ...cols.map(c => num(t[c], 3))]})), {sortKey:2});
}
function playersView(){
  const r = $('#prole').value, Pl = (R.players || []).filter(p => !r || p.role === r);
  const cols = ['rating','pass_acc','passes','key_passes','acc_crosses','drib_succ','tackles','intercepts','aer_pct','duels_won','sot','goals','assists','saves'];
  sortable($('#t-players'), [{h:'Player', c:'l'}, {h:'Role', c:'l'}, {h:'Weighted minutes'}, ...cols.map(c => ({h:c.replace('_',' ')}))],
    Pl.map(p => ({v:[p.name, p.role, p.minutes_w, ...cols.map(c => p[c])], c:[esc(p.name), p.role, int(p.minutes_w), ...cols.map(c => num(p[c], c === 'pass_acc' || c === 'aer_pct' ? 3 : 2))]})), {sortKey:3});
}
$('#prole').onchange = playersView;

// ---- match explorer ----
const RW = R.rows, Cc = RW.cols, NR = Cc.id.length; let page = 0, mx = [];
$('#mx-league').innerHTML = '<option value="">All</option>' + [...new Set(Cc.league)].sort().map(l => `<option>${esc(l)}</option>`).join('');
function mxFilter(){ const lg = $('#mx-league').value, tm = $('#mx-team').value.trim().toLowerCase(), show = $('#mx-show').value, bets = RW.bets[isOU() ? 'ou_' + focusSel.value : focusSel.value];
  mx = []; for (let i = NR-1; i >= 0; i--){ if (lg && Cc.league[i] !== lg) continue; if (tm && !(Cc.home[i].toLowerCase().includes(tm) || Cc.away[i].toLowerCase().includes(tm))) continue; if (show === 'bets' && (!bets || bets[i] < 0)) continue; mx.push(i); }
  page = 0; mxDraw(); }
function mxDraw(){
  const m = focusSel.value, mp = RW.probs[m], mk = RW.probs.market, bets = RW.bets[m], s0 = page*50, sl = mx.slice(s0, s0+50);
  const tcols = ['md_press_vs_build','md_cross_vs_air','md_direct_vs_line','style_distance'].filter(c => Cc[c]);
  let h = '<thead><tr><th class="l">Date</th><th class="l">League</th><th class="l">Match</th><th>Score</th><th>Market H/D/A</th><th>Model H/D/A</th>' + tcols.map(c => `<th>${c}</th>`).join('') + '<th>Bet</th></tr></thead><tbody>';
  for (const i of sl){ const b = bets ? bets[i] : -1;
    h += `<tr data-i="${i}" style="cursor:pointer"><td class="l">${Cc.date[i]}</td><td class="l">${esc(Cc.league[i])}</td><td class="l">${esc(Cc.home[i])} – ${esc(Cc.away[i])}</td><td>${Cc.hg[i]}–${Cc.ag[i]}</td>
      <td>${mk ? [0,1,2].map(k => pctu(mk[k][i],0)).join(' / ') : '–'}</td><td>${mp ? [0,1,2].map(k => pctu(mp[k][i],0)).join(' / ') : '–'}</td>${tcols.map(c => `<td>${num(Cc[c][i],3)}</td>`).join('')}
      <td>${b >= 0 ? `<span class="pill bet">${['Home','Draw','Away'][b]}</span>` : ''}</td></tr>`; }
  $('#t-mx').innerHTML = h + '</tbody>';
  $('#t-mx').querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => { const nx = tr.nextElementSibling; if (nx && nx.classList.contains('detail')) { nx.remove(); return; }
    const i = +tr.dataset.i; const parts = Object.keys(Cc).filter(k => !['id','date','league','home','away','hg','ag'].includes(k)).map(k => `<div><span>${esc(k)}</span> ${Cc[k][i] ?? '–'}</div>`);
    for (const mm of R.models_1x2) if (RW.probs[mm]) parts.unshift(`<div><span>${esc(mm)}</span> ${[0,1,2].map(k => pctu(RW.probs[mm][k][i],1)).join(' / ')}</div>`);
    tr.insertAdjacentHTML('afterend', `<tr class="detail"><td colspan="${7 + tcols.length}"><div class="kv">${parts.join('')}</div></td></tr>`); });
  $('#pg-info').textContent = mx.length ? `${s0+1}–${Math.min(s0+50, mx.length)} of ${mx.length.toLocaleString()}` : 'No matches';
  $('#mx-note').textContent = `${NR.toLocaleString()} evaluated matches with every model's probabilities and the main tactical attributes. Click a row for everything; the CSV holds every column.`;
}
$('#pg-prev').onclick = () => { if (page > 0){ page--; mxDraw(); } }; $('#pg-next').onclick = () => { if ((page+1)*50 < mx.length){ page++; mxDraw(); } };
$('#mx-league').onchange = mxFilter; $('#mx-show').onchange = mxFilter; $('#mx-team').oninput = mxFilter;
$('#mx-csv').onclick = () => { const keys = Object.keys(Cc), head = [...keys]; for (const m of R.models_1x2) if (RW.probs[m]) head.push(`${m}_p_home`, `${m}_p_draw`, `${m}_p_away`);
  const q = v => v == null ? '' : (/[",\n]/.test(String(v)) ? '"' + String(v).replace(/"/g,'""') + '"' : String(v)); const out = [head.join(',')];
  for (let i = 0; i < NR; i++){ const r = keys.map(k => q(Cc[k][i])); for (const m of R.models_1x2) if (RW.probs[m]) r.push(q(RW.probs[m][0][i]), q(RW.probs[m][1][i]), q(RW.probs[m][2][i])); out.push(r.join(',')); }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([out.join('\n')], {type:'text/csv'})); a.download = 'tactical_matches.csv'; a.click(); };

function staticPanels(){
  const F = R.coverage.found || {}, rows = [];
  for (const [grp, m] of Object.entries(F)) for (const [k, v] of Object.entries(m)) rows.push(`<tr><td class="l">${grp}</td><td class="l">${esc(k)}</td><td class="l">${esc(v)}</td></tr>`);
  $('#t-found').innerHTML = '<thead><tr><th class="l">Level</th><th class="l">Attribute</th><th class="l">Sportmonks type used</th></tr></thead><tbody>' + (rows.join('') || '<tr><td class="note">Nothing found</td></tr>') + '</tbody>';
  $('#t-famcov').innerHTML = '<tbody>' + Object.entries(R.coverage.tactical_coverage || {}).map(([k, v]) => `<tr><td class="l"><span class="fam"></span>${esc(k)}</td><td>${pctu(v)}</td></tr>`).join('') + '</tbody>';
  $('#safeguards').innerHTML = [
    'Same engine pattern as backtest_models.py: each day\'s matches are described from the state before that day; results, player statistics, team statistics and events are only read afterwards.',
    'Starting line-ups, formations and bench are used for the current match (published about an hour before kickoff); every player rating and per-90 number comes from earlier matches only, shrunk towards the positional average.',
    'In-play rows use only events, momentum and pressure from before the checkpoint minute; the minute-travel test recomputes them with all later events deleted.',
    'Models retrain every ' + R.params.retrain_months + ' month(s) on earlier matches; boosting decides how much of its correction to trust on the latest 20% of each window.',
    'In-play models use the pre-match models\' out-of-sample forecasts; family importance comes from a model trained on discovery and scored on the hold-out.',
    `Leakage alarm: any model beating the closing market by more than ${(R.params.alarm_gain*1000).toFixed(0)}×10⁻³ log loss is flagged at the top.`].map(x => `<li>${x}</li>`).join('');
  const L = R.leak;
  $('#t-leak').innerHTML = L.skipped ? '<tr><td class="note">Time-travel test skipped</td></tr>' : '<thead><tr><th class="l">Engine</th><th class="l">Cutoff</th><th>Matches</th><th>Features</th><th>Changed</th></tr></thead><tbody>' +
    L.tests.map(t => `<tr><td class="l">${t.engine || 'base'}</td><td class="l">${t.day}</td><td>${t.matches}</td><td>${t.features}</td><td class="${t.mismatches?'neg':'pos'}">${t.mismatches}</td></tr>`).join('') + '</tbody>';
}

function draw(){ hero(); valueAdded(); importanceView(); matchups(); leaderboard(); inplayView(); teamMap(); playersView(); mxFilter(); }
mktSel.onchange = () => { fillControls(); draw(); }; for (const s of [priceSel, perSel, focusSel]) s.onchange = draw;
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; draw(); };
fillControls(); header(); staticPanels(); draw();
</script>
</body>
</html>
'''

if __name__ == "__main__":
    main()
