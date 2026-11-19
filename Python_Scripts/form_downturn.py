#!/usr/bin/env python3
"""
form_downturn.py - spotting a player's drop in form before it happens
=====================================================================

Reads the Sportmonks PostgreSQL database (schema.sql), builds a context-adjusted
performance history for every player, and trains gradient-boosted models
(XGBoost, or scikit-learn's histogram boosting if XGBoost is missing) that
predict three kinds of downturn BEFORE the match in question:

  next       Single game: will the player's next performance be clearly below
             his own norm? (bottom ~25% of "performance minus own baseline")
  sustained  Long term: will his average over his next K appearances (default
             6, within 150 days) sit clearly below his own norm?
  dropped    Coach's verdict: a regular starter (70%+ of the last 10 team
             matches) starts 2 or fewer of his team's next 6 matches
             (rows where an injury or transfer explains it are excluded).

"Performance" is the Sportmonks match rating adjusted for context (see below).

CONFIDENCE SIGNALS (each one adjusted for context, then tracked as recent vs
usual for that player)
  dribbling: attempts per 90 (reluctance) and success rate
  shooting: shots per 90, accuracy (on target / all), xG per shot, big chances
            missed, goals minus xG (finishing luck), drought (xG since last goal)
  passing:  accuracy, volume, risk-taking (key-pass rate, final-third share,
            long-ball share, backward-pass share = "negative passes" if your
            plan has BACKWARD_PASSES), cross and long-ball accuracy
  involvement: touches per 90, fouls drawn
  ball security: turnovers (possession lost / dispossessed) per touch AND per 90
  frustration: fouls per 90, cards
  coach trust: minutes taken off earlier than this coach/team usually does with
            this position in this game state (injury substitutions excluded),
            hooked before 60' / at half-time, subbed while losing, losing the
            starting place (starts in the last 5 vs last 20 team matches)
  duels & defending, errors (leading to shots/goals), offsides, missed penalties,
  goalkeeping (save % vs expected)
  plus workload (minutes in the last 7/14/28 days, rest), injury history,
  off-field (transfer, transfer rumours, new coach), age, role familiarity, the
  team's own form and the opponent and market for the match being predicted.

NORMALISATION - every match stat becomes "actual vs expected", where expected =
  league x position baseline (shrunk to the position's global rate)
  x the player's team (how that team's players in that position usually do)
  x the opponent (what that opponent usually allows that position)
  x home/away x game state (minutes the player spent winning/drawing/losing).
Counts use per-90 log ratios, success rates use binomial z-scores, the rating an
additive residual. Positions come from the line-up (detailed position when
available: GK, CB, FB, DM, CM, AM, wide, striker).

NO LEAKAGE FOR THE MATCH BEING TESTED
  * All baselines above are time-decayed sums over EARLIER DAYS only (a match
    never sees anything from its own day, not even other kick-offs).
  * Player features use his earlier appearances only. For the match being
    predicted only genuinely pre-match facts are used: line-up (started, role),
    home/away, opponent, pre-kick-off odds, rest, injuries/transfers/rumours
    dated before the match day.
  * Labels look forward, so training rows are PURGED: a row may only train a
    model if its whole label window ended before the first day being predicted.
  * Walk-forward: models retrain every few months on earlier data only; the
    downturn thresholds are frozen from the first training window.
  * Time-travel test: the whole pipeline is re-run with every match after a
    cutoff day deleted and that day's match data blanked; every feature for that
    day's matches must come out identical. Verdict shown on the dashboard.
  * Feature importance comes from a model trained on the discovery period and
    scored on the later hold-out period.

USAGE (same folder as backtest_models.py / load_league.py, or pass --dsn)
    python3 form_downturn.py
    python3 form_downturn.py --leagues 8,82 --cache .cache --export-csv form.csv
    python3 form_downturn.py --demo          # synthetic data, to see the dashboard
"""
from __future__ import annotations

import argparse
import html as htmlmod
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bootstrap():
    import importlib
    import importlib.util
    need = [p for m, p in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn")) if importlib.util.find_spec(m) is None]
    opt = []
    if importlib.util.find_spec("xgboost") is None:
        opt.append("xgboost")
    if "--demo" not in sys.argv and importlib.util.find_spec("psycopg") is None and importlib.util.find_spec("psycopg2") is None:
        opt.append("psycopg[binary]")
    if not need and not opt:
        return
    if os.environ.get("FORM_NO_INSTALL") or os.environ.get("BACKTEST_NO_INSTALL"):
        if need:
            sys.exit(f"Missing packages: {' '.join(need)}  ->  {sys.executable} -m pip install {' '.join(need)}")
        return

    def pip(pk):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", "--retries", "1", "--timeout", "15", *pk],
                           capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or r.stdout or "")
    if need:
        ok, err = pip(need)
        if not ok:
            sys.exit(f"Could not install {need}:\n{err[-800:]}\nInstall by hand: {sys.executable} -m pip install {' '.join(need)}")
    for p in opt:
        ok, err = pip([p])
        if not ok:
            why = [ln for ln in err.strip().splitlines() if ln.strip()][-1:] or ["unknown reason"]
            print(f"(optional package {p} could not be installed: {why[0][:160]}\n continuing without it; to add it yourself run:  "
                  f"{sys.executable} -m pip install {p})", file=sys.stderr, flush=True)
    importlib.invalidate_caches()


if __name__ == "__main__":
    _bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
try:
    import xgboost as xgb  # noqa: E402
    HAVE_XGB, XGB_ERR, XGB_FULL = True, "", ""
except Exception as _e:  # noqa: BLE001
    _lines = [ln.strip() for ln in str(_e).splitlines() if ln.strip()]
    HAVE_XGB, XGB_ERR, XGB_FULL = False, (_lines[0][:200] if _lines else type(_e).__name__), str(_e)

T0 = time.time()
DAY0 = pd.Timestamp("2000-01-01")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# =============================================================================
# Database access (psycopg 3 -> psycopg2 -> psql CLI), as in backtest_models.py
# =============================================================================
ACTIVE_CONNS: set = set()     # open query connections, cancelled on Ctrl-C
RUNNING: dict = {}            # query name -> start time


def stop_queries():
    import signal as _sig
    for c in list(ACTIVE_CONNS):
        try:
            if isinstance(c, subprocess.Popen):
                c.send_signal(_sig.SIGINT)     # psql cancels its query on SIGINT
            else:
                c.cancel()                    # psycopg: asks PostgreSQL to abort the running statement
        except Exception:  # noqa: BLE001
            pass


def _on_sigint(signum, frame):
    print("\nInterrupted - cancelling database queries and stopping.", file=sys.stderr, flush=True)
    stop_queries()
    os._exit(130)              # do not wait for worker threads stuck inside the database driver


def default_dsn():
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"], "DATABASE_URL"
    sys.path.insert(0, str(HERE))
    try:
        import load_league  # type: ignore
        cfg = getattr(load_league, "DATABASE", None)
        if isinstance(cfg, dict):
            km = {"database": "dbname", "db": "dbname", "username": "user", "passwd": "password"}
            cfg = " ".join(f"{km.get(k, k)}='{str(v)}'" for k, v in cfg.items() if v is not None)
        if cfg:
            return str(cfg), "load_league.DATABASE"
    except Exception:  # noqa: BLE001
        pass
    return "", "libpq defaults"


def _coerce(df):
    for c in df.columns:
        if c in ("starting_at", "start_date", "end_date", "date", "dob"):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        elif df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().any() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


class DB:
    def __init__(self, dsn):
        src = "--dsn"
        if not dsn:
            dsn, src = default_dsn()
        self.dsn, self.conn = dsn, None
        try:
            import psycopg  # type: ignore
            self.conn, self.kind = psycopg.connect(dsn), "psycopg"
        except ImportError:
            try:
                import psycopg2  # type: ignore
                self.conn, self.kind = psycopg2.connect(dsn), "psycopg2"
            except ImportError:
                self.kind = "psql"
        self.name = str(self.query("SELECT current_database() AS db").db.iloc[0])
        log(f"database driver {self.kind}, connected to '{self.name}' (settings from {src})")

    def query(self, sql):
        sql = sql.strip().rstrip(";")
        if self.kind == "psql":
            cmd = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1"] + ([self.dsn] if self.dsn else [])
            pr = subprocess.Popen(cmd + ["-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ACTIVE_CONNS.add(pr)
            try:
                out_b, err_b = pr.communicate()
            finally:
                ACTIVE_CONNS.discard(pr)
            if pr.returncode:
                raise RuntimeError(err_b.decode().strip())
            t = out_b.decode()
            return _coerce(pd.read_csv(io.StringIO(t), low_memory=False)) if t.strip() else pd.DataFrame()
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        finally:
            self.conn.rollback()
        return _coerce(pd.DataFrame(rows, columns=cols))

    def exists(self, t):
        return int(self.query(f"SELECT (to_regclass('{t}') IS NOT NULL)::int AS ok").ok.iloc[0]) == 1


# =============================================================================
# What to read. Sportmonks developer names; the first alias present is used.
# =============================================================================
STATS = {
    "minutes": ["MINUTES_PLAYED"], "rating": ["RATING"], "touches": ["TOUCHES"],
    "passes": ["PASSES", "TOTAL_PASSES"], "acc_passes": ["ACCURATE_PASSES", "SUCCESSFUL_PASSES"],
    "key_passes": ["KEY_PASSES"], "crosses": ["TOTAL_CROSSES", "CROSSES"], "acc_crosses": ["ACCURATE_CROSSES"],
    "long_balls": ["LONG_BALLS", "TOTAL_LONG_BALLS"], "acc_long_balls": ["LONG_BALLS_WON", "ACCURATE_LONG_BALLS", "SUCCESSFUL_LONG_BALLS"],
    "backward_passes": ["BACKWARD_PASSES", "PASSES_BACKWARD", "BACKWARD_PASS"],
    "final_third_passes": ["PASSES_IN_FINAL_THIRD", "FINAL_THIRD_PASSES", "ACCURATE_PASSES_IN_FINAL_THIRD"],
    "drib_att": ["DRIBBLED_ATTEMPTS", "DRIBBLE_ATTEMPTS", "TOTAL_DRIBBLES"], "drib_succ": ["SUCCESSFUL_DRIBBLES"],
    "dispossessed": ["DISPOSSESSED"], "poss_lost": ["POSSESSION_LOST", "POSSESSION_LOSTS", "BALL_LOSSES", "TURN_OVER", "TURNOVERS"],
    "shots": ["SHOTS_TOTAL", "TOTAL_SHOTS"], "sot": ["SHOTS_ON_TARGET"], "goals": ["GOALS"], "assists": ["ASSISTS"],
    "big_ch_missed": ["BIG_CHANCES_MISSED"], "big_ch_created": ["BIG_CHANCES_CREATED"], "woodwork": ["HIT_WOODWORK"],
    "offsides": ["OFFSIDES"], "fouls": ["FOULS", "FOULS_COMMITTED"], "fouls_drawn": ["FOULS_DRAWN"],
    "tackles": ["TACKLES", "TOTAL_TACKLES"], "interceptions": ["INTERCEPTIONS"], "recoveries": ["BALL_RECOVERY", "BALL_RECOVERIES"],
    "duels": ["TOTAL_DUELS", "DUELS"], "duels_won": ["DUELS_WON"], "aerials_won": ["AERIALS_WON", "AERIAL_DUELS_WON"],
    "aerials_lost": ["AERIALS_LOST", "AERIAL_DUELS_LOST"], "dribbled_past": ["DRIBBLED_PAST"],
    "err_goal": ["ERROR_LEAD_TO_GOAL", "ERRORS_LEAD_TO_GOAL"], "err_shot": ["ERROR_LEAD_TO_SHOT", "ERRORS_LEAD_TO_SHOT"],
    "pen_missed": ["PENALTIES_MISSES", "PENALTIES_MISSED", "MISSED_PENALTIES"], "saves": ["SAVES"], "conceded": ["GOALS_CONCEDED"],
}
XG_STATS = {"xg": ["EXPECTED_GOALS"], "xgot": ["EXPECTED_GOALS_ON_TARGET"], "xa": ["EXPECTED_ASSISTS"]}
PRE_COLS = ["fixture_id", "team_id", "player_id", "started", "pos", "dpos"]       # known before kick-off (line-up)

# residual metrics: key, kind, numerator, denominator (count: per 90 minutes), label, family
METRICS = [
    ("rating", "level", "rating", None, "Match rating", "Form vs own norm"),
    ("touches", "count", "touches", None, "Touches per 90", "Involvement"),
    ("fouls_drawn", "count", "fouls_drawn", None, "Fouls drawn per 90", "Involvement"),
    ("passes", "count", "passes", None, "Passes per 90", "Passing security"),
    ("pass_acc", "ratio", "acc_passes", "passes", "Pass accuracy", "Passing security"),
    ("long_acc", "ratio", "acc_long_balls", "long_balls", "Long-ball accuracy", "Passing security"),
    ("cross_acc", "ratio", "acc_crosses", "crosses", "Cross accuracy", "Passing security"),
    ("key_passes", "count", "key_passes", None, "Key passes per 90", "Passing risk-taking"),
    ("keypass_rate", "ratio", "key_passes", "passes", "Key passes per pass", "Passing risk-taking"),
    ("final_third_share", "ratio", "final_third_passes", "passes", "Final-third pass share", "Passing risk-taking"),
    ("long_share", "ratio", "long_balls", "passes", "Long-ball share of passes", "Passing risk-taking"),
    ("backward_share", "ratio", "backward_passes", "passes", "Backward ('negative') pass share", "Passing risk-taking"),
    ("drib_att", "count", "drib_att", None, "Dribble attempts per 90", "Dribbling confidence"),
    ("drib_succ", "ratio", "drib_succ", "drib_att", "Dribble success rate", "Dribbling confidence"),
    ("shots", "count", "shots", None, "Shots per 90", "Shooting & finishing"),
    ("shot_acc", "ratio", "sot", "shots", "Shot accuracy (on target / all)", "Shooting & finishing"),
    ("xg_per_shot", "ratio", "xg", "shots", "xG per shot (chance quality)", "Shooting & finishing"),
    ("big_ch_missed", "count", "big_ch_missed", None, "Big chances missed per 90", "Shooting & finishing"),
    ("turnovers", "count", "turnovers", None, "Turnovers per 90", "Ball security"),
    ("turnover_rate", "ratio", "turnovers", "touches", "Turnovers per touch", "Ball security"),
    ("fouls", "count", "fouls", None, "Fouls committed per 90", "Frustration & discipline"),
    ("cards", "count", "cards", None, "Cards per 90", "Frustration & discipline"),
    ("duels", "count", "duels", None, "Duels per 90", "Duels & defending"),
    ("duel_win", "ratio", "duels_won", "duels", "Duel win rate", "Duels & defending"),
    ("aerial_win", "ratio", "aerials_won", "aerials", "Aerial win rate", "Duels & defending"),
    ("def_actions", "count", "def_actions", None, "Tackles + interceptions per 90", "Duels & defending"),
    ("dribbled_past", "count", "dribbled_past", None, "Dribbled past per 90", "Duels & defending"),
    ("offsides", "count", "offsides", None, "Offsides per 90", "Errors & incidents"),
    ("errors", "count", "errors", None, "Errors leading to shot/goal per 90", "Errors & incidents"),
    ("save_pct", "ratio", "saves", "shots_faced", "Save % (keepers)", "Goalkeeping"),
    ("played_to", "level", "played_to", None, "Minute taken off vs usual (starters)", "Coach trust (subs & selection)"),
]
MKEY = {m[0]: m for m in METRICS}
RAW_EW = {   # raw per-appearance values tracked with the same recent/usual windows
    "rating_raw": ("Raw match rating", "Form vs own norm"), "finishing": ("Goals minus xG", "Shooting & finishing"),
    "xg_app": ("xG per appearance", "Shooting & finishing"), "goals_app": ("Goals per appearance", "Shooting & finishing"),
    "woodwork_app": ("Woodwork hits", "Shooting & finishing"), "hooked": ("Hooked before 60' (no injury)", "Coach trust (subs & selection)"),
    "hooked_ht": ("Hooked at half-time", "Coach trust (subs & selection)"), "off_losing": ("Taken off while losing", "Coach trust (subs & selection)"),
    "sub_on": ("Came on as a substitute", "Coach trust (subs & selection)"), "mins_app": ("Minutes per appearance", "Workload & fatigue"),
    "red": ("Red cards", "Frustration & discipline"), "pen_miss": ("Missed penalties", "Errors & incidents"),
}
ROLES = ["GK", "CB", "FB", "D", "DM", "CM", "AM", "W", "ST", "U"]
ROLE_GROUP = {"GK": "GK", "CB": "DEF", "FB": "DEF", "D": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "W": "ATT", "ST": "ATT", "U": "MID"}
TARGETS = {
    "next": ("Next game below his norm", "Will the player's next performance (context-adjusted rating) fall in the bottom quarter relative to his own baseline?"),
    "sustained": ("Sustained downturn", "Will his average over his next K appearances fall in the bottom quarter relative to his own baseline?"),
    "dropped": ("Loses his starting place", "A regular starter (70%+ of the last 10 team matches) starts 2 or fewer of the next 6 (injuries and transfers excluded)."),
}


def role_of(dpos, pos):
    s, p = str(dpos or "").upper(), str(pos or "").upper()
    if "GOAL" in s or "GOAL" in p or p in ("G", "GK"):
        return "GK"
    if s:
        if "CENTRE_BACK" in s or "CENTER_BACK" in s or "CENTRE-BACK" in s:
            return "CB"
        if "BACK" in s:
            return "FB"
        if "DEFENSIVE_MID" in s:
            return "DM"
        if "ATTACKING_MID" in s:
            return "AM"
        if "WING" in s or "LEFT_MID" in s or "RIGHT_MID" in s:
            return "W"
        if "FORWARD" in s or "STRIKER" in s:
            return "ST"
        if "MID" in s:
            return "CM"
        if "DEF" in s:
            return "D"
    if "DEF" in p:
        return "D"
    if "MID" in p:
        return "CM"
    if "ATT" in p or "FORW" in p or "STRIK" in p:
        return "ST"
    return "U"


# =============================================================================
# Extraction
# =============================================================================
FINISHED = "('FT', 'AET', 'FT_PEN')"


def resolve(types, spec, present):
    by = defaultdict(list)
    for r in types.itertuples():
        for k in {str(r.dev or ""), str(r.name or "").replace(" ", "_")}:
            if k:
                by[k.upper()].append(int(r.type_id))
    out = {}
    for k, aliases in spec.items():
        for al in aliases:
            ids = sorted({i for i in by.get(al, []) if i in present})
            if ids:
                out[k] = (al, ids)
                break
    return out


def pivot(table, keys, found, where=None):
    cols = ", ".join(f"max(t.value) FILTER (WHERE t.type_id IN ({','.join(map(str, ids))})) AS {k}" for k, (_, ids) in found.items())
    allids = ",".join(str(i) for _, ids in found.values() for i in ids)
    kk = ", ".join(f"t.{c}" for c in keys)
    extra = f" AND {where}" if where else ""
    return f"SELECT {kk}, {cols} FROM {table} t WHERE t.type_id IN ({allids}) AND t.player_id IS NOT NULL{extra} GROUP BY {kk}"


def extract(db, args):
    types = db.query("SELECT type_id, upper(coalesce(developer_name,'')) AS dev, upper(coalesce(name,'')) AS name FROM core.types")

    def present(tbl):
        if not db.exists(tbl):
            return set()
        s = db.query(f"SELECT DISTINCT type_id FROM {tbl} TABLESAMPLE SYSTEM (5)")
        if len(s) < 5:
            s = db.query(f"SELECT DISTINCT type_id FROM {tbl}")
        return set(s.type_id.dropna().astype(int)) if len(s) else set()
    fs = resolve(types, STATS, present("football.fixture_lineup_details"))
    fx = resolve(types, XG_STATS, present("football.fixture_xg_players"))
    log(f"player stats found: {len(fs)}/{len(STATS)} · player expected-metrics: {len(fx)}/{len(XG_STATS)}")
    if "minutes" not in fs:
        sys.exit("MINUTES_PLAYED not found in football.fixture_lineup_details - load line-up details first.")
    Q = {
        "lineups": """
            SELECT l.fixture_id, l.team_id, l.player_id, (lt.developer_name = 'LINEUP')::int AS started,
                   upper(coalesce(pt.developer_name, pt.name, '')) AS pos, upper(coalesce(dt.developer_name, dt.name, '')) AS dpos
            FROM football.fixture_lineups l
            JOIN football.fixtures f ON f.fixture_id = l.fixture_id
            JOIN core.states st ON st.state_id = f.state_id AND st.developer_name IN """ + FINISHED + """
            LEFT JOIN core.types lt ON lt.type_id = l.type_id
            LEFT JOIN core.types pt ON pt.type_id = l.position_id
            LEFT JOIN core.types dt ON dt.type_id = l.detailed_position_id
            WHERE l.player_id IS NOT NULL AND l.team_id IS NOT NULL""",
        "stats": pivot("football.fixture_lineup_details", ["fixture_id", "player_id"], fs),
        "fixtures": f"""
            WITH sc AS (SELECT fixture_id,
                   max(goals) FILTER (WHERE location='home' AND description='CURRENT') AS h_cur, max(goals) FILTER (WHERE location='away' AND description='CURRENT') AS a_cur,
                   max(goals) FILTER (WHERE location='home' AND description='2ND_HALF') AS h_90, max(goals) FILTER (WHERE location='away' AND description='2ND_HALF') AS a_90
                   FROM football.fixture_scores WHERE description IN ('CURRENT','2ND_HALF') GROUP BY fixture_id)
            SELECT f.fixture_id, f.league_id, l.name AS league, f.season_id, f.starting_at,
                   COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
                   CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.h_90, sc.h_cur) ELSE sc.h_cur END AS hg,
                   CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.a_90, sc.a_cur) ELSE sc.a_cur END AS ag
            FROM football.fixtures f JOIN core.states st ON st.state_id = f.state_id
            LEFT JOIN sc ON sc.fixture_id = f.fixture_id
            LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
            LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
            LEFT JOIN football.leagues l ON l.league_id = f.league_id
            WHERE st.developer_name IN {FINISHED}""",
        "events": """
            SELECT e.fixture_id, e.team_id, e.player_id, e.related_player_id, e.minute, coalesce(e.extra_minute, 0) AS extra,
                   e.result, coalesce(e.injured, false)::int AS injured, upper(t.developer_name) AS kind
            FROM football.fixture_events e JOIN core.types t ON t.type_id = e.type_id
            WHERE upper(t.developer_name) IN ('GOAL','OWNGOAL','PENALTY','SUBSTITUTION','REDCARD','YELLOWREDCARD','YELLOWCARD','MISSED_PENALTY')
              AND e.rescinded IS NOT TRUE AND e.minute IS NOT NULL""",
        "odds": f"""
            WITH o AS (
              SELECT o.fixture_id, o.bookmaker_id, o.decimal_odds AS price,
                     CASE WHEN lower(coalesce(o.label,'')) IN ('home','1') OR lower(coalesce(o.original_label,''))='1' THEN 'H'
                          WHEN lower(coalesce(o.label,'')) IN ('draw','x') OR lower(coalesce(o.original_label,''))='x' THEN 'D'
                          WHEN lower(coalesce(o.label,'')) IN ('away','2') OR lower(coalesce(o.original_label,''))='2' THEN 'A' END AS side
              FROM odds.prematch_odds o JOIN football.fixtures f ON f.fixture_id = o.fixture_id
              WHERE o.market_id = {int(args.market_1x2)} AND o.decimal_odds > 1
                AND (o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at)),
            b AS (SELECT fixture_id, bookmaker_id, avg(1/price) FILTER (WHERE side='H') AS ih, avg(1/price) FILTER (WHERE side='D') AS id,
                         avg(1/price) FILTER (WHERE side='A') AS ia FROM o GROUP BY 1, 2)
            SELECT fixture_id, avg(ih/(ih+id+ia)) AS p_h, avg(id/(ih+id+ia)) AS p_d, avg(ia/(ih+id+ia)) AS p_a
            FROM b WHERE ih IS NOT NULL AND id IS NOT NULL AND ia IS NOT NULL GROUP BY 1""",
        "team_xg": """SELECT x.fixture_id, x.team_id, max(x.value) AS xg FROM football.fixture_xg x JOIN core.types t ON t.type_id = x.type_id
                      WHERE t.developer_name = 'EXPECTED_GOALS' GROUP BY 1, 2""",
        "players": "SELECT player_id, COALESCE(display_name, common_name, name) AS name, date_of_birth AS dob FROM football.players",
        "teams": "SELECT team_id, name FROM football.teams",
        "sidelined": "SELECT player_id, team_id, lower(coalesce(category,'')) AS category, start_date, end_date FROM football.sidelined WHERE player_id IS NOT NULL",
        "transfers": "SELECT player_id, to_team_id, date FROM football.transfers WHERE player_id IS NOT NULL AND date IS NOT NULL",
        "rumours": "SELECT player_id, date FROM football.transfer_rumours WHERE player_id IS NOT NULL AND date IS NOT NULL",
        "coaches": "SELECT fixture_id, team_id, coach_id FROM football.fixture_coaches WHERE team_id IS NOT NULL",
    }
    if fx:
        Q["xg"] = pivot("football.fixture_xg_players", ["fixture_id", "player_id"], fx)
    cache = Path(args.cache) if args.cache else None
    out = {}
    import concurrent.futures as cf
    import threading

    def run(name, sql):
        c = DB.__new__(DB)
        c.dsn, c.kind, c.name, c.conn = db.dsn, db.kind, db.name, None
        if db.kind == "psycopg":
            import psycopg  # type: ignore
            c.conn = psycopg.connect(db.dsn)
        elif db.kind == "psycopg2":
            import psycopg2  # type: ignore
            c.conn = psycopg2.connect(db.dsn)
        if c.conn is not None:
            ACTIVE_CONNS.add(c.conn)
        RUNNING[name] = time.time()
        try:
            return c.query(sql), None
        except Exception as e:  # noqa: BLE001
            return pd.DataFrame(), e
        finally:
            RUNNING.pop(name, None)
            if c.conn is not None:
                ACTIVE_CONNS.discard(c.conn)
                try:
                    c.conn.close()
                except Exception:  # noqa: BLE001
                    pass
    todo = []
    for k, sql in Q.items():
        p = cache / f"form_{k}.pkl" if cache else None
        if p and p.exists() and not args.refresh:
            out[k] = pd.read_pickle(p)
            log(f"{k:10s} {len(out[k]):>11,} rows (cache)")
        else:
            todo.append((k, sql, p))
    weight = {"lineups": 3, "events": 2, "odds": 2}
    todo.sort(key=lambda x: -weight.get(x[0], 1))
    n_tot, n_done = len(todo), 0
    if todo:
        log(f"reading {n_tot} queries with {args.db_jobs} parallel connections - the player stats query is the slow one; Ctrl-C stops cleanly")
    ex = cf.ThreadPoolExecutor(max_workers=max(1, min(n_tot or 1, args.db_jobs)))
    futs = {ex.submit(run, k, sql): (k, p) for k, sql, p in todo}
    pending, last_beat = set(futs), time.time()
    while pending:
        done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)   # short waits keep Ctrl-C responsive
        for f in done:
            k, p = futs[f]
            res, err = f.result()
            n_done += 1
            if err is not None:
                if k in ("lineups", "fixtures", "stats"):
                    stop_queries()
                    sys.exit(f"Required query '{k}' failed: {err}")
                log(f"{k}: not available ({str(err).splitlines()[0][:100]}) - continuing without it")
            out[k] = res
            log(f"{k:10s} {len(res):>11,} rows   [{n_done}/{n_tot} queries done]")
            if p is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                res.to_pickle(p)
        if pending and time.time() - last_beat > 30:
            last_beat = time.time()
            run_ = sorted(RUNNING.items(), key=lambda kv: kv[1])[:4]
            log(f"  still reading: {n_done}/{n_tot} done; running " + ", ".join(f"{k} ({time.time() - t0:.0f}s)" for k, t0 in run_))
    ex.shutdown(wait=False)
    if not len(out["stats"]):
        sys.exit("No player statistics returned from football.fixture_lineup_details.")
    if "xg" not in out:
        out["xg"] = pd.DataFrame()
    apps = out["lineups"].merge(out["stats"], on=["fixture_id", "player_id"], how="left")
    if len(out.get("xg", [])):
        apps = apps.merge(out["xg"], on=["fixture_id", "player_id"], how="left")
    raw = {"apps": apps, "fixtures": out["fixtures"], "events": out["events"], "odds": out["odds"], "team_xg": out["team_xg"],
           "players": out["players"], "teams": out["teams"], "sidelined": out["sidelined"], "transfers": out["transfers"],
           "rumours": out["rumours"], "coaches": out["coaches"],
           "found": {**{k: v[0] for k, v in fs.items()}, **{k: v[0] for k, v in fx.items()}}, "demo": False}
    return raw


# =============================================================================
# Synthetic database (--demo): same structure, with a hidden "confidence" state
# that moves stats a little before ratings follow. Only for checking the
# pipeline and the dashboard - the findings it shows are made up.
# =============================================================================
def demo_data(seed=11, n_leagues=3, n_teams=14, seasons=(2020, 2021, 2022, 2023)):
    rng = np.random.default_rng(seed)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    slots = ["GK"] * 2 + ["CB"] * 4 + ["FB"] * 3 + ["DM"] * 2 + ["CM"] * 3 + ["AM"] * 2 + ["W"] * 3 + ["ST"] * 3
    POS = {"GK": "GOALKEEPER", "CB": "DEFENDER", "FB": "DEFENDER", "DM": "MIDFIELDER", "CM": "MIDFIELDER", "AM": "MIDFIELDER", "W": "ATTACKER", "ST": "ATTACKER"}
    DPOS = {"GK": "GOALKEEPER", "CB": "CENTRE_BACK", "FB": "LEFT_BACK", "DM": "DEFENSIVE_MIDFIELD", "CM": "CENTRAL_MIDFIELD", "AM": "ATTACKING_MIDFIELD",
            "W": "LEFT_WING", "ST": "CENTRE_FORWARD"}
    XI = ["GK", "CB", "CB", "FB", "FB", "DM", "CM", "AM", "W", "W", "ST"]
    RATE = {  # per-90 base rates by role: touches, drib, shots, keyp, fouls, duels, def
        "GK": (35, .1, 0, .05, .1, 2, .3), "CB": (65, .3, .5, .2, 1.1, 8, 3.5), "FB": (70, 1.0, .5, .8, 1.2, 9, 3), "DM": (75, .7, .8, .8, 1.5, 10, 3.5),
        "CM": (70, 1.0, 1.0, 1.2, 1.2, 9, 2.5), "AM": (60, 2.0, 2.0, 2.0, 1.0, 9, 1.5), "W": (55, 3.0, 2.0, 1.5, .9, 11, 1.2), "ST": (40, 1.5, 3.0, 1.0, 1.4, 12, .8)}
    teams, players, pid = [], {}, 1000
    for lg in range(n_leagues):
        for k in range(n_teams):
            tid = 100 * (lg + 1) + k
            teams.append((tid, lg, rng.normal(0, .35)))
            for r in slots:
                players[pid] = {"team": tid, "role": r, "ab": rng.normal(0, .35), "c": 0.0, "out_until": -1, "dob": pd.Timestamp(rng.integers(1988, 2004), rng.integers(1, 13), 1)}
                pid += 1
    strength = {t: s for t, _, s in teams}
    squad = defaultdict(list)
    for p, v in players.items():
        squad[v["team"]].append(p)
    apps, fixtures, events, odds, sidel, transfers, rumours, coaches, txg = [], [], [], [], [], [], [], [], []
    fid, coach_of, next_coach = 1, {t: 5000 + t for t, _, _ in teams}, 9000
    for season in seasons:
        # summer transfers inside each league
        if season != seasons[0]:
            for p in rng.choice(list(players), size=len(players) // 12, replace=False):
                v = players[p]
                lg_t = [t for t, l, _ in teams if l == (v["team"] // 100 - 1) and t != v["team"]]
                to = int(rng.choice(lg_t))
                squad[v["team"]].remove(p)
                squad[to].append(p)
                v["team"] = to
                transfers.append((p, to, pd.Timestamp(season, 7, int(rng.integers(1, 28)), tz="UTC")))
        for lg in range(n_leagues):
            ts = [t for t, l, _ in teams if l == lg]
            n = len(ts)
            rounds = []
            arr = ts[:]
            for r in range(n - 1):
                rounds.append([(arr[i], arr[n - 1 - i]) for i in range(n // 2)])
                arr = [arr[0]] + [arr[-1]] + arr[1:-1]
            rounds += [[(b, a) for a, b in rd] for rd in rounds]
            for ri, rd in enumerate(rounds):
                day = pd.Timestamp(season, 8, 10) + pd.Timedelta(days=7 * ri + (ri // 19) * 21 + lg)
                if ri == len(rounds) // 2:
                    for t in ts:
                        if rng.random() < .15:
                            coach_of[t] = next_coach
                            next_coach += 1
                for h, a in rd:
                    ko = pd.Timestamp(day.year, day.month, day.day, int(rng.choice([13, 15, 17, 20])), tz="UTC")
                    dn = (day - DAY0).days
                    lineup = {}
                    for t in (h, a):
                        avail = [p for p in squad[t] if players[p]["out_until"] < dn]
                        for p in avail:   # confidence: persistent, with occasional slumps and recoveries
                            v = players[p]
                            v["c"] = .88 * v["c"] + rng.normal(0, .28) + (-1.1 if rng.random() < .02 else 0) + (.8 if rng.random() < .015 else 0)
                        xi = []
                        for r in XI:
                            cand = [p for p in avail if players[p]["role"] == r and p not in xi] or [p for p in avail if p not in xi]
                            if not cand:
                                continue
                            sc = [players[p]["ab"] + .35 * players[p]["c"] + rng.normal(0, .25) for p in cand]
                            xi.append(cand[int(np.argmax(sc))])
                        bench = [p for p in avail if p not in xi][:9]
                        lineup[t] = (xi, bench)
                        coaches.append((fid, t, coach_of[t]))
                    sh, sa = strength[h] + .15, strength[a]
                    p_h = sig(1.1 * (sh - sa) + .05) * .78
                    p_a = (1 - p_h) * sig(-1.2 * (sh - sa) - .1) * 1.05
                    p_a = min(p_a, 1 - p_h - .15)
                    odds.append((fid, p_h, 1 - p_h - p_a, p_a))
                    rows_t, goals_t = {}, []
                    for t, o, s_, so in ((h, a, sh, sa), (a, h, sa, sh)):
                        xi, bench = lineup[t]
                        perf = {p: players[p]["ab"] + .6 * players[p]["c"] + rng.normal(0, .5) for p in xi}
                        nsub = int(rng.integers(2, 6))
                        outs = sorted(xi[1:], key=lambda p: perf[p])[:nsub]
                        on_m, off_m, inj = {p: 0 for p in xi}, {p: 90 for p in xi}, {}
                        for k_, p in enumerate(outs):
                            if not bench:
                                break
                            m = int(np.clip(rng.normal(66 - 8 * (perf[p] < -.6), 11), 30, 89))
                            if rng.random() < .07:
                                m = int(rng.integers(5, 85))
                                inj[p] = 1
                                players[p]["out_until"] = dn + int(rng.integers(8, 50))
                                sidel.append((p, t, "injury", pd.Timestamp(day, tz="UTC"), pd.Timestamp(day + pd.Timedelta(days=int(players[p]["out_until"] - dn)), tz="UTC")))
                            same = [b for b in bench if players[b]["role"] == players[p]["role"]] or bench
                            q = same[0]
                            bench.remove(q)
                            off_m[p], on_m[q] = m, m
                            events.append((fid, t, q, p, m, 0, None, inj.get(p, 0), "SUBSTITUTION"))
                        for p in list(on_m):
                            v = players[p]
                            r, c, ab = v["role"], v["c"], v["ab"]
                            mins = off_m.get(p, 90) - on_m[p]
                            if mins <= 0:
                                continue
                            f90 = mins / 90
                            T, D, S, K, F, U, A = RATE[r]
                            team_f = np.exp(.25 * (s_ - so))
                            tch = rng.poisson(T * f90 * team_f * np.exp(.12 * c))
                            ps = int(tch * .78)
                            pacc = rng.binomial(ps, sig(1.6 + .28 * c + .25 * ab + .3 * (s_ - so)))
                            bwd = rng.binomial(ps, sig(-1.3 - .25 * c))
                            datt = rng.poisson(D * f90 * np.exp(.3 * c))
                            dsuc = rng.binomial(datt, sig(-.1 + .45 * c + .3 * ab))
                            shots = rng.poisson(S * f90 * team_f * np.exp(.2 * c + .2 * ab))
                            xg = float(np.sum(rng.gamma(1.2, .085, shots))) * np.exp(.15 * ab)
                            sot = rng.binomial(shots, sig(-.4 + .4 * c))
                            g = min(rng.poisson(xg * np.exp(.15 * c)), sot) if shots else 0
                            fouls = rng.poisson(F * f90 * np.exp(-.3 * c))
                            duels = rng.poisson(U * f90)
                            dwon = rng.binomial(duels, sig(.35 * c + .3 * ab))
                            disp = rng.poisson(max(tch, 1) * .035 * np.exp(-.35 * c))
                            rows_t[p] = dict(fixture_id=fid, team_id=t, player_id=p, started=int(on_m[p] == 0), pos=POS[r], dpos=DPOS[r],
                                             minutes=mins, touches=tch, passes=ps, acc_passes=pacc, backward_passes=bwd, key_passes=rng.poisson(K * f90 * np.exp(.25 * c)),
                                             drib_att=datt, drib_succ=dsuc, shots=shots, sot=sot, goals=g, xg=round(xg, 3), fouls=fouls,
                                             fouls_drawn=rng.poisson(1.0 * f90), duels=duels, duels_won=dwon, dispossessed=disp,
                                             tackles=rng.poisson(A * .55 * f90), interceptions=rng.poisson(A * .45 * f90), assists=0,
                                             big_ch_missed=rng.poisson(.15 * S * f90 * np.exp(-.3 * c)), saves=rng.poisson(2.5) if r == "GK" else 0,
                                             _c=c, _ab=ab, _mins=(on_m[p], off_m.get(p, 90)))
                            for _ in range(g):
                                goals_t.append((rng.uniform(on_m[p] + .5, off_m.get(p, 90) - .1), t, p))
                        for p in xi + list(on_m):
                            if p in rows_t and rng.random() < .09 * np.exp(-.3 * players[p]["c"]):
                                events.append((fid, t, p, None, int(rng.uniform(*rows_t[p]["_mins"])), 0, None, 0, "YELLOWCARD"))
                        for b in bench:   # unused substitutes are on the team sheet too
                            rows_t[b] = dict(fixture_id=fid, team_id=t, player_id=b, started=0, pos=POS[players[b]["role"]], dpos=DPOS[players[b]["role"]], minutes=None)
                    goals_t.sort()
                    hg_ = ag_ = 0
                    for tm, t, p in goals_t:
                        hg_, ag_ = hg_ + (t == h), ag_ + (t == a)
                        events.append((fid, t, p, None, int(tm), 0, f"{hg_}-{ag_}", 0, "GOAL"))
                    for t, gf, ga in ((h, hg_, ag_), (a, ag_, hg_)):
                        txg.append((fid, t, sum(r_.get("xg", 0) or 0 for r_ in rows_t.values() if r_["team_id"] == t)))
                    for p, r_ in rows_t.items():
                        if r_.get("minutes"):
                            t = r_["team_id"]
                            gd = (hg_ - ag_) * (1 if t == h else -1)
                            rr = 6.55 + .3 * r_.pop("_ab") + .3 * r_.pop("_c") + .55 * r_["goals"] + .08 * np.sign(gd) + rng.normal(0, .33)
                            r_["rating"] = round(float(np.clip(rr, 4, 10)), 1)
                            r_.pop("_mins")
                        apps.append(r_)
                    fixtures.append((fid, 1000 + lg, f"Demo League {lg + 1}", season * 10 + lg, ko, h, a, hg_, ag_))
                    fid += 1
        for p, v in players.items():
            if rng.random() < .08:
                rumours.append((p, pd.Timestamp(season + 1, int(rng.integers(1, 6)), int(rng.integers(1, 28)), tz="UTC")))
    A = pd.DataFrame(apps)
    raw = {"apps": A, "fixtures": pd.DataFrame(fixtures, columns=["fixture_id", "league_id", "league", "season_id", "starting_at", "home_id", "away_id", "hg", "ag"]),
           "events": pd.DataFrame(events, columns=["fixture_id", "team_id", "player_id", "related_player_id", "minute", "extra", "result", "injured", "kind"]),
           "odds": pd.DataFrame(odds, columns=["fixture_id", "p_h", "p_d", "p_a"]), "team_xg": pd.DataFrame(txg, columns=["fixture_id", "team_id", "xg"]),
           "players": pd.DataFrame([(p, f"Demo player {p}", v["dob"]) for p, v in players.items()], columns=["player_id", "name", "dob"]),
           "teams": pd.DataFrame([(t, f"Demo FC {t}") for t, _, _ in teams], columns=["team_id", "name"]),
           "sidelined": pd.DataFrame(sidel, columns=["player_id", "team_id", "category", "start_date", "end_date"]),
           "transfers": pd.DataFrame(transfers, columns=["player_id", "to_team_id", "date"]),
           "rumours": pd.DataFrame(rumours, columns=["player_id", "date"]), "coaches": pd.DataFrame(coaches, columns=["fixture_id", "team_id", "coach_id"]),
           "found": {k: STATS.get(k, XG_STATS.get(k, [k]))[0] for k in A.columns if k in STATS or k in XG_STATS}, "demo": True}
    return raw


# =============================================================================
# Raw preparation (decisions about the data FORMAT, made once on all data)
# =============================================================================
def to_day(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert(None).dt.normalize()


def prepare_raw(raw):
    """Orientation of substitution events (which column is the player going off) is a property of the data
    format, so it is decided once from all data and then fixed; everything else stays raw."""
    ev = raw["events"]
    sub = ev[ev.kind == "SUBSTITUTION"]
    st = raw["apps"].loc[raw["apps"].started == 1, ["fixture_id", "player_id"]]
    sset = set(zip(st.fixture_id.astype("int64"), st.player_id.astype("int64")))
    a = np.mean([(int(f), int(p)) in sset for f, p in zip(sub.fixture_id, sub.player_id) if pd.notna(p)]) if len(sub) else 0.0
    raw["sub_out_col"] = "player_id" if a > 0.5 else "related_player_id"
    log(f"substitution events: '{raw['sub_out_col']}' is the player going off ({a:.0%} of player_id are starters)")
    for k in ("sidelined", "transfers", "rumours"):
        if k not in raw or raw[k] is None:
            raw[k] = pd.DataFrame()
    return raw


# =============================================================================
# Point-in-time, time-decayed group sums (the heart of the leak-proof normalisation)
# =============================================================================
def kc(*codes):
    """Combine small integer codes into one int64 key."""
    k = np.zeros(len(codes[0]), np.int64)
    for c in codes:
        k = k * 1_000_003 + (np.asarray(c, np.int64) + 1)
    return k


def decayed(obs_key, obs_t, V, q_key, q_t, tau):
    """For each query (key, day): sum over observations with the same key on EARLIER days of
    V * exp(-(day - obs_day) / tau). Observations on the query day itself are never included."""
    V = np.nan_to_num(np.asarray(V, np.float64))
    m = V.shape[1]
    df = pd.DataFrame(V, columns=[f"v{i}" for i in range(m)])
    df["k"], df["t"] = np.asarray(obs_key, np.int64), np.asarray(obs_t, np.float64)
    g = df.groupby(["k", "t"], sort=True).sum()
    k = g.index.get_level_values(0).to_numpy(np.int64)
    tt = g.index.get_level_values(1).to_numpy(np.float64)
    W = g.to_numpy(np.float64)
    t0 = pd.Series(tt).groupby(k).transform("min").to_numpy()
    e = np.exp((tt - t0) / tau)
    incl = pd.DataFrame(W * e[:, None]).groupby(k).cumsum().to_numpy() / e[:, None]
    obs = pd.DataFrame({"k": k, "t": tt, "tobs": tt, "i": np.arange(len(k))}).sort_values("t", kind="stable")
    q = pd.DataFrame({"k": np.asarray(q_key, np.int64), "t": np.asarray(q_t, np.float64), "qi": np.arange(len(q_key))}).sort_values("t", kind="stable")
    mg = pd.merge_asof(q, obs, on="t", by="k", allow_exact_matches=False, direction="backward")
    out = np.zeros((len(q_key), m))
    ii, qi = mg["i"].to_numpy(), mg["qi"].to_numpy()
    good = np.isfinite(ii)
    gi = ii[good].astype(np.int64)
    out[qi[good]] = incl[gi] * np.exp(-(mg["t"].to_numpy()[good] - mg["tobs"].to_numpy()[good]) / tau)[:, None]
    return out


def ew_prior(pcode, pos, X, alphas):
    """Exponentially weighted means of each row's EARLIER rows for the same entity (missing values skipped).
    pcode: entity code per row; pos: 0,1,2.. order of the row within its entity."""
    n, m = X.shape
    P = int(pcode.max()) + 1 if n else 1
    order = np.argsort(pos, kind="stable")
    ps = pos[order]
    bounds = np.searchsorted(ps, np.arange(int(ps.max()) + 2 if n else 1))
    outs = {a: np.full((n, m), np.nan, np.float32) for a in alphas}
    S = {a: np.zeros((P, m)) for a in alphas}
    W = {a: np.zeros((P, m)) for a in alphas}
    for k in range(len(bounds) - 1):
        rows = order[bounds[k]:bounds[k + 1]]
        if not len(rows):
            continue
        p = pcode[rows]
        x = X[rows]
        ok = np.isfinite(x)
        xz = np.where(ok, x, 0.0)
        for a in alphas:
            s, w = S[a][p], W[a][p]
            outs[a][rows] = np.where(w > 1e-9, s / np.where(w > 1e-9, w, 1), np.nan)
            S[a][p] = np.where(ok, (1 - a) * s + a * xz, s)
            W[a][p] = np.where(ok, (1 - a) * w + a, w)
    return outs


def window_count(key_sorted, vals_cum, key_q, lo_off, hi_off):
    """Sum of values whose key lies in [key_q + lo_off, key_q + hi_off) (keys sorted ascending)."""
    lo = np.searchsorted(key_sorted, key_q + lo_off, side="left")
    hi = np.searchsorted(key_sorted, key_q + hi_off, side="left")
    return vals_cum[hi] - vals_cum[lo]


# =============================================================================
# Build: appearances with match context -> context-adjusted residuals -> features
# =============================================================================
def build_base(raw, cutoff=None, ghosts=True):
    fx = raw["fixtures"].dropna(subset=["home_id", "away_id", "starting_at"]).copy()
    fx["day"] = to_day(fx.starting_at)
    fx = fx.dropna(subset=["day"])
    apps = raw["apps"].copy()
    ev = raw["events"].copy()
    side = {k: (raw.get(k) if raw.get(k) is not None else pd.DataFrame()).copy() for k in ("sidelined", "transfers", "rumours")}
    if cutoff is not None:
        # time travel: the world ends on the cutoff day, and that day's matches have not been played yet
        fx = fx[fx.day <= cutoff]
        today = set(fx.fixture_id[fx.day == cutoff].astype("int64"))
        fx.loc[fx.day == cutoff, ["hg", "ag"]] = np.nan
        apps = apps[apps.fixture_id.isin(fx.fixture_id)].copy()
        post = [c for c in apps.columns if c not in PRE_COLS]
        apps.loc[apps.fixture_id.isin(today), post] = np.nan
        ev = ev[ev.fixture_id.isin(fx.fixture_id) & ~ev.fixture_id.isin(today)]
        end = pd.Timestamp(cutoff)
        if len(side["sidelined"]):
            s = side["sidelined"]
            s = s[to_day(s.start_date) <= end].copy()
            s.loc[to_day(s.end_date) > end, "end_date"] = pd.NaT
            side["sidelined"] = s
        for k in ("transfers", "rumours"):
            if len(side[k]):
                side[k] = side[k][to_day(side[k].date) <= end]
    fx["t"] = (fx.day - DAY0).dt.days.astype(float)
    d = apps.merge(fx[["fixture_id", "league_id", "league", "season_id", "day", "t", "home_id", "away_id", "hg", "ag"]], on="fixture_id")
    d = d[(d.team_id == d.home_id) | (d.team_id == d.away_id)].copy()
    for c in ("fixture_id", "team_id", "player_id"):
        d[c] = d[c].astype("int64")
    d["home"] = (d.team_id == d.home_id).astype(float)
    d["opp_id"] = np.where(d.home == 1, d.away_id, d.home_id).astype("int64")
    d["gf"] = np.where(d.home == 1, d.hg, d.ag)
    d["ga"] = np.where(d.home == 1, d.ag, d.hg)
    d["started"] = d.started.fillna(0).astype(float)
    d["role"] = [role_of(a, b) for a, b in zip(d.dpos.fillna(""), d.pos.fillna(""))]
    for k in list(STATS) + list(XG_STATS):
        if k not in d:
            d[k] = np.nan
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d["ghost"] = 0
    # ---------------- events: substitutions, cards, penalties, score timeline
    ev = ev.dropna(subset=["minute"]).copy()
    ev["tm"] = ev.minute.astype(float) + np.minimum(ev.extra.fillna(0).astype(float), 9) * 0.1
    outc = raw.get("sub_out_col", "related_player_id")
    inc = "player_id" if outc == "related_player_id" else "related_player_id"
    sub = ev[ev.kind == "SUBSTITUTION"]
    off = sub.dropna(subset=[outc]).groupby(["fixture_id", outc]).agg(off_t=("tm", "min"), off_inj=("injured", "max")).reset_index().rename(columns={outc: "player_id"})
    on = sub.dropna(subset=[inc]).groupby(["fixture_id", inc]).agg(on_t=("tm", "min")).reset_index().rename(columns={inc: "player_id"})
    reds = ev[ev.kind.isin(["REDCARD", "YELLOWREDCARD"])].dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).agg(red_t=("tm", "min"), red=("tm", "size")).reset_index()
    yel = ev[ev.kind == "YELLOWCARD"].dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).size().rename("yel_ev").reset_index()
    pm = ev[ev.kind == "MISSED_PENALTY"].dropna(subset=["player_id"]).groupby(["fixture_id", "player_id"]).size().rename("pen_miss_ev").reset_index()
    for t_ in (off, on, reds, yel, pm):
        t_["fixture_id"] = t_.fixture_id.astype("int64")
        t_["player_id"] = t_.player_id.astype("int64")
        d = d.merge(t_, on=["fixture_id", "player_id"], how="left")
    has_subs = set(sub.fixture_id.astype("int64"))
    d["subs_known"] = d.fixture_id.isin(has_subs).astype(float)
    d["red"] = d.red.fillna(0).where(d.minutes.notna())
    d["cards"] = (d.yel_ev.fillna(0) + 2 * d.red).where(d.minutes.notna())
    d["pen_miss"] = d.pen_miss_ev.fillna(0).where(d.minutes.notna())
    if d.pen_missed.notna().any():
        d["pen_miss"] = d.pen_missed.where(d.pen_missed.notna(), d.pen_miss)
    played = d.minutes.fillna(0) > 0
    endt = ev.groupby("fixture_id").tm.max().clip(lower=90)
    d["end_t"] = d.fixture_id.map(endt).fillna(90.0).clip(upper=125)
    d["on"] = np.where(d.started == 1, 0.0, d.on_t)
    d.loc[played & d.on.isna(), "on"] = (d.end_t - d.minutes).clip(lower=0)
    d["off"] = np.fmin(d.off_t, d.red_t)
    d["off"] = d.off.where(d.off.notna(), d.end_t)
    d.loc[~played, ["on", "off"]] = np.nan
    # score timeline from goal events (90 minutes); fixtures whose goals don't add up get no game state
    gl = ev[ev.kind.isin(["GOAL", "OWNGOAL", "PENALTY"]) & (ev.tm < 91)].copy()
    gl = gl.merge(fx[["fixture_id", "home_id", "hg", "ag"]], on="fixture_id")
    gl = gl.sort_values(["fixture_id", "tm"], kind="stable")
    hinc = ((gl.team_id == gl.home_id) != (gl.kind == "OWNGOAL")).astype(int)
    gl["ch"] = hinc.groupby(gl.fixture_id).cumsum()
    gl["ca"] = (1 - hinc).groupby(gl.fixture_id).cumsum()
    pr = gl.result.astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$").astype(float)
    use = pr.notna().all(axis=1)
    gl["sh"] = np.where(use, pr[0], gl.ch)
    gl["sa"] = np.where(use, pr[1], gl.ca)
    last = gl.groupby("fixture_id")[["sh", "sa"]].last()
    fxs = fx.set_index("fixture_id")[["hg", "ag"]]
    tot = gl.groupby("fixture_id").size()
    okfx = set(fxs.index[(fxs.hg.fillna(-1) == 0) & (fxs.ag.fillna(-1) == 0) & ~fxs.index.isin(tot.index)])
    chk = last.join(fxs, how="inner")
    okfx |= set(chk.index[(chk.sh == chk.hg) & (chk.sa == chk.ag)])
    seg = pd.concat([pd.DataFrame({"fixture_id": list(okfx), "s0": 0.0, "dh": 0.0}),
                     pd.DataFrame({"fixture_id": gl.fixture_id, "s0": gl.tm, "dh": gl.sh - gl.sa})[gl.fixture_id.isin(okfx)]])
    seg = seg.sort_values(["fixture_id", "s0"], kind="stable")
    seg["s1"] = seg.groupby("fixture_id").s0.shift(-1).fillna(200.0)
    seg["fixture_id"] = seg.fixture_id.astype("int64")
    d = d.reset_index(drop=True)
    d["rid"] = np.arange(len(d))
    okfx = sorted(int(x) for x in okfx)
    R = d.loc[played & d.fixture_id.isin(okfx), ["rid", "fixture_id", "on", "off", "home"]]
    M = R.merge(seg, on="fixture_id")
    ov = np.clip(np.minimum(M.off, M.s1) - np.maximum(M.on, M.s0), 0, None)
    dd = M.dh * (2 * M.home - 1)
    agg = pd.DataFrame({"rid": M.rid, "w": ov * (dd > 0), "dr": ov * (dd == 0), "l": ov * (dd < 0), "tot": ov}).groupby("rid").sum()
    for c in ("w", "dr", "l"):
        d[f"sh_{c}"] = d.rid.map(agg[c] / agg.tot.replace(0, np.nan))
    at_off = M[(M.s0 <= M.off) & (M.off < M.s1)]
    d["gd_at_off"] = d.rid.map(pd.Series((at_off.dh * (2 * at_off.home - 1)).to_numpy(), index=at_off.rid.to_numpy()))
    sh = d[["sh_w", "sh_dr", "sh_l"]].to_numpy()
    dom = np.where(np.nanmax(np.nan_to_num(sh, nan=-1), axis=1) >= 0.6, np.nanargmax(np.nan_to_num(sh, nan=-1), axis=1), -1)
    d["state_bucket"] = np.where(np.isfinite(sh).all(axis=1), dom, -1)
    # ---------------- derived counts
    def s(c):
        return d[c] if d[c].notna().any() else None
    if s("poss_lost") is not None:
        d["turnovers"] = d.poss_lost
    elif s("dispossessed") is not None:
        d["turnovers"] = d.dispossessed + (d.drib_att - d.drib_succ).clip(lower=0).fillna(0)
    else:
        d["turnovers"] = np.nan
    d["def_actions"] = d[["tackles", "interceptions"]].sum(axis=1, min_count=1)
    d["errors"] = d[["err_goal", "err_shot"]].sum(axis=1, min_count=1)
    d["aerials"] = d[["aerials_won", "aerials_lost"]].sum(axis=1, min_count=2)
    d["shots_faced"] = d[["saves", "conceded"]].sum(axis=1, min_count=2)
    d.loc[d.role != "GK", ["saves", "shots_faced"]] = np.nan
    pl_to = np.where(d.off_t.notna() | d.red_t.notna(), d.off, 90.0)
    valid_pt = (d.started == 1) & played & (d.subs_known == 1) & (d.off_inj.fillna(0) == 0) & d.red_t.isna()
    d["played_to"] = np.where(valid_pt, pl_to, np.nan)
    sub_off = (d.started == 1) & d.off_t.notna() & (d.off_inj.fillna(0) == 0)
    known = played & (d.subs_known == 1) & (d.started == 1)
    d["hooked"] = np.where(known, (sub_off & (d.off_t < 60)).astype(float), np.nan)
    d["hooked_ht"] = np.where(known, (sub_off & (d.off_t <= 46)).astype(float), np.nan)
    d["off_losing"] = np.where(known & d.gd_at_off.notna() | (known & ~sub_off), (sub_off & (d.gd_at_off < 0)).astype(float), np.nan)
    d["sub_on"] = np.where(played, (d.started == 0).astype(float), np.nan)
    d["rating_raw"] = d.rating
    d["finishing"] = d.goals - d.xg
    d["xg_app"], d["goals_app"], d["woodwork_app"] = d.xg, d.goals, d.woodwork
    d["mins_app"] = d.minutes.where(played)
    # ---------------- ghost rows: "before each active player's next match", for the watchlist
    if ghosts and cutoff is None and len(d):
        tmax = d.t.max()
        lastp = d[played].sort_values("t").groupby("player_id").tail(1)
        lastp = lastp[lastp.t >= tmax - 60]
        g = pd.DataFrame({"fixture_id": -1 - lastp.team_id.to_numpy(), "team_id": lastp.team_id.to_numpy(), "player_id": lastp.player_id.to_numpy(),
                          "league_id": lastp.league_id.to_numpy(), "league": lastp.league.to_numpy(), "season_id": lastp.season_id.to_numpy(),
                          "day": DAY0 + pd.Timedelta(days=float(tmax) + 1), "t": tmax + 1, "role": lastp.role.to_numpy(), "ghost": 1,
                          "opp_id": -1, "home": np.nan, "started": np.nan})
        d = pd.concat([d, g], ignore_index=True)
    d = d.sort_values(["player_id", "t", "fixture_id"], kind="stable").reset_index(drop=True)
    side["fx"] = fx
    return d, side


def normalise(d, args, fixed=None):
    """Adds z_<metric>: actual vs context-expected, with every baseline built from earlier days only."""
    played = (d.minutes.fillna(0) >= 15).to_numpy()
    n = len(d)
    lg = pd.factorize(d.league_id)[0]
    rc = d.role.map({r: i for i, r in enumerate(ROLES)}).fillna(len(ROLES) - 1).to_numpy(np.int64)
    teams = pd.factorize(pd.concat([d.team_id, d.opp_id]))[0]
    tc, oc = teams[:n], teams[n:]
    hm = d.home.fillna(-1).to_numpy(np.int64)
    sb = d.state_bucket.fillna(-1).to_numpy(np.int64)
    t = d.t.to_numpy(float)
    shares = d[["sh_w", "sh_dr", "sh_l"]].to_numpy(float)
    has_sh = np.isfinite(shares).all(axis=1)
    keys = {"base": (kc(lg, rc), kc(lg, rc), 365.0), "glob": (kc(rc), kc(rc), 730.0), "team": (kc(tc, rc), kc(tc, rc), 150.0),
            "opp": (kc(oc, rc), kc(oc, rc), 150.0), "venue": (kc(rc, hm), kc(rc, hm), 730.0)}
    sobs = kc(rc, sb)
    avail = [] if fixed is None else list(fixed)
    for key, kind, num, den, label, fam in (METRICS if fixed is None else []):
        if num not in d or d[num].notna().mean() < 0.03 or (den and (den not in d or d[den].notna().mean() < 0.03)):
            continue
        avail.append(key)
    log(f"normalising {len(avail)} metrics against league x position, team, opponent, venue and game state ...")
    cov = {}
    for chunk in [avail[i:i + 6] for i in range(0, len(avail), 6)]:
        Ns, Ds, Cs, valid = [], [], [], []
        for key in chunk:
            _, kind, num, den, _, _ = MKEY[key]
            N = d[num].to_numpy(float)
            if kind == "count":
                D = d.minutes.to_numpy(float) / 90
                ok = played & np.isfinite(N)
            elif kind == "ratio":
                D = d[den].to_numpy(float)
                ok = played & np.isfinite(N) & np.isfinite(D) & (D >= 1)
            else:
                D = np.ones(n)
                ok = np.isfinite(N) & (played if key != "played_to" else True)
            ok &= d.ghost.to_numpy() == 0
            Ns.append(np.where(ok, N, 0))
            Ds.append(np.where(ok, D, 0))
            Cs.append(ok.astype(float))
            valid.append((ok, N, D))
            cov[key] = float(ok.sum() / max(1, ((d.minutes.fillna(0) > 0) & (d.ghost == 0)).sum()))
        V = np.column_stack(Ns + Ds + Cs)
        m = len(chunk)
        L = {nm: decayed(ok_k, t, V, q_k, t, tau) for nm, (ok_k, q_k, tau) in keys.items()}
        ST = [decayed(sobs, t, V, kc(rc, np.full(n, b)), t, 730.0) for b in range(3)]
        for j, key in enumerate(chunk):
            _, kind, _, _, _, _ = MKEY[key]
            ok, N, D = valid[j]
            gN, gD, gC = L["glob"][:, j], L["glob"][:, m + j], L["glob"][:, 2 * m + j]
            with np.errstate(all="ignore"):
                g = gN / gD
                dbar = np.where(gC > 0, gD / gC, np.nan)

                def shrink(S, target, K):
                    return (S[:, j] + K * dbar * target) / (S[:, m + j] + K * dbar)
                base = shrink(L["base"], g, 30)
                team = shrink(L["team"], base, 8)
                opp = shrink(L["opp"], base, 8)
                ven = shrink(L["venue"], g, 50)
                sts = np.column_stack([shrink(ST[b], g, 50) for b in range(3)])
                if kind == "count":
                    f = lambda a, b: np.clip(np.where(b > 0, a / b, 1), 0.4, 2.5)  # noqa: E731
                    sf = np.where(has_sh, (np.nan_to_num(shares) * np.nan_to_num(f(sts, g[:, None]), nan=1.0)).sum(axis=1), 1.0)
                    exp = base * f(team, base) * f(opp, base) * f(ven, g) * sf
                    z = np.log((N + 1) / (exp * D + 1))
                else:
                    sa = np.where(has_sh, (np.nan_to_num(shares) * np.nan_to_num(sts - g[:, None])).sum(axis=1), 0.0)
                    exp = base + (team - base) + (opp - base) + (ven - g) + sa
                    if kind == "ratio":
                        e = np.clip(exp, 0.02, 0.98)
                        z = (N - e * D) / np.sqrt(D * e * (1 - e) + 1)
                    else:
                        z = N - exp
            d[f"z_{key}"] = np.where(ok & np.isfinite(z), np.clip(z, -6, 6), np.nan).astype(np.float32)
    return avail, cov


def features(d, side, avail, args):
    """Player history features (earlier appearances only) plus pre-match facts about the match itself."""
    n = len(d)
    pcode = pd.factorize(d.player_id)[0]
    pos = d.groupby("player_id").cumcount().to_numpy()
    t = d.t.to_numpy(float)
    F = {}
    ew_cols = [f"z_{k}" for k in avail] + [c for c in RAW_EW if c in d and d[c].notna().any()]
    X = d[ew_cols].to_numpy(float)
    aS, aL = 1 - 0.5 ** (1 / 3), 1 - 0.5 ** (1 / 15)
    E = ew_prior(pcode, pos, X, [aS, aL, 1.0])
    for i, c in enumerate(ew_cols):
        b = c[2:] if c.startswith("z_") else c
        F[f"{b}_s"], F[f"{b}_l"] = E[aS][:, i], E[aL][:, i]
        F[f"{b}_d"] = E[aS][:, i] - E[aL][:, i]
    if "z_rating" in d:
        r = d.z_rating.to_numpy(float)
        F["rating_1"] = E[1.0][:, ew_cols.index("z_rating")]
        sq = ew_prior(pcode, pos, (r ** 2)[:, None], [aL])[aL][:, 0]
        F["rating_vol"] = np.sqrt(np.clip(sq - F["rating_l"].astype(float) ** 2, 0, None))
        F["n_rated_prior"] = pd.Series(np.isfinite(r).astype(int)).groupby(pcode).cumsum().to_numpy() - np.isfinite(r)
    for c in ("hooked", "off_losing", "red", "pen_miss"):
        if c in ew_cols:
            F[f"{c}_1"] = E[1.0][:, ew_cols.index(c)]
    # ---- appearance counts, rest, workload (earlier days only)
    played = (d.minutes.fillna(0) > 0).to_numpy()
    F["apps_prior"] = pd.Series(played.astype(int)).groupby(pcode).cumsum().to_numpy() - played
    key = pcode.astype(np.int64) * 100000 + t.astype(np.int64)
    mins = d.minutes.fillna(0).to_numpy(float)
    cm, ca = np.concatenate([[0], np.cumsum(mins)]), np.concatenate([[0], np.cumsum(played.astype(float))])
    for w in (7, 14, 28):
        F[f"mins_{w}d"] = window_count(key, cm, key, -w, 0)
    F["apps_14d"] = window_count(key, ca, key, -14, 0)
    lastt = pd.Series(np.where(played, t, np.nan)).groupby(pcode).transform(lambda s: s.shift(1).ffill()).to_numpy()
    F["rest_days"] = np.clip(t - lastt, 0, 60)
    # ---- drought: appearances and xG since the player's last goal
    if d.goals.notna().any():
        gls = d.goals.fillna(0).to_numpy(float)
        xgv = d.xg.fillna(0).to_numpy(float) if d.xg.notna().any() else d.shots.fillna(0).to_numpy(float) * 0.1
        cx = pd.Series(xgv).groupby(pcode).cumsum().to_numpy()
        at_goal = pd.Series(np.where(gls > 0, cx, np.nan)).groupby(pcode).transform(lambda s: s.ffill().shift(1)).to_numpy()
        ap_goal = pd.Series(np.where(gls > 0, F["apps_prior"] + 1, np.nan)).groupby(pcode).transform(lambda s: s.ffill().shift(1)).to_numpy()
        cprev = cx - xgv
        F["drought_xg"] = cprev - np.nan_to_num(at_goal)
        F["drought_apps"] = F["apps_prior"] - np.nan_to_num(ap_goal)
    # ---- selection: starts in the team's last 5 / 10 / 20 matches (team fixture sequence)
    fx = side["fx"]
    tm = pd.concat([fx[["fixture_id", "home_id", "t"]].rename(columns={"home_id": "team_id"}), fx[["fixture_id", "away_id", "t"]].rename(columns={"away_id": "team_id"})])
    tm["team_id"] = tm.team_id.astype("int64")
    tm = tm.sort_values(["team_id", "t", "fixture_id"], kind="stable")
    tm["tidx"] = tm.groupby("team_id").cumcount()
    tidx = d[["fixture_id", "team_id"]].merge(tm[["fixture_id", "team_id", "tidx"]], on=["fixture_id", "team_id"], how="left").tidx.to_numpy(float).copy()
    gh = d.ghost.to_numpy() == 1
    if gh.any():
        mx = tm.groupby("team_id").tidx.max()
        tidx[gh] = d.team_id[gh].map(mx).to_numpy(float) + 1
    d["tidx"] = tidx
    ptc = pd.factorize(pd.MultiIndex.from_arrays([d.player_id, d.team_id]))[0]
    k2 = ptc.astype(np.int64) * 10_000_000 + np.nan_to_num(tidx, nan=-1).astype(np.int64)
    o2 = np.argsort(k2, kind="stable")
    ks = k2[o2]
    stv = np.nan_to_num(d.started.to_numpy(float))[o2]
    cst = np.concatenate([[0], np.cumsum(stv)])
    for w in (5, 10, 20):
        F[f"starts_{w}"] = window_count(ks, cst, k2, -w, 0) / w
    F["start_drop"] = F["starts_5"] - F["starts_20"]
    F["apps_for_team"] = pd.Series(played.astype(int)).groupby(ptc).cumsum().to_numpy() - played
    # ---- role familiarity: share of his recent appearances in today's role (line-up known pre-match)
    R1 = np.zeros((n, len(ROLES)))
    ri = d.role.map({r: i for i, r in enumerate(ROLES)}).fillna(len(ROLES) - 1).astype(int).to_numpy()
    R1[np.arange(n), ri] = 1.0
    R1[~played] = np.nan
    RE = ew_prior(pcode, pos, R1, [aL])[aL]
    F["role_familiarity"] = RE[np.arange(n), ri]
    for gname in ("GK", "DEF", "MID", "ATT"):
        F[f"is_{gname}"] = d.role.map(ROLE_GROUP).eq(gname).astype(float).to_numpy()
    F["home"], F["started_today"] = d.home.to_numpy(float), d.started.to_numpy(float)
    # ---- age
    pl = side["players"]
    if len(pl) and "dob" in pl:
        dob = pd.Series(to_day(pl.dob).to_numpy(), index=pl.player_id.astype("int64"))
        dob = dob[~dob.index.duplicated()]
        F["age"] = ((d.day - d.player_id.map(dob)).dt.days / 365.25).to_numpy(float)
    # ---- injuries (spells that ENDED before the match day), transfers, rumours (dated before the match day)
    sl = side["sidelined"]
    if len(sl):
        sl = sl.dropna(subset=["player_id", "end_date"]).copy()
        sl = sl[~sl.category.astype(str).str.contains("suspen")]
        sl["te"] = (to_day(sl.end_date) - DAY0).dt.days.astype(float)
        sl["ts"] = (to_day(sl.start_date) - DAY0).dt.days.astype(float)
        sl["player_id"] = sl.player_id.astype("int64")
        sl = sl.dropna(subset=["te"]).sort_values("te")
        q = pd.DataFrame({"player_id": d.player_id, "t": t, "i": np.arange(n)}).sort_values("t")
        m = pd.merge_asof(q, sl[["player_id", "te", "ts"]].rename(columns={"te": "t"}).assign(te=lambda x: x.t), on="t", by="player_id",
                          allow_exact_matches=False).sort_values("i")
        F["days_since_injury"] = np.clip(t - m.te.to_numpy(), 0, 730)
        F["last_injury_len"] = (m.te - m.ts).to_numpy()
        ks_ = sl.player_id.to_numpy(np.int64) * 100000 + sl.t.to_numpy(np.int64) if "t" in sl else None
        sl2 = sl.assign(k=sl.player_id.to_numpy(np.int64) * 100000 + sl.te.to_numpy(np.int64)).sort_values("k")
        kq = d.player_id.to_numpy(np.int64) * 100000 + t.astype(np.int64)
        ones = np.concatenate([[0], np.cumsum(np.ones(len(sl2)))])
        F["injuries_365d"] = window_count(sl2.k.to_numpy(), ones, kq, -365, 0)
        del ks_
    tr = side["transfers"]
    if len(tr):
        tr = tr.dropna(subset=["date"]).copy()
        tr["t"] = (to_day(tr.date) - DAY0).dt.days.astype(float)
        tr["player_id"] = tr.player_id.astype("int64")
        q = pd.DataFrame({"player_id": d.player_id, "t": t, "i": np.arange(n)}).sort_values("t")
        m = pd.merge_asof(q, tr[["player_id", "t"]].assign(tt=lambda x: x.t).sort_values("t"), on="t", by="player_id", allow_exact_matches=False).sort_values("i")
        F["days_since_transfer"] = np.clip(t - m.tt.to_numpy(), 0, 1500)
    ru = side["rumours"]
    if len(ru):
        ru = ru.dropna(subset=["date"]).copy()
        k_ = np.sort(ru.player_id.astype("int64").to_numpy() * 100000 + (to_day(ru.date) - DAY0).dt.days.astype("int64").to_numpy())
        ones = np.concatenate([[0], np.arange(1, len(k_) + 1, dtype=float)])
        kq = d.player_id.to_numpy(np.int64) * 100000 + t.astype(np.int64)
        F["rumours_30d"] = window_count(k_, ones, kq, -30, 0)
        F["rumours_120d"] = window_count(k_, ones, kq, -120, 0)
    # ---- team & opponent form (team-match rows, earlier matches only), coach tenure, market for this match
    T = team_table(d, side)
    tf = ["gf", "ga", "pts", "xgf", "xga", "expts"]
    TX = T[tf].to_numpy(float)
    tcode = pd.factorize(T.team_id)[0]
    tpos = T.groupby("team_id").cumcount().to_numpy()
    TE = ew_prior(tcode, tpos, TX, [1 - 0.5 ** (1 / 4)])[1 - 0.5 ** (1 / 4)]
    for i, c in enumerate(tf):
        T[f"f_{c}"] = TE[:, i]
    T["coach_tenure"] = np.clip(T.coach_run.to_numpy(float), 0, 300)
    T["coach_new"] = (T.coach_run < 6).astype(float).where(T.coach_id.notna())
    tk = T.set_index(["fixture_id", "team_id"])
    J = d[["fixture_id", "team_id", "opp_id"]].copy()
    tj = J.join(tk[[f"f_{c}" for c in tf] + ["coach_tenure", "coach_new", "p_win", "p_loss"]], on=["fixture_id", "team_id"])
    oj = J.join(tk[[f"f_{c}" for c in tf]], on=["fixture_id", "opp_id"])
    for c in tf:
        F[f"team_{c}"] = tj[f"f_{c}"].to_numpy(float)
        F[f"opp_{c}"] = oj[f"f_{c}"].to_numpy(float)
    F["coach_tenure"], F["coach_new"] = tj.coach_tenure.to_numpy(float), tj.coach_new.to_numpy(float)
    F["p_team_win"], F["p_team_loss"] = tj.p_win.to_numpy(float), tj.p_loss.to_numpy(float)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()}, index=d.index)


def team_table(d, side):
    fx = side["fx"]
    rows = []
    for h in (1, 0):
        x = fx[["fixture_id", "t", "home_id", "away_id", "hg", "ag"]].copy()
        x["team_id"] = x.home_id if h else x.away_id
        x["gf"], x["ga"] = (x.hg, x.ag) if h else (x.ag, x.hg)
        x["home"] = h
        rows.append(x[["fixture_id", "t", "team_id", "gf", "ga", "home"]])
    T = pd.concat(rows, ignore_index=True)
    T["team_id"] = T.team_id.astype("int64")
    T["pts"] = np.select([T.gf > T.ga, T.gf == T.ga], [3.0, 1.0], 0.0)
    T.loc[T.gf.isna() | T.ga.isna(), "pts"] = np.nan
    od = side["odds"]
    if len(od):
        o = od.set_index(od.fixture_id.astype("int64"))
        o = o[~o.index.duplicated()]
        ph, pdw, pa = (T.fixture_id.map(o[c]) for c in ("p_h", "p_d", "p_a"))
        T["p_win"] = np.where(T.home == 1, ph, pa)
        T["p_loss"] = np.where(T.home == 1, pa, ph)
        T["expts"] = T.pts - (3 * T.p_win + pdw)
    else:
        T["p_win"] = T["p_loss"] = T["expts"] = np.nan
    xg = side["team_xg"]
    if len(xg):
        xg = xg.dropna().astype({"fixture_id": "int64", "team_id": "int64"})
        xk = xg.set_index(["fixture_id", "team_id"]).xg
        xk = xk[~xk.index.duplicated()]
        T["xgf"] = xk.reindex(pd.MultiIndex.from_arrays([T.fixture_id.astype("int64"), T.team_id.astype("int64")])).to_numpy()
        opp = T[["fixture_id", "team_id", "xgf"]].rename(columns={"team_id": "o", "xgf": "xga"})
        T = T.merge(opp, on="fixture_id")
        T = T[T.o != T.team_id].drop(columns="o")
    else:
        T["xgf"] = T["xga"] = np.nan
    co = side["coaches"]
    if len(co):
        co = co.dropna().astype({"fixture_id": "int64", "team_id": "int64"})
        c = co.drop_duplicates(["fixture_id", "team_id"]).set_index(["fixture_id", "team_id"]).coach_id
        T["coach_id"] = c.reindex(pd.MultiIndex.from_arrays([T.fixture_id.astype("int64"), T.team_id.astype("int64")])).to_numpy()
    else:
        T["coach_id"] = np.nan
    gh = d[d.ghost == 1][["fixture_id", "team_id", "t"]].drop_duplicates(["fixture_id", "team_id"])
    if len(gh):
        T = pd.concat([T, gh.assign(home=np.nan)], ignore_index=True)
    T = T.sort_values(["team_id", "t", "fixture_id"], kind="stable").reset_index(drop=True)
    T["coach_id"] = T.groupby("team_id").coach_id.ffill()
    chg = (T.coach_id != T.groupby("team_id").coach_id.shift(1)) | (T.team_id != T.team_id.shift(1))
    run = chg.cumsum()
    T["coach_run"] = T.groupby(run).cumcount().astype(float)
    return T


def labels(d, side, args):
    """Targets look FORWARD (that is the point). Each gets label_end = last day it depends on, for purging."""
    n = len(d)
    played = (d.minutes.fillna(0) >= args.min_minutes) & (d.ghost == 0)
    if "z_rating" in d and d.z_rating.notna().mean() > 0.3 * played.mean():
        perf = d.z_rating.to_numpy(float)
        base = d["f_rating_l"].to_numpy(float)
        nprior = d["f_n_rated_prior"].to_numpy(float)
        perf_name = "context-adjusted rating"
    else:   # fallback: average of the available adjusted indicators
        zc = [c for c in d.columns if c.startswith("z_") and c not in ("z_played_to",)]
        perf = np.nanmean(d[zc].to_numpy(float), axis=1)
        d["z_perf"] = perf
        base = pd.Series(perf).groupby(pd.factorize(d.player_id)[0]).transform(lambda s: s.shift(1).ewm(halflife=15, ignore_na=True).mean()).to_numpy()
        nprior = d["f_apps_prior"].to_numpy(float)
        perf_name = "composite of adjusted indicators (no ratings found)"
    ok = played.to_numpy() & np.isfinite(perf) & np.isfinite(base) & (nprior >= args.min_history)
    t = d.t.to_numpy(float)
    d["d_next"] = np.where(ok, perf - base, np.nan)
    d["lend_next"] = np.where(ok, t, np.nan)
    # sustained: mean over the next K qualifying appearances (incl. this one) within the horizon
    e = pd.DataFrame({"pid": d.player_id.to_numpy(), "t": t, "perf": perf, "base": base, "ok": ok, "i": np.arange(n)})
    q = e[np.isfinite(perf) & played.to_numpy()].copy()
    K = args.k_games
    g = q.groupby("pid")
    vals = np.column_stack([g.perf.shift(-j).to_numpy() for j in range(K)])
    tend = g.t.shift(-(K - 1)).to_numpy()
    good = q.ok.to_numpy() & np.isfinite(vals).all(axis=1) & (tend - q.t.to_numpy() <= args.horizon_days)
    ds = np.full(n, np.nan)
    le = np.full(n, np.nan)
    ds[q.i.to_numpy()[good]] = vals[good].mean(axis=1) - q.base.to_numpy()[good]
    le[q.i.to_numpy()[good]] = tend[good]
    d["d_sustained"], d["lend_sustained"] = ds, le
    # dropped: regular starter, starts <= 2 of the team's next 6 matches; injuries / transfers in the window excluded
    tm = side["team_seq"]
    k2, ks, cst = side["k2"], side["ks"], side["cst"]
    nxt = window_count(ks, cst, k2, 1, 7)
    tidx = d.tidx.to_numpy(float)
    tmax = d.team_id.map(tm.groupby("team_id").tidx.max()).to_numpy(float)
    endday = pd.Series(tm.t.to_numpy(), index=pd.MultiIndex.from_arrays([tm.team_id, tm.tidx]))
    endday = endday[~endday.index.duplicated()]
    lend = endday.reindex(pd.MultiIndex.from_arrays([d.team_id.to_numpy(), np.nan_to_num(tidx + 6, nan=-1).astype(int)])).to_numpy()
    elig = (d.started.to_numpy(float) == 1) & (d["f_starts_10"].to_numpy(float) >= 0.7) & (tidx + 6 <= tmax) & (d.ghost.to_numpy() == 0)
    excl = np.zeros(n, bool)
    sl = side["sidelined"]
    if len(sl):
        s = sl.dropna(subset=["player_id", "start_date"])
        k_ = np.sort(s.player_id.astype("int64").to_numpy() * 100000 + (to_day(s.start_date) - DAY0).dt.days.astype("int64").to_numpy())
        ones = np.concatenate([[0], np.arange(1, len(k_) + 1, dtype=float)])
        kq = d.player_id.to_numpy(np.int64) * 100000 + t.astype(np.int64)
        span = np.nan_to_num(lend - t, nan=0).astype(np.int64)
        lo = np.searchsorted(k_, kq + 1)          # spells starting after today ...
        hi = np.searchsorted(k_, kq + span + 1)   # ... up to the label end
        excl |= (ones[hi] - ones[lo]) > 0
    tr = side["transfers"]
    if len(tr):
        k_ = np.sort(tr.player_id.astype("int64").to_numpy() * 100000 + (to_day(tr.date) - DAY0).dt.days.astype("int64").to_numpy())
        ones = np.concatenate([[0], np.arange(1, len(k_) + 1, dtype=float)])
        kq = d.player_id.to_numpy(np.int64) * 100000 + t.astype(np.int64)
        span = np.nan_to_num(lend - t, nan=0).astype(np.int64)
        excl |= (ones[np.searchsorted(k_, kq + span + 1)] - ones[np.searchsorted(k_, kq + 1)]) > 0
    elig &= ~excl & np.isfinite(lend)
    d["y_dropped"] = np.where(elig, (nxt <= 2).astype(float), np.nan)
    d["lend_dropped"] = np.where(elig, lend, np.nan)
    return perf_name


# =============================================================================
# Pipeline
# =============================================================================
def build_all(raw, args, cutoff=None, fixed=None):
    d, side = build_base(raw, cutoff=cutoff, ghosts=cutoff is None)
    for k in ("players", "odds", "team_xg", "coaches"):
        side[k] = raw.get(k) if raw.get(k) is not None else pd.DataFrame()
    avail, cov = normalise(d, args, fixed)
    feat = features(d, side, avail, args)
    return d, side, feat, avail, cov


def feature_meta(cols):
    """family, human label and whether it belongs to the mean-reversion baseline set."""
    meta = {}
    SUF = {"_s": "recent (~3 games)", "_l": "usual (~15 games)", "_d": "change: recent minus usual", "_1": "last game"}
    ML = {m[0]: (m[4], m[5]) for m in METRICS}
    ML.update(RAW_EW)
    OTHER = {
        "rating_vol": ("Inconsistency of adjusted rating", "Form vs own norm"), "n_rated_prior": ("Rated appearances so far", "Player profile"),
        "apps_prior": ("Appearances so far", "Player profile"), "age": ("Age", "Player profile"), "role_familiarity": ("Share of recent games in today's role", "Player profile"),
        "is_GK": ("Goalkeeper", "Player profile"), "is_DEF": ("Defender", "Player profile"), "is_MID": ("Midfielder", "Player profile"), "is_ATT": ("Attacker", "Player profile"),
        "mins_7d": ("Minutes, last 7 days", "Workload & fatigue"), "mins_14d": ("Minutes, last 14 days", "Workload & fatigue"),
        "mins_28d": ("Minutes, last 28 days", "Workload & fatigue"), "apps_14d": ("Appearances, last 14 days", "Workload & fatigue"),
        "rest_days": ("Days since last appearance", "Workload & fatigue"),
        "drought_xg": ("xG accumulated since last goal", "Shooting & finishing"), "drought_apps": ("Appearances since last goal", "Shooting & finishing"),
        "starts_5": ("Start share, team's last 5", "Coach trust (subs & selection)"), "starts_10": ("Start share, team's last 10", "Coach trust (subs & selection)"),
        "starts_20": ("Start share, team's last 20", "Coach trust (subs & selection)"), "start_drop": ("Start share: last 5 minus last 20", "Coach trust (subs & selection)"),
        "apps_for_team": ("Appearances for current team", "Off-field & club changes"),
        "days_since_injury": ("Days since last injury ended", "Injury history"), "last_injury_len": ("Length of last injury (days)", "Injury history"),
        "injuries_365d": ("Injuries ended in last year", "Injury history"), "days_since_transfer": ("Days since last transfer", "Off-field & club changes"),
        "rumours_30d": ("Transfer rumours, last 30 days", "Off-field & club changes"), "rumours_120d": ("Transfer rumours, last 120 days", "Off-field & club changes"),
        "coach_tenure": ("Coach's matches in charge", "Off-field & club changes"), "coach_new": ("New coach (under 6 matches)", "Off-field & club changes"),
        "home": ("Home match", "Match & team context"), "started_today": ("Starts this match (line-up)", "Match & team context"),
        "p_team_win": ("Market: team win probability", "Match & team context"), "p_team_loss": ("Market: team loss probability", "Match & team context"),
    }
    TEAM = {"gf": "goals for", "ga": "goals against", "pts": "points", "xgf": "xG for", "xga": "xG against", "expts": "points vs market"}
    for c in cols:
        if c in OTHER:
            lab, fam = OTHER[c]
        elif c.startswith(("team_", "opp_")):
            w, k = c.split("_", 1)
            lab, fam = f"{'Team' if w == 'team' else 'Opponent'} form: {TEAM.get(k, k)}", "Match & team context"
        else:
            b, suf = c[:-2], c[-2:]
            lab0, fam = ML.get(b, (b, "Other"))
            lab = f"{lab0} - {SUF.get(suf, suf)}" + (" (vs expected)" if b in ML and b not in RAW_EW else "")
        meta[c] = {"label": lab, "family": fam}
    base = [c for c in cols if meta[c]["family"] in ("Form vs own norm", "Player profile", "Match & team context")]
    return meta, base


def months_blocks(t_min, t_max, args):
    start = DAY0 + pd.Timedelta(days=float(t_min) + args.burn_in_days)
    start = pd.Timestamp(start.year, start.month, 1)
    end = DAY0 + pd.Timedelta(days=float(t_max) + 1)
    edges = pd.date_range(start, end, freq=f"{args.retrain_months}MS")
    if len(edges) == 0 or edges[-1] <= end:          # the last block must reach past the final match day
        last = edges[-1] if len(edges) else start
        edges = edges.append(pd.DatetimeIndex([last + pd.DateOffset(months=args.retrain_months)]))
    if len(edges) == 1:
        edges = pd.DatetimeIndex([start]).append(edges) if edges[0] > start else edges
    return [((a - DAY0).days, (b - DAY0).days) for a, b in zip(edges[:-1], edges[1:])]


def make_model(n_trees, seed, threads):
    if HAVE_XGB:
        return xgb.XGBClassifier(n_estimators=n_trees, learning_rate=0.05, max_depth=5, min_child_weight=30, subsample=0.8, colsample_bytree=0.6,
                                 reg_lambda=5.0, tree_method="hist", max_bin=128, n_jobs=threads, random_state=seed, eval_metric="logloss")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=n_trees, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=5.0,
                                          max_bins=63, early_stopping=False, random_state=seed)


class Fitted:
    """A fitted learner plus the columns it was trained on (columns with no data in a window are dropped)."""
    def __init__(self, m, keep):
        self.m, self.keep = m, keep

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, self.keep])

    def contribs(self, X):
        sv = self.m.get_booster().predict(xgb.DMatrix(X[:, self.keep]), pred_contribs=True)[:, :-1]
        out = np.zeros((len(X), len(self.keep)))
        out[:, self.keep] = sv
        return out


def fit_tuned(X, y, t, args, seed=0):
    """Number of trees chosen on the LATEST 15% of the training dates (time split, never random), then refit on all."""
    cut = np.quantile(t, 0.85)
    fm, vm = t < cut, t >= cut
    ref = X[fm] if fm.sum() > 100 else X
    keep = np.isfinite(ref).any(axis=0) & (np.nan_to_num(np.nanstd(ref, axis=0)) > 0)
    X = X[:, keep]
    best = 150
    if fm.sum() > 1500 and vm.sum() > 300 and len(np.unique(y[fm])) == 2 and len(np.unique(y[vm])) == 2:
        if HAVE_XGB:
            m = make_model(args.max_trees, seed, args.threads).fit(X[fm], y[fm], eval_set=[(X[vm], y[vm])], verbose=False)
            losses = m.evals_result()["validation_0"]["logloss"]
        else:
            m = make_model(args.max_trees, seed, args.threads).fit(X[fm], y[fm])
            yv = y[vm]
            losses = [-(yv * np.log(np.clip(p[:, 1], 1e-6, 1)) + (1 - yv) * np.log(np.clip(p[:, 0], 1e-6, 1))).mean() for p in m.staged_predict_proba(X[vm])]
        best = int(np.argmin(losses)) + 1
    return Fitted(make_model(max(best, 20), seed, args.threads).fit(X, y), keep)


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def walk_forward(df, target, feats, base_feats, blocks, args):
    y = df[f"y_{target}"].to_numpy(float)
    t = df.t.to_numpy(float)
    le = df[f"lend_{target}"].to_numpy(float)
    ok = np.isfinite(y) & (df.ghost.to_numpy() == 0)
    P = {"baseline": np.full(len(df), np.nan), "full": np.full(len(df), np.nan)}
    Xf, Xb = df[feats].to_numpy(np.float32), df[base_feats].to_numpy(np.float32)
    rng = np.random.default_rng(3)
    for b0, b1 in blocks:
        tr = ok & (le < b0) & (t >= b0 - args.train_window_days)          # purged: whole label window before the block
        te = ok & (t >= b0) & (t < b1)
        if tr.sum() < args.min_train or te.sum() == 0:
            continue
        idx = np.where(tr)[0]
        if len(idx) > args.max_train_rows:
            idx = np.sort(rng.choice(idx, args.max_train_rows, replace=False))
        for name, X in (("baseline", Xb), ("full", Xf)):
            m = fit_tuned(X[idx], y[idx], t[idx], args)
            P[name][te] = proba(m, X[te])
        log(f"  {target:9s} block {DAY0 + pd.Timedelta(days=b0):%Y-%m}: trained on {len(idx):,} purged rows, predicted {te.sum():,}")
    return P


# =============================================================================
# Analysis
# =============================================================================
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        return None
    return float(roc_auc_score(y[ok], s[ok]))


def metrics(y, p):
    from sklearn.metrics import average_precision_score
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return None
    y, p = y[ok], np.clip(p[ok], 1e-6, 1 - 1e-6)
    br = y.mean()
    top = p >= np.quantile(p, 0.9)
    return {"n": int(len(y)), "base_rate": float(br), "auc": auc(y, p), "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "brier": float(((p - y) ** 2).mean()),
            "top10_rate": float(y[top].mean()), "lift10": float(y[top].mean() / br) if br > 0 else None,
            "ll_null": float(-(br * np.log(br) + (1 - br) * np.log(1 - br)))}


def paired_gain(y, pa, pb, week, B=600, seed=4):
    """Log-loss of b minus a per row (positive = a better), 95% interval from resampling whole weeks."""
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y)
    if ok.sum() < 100:
        return None
    ll = lambda p: -(y[ok] * np.log(np.clip(p[ok], 1e-6, 1)) + (1 - y[ok]) * np.log(np.clip(1 - p[ok], 1e-6, 1)))  # noqa: E731
    g = pd.DataFrame({"x": ll(pb) - ll(pa), "w": week[ok]}).groupby("w").x.agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(s), (B, len(s)))
    bs = s[ix].sum(1) / c[ix].sum(1)
    return {"gain": float(s.sum() / c.sum()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B)), "n": int(c.sum())}


def deciles(x, y, cuts):
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.searchsorted(cuts, x[ok], side="right")
    out = []
    for k in range(len(cuts) + 1):
        m = b == k
        out.append([int(k), float(y[ok][m].mean()) if m.sum() >= 15 else None, int(m.sum()), float(np.median(x[ok][m])) if m.sum() else None])
    return out


def univariate(df, feats, target, hold_t, eval_mask):
    y = df[f"y_{target}"].to_numpy(float)
    t = df.t.to_numpy(float)
    disc, hold = eval_mask & (t < hold_t), eval_mask & (t >= hold_t)
    out = {}
    for c in feats:
        x = df[c].to_numpy(float)
        a_d, a_h = auc(y[disc], x[disc]), auc(y[hold], x[hold])
        if a_d is None:
            continue
        sgn = 1 if a_d >= 0.5 else -1
        xd = x[disc][np.isfinite(x[disc])]
        cuts = np.unique(np.quantile(xd, np.linspace(0.1, 0.9, 9))) if len(xd) > 50 else np.array([])
        out[c] = {"auc_d": a_d if sgn > 0 else 1 - a_d, "auc_h": (a_h if sgn > 0 else 1 - a_h) if a_h is not None else None, "dir": sgn,
                  "cov": float(np.isfinite(x[eval_mask]).mean()), "cuts": cuts.tolist(),
                  "dec_d": deciles(x[disc], y[disc], cuts), "dec_h": deciles(x[hold], y[hold], cuts)}
    return out


def importance(df, target, feats, meta, hold_t, args, uni):
    """Model trained on the discovery period (purged), explained on the hold-out period."""
    y = df[f"y_{target}"].to_numpy(float)
    t = df.t.to_numpy(float)
    le = df[f"lend_{target}"].to_numpy(float)
    ok = np.isfinite(y) & (df.ghost.to_numpy() == 0)
    tr = ok & (le < hold_t) & (t >= hold_t - args.train_window_days)
    ho = ok & (t >= hold_t)
    if tr.sum() < args.min_train or ho.sum() < 300:
        return {}
    rng = np.random.default_rng(7)
    idx = np.where(tr)[0]
    if len(idx) > args.max_train_rows:
        idx = np.sort(rng.choice(idx, args.max_train_rows, replace=False))
    X = df[feats].to_numpy(np.float32)
    m = fit_tuned(X[idx], y[idx], t[idx], args)
    hi = np.where(ho)[0]
    if len(hi) > args.explain_rows:
        hi = np.sort(rng.choice(hi, args.explain_rows, replace=False))
    Xh, yh = X[hi], y[hi]
    p0 = proba(m, Xh)
    a0 = auc(yh, p0)
    ll = lambda p: float(-(yh * np.log(np.clip(p, 1e-6, 1)) + (1 - yh) * np.log(np.clip(1 - p, 1e-6, 1))).mean())  # noqa: E731
    l0 = ll(p0)
    fams = defaultdict(list)
    for i, c in enumerate(feats):
        fams[meta[c]["family"]].append(i)
    res = {"auc": a0, "logloss": l0, "families": {}, "features": {}, "roles": {}, "method": "SHAP (XGBoost)" if HAVE_XGB else "permutation"}

    def perm(cols, rows=None):
        Z = Xh.copy() if rows is None else Xh[rows].copy()
        pr = rng.permutation(len(Z))
        Z[:, cols] = Z[pr][:, cols]
        p = proba(m, Z)
        yy = yh if rows is None else yh[rows]
        return p, yy
    for fam, cols in fams.items():
        drops = []
        for rep in range(args.perm_repeats):
            p, _ = perm(cols)
            drops.append((a0 - (auc(yh, p) or a0), ll(p) - l0))
        res["families"][fam] = {"auc_drop": float(np.mean([x[0] for x in drops])), "ll_inc": float(np.mean([x[1] for x in drops])), "n": len(cols)}
    if HAVE_XGB:
        sv = m.contribs(Xh)
        for i, c in enumerate(feats):
            x = Xh[:, i]
            okx = np.isfinite(x)
            direction = float(np.corrcoef(x[okx], sv[okx, i])[0, 1]) if okx.sum() > 30 and np.std(x[okx]) > 0 and np.std(sv[okx, i]) > 0 else 0.0
            res["features"][c] = {"imp": float(np.abs(sv[:, i]).mean()), "dir": 1 if direction >= 0 else -1, "strength": direction}
        res["imp_unit"] = "mean |SHAP| (log-odds)"
    else:
        for i, c in enumerate(feats):
            p, _ = perm([i])
            u = uni.get(c, {})
            res["features"][c] = {"imp": float(a0 - (auc(yh, p) or a0)), "dir": u.get("dir", 1)}
        res["imp_unit"] = "AUC drop when scrambled"
    res["model"] = m
    # by position group: discrimination and which families matter there
    grp = df.role.map(ROLE_GROUP).to_numpy()[hi]
    for gname in ("GK", "DEF", "MID", "ATT"):
        rows = np.where(grp == gname)[0]
        if len(rows) < 200 or len(np.unique(yh[rows])) < 2:
            continue
        ag = auc(yh[rows], p0[rows])
        fd = {}
        for fam, cols in fams.items():
            p, yy = perm(cols, rows)
            fd[fam] = float(ag - (auc(yy, p) or ag))
        res["roles"][gname] = {"n": int(len(rows)), "auc": ag, "rate": float(yh[rows].mean()), "families": fd}
    return res


def leak_test(raw, args, full_d, full_feat, picks, avail):
    tests, passed = [], True
    key = ["fixture_id", "player_id"]
    for c in picks:
        dc, _, fc, _, _ = build_all(raw, args, cutoff=c, fixed=avail)
        a = pd.concat([full_d[key], full_feat], axis=1)[(full_d.t == (c - DAY0).days) & (full_d.ghost == 0)].set_index(key).sort_index()
        b = pd.concat([dc[key], fc], axis=1)[dc.t == (c - DAY0).days].set_index(key).sort_index()
        cols = [x for x in a.columns if x in b.columns]
        missing = [x for x in a.columns if x not in b.columns]
        b = b.reindex(a.index)
        A, Bv = a[cols].to_numpy(float), b[cols].to_numpy(float)
        same = np.isclose(A, Bv, rtol=1e-5, atol=1e-5, equal_nan=True)
        bad = int((~same).sum()) + len(missing) * len(a)
        worst = []
        if bad:
            cnt = (~same).sum(axis=0)
            worst = [cols[i] for i in np.argsort(-cnt)[:5] if cnt[i] > 0] + missing[:3]
        tests.append({"day": str(c.date()), "rows": int(len(a)), "features": len(cols), "mismatches": bad, "worst": worst})
        passed &= bad == 0
        log(f"  time-travel test {c.date()}: {len(a)} player rows x {len(cols)} features, mismatches = {bad}" + (f"  {worst}" if worst else ""))
    return {"tests": tests, "passed": passed}


def explain_rows(m, Xr, feats, top_feats, med):
    """Top 3 features pushing each row's risk up."""
    out = []
    if HAVE_XGB:
        sv = m.contribs(Xr)
        for r in range(len(Xr)):
            o = np.argsort(-sv[r])[:3]
            out.append([[feats[i], float(sv[r, i]), None if not np.isfinite(Xr[r, i]) else float(Xr[r, i])] for i in o if sv[r, i] > 0])
        return out
    p0 = proba(m, Xr)
    eff = np.zeros((len(Xr), len(top_feats)))
    for j, c in enumerate(top_feats):
        i = feats.index(c)
        Z = Xr.copy()
        Z[:, i] = med[i]
        eff[:, j] = p0 - proba(m, Z)
    for r in range(len(Xr)):
        o = np.argsort(-eff[r])[:3]
        out.append([[top_feats[j], float(eff[r, j]), None if not np.isfinite(Xr[r, feats.index(top_feats[j])]) else float(Xr[r, feats.index(top_feats[j])])]
                    for j in o if eff[r, j] > 0])
    return out


# =============================================================================
# Main
# =============================================================================
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic database (to see the dashboard)")
    ap.add_argument("--out", default="form_downturn_dashboard.html")
    ap.add_argument("--export-csv", help="write every row's features, labels and out-of-sample predictions")
    ap.add_argument("--title", default="Predicting a player's drop in form")
    ap.add_argument("--leagues", help="comma-separated league_ids to evaluate (history always uses everything)")
    ap.add_argument("--holdout-from", help="start of the hold-out period (default: 60%% through the evaluated period)")
    ap.add_argument("--burn-in-days", type=int, default=365)
    ap.add_argument("--train-window-days", type=int, default=1095)
    ap.add_argument("--retrain-months", type=int, default=6, help="months between retrains (lower = slower, slightly fresher)")
    ap.add_argument("--min-train", type=int, default=3000)
    ap.add_argument("--max-train-rows", type=int, default=300_000, help="random sample cap per training window (speed)")
    ap.add_argument("--max-trees", type=int, default=600)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--min-minutes", type=int, default=30, help="appearances shorter than this are not scored as performances")
    ap.add_argument("--min-history", type=int, default=8, help="rated appearances needed before a player's baseline counts")
    ap.add_argument("--k-games", type=int, default=6, help="appearances averaged for the sustained-downturn target")
    ap.add_argument("--horizon-days", type=int, default=150, help="the K appearances must fall within this many days")
    ap.add_argument("--downturn-quantile", type=float, default=0.25, help="bottom share that counts as a downturn (frozen on the first window)")
    ap.add_argument("--market-1x2", type=int, default=1)
    ap.add_argument("--leak-tests", type=int, default=2)
    ap.add_argument("--skip-leak-test", action="store_true")
    ap.add_argument("--perm-repeats", type=int, default=2)
    ap.add_argument("--explain-rows", type=int, default=60_000)
    ap.add_argument("--watchlist", type=int, default=80)
    ap.add_argument("--db-jobs", type=int, default=6, help="parallel database connections while reading")
    ap.add_argument("--work-mem", default="256MB", help="PostgreSQL work_mem for this script's sessions (bigger = fewer disk spills)")
    ap.add_argument("--cache")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    import signal
    signal.signal(signal.SIGINT, _on_sigint)
    if args.work_mem:
        os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + f" -c work_mem={args.work_mem}").strip()
    log(f"learner: {'XGBoost ' + xgb.__version__ if HAVE_XGB else 'scikit-learn HistGradientBoosting (XGBoost unavailable: ' + XGB_ERR + ')'}")
    if not HAVE_XGB and "No module named" not in XGB_ERR:
        full = XGB_FULL.lower()
        if "libomp" in full or "openmp" in full:
            log("  XGBoost is installed but cannot load the OpenMP runtime. On a Mac run:  brew install libomp   (then rerun)")
        else:
            log("  XGBoost is installed but failed to load. Full error:\n" + XGB_FULL.strip()[:1500])
    elif not HAVE_XGB:
        log(f"  this Python ({sys.executable}) cannot see XGBoost - install it into this one:  {sys.executable} -m pip install xgboost")
    if args.demo:
        log("building the synthetic demo database ...")
        raw = demo_data()
        db_name = "synthetic demo"
        if args.min_train == 3000:
            args.min_train = 1500
    else:
        db = DB(args.dsn)
        db_name = db.name
        raw = extract(db, args)
    raw = prepare_raw(raw)
    log("building appearances, context-adjusted residuals and features ...")
    d, side, feat, avail, cov = build_all(raw, args)
    feat = feat.loc[:, (feat[d.ghost.to_numpy() == 0].notna().mean() > 0.01) & (feat.nunique() > 1)]
    side["team_seq"] = None
    log(f"{len(d):,} line-up rows ({int((d.ghost == 0).sum()):,} real), {feat.shape[1]} features")
    # selection windows for the 'dropped' target (same arrays as the starts_N features)
    fx = side["fx"]
    tm = pd.concat([fx[["fixture_id", "home_id", "t"]].rename(columns={"home_id": "team_id"}), fx[["fixture_id", "away_id", "t"]].rename(columns={"away_id": "team_id"})])
    tm["team_id"] = tm.team_id.astype("int64")
    tm = tm.sort_values(["team_id", "t", "fixture_id"], kind="stable")
    tm["tidx"] = tm.groupby("team_id").cumcount()
    side["team_seq"] = tm
    ptc = pd.factorize(pd.MultiIndex.from_arrays([d.player_id, d.team_id]))[0]
    k2 = ptc.astype(np.int64) * 10_000_000 + np.nan_to_num(d.tidx.to_numpy(float), nan=-1).astype(np.int64)
    o2 = np.argsort(k2, kind="stable")
    side["k2"], side["ks"] = k2, k2[o2]
    side["cst"] = np.concatenate([[0], np.cumsum(np.nan_to_num(d.started.to_numpy(float))[o2])])
    fcols = list(feat.columns)
    df = pd.concat([d, feat.add_prefix("f_")], axis=1)
    perf_name = labels(df, side, args)
    feats = ["f_" + c for c in fcols]
    meta0, base0 = feature_meta(fcols)
    meta = {"f_" + k: v for k, v in meta0.items()}
    base_feats = ["f_" + c for c in base0]
    log(f"performance measure: {perf_name}; features: {len(feats)} (baseline set {len(base_feats)})")
    tmin, tmax = df.t[df.ghost == 0].min(), df.t[df.ghost == 0].max()
    blocks = months_blocks(tmin, tmax, args)
    if not blocks:
        sys.exit("Not enough history for a walk-forward test (lower --burn-in-days).")
    first = blocks[0][0]
    taus = {}
    for tg in ("next", "sustained"):
        dv, le = df[f"d_{tg}"].to_numpy(float), df[f"lend_{tg}"].to_numpy(float)
        early = np.isfinite(dv) & (le < first)
        if early.sum() < 300:
            early = np.isfinite(dv) & (df.t.to_numpy() <= np.nanquantile(df.t[np.isfinite(dv)], 0.25))
        taus[tg] = float(np.quantile(dv[early], args.downturn_quantile))
        df[f"y_{tg}"] = np.where(np.isfinite(dv), (dv <= taus[tg]).astype(float), np.nan)
        log(f"target {tg}: downturn = performance {taus[tg]:+.3f} or worse vs own baseline (frozen from {early.sum():,} rows before the first test month)")
    if args.leagues:
        keep = df.league_id.isin([int(x) for x in args.leagues.split(",")]).to_numpy()
    else:
        keep = np.ones(len(df), bool)
    # ---- time-travel test
    leak = {"skipped": True, "passed": True, "tests": []}
    if not args.skip_leak_test:
        log("time-travel test: re-running the whole pipeline with the future deleted ...")
        days = np.sort(df.t[(df.ghost == 0) & (df.t >= first)].unique())
        rng = np.random.default_rng(5)
        picks = sorted(rng.choice(days, size=min(args.leak_tests, len(days)), replace=False))
        leak = leak_test(raw, args, d, feat, [DAY0 + pd.Timedelta(days=float(x)) for x in picks], avail)
        leak["skipped"] = False
    # ---- walk-forward
    preds, evals, uni, imps = {}, {}, {}, {}
    t = df.t.to_numpy(float)
    week = (df.day.dt.isocalendar().week.astype(int) + 100 * df.day.dt.year).to_numpy()
    hold_t = None
    for tg in TARGETS:
        if df[f"y_{tg}"].notna().sum() < args.min_train:
            log(f"target {tg}: too few labelled rows - skipped")
            continue
        log(f"walk-forward: {tg} ...")
        P = walk_forward(df, tg, feats, base_feats, blocks, args)
        preds[tg] = P
        em = np.isfinite(P["full"]) & np.isfinite(P["baseline"]) & keep
        if em.sum() < 200:
            continue
        if hold_t is None:
            hold_t = float((pd.Timestamp(args.holdout_from) - DAY0).days) if args.holdout_from else float(np.floor(np.quantile(t[em], 0.6)))
        y = df[f"y_{tg}"].to_numpy(float)
        rec = {"models": {}, "gain": {}}
        for pn, pm in (("discovery", em & (t < hold_t)), ("holdout", em & (t >= hold_t)), ("all", em)):
            rec["models"][pn] = {k: metrics(y[pm], v[pm]) for k, v in P.items()}
            rec["gain"][pn] = paired_gain(y[pm], P["full"][pm], P["baseline"][pm], week[pm])
        hm = em & (t >= hold_t)
        pr = P["full"][hm]
        cuts = np.quantile(pr, np.linspace(0.1, 0.9, 9))
        b = np.searchsorted(cuts, pr, side="right")
        rec["lift"] = [[k + 1, float(pr[b == k].mean()), float(y[hm][b == k].mean()), int((b == k).sum())] for k in range(10) if (b == k).sum()]
        prb = P["baseline"][hm]
        cb = np.quantile(prb, np.linspace(0.1, 0.9, 9))
        bb = np.searchsorted(cb, prb, side="right")
        rec["lift_base"] = [[k + 1, float(prb[bb == k].mean()), float(y[hm][bb == k].mean()), int((bb == k).sum())] for k in range(10) if (bb == k).sum()]
        mo = pd.DataFrame({"m": df.day.dt.to_period("Q").astype(str).to_numpy()[em], "y": y[em], "f": P["full"][em], "b": P["baseline"][em]})
        rec["by_quarter"] = [[q, auc(g.y.to_numpy(), g.f.to_numpy()), auc(g.y.to_numpy(), g.b.to_numpy()), int(len(g))] for q, g in mo.groupby("m")]
        evals[tg] = rec
        log(f"  {tg}: univariate signals ...")
        uni[tg] = univariate(df, feats, tg, hold_t, em)
        log(f"  {tg}: importance (discovery model, hold-out explanation) ...")
        imps[tg] = importance(df, tg, feats, meta, hold_t, args, uni[tg])
        h = rec["models"]["holdout"]
        log(f"  {tg}: hold-out AUC full {h['full']['auc']:.3f} vs baseline {h['baseline']['auc']:.3f} · base rate {h['full']['base_rate']:.1%} · top-10% lift {h['full']['lift10']:.2f}x")
    if not evals:
        sys.exit("No target could be evaluated - not enough history.")
    # ---- watchlist: models trained on everything labelled so far, applied before each player's next match
    log("watchlist ...")
    gh = np.where(df.ghost.to_numpy() == 1)[0]
    watch = []
    if len(gh):
        Xall = df[feats].to_numpy(np.float32)
        risk, drivers = {}, {}
        for tg in evals:
            y = df[f"y_{tg}"].to_numpy(float)
            le = df[f"lend_{tg}"].to_numpy(float)
            tr = np.isfinite(y) & (le <= tmax) & (t >= tmax - args.train_window_days) & (df.ghost.to_numpy() == 0)
            idx = np.where(tr)[0]
            if len(idx) > args.max_train_rows:
                idx = np.sort(np.random.default_rng(9).choice(idx, args.max_train_rows, replace=False))
            m = fit_tuned(Xall[idx], y[idx], t[idx], args)
            risk[tg] = proba(m, Xall[gh])
            top = [c for c, _ in sorted(imps.get(tg, {}).get("features", {}).items(), key=lambda kv: -kv[1]["imp"])[:20]] or feats[:20]
            med = np.nanmedian(Xall[idx], axis=0)
            drivers[tg] = explain_rows(m, Xall[gh], feats, top, med)
        names = dict(zip(raw["players"].player_id.astype("int64"), raw["players"].name)) if len(raw["players"]) else {}
        tnames = dict(zip(raw["teams"].team_id.astype("int64"), raw["teams"].name)) if len(raw["teams"]) else {}
        G = df.iloc[gh]
        order_tg = "sustained" if "sustained" in risk else list(risk)[0]
        rank = np.argsort(-risk[order_tg])
        rated = G["f_n_rated_prior"].to_numpy(float) if "f_n_rated_prior" in G else G["f_apps_prior"].to_numpy(float)
        for r in rank:
            if rated[r] < args.min_history:
                continue
            row = G.iloc[r]
            watch.append({"player": names.get(int(row.player_id), f"player {int(row.player_id)}"), "team": tnames.get(int(row.team_id), str(int(row.team_id))),
                          "league": row.league, "role": row.role, "age": row.get("f_age"), "apps": row.get("f_apps_prior"),
                          "rating_l": row.get("f_rating_l"), "rating_d": row.get("f_rating_d"),
                          "risk": {tg: float(risk[tg][r]) for tg in risk}, "drivers": {tg: drivers[tg][r] for tg in drivers}})
            if len(watch) >= args.watchlist:
                break
        base_rates = {tg: float(np.nanmean(df[f"y_{tg}"])) for tg in risk}
    else:
        base_rates = {}
    # ---- payload
    tlab = {}
    for tg, (title, desc) in TARGETS.items():
        if tg not in evals:
            continue
        extra = ""
        if tg in taus:
            extra = f" Downturn = {taus[tg]:+.2f} or worse vs his baseline ({perf_name})" + (f", averaged over {args.k_games} appearances within {args.horizon_days} days." if tg == "sustained" else ".")
        tlab[tg] = {"title": title, "desc": desc + extra, "tau": taus.get(tg), "rows": int(np.isfinite(df[f"y_{tg}"]).sum())}
    feat_out = {}
    for tg in evals:
        I = imps.get(tg, {})
        fi = I.get("features", {})
        rows = []
        for c in feats:
            u = uni[tg].get(c)
            if u is None and c not in fi:
                continue
            rows.append({"f": c[2:], "label": meta[c]["label"], "family": meta[c]["family"], "imp": fi.get(c, {}).get("imp"), "dir": fi.get(c, {}).get("dir", (u or {}).get("dir")),
                         **({k: u[k] for k in ("auc_d", "auc_h", "cov", "cuts", "dec_d", "dec_h")} if u else {}), "udir": (u or {}).get("dir")})
        feat_out[tg] = sorted(rows, key=lambda r: -(r["imp"] or 0))
    famo = {tg: {"families": imps[tg].get("families", {}), "roles": imps[tg].get("roles", {}), "auc": imps[tg].get("auc"), "unit": imps[tg].get("imp_unit"),
                 "method": imps[tg].get("method")} for tg in evals if imps.get(tg)}
    covr = {"rows": int((d.ghost == 0).sum()), "players": int(d.player_id[d.ghost == 0].nunique()), "fixtures": int(d.fixture_id[d.ghost == 0].nunique()),
            "leagues": int(d.league_id[d.ghost == 0].nunique()), "from": str(d.day[d.ghost == 0].min().date()), "to": str(d.day[d.ghost == 0].max().date()),
            "holdout_from": str((DAY0 + pd.Timedelta(days=hold_t)).date()), "found": raw.get("found", {}), "metrics": {MKEY[k][4]: v for k, v in cov.items()},
            "missing_metrics": [m[4] for m in METRICS if m[0] not in avail], "perf": perf_name, "game_state": float(d.state_bucket.ge(0)[d.minutes.fillna(0) > 0].mean()),
            "subs_known": float(d.subs_known[d.ghost == 0].mean()), "base_rates": base_rates, "learner": "XGBoost" if HAVE_XGB else "HistGradientBoosting (scikit-learn)",
            "n_features": len(feats), "n_base": len(base_feats), "runtime_s": round(time.time() - T0)}
    alarms = [tg for tg in evals if (evals[tg]["models"]["holdout"]["full"] or {}).get("auc", 0) > (0.85 if tg != "dropped" else 0.92)]
    payload = clean({"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "db": db_name, "demo": bool(raw.get("demo")),
                     "params": {k: v for k, v in vars(args).items() if k not in ("dsn",)}, "targets": tlab, "eval": evals, "features": feat_out,
                     "imp": famo, "watch": watch, "coverage": covr, "leak": leak, "alarms": alarms,
                     "families": sorted({v["family"] for v in meta.values()})})
    if args.export_csv:
        out = df[["fixture_id", "player_id", "team_id", "league_id", "day", "role", "ghost"] + [c for c in df.columns if c.startswith(("y_", "d_", "lend_", "z_"))] + feats].copy()
        for tg, P in preds.items():
            for k, v in P.items():
                out[f"pred_{tg}_{k}"] = v
        out.to_csv(args.export_csv, index=False)
        log(f"CSV written: {args.export_csv}")
    page = TEMPLATE.replace("__TITLE__", htmlmod.escape(args.title)).replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    Path(args.out).write_text(page, encoding="utf-8")
    log(f"dashboard written: {args.out} ({len(page) / 1e6:.1f} MB)")
    if not leak.get("skipped") and not leak["passed"]:
        log("!!! TIME-TRAVEL TEST FAILED - some features change when the future is deleted. See the dashboard.")
    for tg in alarms:
        log(f"!!! {tg}: hold-out AUC is implausibly high - check for leakage before trusting it.")


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--ctrl:#a9a79f;--good:#006300;--bad:#d03b3b;--goodbg:rgba(12,163,12,.10);--badbg:rgba(208,59,59,.09);--wash:rgba(42,120,214,.07);--warnbg:rgba(235,104,52,.12)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.10);--warnbg:rgba(217,89,38,.16)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.10);--warnbg:rgba(217,89,38,.16)}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh;padding:26px clamp(14px,3.5vw,44px) 60px}
.wrap{max-width:1320px;margin:0 auto}
h1{font-size:24px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:15px;font-weight:600;margin:0 0 2px}h3{font-size:13px;font-weight:600;margin:12px 0 4px;color:var(--ink2)}
.sub{color:var(--ink2);margin:0 0 14px;max-width:980px}.note{color:var(--muted);font-size:12px;margin:2px 0 10px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;padding:4px 10px;border-radius:99px;border:1px solid var(--ring);background:var(--surface);color:var(--ink2)}
.badge.ok{background:var(--goodbg);color:var(--good)}.badge.bad{background:var(--badbg);color:var(--bad)}.badge.warn{background:var(--warnbg);color:var(--ink)}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:0 0 18px;padding:10px 14px;background:var(--surface);border:1px solid var(--ring);border-radius:10px;position:sticky;top:8px;z-index:6}
.controls label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--ink2)}
select,button,input{font:inherit;color:var(--ink);background:var(--page);border:1px solid var(--ring);border-radius:7px;padding:5px 9px}button{cursor:pointer}button:hover{background:var(--wash)}
.grid{display:grid;gap:14px}.g2{grid-template-columns:repeat(auto-fit,minmax(520px,1fr))}.g5{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:15px 17px;min-width:0}
.hero{display:flex;flex-wrap:wrap;gap:26px;align-items:flex-end}.big{font-size:50px;font-weight:650;line-height:1;letter-spacing:-.02em}
.lbl{color:var(--ink2);font-size:12.5px}.tile .val{font-size:24px;font-weight:620;margin:3px 0 2px}.tile .ci{color:var(--muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}.section{margin:30px 0 10px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:4px 0 6px;flex-wrap:wrap}.key{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}.ln{width:14px;height:2px;display:inline-block;border-radius:1px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{font-family:inherit;font-size:11px;fill:var(--muted)}svg .lab{fill:var(--ink2);font-size:11.5px}svg .val{fill:var(--ink);font-size:11px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none;z-index:1}
td:first-child,th:first-child,.l{text-align:left}td{font-variant-numeric:tabular-nums}tr.sel td{background:var(--wash)}tbody tr:hover td{background:var(--wash)}
.scroll{max-height:520px;overflow:auto}.wide{overflow-x:auto}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);box-shadow:0 4px 18px rgba(0,0,0,.14);border-radius:8px;padding:8px 10px;font-size:12px;opacity:0;transition:opacity .08s;z-index:30;max-width:320px}
.hit{fill:transparent}.why{color:var(--muted);font-size:11.5px;white-space:normal;max-width:520px}ul.m li{margin:3px 0;color:var(--ink2)}
.demo{background:var(--warnbg);border:1px solid var(--s2);border-radius:10px;padding:10px 14px;margin:0 0 14px;font-size:13.5px}
td.heat{text-align:center;color:var(--ink)}.drv{white-space:normal;min-width:260px;max-width:420px;text-align:left;color:var(--ink2);font-size:12px}
.arrow{font-weight:700}
</style></head>
<body><div class="viz-root"><div class="wrap">
<h1>__TITLE__</h1><p class="sub" id="sub"></p><div id="demo"></div><div class="badges" id="badges"></div>
<div class="controls">
  <label>Downturn to predict<select id="target"></select></label>
  <label>Period<select id="period"><option value="holdout">Hold-out (honest)</option><option value="discovery">Discovery</option><option value="all">All</option></select></label>
  <button id="theme" type="button">Theme</button>
</div>
<div class="card hero" id="hero"></div>
<div class="grid g5" id="tiles" style="margin-top:14px"></div>

<div class="section"><h2>Which kinds of signal predict the downturn?</h2><p class="note" id="fam-note"></p></div>
<div class="grid g2">
  <div class="card"><h2>By family of signals</h2><p class="note">How much the hold-out AUC falls when every feature in the family is scrambled at once (×100). Bigger = the model leans on it and it works on unseen matches.</p><div id="c-fam"></div></div>
  <div class="card"><h2>Strongest individual signals</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s2)"></span>Higher value → more risk</span><span class="key"><span class="sw" style="background:var(--s1)"></span>Higher value → less risk</span></div><p class="note" id="feat-note"></p><div id="c-feat"></div></div>
</div>

<div class="section"><h2>Feature explorer</h2><p class="note">Downturn rate by decile of the chosen signal. Decile cut points come from the discovery period; a real signal slopes the same way in the hold-out. Click any row of the table to chart it.</p></div>
<div class="grid g2">
  <div class="card"><label class="lbl">Signal <select id="fsel" style="max-width:100%"></select></label><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Hold-out</span><span class="key"><span class="ln" style="background:var(--s2)"></span>Discovery</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Overall rate</span></div><div id="c-dec"></div><p class="note" id="dec-note"></p></div>
  <div class="card"><h2>All signals</h2><label class="lbl">Family <select id="ffam"><option value="">All</option></select></label><div class="scroll" style="margin-top:8px"><table id="t-feat"></table></div></div>
</div>

<div class="section"><h2>By position</h2><p class="note">Hold-out AUC of the full model within each position group, and the AUC lost when each family is scrambled for that group only (×100). Darker = more important there.</p></div>
<div class="card wide"><table id="t-roles"></table></div>

<div class="section"><h2>How good are the forecasts?</h2><p class="note">Full model = every signal. Baseline = mean reversion only (recent vs usual rating, inconsistency) plus player profile and match context. The gap is what the confidence indicators add.</p></div>
<div class="grid g2">
  <div class="card"><h2>Actual downturn rate by predicted-risk decile</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>Full model</span><span class="key"><span class="sw" style="background:var(--s2);border-radius:99px"></span>Baseline</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Overall rate</span></div><div id="c-lift"></div></div>
  <div class="card"><h2>AUC by quarter (out of sample)</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Full model</span><span class="key"><span class="ln" style="background:var(--s2)"></span>Baseline</span></div><div id="c-q"></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>All targets and periods</h2><div class="wide"><table id="t-eval"></table></div></div>

<div class="section"><h2>Watchlist: risk before each player's next match</h2><p class="note" id="watch-note"></p></div>
<div class="card"><div class="controls" style="position:static;margin:0 0 8px;padding:6px 8px"><label>Position<select id="wrole"><option value="">All</option><option>GK</option><option>DEF</option><option>MID</option><option>ATT</option></select></label><label>Search<input id="wq" placeholder="player or team"></label></div><div class="scroll" style="max-height:640px"><table id="t-watch"></table></div></div>

<div class="section"><h2>Data, safeguards and method</h2></div>
<div class="grid g2">
  <div class="card"><h2>How leakage is prevented</h2><ul class="m" id="safe"></ul><h3>Time-travel tests</h3><table id="t-leak"></table></div>
  <div class="card"><h2>Coverage</h2><table id="t-cov"></table><h3>Context-adjusted indicators (share of appearances with data)</h3><div class="scroll" style="max-height:300px"><table id="t-met"></table></div></div>
</div>
</div></div><div class="tip" id="tip"></div>
<script>
const R = __DATA__;
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const pctu = (v, d=1) => v == null ? '–' : (v*100).toFixed(d) + '%';
const num = (v, d=2) => v == null ? '–' : (+v).toFixed(d);
const int = v => v == null ? '–' : Math.round(v).toLocaleString();
const sgn = (v, d=1) => v == null ? '–' : (v >= 0 ? '+' : '') + (+v).toFixed(d);
const cls = v => v == null ? '' : v >= 0 ? 'pos' : 'neg';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tip = $('#tip');
function showTip(e, h){ tip.innerHTML = h; tip.style.opacity = 1; const w = tip.offsetWidth, x = e.clientX + 14 + w > innerWidth ? e.clientX - w - 14 : e.clientX + 14; tip.style.left = x + 'px'; tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8) + 'px'; }
function hideTip(){ tip.style.opacity = 0; }
const NS = 'http://www.w3.org/2000/svg';
function el(tag, a, p){ const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
function svg(host, w, h){ host.innerHTML = ''; return el('svg', {viewBox:`0 0 ${w} ${h}`, role:'img'}, host); }
function text(p, x, y, s, a={}){ const t = el('text', Object.assign({x, y}, a), p); t.textContent = s; return t; }
function ticks(lo, hi, n=5){ const span = hi - lo || 1, s0 = span/n, mag = Math.pow(10, Math.floor(Math.log10(s0))); const st = [1,2,2.5,5,10].map(m=>m*mag).find(s => span/s <= n) || mag*10; const o=[]; for(let v=Math.ceil(lo/st)*st; v<=hi+1e-9; v+=st) o.push(+v.toFixed(10)); return o; }
function ext(vals, zero=true){ let v = vals.filter(x => x != null && isFinite(x)); if (zero) v.push(0); if (!v.length) v=[0,1]; let lo = Math.min(...v), hi = Math.max(...v); if (lo === hi){ lo -= .5; hi += .5; } const p = (hi-lo)*.08; return [zero && lo >= 0 ? 0 : lo-p, hi+p]; }
function hit(node, html){ node.addEventListener('mousemove', e => showTip(e, html)); node.addEventListener('mouseleave', hideTip); }
function empty(host, msg){ host.innerHTML = `<p class="note" style="padding:40px 0;text-align:center">${msg}</p>`; }
function hbarPath(x0, x1, y, h){ const r = Math.min(4, Math.abs(x1-x0)/2, h/2); if (x1 >= x0) return `M${x0},${y}H${x1-r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1-r},${y+h}H${x0}Z`; return `M${x0},${y}H${x1+r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1+r},${y+h}H${x0}Z`; }
function vbarPath(x, w, y0, y1){ const r = Math.min(4, Math.abs(y1-y0)/2, w/2); return `M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z`; }
function hbars(host, rows, fmt, o={}){
  if (!rows.length) return empty(host, 'No data');
  const W = 600, rowH = o.rowH || 28, top = 4, left = o.left || 210, right = 60, H = top + rows.length*rowH + 24, s = svg(host, W, H);
  const [lo, hi] = ext(rows.map(r => r.v)); const x = v => left + (v-lo)/(hi-lo)*(W-left-right);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-20, stroke:css('--grid')}, s); text(s, x(t), H-5, fmt(t), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:top, y2:H-20, stroke:css('--axis')}, s);
  rows.forEach((r, i) => { const y = top + i*rowH + (rowH-16)/2, h = 16;
    const lb = r.label.replace(' (vs expected)', ''); text(s, left-8, y+h/2+4, lb.length > 44 ? lb.slice(0,43) + '…' : lb, {'text-anchor':'end', class:'lab'});
    if (r.v != null) el('path', {d: hbarPath(x(0), x(r.v), y, h), fill: r.color || css('--s1')}, s);
    if (r.v != null) text(s, Math.max(x(r.v), x(0)) + 6, y+h/2+4, fmt(r.v), {class:'val'});
    const hr = el('rect', {x:0, y:y-(rowH-h)/2, width:W, height:rowH, class:'hit', style: r.onclick ? 'cursor:pointer' : ''}, s); hit(hr, r.tip || ''); if (r.onclick) hr.addEventListener('click', r.onclick); });
}
function lines(host, series, o={}){
  const pts = series.flatMap(s => s.pts); if (!pts.length) return empty(host, o.empty || 'No data');
  const W = 600, H = 260, top = 12, bottom = 34, left = 54, right = 18, s = svg(host, W, H);
  const xs = pts.map(p => p[0]), x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const [lo, hi] = ext(pts.map(p => p[1]).concat(o.ref != null ? [o.ref] : []), false);
  const x = v => left + (v-x0)/(x1-x0)*(W-left-right), y = v => top + (hi-v)/(hi-lo)*(H-top-bottom);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, o.fmt(t), {'text-anchor':'end'}); }
  (o.xticks || ticks(x0, x1, 8)).forEach(t => text(s, x(t), H-bottom+16, o.xfmt ? o.xfmt(t) : t, {'text-anchor':'middle'}));
  if (o.xtitle) text(s, left + (W-left-right)/2, H-2, o.xtitle, {'text-anchor':'middle'});
  if (o.ref != null) el('line', {x1:left, x2:W-right, y1:y(o.ref), y2:y(o.ref), stroke:css('--axis'), 'stroke-width':1.5, 'stroke-dasharray':'4 3'}, s);
  for (const se of series){ if (!se.pts.length) continue;
    el('path', {d: se.pts.map((p,j)=>(j?'L':'M')+x(p[0]).toFixed(1)+','+y(p[1]).toFixed(1)).join(''), fill:'none', stroke:se.color, 'stroke-width':2, 'stroke-linejoin':'round'}, s);
    se.pts.forEach(p => { el('circle', {cx:x(p[0]), cy:y(p[1]), r:4, fill:se.color, stroke:css('--surface'), 'stroke-width':2}, s);
      hit(el('circle', {cx:x(p[0]), cy:y(p[1]), r:11, class:'hit'}, s), `<b>${esc(se.name)}</b><br>${o.tipx ? o.tipx(p) : p[0]}<br>${o.fmt(p[1])}${p[2] != null ? ' · ' + int(p[2]) + ' rows' : ''}`); }); }
}
function liftChart(host, full, base, rate){
  if (!full || !full.length) return empty(host, 'No hold-out predictions');
  const W = 600, H = 260, top = 12, bottom = 40, left = 54, right = 10, s = svg(host, W, H);
  const [lo, hi] = ext(full.map(r => r[2]).concat((base||[]).map(r => r[2]), [rate]));
  const y = v => top + (hi-v)/(hi-lo)*(H-top-bottom), bw = (W-left-right)/10, w = Math.min(26, bw*.55);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, pctu(t,0), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis')}, s);
  full.forEach(r => { const cx = left + bw*(r[0]-1) + bw/2; el('path', {d: vbarPath(cx-w/2, w, y(0), y(r[2])), fill:css('--s1')}, s);
    text(s, cx, H-bottom+16, r[0], {'text-anchor':'middle', class:'lab'});
    hit(el('rect', {x:cx-bw/2, y:top, width:bw, height:H-top-bottom, class:'hit'}, s), `<b>Risk decile ${r[0]}</b><br>Full model: predicted ${pctu(r[1])}, actual ${pctu(r[2])}<br>${int(r[3])} rows` + (base && base[r[0]-1] ? `<br>Baseline decile ${r[0]}: actual ${pctu(base[r[0]-1][2])}` : '')); });
  (base||[]).forEach(r => { const cx = left + bw*(r[0]-1) + bw/2; el('circle', {cx, cy:y(r[2]), r:5, fill:css('--s2'), stroke:css('--surface'), 'stroke-width':2}, s); });
  el('line', {x1:left, x2:W-right, y1:y(rate), y2:y(rate), stroke:css('--axis'), 'stroke-width':1.5, 'stroke-dasharray':'4 3'}, s);
  text(s, left + (W-left-right)/2, H-4, 'Predicted-risk decile (1 = safest, 10 = riskiest)', {'text-anchor':'middle'});
}
function sortable(t, head, rows, o={}){
  let key = o.sortKey ?? null, dir = o.sortDir ?? -1;
  function draw(){ const r = rows.slice(); if (key != null) r.sort((a, b) => { const x = a.v[key], y = b.v[key]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    t.innerHTML = '<thead><tr>' + head.map((h, i) => `<th data-i="${i}" class="${h.c||''}" title="${esc(h.t||'')}">${h.h}${key===i?(dir>0?' ▲':' ▼'):''}</th>`).join('') + '</tr></thead><tbody>' +
      r.map(x => `<tr data-k="${esc(x.k||'')}" class="${x.sel?'sel':''}" ${o.onRow?'style="cursor:pointer"':''}>` + x.c.map((c, i) => `<td class="${head[i].c||''}" ${x.st && x.st[i] ? `style="${x.st[i]}"` : ''}>${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
    t.querySelectorAll('th').forEach(th => th.onclick = () => { const i = +th.dataset.i; dir = key === i ? -dir : -1; key = i; draw(); });
    if (o.onRow) t.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => o.onRow(tr.dataset.k)); }
  draw();
}
// ------------------------------------------------------------------
const tsel = $('#target'), psel = $('#period');
tsel.innerHTML = Object.entries(R.targets).map(([k, v]) => `<option value="${k}">${esc(v.title)}</option>`).join('');
const TG = () => tsel.value, PER = () => psel.value;
const FL = Object.fromEntries(Object.values(R.features).flat().map(f => [f.f, f.label]));
const fname = f => FL[f] || f;
function header(){
  const C = R.coverage;
  $('#sub').innerHTML = `${int(C.rows)} appearances of ${int(C.players)} players in ${int(C.fixtures)} matches across ${C.leagues} competitions, ${C.from} → ${C.to}. Performance = ${esc(C.perf)}. ${C.n_features} signals per player-match; learner: ${esc(C.learner)}. Hold-out from <b>${C.holdout_from}</b>. Database <b>${esc(R.db)}</b> · ${R.generated}.`;
  if (R.demo) $('#demo').innerHTML = '<div class="demo"><b>Demo run on synthetic data.</b> The players, matches and every finding on this page are invented, generated to exercise the pipeline. Run the script against your database for real results.</div>';
  const L = R.leak, b = [];
  if (L.skipped) b.push('<span class="badge">Time-travel test skipped</span>');
  else b.push(`<span class="badge ${L.passed?'ok':'bad'}">${L.passed?'✓':'✕'} Time-travel test ${L.passed?'passed':'FAILED'}: ${L.tests.length} cutoff days, ${int(L.tests.reduce((a,t)=>a+t.rows*t.features,0))} feature values, ${L.tests.reduce((a,t)=>a+t.mismatches,0)} changed</span>`);
  b.push('<span class="badge ok">✓ Training rows purged: label window ends before the predicted month</span>', '<span class="badge">Baselines from earlier days only</span>', `<span class="badge">Retrained every ${R.params.retrain_months} months</span>`);
  if (R.alarms.length) b.push(`<span class="badge bad">⚠ Implausibly strong: ${R.alarms.map(esc).join(', ')}. Treat as a leak until proven otherwise.</span>`);
  $('#badges').innerHTML = b.join('');
}
function hero(){
  const E = R.eval[TG()], per = PER(), M = E.models[per], g = E.gain[per], T = R.targets[TG()];
  if (!M || !M.full){ $('#hero').innerHTML = '<div class="lbl">Not enough rows in this period.</div>'; $('#tiles').innerHTML = ''; return; }
  const f = M.full, b = M.baseline, d = f.auc - b.auc;
  $('#hero').innerHTML = `<div><div class="big">${num(f.auc,3)}</div><div class="lbl">AUC, ${per} · ${esc(T.title)}<br><span class="note">0.5 = coin toss, 1 = perfect</span></div></div>
    <div style="max-width:700px">${esc(T.desc)}<br><br>With every signal the model ranks a future downturn above a non-downturn ${pctu(f.auc,1)} of the time, against ${pctu(b.auc,1)} for mean reversion and context alone (${sgn(d*100,1)} points).
    The riskiest 10% of rows turn bad ${pctu(f.top10_rate,0)} of the time vs ${pctu(f.base_rate,0)} overall (${num(f.lift10,2)}× lift).
    ${g ? `Log-loss gain from the confidence signals: ${sgn(g.gain*1000,2)}×10⁻³ (95% interval ${sgn(g.lo*1000,2)} to ${sgn(g.hi*1000,2)}; ${g.lo > 0 ? '<b>unlikely to be chance</b>' : 'could be chance'}).` : ''}</div>`;
  const tiles = [['Rows scored', int(f.n), `${per} period`], ['Downturn rate', pctu(f.base_rate), 'share of rows labelled a downturn'],
    ['Baseline AUC', num(b.auc,3), 'mean reversion + context'], ['Top-10% lift', num(f.lift10,2) + '×', `${pctu(f.top10_rate,0)} vs ${pctu(f.base_rate,0)}`],
    ['Precision-recall AUC', num(f.pr_auc,3), `baseline ${num(b.pr_auc,3)} · chance ${num(f.base_rate,3)}`]];
  $('#tiles').innerHTML = tiles.map(t => `<div class="card tile"><div class="lbl">${t[0]}</div><div class="val">${t[1]}</div><div class="ci">${t[2]}</div></div>`).join('');
}
function famView(){
  const I = R.imp[TG()];
  if (!I){ empty($('#c-fam'), 'Not enough data'); empty($('#c-feat'), ''); return; }
  $('#fam-note').textContent = `A model trained on the discovery period only, explained on the hold-out (AUC ${num(I.auc,3)}). Individual signals ranked by ${I.unit}.`;
  const F = Object.entries(I.families).sort((a, b) => b[1].auc_drop - a[1].auc_drop);
  hbars($('#c-fam'), F.map(([k, v]) => ({label:k, v:v.auc_drop*100, color: v.auc_drop > 0.0005 ? css('--s1') : css('--ctrl'),
    tip:`<b>${esc(k)}</b> · ${v.n} signals<br>AUC −${num(v.auc_drop*100,2)} points when scrambled<br>log loss +${num(v.ll_inc*1000,2)}×10⁻³`})), v => num(v,1));
  const top = R.features[TG()].filter(f => f.imp != null).slice(0, 22);
  $('#feat-note').textContent = I.method === 'permutation' ? 'Bar = AUC lost (×100) when the signal is scrambled; direction from the discovery period.' : 'Bar = average absolute contribution to the log-odds of a downturn (SHAP); direction from how the contribution moves with the value.';
  const sc = I.method === 'permutation' ? 100 : 1;
  hbars($('#c-feat'), top.map(f => ({label:f.label, v:f.imp*sc, color: f.dir > 0 ? css('--s2') : css('--s1'), onclick: () => { $('#fsel').value = f.f; decView(); },
    tip:`<b>${esc(f.label)}</b><br>${esc(f.family)}<br><span class="arrow">${f.dir > 0 ? '↑ higher → more risk' : '↓ higher → less risk'}</span><br>${I.unit}: ${num(f.imp*sc,3)}<br>Alone: AUC ${num(f.auc_d,3)} discovery, ${num(f.auc_h,3)} hold-out<br>(click to chart)`})), v => num(v, 2), {left:290, rowH:24});
}
function decView(){
  const fs = R.features[TG()], sel = $('#fsel');
  const cur = sel.value; sel.innerHTML = fs.filter(f => f.dec_d).map(f => `<option value="${f.f}">${esc(f.label)}</option>`).join('');
  if (fs.some(f => f.f === cur && f.dec_d)) sel.value = cur;
  const f = fs.find(x => x.f === sel.value); const rate = R.eval[TG()].models.holdout?.full?.base_rate;
  if (!f){ empty($('#c-dec'), 'No data'); return; }
  const ser = k => (f[k] || []).filter(r => r[1] != null).map(r => [r[0]+1, r[1], r[2], r[3]]);
  lines($('#c-dec'), [{name:'Discovery', color:css('--s2'), pts:ser('dec_d')}, {name:'Hold-out', color:css('--s1'), pts:ser('dec_h')}],
    {fmt: v => pctu(v,0), ref: rate, xticks:[1,2,3,4,5,6,7,8,9,10], xtitle:'Decile of the signal (1 = lowest values)', tipx: p => `Decile ${p[0]} · median value ${num(p[3],3)}`});
  $('#dec-note').innerHTML = `<b>${esc(f.label)}</b> (${esc(f.family)}). On its own: AUC ${num(f.auc_d,3)} in discovery, ${num(f.auc_h,3)} in hold-out, ${f.udir > 0 ? 'higher values → more downturns' : 'higher values → fewer downturns'}. Data for ${pctu(f.cov,0)} of rows.`;
}
function featTable(){
  const fam = $('#ffam'), cur = fam.value;
  fam.innerHTML = '<option value="">All</option>' + R.families.map(f => `<option>${esc(f)}</option>`).join(''); fam.value = cur;
  const I = R.imp[TG()] || {}, sc = I.method === 'permutation' ? 100 : 1;
  const rows = R.features[TG()].filter(f => !fam.value || f.family === fam.value).map(f => { const same = f.auc_h != null && f.auc_d != null && (f.auc_h - .5) * (f.auc_d - .5) > 0 && f.auc_d > .5;
    return {k:f.f, v:[f.label, f.family, f.imp, f.dir, f.auc_d, f.auc_h, same, f.cov],
      c:[`${esc(f.label)}<div class="why">${esc(f.f)}</div>`, esc(f.family), num(f.imp == null ? null : f.imp*sc, 3), f.dir > 0 ? '<span class="arrow">↑</span> more risk' : '<span class="arrow">↓</span> less risk', num(f.auc_d,3), num(f.auc_h,3), same ? '✓' : '', pctu(f.cov,0)]}; });
  sortable($('#t-feat'), [{h:'Signal', c:'l'}, {h:'Family', c:'l'}, {h:'Importance'}, {h:'Higher value'}, {h:'AUC disc.', t:'univariate AUC, discovery (oriented)'}, {h:'AUC hold-out', t:'univariate AUC, hold-out, same orientation'}, {h:'Holds up'}, {h:'Coverage'}], rows,
    {sortKey:2, onRow: k => { $('#fsel').value = k; decView(); }});
}
function rolesView(){
  const I = R.imp[TG()];
  if (!I || !I.roles || !Object.keys(I.roles).length){ $('#t-roles').innerHTML = '<tr><td class="note">Not enough data</td></tr>'; return; }
  const fams = Object.entries(I.families).sort((a, b) => b[1].auc_drop - a[1].auc_drop).map(x => x[0]);
  const G = ['GK','DEF','MID','ATT'].filter(g => I.roles[g]);
  const mx = Math.max(1e-6, ...G.flatMap(g => fams.map(f => I.roles[g].families[f] || 0)));
  const head = [{h:'Position', c:'l'}, {h:'Rows'}, {h:'Downturn rate'}, {h:'AUC'}, ...fams.map(f => ({h:f.replace(' & ', ' &amp; ').split(' (')[0]}))];
  const rows = G.map(g => { const r = I.roles[g]; const st = [null, null, null, null];
    const c = [g, int(r.n), pctu(r.rate), num(r.auc,3)];
    fams.forEach(f => { const v = r.families[f] || 0, a = Math.max(0, Math.min(1, v / mx)); c.push(num(v*100, 1)); st.push(`background:color-mix(in srgb, var(--s1) ${Math.round(a*70)}%, transparent);text-align:center`); });
    return {v:[g, r.n, r.rate, r.auc, ...fams.map(f => r.families[f])], c, st}; });
  sortable($('#t-roles'), head, rows);
}
function evalView(){
  const E = R.eval[TG()], rate = E.models.holdout?.full?.base_rate;
  liftChart($('#c-lift'), E.lift, E.lift_base, rate);
  const q = E.by_quarter.filter(r => r[1] != null && r[3] >= 100);
  lines($('#c-q'), [{name:'Baseline', color:css('--s2'), pts:q.map((r, i) => [i, r[2], r[3]])}, {name:'Full model', color:css('--s1'), pts:q.map((r, i) => [i, r[1], r[3]])}],
    {fmt: v => v.toFixed(2), xticks: q.map((r, i) => i).filter((i, k, a) => a.length < 10 || k % Math.ceil(a.length/8) === 0), xfmt: i => q[i] ? q[i][0] : '', tipx: p => q[p[0]][0]});
  const rows = [];
  for (const [tg, e] of Object.entries(R.eval)) for (const pn of ['discovery','holdout']) for (const m of ['full','baseline']) { const x = e.models[pn]?.[m]; if (!x) continue;
    const g = m === 'full' ? e.gain[pn] : null;
    rows.push({sel: tg === TG() && pn === PER(), v:[R.targets[tg].title, pn, m, x.n, x.base_rate, x.auc, x.pr_auc, x.logloss, x.ll_null, x.lift10, g?.gain],
      c:[esc(R.targets[tg].title), pn, m, int(x.n), pctu(x.base_rate), `<b>${num(x.auc,3)}</b>`, num(x.pr_auc,3), num(x.logloss,4), num(x.ll_null,4), num(x.lift10,2)+'×', g ? `<span class="${cls(g.gain)}">${sgn(g.gain*1000,2)}</span> <span class="note">(${sgn(g.lo*1000,2)} to ${sgn(g.hi*1000,2)})</span>` : '']}); }
  sortable($('#t-eval'), [{h:'Target', c:'l'}, {h:'Period', c:'l'}, {h:'Model', c:'l'}, {h:'Rows'}, {h:'Rate'}, {h:'AUC'}, {h:'PR-AUC'}, {h:'Log loss'}, {h:'No-skill log loss'}, {h:'Top-10% lift'}, {h:'Gain vs baseline ×10⁻³'}], rows);
}
function watchView(){
  const W = R.watch || [], tg = TG(), role = $('#wrole').value, q = $('#wq').value.trim().toLowerCase();
  const br = R.coverage.base_rates || {};
  $('#watch-note').textContent = W.length ? `Models retrained on everything labelled so far, applied to each player's state after his latest appearance (players active in the last 60 days, ${R.params.min_history}+ rated games). Sorted by the selected target. Drivers = signals pushing this player's risk up most. Typical rates: ${Object.entries(br).map(([k, v]) => R.targets[k].title.toLowerCase() + ' ' + pctu(v,0)).join(', ')}.` : 'No active players at the end of the data.';
  const T = Object.keys(R.targets);
  const rows = W.filter(w => (!role || (w.role && ({GK:'GK',CB:'DEF',FB:'DEF',D:'DEF',DM:'MID',CM:'MID',AM:'MID',W:'ATT',ST:'ATT',U:'MID'})[w.role] === role)) && (!q || (w.player + ' ' + w.team).toLowerCase().includes(q)))
    .map(w => ({v:[w.player, w.team, w.role, w.age, w.apps, w.rating_d, ...T.map(k => w.risk[k])],
      c:[`<b>${esc(w.player)}</b><div class="why">${esc(w.league)}</div>`, esc(w.team), w.role, num(w.age,1), int(w.apps), `<span class="${cls(w.rating_d)}">${sgn(w.rating_d,2)}</span>`,
        ...T.map(k => `${k === tg ? '<b>' : ''}${pctu(w.risk[k],0)}${k === tg ? '</b>' : ''}`),
        `<div class="drv">${(w.drivers[tg] || []).map(x => `${esc(fname(x[0].replace(/^f_/, '')))} <span class="note">(${x[2] == null ? 'n/a' : num(x[2],2)})</span>`).join(' · ') || '–'}</div>`]}));
  sortable($('#t-watch'), [{h:'Player', c:'l'}, {h:'Team', c:'l'}, {h:'Role', c:'l'}, {h:'Age'}, {h:'Apps'}, {h:'Rating trend', t:'adjusted rating: recent minus usual'}, ...T.map(k => ({h:R.targets[k].title})), {h:'Main drivers (value)', c:'l'}], rows, {sortKey: 6 + T.indexOf(tg)});
}
function staticPanels(){
  const C = R.coverage, P = R.params;
  $('#safe').innerHTML = [
    'Every statistic is turned into "actual vs expected" with expectations built from time-decayed sums over <b>earlier days only</b>: league × position, the player\'s team, the opponent, home/away and the minutes he spent winning, drawing and losing. Nothing from the match day itself enters any baseline.',
    'Player features use his earlier appearances only. For the match being predicted only pre-match facts are used: line-up (started, position), home/away, opponent form, pre-kick-off odds, rest, injuries ended, transfers and rumours dated before the day.',
    `Labels look forward (next game, next ${P.k_games} appearances within ${P.horizon_days} days, next 6 team matches). Training rows are <b>purged</b>: a row only trains a model if its whole label window ended before the first day being predicted.`,
    `Walk-forward: retrained every ${P.retrain_months} months on the previous ${Math.round(P.train_window_days/365*10)/10} years; tree counts chosen on the latest 15% of training dates. Downturn thresholds frozen from data before the first predicted month.`,
    'Time-travel test: the whole pipeline is re-run with every later match deleted and the test day\'s match data blanked; features for that day must not change.',
    'Importance and the position breakdown come from a model trained on the discovery period and scored on the hold-out.'].map(x => `<li>${x}</li>`).join('');
  const L = R.leak;
  $('#t-leak').innerHTML = L.skipped ? '<tr><td class="note">Skipped</td></tr>' : '<thead><tr><th class="l">Cutoff day</th><th>Player rows</th><th>Features</th><th>Changed</th></tr></thead><tbody>' +
    L.tests.map(t => `<tr><td class="l">${t.day}</td><td>${int(t.rows)}</td><td>${t.features}</td><td class="${t.mismatches?'neg':'pos'}">${t.mismatches}${t.worst && t.worst.length ? ' <span class="note">' + t.worst.map(esc).join(', ') + '</span>' : ''}</td></tr>`).join('') + '</tbody>';
  const cv = [['Appearances', int(C.rows)], ['Players', int(C.players)], ['Matches', int(C.fixtures)], ['Competitions', C.leagues], ['Period', `${C.from} → ${C.to}`],
    ['Hold-out from', C.holdout_from], ['Game state known (goal timeline)', pctu(C.game_state)], ['Substitution events available', pctu(C.subs_known)],
    ['Signals per player-match', `${C.n_features} (baseline set ${C.n_base})`], ['Learner', esc(C.learner)], ['Run time', `${Math.floor(C.runtime_s/60)} min ${C.runtime_s%60} s`],
    ['Indicators not available in this database', C.missing_metrics.length ? esc(C.missing_metrics.join(', ')) : 'none']];
  $('#t-cov').innerHTML = '<tbody>' + cv.map(r => `<tr><td class="l">${r[0]}</td><td style="white-space:normal">${r[1]}</td></tr>`).join('') + '</tbody>';
  $('#t-met').innerHTML = '<tbody>' + Object.entries(C.metrics).sort((a,b) => b[1]-a[1]).map(([k, v]) => `<tr><td class="l">${esc(k)}</td><td>${pctu(v,0)}</td></tr>`).join('') + '</tbody>';
}
function draw(){ hero(); famView(); decView(); featTable(); rolesView(); evalView(); watchView(); }
tsel.onchange = draw; psel.onchange = () => { hero(); evalView(); };
$('#fsel').onchange = decView; $('#ffam').onchange = featTable; $('#wrole').onchange = watchView; $('#wq').oninput = watchView;
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; draw(); };
header(); staticPanels(); draw();
</script></body></html>
'''

if __name__ == "__main__":
    main()
