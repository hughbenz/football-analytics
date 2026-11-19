#!/usr/bin/env python3
"""
backtest_models.py - walk-forward prediction models vs the betting market
===========================================================================

Builds several football prediction models from the Sportmonks PostgreSQL
database, backtests them match by match against real outcomes and closing odds,
tests a set of pre-declared betting "angles", and writes everything to one HTML
dashboard (plus an optional CSV of every feature and prediction).

NO LOOK-AHEAD - how the script guarantees it
--------------------------------------------
1. Features come from a day-by-day state machine. For each calendar day it first
   writes the feature rows for that day's matches from the current state, and only
   then reads that day's results into the state. A match can never see its own
   result, stats, xG, ratings, cards, or anything played later that day.
2. Pre-match information is used only where it is genuinely known before kickoff:
   past fixtures (rest days, congestion - never future fixtures: knockout rounds and
   play-offs only appear in the fixture list once results are known), venues, the referee appointment,
   injuries/suspensions, weather forecast, starting line-ups (published ~1h before),
   and odds whose last update is not after kickoff (later prices are discarded).
3. Season-level tables (standings, team/player/coach/referee season stats,
   topscorers) are NOT read: they contain future matches. League tables are
   rebuilt from results as they happen.
4. Models are retrained every month using only matches before that month; the
   stacked ensemble only learns from earlier out-of-sample predictions.
5. Automatic time-travel test: the engine is re-run with every result from a
   cutoff date onwards deleted and every later fixture removed. Feature rows for matches on the cutoff date must be
   identical. The verdict is shown on the dashboard.
6. Bet thresholds are chosen on the discovery period only, then applied unchanged
   to the hold-out period. Hold-out figures are the honest ones.

Models (1X2 unless noted)
-------------------------
  market        de-vigged consensus of every bookmaker's closing price (benchmark)
  sportmonks    Sportmonks' own pre-match probabilities (benchmark, if loaded)
  elo           goal-difference Elo across all competitions -> multinomial logit
  poisson       time-decayed attack/defence goal ratings (Dixon-Coles style)
  xg_poisson    the same ratings fitted on expected goals instead of goals
  fundamentals  ridge multinomial logit on 100+ features, no odds at all
  market_plus   market log-odds as an anchor + ridge-shrunk features: "where is the
                market wrong?" (main edge hunter)
  gbm           gradient boosting on the same inputs (needs scikit-learn)
  stack         learns how to weight the models from their earlier OOS predictions
  *_open        same idea priced at the OPENING line without line-up info; scored
                by closing-line value (CLV), the least noisy measure of edge
  Over/Under 2.5: market, poisson, xg_poisson, ou_plus, gbm_ou

Usage
-----
  Put next to load_league.py (uses load_league.DATABASE), or pass --dsn.
    pip install pandas numpy scipy "psycopg[binary]" scikit-learn
    python3 backtest_models.py                      # everything
    python3 backtest_models.py --leagues 8,82 --export-csv predictions.csv
  Options: --holdout-from 2024-07-01  --burn-in-days 365  --retrain-months 1
           --no-gbm  --skip-leak-test  --cache .cache  --max-odds 10
"""
from __future__ import annotations

import argparse
import bisect
import html as htmlmod
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import warnings


# =============================================================================
# Self-install: missing packages are installed into the Python running this
# script (or, if that Python is locked, into a .venv next to the script).
# Set BACKTEST_NO_INSTALL=1 to turn this off.
# =============================================================================
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy"}          # import name -> pip name
OPTIONAL = {"sklearn": "scikit-learn", "threadpoolctl": "threadpoolctl", "psutil": "psutil"}


def ensure_packages() -> None:
    import importlib
    import importlib.util
    import shlex
    missing = lambda d: [pip for mod, pip in d.items() if importlib.util.find_spec(mod) is None]  # noqa: E731
    req, opt = missing(REQUIRED), missing(OPTIONAL)
    if importlib.util.find_spec("psycopg") is None and importlib.util.find_spec("psycopg2") is None:
        opt.append("psycopg[binary]")            # without it the script falls back to the psql command
    if not req and not opt:
        return
    hint = lambda pk: f"{shlex.quote(sys.executable)} -m pip install " + " ".join(shlex.quote(p) for p in pk)  # noqa: E731
    if os.environ.get("BACKTEST_NO_INSTALL"):
        if req:
            sys.exit(f"Missing packages: {' '.join(req)}\nInstall with:  {hint(req + opt)}")
        return

    def pip(py, pk):
        r = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *pk], capture_output=True, text=True)
        if r.returncode != 0 and "No module named pip" in (r.stderr or ""):
            subprocess.run([py, "-m", "ensurepip", "--upgrade"], capture_output=True)
            r = subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "-q", *pk], capture_output=True, text=True)
        return r.returncode == 0, (r.stderr or r.stdout or "").strip()

    print(f"Installing missing packages into {sys.executable}: {', '.join(req + opt)} ...", file=sys.stderr, flush=True)
    ok, err = pip(sys.executable, req + opt)
    if not ok and "externally-managed-environment" in err:
        # Homebrew / system Python refuses installs: use (or create) a virtual environment beside the script
        venv = Path(__file__).resolve().parent / ".venv"
        vpy = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if not vpy.exists():
            print(f"This Python is managed by the system; creating a virtual environment at {venv} ...", file=sys.stderr, flush=True)
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        everything = list(REQUIRED.values()) + list(OPTIONAL.values()) + ["psycopg[binary]"]
        vok, verr = pip(str(vpy), everything)
        if not vok:
            vok, verr = pip(str(vpy), list(REQUIRED.values()))
        if vok:
            print(f"Re-running with {vpy}  (next time you can run it directly with that Python)", file=sys.stderr, flush=True)
            os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])
        ok, err = False, verr
    if not ok and opt:                       # retry without the optional extras
        ok2, err2 = pip(sys.executable, req) if req else (True, "")
        if ok2:
            print(f"Optional packages not installed ({', '.join(opt)}); continuing without them.", file=sys.stderr, flush=True)
            ok, err = True, ""
        else:
            err = err2 or err
    if not ok:
        sys.exit(f"Automatic install failed:\n{err[-1200:]}\n\nInstall by hand:  {hint(req + opt)}")
    importlib.invalidate_caches()


if __name__ == "__main__":      # worker processes import this module too; only the main run installs
    ensure_packages()
    if "--install-only" in sys.argv:    # used by tactical_model.py
        sys.exit(0)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

T0 = time.time()
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)   # no rating history yet -> NaN, handled


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def fmt_dur(sec) -> str:
    if sec is None or not math.isfinite(sec):
        return "estimating..."
    sec = int(max(0, sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else (f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s")


class Progress:
    """Prints elapsed time, stage progress and time remaining every `every` seconds."""
    # share of total run time each stage typically takes (measured on test runs)
    WEIGHTS = {"reading database": 4, "building features": 5, "time-travel test": 4, "training models": 82, "scoring": 5}

    def __init__(self, every: float, skip=()):
        import threading
        self.every, self.t0 = every, time.time()
        self.plan = [s for s in self.WEIGHTS if s not in skip]
        self.stage, self.total, self.done, self.ts, self.done_w, self.note = None, 1.0, 0.0, self.t0, 0.0, ""
        self.lock, self.stop = threading.Lock(), threading.Event()
        if every > 0:
            threading.Thread(target=self._loop, daemon=True).start()

    def start(self, stage: str, total: float):
        with self.lock:
            if self.stage in self.WEIGHTS:
                self.done_w += self.WEIGHTS[self.stage]
            self.stage, self.total, self.done, self.ts, self.note = stage, max(float(total), 1e-9), 0.0, time.time(), ""

    def add(self, n: float = 1.0, note: str = ""):
        with self.lock:
            self.done = min(self.total, self.done + n)
            if note:
                self.note = note

    def close(self):
        self.stop.set()

    def line(self) -> str:
        with self.lock:
            now = time.time()
            f = self.done / self.total
            el = now - self.ts
            stage_left = el * (1 - f) / f if f >= (0.10 if self.stage == "training models" else 0.03) and el > 5 else None
            tot_w = sum(self.WEIGHTS[s] for s in self.plan) or 1
            overall = (self.done_w + self.WEIGHTS.get(self.stage, 0) * f) / tot_w
            run_left = None
            if self.stage == "training models" and stage_left is not None:
                after = sum(self.WEIGHTS[s] for s in self.plan[self.plan.index(self.stage) + 1:])
                run_left = stage_left * (1 + after / self.WEIGHTS[self.stage])   # training dominates: scale from its own pace
            elif overall >= 0.05:
                run_left = (now - self.t0) * (1 - overall) / overall
            note = f" ({self.note})" if self.note else ""
            return (f"⏱  {fmt_dur(now - self.t0)} elapsed · {self.stage} {f * 100:4.1f}%{note} · "
                    f"stage ~{fmt_dur(stage_left)} left · whole run ~{fmt_dur(run_left)} left · {overall * 100:.0f}% overall")

    def _loop(self):
        while not self.stop.wait(self.every):
            if self.stage:
                print(self.line(), file=sys.stderr, flush=True)


PROGRESS: "Progress | None" = None


def progress(n: float = 1.0, note: str = ""):
    if PROGRESS is not None:
        PROGRESS.add(n, note)


# =============================================================================
# Database access (psycopg 3 -> psycopg2 -> psql CLI)
# =============================================================================
def default_dsn() -> tuple[str, str]:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"], "DATABASE_URL"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import load_league  # type: ignore
        cfg = getattr(load_league, "DATABASE", None)
        if isinstance(cfg, dict):
            keymap = {"database": "dbname", "db": "dbname", "username": "user", "passwd": "password"}
            cfg = " ".join(f"{keymap.get(k, k)}='{str(v).replace(chr(39), chr(92) + chr(39))}'"
                           for k, v in cfg.items() if v is not None)
        if cfg:
            return str(cfg), "load_league.DATABASE"
    except Exception:
        pass
    return "", "libpq defaults (PG* environment variables)"


DATE_COLS = {"starting_at", "start_date", "end_date"}


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in DATE_COLS:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            vals = set(df[c].dropna().astype(str).str.lower().unique()[:50])
            if vals and vals <= {"t", "f", "true", "false"}:
                df[c] = df[c].astype(str).str.lower().map({"t": True, "true": True, "f": False, "false": False})
                continue
            conv = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().any() and conv.notna().sum() == df[c].notna().sum():
                df[c] = conv
    return df


class DB:
    def __init__(self, dsn: str | None):
        source = "--dsn"
        if not dsn:
            dsn, source = default_dsn()
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
        name = self.query("SELECT current_database() AS db").db.iloc[0]
        log(f"database driver: {self.kind} · connected to '{name}' (settings from {source})")
        ok = self.query("SELECT (to_regclass('football.fixtures') IS NOT NULL)::int AS ok").ok.iloc[0]
        if int(ok) != 1:
            sys.exit(f"Database '{name}' has no football.fixtures table. Pass --dsn, set DATABASE_URL, "
                     "or put this script next to load_league.py.")
        self.name = name

    def query(self, sql: str) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        if self.kind == "psql":
            cmd = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1"] + ([self.dsn] if self.dsn else [])
            res = subprocess.run(cmd + ["-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"], capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode().strip())
            data = res.stdout.decode()
            return _coerce(pd.read_csv(io.StringIO(data), low_memory=False)) if data.strip() else pd.DataFrame()
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        finally:
            self.conn.rollback()
        return _coerce(pd.DataFrame(rows, columns=cols))

    def clone(self) -> "DB":
        """A second connection with the same settings (connections are not shared between threads)."""
        c = DB.__new__(DB)
        c.dsn, c.kind, c.name, c.conn = self.dsn, self.kind, getattr(self, "name", ""), None
        if self.kind == "psycopg":
            import psycopg  # type: ignore
            c.conn = psycopg.connect(self.dsn)
        elif self.kind == "psycopg2":
            import psycopg2  # type: ignore
            c.conn = psycopg2.connect(self.dsn)
        return c

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass

    def exists(self, table: str) -> bool:
        return int(self.query(f"SELECT (to_regclass('{table}') IS NOT NULL)::int AS ok").ok.iloc[0]) == 1


# =============================================================================
# SQL extracts. Odds rows updated after kickoff are excluded (post-match prices).
# =============================================================================
FINISHED = "('FT', 'AET', 'FT_PEN')"

SQL = {
"fixtures": f"""
    WITH sc AS (
        SELECT fixture_id,
               max(goals) FILTER (WHERE location = 'home' AND description = 'CURRENT')  AS h_cur,
               max(goals) FILTER (WHERE location = 'away' AND description = 'CURRENT')  AS a_cur,
               max(goals) FILTER (WHERE location = 'home' AND description = '2ND_HALF') AS h_90,
               max(goals) FILTER (WHERE location = 'away' AND description = '2ND_HALF') AS a_90
        FROM football.fixture_scores WHERE description IN ('CURRENT', '2ND_HALF') GROUP BY fixture_id)
    SELECT f.fixture_id, f.league_id, l.name AS league,
           CASE WHEN lower(COALESCE(l.type, 'league')) = 'league'
                AND COALESCE(l.sub_type, '') !~* '(cup|play|international|friendly)' THEN 'league' ELSE 'cup' END AS league_type,
           f.season_id, s.name AS season, f.starting_at, f.venue_id,
           COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
           ht.name AS home, at.name AS away, st.developer_name AS state,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN COALESCE(sc.h_90, sc.h_cur) ELSE sc.h_cur END AS hg,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN COALESCE(sc.a_90, sc.a_cur) ELSE sc.a_cur END AS ag
    FROM football.fixtures f
    JOIN core.states st ON st.state_id = f.state_id
    JOIN sc ON sc.fixture_id = f.fixture_id
    LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
    LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
    LEFT JOIN football.leagues l ON l.league_id = f.league_id
    LEFT JOIN football.seasons s ON s.season_id = f.season_id
    LEFT JOIN football.teams ht ON ht.team_id = COALESCE(f.home_team_id, hp.team_id)
    LEFT JOIN football.teams at ON at.team_id = COALESCE(f.away_team_id, ap.team_id)
    WHERE st.developer_name IN {FINISHED}""",
# The published fixture list (any state) - known in advance, used for rest / next match
"schedule": """
    SELECT f.fixture_id, f.starting_at, f.league_id, f.season_id,
           CASE WHEN lower(COALESCE(l.type, 'league')) = 'league'
                AND COALESCE(l.sub_type, '') !~* '(cup|play|international|friendly)' THEN 'league' ELSE 'cup' END AS league_type,
           COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id
    FROM football.fixtures f
    LEFT JOIN football.leagues l ON l.league_id = f.league_id
    LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
    LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
    WHERE f.starting_at IS NOT NULL""",
"venues": "SELECT venue_id, latitude AS lat, longitude AS lon FROM football.venues WHERE latitude IS NOT NULL",
"team_venues": "SELECT team_id, venue_id FROM football.teams WHERE venue_id IS NOT NULL",
"teamstats": """
    SELECT s.fixture_id, s.team_id,
        max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_ON_TARGET')   AS sot,
        max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_TOTAL')       AS shots,
        max(s.value) FILTER (WHERE t.developer_name = 'DANGEROUS_ATTACKS') AS da,
        max(s.value) FILTER (WHERE t.developer_name = 'BALL_POSSESSION')   AS poss,
        max(s.value) FILTER (WHERE t.developer_name = 'CORNERS')           AS corners,
        max(s.value) FILTER (WHERE t.developer_name = 'BIG_CHANCES_CREATED') AS big_chances
    FROM football.fixture_statistics s JOIN core.types t ON t.type_id = s.type_id
    WHERE t.developer_name IN ('SHOTS_ON_TARGET','SHOTS_TOTAL','DANGEROUS_ATTACKS','BALL_POSSESSION','CORNERS','BIG_CHANCES_CREATED')
    GROUP BY 1, 2""",
"xg": """
    SELECT x.fixture_id, x.team_id, max(x.value) AS xg
    FROM football.fixture_xg x JOIN core.types t ON t.type_id = x.type_id
    WHERE t.developer_name = 'EXPECTED_GOALS' GROUP BY 1, 2""",
"cards": """
    SELECT e.fixture_id, e.team_id,
           count(*) FILTER (WHERE t.developer_name = 'YELLOWCARD') AS yellows,
           count(*) FILTER (WHERE t.developer_name IN ('REDCARD', 'YELLOWREDCARD')) AS reds
    FROM football.fixture_events e JOIN core.types t ON t.type_id = e.type_id
    WHERE t.developer_name IN ('YELLOWCARD', 'REDCARD', 'YELLOWREDCARD') AND e.rescinded IS NOT TRUE AND e.team_id IS NOT NULL
    GROUP BY 1, 2""",
"lineups": """
    WITH d AS (
        SELECT ld.lineup_id,
               max(ld.value) FILTER (WHERE t.developer_name = 'MINUTES_PLAYED') AS minutes,
               max(ld.value) FILTER (WHERE t.developer_name = 'RATING') AS rating
        FROM football.fixture_lineup_details ld JOIN core.types t ON t.type_id = ld.type_id
        WHERE t.developer_name IN ('MINUTES_PLAYED', 'RATING')
        GROUP BY ld.lineup_id)
    SELECT l.fixture_id, l.team_id, l.player_id, (lt.developer_name = 'LINEUP')::int AS started, d.minutes, d.rating
    FROM football.fixture_lineups l
    LEFT JOIN core.types lt ON lt.type_id = l.type_id
    LEFT JOIN d ON d.lineup_id = l.lineup_id
    WHERE l.player_id IS NOT NULL AND l.team_id IS NOT NULL""",
"coaches": "SELECT fixture_id, team_id, coach_id FROM football.fixture_coaches WHERE team_id IS NOT NULL",
"referees": """
    SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
    FROM football.fixture_officials fo LEFT JOIN core.types t ON t.type_id = fo.type_id
    WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
    ORDER BY fo.fixture_id, fo.fixture_official_id""",
"sidelined": """
    SELECT fs.fixture_id, COALESCE(fs.team_id, s.team_id) AS team_id, s.player_id, s.category
    FROM football.fixture_sidelined fs JOIN football.sidelined s ON s.sidelined_id = fs.sidelined_id
    WHERE s.player_id IS NOT NULL""",
"weather": """
    SELECT fixture_id, wind_speed, temperature_day AS temp, humidity, lower(coalesce(description, '')) AS wdesc,
           lower(coalesce(metric, '')) AS metric
    FROM football.fixture_weather""",
"rivals": "SELECT team_id, rival_team_id FROM football.rivals",
"predictions": """
    SELECT p.fixture_id, avg(core.num(p.predictions ->> 'home')) AS sm_h,
           avg(core.num(p.predictions ->> 'draw')) AS sm_d, avg(core.num(p.predictions ->> 'away')) AS sm_a
    FROM football.predictions p JOIN core.types t ON t.type_id = p.type_id
    WHERE t.developer_name = 'FULLTIME_RESULT_PROBABILITY' GROUP BY p.fixture_id""",
}

SIDE_1X2 = """CASE WHEN lower(coalesce(o.label,'')) IN ('home','1') OR lower(coalesce({ol},'')) = '1' THEN 'H'
                   WHEN lower(coalesce(o.label,'')) IN ('draw','x') OR lower(coalesce({ol},'')) = 'x' THEN 'D'
                   WHEN lower(coalesce(o.label,'')) IN ('away','2') OR lower(coalesce({ol},'')) = '2' THEN 'A' END"""
SIDE_OU = """CASE WHEN lower(coalesce(o.label,'')) = 'over' THEN 'O' WHEN lower(coalesce(o.label,'')) = 'under' THEN 'U' END"""


def odds_sql(args, kind: str, opening: bool) -> str:
    if kind == "1x2":
        mk = f"(market_id = {args.market_1x2} OR developer_name = 'FULLTIME_RESULT')"
        sides, extra = ("H", "D", "A"), ""
    else:
        mk = f"(market_id = {args.market_ou})"
        sides, extra = ("O", "U"), " AND core.num(o.total) = 2.5"
    side_pm = (SIDE_1X2.format(ol="o.original_label") if kind == "1x2" else SIDE_OU)
    side_pr = (SIDE_1X2.format(ol="NULL") if kind == "1x2" else SIDE_OU)
    piv = ", ".join(f"avg(price) FILTER (WHERE side = '{s}') AS o_{s.lower()}" for s in sides)
    if opening:
        return f"""
        WITH mk AS (SELECT market_id FROM odds.markets WHERE {mk}),
        po AS (SELECT o.odd_id, o.fixture_id, o.bookmaker_id, {side_pr} AS side
               FROM odds.premium_odds o WHERE o.market_id IN (SELECT market_id FROM mk){extra}),
        h AS (SELECT DISTINCT ON (h.odd_id) h.odd_id, h.decimal_odds AS price
              FROM odds.premium_odds_history h JOIN po ON po.odd_id = h.odd_id
              JOIN football.fixtures f ON f.fixture_id = po.fixture_id
              WHERE h.decimal_odds > 1 AND h.bookmaker_update < f.starting_at
              ORDER BY h.odd_id, h.bookmaker_update)
        SELECT po.fixture_id, po.bookmaker_id, {piv}
        FROM h JOIN po ON po.odd_id = h.odd_id WHERE side IS NOT NULL GROUP BY 1, 2"""
    return f"""
    WITH mk AS (SELECT market_id FROM odds.markets WHERE {mk}),
    o AS (
        SELECT o.fixture_id, o.bookmaker_id, o.decimal_odds AS price, {side_pm} AS side
        FROM odds.prematch_odds o JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE o.market_id IN (SELECT market_id FROM mk) AND o.decimal_odds > 1{extra}
          AND (o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at)
        UNION ALL
        SELECT o.fixture_id, o.bookmaker_id, o.decimal_odds, {side_pr}
        FROM odds.premium_odds o JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE o.market_id IN (SELECT market_id FROM mk) AND o.decimal_odds > 1{extra}
          AND (o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at))
    SELECT o.fixture_id, o.bookmaker_id, b.name AS bookmaker, {piv}
    FROM o LEFT JOIN odds.bookmakers b ON b.bookmaker_id = o.bookmaker_id
    WHERE side IS NOT NULL GROUP BY 1, 2, 3"""


def extract(db: DB, args) -> dict[str, pd.DataFrame]:
    """Runs the extracts concurrently on separate connections (the slow odds queries overlap)."""
    import concurrent.futures as cf
    queries = dict(SQL)
    queries["odds_1x2"] = odds_sql(args, "1x2", False)
    queries["open_1x2"] = odds_sql(args, "1x2", True)
    queries["odds_ou"] = odds_sql(args, "ou", False)
    queries["open_ou"] = odds_sql(args, "ou", True)
    required = {"fixtures", "schedule", "odds_1x2"}
    cache = Path(args.cache) if args.cache else None
    out, todo = {}, []
    for name, sql in queries.items():
        path = cache / f"bt_{name}.pkl" if cache else None
        if path and path.exists() and not args.refresh:
            out[name] = pd.read_pickle(path)
            log(f"{name:12s} {len(out[name]):>11,} rows (cache)")
        else:
            todo.append((name, sql, path))
    # biggest first: odds, line-ups, statistics
    weight = {"odds_1x2": 9, "odds_ou": 8, "open_1x2": 7, "open_ou": 6, "lineups": 5, "teamstats": 4, "cards": 3}
    todo.sort(key=lambda t: -weight.get(t[0], 0))
    jobs = max(1, min(len(todo), args.db_jobs or max(2, min(6, usable_cpus()))))
    if todo:
        log(f"reading {len(todo)} extracts with {jobs} parallel database connection(s) ...")
    if PROGRESS:
        PROGRESS.start("reading database", max(1, sum(weight.get(t[0], 0) + 1 for t in todo)))

    def run(name, sql):
        conn = db.clone()
        try:
            return conn.query(sql), None
        except Exception as e:  # noqa: BLE001
            return None, e
        finally:
            conn.close()

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run, name, sql): (name, path) for name, sql, path in todo}
        for f in cf.as_completed(futs):
            name, path = futs[f]
            res, err = f.result()
            if err is not None:
                if name in required:
                    sys.exit(f"Required query '{name}' failed: {err}")
                log(f"{name}: not available ({str(err).splitlines()[0][:110]}) - continuing without it")
                res = pd.DataFrame()
            out[name] = res
            progress(weight.get(name, 0) + 1, f"{name} done")
            log(f"{name:12s} {len(res):>11,} rows")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                res.to_pickle(path)
    return {k: out[k] for k in queries}


# =============================================================================
# Market: de-vigged consensus, prices, microstructure
# =============================================================================
def devig_books(df: pd.DataFrame, sides: list[str]) -> pd.DataFrame:
    cols = [f"o_{s}" for s in sides]
    df = df.dropna(subset=cols)
    df = df[(df[cols] > 1).all(axis=1)]
    inv = 1 / df[cols].to_numpy(float)
    over = inv.sum(axis=1)
    df = df.assign(overround=over - 1, **{f"q_{s}": inv[:, i] / over for i, s in enumerate(sides)})
    return df[(df.overround > -0.02) & (df.overround < 0.35)]


def market_table(d: dict, args) -> pd.DataFrame:
    sharp = args.sharp_bookmaker.lower()
    out = []
    for kind, sides in (("1x2", ["h", "d", "a"]), ("ou", ["o", "u"])):
        c = d.get(f"odds_{kind}", pd.DataFrame())
        if c.empty:
            continue
        c = devig_books(c, sides)
        c["is_sharp"] = c.bookmaker.fillna("").str.lower().str.contains(sharp) if sharp else False
        g = c.groupby("fixture_id")
        pre = "" if kind == "1x2" else "ou_"
        m = pd.DataFrame({f"{pre}n_books": g.size(), f"{pre}overround": g.overround.mean()})
        for s in sides:
            m[f"{pre}p_{s}"] = g[f"q_{s}"].mean()
            m[f"{pre}avg_{s}"] = g[f"o_{s}"].mean()
            m[f"{pre}best_{s}"] = g[f"o_{s}"].max()
        tot = m[[f"{pre}p_{s}" for s in sides]].sum(axis=1)
        for s in sides:
            m[f"{pre}p_{s}"] /= tot
        m[f"{pre}disp"] = g[f"q_{sides[0]}"].std()
        sh = c[c.is_sharp].groupby("fixture_id")
        if len(c[c.is_sharp]):
            for s in sides:
                m[f"{pre}pin_{s}"] = sh[f"o_{s}"].mean()
                m[f"{pre}pinp_{s}"] = sh[f"q_{s}"].mean()
        soft = c[~c.is_sharp].groupby("fixture_id")
        for s in sides:
            m[f"{pre}bestsoft_{s}"] = soft[f"o_{s}"].max()
        o = d.get(f"open_{kind}", pd.DataFrame())
        if not o.empty:
            o = devig_books(o, sides)
            go = o.groupby("fixture_id")
            for s in sides:
                m[f"{pre}open_p_{s}"] = go[f"q_{s}"].mean()
                m[f"{pre}open_{s}"] = go[f"o_{s}"].mean()
        out.append(m)
    mk = pd.concat(out, axis=1) if out else pd.DataFrame()
    return mk


# =============================================================================
# Helpers
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    r = math.radians
    a = math.sin(r(lat2 - lat1) / 2) ** 2 + math.cos(r(lat1)) * math.cos(r(lat2)) * math.sin(r(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1, a)))


GOALS = np.arange(11)
LOGFACT = np.array([math.lgamma(k + 1) for k in GOALS])


def poisson_probs(lh: np.ndarray, la: np.ndarray, rho: float = 0.0):
    """1X2 + P(over 2.5) from independent Poisson goals (optional Dixon-Coles rho)."""
    lh, la = np.clip(lh, 0.05, 6), np.clip(la, 0.05, 6)
    ph = np.exp(GOALS[None, :] * np.log(lh[:, None]) - lh[:, None] - LOGFACT[None, :])
    pa = np.exp(GOALS[None, :] * np.log(la[:, None]) - la[:, None] - LOGFACT[None, :])
    m = ph[:, :, None] * pa[:, None, :]
    if rho:
        m[:, 0, 0] *= 1 - lh * la * rho
        m[:, 0, 1] *= 1 + lh * rho
        m[:, 1, 0] *= 1 + la * rho
        m[:, 1, 1] *= 1 - rho
    tot = m.sum(axis=(1, 2))
    i, j = np.meshgrid(GOALS, GOALS, indexing="ij")
    p_h = (m * (i > j)).sum(axis=(1, 2)) / tot
    p_d = (m * (i == j)).sum(axis=(1, 2)) / tot
    p_o = (m * (i + j > 2)).sum(axis=(1, 2)) / tot
    return np.c_[p_h, p_d, 1 - p_h - p_d], p_o


class EW:
    """Bias-corrected exponentially weighted mean."""
    __slots__ = ("s", "w")

    def __init__(self):
        self.s, self.w = 0.0, 0.0

    def add(self, x, a):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.s = (1 - a) * self.s + a * x
        self.w = (1 - a) * self.w + a

    def get(self):
        return self.s / self.w if self.w > 1e-9 else np.nan


A_SHORT, A_LONG = 1 - 0.5 ** (1 / 6), 1 - 0.5 ** (1 / 25)
EW_METRICS = ["gf", "ga", "xgf", "xga", "sotf", "sota", "shf", "sha", "daf", "daa", "poss", "corf", "cora",
              "pts", "expts", "luck", "dluck", "cards", "bigf"]


# =============================================================================
# The leakage-proof feature engine
# =============================================================================
class Engine:
    """Walks through the calendar. emit() reads only pre-match information and the
    state; update() is the only place results enter the state, and it runs after
    every match of the day has been emitted."""

    def __init__(self, d: dict, mk: pd.DataFrame, args):
        self.args = args
        fx = d["fixtures"].dropna(subset=["home_id", "away_id", "starting_at"]).copy()
        fx[["home_id", "away_id"]] = fx[["home_id", "away_id"]].astype("int64")
        fx["day"] = fx.starting_at.dt.tz_convert(None).dt.normalize()
        self.fx = fx.sort_values("starting_at").reset_index(drop=True)
        self.mkp = {}
        if len(mk) and "p_h" in mk:
            ok = mk[["p_h", "p_d", "p_a"]].dropna()
            self.mkp = {int(k): (a, b, c) for k, a, b, c in zip(ok.index, ok.p_h, ok.p_d, ok.p_a)}
        # pre-match information
        sch = d["schedule"].dropna(subset=["home_id", "away_id"])
        self.team_sched = defaultdict(list)
        for r in sch.itertuples():
            for t in (int(r.home_id), int(r.away_id)):
                self.team_sched[t].append((int(r.starting_at.timestamp()), r.league_type != "league"))
        for t in self.team_sched:
            self.team_sched[t].sort()
        # NB: only fixtures BEFORE the current kickoff are ever read from the schedule. Future fixtures
        # are not reliably known in advance (knockout rounds, play-offs, rearranged games appear only
        # once results are in), so nothing - "next match", "matches remaining" - is derived from them.
        ven = {int(r.venue_id): (float(r.lat), float(r.lon)) for r in d["venues"].itertuples()} if len(d["venues"]) else {}
        self.venue = ven
        self.team_venue = {int(r.team_id): int(r.venue_id) for r in d["team_venues"].itertuples()} if len(d["team_venues"]) else {}
        self.coach = {(int(r.fixture_id), int(r.team_id)): int(r.coach_id) for r in d["coaches"].itertuples()} if len(d["coaches"]) else {}
        self.ref = {int(r.fixture_id): int(r.referee_id) for r in d["referees"].itertuples()} if len(d["referees"]) else {}
        self.sidelined = defaultdict(list)
        for r in d["sidelined"].itertuples() if len(d["sidelined"]) else []:
            if pd.notna(r.team_id):
                self.sidelined[(int(r.fixture_id), int(r.team_id))].append(int(r.player_id))
        self.weather = {int(r.fixture_id): r for r in d["weather"].itertuples()} if len(d["weather"]) else {}
        self.rivals = {(int(r.team_id), int(r.rival_team_id)) for r in d["rivals"].itertuples()} if len(d["rivals"]) else set()
        self.preds = d["predictions"].set_index("fixture_id") if len(d["predictions"]) else pd.DataFrame()
        lu = d["lineups"]
        self.starters, self.apps = defaultdict(list), defaultdict(list)
        if len(lu):
            lu = lu.dropna(subset=["fixture_id", "team_id", "player_id"])
            for fid, tid, pid, st, mins, rat in zip(lu.fixture_id.astype("int64"), lu.team_id.astype("int64"), lu.player_id.astype("int64"),
                                                    lu.started.fillna(0), lu.minutes, lu.rating):
                if st == 1:
                    self.starters[(fid, tid)].append(pid)             # pre-match (line-up announcement)
                self.apps[(fid, tid)].append((pid, st, mins, rat))     # result data (minutes, rating)
        # result information (only read in update)
        self.tstats = {}
        if len(d["teamstats"]):
            tsd = d["teamstats"].dropna(subset=["fixture_id", "team_id"])
            for r in tsd.to_dict("records"):
                self.tstats[(int(r.pop("fixture_id")), int(r.pop("team_id")))] = r
        self.xg = {(int(r.fixture_id), int(r.team_id)): r.xg for r in d["xg"].itertuples()} if len(d["xg"]) else {}
        self.cards = {(int(r.fixture_id), int(r.team_id)): (r.yellows or 0) + 2 * (r.reds or 0) for r in d["cards"].itertuples()} if len(d["cards"]) else {}
        self.results = {int(r.fixture_id): (r.hg, r.ag) for r in self.fx.itertuples()}
        self.reset_state()

    def reset_state(self):
        self.elo = {}
        self.ew = defaultdict(lambda: {m: EW() for m in EW_METRICS})
        self.ewl = defaultdict(lambda: {m: EW() for m in ("gf", "ga", "xgf", "xga")})
        self.table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # season -> team -> [pts, gd, gf, played]
        self.league_teams = defaultdict(set)   # (league, season) -> teams
        self.league_seasons = defaultdict(list)
        self.pshare = defaultdict(dict)        # team -> player -> EW share
        self.prating = {}                      # player -> EW
        self.team_xi_rating = defaultdict(EW)
        self.coach_n = defaultdict(int)
        self.coach_last = {}
        self.coach_first = {}
        self.expo = defaultdict(deque)
        self.ref_ew = defaultdict(lambda: {"home": EW(), "goals": EW(), "cards": EW(), "n": 0})
        self.league_ew = defaultdict(lambda: {"home": EW(), "draw": EW(), "goals": EW()})
        self.h2h = defaultdict(lambda: deque(maxlen=6))
        self.store = defaultdict(list)         # league -> [(daynum, home, away, hg, ag, hxg, axg)]
        self.max_result_day = None
        self.team_league = {}

    # ---------- pre-match ----------
    def sched_feats(self, team, ts):
        """Rest and congestion from fixtures that kicked off BEFORE this one (never later ones)."""
        s = self.team_sched.get(team, [])
        i = bisect.bisect_left(s, (ts, False))          # entries before index i started earlier
        prev = s[i - 1] if i > 0 else None
        n14 = sum(1 for k in range(max(0, i - 8), i) if ts - s[k][0] <= 14 * 86400)
        return dict(rest=min((ts - prev[0]) / 86400, 21) if prev else np.nan,
                    cong14=n14, prev_cup=float(prev[1]) if prev else np.nan)

    def poisson_day(self, league, daynum, pairs, xg=False):
        rows = [r for r in self.store[league] if daynum - 730 < r[0] < daynum]
        if len(rows) < 60:
            return {}
        a = np.array(rows, dtype=float)
        teams = sorted(set(a[:, 1].astype(int)) | set(a[:, 2].astype(int)) | {t for p in pairs for t in p})
        ix = {t: i for i, t in enumerate(teams)}
        hi, ai = np.array([ix[int(t)] for t in a[:, 1]]), np.array([ix[int(t)] for t in a[:, 2]])
        hg, ag = (a[:, 5], a[:, 6]) if xg else (a[:, 3], a[:, 4])
        ok = ~(np.isnan(hg) | np.isnan(ag))
        if ok.sum() < 60:
            return {}
        hi, ai, hg, ag = hi[ok], ai[ok], hg[ok], ag[ok]
        w = 0.5 ** ((daynum - a[ok, 0]) / self.args.half_life_days)
        n = len(teams)
        mh, ma = (w * hg).sum() / w.sum(), (w * ag).sum() / w.sum()
        att, dfn = np.ones(n), np.ones(n)
        k = 2.0 * (mh + ma) / 2
        seen = np.bincount(hi, w, n) + np.bincount(ai, w, n)
        for _ in range(25):
            num = np.bincount(hi, w * hg, n) + np.bincount(ai, w * ag, n) + k
            den = np.bincount(hi, w * mh * dfn[ai], n) + np.bincount(ai, w * ma * dfn[hi], n) + k
            att = num / den
            num = np.bincount(ai, w * hg, n) + np.bincount(hi, w * ag, n) + k
            den = np.bincount(ai, w * mh * att[hi], n) + np.bincount(hi, w * ma * att[ai], n) + k
            dfn = num / den
            g = np.exp(np.log(att).mean()); att /= g; dfn *= g
        out = {}
        for h, aw in pairs:
            ah, dh = (att[ix[h]], dfn[ix[h]]) if seen[ix[h]] > 0.5 else (0.85, 1.15)   # unseen = promoted prior
            aa, da = (att[ix[aw]], dfn[ix[aw]]) if seen[ix[aw]] > 0.5 else (0.85, 1.15)
            out[(h, aw)] = (mh * ah * da, ma * aa * dh)
        return out

    def team_feats(self, f, team, opp, home, ts):
        F = {}
        e = self.ew[team]
        for m in EW_METRICS:
            F[m] = e[m].get()
        for m, v in self.ewl[team].items():
            F[m + "_l"] = v.get()
        F["elo"] = self.elo.get(team, np.nan)
        F.update(self.sched_feats(team, ts))
        # travel to the venue (away side mainly)
        hv = self.venue.get(self.team_venue.get(team, -1))
        mv = self.venue.get(int(f.venue_id)) if pd.notna(f.venue_id) else None
        F["travel"] = haversine(*hv, *mv) if hv and mv else np.nan
        # league table so far (league competitions only)
        if f.league_type == "league":
            tab = self.table[f.season_id]
            # league size from teams seen so far this season and last season's size (no future fixtures)
            prev_s = [x for x in self.league_seasons[f.league_id] if x != f.season_id]
            n = max(len(self.league_teams[(f.league_id, f.season_id)] | {int(f.home_id), int(f.away_id)}),
                    len(self.league_teams[(f.league_id, prev_s[-1])]) if prev_s else 0, 2)
            rec = tab[team]
            order = sorted(tab.items(), key=lambda kv: (-kv[1][0], -kv[1][1], -kv[1][2]))
            pts_sorted = [v[0] for _, v in order] + [0] * max(0, n - len(order))
            pos = [t for t, _ in order].index(team) + 1 if team in tab else (len(order) + 1)
            played, mp = rec[3], 2 * (n - 1)                # double round robin
            remaining = max(mp - played, 0)
            drop_pts = pts_sorted[n - 3] if n >= 10 and len(pts_sorted) > n - 3 else np.nan
            eur_pts = pts_sorted[3] if len(pts_sorted) > 3 else np.nan
            F.update(pos=(pos - 1) / max(n - 1, 1), ppg=rec[0] / played if played else np.nan,
                     progress=played / mp if mp else np.nan, remaining=remaining,
                     safe_margin=rec[0] - drop_pts if played else np.nan, europe_gap=rec[0] - eur_pts if played else np.nan)
            F["releg_fight"] = float(played > 0 and remaining <= 12 and F["safe_margin"] <= 6)
            F["nothing"] = float(played > 0 and remaining <= 8 and F["safe_margin"] > 1.8 * remaining and F["europe_gap"] < -1.8 * remaining)
            prev = [s for s in self.league_seasons[f.league_id] if s != f.season_id]
            F["promoted"] = float(team not in self.league_teams[(f.league_id, prev[-1])]) if prev else np.nan
        # coach
        c = self.coach.get((f.fixture_id, team))
        F["tenure"] = self.coach_n[(c, team)] if c else np.nan
        F["coach_new"] = float(F["tenure"] < 10) if c else np.nan
        last = self.coach_last.get((c, opp)) if c else None
        F["club_days"] = (f.day - last).days if last is not None else np.nan
        F["club_fam"] = float(last is not None and (f.day - last).days <= 730 and self.coach_n[(c, opp)] >= 5)
        # players: absences (pre-match) and line-up (announced ~1h before kickoff)
        shares = self.pshare.get(team, {})
        top11 = set(sorted(shares, key=shares.get, reverse=True)[:11])
        absent = self.sidelined.get((f.fixture_id, team), [])
        F["absent_share"] = sum(shares.get(p, 0) for p in absent)
        F["absent_top11"] = float(sum(1 for p in absent if p in top11))
        xi = self.starters.get((f.fixture_id, team), [])
        if xi and shares:
            top_sum = sum(sorted(shares.values(), reverse=True)[:11]) or 1
            F["xi_regular"] = sum(shares.get(p, 0) for p in xi) / top_sum
            team_mean = np.nanmean([self.prating[p].get() for p in shares if p in self.prating] or [np.nan])
            r = [self.prating[p].get() if p in self.prating else team_mean - 0.15 for p in xi]
            F["xi_rating"] = float(np.nanmean(r)) if r else np.nan
            F["xi_rating_delta"] = F["xi_rating"] - self.team_xi_rating[team].get()
        else:
            F["xi_regular"] = F["xi_rating"] = F["xi_rating_delta"] = np.nan
        oxi = self.starters.get((f.fixture_id, opp), [])
        if c and oxi:
            cut = f.day - pd.Timedelta(days=730)
            F["n_former"] = float(sum(1 for p in oxi if sum(1 for dd in self.expo[(p, c)] if dd >= cut) >= 5))
        else:
            F["n_former"] = np.nan
        return F

    def emit(self, f, ts):
        if self.max_result_day is not None:
            assert self.max_result_day < f.day, "look-ahead: state contains a result from this day or later"
        h, a = int(f.home_id), int(f.away_id)
        R = dict(fixture_id=int(f.fixture_id), day=f.day, league_id=f.league_id, league=f.league, league_type=f.league_type,
                 season_id=f.season_id, season=f.season, home_id=h, away_id=a, home=f.home, away=f.away)
        H, A = self.team_feats(f, h, a, True, ts), self.team_feats(f, a, h, False, ts)
        for k, v in H.items():
            R["h_" + k] = v
        for k, v in A.items():
            R["a_" + k] = v
        R["elo_diff"] = H["elo"] - A["elo"] if not (np.isnan(H["elo"]) or np.isnan(A["elo"])) else np.nan
        R["derby"] = float((h, a) in self.rivals or (a, h) in self.rivals)
        hh = self.h2h[frozenset((h, a))]
        R["h2h_n"] = len(hh)
        # points the current home side took in recent meetings (any venue)
        R["h2h_pts"] = float(np.mean([p if hm == h else {0: 3, 1: 1, 3: 0}[p] for hm, p in hh])) if hh else np.nan
        rf = self.ref.get(int(f.fixture_id))
        if rf is not None:
            st = self.ref_ew[rf]
            R.update(ref_n=st["n"], ref_home=st["home"].get(), ref_goals=st["goals"].get(), ref_cards=st["cards"].get())
        else:
            R.update(ref_n=0, ref_home=np.nan, ref_goals=np.nan, ref_cards=np.nan)
        le = self.league_ew[f.league_id]
        R.update(lg_home=le["home"].get(), lg_draw=le["draw"].get(), lg_goals=le["goals"].get())
        w = self.weather.get(int(f.fixture_id))
        if w is not None:
            ws = w.wind_speed
            if pd.notna(ws) and "imperial" in str(w.metric):
                ws = ws * 1.609
            R.update(wind=ws, temp=w.temp, humidity=w.humidity, rain=float(any(k in str(w.wdesc) for k in ("rain", "snow", "storm", "drizzle"))))
        else:
            R.update(wind=np.nan, temp=np.nan, humidity=np.nan, rain=np.nan)
        return R

    # ---------- results ----------
    def update(self, f):
        fid, h, a = int(f.fixture_id), int(f.home_id), int(f.away_id)
        hg, ag = self.results.get(fid, (np.nan, np.nan))
        self.max_result_day = f.day if self.max_result_day is None else max(self.max_result_day, f.day)
        if pd.isna(hg) or pd.isna(ag):
            return
        hg, ag = float(hg), float(ag)
        self.league_teams[(f.league_id, f.season_id)].update((h, a))
        if f.season_id not in self.league_seasons[f.league_id]:
            self.league_seasons[f.league_id].append(f.season_id)
        # Elo (new teams start at their league's average minus a promotion discount)
        for t in (h, a):
            if t not in self.elo:
                peers = [self.elo[x] for x in self.league_teams[(f.league_id, f.season_id)] if x in self.elo]
                newcomer = len(self.league_seasons[f.league_id]) > 1   # league already has an earlier season
                self.elo[t] = (np.mean(peers) - (60 if newcomer else 0)) if peers else 1500.0
        exp_h = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - 60) / 400))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = 20 * (1 + math.log1p(abs(hg - ag)))
        self.elo[h] += k * (s_h - exp_h)
        self.elo[a] -= k * (s_h - exp_h)
        # market expectation for residual metrics (closing prices are pre-match, fine to use for the past)
        m = self.mkp.get(fid)
        exp_pts_h = 3 * m[0] + m[1] if m else np.nan
        exp_pts_a = 3 * m[2] + m[1] if m else np.nan
        pts_h, pts_a = (3, 0) if hg > ag else ((1, 1) if hg == ag else (0, 3))
        for t, o, gf, ga, pts, ex in ((h, a, hg, ag, pts_h, exp_pts_h), (a, h, ag, hg, pts_a, exp_pts_a)):
            st, so = self.tstats.get((fid, t), {}), self.tstats.get((fid, o), {})
            xgf, xga = self.xg.get((fid, t), np.nan), self.xg.get((fid, o), np.nan)
            vals = dict(gf=gf, ga=ga, xgf=xgf, xga=xga, sotf=st.get("sot"), sota=so.get("sot"), shf=st.get("shots"), sha=so.get("shots"),
                        daf=st.get("da"), daa=so.get("da"), poss=st.get("poss"), corf=st.get("corners"), cora=so.get("corners"),
                        pts=pts, expts=pts - ex if pd.notna(ex) else np.nan,
                        luck=gf - xgf if pd.notna(xgf) else np.nan, dluck=xga - ga if pd.notna(xga) else np.nan,
                        cards=self.cards.get((fid, t)), bigf=st.get("big_chances"))
            e = self.ew[t]
            for mname, v in vals.items():
                e[mname].add(None if v is None or pd.isna(v) else float(v), A_SHORT)
            for mname in ("gf", "ga", "xgf", "xga"):
                v = vals[mname]
                self.ewl[t][mname].add(None if pd.isna(v) else float(v), A_LONG)
            # players: minutes share and rating
            apps = self.apps.get((fid, t), [])
            if apps:
                sh = self.pshare[t]
                played = {}
                for pid, stt, mins, rat in apps:
                    mins = mins if pd.notna(mins) else (90 if stt == 1 else 0)
                    played[pid] = min(mins, 90) / 90
                    if pd.notna(rat):
                        self.prating.setdefault(pid, EW()).add(float(rat), 0.15)
                for pid in set(sh) | set(played):
                    sh[pid] = (1 - 0.15) * sh.get(pid, 0) + 0.15 * played.get(pid, 0)
                for pid in [p for p, v in sh.items() if v < 0.01]:
                    del sh[pid]
                xi = [p for p, stt, _, _ in apps if stt == 1]
                rs = [self.prating[p].get() for p in xi if p in self.prating]
                if rs:
                    self.team_xi_rating[t].add(float(np.mean(rs)), 0.15)
            c = self.coach.get((fid, t))
            if c:
                self.coach_n[(c, t)] += 1
                self.coach_last[(c, t)] = f.day
                for pid, stt, mins, _ in apps:
                    if stt == 1 or (pd.notna(mins) and mins > 0):
                        dq = self.expo[(pid, c)]
                        dq.append(f.day)
                        while dq and (f.day - dq[0]).days > 760:
                            dq.popleft()
        if f.league_type == "league":
            tab = self.table[f.season_id]
            for t, gf, ga, pts in ((h, hg, ag, pts_h), (a, ag, hg, pts_a)):
                r = tab[t]; r[0] += pts; r[1] += gf - ga; r[2] += gf; r[3] += 1
            daynum = (f.day - pd.Timestamp("2000-01-01")).days
            self.store[f.league_id].append((daynum, h, a, hg, ag, self.xg.get((fid, h), np.nan), self.xg.get((fid, a), np.nan)))
        rf = self.ref.get(fid)
        if rf is not None:
            st = self.ref_ew[rf]
            st["n"] += 1
            if pd.notna(exp_pts_h):
                st["home"].add(pts_h - exp_pts_h, 1 - 0.5 ** (1 / 30))
            st["goals"].add(hg + ag, 1 - 0.5 ** (1 / 30))
            cc = (self.cards.get((fid, h)) or 0) + (self.cards.get((fid, a)) or 0)
            st["cards"].add(cc, 1 - 0.5 ** (1 / 30))
        le = self.league_ew[f.league_id]
        le["home"].add(float(hg > ag), 0.01); le["draw"].add(float(hg == ag), 0.01); le["goals"].add(hg + ag, 0.01)
        self.h2h[frozenset((h, a))].append((h, pts_h))

    def run(self, cutoff=None, only_day=None):
        """cutoff: pretend the world ends on this day (time-travel test). Every result from the
        cutoff day on is deleted, and so is every fixture scheduled after it - a knockout tie's
        next round, a play-off, a rearranged match - since those only appear once results are known."""
        self.reset_state()
        saved = None
        if cutoff is not None:
            saved = (self.results, self.tstats, self.xg, self.cards, self.apps, self.team_sched)
            end = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()
            self.team_sched = {t: [e for e in v if e[0] < end] for t, v in self.team_sched.items()}
            late = set(self.fx.fixture_id[self.fx.day >= cutoff].astype(int))
            self.results = {k: (v if k not in late else (np.nan, np.nan)) for k, v in self.results.items()}
            self.tstats = {k: v for k, v in self.tstats.items() if int(k[0]) not in late}
            self.xg = {k: v for k, v in self.xg.items() if k[0] not in late}
            self.cards = {k: v for k, v in self.cards.items() if k[0] not in late}
            self.apps = defaultdict(list, {k: v for k, v in self.apps.items() if k[0] not in late})
        out = []
        days = self.fx.groupby("day", sort=True)
        nd = len(days)
        track = PROGRESS is not None and only_day is None and cutoff is None
        if track:
            PROGRESS.start("building features", nd)
        for i, (day, g) in enumerate(days):
            if only_day is not None and day > only_day:
                break
            ts_list = [int(x.timestamp()) for x in g.starting_at]
            rows = [self.emit(f, ts) for f, ts in zip(g.itertuples(), ts_list)]
            # Poisson goal / xG ratings for today's league matches (fitted on earlier days only)
            daynum = (day - pd.Timestamp("2000-01-01")).days
            for lg, gg in g[g.league_type == "league"].groupby("league_id"):
                pairs = list(zip(gg.home_id.astype(int), gg.away_id.astype(int)))
                for xg, pre in ((False, "pois"), (True, "xpois")):
                    lam = self.poisson_day(lg, daynum, pairs, xg)
                    for r in rows:
                        key = (r["home_id"], r["away_id"])
                        if r["league_id"] == lg and key in lam:
                            r[f"{pre}_lh"], r[f"{pre}_la"] = lam[key]
            if only_day is None or day == only_day:
                out.extend(rows)
            for f in g.itertuples():
                self.update(f)
            if track:
                progress(1, str(day.date()))
            if i % 250 == 0 and only_day is None:
                log(f"  features: day {i:,}/{nd:,} ({day.date()})")
        if saved:
            self.results, self.tstats, self.xg, self.cards, self.apps, self.team_sched = saved
        return pd.DataFrame(out)


def _leak_run(cut):
    return _W["engine"].run(cutoff=cut, only_day=cut)


def leak_test(engine: Engine, full: pd.DataFrame, args) -> dict:
    """Re-run the engine with every result from the cutoff day on deleted; features
    for matches on the cutoff day must not change."""
    days = sorted(full.day.unique())
    rng = np.random.default_rng(5)
    picks = sorted(rng.choice(days[len(days) // 3: -1], size=min(args.leak_tests, max(1, len(days) // 3)), replace=False))
    res = {"tests": [], "passed": True}
    picks = [pd.Timestamp(c) for c in picks]
    if PROGRESS:
        PROGRESS.start("time-travel test", len(picks))
    truncs = {}
    if len(picks) > 1 and sys.platform.startswith("linux") and usable_cpus() > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        _W["engine"] = engine
        with cf.ProcessPoolExecutor(max_workers=min(len(picks), usable_cpus()), mp_context=mp.get_context("fork")) as ex:
            for cut, tr in zip(picks, ex.map(_leak_run, picks)):
                truncs[cut] = tr
        _W.pop("engine", None)
    for cut in picks:
        trunc = truncs[cut] if cut in truncs else engine.run(cutoff=cut, only_day=cut)
        a = full[full.day == cut].set_index("fixture_id").sort_index()
        b = trunc.set_index("fixture_id").sort_index()
        cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
        diff = (a[cols] - b[cols]).abs()
        both_nan = a[cols].isna() & b[cols].isna()
        mism = ((diff > 1e-9) | (a[cols].isna() != b[cols].isna())) & ~both_nan
        n_bad = int(mism.to_numpy().sum())
        res["tests"].append({"day": str(cut.date()), "matches": int(len(a)), "features": len(cols), "mismatches": n_bad})
        res["passed"] &= n_bad == 0
        progress(1, f"cutoff {cut.date()}")
        log(f"  time-travel test {cut.date()}: {len(a)} matches x {len(cols)} features, mismatches = {n_bad}")
    return res


# =============================================================================
# Models
# =============================================================================
def add_derived(df: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    df = df.join(mk, on="fixture_id")
    eps = 1e-4
    for pre, cols in (("mk", ("p_h", "p_d", "p_a")), ("op", ("open_p_h", "open_p_d", "open_p_a"))):
        if all(c in df for c in cols):
            ph, pd_, pa = (df[c].clip(eps, 1 - eps) for c in cols)
            df[f"{pre}_lr_h"], df[f"{pre}_lr_a"] = np.log(ph / pd_), np.log(pa / pd_)
        else:
            df[f"{pre}_lr_h"] = df[f"{pre}_lr_a"] = np.nan
    if "ou_p_o" in df:
        df["mk_lo"] = np.log(df.ou_p_o.clip(eps, 1 - eps) / (1 - df.ou_p_o.clip(eps, 1 - eps)))
    else:
        df["mk_lo"] = np.nan
    df["op_lo"] = np.log(df.ou_open_p_o / (1 - df.ou_open_p_o)) if "ou_open_p_o" in df else np.nan
    df["move_h"] = df.p_h - df.open_p_h if "open_p_h" in df else np.nan
    df["move_a"] = df.p_a - df.open_p_a if "open_p_a" in df else np.nan
    df["pin_diff_h"] = df.pinp_h - df.p_h if "pinp_h" in df else np.nan
    # Poisson model outputs
    for pre in ("pois", "xpois"):
        if f"{pre}_lh" in df:
            ok = df[f"{pre}_lh"].notna()
            p3, po = poisson_probs(df.loc[ok, f"{pre}_lh"].to_numpy(float), df.loc[ok, f"{pre}_la"].to_numpy(float))
            for i, s in enumerate("hda"):
                df[f"{pre}_p_{s}"] = np.nan
                df.loc[ok, f"{pre}_p_{s}"] = p3[:, i]
            df[f"{pre}_p_o"] = np.nan
            df.loc[ok, f"{pre}_p_o"] = po
        else:
            for s in ("h", "d", "a", "o"):
                df[f"{pre}_p_{s}"] = np.nan
            df[f"{pre}_lh"] = df[f"{pre}_la"] = np.nan
    # symmetric differences and totals
    for m in EW_METRICS + ["gf_l", "ga_l", "xgf_l", "xga_l", "rest", "cong14", "travel", "pos", "ppg", "tenure",
                           "absent_share", "absent_top11", "xi_regular", "xi_rating", "xi_rating_delta", "n_former", "club_fam"]:
        if f"h_{m}" in df:
            df[f"d_{m}"] = df[f"h_{m}"] - df[f"a_{m}"]
    df["s_xg"] = df.h_xgf + df.a_xgf + df.h_xga + df.a_xga
    df["s_goals"] = df.h_gf + df.a_gf + df.h_ga + df.a_ga
    df["s_sot"] = df.h_sotf + df.a_sotf
    df["xpois_tot"] = df.xpois_lh + df.xpois_la
    df["pois_tot"] = df.pois_lh + df.pois_la
    if {"sm_h", "sm_d", "sm_a"} <= set(df.columns):
        s = df[["sm_h", "sm_d", "sm_a"]].sum(axis=1).replace(0, np.nan)
        for c in ("h", "d", "a"):
            df[f"sm_p_{c}"] = df[f"sm_{c}"] / s
    df["y"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    df["y_over"] = (df.hg + df.ag > 2.5).astype(float)
    df["month"] = df.day.dt.to_period("M")
    return df.copy()


LATE = {"h_xi_regular", "a_xi_regular", "h_xi_rating", "a_xi_rating", "h_xi_rating_delta", "a_xi_rating_delta",
        "h_n_former", "a_n_former", "d_xi_regular", "d_xi_rating", "d_xi_rating_delta", "d_n_former"}
NOT_FEATURES = {"fixture_id", "day", "league_id", "league", "league_type", "season_id", "season", "home_id", "away_id",
                "home", "away", "hg", "ag", "y", "y_over", "month", "starting_at", "state", "venue_id"}


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    mkt_cols = {c for c in df.columns if c.startswith(("p_", "avg_", "best_", "pin_", "pinp_", "bestsoft_", "open_", "ou_", "n_books", "overround",
                                                       "disp", "mk_", "op_", "move_", "pin_diff", "sm_"))}
    num = [c for c in df.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().mean() > 0.05]
    fund = [c for c in num if c not in mkt_cols and not c.startswith(("pois_p", "xpois_p")) and c not in LATE]
    late = [c for c in num if c in LATE]
    micro = [c for c in ("disp", "overround", "n_books", "move_h", "move_a", "pin_diff_h") if c in num]
    return {
        "elo": ["elo_diff", "lg_home"],
        "fundamentals": fund + late + [c for c in ("pois_p_h", "pois_p_a", "xpois_p_h", "xpois_p_a") if c in num],
        "market_plus": ["mk_lr_h", "mk_lr_a"] + micro + fund + late,
        "gbm": ["mk_lr_h", "mk_lr_a"] + micro + fund + late,
        "market_plus_open": ["op_lr_h", "op_lr_a"] + fund,
        "gbm_open": ["op_lr_h", "op_lr_a"] + fund,
        "ou_plus": ["mk_lo"] + micro + fund + late,
        "gbm_ou": ["mk_lo"] + micro + fund + late,
        "anchors": ["mk_lr_h", "mk_lr_a", "op_lr_h", "op_lr_a", "mk_lo"],
    }


class Prep:
    """Median-impute, standardise, clip - parameters from the training rows only."""

    def __init__(self, X: pd.DataFrame):
        self.cols = list(X.columns)
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1).fillna(1)

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.cols].fillna(self.med) - self.mu) / self.sd
        return np.clip(Z.fillna(0).to_numpy(float), -5, 5)


def fit_mnl(X, y, K, lam, pen_mask, w0=None):
    """Ridge multinomial (K classes, class 0 baseline-free softmax) via L-BFGS."""
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    Y = np.eye(K)[y]
    pm = np.r_[0.0, pen_mask][:, None]

    def f(wf):
        W = wf.reshape(p + 1, K)
        Z = Xb @ W
        lse = logsumexp(Z, axis=1)
        nll = (lse - (Z * Y).sum(1)).mean()
        P = np.exp(Z - lse[:, None])
        g = Xb.T @ (P - Y) / n + lam * pm * W
        return nll + 0.5 * lam * (pm * W * W).sum(), g.ravel()

    w0 = np.zeros((p + 1) * K) if w0 is None or w0.size != (p + 1) * K else w0
    # tight tolerances: the ridge problem is convex, so every run (any core count) lands on the same optimum
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-8})
    return r.x


def mnl_predict(wf, X, K):
    W = wf.reshape(X.shape[1] + 1, K)
    Z = np.c_[np.ones(len(X)), X] @ W
    return np.exp(Z - logsumexp(Z, axis=1)[:, None])


LAM_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def tune_lam(X, y, K, pen, grid):
    """Pick the ridge strength on the latest 20% of the training window (still all in the past)."""
    cut = int(len(y) * 0.8)
    best, best_ll = grid[len(grid) // 2], np.inf
    for lam in grid:
        wf = fit_mnl(X[:cut], y[:cut], K, lam, pen)
        p = mnl_predict(wf, X[cut:], K)
        ll = -np.log(np.clip(p[np.arange(len(y) - cut), y[cut:]], 1e-9, 1)).mean()
        if ll < best_ll:
            best, best_ll = lam, ll
    return best


# =============================================================================
# Compute: detect what this machine can give, and size the work to fit
# =============================================================================
def usable_cpus() -> int:
    """Logical CPUs this process may use: CPU affinity and container (cgroup) quotas respected."""
    n = os.cpu_count() or 1
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    for qf, pf in (("/sys/fs/cgroup/cpu.max", None), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if pf is None:
                q, per = open(qf).read().split()[:2]
                if q != "max":
                    n = min(n, max(1, int(q) // int(per)))
            else:
                q, per = int(open(qf).read()), int(open(pf).read())
                if q > 0:
                    n = min(n, max(1, q // per))
        except (OSError, ValueError):
            pass
    return max(1, n)


def physical_cores() -> int | None:
    """Physical cores (hyper-threads add little to numerical work)."""
    try:
        import psutil  # type: ignore
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    try:
        cores, phys = set(), None
        for line in open("/proc/cpuinfo"):
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                cores.add((phys, line.split(":")[1].strip()))
        if cores:
            return len(cores)
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout)
        except (OSError, ValueError):
            pass
    return None


def available_memory() -> float | None:
    """Bytes of RAM free for this process (container limits respected)."""
    avail = None
    try:
        import psutil  # type: ignore
        avail = float(psutil.virtual_memory().available)
    except ImportError:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"):
                    avail = float(line.split()[1]) * 1024
        except OSError:
            pass
        if avail is None and sys.platform == "darwin":
            try:
                avail = 0.5 * float(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            except (OSError, ValueError):
                pass
        if avail is None and sys.platform == "win32":
            try:
                import ctypes

                class MS(ctypes.Structure):
                    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                                ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                                ("ae", ctypes.c_ulonglong)]
                m = MS(); m.l = ctypes.sizeof(MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
                avail = float(m.avail)
            except Exception:
                pass
    try:  # cgroup v2 memory limit
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        if lim != "max":
            used = float(open("/sys/fs/cgroup/memory.current").read())
            avail = min(avail or np.inf, float(lim) - used)
    except (OSError, ValueError):
        pass
    return avail


def compute_plan(args, n_chunks: int, df_bytes: float) -> dict:
    """Worker processes x threads per worker that keep every core busy without
    oversubscribing them or running out of memory."""
    logical = usable_cpus()
    phys = physical_cores() or logical
    budget = max(1, min(phys, logical))                 # one heavy thread per physical core
    if budget >= 4 and not args.jobs:
        budget -= 1                                      # leave a core for the OS / database
    mem = available_memory()
    per_worker = 3.0 * df_bytes + 450e6                  # data copy + model working set + scikit-learn
    mem_cap = max(1, int(0.7 * mem // per_worker)) if mem else budget
    workers = args.jobs or max(1, min(n_chunks, budget, mem_cap))
    threads = args.threads or max(1, budget // workers)
    why = "chunks" if workers == n_chunks else ("memory" if workers == mem_cap and mem_cap < budget else "cores")
    return {"logical": logical, "physical": phys, "budget": budget, "mem_gb": round(mem / 1e9, 1) if mem else None,
            "per_worker_gb": round(per_worker / 1e9, 2), "workers": int(workers), "threads": int(threads), "limited_by": why,
            "db_jobs": args.db_jobs or max(2, min(6, logical))}


def set_thread_limit(threads: int):
    """Cap BLAS / OpenMP threads inside this process (numpy, scipy, scikit-learn)."""
    try:
        from threadpoolctl import threadpool_limits  # installed with scikit-learn
        return threadpool_limits(limits=threads)
    except ImportError:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = str(threads)
        return None


# =============================================================================
# Walk-forward training. Work is split into chunks of `retune_every` retrains:
# hyper-parameters are tuned at the start of each chunk, so chunks are
# independent and run in parallel - results do not depend on the core count.
# =============================================================================
_W: dict = {}


def _init_worker(df, fs, args, flags, threads, queue=None):
    _W.update(df=df, fs=fs, args=args, flags=flags, limiter=set_thread_limit(threads), queue=queue)


def _report(work, label):
    """Tell the parent a training block is finished (directly, or via a queue from a worker process)."""
    q = _W.get("queue")
    if q is not None:
        q.put((work, label))
    else:
        progress(work, label)


def _train_chunk(blocks):
    """Train every base model for a run of consecutive retrain blocks. Uses only rows dated
    before each block's first day. Returns predictions to be written back by the parent."""
    df, fs, args, (have_gbm, has_open, has_ou) = _W["df"], _W["fs"], _W["args"], _W["flags"]
    t0 = time.time()
    out = {"P": [], "Q": [], "coefs": [], "lam": [], "log": []}
    warm, lam_cache = {}, {}
    y_all = df.y.to_numpy()
    has_mk = df.mk_lr_h.notna().to_numpy()
    for k, block in enumerate(blocks):
        m0 = block[0].start_time
        train = ((df.day < m0) & (df.day >= m0 - pd.Timedelta(days=args.train_window_days))).to_numpy() & has_mk
        test = df.month.isin(block).to_numpy() & has_mk
        if train.sum() < args.min_train or test.sum() == 0:
            _report(float(train.sum()), str(block[0]))
            continue
        tr, te = df[train], df[test]
        ytr = y_all[train]
        tune = k == 0
        for name in ("elo", "fundamentals", "market_plus") + (("market_plus_open",) if has_open else ()):
            cols = [c for c in fs[name] if c in df]
            if name == "market_plus_open":
                ok_tr, ok_te = tr.op_lr_h.notna().to_numpy(), te.op_lr_h.notna().to_numpy()
            else:
                ok_tr, ok_te = np.ones(len(tr), bool), np.ones(len(te), bool)
            if ok_tr.sum() < args.min_train:
                continue
            prep = Prep(tr.loc[ok_tr, cols])
            Xtr, Xte = prep(tr.loc[ok_tr, cols]), prep(te.loc[ok_te, cols])
            pen = np.array([0.0 if c in fs["anchors"] else 1.0 for c in cols])
            if name == "elo":
                lam = 1e-4
            else:
                if name not in lam_cache or tune:
                    lam_cache[name] = tune_lam(Xtr, ytr[ok_tr], 3, pen, LAM_GRID)
                    out["lam"].append((name, str(block[0]), lam_cache[name]))
                lam = lam_cache[name]
            wf = fit_mnl(Xtr, ytr[ok_tr], 3, lam, pen, warm.get(name))
            warm[name] = wf
            out["P"].append((name, np.where(test)[0][ok_te], mnl_predict(wf, Xte, 3)))
            if name == "market_plus":
                W = wf.reshape(len(cols) + 1, 3)[1:]
                out["coefs"].append((str(block[0]), cols, (W[:, 0] - W[:, 1]).tolist(), (W[:, 2] - W[:, 1]).tolist()))
        if have_gbm:
            # boosting on what the market gets wrong: regress (outcome - market probability)
            from sklearn.ensemble import HistGradientBoostingRegressor
            for name, K in (("gbm", 3), ("gbm_open", 3), ("gbm_ou", 2)):
                if name == "gbm_open" and not has_open or name == "gbm_ou" and not has_ou:
                    continue
                cols = [c for c in fs[name] if c in df]
                pcols = {"gbm": ["p_h", "p_d", "p_a"], "gbm_open": ["open_p_h", "open_p_d", "open_p_a"], "gbm_ou": ["ou_p_o"]}[name]
                ok_tr = tr[pcols].notna().all(axis=1).to_numpy().copy()
                ok_te = te[pcols].notna().all(axis=1).to_numpy().copy()
                if ok_tr.sum() < args.min_train or ok_te.sum() == 0:
                    continue
                Xtr, Xte = tr.loc[ok_tr, cols].to_numpy(float), te.loc[ok_te, cols].to_numpy(float)
                Mtr, Mte = tr.loc[ok_tr, pcols].to_numpy(float), te.loc[ok_te, pcols].to_numpy(float)
                Ytr = np.eye(3)[ytr[ok_tr]] if K == 3 else tr.y_over.to_numpy(float)[ok_tr][:, None]

                def boost(Xa, Ya, Ma, Xb):
                    keep = np.isfinite(Xa).any(axis=0)          # drop features with no data in this window
                    Xa, Xb = Xa[:, keep], Xb[:, keep]
                    res = np.zeros((len(Xb), Ma.shape[1]))
                    for j in range(Ma.shape[1]):
                        reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=200,
                                                            l2_regularization=5.0, random_state=j)
                        reg.fit(Xa, Ya[:, j] - Ma[:, j])
                        res[:, j] = reg.predict(Xb)
                    return res

                def combine(M, G, s_):
                    if M.shape[1] == 1:
                        return np.clip(M[:, 0] + s_ * G[:, 0], 0.01, 0.99)
                    Pn = np.clip(M + s_ * G, 0.005, 1)
                    return Pn / Pn.sum(axis=1, keepdims=True)
                if name not in lam_cache or tune:
                    # how much of the boosted correction to trust, chosen on the latest 20% of the window
                    cut = int(len(Xtr) * 0.8)
                    Gv = boost(Xtr[:cut], Ytr[:cut], Mtr[:cut], Xtr[cut:])
                    yv = Ytr[cut:]
                    best_s, best_ll = 0.0, np.inf
                    for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
                        pv = combine(Mtr[cut:], Gv, s_)
                        ll = (-(yv[:, 0] * np.log(pv) + (1 - yv[:, 0]) * np.log(1 - pv)).mean() if K == 2
                              else -np.log((pv * yv).sum(1)).mean())
                        if ll < best_ll - 1e-6:
                            best_s, best_ll = s_, ll
                    lam_cache[name] = best_s
                    out["lam"].append((name, str(block[0]), best_s))
                G = boost(Xtr, Ytr, Mtr, Xte)
                idx = np.where(test)[0][ok_te]
                if K == 3:
                    out["P"].append((name, idx, combine(Mte, G, lam_cache[name])))
                else:
                    out["Q"].append((name, idx, combine(Mte, G, lam_cache[name])))
        if has_ou:
            cols = [c for c in fs["ou_plus"] if c in df]
            ok_tr, ok_te = tr.mk_lo.notna().to_numpy(), te.mk_lo.notna().to_numpy()
            if ok_tr.sum() >= args.min_train:
                prep = Prep(tr.loc[ok_tr, cols])
                pen = np.array([0.0 if c in fs["anchors"] else 1.0 for c in cols])
                Xo, yo_ = prep(tr.loc[ok_tr, cols]), tr.y_over.to_numpy(int)[ok_tr]
                if "ou" not in lam_cache or tune:
                    lam_cache["ou"] = tune_lam(Xo, yo_, 2, pen, LAM_GRID)
                    out["lam"].append(("ou_plus", str(block[0]), lam_cache["ou"]))
                wf = fit_mnl(Xo, yo_, 2, lam_cache["ou"], pen, warm.get("ou"))
                warm["ou"] = wf
                out["Q"].append(("ou_plus", np.where(test)[0][ok_te], mnl_predict(wf, prep(te.loc[ok_te, cols]), 2)[:, 1]))
        out["log"].append(f"{block[0]}: {train.sum():,} training matches, {test.sum():,} predicted")
        _report(float(train.sum()), str(block[0]))
    out["seconds"] = time.time() - t0
    out["span"] = f"{blocks[0][0]}..{blocks[-1][-1]}"
    return out


def walk_forward(df: pd.DataFrame, fs: dict, args) -> tuple[dict, dict, dict]:
    """Retrain every `retrain_months`; each block is predicted by models trained on earlier months only."""
    import concurrent.futures as cf
    import multiprocessing as mp
    n = len(df)
    P = {m: np.full((n, 3), np.nan) for m in ("elo", "fundamentals", "market_plus", "gbm", "market_plus_open", "gbm_open", "stack")}
    Q = {m: np.full(n, np.nan) for m in ("ou_plus", "gbm_ou")}
    coefs = defaultdict(list)
    try:
        import sklearn  # noqa: F401
        have_gbm = not args.no_gbm
    except ImportError:
        have_gbm = False
        log("scikit-learn not installed - gradient boosting models skipped")
    flags = (have_gbm, df.op_lr_h.notna().mean() > 0.3, df.mk_lo.notna().mean() > 0.3)
    months = sorted(df.month.unique())
    start = df.day.min() + pd.Timedelta(days=args.burn_in_days)
    eval_months = [m for m in months if m.start_time >= start]
    step = args.retrain_months
    blocks = [eval_months[i: i + step] for i in range(0, len(eval_months), step)]
    chunks = [blocks[i: i + args.retune_every] for i in range(0, len(blocks), args.retune_every)]
    plan = compute_plan(args, len(chunks), float(df.memory_usage(deep=True).sum()))
    log(f"compute: {plan['logical']} usable CPUs ({plan['physical']} physical), "
        f"{plan['mem_gb'] if plan['mem_gb'] is not None else '?'} GB RAM free -> {plan['workers']} training worker(s) x "
        f"{plan['threads']} thread(s) for {len(blocks)} retrains in {len(chunks)} chunks (limited by {plan['limited_by']})")
    results = []
    t0 = time.time()
    if PROGRESS:   # work of a block ~ size of its training set
        has_mk = df.mk_lr_h.notna().to_numpy()
        days = df.day.to_numpy()
        tw = np.timedelta64(args.train_window_days, "D")
        work = sum(float(((days < np.datetime64(b[0].start_time)) & (days >= np.datetime64(b[0].start_time) - tw) & has_mk).sum()) for b in blocks)
        PROGRESS.start("training models", work)
    if plan["workers"] <= 1:
        _init_worker(df, fs, args, flags, plan["threads"])
        for i, ch in enumerate(chunks):
            r = _train_chunk(ch)
            results.append(r)
            log(f"  chunk {i + 1}/{len(chunks)} ({r['span']}) in {r['seconds']:.0f}s")
        _W.clear()
    else:
        import queue as queue_mod
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        q = ctx.Queue()
        order = sorted(range(len(chunks)), key=lambda i: -i)        # later chunks train on more data: start them first
        with cf.ProcessPoolExecutor(max_workers=plan["workers"], mp_context=ctx, initializer=_init_worker,
                                    initargs=(df, fs, args, flags, plan["threads"], q)) as ex:
            pending = {ex.submit(_train_chunk, chunks[i]) for i in order}
            n_done = 0

            def drain():
                while True:
                    try:
                        w, label = q.get_nowait()
                    except queue_mod.Empty:
                        return
                    progress(w, label)
            while pending:
                done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
                drain()
                for f in done:
                    r = f.result()
                    results.append(r)
                    n_done += 1
                    log(f"  chunk {n_done}/{len(chunks)} ({r['span']}) in {r['seconds']:.0f}s · elapsed {time.time() - t0:.0f}s")
            drain()
    lam_log = defaultdict(list)
    for r in sorted(results, key=lambda r: r["span"]):
        for name, idx, pred in r["P"]:
            P[name][idx] = pred
        for name, idx, pred in r["Q"]:
            Q[name][idx] = pred
        for month, cols, home, away in r["coefs"]:
            coefs["month"].append(month); coefs["cols"] = cols
            coefs["home"].append(home); coefs["away"].append(away)
        for name, month, val in r["lam"]:
            lam_log[name].append((month, val))
    coefs["lambda"] = dict(lam_log)
    coefs["compute"] = plan
    # stacked ensemble: needs every base model's earlier out-of-sample predictions, so it runs last
    base = ["market", "poisson", "xg_poisson", "elo", "fundamentals", "market_plus"] + (["gbm"] if have_gbm else [])
    S = stack_inputs(df, P, base)
    okS = ~np.isnan(S).any(axis=1)
    y_all, has_mk = df.y.to_numpy(), df.mk_lr_h.notna().to_numpy()
    warm = None
    for block in blocks:
        m0 = block[0].start_time
        prior = (df.day < m0).to_numpy() & okS
        test = df.month.isin(block).to_numpy() & has_mk & okS
        if prior.sum() >= 1500 and test.any():
            warm = fit_mnl(S[prior], y_all[prior], 3, 1e-3, np.ones(S.shape[1]), warm)
            P["stack"][test] = mnl_predict(warm, S[test], 3)
    log(f"walk-forward training done in {time.time() - t0:.0f}s")
    return P, Q, coefs


def stack_inputs(df, P, base):
    cols = []
    for m in base:
        p = model_probs(df, P, m)
        p = np.clip(p, 1e-4, 1)
        cols += [np.log(p[:, 0] / p[:, 1]), np.log(p[:, 2] / p[:, 1])]
    return np.column_stack(cols)


def model_probs(df, P, m):
    if m in P:
        return P[m]
    pre = {"market": "", "poisson": "pois_", "xg_poisson": "xpois_", "sportmonks": "sm_", "market_open": "open_"}[m]
    cols = [f"{pre}p_{s}" for s in "hda"]
    if not all(c in df for c in cols):
        return np.full((len(df), 3), np.nan)
    return df[cols].to_numpy(float)


def model_ou(df, Q, m):
    if m in Q:
        return Q[m]
    col = {"market": "ou_p_o", "poisson": "pois_p_o", "xg_poisson": "xpois_p_o", "market_open": "ou_open_p_o"}[m]
    return df[col].to_numpy(float) if col in df else np.full(len(df), np.nan)


# =============================================================================
# Evaluation, betting, angles
# =============================================================================
def boot_roi(pl: np.ndarray, week: np.ndarray, B=1000, seed=3):
    """Block bootstrap by calendar week (bets on the same weekend are correlated)."""
    if len(pl) < 5:
        return (np.nan, np.nan, np.nan)
    df = pd.DataFrame({"pl": pl, "w": week}).groupby("w").pl.agg(["sum", "size"])
    s, c = df["sum"].to_numpy(), df["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1.0, max(p, 1 / B)))


def bet_table(P3, prices, y, max_odds, tau):
    """Back the selection with the biggest positive edge when edge >= tau."""
    edge = P3 * prices - 1
    edge = np.where(np.isfinite(edge) & (prices <= max_odds), edge, -np.inf)
    sel = np.argmax(edge, axis=1)
    e = edge[np.arange(len(sel)), sel]
    bet = e >= tau
    price = prices[np.arange(len(sel)), sel]
    pl = np.where(sel == y, price - 1, -1.0)
    return bet, sel, e, price, pl


TAUS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]


def metrics_block(P3, y, ok):
    p = np.clip(P3[ok], 1e-6, 1)
    yy = y[ok]
    if len(yy) == 0:
        return {}
    ll = float(-np.log(p[np.arange(len(yy)), yy]).mean())
    Y = np.eye(3)[yy]
    brier = float(((p - Y) ** 2).sum(1).mean())
    cp, cy = np.cumsum(p, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    rps = float(((cp - cy) ** 2).sum(1).mean() / 2)
    acc = float((p.argmax(1) == yy).mean())
    return {"n": int(len(yy)), "logloss": ll, "brier": brier, "rps": rps, "acc": acc}


def evaluate(df, P, Q, args, models_1x2, models_ou):
    E = df.eval_row.to_numpy()
    disc = E & (df.day < df.holdout_from).to_numpy()
    hold = E & (df.day >= df.holdout_from).to_numpy()
    periods = {"discovery": disc, "holdout": hold, "all": E}
    y, yo = df.y.to_numpy(), df.y_over.to_numpy()
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    price_sets = {"avg": ["avg_h", "avg_d", "avg_a"], "best": ["best_h", "best_d", "best_a"], "sharp": ["pin_h", "pin_d", "pin_a"],
                  "open": ["open_h", "open_d", "open_a"]}
    price_sets = {k: v for k, v in price_sets.items() if all(c in df for c in v) and df[v[0]].notna().mean() > 0.2}
    out = {"models": {}, "ou": {}, "prices": list(price_sets)}
    per_row = {}
    mk_ref = model_probs(df, P, "market")
    for m in models_1x2:
        P3 = model_probs(df, P, m)
        have = ~np.isnan(P3).any(axis=1) & ~np.isnan(mk_ref).any(axis=1)
        rec = {"metrics": {}, "bets": {}}
        for pn, pm in periods.items():
            ok = pm & have
            rec["metrics"][pn] = metrics_block(P3, y, ok)
            rec["metrics"][pn]["mk_logloss"] = metrics_block(mk_ref, y, ok).get("logloss")
        for pr, cols in price_sets.items():
            if m.endswith("_open") and pr != "open" or (not m.endswith("_open") and pr == "open"):
                continue
            prices = df[cols].to_numpy(float)
            okp = have & ~np.isnan(prices).any(axis=1)
            curve = {}
            for pn, pm in periods.items():
                curve[pn] = []
                for tau in TAUS:
                    bet, sel, e, price, pl = bet_table(P3, prices, y, args.max_odds, tau)
                    b = bet & pm & okp
                    curve[pn].append({"tau": tau, "n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None})
            # choose tau on discovery only
            cands = [c for c in curve["discovery"] if c["n"] >= args.min_bets and c["roi"] is not None]
            tau = max(cands, key=lambda c: c["roi"])["tau"] if cands else 0.05
            bet, sel, e, price, pl = bet_table(P3, prices, y, args.max_odds, tau)
            res = {"tau": tau, "curve": curve, "periods": {}}
            for pn, pm in periods.items():
                b = bet & pm & okp
                lo, hi, pv = boot_roi(pl[b], week[b])
                rec_p = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                         "profit": float(pl[b].sum()), "win": float((sel[b] == y[b]).mean()) if b.sum() else None,
                         "avg_price": float(price[b].mean()) if b.sum() else None}
                if m.endswith("_open") and "p_h" in df:
                    fair = mk_ref[np.arange(len(sel)), sel]
                    clv = price * fair - 1
                    okc = b & np.isfinite(clv)
                    lo2, hi2, pv2 = boot_roi(clv[okc], week[okc])
                    rec_p.update(clv=float(clv[okc].mean()) if okc.sum() else None, clv_lo=lo2, clv_hi=hi2, clv_p=pv2)
                res["periods"][pn] = rec_p
            # edge buckets, cumulative, monthly, league, kelly (all eval rows at chosen tau)
            b_all = bet & E & okp
            buckets = [(0, .02), (.02, .04), (.04, .06), (.06, .10), (.10, .15), (.15, 9)]
            res["edge_buckets"] = []
            ball = (e >= 0) & E & okp
            for lo_, hi_ in buckets:
                bb = ball & (e >= lo_) & (e < hi_)
                l2, h2, _ = boot_roi(pl[bb], week[bb])
                res["edge_buckets"].append({"label": f"{lo_*100:.0f}–{hi_*100:.0f}%" if hi_ < 9 else f"{lo_*100:.0f}%+",
                                            "n": int(bb.sum()), "roi": float(pl[bb].mean()) if bb.sum() else None, "lo": l2, "hi": h2})
            order = np.argsort(df.day.to_numpy()[b_all], kind="stable")
            cum = np.cumsum(pl[b_all][order])
            dts = df.day.dt.strftime("%Y-%m-%d").to_numpy()[b_all][order]
            stepn = max(1, len(cum) // 400)
            res["cum"] = [[dts[i], round(float(cum[i]), 2)] for i in list(range(0, len(cum), stepn)) + ([len(cum) - 1] if len(cum) else [])]
            mo = pd.DataFrame({"m": df.month.astype(str).to_numpy()[b_all], "pl": pl[b_all]}).groupby("m").pl.agg(["sum", "size"])
            res["monthly"] = [[k, round(float(r["sum"]), 2), int(r["size"])] for k, r in mo.iterrows()]
            lg = pd.DataFrame({"lg": df.league.to_numpy()[b_all], "pl": pl[b_all], "h": hold[b_all]})
            res["leagues"] = {k: {"n": int(len(g)), "roi": float(g.pl.mean()), "n_hold": int(g.h.sum()),
                                  "roi_hold": float(g.pl[g.h].mean()) if g.h.sum() else None} for k, g in lg.groupby("lg")}
            # quarter-Kelly bankroll, stakes capped at 2.5%
            pk = P3[np.arange(len(sel)), sel]
            kel = np.clip((pk * price - 1) / (price - 1), 0, None) * 0.25
            bank, curve_k = 100.0, []
            dd = pd.DataFrame({"d": df.day.to_numpy()[b_all], "k": np.minimum(kel[b_all], 0.025), "pl": pl[b_all]}).sort_values("d")
            for d_, g in dd.groupby("d"):
                stake = bank * g.k.to_numpy()
                bank += float((stake * g.pl.to_numpy()).sum())
                curve_k.append([str(pd.Timestamp(d_).date()), round(bank, 2)])
            res["kelly"] = curve_k[:: max(1, len(curve_k) // 400)] + curve_k[-1:]
            rec["bets"][pr] = res
            if pr in ("avg", "open"):
                per_row[m] = (np.where(bet & okp & E, sel, -1), e)
        # calibration (pooled over the three outcomes)
        okc = E & have
        pp, yy = P3[okc].ravel(), np.eye(3)[y[okc]].ravel()
        bins = np.clip((pp * 20).astype(int), 0, 19)
        cal = pd.DataFrame({"b": bins, "p": pp, "y": yy}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        rec["calibration"] = [[round(float(r.p), 4), round(float(r.y), 4), int(r.n)] for r in cal.itertuples() if r.n >= 30]
        out["models"][m] = rec
    # Over/Under 2.5
    if "ou_avg_o" in df:
        po_cols = {"avg": ["ou_avg_o", "ou_avg_u"], "best": ["ou_best_o", "ou_best_u"], "sharp": ["ou_pin_o", "ou_pin_u"]}
        po_cols = {k: v for k, v in po_cols.items() if all(c in df for c in v) and df[v[0]].notna().mean() > 0.2}
        mko = model_ou(df, Q, "market")
        for m in models_ou:
            po = model_ou(df, Q, m)
            have = ~np.isnan(po) & ~np.isnan(mko) & ~np.isnan(yo)
            rec = {"metrics": {}, "bets": {}}
            for pn, pm in periods.items():
                ok = pm & have
                p = np.clip(po[ok], 1e-6, 1 - 1e-6)
                yy = yo[ok]
                pmk = np.clip(mko[ok], 1e-6, 1 - 1e-6)
                rec["metrics"][pn] = {"n": int(ok.sum()), "logloss": float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()) if ok.sum() else None,
                                      "mk_logloss": float(-(yy * np.log(pmk) + (1 - yy) * np.log(1 - pmk)).mean()) if ok.sum() else None,
                                      "brier": float(((p - yy) ** 2).mean()) if ok.sum() else None,
                                      "acc": float(((p > .5) == yy).mean()) if ok.sum() else None}
            for pr, cols in po_cols.items():
                prices = df[cols].to_numpy(float)
                P2 = np.c_[po, 1 - po]
                yi = np.where(yo == 1, 0, 1)
                okp = have & ~np.isnan(prices).any(axis=1)
                edge = np.where(okp[:, None] & (prices <= args.max_odds), P2 * prices - 1, -np.inf)
                sel = edge.argmax(1)
                e = edge[np.arange(len(sel)), sel]
                price = prices[np.arange(len(sel)), sel]
                pl = np.where(sel == yi, price - 1, -1.0)
                curve = {pn: [{"tau": t, "n": int((pm & (e >= t)).sum()), "roi": float(pl[pm & (e >= t)].mean()) if (pm & (e >= t)).sum() else None}
                              for t in TAUS] for pn, pm in periods.items()}
                cands = [c for c in curve["discovery"] if c["n"] >= args.min_bets and c["roi"] is not None]
                tau = max(cands, key=lambda c: c["roi"])["tau"] if cands else 0.05
                res = {"tau": tau, "curve": curve, "periods": {}}
                for pn, pm in periods.items():
                    b = pm & (e >= tau)
                    lo, hi, pv = boot_roi(pl[b], week[b])
                    res["periods"][pn] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                                          "profit": float(pl[b].sum()), "win": float((sel[b] == yi[b]).mean()) if b.sum() else None,
                                          "avg_price": float(price[b].mean()) if b.sum() else None}
                b_all = E & (e >= tau)
                order = np.argsort(df.day.to_numpy()[b_all], kind="stable")
                cum = np.cumsum(pl[b_all][order])
                dts = df.day.dt.strftime("%Y-%m-%d").to_numpy()[b_all][order]
                stepn = max(1, len(cum) // 400)
                res["cum"] = [[dts[i], round(float(cum[i]), 2)] for i in list(range(0, len(cum), stepn)) + ([len(cum) - 1] if len(cum) else [])]
                rec["bets"][pr] = res
                if pr == "avg":
                    per_row["ou_" + m] = (np.where(E & (e >= tau), sel, -1), e)
            okc = E & have
            bins = np.clip((po[okc] * 20).astype(int), 0, 19)
            cal = pd.DataFrame({"b": bins, "p": po[okc], "y": yo[okc]}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            rec["calibration"] = [[round(float(r.p), 4), round(float(r.y), 4), int(r.n)] for r in cal.itertuples() if r.n >= 30]
            out["ou"][m] = rec
    return out, per_row


def angles(df, args):
    """Pre-declared, rationalised rules. Each backs one selection per qualifying match."""
    E = df.eval_row.to_numpy()
    hold = (df.day >= df.holdout_from).to_numpy()
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    y, yo = df.y.to_numpy(), df.y_over.to_numpy()
    g = lambda c: df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)  # noqa: E731
    rules = []

    def side_rule(key, title, why, cond_h, cond_a):
        rules.append((key, title, why, "1x2", np.nan_to_num(cond_h).astype(bool), np.nan_to_num(cond_a).astype(bool)))

    side_rule("sharp_value", "Soft book beats the sharp price",
              "Best non-sharp price is 3%+ above the sharp bookmaker's fair (de-vigged) price: classic line-shopping value.",
              g("bestsoft_h") > 1.03 / g("pinp_h"), g("bestsoft_a") > 1.03 / g("pinp_a"))
    side_rule("steam", "Follow late money (steam)", "Market probability rose 3+ points from opening to close: informed money tends to be right.",
              g("move_h") >= .03, g("move_a") >= .03)
    side_rule("drift", "Back drifters (contrarian)", "Market probability fell 3+ points: does the market over-react to drift?",
              g("move_h") <= -.03, g("move_a") <= -.03)
    side_rule("fatigue", "Opponent is tired", "Opponent had 3 days or less rest; we had 6+. Markets under-price fatigue.",
              (g("a_rest") <= 3.5) & (g("h_rest") >= 6), (g("h_rest") <= 3.5) & (g("a_rest") >= 6))
    side_rule("luck_reversion", "Fade lucky finishing", "Opponent has scored 0.35+ goals per match above its xG recently; we have not. Results regress to xG.",
              (g("a_luck") >= .35) & (g("h_luck") <= 0), (g("h_luck") >= .35) & (g("a_luck") <= 0))
    side_rule("xg_gap", "xG ratings beat the price", "xG-based Poisson gives this side 5+ points more win probability than the market.",
              g("xpois_p_h") - g("p_h") >= .05, g("xpois_p_a") - g("p_a") >= .05)
    side_rule("market_underrates", "Market keeps under-rating a team", "Team has beaten market-expected points by 0.25+/match recently; opponent has not.",
              (g("h_expts") >= .25) & (g("a_expts") <= 0), (g("a_expts") >= .25) & (g("h_expts") <= 0))
    side_rule("motivation", "Relegation fight vs nothing to play for", "Late season: we are fighting relegation, the opponent has nothing left to play for.",
              (g("h_releg_fight") == 1) & (g("a_nothing") == 1), (g("a_releg_fight") == 1) & (g("h_nothing") == 1))
    side_rule("rotation", "Opponent rotates", "Opponent's starting XI holds under 75% of its usual starters' minutes.",
              g("a_xi_regular") < .75, g("h_xi_regular") < .75)
    side_rule("absences", "Opponent missing key players", "Opponent has 2+ of its usual top-11 sidelined; we have none.",
              (g("a_absent_top11") >= 2) & (g("h_absent_top11") == 0), (g("h_absent_top11") >= 2) & (g("a_absent_top11") == 0))
    side_rule("referee_home", "Home-friendly referee", "Referee (20+ matches) has given home sides 0.15+ points per match above market expectation.",
              (g("ref_n") >= 20) & (g("ref_home") >= .15), np.zeros(len(df)))
    side_rule("coach_familiarity", "Coach knows the opponent", "Our coach managed the opponent within 2 years, or has coached 3+ of today's opposition starters.",
              (g("h_club_fam") == 1) | (g("h_n_former") >= 3), (g("a_club_fam") == 1) | (g("a_n_former") >= 3))
    side_rule("new_manager", "New-manager bounce", "Coach in his first 3 matches in charge.",
              g("h_tenure") <= 2, g("a_tenure") <= 2)
    side_rule("promoted_early", "Promoted team, first 8 games", "Market may anchor on last season's lower-league status.",
              (g("h_promoted") == 1) & (g("h_progress") < .22), (g("a_promoted") == 1) & (g("a_progress") < .22))
    side_rule("home_dog", "Home underdog", "Home side priced 3.0+: home-underdog bias is a documented anomaly.",
              g("avg_h") >= 3.0, np.zeros(len(df)))
    side_rule("long_travel", "Away side travelled far", "Away team travelled 800km+ on 4 days rest or less.",
              (g("a_travel") >= 800) & (g("a_rest") <= 4), np.zeros(len(df)))
    fav = np.nan_to_num(g("avg_h") <= 1.5) | np.nan_to_num(g("avg_a") <= 1.5)
    rules.append(("favourite", "Short-priced favourite", "Favourite-longshot bias: heavy favourites (1.50 or shorter) are under-priced.",
                  "1x2", np.nan_to_num(g("avg_h") <= 1.5).astype(bool), np.nan_to_num(g("avg_a") <= 1.5).astype(bool) & ~np.nan_to_num(g("avg_h") <= 1.5).astype(bool)))
    draw_c = (np.abs(g("p_h") - g("p_a")) < .08) & (g("ou_p_o") < .48)
    rules.append(("tight_draw", "Tight, low-scoring match: draw", "Evenly matched sides in a low-total game: draws are often under-priced.", "draw", np.nan_to_num(draw_c).astype(bool), None))
    rules.append(("derby_draw", "Derby: draw", "Local derbies are cagey; back the draw.", "draw", np.nan_to_num(g("derby") == 1).astype(bool), None))
    rules.append(("wind_under", "High wind: under 2.5", "Wind 30 km/h+ disrupts passing and shooting.", "under", np.nan_to_num(g("wind") >= 30).astype(bool), None))
    rules.append(("ref_goals_under", "Low-scoring referee: under 2.5", "Referee's matches average 0.4+ fewer goals than the league.", "under",
                  np.nan_to_num((g("ref_n") >= 20) & (g("ref_goals") - g("lg_goals") <= -.4)).astype(bool), None))
    rules.append(("xg_total_over", "xG says goals: over 2.5", "xG-based total is 0.35+ goals above what the over/under price implies.", "over",
                  np.nan_to_num(g("xpois_p_o") - g("ou_p_o") >= .06).astype(bool), None))

    out = []
    price_cols = {"avg": ("avg_h", "avg_d", "avg_a", "ou_avg_o", "ou_avg_u"), "best": ("best_h", "best_d", "best_a", "ou_best_o", "ou_best_u")}
    for key, title, why, kind, ch, ca in rules:
        rec = {"key": key, "title": title, "why": why, "periods": {}}
        for pr, (ph_, pd__, pa_, po_, pu_) in price_cols.items():
            if kind == "1x2":
                ch2, ca = ch & ~ca, ca & ~ch     # skip matches where both sides qualify
                price = np.where(ch2, g(ph_), np.where(ca, g(pa_), np.nan))
                if key == "sharp_value":
                    price = np.where(ch2, g("bestsoft_h"), np.where(ca, g("bestsoft_a"), np.nan))
                win = np.where(ch2, y == 0, y == 2)
                mask = ch2 | ca
                imp = np.where(ch2, g("p_h"), g("p_a"))
            elif kind == "draw":
                price, win, mask, imp = g(pd__), y == 1, ch, g("p_d")
            else:
                over = kind == "over"
                price, win, mask = g(po_ if over else pu_), (yo == 1) if over else (yo == 0), ch & ~np.isnan(yo)
                imp = g("ou_p_o") if over else 1 - g("ou_p_o")
            mask = mask & E & np.isfinite(price) & (price <= args.max_odds)
            pl = np.where(win, price - 1, -1.0)
            for pn, pm in (("discovery", ~hold), ("holdout", hold), ("all", np.ones(len(df), bool))):
                b = mask & pm
                lo, hi, pv = boot_roi(pl[b], week[b])
                rec["periods"].setdefault(pn, {})[pr] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                                                         "win": float(win[b].mean()) if b.sum() else None, "implied": float(np.nanmean(imp[b])) if b.sum() else None}
        out.append(rec)
    return out


# =============================================================================
# Dashboard payload
# =============================================================================
def r3(a):
    return [None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else round(float(x), 3) for x in a]


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


def build_payload(df, P, Q, evals, per_row, angle_res, coefs, leak, args, models_1x2, models_ou, cov):
    E = df.eval_row.to_numpy()
    d = df[E]
    cols = {"id": d.fixture_id.astype(int).tolist(), "date": d.day.dt.strftime("%Y-%m-%d").tolist(), "league": d.league.fillna("").tolist(),
            "season": d.season.fillna("").astype(str).tolist(), "home": d.home.fillna("").tolist(), "away": d.away.fillna("").tolist(),
            "hg": d.hg.astype(int).tolist(), "ag": d.ag.astype(int).tolist()}
    for c in ("avg_h", "avg_d", "avg_a", "best_h", "best_d", "best_a", "pin_h", "pin_d", "pin_a", "open_h", "open_d", "open_a",
              "ou_p_o", "ou_avg_o", "ou_avg_u", "move_h", "move_a", "disp", "elo_diff", "h_rest", "a_rest", "a_travel", "h_xgf", "h_xga", "a_xgf", "a_xga",
              "h_luck", "a_luck", "h_expts", "a_expts", "h_pos", "a_pos", "h_xi_regular", "a_xi_regular", "h_xi_rating_delta", "a_xi_rating_delta",
              "h_absent_top11", "a_absent_top11", "h_n_former", "a_n_former", "h_club_fam", "a_club_fam", "h_tenure", "a_tenure",
              "ref_n", "ref_home", "ref_goals", "wind", "derby", "h_releg_fight", "a_releg_fight", "h_nothing", "a_nothing", "xpois_lh", "xpois_la"):
        if c in d:
            cols[c] = r3(d[c].to_numpy(float))
    probs = {}
    for m in models_1x2:
        p = model_probs(df, P, m)[E]
        probs[m] = [r3(p[:, i]) for i in range(3)]
    ou = {m: r3(model_ou(df, Q, m)[E]) for m in models_ou}
    bets = {m: [int(x) for x in v[0][E]] for m, v in per_row.items()}
    edges = {m: r3(v[1][E]) for m, v in per_row.items()}
    imp = {}
    if coefs.get("cols"):
        Hm, Am = np.array(coefs["home"]), np.array(coefs["away"])
        anchors = {}
        for i, c in enumerate(coefs["cols"]):
            if c in ("mk_lr_h", "mk_lr_a"):
                anchors[c] = {"home": float(Hm[:, i].mean()), "away": float(Am[:, i].mean())}
                continue
            imp[c] = {"home_mean": float(Hm[:, i].mean()), "home_sd": float(Hm[:, i].std()), "home_pos": float((Hm[:, i] > 0).mean()),
                      "away_mean": float(Am[:, i].mean()), "away_sd": float(Am[:, i].std()), "away_pos": float((Am[:, i] > 0).mean()),
                      "home_path": [round(float(v), 4) for v in Hm[:, i]]}
        imp = dict(sorted(imp.items(), key=lambda kv: -max(abs(kv[1]["home_mean"]), abs(kv[1]["away_mean"])))[:40])
        imp["__anchors__"] = anchors
    return clean({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "db": args.db_name,
                  "params": {k: v for k, v in vars(args).items() if k not in ("dsn",)},
                  "holdout_from": str(pd.Timestamp(df.holdout_from.iloc[0]).date()),
                  "models_1x2": models_1x2, "models_ou": models_ou, "eval": evals, "angles": angle_res,
                  "importance": imp, "importance_months": coefs.get("month", []), "lambda": coefs.get("lambda", {}), "leak": leak, "coverage": cov,
                  "rows": {"cols": cols, "probs": probs, "ou": ou, "bets": bets, "edges": edges}})


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", help="PostgreSQL connection string (default: DATABASE_URL, else load_league.DATABASE)")
    ap.add_argument("--out", default="model_backtest_dashboard.html")
    ap.add_argument("--export-csv", help="write every feature and prediction to this CSV")
    ap.add_argument("--title", default="Football prediction models vs the market")
    ap.add_argument("--leagues", help="comma-separated league_ids to evaluate/bet (history always uses everything)")
    ap.add_argument("--holdout-from", help="hold-out start date (default: 60%% through the evaluation period)")
    ap.add_argument("--burn-in-days", type=int, default=365, help="history to accumulate before the first prediction")
    ap.add_argument("--train-window-days", type=int, default=1460)
    ap.add_argument("--retrain-months", type=int, default=1)
    ap.add_argument("--min-train", type=int, default=1500)
    ap.add_argument("--half-life-days", type=float, default=180, help="time decay of the Poisson ratings")
    ap.add_argument("--retune-every", type=int, default=6, help="re-tune ridge strength every N retrains (inner time split)")
    ap.add_argument("--market-1x2", type=int, default=1)
    ap.add_argument("--market-ou", type=int, default=80, help="Sportmonks Over/Under market id (line 2.5 is used)")
    ap.add_argument("--sharp-bookmaker", default="Pinnacle")
    ap.add_argument("--max-odds", type=float, default=10.0)
    ap.add_argument("--min-bets", type=int, default=150, help="min discovery bets for a threshold to be eligible")
    ap.add_argument("--leak-tests", type=int, default=3)
    ap.add_argument("--alarm-gain", type=float, default=0.006, help="warn if a model beats market log loss by more than this")
    ap.add_argument("--skip-leak-test", action="store_true")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--jobs", type=int, default=0, help="training worker processes (default: auto from cores and memory)")
    ap.add_argument("--threads", type=int, default=0, help="threads per worker (default: auto)")
    ap.add_argument("--progress-every", type=float, default=30, help="seconds between time-remaining updates (0 = off)")
    ap.add_argument("--db-jobs", type=int, default=0, help="parallel database connections for the extracts (default: auto, 2-6)")
    ap.add_argument("--cache")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    global PROGRESS
    PROGRESS = Progress(args.progress_every, skip=("time-travel test",) if args.skip_leak_test else ())
    db = DB(args.dsn)
    args.db_name = db.name
    d = extract(db, args)
    mk = market_table(d, args)
    log(f"market table: {len(mk):,} fixtures with 1X2 prices")
    if len(d["predictions"]):
        mk = mk.join(d["predictions"].set_index("fixture_id"), how="left")
    log("building features day by day ...")
    eng = Engine(d, mk, args)
    feats = eng.run()
    log(f"feature rows: {len(feats):,} x {feats.shape[1]} columns")
    leak = {"skipped": True}
    if not args.skip_leak_test:
        log("time-travel leakage test ...")
        leak = leak_test(eng, feats, args)
        leak["skipped"] = False
        if not leak["passed"]:
            log("WARNING: time-travel test found differences - see dashboard")
    df = feats.merge(d["fixtures"][["fixture_id", "hg", "ag"]], on="fixture_id")
    df = add_derived(df, mk)
    df = df.sort_values(["day", "fixture_id"]).reset_index(drop=True)
    fs = feature_sets(df)
    log(f"features: fundamentals {len(fs['fundamentals'])}, market_plus {len(fs['market_plus'])}")
    log("walk-forward training ...")
    P, Q, coefs = walk_forward(df, fs, args)
    models_1x2 = ["market"] + (["sportmonks"] if "sm_p_h" in df else []) + ["elo", "poisson", "xg_poisson", "fundamentals", "market_plus"]
    models_1x2 += [m for m in ("gbm", "stack") if not np.isnan(P[m]).all()]
    models_1x2 += [m for m in ("market_open", "market_plus_open", "gbm_open")
                   if (m == "market_open" and "open_p_h" in df and df.open_p_h.notna().mean() > .3) or (m in P and not np.isnan(P[m]).all())]
    models_ou = [m for m in ("market", "poisson", "xg_poisson", "ou_plus", "gbm_ou")
                 if not np.isnan(model_ou(df, Q, m)).all()] if "ou_p_o" in df else []
    # evaluation rows: every 1X2 model has a prediction (after burn-in, training available)
    have = np.ones(len(df), bool)
    for m in ("market", "elo", "fundamentals", "market_plus"):
        have &= ~np.isnan(model_probs(df, P, m)).any(axis=1)
    if args.leagues:
        have &= df.league_id.isin([int(x) for x in args.leagues.split(",")]).to_numpy()
    df["eval_row"] = have
    if have.sum() == 0:
        sys.exit("No rows could be evaluated - not enough history or odds.")
    ev_days = df.day[have]
    hf = pd.Timestamp(args.holdout_from) if args.holdout_from else ev_days.quantile(0.6).normalize()
    df["holdout_from"] = hf
    log(f"evaluating {have.sum():,} matches; discovery < {hf.date()} <= hold-out")
    PROGRESS.start("scoring", 3)
    evals, per_row = evaluate(df, P, Q, args, models_1x2, models_ou)
    # Alarm: beating the closing market's log loss by this much is not realistic for any model.
    # It almost always means information from after kickoff has crept in.
    alarms = []
    for m, r in list(evals["models"].items()) + [("ou:" + k, v) for k, v in evals.get("ou", {}).items()]:
        mt = r["metrics"].get("holdout", {})
        if mt.get("logloss") is not None and mt.get("mk_logloss") is not None and mt["mk_logloss"] - mt["logloss"] > args.alarm_gain:
            alarms.append({"model": m, "gain": mt["mk_logloss"] - mt["logloss"]})
            log(f"!!! WARNING: {m} beats the closing market's log loss by {(mt['mk_logloss'] - mt['logloss']) * 1000:.1f}e-3 in the "
                f"hold-out. That is implausibly large - check for leakage before trusting any result.")
    leak["alarms"] = alarms
    progress(1, "models scored")
    angle_res = angles(df, args)
    progress(1, "angles scored")
    cov = {"fixtures": int(len(df)), "eval": int(have.sum()), "leagues": int(df[have].league_id.nunique()),
           "from": str(ev_days.min().date()), "to": str(ev_days.max().date()),
           "with_xg": float(df.h_xgf.notna()[have].mean()), "with_lineups": float(df.h_xi_regular.notna()[have].mean()),
           "with_open": float(df.open_p_h.notna()[have].mean()) if "open_p_h" in df else 0.0,
           "with_sharp": float(df.pin_h.notna()[have].mean()) if "pin_h" in df else 0.0,
           "with_ou": float(df.ou_p_o.notna()[have].mean()) if "ou_p_o" in df else 0.0,
           "with_ref": float((df.ref_n > 0)[have].mean()), "with_weather": float(df.wind.notna()[have].mean()),
           "with_sidelined": float((df.h_absent_share > 0)[have].mean()),
           "n_features": len(fs["market_plus"]), "avg_overround": float(df.overround[have].mean()),
           "n_discovery": int((have & (df.day < hf)).sum()), "n_holdout": int((have & (df.day >= hf)).sum()),
           "compute": coefs.get("compute", {}), "runtime_s": round(time.time() - T0)}
    if args.export_csv:
        out = df.copy()
        for m in models_1x2:
            p = model_probs(df, P, m)
            for i, s in enumerate("hda"):
                out[f"pred_{m}_{s}"] = p[:, i]
        for m in models_ou:
            out[f"pred_ou_{m}"] = model_ou(df, Q, m)
        out.drop(columns=["month"]).to_csv(args.export_csv, index=False)
        log(f"CSV written: {args.export_csv}")
    payload = build_payload(df, P, Q, evals, per_row, angle_res, coefs, leak, args, models_1x2, models_ou, cov)
    page = DASHBOARD.replace("__TITLE__", htmlmod.escape(args.title)).replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    Path(args.out).write_text(page, encoding="utf-8")
    PROGRESS.close()
    log(f"dashboard written: {args.out} ({len(page) / 1e6:.1f} MB)")
    for m in models_1x2:
        r = evals["models"][m]
        b = r["bets"].get("open" if m.endswith("_open") else "avg", {}).get("periods", {}).get("holdout", {})
        mt = r["metrics"]["holdout"]
        log(f"  {m:18s} hold-out logloss {mt.get('logloss', float('nan')):.4f} (market {mt.get('mk_logloss') or float('nan'):.4f})"
            f"  bets {b.get('n', 0):5d}  ROI {(b.get('roi') or 0) * 100:+6.2f}%")


DASHBOARD = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--ctrl:#a9a79f;--good:#006300;--bad:#d03b3b;--goodbg:rgba(12,163,12,.10);--badbg:rgba(208,59,59,.09);--wash:rgba(42,120,214,.07)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.10)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--goodbg:rgba(12,163,12,.16);--badbg:rgba(230,103,103,.14);--wash:rgba(57,135,229,.10)}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh;padding:26px clamp(14px,3.5vw,44px) 60px}
.wrap{max-width:1320px;margin:0 auto}
h1{font-size:24px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;font-weight:600;margin:0 0 2px}
h3{font-size:13px;font-weight:600;margin:12px 0 4px;color:var(--ink2)}
.sub{color:var(--ink2);margin:0 0 14px;max-width:960px}
.note{color:var(--muted);font-size:12px;margin:2px 0 10px}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;padding:4px 10px;border-radius:99px;border:1px solid var(--ring);background:var(--surface);color:var(--ink2)}
.badge.ok{background:var(--goodbg);color:var(--good)}.badge.bad{background:var(--badbg);color:var(--bad)}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin:0 0 18px;padding:10px 14px;background:var(--surface);border:1px solid var(--ring);border-radius:10px;position:sticky;top:8px;z-index:6}
.controls label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--ink2)}
select,button,input{font:inherit;color:var(--ink);background:var(--page);border:1px solid var(--ring);border-radius:7px;padding:5px 9px}
button{cursor:pointer}button:hover{background:var(--wash)}
.grid{display:grid;gap:14px}.g2{grid-template-columns:repeat(auto-fit,minmax(520px,1fr))}.g3{grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}.g5{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:15px 17px;min-width:0}
.hero{display:flex;flex-wrap:wrap;gap:26px;align-items:flex-end}
.big{font-size:50px;font-weight:650;line-height:1;letter-spacing:-.02em}
.lbl{color:var(--ink2);font-size:12.5px}
.tile .val{font-size:24px;font-weight:620;margin:3px 0 2px}.tile .ci{color:var(--muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.section{margin:30px 0 10px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:4px 0 6px;flex-wrap:wrap}
.key{display:inline-flex;align-items:center;gap:6px}.sw{width:10px;height:10px;border-radius:3px;display:inline-block}.ln{width:14px;height:2px;display:inline-block;border-radius:1px}
svg{display:block;width:100%;height:auto;overflow:visible}svg text{font-family:inherit;font-size:11px;fill:var(--muted)}
svg .lab{fill:var(--ink2);font-size:11.5px}svg .val{fill:var(--ink);font-size:11px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none;z-index:1}
td:first-child,th:first-child,.l{text-align:left}td{font-variant-numeric:tabular-nums}
tr.sel td{background:var(--wash)}tbody tr:hover td{background:var(--wash)}
.scroll{max-height:460px;overflow:auto}.wide{overflow-x:auto}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);box-shadow:0 4px 18px rgba(0,0,0,.14);border-radius:8px;padding:8px 10px;font-size:12px;opacity:0;transition:opacity .08s;z-index:30;max-width:300px}
.pill{display:inline-block;font-size:11px;padding:0 7px;border-radius:99px;border:1px solid var(--ring);color:var(--ink2)}
.pill.bet{border-color:var(--s1);color:var(--ink)}
.hit{fill:transparent}.why{color:var(--muted);font-size:11.5px;white-space:normal;max-width:560px}
.pager{display:flex;gap:8px;align-items:center;margin-top:8px;font-size:12.5px;color:var(--ink2)}
.detail td{white-space:normal;text-align:left;background:var(--page)}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:2px 16px;font-size:12px}
.kv span{color:var(--muted)}
ul.m li{margin:3px 0;color:var(--ink2)}
.tabs{display:flex;gap:6px;margin:6px 0 8px}.tabs button.on{background:var(--wash);border-color:var(--s1)}
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

<div class="section"><h2>Model leaderboard</h2><p class="note">Click a column to sort, a row to focus that model. Log-loss gain = market log loss minus model log loss on the same matches (positive = better forecaster than the market). Bets: back the outcome with the largest edge when edge ≥ τ; τ was picked on the discovery period only.</p></div>
<div class="card wide"><table id="t-lead"></table></div>

<div class="grid g2" style="margin-top:14px">
  <div class="card"><h2>Return per bet by model</h2><p class="note" id="roi-note"></p><div id="c-roi"></div></div>
  <div class="card"><h2>Forecast quality vs the market</h2><p class="note">Log-loss gain ×1000 on the same matches. Above zero = the model forecasts better than the closing market.</p><div id="c-ll"></div></div>
</div>

<div class="section"><h2>Focus: <span id="fname"></span></h2><p class="note" id="fdesc"></p></div>
<div class="grid g2">
  <div class="card"><h2>Cumulative profit (1-unit stakes)</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Profit</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Hold-out starts</span></div><div id="c-cum"></div></div>
  <div class="card"><h2>Bankroll, quarter-Kelly staking (start 100)</h2><p class="note">Stake = ¼ × Kelly fraction, capped at 2.5% of bankroll. Shows the path a disciplined bettor would live through.</p><div id="c-kelly"></div></div>
  <div class="card"><h2>Return vs edge threshold τ</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Hold-out</span><span class="key"><span class="ln" style="background:var(--s2)"></span>Discovery</span></div><div id="c-tau"></div></div>
  <div class="card"><h2>Does a bigger edge mean a bigger return?</h2><p class="note">All evaluated matches where the model saw positive edge, grouped by edge size. A real edge should rise left to right.</p><div id="c-edge"></div></div>
  <div class="card"><h2>Calibration</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>Predicted vs actual</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Perfect</span></div><div id="c-cal"></div></div>
  <div class="card"><h2>Monthly profit</h2><div id="c-month"></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>By league</h2><div class="scroll"><table id="t-league"></table></div></div>

<div class="section"><h2>Betting angles</h2><p class="note">Pre-declared rules, each with a reason written before testing. Discovery and hold-out are shown separately: an angle only matters if it holds up in the hold-out.</p></div>
<div class="grid">
  <div class="card"><h2>Hold-out return with 95% interval</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>Hold-out, interval above zero</span><span class="key"><span class="sw" style="background:var(--ctrl)"></span>Hold-out, could be chance</span><span class="key"><span class="sw" style="background:var(--s2);border-radius:99px"></span>Discovery</span></div><div id="c-angles"></div></div>
  <div class="card"><h2>Angle detail</h2><div class="scroll" style="max-height:860px"><table id="t-angles"></table></div></div>
</div>

<div class="section"><h2>What moves the edge model</h2><p class="note">Standardised coefficients of the market-anchored model (market_plus), averaged over every monthly refit. "Same sign" = share of refits where the coefficient kept its direction; stable drivers are more believable.</p></div>
<div class="grid g2">
  <div class="card"><div class="tabs"><button id="imp-home" class="on" type="button">Home win vs draw</button><button id="imp-away" type="button">Away win vs draw</button></div><div id="c-imp"></div></div>
  <div class="card"><h2>Driver table</h2><div class="scroll" style="max-height:560px"><table id="t-imp"></table></div><h3>Regularisation chosen inside each training window</h3><div id="lam"></div></div>
</div>

<div class="section"><h2>Match explorer</h2><p class="note" id="mx-note"></p></div>
<div class="card">
  <div class="controls" style="position:static;margin-bottom:10px">
    <label>League<select id="mx-league"></select></label>
    <label>Team<input id="mx-team" placeholder="search"></label>
    <label>Show<select id="mx-show"><option value="all">All matches</option><option value="bets">Focus model bets</option><option value="won">Focus model winning bets</option><option value="lost">Focus model losing bets</option></select></label>
    <label>Period<select id="mx-period"><option value="all">All</option><option value="holdout">Hold-out</option><option value="discovery">Discovery</option></select></label>
    <button id="mx-csv" type="button">Download all rows (CSV)</button>
  </div>
  <div class="wide"><table id="t-mx"></table></div>
  <div class="pager"><button id="pg-prev" type="button">‹ Prev</button><span id="pg-info"></span><button id="pg-next" type="button">Next ›</button></div>
</div>

<div class="section"><h2>Leakage safeguards, data and method</h2></div>
<div class="grid g2">
  <div class="card"><h2>How look-ahead is prevented</h2><ul class="m" id="safeguards"></ul><h3>Time-travel tests</h3><table id="t-leak"></table></div>
  <div class="card"><h2>Coverage</h2><table id="t-cov"></table><h3>Models</h3><ul class="m" id="mdesc"></ul></div>
</div>
</div></div>
<div class="tip" id="tip"></div>
<script>
const R = __DATA__;
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const pct = (v, d=1) => v == null ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(d) + '%';
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
function el(tag, attrs, parent){ const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
function svg(host, w, h){ host.innerHTML = ''; return el('svg', {viewBox:`0 0 ${w} ${h}`, role:'img'}, host); }
function text(p, x, y, s, a={}){ const t = el('text', Object.assign({x, y}, a), p); t.textContent = s; return t; }
function ticks(lo, hi, n=5){ const span = hi - lo || 1, step0 = span/n, mag = Math.pow(10, Math.floor(Math.log10(step0))); const step = [1,2,2.5,5,10].map(m=>m*mag).find(s => span/s <= n) || mag*10; const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(10)); return out; }
function ext(vals, zero=true){ let v = vals.filter(x => x != null && isFinite(x)); if (zero) v.push(0); if (!v.length) v=[0,1]; let lo = Math.min(...v), hi = Math.max(...v); if (lo === hi){ lo -= 1; hi += 1; } const p = (hi-lo)*.07; return [lo-p, hi+p]; }
function hit(node, html){ node.addEventListener('mousemove', e => showTip(e, html)); node.addEventListener('mouseleave', hideTip); }
function empty(host, msg){ host.innerHTML = `<p class="note" style="padding:40px 0;text-align:center">${msg}</p>`; }
function hbarPath(x0, x1, y, h){ const r = Math.min(4, Math.abs(x1-x0)/2, h/2); if (x1 >= x0) return `M${x0},${y}H${x1-r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1-r},${y+h}H${x0}Z`; return `M${x0},${y}H${x1+r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1+r},${y+h}H${x0}Z`; }
function vbarPath(x, w, y0, y1){ const r = Math.min(4, Math.abs(y1-y0)/2, w/2); if (y1 <= y0) return `M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z`; return `M${x},${y0}V${y1-r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1-r}V${y0}Z`; }

// horizontal bars with optional CI and optional second marker
function hbars(host, rows, fmt, opts={}){
  if (!rows.length) return empty(host, 'No data');
  const W = opts.W || 600, rowH = opts.rowH || 30, top = 4, left = opts.left || 150, right = 64, H = top + rows.length*rowH + 24;
  const s = svg(host, W, H); const [lo, hi] = ext(rows.flatMap(r => [r.v, r.lo, r.hi, r.v2]));
  const x = v => left + (v-lo)/(hi-lo)*(W-left-right);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-20, stroke:css('--grid')}, s); text(s, x(t), H-5, fmt(t), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:top, y2:H-20, stroke:css('--axis')}, s);
  rows.forEach((r, i) => {
    const y = top + i*rowH + (rowH-16)/2, h = 16;
    text(s, left-8, y+h/2+4, r.label, {'text-anchor':'end', class:'lab'});
    if (r.v != null) el('path', {d: hbarPath(x(0), x(r.v), y, h), fill: r.color || css('--s1')}, s);
    if (r.lo != null && r.hi != null){ el('line', {x1:x(r.lo), x2:x(r.hi), y1:y+h/2, y2:y+h/2, stroke:css('--ink2'), 'stroke-width':1.5}, s);
      for (const e of [r.lo, r.hi]) el('line', {x1:x(e), x2:x(e), y1:y+h/2-4, y2:y+h/2+4, stroke:css('--ink2'), 'stroke-width':1.5}, s); }
    if (r.v2 != null) el('circle', {cx:x(r.v2), cy:y+h/2, r:4.5, fill:css('--s2'), stroke:css('--surface'), 'stroke-width':2}, s);
    if (r.v != null) text(s, Math.max(x(r.v), x(0), r.hi != null ? x(r.hi) : 0, r.v2 != null ? x(r.v2)+5 : 0) + 6, y+h/2+4, fmt(r.v), {class:'val'});
    hit(el('rect', {x:0, y:y-(rowH-h)/2, width:W, height:rowH, class:'hit'}, s), r.tip || '');
    if (r.onclick) s.lastChild.addEventListener('click', r.onclick);
  });
}

// columns with CI
function cols(host, rows, fmt, xTitle, every=1){
  if (!rows.length) return empty(host, 'No data');
  const W = 600, H = 250, top = 12, bottom = 54, left = 54, right = 10;
  const s = svg(host, W, H); const [lo, hi] = ext(rows.flatMap(r => [r.v, r.lo, r.hi]));
  const y = v => top + (hi-v)/(hi-lo)*(H-top-bottom), bw = (W-left-right)/rows.length, w = Math.min(24, bw*.55);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, fmt(t), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis')}, s);
  rows.forEach((r, i) => {
    const cx = left + bw*i + bw/2;
    if (i % every === 0) text(s, cx, H-bottom+16, r.label, {'text-anchor':'middle', class:'lab'});
    if (r.n != null) text(s, cx, H-bottom+30, 'n=' + r.n.toLocaleString(), {'text-anchor':'middle'});
    if (r.v != null) el('path', {d: vbarPath(cx-w/2, w, y(0), y(r.v)), fill: r.color || (r.small ? css('--ctrl') : css('--s1'))}, s);
    if (r.lo != null && r.hi != null){ el('line', {x1:cx, x2:cx, y1:y(r.lo), y2:y(r.hi), stroke:css('--ink2'), 'stroke-width':1.5}, s);
      for (const e of [r.lo, r.hi]) el('line', {x1:cx-4, x2:cx+4, y1:y(e), y2:y(e), stroke:css('--ink2'), 'stroke-width':1.5}, s); }
    hit(el('rect', {x:cx-bw/2, y:top, width:bw, height:H-top-bottom, class:'hit'}, s), r.tip || '');
  });
  if (xTitle) text(s, left+(W-left-right)/2, H-3, xTitle, {'text-anchor':'middle'});
}

// line chart; x either dates ('YYYY-MM-DD') or numbers
function lines(host, series, opts={}){
  const pts = series.flatMap(s => s.pts);
  if (!pts.length) return empty(host, opts.empty || 'No bets for this selection');
  const isDate = typeof pts[0][0] === 'string', X = v => isDate ? Date.parse(v) : v;
  const W = 600, H = 250, top = 12, bottom = 28, left = 54, right = 62, s = svg(host, W, H);
  const xs = pts.map(p => X(p[0])), x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0+1 : Math.max(...xs);
  const [lo, hi] = ext(pts.map(p => p[1]).concat(opts.yref != null ? [opts.yref] : []), opts.zero !== false);
  const x = v => left + (X(v)-x0)/(x1-x0)*(W-left-right), y = v => top + (hi-v)/(hi-lo)*(H-top-bottom);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-7, y(t)+4, (opts.fmt||(v=>v.toLocaleString()))(t), {'text-anchor':'end'}); }
  if (opts.yref != null) el('line', {x1:left, x2:W-right, y1:y(opts.yref), y2:y(opts.yref), stroke:css('--axis')}, s);
  if (isDate){ const y0 = new Date(x0).getUTCFullYear(), y1 = new Date(x1).getUTCFullYear(); for (let yy=y0; yy<=y1+1; yy++){ const t = Date.UTC(yy,0,1); if (t>=x0 && t<=x1) text(s, left+(t-x0)/(x1-x0)*(W-left-right), H-8, yy, {'text-anchor':'middle'}); } }
  else for (const t of ticks(x0, x1, 6)) text(s, x(t), H-8, (opts.xfmt||(v=>v))(t), {'text-anchor':'middle'});
  if (opts.vline){ const vx = x(opts.vline); if (vx>=left && vx<=W-right){ el('line', {x1:vx, x2:vx, y1:top, y2:H-bottom, stroke:css('--axis'), 'stroke-width':1.5}, s); text(s, vx+4, top+10, opts.vlabel||'', {}); } }
  for (const se of series){ if (!se.pts.length) continue;
    el('path', {d: se.pts.map((p,j)=>(j?'L':'M')+x(p[0]).toFixed(1)+','+y(p[1]).toFixed(1)).join(''), fill:'none', stroke:se.color, 'stroke-width':2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}, s);
    if (se.dots) se.pts.forEach(p => el('circle', {cx:x(p[0]), cy:y(p[1]), r:4, fill:se.color, stroke:css('--surface'), 'stroke-width':2}, s));
    const l = se.pts[se.pts.length-1]; if (se.endLabel){ el('circle', {cx:x(l[0]), cy:y(l[1]), r:4, fill:se.color, stroke:css('--surface'), 'stroke-width':2}, s); text(s, x(l[0])+7, y(l[1])+4, se.endLabel, {class:'val'}); } }
  const cross = el('line', {y1:top, y2:H-bottom, stroke:css('--axis'), opacity:0}, s);
  const dots = series.map(se => el('circle', {r:4, fill:se.color, stroke:css('--surface'), 'stroke-width':2, opacity:0}, s));
  const hr = el('rect', {x:left, y:top, width:W-left-right, height:H-top-bottom, class:'hit'}, s);
  hr.addEventListener('mousemove', e => { const b = s.getBoundingClientRect(), mx = (e.clientX-b.left)/b.width*W, xv = x0 + (mx-left)/(W-left-right)*(x1-x0);
    let html = '', cx = null;
    series.forEach((se, k) => { if (!se.pts.length) return; let best = se.pts[0]; for (const p of se.pts) if (Math.abs(X(p[0])-xv) < Math.abs(X(best[0])-xv)) best = p;
      cx = x(best[0]); dots[k].setAttribute('cx', cx); dots[k].setAttribute('cy', y(best[1])); dots[k].setAttribute('opacity', 1);
      html += (k ? '' : `<b>${opts.xlabel ? opts.xlabel(best[0]) : best[0]}</b><br>`) + `${esc(se.name)}: <b>${(opts.tfmt||opts.fmt||(v=>v))(best[1])}</b>${best[2]!=null?` · ${opts.extra?opts.extra(best):''}`:''}<br>`; });
    if (cx != null){ cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('opacity', 1); }
    showTip(e, html); });
  hr.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); dots.forEach(d => d.setAttribute('opacity', 0)); });
}

function calChart(host, pts){
  if (!pts || !pts.length) return empty(host, 'No data');
  const W = 600, H = 270, top = 10, bottom = 32, left = 46, right = 12, s = svg(host, W, H);
  const mx = Math.min(1, Math.max(...pts.map(p => Math.max(p[0], p[1]))) * 1.08);
  const x = v => left + v/mx*(W-left-right), y = v => top + (1-v/mx)*(H-top-bottom);
  for (const t of ticks(0, mx, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-bottom, stroke:css('--grid')}, s); el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s);
    text(s, x(t), H-bottom+15, Math.round(t*100)+'%', {'text-anchor':'middle'}); text(s, left-7, y(t)+4, Math.round(t*100)+'%', {'text-anchor':'end'}); }
  text(s, left+(W-left-right)/2, H-2, 'Predicted probability', {'text-anchor':'middle'});
  el('line', {x1:x(0), y1:y(0), x2:x(mx), y2:y(mx), stroke:css('--axis'), 'stroke-width':1.5}, s);
  el('path', {d: pts.map((p,j)=>(j?'L':'M')+x(p[0])+','+y(p[1])).join(''), fill:'none', stroke:css('--s1'), 'stroke-width':2}, s);
  pts.forEach(p => { const r = Math.max(4, Math.min(8, Math.sqrt(p[2])/6)); el('circle', {cx:x(p[0]), cy:y(p[1]), r, fill:css('--s1'), stroke:css('--surface'), 'stroke-width':2}, s);
    hit(el('circle', {cx:x(p[0]), cy:y(p[1]), r:r+8, class:'hit'}, s), `Predicted ${pctu(p[0])}<br>Happened ${pctu(p[1])}<br>${p[2].toLocaleString()} outcomes`); });
}

function sortable(t, head, rows, opts={}){
  let key = opts.sortKey ?? null, dir = opts.sortDir ?? -1;
  function draw(){
    const r = rows.slice(); if (key != null) r.sort((a, b) => { const x = a.v[key], y = b.v[key]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    t.innerHTML = '<thead><tr>' + head.map((h, i) => `<th data-i="${i}" class="${h.c||''}" title="${esc(h.t||'')}">${h.h}${key===i?(dir>0?' ▲':' ▼'):''}</th>`).join('') + '</tr></thead><tbody>' +
      r.map(x => `<tr data-k="${esc(x.k||'')}" class="${x.sel?'sel':''}">` + x.c.map((c, i) => `<td class="${head[i].c||''}">${c}</td>`).join('') + '</tr>').join('') + '</tbody>';
    t.querySelectorAll('th').forEach(th => th.onclick = () => { const i = +th.dataset.i; dir = key === i ? -dir : -1; key = i; draw(); });
    if (opts.onRow) t.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => opts.onRow(tr.dataset.k));
  }
  draw();
}

// ---------------- state & controls ----------------
const DESC = {
  market:'De-vigged consensus of every bookmaker\'s closing price. The benchmark.', sportmonks:'Sportmonks\' own pre-match probabilities.',
  elo:'Goal-difference Elo across every competition, mapped to 1X2 with a multinomial logit.',
  poisson:'Time-decayed attack/defence ratings from goals (Dixon-Coles style), 2-year window, 180-day half-life.',
  xg_poisson:'The same ratings fitted on expected goals: less noisy than goals.',
  fundamentals:'Ridge multinomial logit on every feature, but no odds at all. Can football data alone match the market?',
  market_plus:'Market log-odds as an unpenalised anchor plus every feature with ridge shrinkage: learns only where the market is systematically wrong.',
  gbm:'Gradient-boosted trees modelling the market\'s errors (outcome − market probability), shrunk by an amount chosen inside the training window.',
  stack:'Learns how much to trust each model from their earlier out-of-sample predictions.',
  market_open:'De-vigged opening line (premium odds history).', market_plus_open:'market_plus at the opening line, without line-up information. Judged by closing-line value.',
  gbm_open:'Boosted market-error model at the opening line.', ou_plus:'Over/Under market logit + features with ridge shrinkage.', gbm_ou:'Boosted errors of the Over/Under market.'
};
const PRICE_NAME = {avg:'Average closing price', best:'Best closing price (line shopping)', sharp:'Sharp bookmaker closing price', open:'Opening price'};
const mktSel = $('#mkt'), priceSel = $('#price'), perSel = $('#period'), focusSel = $('#focus');
let impSide = 'home';
const P = () => perSel.value, isOU = () => mktSel.value === 'ou';
function models(){ return isOU() ? R.models_ou : R.models_1x2; }
function rec(m){ return isOU() ? R.eval.ou[m] : R.eval.models[m]; }
function priceFor(m){ const r = rec(m); if (!r) return null; if (!isOU() && m.endsWith('_open')) return 'open'; return r.bets[priceSel.value] ? priceSel.value : null; }
function fillControls(){
  const prices = isOU() ? Object.keys(R.eval.ou[R.models_ou[0]]?.bets || {}) : R.eval.prices.filter(p => p !== 'open');
  const cur = priceSel.value; priceSel.innerHTML = prices.map(p => `<option value="${p}">${PRICE_NAME[p]}</option>`).join(''); if (prices.includes(cur)) priceSel.value = cur;
  const cf = focusSel.value; focusSel.innerHTML = models().map(m => `<option value="${m}">${m}</option>`).join('');
  focusSel.value = models().includes(cf) ? cf : (isOU() ? (R.models_ou.includes('ou_plus') ? 'ou_plus' : R.models_ou[0]) : (R.models_1x2.includes('stack') ? 'stack' : 'market_plus'));
}

function header(){
  const C = R.coverage;
  $('#sub').innerHTML = `${int(C.eval)} matches evaluated across ${C.leagues} competitions, ${C.from} → ${C.to}. Discovery: ${int(C.n_discovery)} matches · hold-out from <b>${R.holdout_from}</b>: ${int(C.n_holdout)} matches. Database <b>${esc(R.db)}</b> · generated ${R.generated}.`;
  const L = R.leak; const b = [];
  if (L.skipped) b.push(`<span class="badge">Time-travel test skipped</span>`);
  else b.push(`<span class="badge ${L.passed?'ok':'bad'}">${L.passed?'✓':'✕'} Time-travel test ${L.passed?'passed':'FAILED'}: ${L.tests.length} cutoff days, ${L.tests.reduce((a,t)=>a+t.features*t.matches,0).toLocaleString()} feature values, ${L.tests.reduce((a,t)=>a+t.mismatches,0)} changed</span>`);
  if (L.alarms && L.alarms.length) b.push(`<span class="badge bad">⚠ Implausibly strong: ${L.alarms.map(a => esc(a.model) + ' (' + sgn(a.gain*1000, 1) + '×10⁻³)').join(', ')} beat the closing market by too much. Treat as a leak until proven otherwise.</span>`);
  b.push(`<span class="badge">Thresholds fixed before ${R.holdout_from}</span>`, `<span class="badge">Post-kickoff odds discarded</span>`, `<span class="badge">Monthly walk-forward retraining</span>`);
  $('#badges').innerHTML = b.join('');
}

function heroAndTiles(){
  const per = P(), ms = models().filter(m => m !== 'market' && m !== 'market_open');
  const cand = ms.map(m => { const pr = priceFor(m); const x = pr ? rec(m).bets[pr].periods[per] : null; return {m, pr, x}; }).filter(c => c.x && c.x.n >= 100 && c.x.roi != null && c.x.lo != null);
  cand.sort((a, b) => b.x.lo - a.x.lo);   // most reliable: highest lower bound of the 95% interval
  const best = cand[0];
  const llg = ms.map(m => { const mt = rec(m).metrics[per]; return {m, g: mt && mt.logloss != null && mt.mk_logloss != null ? mt.mk_logloss - mt.logloss : null}; }).filter(x => x.g != null).sort((a,b) => b.g - a.g)[0];
  if (!best){ $('#hero').innerHTML = '<div class="lbl">Not enough bets in this period.</div>'; }
  else { const x = best.x, sig = x.p != null && x.p < 0.05;
    $('#hero').innerHTML = `<div><div class="big ${cls(x.roi)}">${pct(x.roi)}</div><div class="lbl">${per === 'holdout' ? 'Hold-out' : per === 'discovery' ? 'Discovery' : 'All-period'} return per bet · <b>${esc(best.m)}</b><br><span class="note">most reliable model: highest lower bound of its 95% interval, 100+ bets</span></div></div>
      <div style="max-width:640px">${int(x.n)} bets at ${PRICE_NAME[best.pr].toLowerCase()} (τ = ${(rec(best.m).bets[best.pr].tau*100).toFixed(0)}% edge), profit ${sgn(x.profit)} units.
      95% interval ${pct(x.lo)} to ${pct(x.hi)}; ${sig ? `<b>unlikely to be chance</b> (p = ${num(x.p,3)})` : `could be chance (p = ${num(x.p,2)})`}.
      ${per !== 'holdout' ? '<br><span class="note">Switch Period to Hold-out for the honest figure.</span>' : ''}
      ${cand.length > 5 ? `<div class="note" style="margin-top:6px">Picked as the best of ${cand.length} models, so expect some of this to be luck of the draw.</div>` : ''}</div>`; }
  const n = ms.map(m => rec(m).metrics[per]?.n || 0); const mk = rec('market')?.metrics[per];
  const pos = cand.filter(c => c.x.roi > 0).length;
  const tiles = [
    ['Matches in period', int(mk?.n), isOU() ? 'with Over/Under prices' : 'with 1X2 prices'],
    ['Market log loss', num(mk?.logloss, 4), `${isOU() ? 'Brier ' + num(mk?.brier,4) : 'RPS ' + num(mk?.rps,4)} · accuracy ${pctu(mk?.acc)}`],
    ['Best forecaster', llg ? esc(llg.m) : '–', llg ? `log-loss gain ${sgn(llg.g*1000,2)} ×10⁻³ vs market` : ''],
    ['Models in profit', `${pos} of ${cand.length}`, `at ${PRICE_NAME[priceSel.value]?.toLowerCase() || ''}`],
    ['Average bookmaker margin', pctu(R.coverage.avg_overround, 2), 'the hurdle every bet must clear'],
  ];
  $('#tiles').innerHTML = tiles.map(t => `<div class="card tile"><div class="lbl">${t[0]}</div><div class="val">${t[1]}</div><div class="ci">${t[2]}</div></div>`).join('');
}

function leaderboard(){
  const per = P(), focus = focusSel.value;
  const head = [{h:'Model', c:'l'}, {h:'Log loss'}, {h:'Gain ×10⁻³', t:'market log loss − model log loss, ×1000'}, ...(isOU()?[{h:'Brier'}]:[{h:'RPS'}]), {h:'Accuracy'},
    {h:'τ'}, {h:'Bets'}, {h:'Won'}, {h:'Avg price'}, {h:'Return'}, {h:'95% CI'}, {h:'p'}, {h:'Profit'}, ...(!isOU()?[{h:'CLV', t:'closing-line value: price taken × closing fair probability − 1'}]:[])];
  const rows = models().map(m => { const r = rec(m), mt = r.metrics[per] || {}, pr = priceFor(m), b = pr ? r.bets[pr] : null, x = b ? b.periods[per] : null;
    const gain = mt.logloss != null && mt.mk_logloss != null ? (mt.mk_logloss - mt.logloss)*1000 : null;
    const v = [m, mt.logloss, gain, isOU() ? mt.brier : mt.rps, mt.acc, b?.tau, x?.n, x?.win, x?.avg_price, x?.roi, x?.lo, x?.p, x?.profit, x?.clv];
    const c = [`${esc(m)}${pr==='open'?' <span class="pill">open</span>':''}`, num(mt.logloss,4), `<span class="${cls(gain)}">${gain==null?'–':sgn(gain,2)}</span>`, num(isOU()?mt.brier:mt.rps,4), pctu(mt.acc),
      b ? (b.tau*100).toFixed(0)+'%' : '–', int(x?.n), pctu(x?.win), num(x?.avg_price), `<b class="${cls(x?.roi)}">${pct(x?.roi)}</b>`, x?.lo!=null?`${pct(x.lo)} to ${pct(x.hi)}`:'–', num(x?.p,3), `<span class="${cls(x?.profit)}">${sgn(x?.profit)}</span>`];
    if (!isOU()) c.push(x?.clv != null ? `<span class="${cls(x.clv)}">${pct(x.clv,2)}</span>` : '–');
    return {k:m, v, c, sel: m === focus}; });
  sortable($('#t-lead'), head, rows, {onRow: k => { focusSel.value = k; draw(); }});
  const rr = models().map(m => { const pr = priceFor(m); const x = pr ? rec(m).bets[pr].periods[per] : null; return {m, x}; }).filter(o => o.x && o.x.n > 0);
  $('#roi-note').textContent = `${per === 'holdout' ? 'Hold-out' : per === 'discovery' ? 'Discovery' : 'All'} period, ${PRICE_NAME[priceSel.value]?.toLowerCase() || ''}; opening-line models use opening prices. Whiskers: 95% interval (weekly block bootstrap).`;
  hbars($('#c-roi'), rr.map(o => ({label:o.m, v:o.x.roi, lo:o.x.lo, hi:o.x.hi, color: o.m === focus ? css('--s1') : css('--ctrl'),
    tip:`<b>${esc(o.m)}</b><br>${int(o.x.n)} bets · won ${pctu(o.x.win)}<br>Return ${pct(o.x.roi)} (${pct(o.x.lo)} to ${pct(o.x.hi)})<br>p = ${num(o.x.p,3)}`, onclick:() => { focusSel.value = o.m; draw(); }})), v => pct(v,0));
  const ll = models().filter(m => m !== 'market').map(m => { const mt = rec(m).metrics[per] || {}; return {m, g: mt.logloss != null && mt.mk_logloss != null ? (mt.mk_logloss - mt.logloss)*1000 : null, n: mt.n}; }).filter(o => o.g != null);
  hbars($('#c-ll'), ll.map(o => ({label:o.m, v:o.g, color: o.m === focus ? css('--s1') : css('--ctrl'), tip:`<b>${esc(o.m)}</b><br>Gain ${sgn(o.g,2)} ×10⁻³ over ${int(o.n)} matches`})), v => sgn(v,1));
}

function focusCharts(){
  const m = focusSel.value, r = rec(m), pr = priceFor(m), b = pr ? r.bets[pr] : null, per = P();
  $('#fname').textContent = m; $('#fdesc').textContent = (DESC[m] || '') + (b ? ` Bets at ${PRICE_NAME[pr].toLowerCase()}, τ = ${(b.tau*100).toFixed(0)}% chosen on discovery.` : '');
  if (!b){ ['#c-cum','#c-kelly','#c-tau','#c-edge','#c-month'].forEach(s => empty($(s), 'This model does not place bets (it is a benchmark).')); }
  else {
    lines($('#c-cum'), [{name:'Profit', color:css('--s1'), pts:b.cum, endLabel: b.cum.length ? sgn(b.cum[b.cum.length-1][1]) + 'u' : ''}], {vline:R.holdout_from, vlabel:'hold-out', yref:0, fmt:v => sgn(v,0), tfmt:v => sgn(v,1)+'u'});
    if (b.kelly) lines($('#c-kelly'), [{name:'Bankroll', color:css('--s1'), pts:b.kelly, endLabel: b.kelly.length ? num(b.kelly[b.kelly.length-1][1],1) : ''}], {vline:R.holdout_from, vlabel:'hold-out', yref:100, zero:false, fmt:v => v.toFixed(0)});
    else empty($('#c-kelly'), 'Not computed for this market');
    const cv = k => b.curve[k].filter(c => c.roi != null && c.n >= 20).map(c => [c.tau*100, c.roi, c.n]);
    lines($('#c-tau'), [{name:'Discovery', color:css('--s2'), pts:cv('discovery'), dots:true}, {name:'Hold-out', color:css('--s1'), pts:cv('holdout'), dots:true}],
      {fmt:v => pct(v,0), xfmt:v => v + '%', xlabel:v => 'τ = ' + v + '%', yref:0, extra:p => p[2].toLocaleString() + ' bets'});
    if (b.edge_buckets) cols($('#c-edge'), b.edge_buckets.map(e => ({label:e.label, v:e.roi, lo:e.lo, hi:e.hi, n:e.n, small:e.n<100,
      tip:`<b>Edge ${e.label}</b><br>${int(e.n)} bets<br>Return ${pct(e.roi)} (${pct(e.lo)} to ${pct(e.hi)})`})), v => pct(v,0), 'Model edge at the price taken · return per bet (grey: under 100 bets)');
    else empty($('#c-edge'), 'Not computed for this market');
    if (b.monthly) cols($('#c-month'), b.monthly.map(x => ({label: x[0].slice(2).replace('-', '/'), v:x[1], n:null, color: x[0] >= R.holdout_from.slice(0,7) ? css('--s1') : css('--ctrl'),
      tip:`<b>${x[0]}</b><br>${x[2]} bets · ${sgn(x[1])} units`})).slice(-36), v => sgn(v,0), 'Last 36 months · blue = hold-out', 6);
    else empty($('#c-month'), 'Not computed for this market');
  }
  calChart($('#c-cal'), r.calibration);
  if (b && b.leagues){ const rows = Object.entries(b.leagues).map(([k, v]) => ({k, v:[k, v.n, v.roi, v.n_hold, v.roi_hold], c:[esc(k), int(v.n), `<span class="${cls(v.roi)}">${pct(v.roi)}</span>`, int(v.n_hold), `<span class="${cls(v.roi_hold)}">${pct(v.roi_hold)}</span>`]}));
    sortable($('#t-league'), [{h:'League', c:'l'}, {h:'Bets (all)'}, {h:'Return (all)'}, {h:'Bets (hold-out)'}, {h:'Return (hold-out)'}], rows, {sortKey:1}); }
  else $('#t-league').innerHTML = '<tr><td class="note">No league breakdown for this selection.</td></tr>';
}

function anglesView(){
  const per = P(), pr = priceSel.value === 'best' ? 'best' : 'avg';
  const A = R.angles.filter(a => a.periods.holdout[pr].n > 0 || a.periods.discovery[pr].n > 0);
  const ord = A.slice().sort((a, b) => (b.periods.holdout[pr].roi ?? -9) - (a.periods.holdout[pr].roi ?? -9));
  hbars($('#c-angles'), ord.map(a => { const h = a.periods.holdout[pr], d = a.periods.discovery[pr];
    return {label:a.title, v:h.n ? h.roi : null, lo:h.lo, hi:h.hi, v2:d.n ? d.roi : null,
      color: h.n >= 50 && h.lo > 0 ? css('--s1') : css('--ctrl'),
      tip:`<b>${esc(a.title)}</b><br>${esc(a.why)}<br>Hold-out: ${int(h.n)} bets, ${pct(h.roi)} (${pct(h.lo)} to ${pct(h.hi)})<br>Discovery: ${int(d.n)} bets, ${pct(d.roi)}`}; }), v => pct(v,0), {W:1240, left:250, rowH:24});
  const rows = A.map(a => { const d = a.periods.discovery[pr], h = a.periods.holdout[pr], al = a.periods.all[pr];
    const held = d.roi != null && h.roi != null && d.roi > 0 && h.roi > 0;
    return {k:a.key, v:[a.title, d.n, d.roi, h.n, h.roi, h.p, h.win, h.implied],
      c:[`<b>${esc(a.title)}</b>${held?' <span class="pill bet">holds</span>':''}<div class="why">${esc(a.why)}</div>`, int(d.n), `<span class="${cls(d.roi)}">${pct(d.roi)}</span>`, int(h.n),
         `<b class="${cls(h.roi)}">${pct(h.roi)}</b>`, num(h.p,3), `${pctu(h.win)} <span class="note">vs ${pctu(h.implied)}</span>`]}; });
  sortable($('#t-angles'), [{h:'Angle', c:'l'}, {h:'Disc. bets'}, {h:'Disc. return'}, {h:'Hold-out bets'}, {h:'Hold-out return'}, {h:'p'}, {h:'Won vs implied'}], rows, {sortKey:4});
}

function importance(){
  const I = Object.assign({}, R.importance), an = I.__anchors__; delete I.__anchors__; const keys = Object.keys(I);
  if (!keys.length){ empty($('#c-imp'), 'Not available'); return; }
  const side = impSide, ord = keys.slice().sort((a, b) => Math.abs(I[b][side+'_mean']) - Math.abs(I[a][side+'_mean'])).slice(0, 22);
  hbars($('#c-imp'), ord.map(k => { const x = I[k], m = x[side+'_mean'], sd = x[side+'_sd'], sp = m >= 0 ? x[side+'_pos'] : 1 - x[side+'_pos'];
    return {label:k.length > 22 ? k.slice(0, 21) + '…' : k, v:m, lo:m - sd, hi:m + sd, color: sp >= .9 ? css('--s1') : css('--ctrl'),
      tip:`<b>${esc(k)}</b><br>Mean coefficient ${num(m,3)} (±${num(sd,3)} across refits)<br>Same sign in ${pctu(sp,0)} of refits`}; }), v => num(v,2), {left:170, rowH:24});
  const rows = keys.map(k => { const x = I[k]; return {k, v:[k, x.home_mean, x.home_pos, x.away_mean, x.away_pos],
    c:[esc(k), `<span class="${cls(x.home_mean)}">${num(x.home_mean,3)}</span>`, pctu(x.home_mean >= 0 ? x.home_pos : 1 - x.home_pos, 0), `<span class="${cls(x.away_mean)}">${num(x.away_mean,3)}</span>`, pctu(x.away_mean >= 0 ? x.away_pos : 1 - x.away_pos, 0)]}; });
  sortable($('#t-imp'), [{h:'Feature', c:'l'}, {h:'Home vs draw'}, {h:'Same sign'}, {h:'Away vs draw'}, {h:'Same sign'}], rows);
  if (an && an.mk_lr_h) $('#lam').insertAdjacentHTML('beforebegin', '');
  $('#lam').innerHTML = (an && an.mk_lr_h ? `<div class="note">Market anchor (unpenalised, not charted): home log-odds ${num(an.mk_lr_h.home,3)} / ${num(an.mk_lr_h.away,3)}, away log-odds ${num(an.mk_lr_a.home,3)} / ${num(an.mk_lr_a.away,3)}. A coefficient of 1 on its own side means the model trusts the market fully.</div>` : '') + (Object.entries(R.lambda || {}).map(([m, v]) => `<div class="note"><b>${esc(m)}</b>${m.startsWith('gbm') ? ' (share of boosted correction kept)' : ' (ridge strength)'}: ${v.map(x => `${x[0]} → ${x[1]}`).join(' · ')}</div>`).join('') || '<div class="note">–</div>');
}

// ---------------- match explorer ----------------
const RW = R.rows, Ccol = RW.cols, NR = Ccol.id.length;
let page = 0, mxRows = [];
const mxLeague = $('#mx-league');
mxLeague.innerHTML = '<option value="">All</option>' + [...new Set(Ccol.league)].sort().map(l => `<option>${esc(l)}</option>`).join('');
const SEL3 = ['Home', 'Draw', 'Away'], SELOU = ['Over', 'Under'];
function mxFilter(){
  const lg = mxLeague.value, tm = $('#mx-team').value.trim().toLowerCase(), show = $('#mx-show').value, per = $('#mx-period').value, m = focusSel.value;
  const bkey = isOU() ? 'ou_' + m : m, bets = RW.bets[bkey];
  mxRows = [];
  for (let i = NR - 1; i >= 0; i--){
    if (lg && Ccol.league[i] !== lg) continue;
    if (tm && !(Ccol.home[i].toLowerCase().includes(tm) || Ccol.away[i].toLowerCase().includes(tm))) continue;
    if (per === 'holdout' && Ccol.date[i] < R.holdout_from) continue;
    if (per === 'discovery' && Ccol.date[i] >= R.holdout_from) continue;
    if (show !== 'all'){ if (!bets || bets[i] < 0) continue; const w = outcomeWon(i, bets[i]); if (show === 'won' && !w) continue; if (show === 'lost' && w) continue; }
    mxRows.push(i);
  }
  page = 0; mxDraw();
}
function outcomeWon(i, sel){ const hg = Ccol.hg[i], ag = Ccol.ag[i]; if (isOU()) return sel === 0 ? hg + ag > 2.5 : hg + ag < 2.5; return sel === (hg > ag ? 0 : hg === ag ? 1 : 2); }
function priceOf(i, sel){ if (isOU()) return sel === 0 ? Ccol.ou_avg_o?.[i] : Ccol.ou_avg_u?.[i]; const m = focusSel.value; const pre = m.endsWith('_open') ? 'open_' : 'avg_'; return Ccol[pre + 'hda'[sel]]?.[i]; }
function mxDraw(){
  const m = focusSel.value, per = 50, t = $('#t-mx'), start = page*per, slice = mxRows.slice(start, start + per);
  const bkey = isOU() ? 'ou_' + m : m, bets = RW.bets[bkey], edges = RW.edges[bkey];
  const head = isOU() ? ['Date','League','Match','Score','Market over','Model over','Over price','Under price','Bet','Edge','P/L']
                      : ['Date','League','Match','Score','Market H / D / A','Model H / D / A','Avg price H / D / A','Bet','Edge','P/L'];
  let h = '<thead><tr>' + head.map((x, k) => `<th class="${k<3?'l':''}">${x}</th>`).join('') + '</tr></thead><tbody>';
  for (const i of slice){
    const sel = bets ? bets[i] : -1, won = sel >= 0 ? outcomeWon(i, sel) : null, pr = sel >= 0 ? priceOf(i, sel) : null, pl = sel >= 0 && pr ? (won ? pr - 1 : -1) : null;
    const mk = RW.probs.market, mp = RW.probs[m];
    const cells = [Ccol.date[i], esc(Ccol.league[i]), `${esc(Ccol.home[i])} – ${esc(Ccol.away[i])}`, `${Ccol.hg[i]}–${Ccol.ag[i]}`];
    if (isOU()) cells.push(pctu(RW.ou.market?.[i]), pctu(RW.ou[m]?.[i]), num(Ccol.ou_avg_o?.[i]), num(Ccol.ou_avg_u?.[i]), sel >= 0 ? `<span class="pill bet">${SELOU[sel]}</span>` : '', sel >= 0 ? pct(edges[i]) : '', pl == null ? '' : `<span class="${cls(pl)}">${sgn(pl,2)}</span>`);
    else cells.push(mk ? [0,1,2].map(k => pctu(mk[k][i], 0)).join(' / ') : '–', mp ? [0,1,2].map(k => pctu(mp[k][i], 0)).join(' / ') : '–',
      ['avg_h','avg_d','avg_a'].map(c => num(Ccol[c]?.[i])).join(' / '), sel >= 0 ? `<span class="pill bet">${SEL3[sel]}</span>` : '', sel >= 0 ? pct(edges[i]) : '', pl == null ? '' : `<span class="${cls(pl)}">${sgn(pl,2)}</span>`);
    h += `<tr data-i="${i}" style="cursor:pointer">` + cells.map((c, k) => `<td class="${k<3?'l':''}">${c}</td>`).join('') + '</tr>';
  }
  t.innerHTML = h + '</tbody>';
  t.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => { const nx = tr.nextElementSibling; if (nx && nx.classList.contains('detail')) { nx.remove(); return; } tr.insertAdjacentHTML('afterend', `<tr class="detail"><td colspan="${head.length}">${detail(+tr.dataset.i)}</td></tr>`); });
  $('#pg-info').textContent = mxRows.length ? `${(start+1).toLocaleString()}–${Math.min(start+per, mxRows.length).toLocaleString()} of ${mxRows.length.toLocaleString()}` : 'No matches';
  $('#mx-note').textContent = `${NR.toLocaleString()} evaluated matches with every model's probabilities and the key pre-match features. Click a row for the full record. The CSV holds every column.`;
}
function detail(i){
  const parts = [];
  for (const m of R.models_1x2) if (RW.probs[m]) parts.push(`<div><span>${esc(m)}</span> ${[0,1,2].map(k => pctu(RW.probs[m][k][i], 1)).join(' / ')}${RW.bets[m] && RW.bets[m][i] >= 0 ? ` · bet ${SEL3[RW.bets[m][i]]}` : ''}</div>`);
  for (const m of R.models_ou) if (RW.ou[m]) parts.push(`<div><span>O/U ${esc(m)}</span> over ${pctu(RW.ou[m][i], 1)}</div>`);
  for (const k of Object.keys(Ccol)) if (!['id','date','league','season','home','away','hg','ag'].includes(k)) parts.push(`<div><span>${esc(k)}</span> ${Ccol[k][i] == null ? '–' : Ccol[k][i]}</div>`);
  return `<div class="note">Fixture ${Ccol.id[i]} · season ${esc(Ccol.season[i])}</div><div class="kv">${parts.join('')}</div>`;
}
$('#pg-prev').onclick = () => { if (page > 0){ page--; mxDraw(); } };
$('#pg-next').onclick = () => { if ((page+1)*50 < mxRows.length){ page++; mxDraw(); } };
for (const s of ['#mx-league', '#mx-show', '#mx-period']) $(s).onchange = mxFilter;
$('#mx-team').oninput = mxFilter;
$('#mx-csv').onclick = () => {
  const keys = Object.keys(Ccol), head = [...keys];
  for (const m of R.models_1x2) if (RW.probs[m]) head.push(`${m}_p_home`, `${m}_p_draw`, `${m}_p_away`);
  for (const m of R.models_ou) if (RW.ou[m]) head.push(`${m}_p_over`);
  for (const m of Object.keys(RW.bets)) head.push(`${m}_bet`, `${m}_edge`);
  const q = v => { if (v == null) return ''; const s = String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const lines_ = [head.join(',')];
  for (let i = 0; i < NR; i++){
    const r = keys.map(k => q(Ccol[k][i]));
    for (const m of R.models_1x2) if (RW.probs[m]) r.push(q(RW.probs[m][0][i]), q(RW.probs[m][1][i]), q(RW.probs[m][2][i]));
    for (const m of R.models_ou) if (RW.ou[m]) r.push(q(RW.ou[m][i]));
    for (const m of Object.keys(RW.bets)) { const b = RW.bets[m][i]; r.push(b < 0 ? '' : (m.startsWith('ou_') ? SELOU[b] : SEL3[b]), q(RW.edges[m][i])); }
    lines_.push(r.join(','));
  }
  const blob = new Blob([lines_.join('\n')], {type:'text/csv'}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'backtest_matches.csv'; a.click();
};

// ---------------- static panels ----------------
function staticPanels(){
  const C = R.coverage, Pm = R.params;
  $('#safeguards').innerHTML = [
    'Features are built day by day: every match on a day is described from the state <b>before</b> that day, and only then are that day\'s results added. A match never sees its own result, stats, xG, ratings or cards, nor anything played later the same day.',
    'Only genuinely pre-match information is used for the current match: past fixtures (rest, congestion; future fixtures are never read because knockout rounds and play-offs only appear once results are known), venue and travel, referee appointment, injuries and suspensions, weather forecast, starting line-ups (published about an hour before kickoff; used only by closing-line models) and odds last updated no later than kickoff.',
    'Season-level tables (standings, team/player/coach/referee season statistics, top scorers) are never read because they include future matches. League tables are rebuilt from results as they happen.',
    `Models are retrained every ${Pm.retrain_months} month(s) on earlier matches only (window ${Pm.train_window_days} days). Regularisation is tuned on the latest 20% of each training window, never on the month being predicted. The stack learns only from earlier out-of-sample predictions.`,
    `Bet thresholds are chosen on the discovery period (before ${R.holdout_from}) and frozen for the hold-out.`,
    'Time-travel test: the engine is re-run with every result from a cutoff day onward deleted <b>and every fixture scheduled after it removed</b>; features for that day\'s matches must not change.',
    `Leakage alarm: any model beating the closing market's log loss by more than ${(Pm.alarm_gain*1000).toFixed(0)}×10⁻³ in the hold-out is flagged at the top of the page.`].map(s => `<li>${s}</li>`).join('');
  const L = R.leak;
  $('#t-leak').innerHTML = L.skipped ? '<tr><td class="note">Skipped (--skip-leak-test)</td></tr>' :
    '<thead><tr><th class="l">Cutoff day</th><th>Matches</th><th>Features</th><th>Changed values</th></tr></thead><tbody>' + L.tests.map(t => `<tr><td class="l">${t.day}</td><td>${t.matches}</td><td>${t.features}</td><td class="${t.mismatches?'neg':'pos'}">${t.mismatches}</td></tr>`).join('') + '</tbody>';
  const cv = [['Matches evaluated', int(C.eval)], ['Feature rows built (all competitions)', int(C.fixtures)], ['Competitions evaluated', C.leagues], ['Period', `${C.from} → ${C.to}`],
    ['Discovery / hold-out matches', `${int(C.n_discovery)} / ${int(C.n_holdout)}`], ['Features in the edge model', C.n_features],
    ['with xG history', pctu(C.with_xg)], ['with starting line-ups', pctu(C.with_lineups)], ['with opening prices', pctu(C.with_open)], ['with sharp prices', pctu(C.with_sharp)],
    ['with Over/Under 2.5 prices', pctu(C.with_ou)], ['with referee history', pctu(C.with_ref)], ['with weather', pctu(C.with_weather)], ['with injury/suspension data', pctu(C.with_sidelined)],
    ['Average bookmaker margin', pctu(C.avg_overround, 2)], ['Burn-in before first prediction', Pm.burn_in_days + ' days'], ['Maximum price backed', Pm.max_odds],
    ['Compute used', C.compute && C.compute.workers ? `${C.compute.workers} worker(s) × ${C.compute.threads} thread(s) on ${C.compute.logical} CPUs, ${C.compute.mem_gb ?? '?'} GB free` : '–'],
    ['Run time', C.runtime_s != null ? `${Math.floor(C.runtime_s/60)} min ${C.runtime_s%60} s` : '–']];
  $('#t-cov').innerHTML = '<tbody>' + cv.map(r => `<tr><td class="l">${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</tbody>';
  $('#mdesc').innerHTML = [...R.models_1x2, ...R.models_ou.filter(m => !R.models_1x2.includes(m))].map(m => `<li><b>${esc(m)}</b>: ${esc(DESC[m] || '')}</li>`).join('');
}

function draw(){ heroAndTiles(); leaderboard(); focusCharts(); anglesView(); importance(); mxFilter(); }
mktSel.onchange = () => { fillControls(); draw(); };
for (const s of [priceSel, perSel, focusSel]) s.onchange = draw;
$('#imp-home').onclick = () => { impSide = 'home'; $('#imp-home').classList.add('on'); $('#imp-away').classList.remove('on'); importance(); };
$('#imp-away').onclick = () => { impSide = 'away'; $('#imp-away').classList.add('on'); $('#imp-home').classList.remove('on'); importance(); };
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; draw(); };
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
fillControls(); header(); staticPanels(); draw();
</script>
</body>
</html>
'''

if __name__ == "__main__":
    main()
