#!/usr/bin/env python3
"""
referee_impact.py - how much does the referee change a football match, and is it priced?
==========================================================================================

Builds a profile of every referee from the Sportmonks PostgreSQL database, using only
matches played BEFORE each fixture. It then measures, out of sample and month by month,
whether knowing the referee improves forecasts of:

  home      match result (1X2)               -> home favouritism, big-club / crowd pressure
  over25    over/under 2.5 goals              -> referees that slow a game down / let it flow
  btts      both teams score                  -> "balancing" referees
  cards     total cards (count)               -> strictness
  card_diff away cards minus home cards       -> home bias in decisions
  red       at least one sending-off          -> what drives sending-off frequency
  pen       at least one penalty
  added2    second-half stoppage time         -> game management
  ht_trail  trailing side scores in the 2nd half, given a one-goal half-time lead   (in-play)
  ht_shut   no further goals in the 2nd half, given a one-goal half-time lead        (in-play)

It builds on backtest_models.py: the same day-by-day leak-proof feature engine, time-travel
test, walk-forward retraining, discovery/hold-out split, block bootstrap and dashboard style.

HOW REFEREE EFFECTS ARE MEASURED (normalisation)
------------------------------------------------
A raw referee average is mostly a measure of the matches the referee is given: top
referees get derbies and title deciders, which have more cards whoever referees them.
So every match first gets a PRE-MATCH EXPECTATION, built only from earlier matches:
  * goals / BTTS / points: the closing market (1X2 + over/under + BTTS, de-vigged; the
    1X2 and O/U prices are turned into expected home and away goals with a Poisson grid);
  * cards / fouls / penalties / shots: both teams' recent rates (what each team commits and
    what each team draws from opponents) times the league rate, shrunk for small samples,
    then adjusted for derby / closeness / cup / late-season stakes by an online regression;
  * stoppage time: league level plus an online regression on what happened in that half
    (goals, cards, substitutions, VAR checks, penalties).
The referee is credited only with the RESIDUAL (actual minus expected), expressed relative
to the league scale, and with the league's average residual removed (so a referee is
compared with the other referees in the same competition). Residuals enter time-decayed
per-referee accumulators and are shrunk towards zero by empirical Bayes:
      trait = sum(w * r) / (sum(w) + k),   k = within-referee variance / between-referee variance
k is re-estimated every 14 days from referees' PAST matches only. A referee with 5 matches
therefore sits near 0; one with 150 consistent matches keeps most of their own average.

Referee traits (28), each a residual as above
  Discipline : cards, fouls, cards-per-foul (strictness given the fouls), reds, second-yellow
               propensity, first-card minute, penalties, VAR checks
  Home       : home points vs market, card / foul / penalty asymmetry, composite decision index
  Pressure   : stoppage time when the home side is losing by one vs winning by one at 90',
               slope of the decision index on crowd size, on team strength (market) and on
               the team's "decision-winning reputation"; favourites' points vs market
  Game flow  : goals vs market, BTTS vs market, goals after the first goal (slowing a game),
               trailing side scoring after the first goal (balancing), card alternation
               between the two sides (evening up - measured against the exact permutation
               expectation), cards to the leading side after it scores, penalties to the
               trailing side, 1st/2nd-half stoppage time, shot volume

"Pressure" is the easiest thing to over-read. It is only measured as a SLOPE: does this
referee's decision balance move consistently with a pre-match pressure measure (crowd
size, favourite status, a team's history of winning decisions) across many matches? A
single match says nothing. The dashboard reports the between-referee spread of each trait,
its split-half reliability and a permutation p-value, so a trait that is really noise is
labelled as such.

NO LOOK-AHEAD - the same guarantees as backtest_models.py, plus a placebo
--------------------------------------------------------------------------
 1. Day-by-day state machine: all of a day's feature rows are written before any of that
    day's results enter the state. Referee accumulators, team rates, league rates and the
    online regressions are only updated after the match.
 2. Pre-match inputs only: the referee appointment, past fixtures, venues, weather, odds
    whose last update is not after kickoff. Season-level tables (referee_season_stats,
    analysis.referee_season_summary, standings, team/player season stats) are NOT read:
    they contain the season's future matches.
 3. In-play targets (ht_*) add only the half-time score; everything else is pre-match.
 4. Models are retrained every --retrain-months using only earlier matches; ridge strength
    and boosting shrinkage are chosen on the latest 20% of each training window.
 5. Time-travel test: the engine is re-run with every result from a cutoff day deleted;
    features for that day's matches must be identical.
 6. Placebo: the engine is re-run with referees shuffled between matches of the same league
    and season. Placebo "referee" features carry the same structure but no real referee
    information; their lift over the context model should be ~0. Real lift must beat it.
 7. Bet thresholds and angle cut-offs are fixed on the discovery period only; the hold-out
    figures are the honest ones. An alarm fires if anything beats the closing market by an
    implausible margin.

Usage
-----
    pip install pandas numpy scipy "psycopg[binary]" scikit-learn
    python3 referee_impact.py                                   # everything
    python3 referee_impact.py --leagues 8,82 --export-csv ref_rows.csv
    python3 referee_impact.py --targets btts,cards,red --no-placebo
  Connection: --dsn, DATABASE_URL, or load_league.DATABASE next to the script (as before).
  Options: --holdout-from 2025-11-01 --burn-in-days 365 --retrain-months 2 --market-btts 14
           --market-cards <id> (enables card-line betting) --upcoming-days 10 --cache .cache
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
import warnings
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# Several training processes run side by side, one thread each. OpenMP (gradient boosting) and BLAS
# must be told before numpy / scikit-learn load: oversubscribed OpenMP can be 100x slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("REFEREE_THREADS", "1"))

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy"}
OPTIONAL = {"sklearn": "scikit-learn", "threadpoolctl": "threadpoolctl"}


def ensure_packages() -> None:
    import importlib.util
    miss = lambda d: [p for m, p in d.items() if importlib.util.find_spec(m) is None]  # noqa: E731
    req, opt = miss(REQUIRED), miss(OPTIONAL)
    if importlib.util.find_spec("psycopg") is None and importlib.util.find_spec("psycopg2") is None:
        opt.append("psycopg[binary]")
    if not req and not opt or os.environ.get("BACKTEST_NO_INSTALL"):
        if req:
            sys.exit(f"Missing packages: {' '.join(req)}  ->  {sys.executable} -m pip install {' '.join(req + opt)}")
        return
    print(f"Installing: {', '.join(req + opt)} ...", file=sys.stderr, flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *req, *opt], capture_output=True, text=True)
    if r.returncode != 0 and req:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *req], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"Automatic install failed:\n{r.stderr[-1000:]}\nInstall by hand: pip install {' '.join(req + opt)}")
    importlib.invalidate_caches()


if __name__ == "__main__":
    ensure_packages()

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import nbinom

T0 = time.time()
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# =============================================================================
# Database access (psycopg 3 -> psycopg2 -> psql CLI), as in backtest_models.py
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
            cfg = " ".join(f"{keymap.get(k, k)}='{str(v).replace(chr(39), chr(92) + chr(39))}'" for k, v in cfg.items() if v is not None)
        if cfg:
            return str(cfg), "load_league.DATABASE"
    except Exception:  # noqa: BLE001
        pass
    return "", "libpq defaults (PG* environment variables)"


DATE_COLS = {"starting_at"}


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
        self.name = self.query("SELECT current_database() AS db").db.iloc[0]
        log(f"database driver: {self.kind} · connected to '{self.name}' (settings from {source})")

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
        c = DB.__new__(DB)
        c.dsn, c.kind, c.name, c.conn = self.dsn, self.kind, self.name, None
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


# =============================================================================
# SQL extracts. Only per-match tables; no season aggregates (they contain the future).
# =============================================================================
FINISHED = "('FT', 'AET', 'FT_PEN')"
LEAGUE_TYPE = """CASE WHEN lower(COALESCE(l.type, 'league')) = 'league'
                AND COALESCE(l.sub_type, '') !~* '(cup|play|international|friendly)' THEN 'league' ELSE 'cup' END"""

SQL = {
"fixtures": f"""
    WITH sc AS (
        SELECT fixture_id,
               max(goals) FILTER (WHERE location = 'home' AND description = 'CURRENT')  AS h_cur,
               max(goals) FILTER (WHERE location = 'away' AND description = 'CURRENT')  AS a_cur,
               max(goals) FILTER (WHERE location = 'home' AND description = '1ST_HALF') AS h_ht,
               max(goals) FILTER (WHERE location = 'away' AND description = '1ST_HALF') AS a_ht,
               max(goals) FILTER (WHERE location = 'home' AND description = '2ND_HALF') AS h_90,
               max(goals) FILTER (WHERE location = 'away' AND description = '2ND_HALF') AS a_90
        FROM football.fixture_scores WHERE description IN ('CURRENT', '1ST_HALF', '2ND_HALF') GROUP BY fixture_id)
    SELECT f.fixture_id, f.league_id, l.name AS league, {LEAGUE_TYPE} AS league_type,
           f.season_id, s.name AS season, f.starting_at, f.venue_id,
           COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
           ht.name AS home, at.name AS away, st.developer_name AS state,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN COALESCE(sc.h_90, sc.h_cur) ELSE sc.h_cur END AS hg,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN COALESCE(sc.a_90, sc.a_cur) ELSE sc.a_cur END AS ag,
           sc.h_ht, sc.a_ht
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
"upcoming": f"""
    SELECT f.fixture_id, f.league_id, l.name AS league, {LEAGUE_TYPE} AS league_type,
           f.season_id, s.name AS season, f.starting_at, f.venue_id,
           COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
           ht.name AS home, at.name AS away
    FROM football.fixtures f
    LEFT JOIN core.states st ON st.state_id = f.state_id
    LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
    LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
    LEFT JOIN football.leagues l ON l.league_id = f.league_id
    LEFT JOIN football.seasons s ON s.season_id = f.season_id
    LEFT JOIN football.teams ht ON ht.team_id = COALESCE(f.home_team_id, hp.team_id)
    LEFT JOIN football.teams at ON at.team_id = COALESCE(f.away_team_id, ap.team_id)
    WHERE COALESCE(st.developer_name, 'NS') IN ('NS', 'TBA', 'DELAYED')
      AND f.starting_at >= now() - interval '1 day' AND f.starting_at <= now() + interval '__DAYS__ days'""",
"schedule": """
    SELECT f.starting_at, COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id
    FROM football.fixtures f
    LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
    LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
    WHERE f.starting_at IS NOT NULL""",
"referees": """
    SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
    FROM football.fixture_officials fo LEFT JOIN core.types t ON t.type_id = fo.type_id
    WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
    ORDER BY fo.fixture_id, fo.fixture_official_id""",
"ref_names": """SELECT referee_id, COALESCE(display_name, common_name, name) AS name, country_id FROM football.referees""",
"events": """
    SELECT e.fixture_id, e.team_id, t.developer_name AS kind, e.minute, e.extra_minute, e.result,
           e.player_id, e.coach_id, COALESCE(e.sort_order, 0) AS so, e.event_id
    FROM football.fixture_events e JOIN core.types t ON t.type_id = e.type_id
    WHERE e.rescinded IS NOT TRUE AND e.minute IS NOT NULL AND e.minute <= 90
      AND (t.developer_name IN ('GOAL', 'OWNGOAL', 'PENALTY', 'MISSED_PENALTY', 'YELLOWCARD', 'REDCARD', 'YELLOWREDCARD', 'SUBSTITUTION')
           OR t.developer_name LIKE 'VAR%')""",
"periods": """
    SELECT fixture_id, sort_order, lower(COALESCE(description, '')) AS descr, time_added FROM football.fixture_periods
    WHERE time_added IS NOT NULL""",
"teamstats": """
    SELECT s.fixture_id, s.team_id,
        max(s.value) FILTER (WHERE t.developer_name = 'FOULS')             AS fouls,
        max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_TOTAL')       AS shots,
        max(s.value) FILTER (WHERE t.developer_name = 'FREE_KICKS')        AS free_kicks,
        max(s.value) FILTER (WHERE t.developer_name = 'BALL_POSSESSION')   AS poss,
        max(s.value) FILTER (WHERE t.developer_name = 'DANGEROUS_ATTACKS') AS da
    FROM football.fixture_statistics s JOIN core.types t ON t.type_id = s.type_id
    WHERE t.developer_name IN ('FOULS', 'SHOTS_TOTAL', 'FREE_KICKS', 'BALL_POSSESSION', 'DANGEROUS_ATTACKS')
    GROUP BY 1, 2""",
"venues": "SELECT venue_id, capacity FROM football.venues WHERE capacity IS NOT NULL AND capacity > 0",
"team_venues": "SELECT team_id, venue_id FROM football.teams WHERE venue_id IS NOT NULL",
"rivals": "SELECT team_id, rival_team_id FROM football.rivals",
"weather": """SELECT fixture_id, wind_speed, lower(COALESCE(metric, '')) AS metric, lower(COALESCE(description, '')) AS wdesc
              FROM football.fixture_weather""",
}

SIDES = {
    "1x2": ("H", "D", "A"), "ou": ("O", "U"), "btts": ("Y", "N"), "cards": ("O", "U"),
}
SIDE_SQL = {
    "1x2": """CASE WHEN lower(coalesce(o.label,'')) IN ('home','1') THEN 'H' WHEN lower(coalesce(o.label,'')) IN ('draw','x') THEN 'D'
                   WHEN lower(coalesce(o.label,'')) IN ('away','2') THEN 'A' END""",
    "ou": "CASE WHEN lower(coalesce(o.label,'')) = 'over' THEN 'O' WHEN lower(coalesce(o.label,'')) = 'under' THEN 'U' END",
    "btts": "CASE WHEN lower(coalesce(o.label,'')) IN ('yes','y') THEN 'Y' WHEN lower(coalesce(o.label,'')) IN ('no','n') THEN 'N' END",
    "cards": "CASE WHEN lower(coalesce(o.label,'')) = 'over' THEN 'O' WHEN lower(coalesce(o.label,'')) = 'under' THEN 'U' END",
}


def odds_sql(kind: str, market: str) -> str:
    """Closing pre-match prices: rows whose last bookmaker update is after kickoff are discarded."""
    sides = SIDES[kind]
    extra = " AND core.num(o.total) = 2.5" if kind == "ou" else ""
    line = ", core.num(o.total) AS line" if kind == "cards" else ""
    gline = ", line" if kind == "cards" else ""
    piv = ", ".join(f"avg(price) FILTER (WHERE side = '{s}') AS o_{s.lower()}" for s in sides)
    return f"""
    WITH o AS (
        SELECT o.fixture_id, o.bookmaker_id, o.decimal_odds AS price, {SIDE_SQL[kind]} AS side{line}
        FROM odds.prematch_odds o JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE ({market}) AND o.decimal_odds > 1{extra}
          AND (o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at)
        UNION ALL
        SELECT o.fixture_id, o.bookmaker_id, o.decimal_odds, {SIDE_SQL[kind]}{line}
        FROM odds.premium_odds o JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE ({market}) AND o.decimal_odds > 1{extra}
          AND (o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at))
    SELECT o.fixture_id, o.bookmaker_id{gline}, {piv}
    FROM o WHERE side IS NOT NULL GROUP BY o.fixture_id, o.bookmaker_id{gline}"""


def extract(db: DB, args) -> dict[str, pd.DataFrame]:
    import concurrent.futures as cf
    q = dict(SQL)
    q["upcoming"] = q["upcoming"].replace("__DAYS__", str(int(args.upcoming_days)))
    q["odds_1x2"] = odds_sql("1x2", f"o.market_id = {args.market_1x2} OR o.market_id IN (SELECT market_id FROM odds.markets WHERE developer_name = 'FULLTIME_RESULT')")
    q["odds_ou"] = odds_sql("ou", f"o.market_id = {args.market_ou}")
    if args.market_btts:
        q["odds_btts"] = odds_sql("btts", f"o.market_id = {args.market_btts}")
    if args.market_cards:
        q["odds_cards"] = odds_sql("cards", f"o.market_id = {args.market_cards}")
    required = {"fixtures", "referees", "events"}
    cache = Path(args.cache) if args.cache else None
    out, todo = {}, []
    for name, sql in q.items():
        path = cache / f"ref_{name}.pkl" if cache else None
        if path and path.exists() and not args.refresh and name != "upcoming":
            out[name] = pd.read_pickle(path)
            log(f"{name:12s} {len(out[name]):>11,} rows (cache)")
        else:
            todo.append((name, sql, path))
    weight = {"odds_1x2": 9, "odds_ou": 8, "odds_btts": 7, "odds_cards": 7, "events": 6, "teamstats": 5}
    todo.sort(key=lambda t: -weight.get(t[0], 0))
    jobs = max(1, min(len(todo), args.db_jobs or 4))
    if todo:
        log(f"reading {len(todo)} extracts with {jobs} parallel connection(s) ...")

    def run(sql):
        conn = db.clone()
        try:
            return conn.query(sql), None
        except Exception as e:  # noqa: BLE001
            return None, e
        finally:
            conn.close()
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run, sql): (name, path) for name, sql, path in todo}
        for f in cf.as_completed(futs):
            name, path = futs[f]
            res, err = f.result()
            if err is not None:
                if name in required:
                    sys.exit(f"Required query '{name}' failed: {err}")
                log(f"{name}: not available ({str(err).splitlines()[0][:110]}) - continuing without it")
                res = pd.DataFrame()
            out[name] = res
            log(f"{name:12s} {len(res):>11,} rows")
            if path and name != "upcoming":
                path.parent.mkdir(parents=True, exist_ok=True)
                res.to_pickle(path)
    return out


# =============================================================================
# Market: de-vigged consensus + expected goals implied by 1X2 and O/U 2.5
# =============================================================================
GOALS = np.arange(11)
LOGFACT = np.array([math.lgamma(k + 1) for k in GOALS])


def poisson_probs(lh, la):
    lh, la = np.clip(lh, 0.05, 6), np.clip(la, 0.05, 6)
    ph = np.exp(GOALS[None, :] * np.log(lh[:, None]) - lh[:, None] - LOGFACT[None, :])
    pa = np.exp(GOALS[None, :] * np.log(la[:, None]) - la[:, None] - LOGFACT[None, :])
    m = ph[:, :, None] * pa[:, None, :]
    i, j = np.meshgrid(GOALS, GOALS, indexing="ij")
    tot = m.sum(axis=(1, 2))
    return ((m * (i > j)).sum((1, 2)) / tot, (m * (i < j)).sum((1, 2)) / tot, (m * (i + j > 2)).sum((1, 2)) / tot)


def implied_lambdas(p_h, p_a, p_o):
    """Expected home/away goals reproducing the market's 1X2 and O/U 2.5 (grid search).
    Without an O/U price the total is pulled weakly towards 2.6 goals."""
    g = np.arange(0.15, 4.01, 0.05)
    LH, LA = np.meshgrid(g, g, indexing="ij")
    LH, LA = LH.ravel(), LA.ravel()
    gh, ga, go = poisson_probs(LH, LA)
    n = len(p_h)
    out = np.full((n, 2), np.nan)
    for s in range(0, n, 1500):
        ph, pa, po = p_h[s:s + 1500, None], p_a[s:s + 1500, None], p_o[s:s + 1500, None]
        err = (gh[None] - ph) ** 2 + (ga[None] - pa) ** 2
        err = err + np.where(np.isnan(po), 0.004 * ((LH + LA)[None] - 2.6) ** 2, (go[None] - np.nan_to_num(po)) ** 2)
        k = np.nanargmin(np.where(np.isnan(ph), np.inf, err), axis=1)
        out[s:s + 1500] = np.c_[LH[k], LA[k]]
    ok = ~(np.isnan(p_h) | np.isnan(p_a))
    out[~ok] = np.nan
    return out


def devig(df: pd.DataFrame, sides) -> pd.DataFrame:
    cols = [f"o_{s.lower()}" for s in sides]
    df = df.dropna(subset=cols)
    df = df[(df[cols] > 1).all(axis=1)]
    inv = 1 / df[cols].to_numpy(float)
    over = inv.sum(axis=1)
    df = df.assign(overround=over - 1, **{f"q_{s.lower()}": inv[:, i] / over for i, s in enumerate(sides)})
    return df[(df.overround > -0.02) & (df.overround < 0.35)]


def market_table(d: dict) -> pd.DataFrame:
    out = []
    for kind, pre in (("1x2", ""), ("ou", "ou_"), ("btts", "btts_")):
        c = d.get(f"odds_{kind}", pd.DataFrame())
        if c is None or c.empty:
            continue
        sides = [s.lower() for s in SIDES[kind]]
        c = devig(c, SIDES[kind])
        g = c.groupby("fixture_id")
        m = pd.DataFrame({f"{pre}n_books": g.size()})
        for s in sides:
            m[f"{pre}p_{s}"] = g[f"q_{s}"].mean()
            m[f"{pre}avg_{s}"] = g[f"o_{s}"].mean()
            m[f"{pre}best_{s}"] = g[f"o_{s}"].max()
        tot = m[[f"{pre}p_{s}" for s in sides]].sum(axis=1)
        for s in sides:
            m[f"{pre}p_{s}"] /= tot
        out.append(m)
    mk = pd.concat(out, axis=1) if out else pd.DataFrame(index=pd.Index([], name="fixture_id"))
    for c in ("p_h", "p_d", "p_a", "ou_p_o", "btts_p_y"):
        if c not in mk:
            mk[c] = np.nan
    lam = implied_lambdas(mk.p_h.to_numpy(float), mk.p_a.to_numpy(float), mk.ou_p_o.to_numpy(float))
    mk["lam_h"], mk["lam_a"] = lam[:, 0], lam[:, 1]
    # card line: per fixture the line whose de-vigged price is closest to 50/50
    c = d.get("odds_cards", pd.DataFrame())
    if c is not None and not c.empty:
        c = devig(c.dropna(subset=["line"]), SIDES["cards"])
        g = c.groupby(["fixture_id", "line"]).agg(p_o=("q_o", "mean"), avg_o=("o_o", "mean"), avg_u=("o_u", "mean"),
                                                  best_o=("o_o", "max"), best_u=("o_u", "max")).reset_index()
        g["dist"] = (g.p_o - 0.5).abs()
        g = g.sort_values(["fixture_id", "dist"]).drop_duplicates("fixture_id").set_index("fixture_id")
        for k in ("line", "p_o", "avg_o", "avg_u", "best_o", "best_u"):
            mk[f"cards_{k}"] = g[k]
    return mk


# =============================================================================
# Small online estimators
# =============================================================================
class EW:
    """Bias-corrected exponentially weighted mean (and variance) with an observation count."""
    __slots__ = ("s", "ss", "w", "n")

    def __init__(self):
        self.s = self.ss = self.w = 0.0
        self.n = 0

    def add(self, x, a):
        if x is None or not np.isfinite(x):
            return
        self.s = (1 - a) * self.s + a * x
        self.ss = (1 - a) * self.ss + a * x * x
        self.w = (1 - a) * self.w + a
        self.n += 1

    def get(self, default=np.nan):
        return self.s / self.w if self.w > 1e-9 else default

    def sd(self, default=np.nan):
        if self.w < 1e-9 or self.n < 5:
            return default
        m = self.s / self.w
        return math.sqrt(max(self.ss / self.w - m * m, 1e-12))


class Acc:
    """Time-decayed weighted sums for one referee trait: shrunk mean = wx / (w + k)."""
    __slots__ = ("w", "wx", "wxx", "n")

    def __init__(self):
        self.w = self.wx = self.wxx = 0.0
        self.n = 0

    def add(self, x, decay, wt=1.0):
        self.w = self.w * decay + wt
        self.wx = self.wx * decay + wt * x
        self.wxx = self.wxx * decay + wt * x * x
        self.n += 1

    def shrunk(self, k):
        return self.wx / (self.w + k) if self.w > 0 else 0.0


class RLS:
    """Recursive least squares with slow forgetting (online linear regression)."""

    def __init__(self, p, forget=0.9995, delta=50.0):
        self.th, self.P, self.f, self.n = np.zeros(p), np.eye(p) * delta, forget, 0

    def predict(self, x):
        return float(self.th @ x) if self.n >= 200 else 0.0

    def update(self, x, y):
        if not np.isfinite(y) or not np.all(np.isfinite(x)):
            return
        Px = self.P @ x
        k = Px / (self.f + x @ Px)
        self.th = self.th + k * (y - self.th @ x)
        self.P = (self.P - np.outer(k, Px)) / self.f
        self.n += 1


def nz(x, d=0.0):
    return d if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


# =============================================================================
# Referee traits
# =============================================================================
TRAITS = [
    # key, group, label, description (positive value means ...)
    ("cards", "Discipline", "Card volume", "Booking points vs what these teams and this kind of fixture normally produce (share of league average). + = more cards."),
    ("fouls", "Discipline", "Fouls called", "Fouls given vs expectation from both teams' foul rates. + = whistles more."),
    ("strict", "Discipline", "Cards per foul", "Booking points beyond what the fouls actually called would normally earn. + = quicker to book."),
    ("reds", "Discipline", "Red cards", "Sending-offs per match vs expectation. + = more reds."),
    ("second_yellow", "Discipline", "Second-yellow propensity", "Share of already-booked players who get a second yellow. + = less forgiving."),
    ("early_card", "Discipline", "First-card minute", "Minute of the first card vs league. − = books early (sets a tone)."),
    ("pens", "Discipline", "Penalties", "Penalties awarded vs expectation from both teams' rates. + = more penalties."),
    ("var", "Discipline", "VAR involvement", "VAR events per match vs league. + = more reviews."),
    ("home_pts", "Home", "Home points vs market", "Home side's points minus market-expected points. + = home does better than priced."),
    ("home_cards", "Home", "Card asymmetry", "(Away − home booking points) minus expectation. + = books the away side more."),
    ("home_fouls", "Home", "Foul asymmetry", "(Fouls given against away − against home) minus expectation. + = penalises the away side more."),
    ("home_pens", "Home", "Penalty asymmetry", "(Home − away penalties) minus expectation. + = more penalties to the home side."),
    ("home_dec", "Home", "Home decision index", "Standardised sum of the three asymmetries. + = decisions favour the home side."),
    ("fergie", "Pressure", "Stoppage time when home needs it", "2nd-half added time (beyond what events explain) when home trails by one at 90', minus when it leads by one. + = more time when home needs a goal."),
    ("crowd_slope", "Pressure", "Crowd-size slope", "How much the home decision index rises per SD of stadium capacity. + = bigger crowd, more home-friendly."),
    ("fav_slope", "Pressure", "Big-team slope", "How much the decision index moves towards the market favourite per unit of its strength edge. + = favours strong teams."),
    ("rep_slope", "Pressure", "Reputation slope", "How much the decision index follows each team's history of winning decisions. + = susceptible to teams that pressure officials."),
    ("fav_pts", "Pressure", "Favourite points vs market", "Market favourite's points minus market-expected points. + = favourites over-perform."),
    ("goals", "Flow", "Goals vs market", "Goals minus market-implied expected goals. + = higher-scoring than priced."),
    ("btts", "Flow", "BTTS vs market", "Both-teams-score outcome minus market probability. + = both sides score more often than priced."),
    ("after_lead", "Flow", "Goals after the first goal", "Goals after the first goal minus market expectation for the remaining time. − = game dies once someone leads (slowing)."),
    ("trailer_scores", "Flow", "Trailing side scores", "Side that conceded first goes on to score, minus market expectation. + = balancing."),
    ("alternation", "Flow", "Card alternation", "How often consecutive cards switch team, minus the exact permutation expectation. + = evens up bookings."),
    ("lead_cards", "Flow", "Cards to the leader", "Cards to the leading side minus cards to the trailing side after the first goal (vs league). + = punishes the leader."),
    ("pen_trailer", "Flow", "Penalties to the trailing side", "Share of penalties awarded while not level that go to the trailing side (vs league). + = balancing."),
    ("added1", "Flow", "1st-half stoppage time", "Added time beyond what the half's events explain. + = more added time."),
    ("added2", "Flow", "2nd-half stoppage time", "Added time beyond what the half's events explain. + = more added time."),
    ("shots", "Flow", "Shot volume", "Total shots vs both teams' shot rates. + = more open games."),
]
TKEYS = [t[0] for t in TRAITS]
SLOPES = {"crowd_slope", "fav_slope", "rep_slope"}
CARD_KINDS = ("YELLOWCARD", "REDCARD", "YELLOWREDCARD")
GOAL_KINDS = ("GOAL", "OWNGOAL", "PENALTY")


def build_facts(d: dict, fx: pd.DataFrame) -> dict:
    """Everything that happened in each match (result data: only read by Engine.update)."""
    facts = {}
    hid = dict(zip(fx.fixture_id.astype(int), fx.home_id.astype(int)))
    aid = dict(zip(fx.fixture_id.astype(int), fx.away_id.astype(int)))
    ev = d["events"]
    groups = {}
    if len(ev):
        ev = ev[ev.fixture_id.isin(hid.keys())].copy()
        ev["extra_minute"] = ev.extra_minute.fillna(0)
        ev = ev.sort_values(["fixture_id", "minute", "extra_minute", "so", "event_id"])
        cols = [ev[c].to_numpy() for c in ("fixture_id", "team_id", "kind", "minute", "extra_minute", "result", "player_id", "coach_id")]
        fids = cols[0].astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(fids)) + 1]
        ends = np.r_[starts[1:], len(fids)]
        for s, e in zip(starts, ends):
            groups[int(fids[s])] = [tuple(c[i] for c in cols[1:]) for i in range(s, e)]
    per = {}
    if len(d.get("periods", [])):
        for r in d["periods"].itertuples():
            half = 1 if ("1st" in r.descr or "first" in r.descr or r.sort_order == 1) else (2 if ("2nd" in r.descr or "second" in r.descr or r.sort_order == 2) else 0)
            if half:
                per.setdefault(int(r.fixture_id), {})[half] = float(r.time_added)
    st = {}
    if len(d.get("teamstats", [])):
        for r in d["teamstats"].itertuples():
            st[(int(r.fixture_id), int(r.team_id))] = r
    for r in fx.itertuples():
        fid, h, a = int(r.fixture_id), hid[int(r.fixture_id)], aid[int(r.fixture_id)]
        F = dict(hg=r.hg, ag=r.ag, ht_h=r.h_ht, ht_a=r.a_ht)
        evs = groups.get(fid)
        p = per.get(fid, {})
        F["added1"], F["added2"] = p.get(1, np.nan), p.get(2, np.nan)
        for side, t in (("h", h), ("a", a)):
            s_ = st.get((fid, t))
            for k in ("fouls", "shots", "free_kicks"):
                F[f"{k}_{side}"] = float(getattr(s_, k)) if s_ is not None and pd.notna(getattr(s_, k, np.nan)) else np.nan
        if not evs:
            F["has_events"] = False
            facts[fid] = F
            continue
        F["has_events"] = True
        c = dict(y_h=0, y_a=0, r_h=0, r_a=0, yr_h=0, yr_a=0, pen_h=0, pen_a=0, var=0, var1=0, var2=0, subs1=0, subs2=0,
                 goals1=0, goals2=0, cards1=0, cards2=0, pens1=0, pens2=0, coach_h=0, coach_a=0, pen_nonlevel=0, pen_trailer=0)
        booked, seq, first_card = set(), [], np.nan
        sh = sa = 0
        timeline_ok = True
        score90 = None
        lead_min, leader = np.nan, None
        aft = dict(lg=0, tg=0, lc=0, tc=0)
        for team, kind, minute, extra, result, player, coach in evs:
            side = "h" if team == h else ("a" if team == a else None)
            half = 1 if minute <= 45 else 2
            if score90 is None and minute >= 90 and extra > 0:
                score90 = sh - sa
            if kind in GOAL_KINDS:
                nh, na = sh, sa
                ok = False
                if isinstance(result, str) and "-" in result:
                    try:
                        nh, na = (int(x) for x in result.split("-")[:2])
                        ok = (nh + na == sh + sa + 1)
                    except ValueError:
                        ok = False
                if not ok:
                    scorer = side if kind != "OWNGOAL" else {"h": "a", "a": "h"}.get(side)
                    if scorer is None:
                        timeline_ok = False
                        continue
                    nh, na = sh + (scorer == "h"), sa + (scorer == "a")
                scorer = "h" if nh > sh else "a"
                c[f"goals{half}"] += 1
                if leader is not None and minute >= lead_min:
                    aft["lg" if scorer == leader else "tg"] += 1
                if sh == sa == 0 and leader is None:
                    lead_min, leader = minute, scorer
                pre_h, pre_a = sh, sa
                sh, sa = nh, na
                if kind == "PENALTY" and side:
                    c[f"pen_{side}"] += 1
                    c[f"pens{half}"] += 1
                    if pre_h != pre_a:
                        c["pen_nonlevel"] += 1
                        c["pen_trailer"] += int((side == "h" and pre_h < pre_a) or (side == "a" and pre_a < pre_h))
                continue
            if kind == "MISSED_PENALTY" and side:
                c[f"pen_{side}"] += 1
                c[f"pens{half}"] += 1
                if sh != sa:
                    c["pen_nonlevel"] += 1
                    c["pen_trailer"] += int((side == "h" and sh < sa) or (side == "a" and sa < sh))
                continue
            if kind in CARD_KINDS:
                if side is None:
                    continue
                if pd.notna(coach) and pd.isna(player):
                    c[f"coach_{side}"] += 1
                    continue
                if kind == "YELLOWCARD":
                    c[f"y_{side}"] += 1
                    if pd.notna(player):
                        booked.add(player)
                elif kind == "REDCARD":
                    c[f"r_{side}"] += 1
                else:
                    c[f"yr_{side}"] += 1
                c[f"cards{half}"] += 1
                seq.append(0 if side == "h" else 1)
                if np.isnan(first_card):
                    first_card = minute + 0.0
                if leader is not None and minute >= lead_min:
                    aft["lc" if side == leader else "tc"] += 1
                continue
            if kind == "SUBSTITUTION":
                c[f"subs{half}"] += 1
                continue
            if kind.startswith("VAR"):
                c["var"] += 1
                c[f"var{half}"] += 1
        if score90 is None:
            score90 = sh - sa
        if (sh, sa) != (r.hg, r.ag):
            timeline_ok = False
        F.update(c)
        F.update(booked=len(booked), seq=seq, first_card=first_card, timeline_ok=timeline_ok, score90=score90 if timeline_ok else np.nan,
                 lead_min=lead_min if timeline_ok else np.nan, leader=leader if timeline_ok else None,
                 aft_lg=aft["lg"], aft_tg=aft["tg"], aft_lc=aft["lc"], aft_tc=aft["tc"])
        F["cards_h"] = c["y_h"] + c["r_h"] + c["yr_h"]
        F["cards_a"] = c["y_a"] + c["r_a"] + c["yr_a"]
        F["bp_h"] = 10 * c["y_h"] + 25 * (c["r_h"] + c["yr_h"])
        F["bp_a"] = 10 * c["y_a"] + 25 * (c["r_a"] + c["yr_a"])
        F["reds_h"], F["reds_a"] = c["r_h"] + c["yr_h"], c["r_a"] + c["yr_a"]
        facts[fid] = F
    return facts


# =============================================================================
# The leak-proof feature engine
# =============================================================================
TEAM_M = ("bp_for", "bp_ag", "cards_for", "fouls_for", "fouls_drawn", "reds_for", "pens_for", "pens_ag", "shots_for", "shots_ag",
          "gf", "ga", "coach", "dec")
LG_M = ("bp", "cards", "fouls", "reds", "pens", "shots", "goals", "btts", "var", "added1", "added2", "fcm", "sy", "cap",
        "phd", "D") + tuple("T_" + k for k in TKEYS)
A_TEAM = 1 - 0.5 ** (1 / 12)
A_LG = 1 - 0.5 ** (1 / 150)


class Engine:
    """emit() reads only pre-match information and the state; update() is the only place
    results enter the state, and runs after every match of the day has been emitted."""

    def __init__(self, d: dict, mk: pd.DataFrame, args, ref_map: dict | None = None, keep_log=True):
        self.args, self.keep_log = args, keep_log
        fx = d["fixtures"].dropna(subset=["home_id", "away_id", "starting_at"]).copy()
        fx[["home_id", "away_id"]] = fx[["home_id", "away_id"]].astype("int64")
        fx["day"] = fx.starting_at.dt.tz_convert(None).dt.normalize()
        self.fx = fx.sort_values(["starting_at", "fixture_id"]).reset_index(drop=True)
        self.mkt = {}
        if len(mk):
            cols = ["p_h", "p_d", "p_a", "ou_p_o", "btts_p_y", "lam_h", "lam_a"]
            mm = mk[cols]
            self.mkt = {int(k): v for k, v in zip(mm.index, mm.to_numpy(float))}
        self.ref = ref_map if ref_map is not None else (
            {int(r.fixture_id): int(r.referee_id) for r in d["referees"].itertuples()} if len(d["referees"]) else {})
        cap = {int(r.venue_id): float(r.capacity) for r in d["venues"].itertuples()} if len(d.get("venues", [])) else {}
        tv = {int(r.team_id): int(r.venue_id) for r in d["team_venues"].itertuples()} if len(d.get("team_venues", [])) else {}
        self.cap, self.team_venue = cap, tv
        self.rivals = {(int(r.team_id), int(r.rival_team_id)) for r in d["rivals"].itertuples()} if len(d.get("rivals", [])) else set()
        self.weather = {}
        for r in d["weather"].itertuples() if len(d.get("weather", [])) else []:
            ws = r.wind_speed * (1.609 if "imperial" in str(r.metric) else 1.0) if pd.notna(r.wind_speed) else np.nan
            self.weather[int(r.fixture_id)] = (ws, float(any(k in str(r.wdesc) for k in ("rain", "snow", "storm", "drizzle"))))
        self.team_sched = defaultdict(list)
        sch = d["schedule"].dropna(subset=["home_id", "away_id", "starting_at"])
        for ts, hh, aa in zip(sch.starting_at, sch.home_id, sch.away_id):
            for t in (int(hh), int(aa)):
                self.team_sched[t].append(int(ts.timestamp()))
        for t in self.team_sched:
            self.team_sched[t].sort()
        self.facts = build_facts(d, self.fx)
        self.reset()

    def reset(self):
        self.team = defaultdict(lambda: {m: EW() for m in TEAM_M})
        self.lg = defaultdict(lambda: {m: EW() for m in LG_M})
        self.refs = {}
        self.rt = defaultdict(lambda: {"pts": Acc(), "dec": Acc()})
        self.k = {key: 30.0 for key in TKEYS}
        self.tau = {key: np.nan for key in TKEYS}
        self.k_hist = []
        self.k_day = None
        self.elo = {}
        self.table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.league_teams = defaultdict(set)
        self.rls_cards, self.rls_fouls = RLS(5), RLS(5)
        self.rls_add1, self.rls_add2 = RLS(6), RLS(6)
        self.dvar = {k: EW() for k in ("hc", "hf", "hp")}
        self.exp = {}
        self.max_result_day = None
        self.ref_log = []

    # ---------- helpers ----------
    def lgv(self, lg, m, default=np.nan):
        e = self.lg[lg][m]
        if e.n >= 20:
            return e.get()
        g = self.lg["ALL"][m]
        return g.get(default) if g.n >= 5 else default

    def ratio(self, team, m, lg_val, K=6.0):
        e = self.team[team][m]
        if not np.isfinite(lg_val) or lg_val <= 0 or e.n == 0:
            return 1.0
        n = min(e.n, 1 / A_TEAM)
        return ((n * e.get() + K * lg_val) / (n + K)) / lg_val

    def ref_state(self, rf):
        if rf not in self.refs:
            self.refs[rf] = {"acc": {k: Acc() for k in TKEYS}, "n": 0, "n_lg": defaultdict(int), "first": None, "last": None, "recent": deque()}
        return self.refs[rf]

    def refresh_k(self, day):
        """Empirical-Bayes shrinkage strength per trait, from referees' past matches only."""
        if self.k_day is not None and (day - self.k_day).days < 14:
            return
        self.k_day = day
        rec = {"day": str(day.date())}
        for key in TKEYS:
            ms, s2, ws = [], [], []
            for st in self.refs.values():
                a = st["acc"][key]
                if a.w >= 8:
                    m = a.wx / a.w
                    ms.append(m); ws.append(a.w); s2.append(max(a.wxx / a.w - m * m, 1e-9))
            if len(ms) >= 10:
                ms, s2, ws = np.array(ms), np.array(s2), np.array(ws)
                tau2 = max(np.var(ms) - np.mean(s2 / ws), 1e-4 * np.mean(s2))
                self.k[key] = float(np.clip(np.mean(s2) / tau2, 3, 500))
                self.tau[key] = math.sqrt(tau2)
            rec[key] = self.k[key]
        self.k_hist.append(rec)

    def rest(self, team, ts):
        s = self.team_sched.get(team, [])
        i = bisect.bisect_left(s, ts)
        return min((ts - s[i - 1]) / 86400, 21) if i > 0 else np.nan

    def venue_cap(self, f):
        v = self.cap.get(int(f.venue_id)) if pd.notna(f.venue_id) else None
        return v if v is not None else self.cap.get(self.team_venue.get(int(f.home_id), -1))

    def cap_z(self, f, lg):
        v = self.venue_cap(f)
        if v is None:
            return np.nan
        e = self.lg[lg]["cap"]
        x = math.log1p(v)
        return (x - e.get(x)) / e.sd(1.0) if e.n >= 10 else np.nan

    # ---------- pre-match ----------
    def emit(self, f, ts, check=True):
        if check and self.max_result_day is not None:
            assert self.max_result_day < f.day, "look-ahead: state contains a result from this day or later"
        fid, h, a, lg = int(f.fixture_id), int(f.home_id), int(f.away_id), f.league_id
        R = dict(fixture_id=fid, day=f.day, league_id=lg, league=f.league, league_type=f.league_type, season_id=f.season_id,
                 season=f.season, home_id=h, away_id=a, home=f.home, away=f.away)
        m = self.mkt.get(fid)
        p_h, p_d, p_a, p_o, p_b, lam_h, lam_a = m if m is not None else (np.nan,) * 7
        L = {k: self.lgv(lg, k) for k in ("bp", "cards", "fouls", "reds", "pens", "shots", "goals", "btts", "var", "added1", "added2", "fcm")}
        if not np.isfinite(lam_h):           # no market: team goal rates
            g = L["goals"] if np.isfinite(L["goals"]) else 2.6
            lam_h = g / 2 * 1.12 * self.ratio(h, "gf", g / 2) * self.ratio(a, "ga", g / 2)
            lam_a = g / 2 * 0.88 * self.ratio(a, "gf", g / 2) * self.ratio(h, "ga", g / 2)
        p_b_x = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        if not np.isfinite(p_b):
            p_b = p_b_x
        derby = float((h, a) in self.rivals or (a, h) in self.rivals)
        close = 1 - abs(p_h - p_a) if np.isfinite(p_h) else np.nan
        cup = float(f.league_type != "league")
        # table & stakes (league competitions only, rebuilt from results)
        stakes, pos_h, pos_a, prog = 0.0, np.nan, np.nan, np.nan
        if f.league_type == "league":
            tab = self.table[f.season_id]
            n = max(len(self.league_teams[(lg, f.season_id)] | {h, a}), 2)
            order = sorted(tab, key=lambda t: -tab[t][0])
            pl = (tab[h][1] + tab[a][1]) / 2 if h in tab and a in tab else 0
            prog = pl / max(2 * (n - 1), 1)
            if order:
                pos_h = (order.index(h) if h in order else len(order)) / max(n - 1, 1)
                pos_a = (order.index(a) if a in order else len(order)) / max(n - 1, 1)
                stakes = float(prog > 0.7 and (min(pos_h, pos_a) < 0.2 or max(pos_h, pos_a) > 0.8))
        ctx = np.array([1.0, derby, nz(close, 0.7) - 0.7, cup, stakes])
        # expected discipline (team rates x league x fixture-context adjustment)
        mult_c = math.exp(self.rls_cards.predict(ctx))
        mult_f = math.exp(self.rls_fouls.predict(ctx))
        bp_h0 = L["bp"] * self.ratio(h, "bp_for", L["bp"]) * self.ratio(a, "bp_ag", L["bp"])
        bp_a0 = L["bp"] * self.ratio(a, "bp_for", L["bp"]) * self.ratio(h, "bp_ag", L["bp"])
        exp_bp_h, exp_bp_a = bp_h0 * mult_c, bp_a0 * mult_c
        cpb = L["cards"] / L["bp"] if np.isfinite(L["cards"]) and L["bp"] > 0 else 0.1
        f_h0 = L["fouls"] * self.ratio(h, "fouls_for", L["fouls"]) * self.ratio(a, "fouls_drawn", L["fouls"])
        f_a0 = L["fouls"] * self.ratio(a, "fouls_for", L["fouls"]) * self.ratio(h, "fouls_drawn", L["fouls"])
        exp_f_h, exp_f_a = f_h0 * mult_f, f_a0 * mult_f
        exp_r_h = L["reds"] * self.ratio(h, "reds_for", L["reds"], 40) * mult_c
        exp_r_a = L["reds"] * self.ratio(a, "reds_for", L["reds"], 40) * mult_c
        exp_p_h = L["pens"] * self.ratio(h, "pens_for", L["pens"], 30) * self.ratio(a, "pens_ag", L["pens"], 30)
        exp_p_a = L["pens"] * self.ratio(a, "pens_for", L["pens"], 30) * self.ratio(h, "pens_ag", L["pens"], 30)
        exp_s_h = L["shots"] * self.ratio(h, "shots_for", L["shots"]) * self.ratio(a, "shots_ag", L["shots"])
        exp_s_a = L["shots"] * self.ratio(a, "shots_for", L["shots"]) * self.ratio(h, "shots_ag", L["shots"])
        capz = self.cap_z(f, lg)
        dec_h, dec_a = self.team[h]["dec"].get(0.0), self.team[a]["dec"].get(0.0)
        self.exp[fid] = dict(bp_h=exp_bp_h, bp_a=exp_bp_a, bp0=bp_h0 + bp_a0, f_h=exp_f_h, f_a=exp_f_a, f0=f_h0 + f_a0,
                             r=exp_r_h + exp_r_a, p_h=exp_p_h, p_a=exp_p_a, s=exp_s_h + exp_s_a, lam_h=lam_h, lam_a=lam_a,
                             p_b=p_b, ph=p_h, pd=p_d, pa=p_a, ctx=ctx, capz=capz, dec_h=dec_h, dec_a=dec_a, L=L)
        w = self.weather.get(fid, (np.nan, np.nan))
        C = dict(c_lam_h=lam_h, c_lam_a=lam_a, c_lam_tot=lam_h + lam_a, c_p_btts_x=p_b_x, c_close=close, c_derby=derby, c_cup=cup,
                 c_stakes=stakes, c_pos_h=pos_h, c_pos_a=pos_a, c_progress=prog, c_cap_z=capz, c_wind=w[0], c_rain=w[1],
                 c_exp_bp=exp_bp_h + exp_bp_a, c_exp_cards=(exp_bp_h + exp_bp_a) * cpb, c_exp_card_diff=(exp_bp_a - exp_bp_h) * cpb,
                 c_exp_fouls=exp_f_h + exp_f_a, c_exp_reds=exp_r_h + exp_r_a, c_exp_pens=exp_p_h + exp_p_a, c_exp_shots=exp_s_h + exp_s_a,
                 c_ctx_card_mult=mult_c, c_rest_h=self.rest(h, ts), c_rest_a=self.rest(a, ts),
                 c_elo_diff=self.elo.get(h, np.nan) - self.elo.get(a, np.nan), c_dec_rep_diff=dec_h - dec_a)
        for k in ("bp", "fouls", "reds", "pens", "goals", "btts", "var", "added2"):
            C[f"c_lg_{k}"] = L[k]
        for side, t in (("h", h), ("a", a)):
            for mname in ("bp_for", "bp_ag", "fouls_for", "fouls_drawn", "reds_for", "coach", "pens_for", "pens_ag"):
                C[f"c_{side}_{mname}"] = self.team[t][mname].get()
        R.update(C)
        # ---- referee ----
        rf = self.ref.get(fid)
        R["ref_id"] = rf if rf is not None else -1
        if rf is None:
            for key in TKEYS:
                R[f"r_{key}"] = np.nan
            R.update(r_has=0.0, r_n=0.0, r_n_lg=0.0, r_days=np.nan, r_rest=np.nan, r_load30=np.nan,
                     rt_n_h=0.0, rt_pts_h=np.nan, rt_dec_h=np.nan, rt_n_a=0.0, rt_pts_a=np.nan, rt_dec_a=np.nan)
            return R
        st = self.refs.get(rf)
        if st is None:
            for key in TKEYS:
                R[f"r_{key}"] = 0.0
            R.update(r_has=1.0, r_n=0.0, r_n_lg=0.0, r_days=0.0, r_rest=np.nan, r_load30=0.0)
        else:
            for key in TKEYS:
                R[f"r_{key}"] = st["acc"][key].shrunk(self.k[key])
            rec = st["recent"]
            R.update(r_has=1.0, r_n=float(st["n"]), r_n_lg=float(st["n_lg"][lg]), r_days=float((f.day - st["first"]).days),
                     r_rest=float(min((f.day - st["last"]).days, 60)), r_load30=float(sum(1 for x in rec if (f.day - x).days <= 30)))
        for side, t in (("h", h), ("a", a)):
            q = self.rt.get((rf, t))
            R[f"rt_n_{side}"] = float(q["pts"].n) if q else 0.0
            R[f"rt_pts_{side}"] = q["pts"].shrunk(8.0) if q else 0.0
            R[f"rt_dec_{side}"] = q["dec"].shrunk(8.0) if q else 0.0
        return R

    # ---------- results ----------
    def update(self, f):
        fid, h, a, lg = int(f.fixture_id), int(f.home_id), int(f.away_id), f.league_id
        self.max_result_day = f.day if self.max_result_day is None else max(self.max_result_day, f.day)
        E = self.exp.pop(fid, None)
        M = self.facts.get(fid)
        if E is None or M is None or pd.isna(M["hg"]) or pd.isna(M["ag"]):
            return
        hg, ag = float(M["hg"]), float(M["ag"])
        L = E["L"]
        T = {}                                   # raw trait residuals for this match: key -> (value, weight)
        pts_h, pts_a = (3, 0) if hg > ag else ((1, 1) if hg == ag else (0, 3))
        tot_g = hg + ag
        btts = float(hg > 0 and ag > 0)
        T["goals"] = (tot_g - E["lam_h"] - E["lam_a"], 1.0)
        T["btts"] = (btts - E["p_b"], 1.0)
        if np.isfinite(E["ph"]):
            exp_h, exp_a = 3 * E["ph"] + E["pd"], 3 * E["pa"] + E["pd"]
            T["home_pts"] = (pts_h - exp_h, 1.0)
            T["fav_pts"] = ((pts_h - exp_h) if E["ph"] >= E["pa"] else (pts_a - exp_a), 1.0)
        has_ev = M.get("has_events", False)
        D = np.nan
        if has_ev:
            bp_h, bp_a = M["bp_h"], M["bp_a"]
            bp_tot = bp_h + bp_a
            lbp = 2 * L["bp"] if np.isfinite(L["bp"]) and L["bp"] > 0 else np.nan
            if np.isfinite(lbp):
                T["cards"] = ((bp_tot - E["bp_h"] - E["bp_a"]) / lbp, 1.0)
                T["home_cards"] = (((bp_a - bp_h) - (E["bp_a"] - E["bp_h"])) / lbp, 1.0)
            T["reds"] = (M["reds_h"] + M["reds_a"] - E["r"], 1.0) if np.isfinite(E["r"]) else None
            T["pens"] = (M["pen_h"] + M["pen_a"] - E["p_h"] - E["p_a"], 1.0) if np.isfinite(E["p_h"]) else None
            T["home_pens"] = ((M["pen_h"] - M["pen_a"]) - (E["p_h"] - E["p_a"]), 1.0) if np.isfinite(E["p_h"]) else None
            T["var"] = (float(M["var"]), 1.0)
            if M["booked"] > 0:
                T["second_yellow"] = ((M["yr_h"] + M["yr_a"]) / M["booked"], float(M["booked"]))
            if np.isfinite(M["first_card"]):
                T["early_card"] = (M["first_card"], 1.0)
            seq = M["seq"]
            if len(seq) >= 3:
                n_, k_ = len(seq), sum(seq)
                sw = sum(1 for i in range(1, n_) if seq[i] != seq[i - 1]) / (n_ - 1)
                T["alternation"] = (sw - 2 * k_ * (n_ - k_) / (n_ * (n_ - 1)), float(n_ - 1))
            if M["pen_nonlevel"] > 0:
                T["pen_trailer"] = (M["pen_trailer"] / M["pen_nonlevel"], float(M["pen_nonlevel"]))
            lm = M.get("lead_min", np.nan)
            if np.isfinite(lm) and lm <= 80 and M.get("leader"):
                rem = (90 - lm) / 90
                lam_t = E["lam_a"] if M["leader"] == "h" else E["lam_h"]
                T["after_lead"] = (M["aft_lg"] + M["aft_tg"] - (E["lam_h"] + E["lam_a"]) * rem, 1.0)
                T["trailer_scores"] = (float(M["aft_tg"] > 0) - (1 - math.exp(-lam_t * rem)), 1.0)
                T["lead_cards"] = (float(M["aft_lc"] - M["aft_tc"]), 1.0)
            # stoppage time beyond what the half's events explain
            for half, rls in ((1, self.rls_add1), (2, self.rls_add2)):
                v = M.get(f"added{half}", np.nan)
                lgm = L[f"added{half}"]
                if np.isfinite(v) and np.isfinite(lgm):
                    x = np.array([1.0, M[f"goals{half}"], M[f"cards{half}"], M[f"subs{half}"], M[f"var{half}"], M[f"pens{half}"]])
                    res = (v - lgm) - rls.predict(x)
                    T[f"added{half}"] = (res, 1.0)
                    rls.update(x, v - lgm)
                    if half == 2:
                        s90 = M.get("score90", np.nan)
                        if np.isfinite(s90) and abs(s90) == 1:
                            T["fergie"] = (-s90 * res, 1.0)
            fh, fa = M.get("fouls_h", np.nan), M.get("fouls_a", np.nan)
            if np.isfinite(fh) and np.isfinite(fa) and np.isfinite(L["fouls"]) and L["fouls"] > 0:
                T["fouls"] = ((fh + fa - E["f_h"] - E["f_a"]) / (2 * L["fouls"]), 1.0)
                T["home_fouls"] = (((fa - fh) - (E["f_a"] - E["f_h"])) / (2 * L["fouls"]), 1.0)
                if np.isfinite(lbp) and fh + fa > 0:
                    T["strict"] = ((bp_tot - (fh + fa) * L["bp"] / L["fouls"]) / lbp, 1.0)
            sh_, sa_ = M.get("shots_h", np.nan), M.get("shots_a", np.nan)
            if np.isfinite(sh_) and np.isfinite(sa_) and np.isfinite(L["shots"]) and L["shots"] > 0:
                T["shots"] = ((sh_ + sa_ - E["s"]) / (2 * L["shots"]), 1.0)
            # composite decision index (home-favouring), each part scaled by its running SD
            parts = []
            for key, dk in (("home_cards", "hc"), ("home_fouls", "hf"), ("home_pens", "hp")):
                if T.get(key) is not None:
                    v = T[key][0]
                    sd = self.dvar[dk].sd(np.nan)
                    if np.isfinite(sd) and sd > 0:
                        parts.append(v / sd)
                    self.dvar[dk].add(v, A_LG)
            if parts:
                D = sum(parts) / math.sqrt(len(parts))
                T["home_dec"] = (D, 1.0)
        # ---- league-relative centring, then referee accumulators ----
        rf = self.ref.get(fid)
        Dc = D - self.lgv(lg, "D", 0.0) if np.isfinite(D) else np.nan
        if np.isfinite(Dc):
            slopes = {"crowd_slope": E["capz"],
                      "fav_slope": (E["ph"] - E["pa"]) - self.lgv(lg, "phd", 0.0) if np.isfinite(E["ph"]) else np.nan,
                      "rep_slope": E["dec_h"] - E["dec_a"]}
            for key, x in slopes.items():
                if np.isfinite(x) and abs(x) >= 0.1:
                    T[key] = (Dc / x, x * x)
        if rf is not None:
            st = self.ref_state(rf)
            decay = 0.5 ** (1 / self.args.ref_half_life)
            for key, tv in T.items():
                if tv is None or not np.isfinite(tv[0]) or not np.isfinite(tv[1]):
                    continue
                raw, wt = tv
                base = 0.0 if key in SLOPES else self.lgv(lg, "T_" + key, 0.0)
                vc = raw - base
                st["acc"][key].add(vc, decay, wt)
                if self.keep_log:
                    self.ref_log.append((fid, f.day, rf, lg, f.season_id, key, raw, vc, wt))
            st["n"] += 1
            st["n_lg"][lg] += 1
            st["first"] = st["first"] or f.day
            st["last"] = f.day
            st["recent"].append(f.day)
            while st["recent"] and (f.day - st["recent"][0]).days > 30:
                st["recent"].popleft()
            if np.isfinite(E["ph"]):
                self.rt[(rf, h)]["pts"].add(pts_h - (3 * E["ph"] + E["pd"]), 0.97)
                self.rt[(rf, a)]["pts"].add(pts_a - (3 * E["pa"] + E["pd"]), 0.97)
            if np.isfinite(D):
                self.rt[(rf, h)]["dec"].add(D, 0.97)
                self.rt[(rf, a)]["dec"].add(-D, 0.97)
        # ---- league and team state ----
        for key, tv in T.items():
            if tv is not None and key not in SLOPES and np.isfinite(tv[0]):
                for g_ in (lg, "ALL"):
                    self.lg[g_]["T_" + key].add(tv[0], A_LG)
        add = lambda m, v: [self.lg[g_][m].add(v, A_LG) for g_ in (lg, "ALL")]  # noqa: E731
        add("goals", tot_g); add("btts", btts)
        if np.isfinite(D):
            add("D", D)
        if np.isfinite(E["ph"]):
            add("phd", E["ph"] - E["pa"])
        v = self.venue_cap(f)
        if v:
            self.lg[lg]["cap"].add(math.log1p(v), A_LG)
        if has_ev:
            for side in ("h", "a"):
                add("bp", M[f"bp_{side}"]); add("cards", M[f"cards_{side}"]); add("reds", M[f"reds_{side}"]); add("pens", M[f"pen_{side}"])
                if np.isfinite(M.get(f"fouls_{side}", np.nan)):
                    add("fouls", M[f"fouls_{side}"])
                if np.isfinite(M.get(f"shots_{side}", np.nan)):
                    add("shots", M[f"shots_{side}"])
            add("var", M["var"])
            for half in (1, 2):
                if np.isfinite(M.get(f"added{half}", np.nan)):
                    add(f"added{half}", M[f"added{half}"])
            if np.isfinite(M["first_card"]):
                add("fcm", M["first_card"])
            # fixture-context regressions for expected cards / fouls (fitted on past matches only)
            if E["bp0"] > 0:
                self.rls_cards.update(E["ctx"], math.log((M["bp_h"] + M["bp_a"] + 10) / (E["bp0"] + 10)))
            fh, fa = M.get("fouls_h", np.nan), M.get("fouls_a", np.nan)
            if np.isfinite(fh) and np.isfinite(fa) and E["f0"] > 0:
                self.rls_fouls.update(E["ctx"], math.log((fh + fa + 2) / (E["f0"] + 2)))
        for t, o, s, os_, gf, ga, sign in ((h, a, "h", "a", hg, ag, 1), (a, h, "a", "h", ag, hg, -1)):
            tm = self.team[t]
            tm["gf"].add(gf, A_TEAM); tm["ga"].add(ga, A_TEAM)
            if has_ev:
                tm["bp_for"].add(M[f"bp_{s}"], A_TEAM); tm["bp_ag"].add(M[f"bp_{os_}"], A_TEAM)
                tm["cards_for"].add(M[f"cards_{s}"], A_TEAM); tm["reds_for"].add(M[f"reds_{s}"], A_TEAM)
                tm["pens_for"].add(M[f"pen_{s}"], A_TEAM); tm["pens_ag"].add(M[f"pen_{os_}"], A_TEAM)
                tm["coach"].add(M[f"coach_{s}"], A_TEAM)
                tm["fouls_for"].add(M.get(f"fouls_{s}", np.nan), A_TEAM); tm["fouls_drawn"].add(M.get(f"fouls_{os_}", np.nan), A_TEAM)
                tm["shots_for"].add(M.get(f"shots_{s}", np.nan), A_TEAM); tm["shots_ag"].add(M.get(f"shots_{os_}", np.nan), A_TEAM)
            if np.isfinite(D):
                tm["dec"].add(sign * D, A_TEAM)
        # Elo + league table
        for t in (h, a):
            self.elo.setdefault(t, 1500.0)
        exp_h = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - 60) / 400))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        kk = 20 * (1 + math.log1p(abs(hg - ag)))
        self.elo[h] += kk * (s_h - exp_h)
        self.elo[a] -= kk * (s_h - exp_h)
        if f.league_type == "league":
            self.league_teams[(lg, f.season_id)].update((h, a))
            tab = self.table[f.season_id]
            tab[h][0] += pts_h; tab[h][1] += 1
            tab[a][0] += pts_a; tab[a][1] += 1

    def run(self, cutoff=None, only_day=None, upcoming: pd.DataFrame | None = None):
        """cutoff: pretend the world ends on this day (time-travel test): every result from the
        cutoff on is deleted and every later fixture is removed from the schedule."""
        self.reset()
        saved = None
        if cutoff is not None:
            saved = (self.facts, self.team_sched)
            end = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()
            self.team_sched = {t: [e for e in v if e < end] for t, v in self.team_sched.items()}
            late = set(self.fx.fixture_id[self.fx.day >= cutoff].astype(int))
            self.facts = {k: v for k, v in self.facts.items() if k not in late}
        out = []
        days = self.fx.groupby("day", sort=True)
        nd = len(days)
        for i, (day, g) in enumerate(days):
            if only_day is not None and day > only_day:
                break
            self.refresh_k(day)
            rows = [self.emit(f, int(f.starting_at.timestamp())) for f in g.itertuples()]
            if only_day is None or day == only_day:
                out.extend(rows)
            for f in g.itertuples():
                self.update(f)
            if i % 250 == 0 and only_day is None and cutoff is None:
                log(f"  features: day {i:,}/{nd:,} ({day.date()})")
        up = []
        if upcoming is not None and len(upcoming) and only_day is None:
            u = upcoming.dropna(subset=["home_id", "away_id", "starting_at"]).copy()
            u[["home_id", "away_id"]] = u[["home_id", "away_id"]].astype("int64")
            u["day"] = u.starting_at.dt.tz_convert(None).dt.normalize()
            last = self.fx.day.max()
            for f in u.sort_values("starting_at").itertuples():
                if f.day >= last:
                    up.append(self.emit(f, int(f.starting_at.timestamp()), check=False))
                    self.exp.pop(int(f.fixture_id), None)
        if saved:
            self.facts, self.team_sched = saved
        return pd.DataFrame(out), pd.DataFrame(up)


def leak_test(engine: Engine, full: pd.DataFrame, args) -> dict:
    days = sorted(full.day.unique())
    rng = np.random.default_rng(5)
    picks = sorted(rng.choice(days[len(days) // 3: -1], size=min(args.leak_tests, max(1, len(days) // 3)), replace=False))
    res = {"tests": [], "passed": True}
    snapshot = dict(engine.__dict__)          # reset() rebinds attributes, so this keeps the full run's state
    engine.keep_log = False
    for cut in (pd.Timestamp(c) for c in picks):
        trunc, _ = engine.run(cutoff=cut, only_day=cut)
        a = full[full.day == cut].set_index("fixture_id").sort_index()
        b = trunc.set_index("fixture_id").sort_index()
        cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
        diff = (a[cols] - b[cols]).abs()
        both_nan = a[cols].isna() & b[cols].isna()
        mism = ((diff > 1e-9) | (a[cols].isna() != b[cols].isna())) & ~both_nan
        n_bad = int(mism.to_numpy().sum())
        bad_cols = [c for c in cols if mism[c].any()][:8]
        res["tests"].append({"day": str(cut.date()), "matches": int(len(a)), "features": len(cols), "mismatches": n_bad, "columns": bad_cols})
        res["passed"] &= n_bad == 0
        log(f"  time-travel test {cut.date()}: {len(a)} matches x {len(cols)} features, mismatches = {n_bad}")
    engine.__dict__.update(snapshot)
    return res


def placebo_map(fx: pd.DataFrame, ref: dict, seed: int) -> dict:
    """Shuffle referee appointments between matches of the same league-season."""
    rng = np.random.default_rng(seed)
    out = {}
    f = fx[fx.fixture_id.astype(int).isin(ref.keys())]
    for _, g in f.groupby(["league_id", "season_id"]):
        ids = g.fixture_id.astype(int).to_numpy()
        refs = np.array([ref[i] for i in ids])
        out.update(zip(ids.tolist(), rng.permutation(refs).tolist()))
    return out


# =============================================================================
# Targets and feature sets
# =============================================================================
def market_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match columns derived from prices and expectations (shared by history and upcoming rows)."""
    eps = 1e-4
    ph, pd_, pa = (df[c].clip(eps, 1 - eps) for c in ("p_h", "p_d", "p_a"))
    df["mk_lr_h"], df["mk_lr_a"] = np.log(ph / pd_), np.log(pa / pd_)
    po = df.ou_p_o.clip(eps, 1 - eps)
    df["mk_lo"] = np.log(po / (1 - po))
    df["btts_anchor"] = df.btts_p_y.where(df.btts_p_y.notna(), df.c_p_btts_x)
    pb = df.btts_anchor.clip(eps, 1 - eps)
    df["mk_blo"] = np.log(pb / (1 - pb))
    df["c_mk_lo_x"] = np.log((1 - np.exp(-df.c_lam_tot) * (1 + df.c_lam_tot + df.c_lam_tot ** 2 / 2)).clip(eps, 1 - eps) /
                             (np.exp(-df.c_lam_tot) * (1 + df.c_lam_tot + df.c_lam_tot ** 2 / 2)).clip(eps, 1 - eps))
    if "cards_p_o" in df:
        df["cards_lo"] = np.log(df.cards_p_o.clip(eps, 1 - eps) / (1 - df.cards_p_o.clip(eps, 1 - eps)))
    df["c_log_exp_cards"] = np.log(df.c_exp_cards.clip(0.3, None))
    return df


def add_targets(df: pd.DataFrame, facts: dict, mk: pd.DataFrame) -> pd.DataFrame:
    df = market_cols(df.join(mk, on="fixture_id"))
    eps = 1e-4
    rows = []
    for fid in df.fixture_id.astype(int):
        M = facts.get(fid)
        if M is None:
            rows.append({})
            continue
        hg, ag, hh, ha = M["hg"], M["ag"], M["ht_h"], M["ht_a"]
        R = dict(hg=hg, ag=ag, y=0 if hg > ag else (1 if hg == ag else 2), y_over=float(hg + ag > 2.5), y_btts=float(hg > 0 and ag > 0),
                 y_added2=M.get("added2", np.nan))
        if M.get("has_events"):
            R.update(y_cards=float(M["cards_h"] + M["cards_a"]), y_card_diff=float(M["cards_a"] - M["cards_h"]),
                     y_red=float(M["reds_h"] + M["reds_a"] > 0), y_pen=float(M["pen_h"] + M["pen_a"] > 0))
        if pd.notna(hh) and pd.notna(ha) and abs(hh - ha) == 1:
            lead_home = hh > ha
            R.update(ht_sub=1.0, ht_home_leads=float(lead_home), ht_goals=float(hh + ha),
                     y_ht_trail=float((ag - ha) > 0 if lead_home else (hg - hh) > 0),
                     y_ht_shut=float(hg + ag == hh + ha))
        rows.append(R)
    T = pd.DataFrame(rows, index=df.index)
    for c in T.columns:
        df[c] = T[c]
    for c in ("y_cards", "y_card_diff", "y_red", "y_pen", "ht_sub", "ht_home_leads", "ht_goals", "y_ht_trail", "y_ht_shut"):
        if c not in df:
            df[c] = np.nan
    # in-play (half-time) expectations: known at half-time, never used by pre-match targets
    lt = np.where(df.ht_home_leads == 1, df.c_lam_a, df.c_lam_h)
    df["ht_lam_trail"] = lt * 0.55
    df["ht_lam_lead"] = np.where(df.ht_home_leads == 1, df.c_lam_h, df.c_lam_a) * 0.55
    df["ht_p_trail"] = 1 - np.exp(-df.ht_lam_trail)
    df["ht_p_shut"] = np.exp(-(df.ht_lam_trail + df.ht_lam_lead))
    df["ht_lo_trail"] = np.log(df.ht_p_trail.clip(eps, 1 - eps) / (1 - df.ht_p_trail.clip(eps, 1 - eps)))
    df["ht_lo_shut"] = np.log(df.ht_p_shut.clip(eps, 1 - eps) / (1 - df.ht_p_shut.clip(eps, 1 - eps)))
    df["month"] = df.day.dt.to_period("M")
    return df


REF_META = ["r_has", "r_n", "r_n_lg", "r_days", "r_rest", "r_load30", "rt_n_h", "rt_pts_h", "rt_dec_h", "rt_n_a", "rt_pts_a", "rt_dec_a"]
REF_COLS = [f"r_{k}" for k in TKEYS] + REF_META

# kind: bin | multi | count | gauss.  anchor: unpenalised market inputs.  pm: market probability columns
# for residual boosting.  extra: in-play columns allowed for that target only.  offset: log-expectation.
TARGETS = {
    "home":     dict(kind="multi", y="y", anchor=["mk_lr_h", "mk_lr_a"], pm=["p_h", "p_d", "p_a"], label="Match result (1X2)",
                     prices=["avg_h", "avg_d", "avg_a"], best=["best_h", "best_d", "best_a"], trait="r_home_dec"),
    "over25":   dict(kind="bin", y="y_over", anchor=["mk_lo"], pm=["ou_p_o"], label="Over 2.5 goals", prices=["ou_avg_o", "ou_avg_u"],
                     best=["ou_best_o", "ou_best_u"], trait="r_goals"),
    "btts":     dict(kind="bin", y="y_btts", anchor=["mk_blo"], pm=["btts_anchor"], label="Both teams score", prices=["btts_avg_y", "btts_avg_n"],
                     best=["btts_best_y", "btts_best_n"], trait="r_btts"),
    "cards":    dict(kind="count", y="y_cards", offset="c_log_exp_cards", label="Total cards", trait="r_cards"),
    "card_diff": dict(kind="gauss", y="y_card_diff", label="Away minus home cards", trait="r_home_cards"),
    "red":      dict(kind="bin", y="y_red", label="Sending-off in the match", trait="r_reds"),
    "pen":      dict(kind="bin", y="y_pen", label="Penalty awarded", trait="r_pens"),
    "added2":   dict(kind="gauss", y="y_added2", label="2nd-half stoppage time", trait="r_added2"),
    "ht_trail": dict(kind="bin", y="y_ht_trail", anchor=["ht_lo_trail"], sub="ht_sub", extra=["ht_home_leads", "ht_goals", "ht_lam_trail", "ht_lam_lead"],
                     label="HT 1-goal lead: trailing side scores in 2nd half", trait="r_trailer_scores"),
    "ht_shut":  dict(kind="bin", y="y_ht_shut", anchor=["ht_lo_shut"], sub="ht_sub", extra=["ht_home_leads", "ht_goals", "ht_lam_trail", "ht_lam_lead"],
                     label="HT 1-goal lead: no more goals", trait="r_after_lead"),
}


def context_cols(df):
    return [c for c in df.columns if c.startswith("c_") and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().mean() > 0.05]


def feature_lists(df, spec):
    ctx = context_cols(df) + [c for c in spec.get("extra", []) if c in df]
    ref = [c for c in REF_COLS if c in df]
    pl = ["pl_" + c for c in ref if "pl_" + c in df]
    anc = [c for c in spec.get("anchor", []) if c in df]
    return {"base": anc + ctx, "ref": anc + ctx + ref, "placebo": anc + ctx + pl}, anc


# =============================================================================
# Models: ridge GLMs (logit / softmax / Poisson / Gaussian) and gradient boosting
# =============================================================================
class Prep:
    """Median-impute, standardise, clip: parameters from the training rows only."""

    def __init__(self, X: pd.DataFrame):
        self.cols = list(X.columns)
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1).fillna(1)

    def __call__(self, X):
        Z = (X[self.cols].fillna(self.med) - self.mu) / self.sd
        return np.clip(Z.fillna(0).to_numpy(float), -5, 5)


def fit_mnl(X, y, K, lam, pen, w0=None):
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    Y = np.eye(K)[y]
    pm = np.r_[0.0, pen][:, None]

    def f(wf):
        W = wf.reshape(p + 1, K)
        Z = Xb @ W
        lse = logsumexp(Z, axis=1)
        P = np.exp(Z - lse[:, None])
        return (lse - (Z * Y).sum(1)).mean() + 0.5 * lam * (pm * W * W).sum(), (Xb.T @ (P - Y) / n + lam * pm * W).ravel()
    w0 = np.zeros((p + 1) * K) if w0 is None or w0.size != (p + 1) * K else w0
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7}).x


def mnl_predict(wf, X, K):
    W = wf.reshape(X.shape[1] + 1, K)
    Z = np.c_[np.ones(len(X)), X] @ W
    return np.exp(Z - logsumexp(Z, axis=1)[:, None])


def fit_pois(X, y, lam, pen, off, w0=None):
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    pm = np.r_[0.0, pen]

    def f(b):
        eta = np.clip(Xb @ b + off, -8, 8)
        mu = np.exp(eta)
        return (mu - y * eta).mean() + 0.5 * lam * (pm * b * b).sum(), Xb.T @ (mu - y) / n + lam * pm * b
    w0 = np.zeros(p + 1) if w0 is None or w0.size != p + 1 else w0
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 600}).x


def fit_ridge(X, y, lam, pen):
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]
    A = Xb.T @ Xb / n + lam * np.diag(np.r_[0.0, pen])
    return np.linalg.solve(A + 1e-9 * np.eye(p + 1), Xb.T @ y / n)


def loss_rows(kind, P, y):
    """Per-row loss: log loss (bin/multi), Poisson deviance (count), squared error (gauss)."""
    if kind == "multi":
        return -np.log(np.clip(P[np.arange(len(y)), y.astype(int)], 1e-9, 1))
    if kind == "bin":
        p = np.clip(P, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if kind == "count":
        mu = np.clip(P, 1e-6, None)
        return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
    return (y - P) ** 2


LAM_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def glm_fit_predict(kind, Xtr, ytr, Xte, pen, offtr=None, offte=None, lam=None, warm=None):
    """Returns (pred, lam, coefficients). lam=None -> tuned on the latest 20% of the training rows."""
    def fit(X, y, l_, w0=None, off=None):
        if kind == "multi":
            return fit_mnl(X, y.astype(int), 3, l_, pen, w0)
        if kind == "bin":
            return fit_mnl(X, y.astype(int), 2, l_, pen, w0)
        if kind == "count":
            return fit_pois(X, y, l_, pen, off, w0)
        return fit_ridge(X, y, l_, pen)

    def pred(w, X, off=None):
        if kind == "multi":
            return mnl_predict(w, X, 3)
        if kind == "bin":
            return mnl_predict(w, X, 2)[:, 1]
        if kind == "count":
            return np.exp(np.clip(np.c_[np.ones(len(X)), X] @ w + off, -8, 8))
        return np.c_[np.ones(len(X)), X] @ w
    if lam is None:
        cut = int(len(ytr) * 0.8)
        best, best_l = LAM_GRID[2], np.inf
        for l_ in LAM_GRID:
            w = fit(Xtr[:cut], ytr[:cut], l_, None, offtr[:cut] if offtr is not None else None)
            ls = loss_rows(kind, pred(w, Xtr[cut:], offtr[cut:] if offtr is not None else None), ytr[cut:]).mean()
            if ls < best_l:
                best, best_l = l_, ls
        lam = best
    w = fit(Xtr, ytr, lam, warm, offtr)
    return pred(w, Xte, offte), lam, w


def gbm_fit_predict(kind, Xtr, ytr, Xte, Mtr=None, Mte=None, s=None, seed=0):
    """Unanchored: HistGradientBoosting with the target's loss (early stopping on a slice of the
    TRAINING rows). Anchored: boost (outcome - market probability) and add a shrunk correction,
    the shrink chosen on the latest 20% of the training rows. Returns (pred, s, model)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    keep = np.isfinite(Xtr).any(axis=0)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    common = dict(learning_rate=0.04, max_depth=3, min_samples_leaf=150, l2_regularization=5.0, max_iter=400,
                  max_bins=127, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed)
    if Mtr is None:
        if kind == "bin":
            m = HistGradientBoostingClassifier(**common).fit(Xtr, ytr.astype(int))
            return m.predict_proba(Xte)[:, 1], None, (m, keep)
        if kind == "multi":
            m = HistGradientBoostingClassifier(**common).fit(Xtr, ytr.astype(int))
            return m.predict_proba(Xte), None, (m, keep)
        m = HistGradientBoostingRegressor(loss="poisson" if kind == "count" else "squared_error", **common).fit(Xtr, ytr)
        return m.predict(Xte), None, (m, keep)
    Ytr = np.eye(3)[ytr.astype(int)] if kind == "multi" else ytr[:, None]

    def boost(Xa, Ya, Ma, Xb):
        res, models = np.zeros((len(Xb), Ma.shape[1])), []
        for j in range(Ma.shape[1]):
            reg = HistGradientBoostingRegressor(learning_rate=0.03, max_depth=3, min_samples_leaf=200, l2_regularization=5.0, max_iter=150, max_bins=127,
                                                random_state=seed + j).fit(Xa, Ya[:, j] - Ma[:, j])
            res[:, j] = reg.predict(Xb)
            models.append(reg)
        return res, models

    def combine(M, G, s_):
        if M.shape[1] == 1:
            return np.clip(M[:, 0] + s_ * G[:, 0], 0.01, 0.99)
        P = np.clip(M + s_ * G, 0.005, 1)
        return P / P.sum(axis=1, keepdims=True)
    if s is None:
        cut = int(len(Xtr) * 0.8)
        Gv, _ = boost(Xtr[:cut], Ytr[:cut], Mtr[:cut], Xtr[cut:])
        best, best_l = 0.0, np.inf
        for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            ls = loss_rows(kind, combine(Mtr[cut:], Gv, s_), ytr[cut:]).mean()
            if ls < best_l - 1e-6:
                best, best_l = s_, ls
        s = best
    G, models = boost(Xtr, Ytr, Mtr, Xte)
    return combine(Mte, G, s), s, (models, keep, s)


def gbm_predict(kind, mdl, X, M=None):
    """Predict with a model returned by gbm_fit_predict (M = market probabilities if anchored)."""
    if M is None:
        m, keep = mdl
        Xk = X[:, keep]
        if kind in ("bin", "multi"):
            pr = m.predict_proba(Xk)
            return pr[:, 1] if kind == "bin" else pr
        return m.predict(Xk)
    models, keep, s_ = mdl
    G = np.column_stack([r.predict(X[:, keep]) for r in models])
    if M.shape[1] == 1:
        return np.clip(M[:, 0] + s_ * G[:, 0], 0.01, 0.99)
    Pn = np.clip(M + s_ * G, 0.005, 1)
    return Pn / Pn.sum(1, keepdims=True)


VARIANTS = ("base", "ref", "placebo")


def usable_cpus() -> int:
    n = os.cpu_count() or 1
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(q) // int(per)))
    except (OSError, ValueError):
        pass
    return max(1, n)


_W: dict = {}


def _init(df, up, args, threads):
    _W.update(df=df, up=up, args=args)
    try:
        from threadpoolctl import threadpool_limits
        _W["lim"] = threadpool_limits(limits=threads)
    except ImportError:
        os.environ["OMP_NUM_THREADS"] = str(threads)


def _train_target(name):
    """Walk-forward for one target: every variant x {glm, gbm}, retrained every --retrain-months,
    trained only on matches before the block. Also: final fit for upcoming fixtures and
    permutation importance (trained on discovery rows, scored on hold-out rows)."""
    df, up, args = _W["df"], _W["up"], _W["args"]
    spec = TARGETS[name]
    kind = spec["kind"]
    t0 = time.time()
    fl, anc = feature_lists(df, spec)
    has_pl = len(fl["placebo"]) > len(fl["base"])
    y_all = df[spec["y"]].to_numpy(float)
    ok_all = np.isfinite(y_all)
    if spec.get("sub"):
        ok_all &= (df[spec["sub"]] == 1).to_numpy()
    for c in anc + spec.get("pm", []):
        ok_all &= df[c].notna().to_numpy() if c in df else False
    if spec.get("offset"):
        ok_all &= df[spec["offset"]].notna().to_numpy()
    n = len(df)
    shape = (n, 3) if kind == "multi" else (n,)
    P = {f"{v}_{a}": np.full(shape, np.nan) for v in VARIANTS for a in ("glm", "gbm")}
    coefs, out_log = [], []
    have_gbm = not args.no_gbm
    try:
        import sklearn  # noqa: F401
    except ImportError:
        have_gbm = False
    months = sorted(df.month.unique())
    start = df.day.min() + pd.Timedelta(days=args.burn_in_days)
    eval_months = [m for m in months if m.start_time >= start]
    blocks = [eval_months[i: i + args.retrain_months] for i in range(0, len(eval_months), args.retrain_months)]
    min_train = args.min_train if not spec.get("sub") else max(300, args.min_train // 3)
    tuned, warm, gbm_models = {}, {}, {}
    for bi, block in enumerate(blocks):
        m0 = block[0].start_time
        train = ((df.day < m0) & (df.day >= m0 - pd.Timedelta(days=args.train_window_days))).to_numpy() & ok_all
        test = df.month.isin(block).to_numpy() & ok_all
        if train.sum() < min_train or test.sum() == 0:
            continue
        retune = bi % args.retune_every == 0
        tr, te = df[train], df[test]
        ytr = y_all[train]
        offtr = tr[spec["offset"]].to_numpy(float) if spec.get("offset") else None
        offte = te[spec["offset"]].to_numpy(float) if spec.get("offset") else None
        idx = np.where(test)[0]
        for v in VARIANTS:
            if v == "placebo" and not has_pl:
                continue
            cols = fl[v]
            prep = Prep(tr[cols])
            Xtr, Xte = prep(tr), prep(te)
            pen = np.array([0.0 if c in anc else 1.0 for c in cols])
            key = f"{v}_glm"
            pred, lam, w = glm_fit_predict(kind, Xtr, ytr, Xte, pen, offtr, offte, None if retune or key not in tuned else tuned[key], warm.get(key))
            tuned[key], warm[key] = lam, w
            P[key][idx] = pred
            if v == "ref":
                K = 3 if kind == "multi" else (2 if kind == "bin" else 1)
                W = w.reshape(len(cols) + 1, K)[1:] if kind in ("multi", "bin") else w[1:][:, None]
                eff = (W[:, 0] - W[:, 1]) if kind == "multi" else (W[:, 1] - W[:, 0] if kind == "bin" else W[:, 0])
                coefs.append((str(block[0]), {c: float(e) for c, e in zip(cols, eff) if c.startswith(("r_", "rt_"))}))
            if have_gbm and (v != "placebo" or args.placebo_gbm):
                # boosted models are refitted every --gbm-refit blocks; in between, the latest model
                # (trained on even earlier matches) predicts - still strictly out of sample
                key = f"{v}_gbm"
                Xte_r = te[cols].to_numpy(float)
                Mte = te[spec["pm"]].to_numpy(float) if spec.get("pm") else None
                if key not in gbm_models or bi - gbm_models[key][1] >= args.gbm_refit:
                    Xtr_r = tr[cols].to_numpy(float)
                    Mtr = tr[spec["pm"]].to_numpy(float) if spec.get("pm") else None
                    pred, s, mdl = gbm_fit_predict(kind, Xtr_r, ytr, Xte_r, Mtr, Mte, None if retune or key not in tuned else tuned[key], seed=bi)
                    if s is not None:
                        tuned[key] = s
                    gbm_models[key] = (mdl, bi)
                else:
                    pred = gbm_predict(kind, gbm_models[key][0], Xte_r, Mte)
                P[key][idx] = pred
        out_log.append((str(block[0]), int(train.sum()), int(test.sum())))
    res = {"name": name, "P": P, "coefs": coefs, "log": out_log, "has_pl": has_pl, "tuned": {k: v for k, v in tuned.items()}}
    # ---- permutation importance: fit on discovery rows, score on hold-out rows ----
    hold_from = df.holdout_from.iloc[0]
    disc = ok_all & (df.day < hold_from).to_numpy() & df.eval_window.to_numpy()
    hold = ok_all & (df.day >= hold_from).to_numpy()
    res["importance"] = None
    if have_gbm and disc.sum() >= min_train and hold.sum() >= 100:
        cols = fl["ref"]
        tr, te = df[disc], df[hold]
        Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        yte = y_all[hold]
        Mtr = tr[spec["pm"]].to_numpy(float) if spec.get("pm") else None
        Mte = te[spec["pm"]].to_numpy(float) if spec.get("pm") else None
        pred, s, mdl = gbm_fit_predict(kind, Xtr, y_all[disc], Xte, Mtr, Mte, seed=11)
        predict_with = lambda X: gbm_predict(kind, mdl, X, Mte)  # noqa: E731
        base_loss = loss_rows(kind, predict_with(Xte), yte).mean()
        rng = np.random.default_rng(7)
        imp = {}
        groups = defaultdict(list)
        for j, c in enumerate(cols):
            g = ("Referee traits" if c.startswith("r_") and c[2:] in TKEYS else "Referee experience / team history" if c.startswith(("r_", "rt_"))
                 else "Market" if c.startswith(("mk_", "c_lam", "c_p_btts", "c_close", "ht_")) else
                 "Team discipline" if c.startswith(("c_h_", "c_a_", "c_dec_rep", "c_elo")) else
                 "League level" if c.startswith("c_lg_") else "Expected counts" if c.startswith("c_exp") or c in ("c_ctx_card_mult", "c_log_exp_cards") else "Match context")
            groups[g].append(j)
            X2 = Xte.copy()
            X2[:, j] = X2[rng.permutation(len(X2)), j]
            imp[c] = float(loss_rows(kind, predict_with(X2), yte).mean() - base_loss)
        gimp = {}
        for g, js in groups.items():
            X2 = Xte.copy()
            perm = rng.permutation(len(X2))
            X2[:, js] = X2[perm][:, js]
            gimp[g] = float(loss_rows(kind, predict_with(X2), yte).mean() - base_loss)
        top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:25])
        res["importance"] = {"features": top, "groups": gimp, "base_loss": float(base_loss), "n_train": int(disc.sum()), "n_test": int(hold.sum())}
    # ---- final fit on the latest window -> upcoming fixtures ----
    res["upcoming"] = None
    if up is not None and len(up) and not spec.get("sub"):
        last = df.day.max() + pd.Timedelta(days=1)
        train = ((df.day < last) & (df.day >= last - pd.Timedelta(days=args.train_window_days))).to_numpy() & ok_all
        okup = np.ones(len(up), bool)
        for c in anc + spec.get("pm", []):
            okup &= up[c].notna().to_numpy() if c in up else False
        if spec.get("offset"):
            okup &= up[spec["offset"]].notna().to_numpy()
        if train.sum() >= min_train and okup.any():
            tr, te = df[train], up[okup]
            cols = fl["ref"]
            ytr = y_all[train]
            offtr = tr[spec["offset"]].to_numpy(float) if spec.get("offset") else None
            offte = te[spec["offset"]].to_numpy(float) if spec.get("offset") else None
            prep = Prep(tr[cols])
            pen = np.array([0.0 if c in anc else 1.0 for c in cols])
            pg, _, _ = glm_fit_predict(kind, prep(tr), ytr, prep(te), pen, offtr, offte, tuned.get("ref_glm"))
            prep_b = Prep(tr[fl["base"]])
            pb, _, _ = glm_fit_predict(kind, prep_b(tr), ytr, prep_b(te), np.array([0.0 if c in anc else 1.0 for c in fl["base"]]),
                                       offtr, offte, tuned.get("base_glm"))
            pgb = None
            if have_gbm:
                if spec.get("pm"):
                    pgb, _, _ = gbm_fit_predict(kind, tr[cols].to_numpy(float), ytr, te[cols].to_numpy(float),
                                                tr[spec["pm"]].to_numpy(float), te[spec["pm"]].to_numpy(float), tuned.get("ref_gbm"), seed=99)
                else:
                    pgb, _, _ = gbm_fit_predict(kind, tr[cols].to_numpy(float), ytr, te[cols].to_numpy(float), seed=99)
            res["upcoming"] = {"idx": np.where(okup)[0], "ref_glm": pg, "base_glm": pb, "ref_gbm": pgb}
    res["seconds"] = time.time() - t0
    return res


def walk_forward(df, up, args, names):
    import concurrent.futures as cf
    import multiprocessing as mp
    cpus = usable_cpus()
    workers = args.jobs or max(1, min(len(names), cpus - 1 if cpus > 2 else cpus))
    threads = max(1, cpus // workers)
    log(f"training {len(names)} targets on {workers} worker(s) x {threads} thread(s) ...")
    out = {}
    if workers <= 1 or not sys.platform.startswith("linux"):
        _init(df, up, args, threads)
        for nme in names:
            r = _train_target(nme)
            out[nme] = r
            log(f"  {nme:10s} done in {r['seconds']:.0f}s")
        return out
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(df, up, args, threads)) as ex:
        futs = {ex.submit(_train_target, nme): nme for nme in names}
        for f in cf.as_completed(futs):
            r = f.result()
            out[r["name"]] = r
            log(f"  {r['name']:10s} done in {r['seconds']:.0f}s")
    return out


# =============================================================================
# Evaluation
# =============================================================================
def boot_mean(x: np.ndarray, week: np.ndarray, B=1000, seed=3):
    """Mean with a block bootstrap by calendar week (matches on the same weekend are correlated)."""
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    g = pd.DataFrame({"x": x, "w": week}).groupby("w").x.agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1.0, max(p, 1 / B)))


TAUS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]


def probs_for_bet(name, spec, P):
    """Model probabilities over the priced selections (1X2: H/D/A; binary: yes/no)."""
    if spec["kind"] == "multi":
        return P
    return np.c_[P, 1 - P]


def evaluate(df, res, args):
    E = df.eval_window.to_numpy()
    hold_from = df.holdout_from.iloc[0]
    disc_m, hold_m = E & (df.day < hold_from).to_numpy(), E & (df.day >= hold_from).to_numpy()
    periods = {"discovery": disc_m, "holdout": hold_m, "all": E}
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    out = {}
    for name, r in res.items():
        spec = TARGETS[name]
        kind = spec["kind"]
        P = r["P"]
        y = df[spec["y"]].to_numpy(float)
        have = np.isfinite(y)
        for k in ("base_glm", "ref_glm"):
            have &= ~np.isnan(P[k]).reshape(len(df), -1).any(axis=1)
        algos = ["glm"] + (["gbm"] if not np.isnan(P["ref_gbm"]).all() else [])
        rec = {"label": spec["label"], "kind": kind, "algos": algos, "metrics": {}, "has_pl": r["has_pl"]}
        # market as a reference where the target has one
        mkt = None
        if spec.get("pm"):
            mkt = df[spec["pm"]].to_numpy(float)
            mkt = mkt if kind == "multi" else mkt[:, 0]
        for pn, pm in periods.items():
            ok = pm & have
            if ok.sum() < 30:
                continue
            m_ = {"n": int(ok.sum()), "y_mean": float(np.nanmean(y[ok]) if kind != "multi" else np.mean(y[ok] == 0))}
            if mkt is not None:
                m_["market_loss"] = float(loss_rows(kind, mkt[ok], y[ok]).mean())
            for a in algos:
                lb = loss_rows(kind, P[f"base_{a}"][ok], y[ok])
                lr = loss_rows(kind, P[f"ref_{a}"][ok], y[ok])
                lo, hi, pv = boot_mean(lb - lr, week[ok])
                d_ = {"base_loss": float(lb.mean()), "ref_loss": float(lr.mean()), "lift": float((lb - lr).mean()), "lo": lo, "hi": hi, "p": pv}
                if r["has_pl"] and not np.isnan(P[f"placebo_{a}"][ok]).any():
                    lp = loss_rows(kind, P[f"placebo_{a}"][ok], y[ok])
                    plo, phi, ppv = boot_mean(lb - lp, week[ok])
                    d_.update(pl_lift=float((lb - lp).mean()), pl_lo=plo, pl_hi=phi)
                    lo2, hi2, pv2 = boot_mean(lp - lr, week[ok])
                    d_.update(vs_pl=float((lp - lr).mean()), vs_pl_lo=lo2, vs_pl_hi=hi2, vs_pl_p=pv2)
                # relative scale: lift as % of the base loss
                d_["lift_pct"] = d_["lift"] / d_["base_loss"] if d_["base_loss"] else None
                m_[a] = d_
            rec["metrics"][pn] = m_
        # calibration of the referee model (hold-out, glm)
        okc = hold_m & have
        if kind == "bin" and okc.sum() > 100:
            pp, yy = P["ref_glm"][okc], y[okc]
            bins = np.clip((pp * 20).astype(int), 0, 19)
            cal = pd.DataFrame({"b": bins, "p": pp, "y": yy}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            rec["calibration"] = [[round(float(q.p), 4), round(float(q.y), 4), int(q.n)] for q in cal.itertuples() if q.n >= 20]
        # observed vs context-expected by quintile of the headline referee trait (hold-out)
        tcol = spec["trait"]
        okq = have & (df.r_n >= 10).to_numpy() & df[tcol].notna().to_numpy()
        dq, hq = okq & disc_m, okq & hold_m
        if dq.sum() > 200 and hq.sum() > 100:
            edges = np.quantile(df[tcol][dq], [0.2, 0.4, 0.6, 0.8])
            qi = np.searchsorted(edges, df[tcol].to_numpy(float), side="right")
            base = P["base_glm"]
            yv = (y == 0).astype(float) if kind == "multi" else y
            bv = base[:, 0] if kind == "multi" else base
            rows_q = []
            for q in range(5):
                mm = hq & (qi == q)
                if mm.sum() < 20:
                    rows_q.append({"q": q + 1, "n": int(mm.sum())})
                    continue
                dlt = yv[mm] - bv[mm]
                lo, hi, _ = boot_mean(dlt, week[mm], B=500)
                rows_q.append({"q": q + 1, "n": int(mm.sum()), "obs": float(yv[mm].mean()), "exp": float(bv[mm].mean()), "diff": float(dlt.mean()), "lo": lo, "hi": hi})
            rec["quintiles"] = {"trait": tcol, "rows": rows_q, "what": "home win" if kind == "multi" else spec["label"]}
        # betting where prices exist
        rec["bets"] = {}
        if spec.get("prices") and all(c in df for c in spec["prices"]) and df[spec["prices"][0]].notna().mean() > 0.2:
            yi = y.astype(int) if kind == "multi" else np.where(y == 1, 0, 1)
            for a in algos:
                for pr_name, pcols in (("avg", spec["prices"]), ("best", spec.get("best", spec["prices"]))):
                    if not all(c in df for c in pcols):
                        continue
                    prices = df[pcols].to_numpy(float)
                    Pm = probs_for_bet(name, spec, P[f"ref_{a}"])
                    okp = have & np.isfinite(prices).all(axis=1) & np.isfinite(Pm).all(axis=1)
                    edge = np.where(okp[:, None] & (prices <= args.max_odds), Pm * prices - 1, -np.inf)
                    sel = edge.argmax(1)
                    e = edge[np.arange(len(sel)), sel]
                    price = prices[np.arange(len(sel)), sel]
                    pl = np.where(sel == yi, price - 1, -1.0)
                    curve = {pn: [{"tau": t, "n": int((pm & (e >= t)).sum()), "roi": float(pl[pm & (e >= t)].mean()) if (pm & (e >= t)).sum() else None}
                                  for t in TAUS] for pn, pm in periods.items()}
                    cands = [c for c in curve["discovery"] if c["n"] >= args.min_bets and c["roi"] is not None]
                    tau = max(cands, key=lambda c: c["roi"])["tau"] if cands else 0.05
                    rb = {"tau": tau, "curve": curve, "periods": {}}
                    for pn, pm in periods.items():
                        b = pm & (e >= tau)
                        lo, hi, pv = boot_mean(pl[b], week[b])
                        rb["periods"][pn] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                                             "avg_price": float(price[b].mean()) if b.sum() else None}
                    bm = E & (e >= tau)
                    order = np.argsort(df.day.to_numpy()[bm], kind="stable")
                    cum = np.cumsum(pl[bm][order])
                    dts = df.day.dt.strftime("%Y-%m-%d").to_numpy()[bm][order]
                    stp = max(1, len(cum) // 300)
                    rb["cum"] = [[dts[i], round(float(cum[i]), 2)] for i in list(range(0, len(cum), stp)) + ([len(cum) - 1] if len(cum) else [])]
                    rec["bets"][f"{a}_{pr_name}"] = rb
        # card lines
        if name == "cards" and "cards_line" in df and df.cards_line.notna().mean() > 0.2:
            rec["bets"].update(card_line_bets(df, P, y, have, periods, week, args))
        # referee coefficients (glm, averaged over refits)
        if r["coefs"]:
            keys = sorted({k for _, c in r["coefs"] for k in c})
            M = np.array([[c.get(k, np.nan) for k in keys] for _, c in r["coefs"]])
            rec["coefs"] = sorted([{"f": k, "mean": float(np.nanmean(M[:, j])), "sd": float(np.nanstd(M[:, j])),
                                    "same_sign": float(np.nanmean(np.sign(M[:, j]) == np.sign(np.nanmean(M[:, j]))))}
                                   for j, k in enumerate(keys)], key=lambda x: -abs(x["mean"]))[:30]
        rec["importance"] = r["importance"]
        out[name] = rec
    return out


def card_line_bets(df, P, y, have, periods, week, args):
    """Over/under a bookmaker card line from the count model: negative binomial with the
    dispersion estimated on discovery rows."""
    out = {}
    disc = periods["discovery"]
    for a in ("glm", "gbm"):
        mu = P[f"ref_{a}"]
        if np.isnan(mu).all():
            continue
        okd = disc & have & np.isfinite(mu)
        m_, v_ = mu[okd], (y[okd] - mu[okd]) ** 2
        alpha = max(1e-3, float((v_.mean() - m_.mean()) / (m_ ** 2).mean()))
        r_ = 1 / alpha
        line = df.cards_line.to_numpy(float)
        prices = df[["cards_avg_o", "cards_avg_u"]].to_numpy(float)
        ok = have & np.isfinite(mu) & np.isfinite(line) & np.isfinite(prices).all(1)
        p_over = np.full(len(df), np.nan)
        p_over[ok] = 1 - nbinom.cdf(np.floor(line[ok]), r_, r_ / (r_ + mu[ok]))
        Pm = np.c_[p_over, 1 - p_over]
        yi = np.where(y > line, 0, np.where(y < line, 1, -1))
        edge = np.where(ok[:, None] & (prices <= args.max_odds), Pm * prices - 1, -np.inf)
        sel = edge.argmax(1)
        e = edge[np.arange(len(sel)), sel]
        price = prices[np.arange(len(sel)), sel]
        pl = np.where(yi == -1, 0.0, np.where(sel == yi, price - 1, -1.0))
        curve = {pn: [{"tau": t, "n": int((pm & (e >= t)).sum()), "roi": float(pl[pm & (e >= t)].mean()) if (pm & (e >= t)).sum() else None} for t in TAUS]
                 for pn, pm in periods.items()}
        cands = [c for c in curve["discovery"] if c["n"] >= args.min_bets and c["roi"] is not None]
        tau = max(cands, key=lambda c: c["roi"])["tau"] if cands else 0.05
        rb = {"tau": tau, "curve": curve, "periods": {}, "alpha": alpha}
        for pn, pm in periods.items():
            b = pm & (e >= tau)
            lo, hi, pv = boot_mean(pl[b], week[b])
            rb["periods"][pn] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                                 "avg_price": float(price[b].mean()) if b.sum() else None}
        out[f"{a}_cardline"] = rb
    return out


def angles(df, args):
    """Pre-declared referee angles. Cut-offs are quantiles of the trait over DISCOVERY rows only
    (referees with 15+ previous matches), then applied unchanged to the hold-out."""
    E = df.eval_window.to_numpy()
    hold_from = df.holdout_from.iloc[0]
    hold = (df.day >= hold_from).to_numpy()
    disc = E & ~hold
    week = df.day.dt.isocalendar().week.astype(int).to_numpy() + 100 * df.day.dt.year.to_numpy()
    g = lambda c: df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)  # noqa: E731
    seasoned = g("r_n") >= 15

    def q(c, p):
        v = g(c)[disc & seasoned]
        v = v[np.isfinite(v)]
        return float(np.quantile(v, p)) if len(v) > 50 else np.nan
    y, yo, yb = g("y"), g("y_over"), g("y_btts")
    rules = []
    rules.append(("balancer_btts", "Balancing referee: BTTS yes", "Referee's BTTS residual in the top 20% and trailing sides score after going behind more than with the median referee.",
                  "btts_y", seasoned & (g("r_btts") >= q("r_btts", .8)) & (g("r_trailer_scores") > q("r_trailer_scores", .5))))
    rules.append(("slow_under", "Game-killing referee: under 2.5", "Goals after the first goal in the bottom 20% and goals-vs-market below median: the referee lets leads be protected.",
                  "under", seasoned & (g("r_after_lead") <= q("r_after_lead", .2)) & (g("r_goals") < q("r_goals", .5))))
    rules.append(("open_over", "Flowing referee: over 2.5", "Goals-vs-market in the top 20% and shot volume above median.",
                  "over", seasoned & (g("r_goals") >= q("r_goals", .8)) & (g("r_shots") > q("r_shots", .5))))
    rules.append(("strict_no_btts", "Strict referee, tight game: BTTS no", "Cards-per-foul in the top 20% and the market sees a close game (|p_home − p_away| < 0.15).",
                  "btts_n", seasoned & (g("r_strict") >= q("r_strict", .8)) & (np.abs(g("p_h") - g("p_a")) < 0.15)))
    hd_hi, hd_lo = q("r_home_dec", .8), q("r_home_dec", .2)
    rules.append(("home_decisions", "Home-friendly decisions: back home", "Home decision index (cards, fouls, penalties) in the top 20%.",
                  "home", seasoned & (g("r_home_dec") >= hd_hi)))
    rules.append(("away_decisions", "Away-friendly decisions: back away", "Home decision index in the bottom 20%.",
                  "away", seasoned & (g("r_home_dec") <= hd_lo)))
    rules.append(("big_team", "Big-team referee: back a short favourite", "Big-team slope in the top 20% and a favourite priced 2.00 or shorter.",
                  "fav", seasoned & (g("r_fav_slope") >= q("r_fav_slope", .8)) & ((g("avg_h") <= 2.0) | (g("avg_a") <= 2.0))))
    rules.append(("fergie_home", "Stoppage-time referee + home favourite", "Stoppage-time-when-home-needs-it in the top 20% and home is favourite (p ≥ 0.45).",
                  "home", seasoned & (g("r_fergie") >= q("r_fergie", .8)) & (g("p_h") >= 0.45)))
    rules.append(("crowd_home", "Crowd-sensitive referee at a big stadium", "Crowd-size slope in the top 20% and stadium capacity 1+ SD above league average.",
                  "home", seasoned & (g("r_crowd_slope") >= q("r_crowd_slope", .8)) & (g("c_cap_z") >= 1)))
    if "cards_avg_o" in df:
        rules.append(("cards_over", "Strict referee: cards over the line", "Card-volume trait in the top 20%.", "cards_o", seasoned & (g("r_cards") >= q("r_cards", .8))))
        rules.append(("cards_under", "Lenient referee: cards under the line", "Card-volume trait in the bottom 20%.", "cards_u", seasoned & (g("r_cards") <= q("r_cards", .2))))
    out = []
    for key, title, why, sel, cond in rules:
        cond = np.nan_to_num(cond).astype(bool)
        rec = {"key": key, "title": title, "why": why, "periods": {}}
        for pr in ("avg", "best"):
            if sel in ("btts_y", "btts_n"):
                price, win, imp = g(f"btts_{pr}_{sel[-1]}"), (yb == 1) if sel == "btts_y" else (yb == 0), g("btts_p_y") if sel == "btts_y" else 1 - g("btts_p_y")
            elif sel in ("over", "under"):
                price, win, imp = g(f"ou_{pr}_{sel[0]}"), (yo == 1) if sel == "over" else (yo == 0), g("ou_p_o") if sel == "over" else 1 - g("ou_p_o")
            elif sel == "home":
                price, win, imp = g(f"{pr}_h"), y == 0, g("p_h")
            elif sel == "away":
                price, win, imp = g(f"{pr}_a"), y == 2, g("p_a")
            elif sel == "fav":
                hf = g("p_h") >= g("p_a")
                price, win, imp = np.where(hf, g(f"{pr}_h"), g(f"{pr}_a")), np.where(hf, y == 0, y == 2), np.where(hf, g("p_h"), g("p_a"))
            else:
                over = sel == "cards_o"
                yc, ln = g("y_cards"), g("cards_line")
                price = g(f"cards_{pr}_{'o' if over else 'u'}")
                win = (yc > ln) if over else (yc < ln)
                imp = g("cards_p_o") if over else 1 - g("cards_p_o")
            mask = cond & E & np.isfinite(price) & (price <= args.max_odds)
            pl = np.where(win, price - 1, -1.0)
            for pn, pm in (("discovery", ~hold), ("holdout", hold), ("all", np.ones(len(df), bool))):
                b = mask & pm
                lo, hi, pv = boot_mean(pl[b], week[b])
                rec["periods"].setdefault(pn, {})[pr] = {"n": int(b.sum()), "roi": float(pl[b].mean()) if b.sum() else None, "lo": lo, "hi": hi, "p": pv,
                                                         "win": float(np.mean(win[b])) if b.sum() else None, "implied": float(np.nanmean(imp[b])) if b.sum() else None}
        out.append(rec)
    return out


# =============================================================================
# Descriptive trait analysis: do referees really differ? (whole sample - not used by models)
# =============================================================================
def trait_analysis(log_rows, args):
    if not log_rows:
        return {}
    L = pd.DataFrame(log_rows, columns=["fixture_id", "day", "ref", "league", "season", "key", "raw", "vc", "w"])
    rng = np.random.default_rng(args.seed)
    out = {}
    for key, g in L.groupby("key"):
        g = g.sort_values("day")
        refs = g.ref.to_numpy()
        codes, uniq = pd.factorize(refs)
        w, x, raw = g.w.to_numpy(float), g.vc.to_numpy(float), g.raw.to_numpy(float)
        sw = np.bincount(codes, w)
        nref = np.bincount(codes)
        big = nref >= args.min_ref_matches

        def ref_sd(xv, c=codes):
            m = np.bincount(c, w * xv, len(uniq)) / np.maximum(np.bincount(c, w, len(uniq)), 1e-9)
            return float(np.sqrt(np.average((m[big] - np.average(m[big], weights=sw[big])) ** 2, weights=sw[big]))) if big.sum() > 2 else np.nan
        if big.sum() < 5:
            continue
        m = np.bincount(codes, w * x) / np.maximum(sw, 1e-9)
        s2 = np.bincount(codes, w * x * x) / np.maximum(sw, 1e-9) - m ** 2
        between = np.average((m[big] - np.average(m[big], weights=sw[big])) ** 2, weights=sw[big])
        noise = np.average(np.maximum(s2[big], 0) / sw[big], weights=sw[big])
        tau = math.sqrt(max(between - noise, 0.0))
        sd_obs = ref_sd(x)
        # permutation: shuffle referee labels within league-season
        grp = pd.factorize(g.league.astype(str) + "_" + g.season.astype(str))[0]
        perm_sd = []
        order = np.lexsort((rng.random(len(codes)), grp))
        for _ in range(args.perm):
            o2 = np.lexsort((rng.random(len(codes)), grp))
            c2 = np.empty_like(codes)
            c2[o2] = codes[order]
            perm_sd.append(ref_sd(x, c2))
        perm_sd = np.array(perm_sd)
        pval = float((np.sum(perm_sd >= sd_obs) + 1) / (len(perm_sd) + 1))
        # split-half reliability (alternate matches of each referee), Spearman-Brown corrected
        rank = g.groupby("ref").cumcount().to_numpy()
        odd = rank % 2 == 1
        mo = np.bincount(codes[odd], (w * x)[odd], len(uniq)) / np.maximum(np.bincount(codes[odd], w[odd], len(uniq)), 1e-9)
        me = np.bincount(codes[~odd], (w * x)[~odd], len(uniq)) / np.maximum(np.bincount(codes[~odd], w[~odd], len(uniq)), 1e-9)
        r = float(np.corrcoef(mo[big], me[big])[0, 1]) if big.sum() > 5 else np.nan
        rel = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
        # population effect: weighted mean of the raw residual with a referee-cluster bootstrap
        mraw = np.bincount(codes, w * raw) / np.maximum(sw, 1e-9)
        bs = []
        for _ in range(300):
            pick = rng.integers(0, len(uniq), len(uniq))
            ww = sw[pick]
            bs.append(np.sum(mraw[pick] * ww) / ww.sum())
        pop = float(np.sum(w * raw) / w.sum())
        # spread between referees in natural units: P90 - P10 of shrunk means
        k_ = noise / max(tau ** 2, 1e-12) if tau > 0 else np.inf
        shr = (m * sw / (sw + (k_ if np.isfinite(k_) else 1e12)))[big]
        out[key] = {"n_matches": int(len(g)), "n_refs": int(big.sum()), "pop_mean": pop, "pop_lo": float(np.percentile(bs, 2.5)), "pop_hi": float(np.percentile(bs, 97.5)),
                    "tau": tau, "sd_obs": sd_obs, "sd_perm": float(np.mean(perm_sd)), "p": pval, "reliability": rel,
                    "p10": float(np.percentile(shr, 10)), "p90": float(np.percentile(shr, 90)),
                    "share_real": float(max(between - noise, 0) / between) if between > 0 else 0.0}
    return out


def referee_table(eng: Engine, names: pd.DataFrame, args):
    names = names.dropna(subset=["referee_id"]) if len(names) else names
    nm = dict(zip(names.referee_id.astype(int), names.name.fillna(""))) if len(names) else {}
    lgname = dict(zip(eng.fx.league_id, eng.fx.league))
    rows = []
    for rf, st in eng.refs.items():
        if st["n"] < args.min_ref_matches:
            continue
        main_lg = max(st["n_lg"].items(), key=lambda kv: kv[1])[0] if st["n_lg"] else None
        r = {"id": int(rf), "name": nm.get(int(rf), str(rf)), "n": st["n"], "league": lgname.get(main_lg, ""), "last": str(st["last"].date()) if st["last"] is not None else ""}
        for k in TKEYS:
            a = st["acc"][k]
            r[k] = a.shrunk(eng.k[k]) if a.n else None
            r[k + "_n"] = a.n
        rows.append(r)
    # percentile of each trait among listed referees
    for k in TKEYS:
        v = np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
        okv = np.isfinite(v)
        ranks = np.full(len(v), np.nan)
        if okv.sum() > 1:
            ranks[okv] = pd.Series(v[okv]).rank(pct=True).to_numpy()
        for r, p_ in zip(rows, ranks):
            r[k + "_pct"] = None if not np.isfinite(p_) else float(p_)
    return rows


# =============================================================================
# Payload + dashboard
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
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def upcoming_payload(up, res, ref_rows):
    if up is None or not len(up):
        return []
    prof = {r["id"]: r for r in ref_rows}
    rows = []
    for i, f in enumerate(up.itertuples()):
        r = {"date": str(f.day.date()), "league": f.league, "home": f.home, "away": f.away, "ref_id": int(f.ref_id), "ref_n": f.r_n,
             "ref": prof.get(int(f.ref_id), {}).get("name", "" if f.ref_id < 0 else str(f.ref_id)),
             "traits": {k: getattr(f, f"r_{k}") for k in TKEYS}, "pred": {}}
        for name, rr in res.items():
            u = rr.get("upcoming")
            if not u:
                continue
            pos = np.where(u["idx"] == i)[0]
            if not len(pos):
                continue
            j = pos[0]
            g = lambda arr: None if arr is None else (arr[j].tolist() if np.ndim(arr) > 1 else float(arr[j]))  # noqa: E731
            r["pred"][name] = {"ref": g(u["ref_glm"]), "base": g(u["base_glm"]), "gbm": g(u["ref_gbm"])}
        mk = {}
        for c in ("p_h", "p_d", "p_a", "ou_p_o", "btts_p_y", "avg_h", "avg_d", "avg_a", "ou_avg_o", "ou_avg_u", "btts_avg_y", "btts_avg_n", "c_exp_cards"):
            if hasattr(f, c):
                mk[c] = getattr(f, c)
        r["market"] = mk
        rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--out", default="referee_impact_dashboard.html")
    ap.add_argument("--export-csv")
    ap.add_argument("--title", default="Referee impact on match outcomes")
    ap.add_argument("--leagues", help="comma-separated league_ids to evaluate (history always uses everything)")
    ap.add_argument("--targets", help="comma-separated subset of: " + ",".join(TARGETS))
    ap.add_argument("--holdout-from")
    ap.add_argument("--burn-in-days", type=int, default=365)
    ap.add_argument("--train-window-days", type=int, default=1460)
    ap.add_argument("--retrain-months", type=int, default=2)
    ap.add_argument("--retune-every", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=1500)
    ap.add_argument("--ref-half-life", type=float, default=60, help="referee memory, in that referee's matches")
    ap.add_argument("--min-ref-matches", type=int, default=20, help="minimum matches for the referee table / trait analysis")
    ap.add_argument("--market-1x2", type=int, default=1)
    ap.add_argument("--market-ou", type=int, default=80)
    ap.add_argument("--market-btts", type=int, default=14, help="Sportmonks 'Both Teams To Score' market id (0 = off: BTTS anchored on goal expectation)")
    ap.add_argument("--market-cards", type=int, default=0, help="Sportmonks card over/under market id (0 = off)")
    ap.add_argument("--max-odds", type=float, default=10.0)
    ap.add_argument("--min-bets", type=int, default=150)
    ap.add_argument("--leak-tests", type=int, default=3)
    ap.add_argument("--skip-leak-test", action="store_true")
    ap.add_argument("--no-placebo", action="store_true")
    ap.add_argument("--perm", type=int, default=200, help="permutations for the trait analysis")
    ap.add_argument("--alarm-gain", type=float, default=0.006)
    ap.add_argument("--upcoming-days", type=int, default=10)
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--gbm-refit", type=int, default=2, help="refit boosted models every N retrain blocks (in between the latest one predicts)")
    ap.add_argument("--placebo-gbm", action="store_true", help="also run the placebo through gradient boosting (slower)")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--db-jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cache")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    db = DB(args.dsn)
    args.db_name = db.name
    d = extract(db, args)
    mk = market_table(d)
    log(f"market table: {mk.p_h.notna().sum():,} fixtures with 1X2, {mk.ou_p_o.notna().sum():,} O/U, {mk.btts_p_y.notna().sum():,} BTTS prices")
    log("building features day by day ...")
    eng = Engine(d, mk, args)
    feats, up = eng.run(upcoming=d.get("upcoming"))
    log(f"feature rows: {len(feats):,} x {feats.shape[1]} · upcoming fixtures: {len(up)}")
    leak = {"skipped": True}
    if not args.skip_leak_test:
        log("time-travel leakage test ...")
        leak = leak_test(eng, feats, args)
        leak["skipped"] = False
    if not args.no_placebo:
        log("placebo run (referees shuffled within league-season) ...")
        peng = Engine(d, mk, args, ref_map=placebo_map(eng.fx, eng.ref, args.seed), keep_log=False)
        pfe, _ = peng.run()
        pcols = [c for c in REF_COLS if c in pfe]
        feats = feats.merge(pfe[["fixture_id"] + pcols].rename(columns={c: "pl_" + c for c in pcols}), on="fixture_id", how="left")
        del peng, pfe
    df = add_targets(feats, eng.facts, mk)
    df = df.sort_values(["day", "fixture_id"]).reset_index(drop=True)
    if len(up):
        up = market_cols(up.join(mk, on="fixture_id"))
        for c in df.columns:
            if c.startswith("pl_") and c not in up:
                up[c] = up[c[3:]]       # placebo columns are not used for upcoming predictions
    start = df.day.min() + pd.Timedelta(days=args.burn_in_days)
    ew = (df.day >= start).to_numpy().copy()
    if args.leagues:
        ew &= df.league_id.isin([int(x) for x in args.leagues.split(",")]).to_numpy()
    df["eval_window"] = ew
    if ew.sum() == 0:
        sys.exit("No rows after the burn-in period - not enough history.")
    hf = pd.Timestamp(args.holdout_from) if args.holdout_from else df.day[ew].quantile(0.6).normalize()
    df["holdout_from"] = hf
    names = [t for t in (args.targets.split(",") if args.targets else TARGETS) if t in TARGETS]
    log(f"walk-forward: {ew.sum():,} matches in the evaluation window; discovery < {hf.date()} <= hold-out")
    res = walk_forward(df, up if len(up) else None, args, names)
    log("scoring ...")
    ev = evaluate(df, res, args)
    alarms = []
    for nme, r in ev.items():
        mt = r["metrics"].get("holdout", {})
        for a in r["algos"]:
            if mt.get("market_loss") is not None and a in mt and mt["market_loss"] - mt[a]["ref_loss"] > args.alarm_gain and r["kind"] in ("bin", "multi"):
                alarms.append({"target": nme, "model": a, "gain": mt["market_loss"] - mt[a]["ref_loss"]})
                log(f"!!! WARNING: {nme}/{a} beats the closing market by {(mt['market_loss'] - mt[a]['ref_loss']) * 1000:.1f}e-3 log loss in the hold-out - check for leakage")
    leak["alarms"] = alarms
    ang = angles(df, args)
    log("trait analysis (permutation tests, split-half reliability) ...")
    ta = trait_analysis(eng.ref_log, args)
    refs = referee_table(eng, d.get("ref_names", pd.DataFrame()), args)
    upr = upcoming_payload(up, res, refs)
    E = df.eval_window.to_numpy()
    cov = {"fixtures": int(len(df)), "eval": int(E.sum()), "leagues": int(df[E].league_id.nunique()), "from": str(df.day[E].min().date()),
           "to": str(df.day[E].max().date()), "holdout_from": str(hf.date()), "n_discovery": int((E & (df.day < hf).to_numpy()).sum()),
           "n_holdout": int((E & (df.day >= hf).to_numpy()).sum()), "with_ref": float((df.ref_id >= 0)[E].mean()),
           "with_events": float(df.y_cards.notna()[E].mean()), "with_1x2": float(df.p_h.notna()[E].mean()), "with_ou": float(df.ou_p_o.notna()[E].mean()),
           "with_btts": float(df.btts_p_y.notna()[E].mean()), "with_cards_line": float(df.cards_line.notna()[E].mean()) if "cards_line" in df else 0.0,
           "with_added": float(df.y_added2.notna()[E].mean()), "with_fouls": float(df.c_exp_fouls.notna()[E].mean()),
           "n_refs": int(len(eng.refs)), "n_refs_listed": len(refs), "placebo": not args.no_placebo, "runtime_s": round(time.time() - T0)}
    if args.export_csv:
        out = df.drop(columns=["month"]).copy()
        for nme, r in res.items():
            for k, arr in r["P"].items():
                if arr.ndim == 2:
                    for i, s in enumerate("hda"):
                        out[f"pred_{nme}_{k}_{s}"] = arr[:, i]
                else:
                    out[f"pred_{nme}_{k}"] = arr
        out.to_csv(args.export_csv, index=False)
        log(f"CSV written: {args.export_csv}")
    payload = clean({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "db": args.db_name,
                     "params": {k: v for k, v in vars(args).items() if k != "dsn"}, "coverage": cov, "leak": leak,
                     "targets": ev, "order": names, "angles": ang, "traits": [{"key": k, "group": g_, "label": l_, "desc": d_} for k, g_, l_, d_ in TRAITS],
                     "trait_stats": ta, "k_hist": eng.k_hist[-1] if eng.k_hist else {}, "referees": refs, "upcoming": upr})
    page = DASHBOARD.replace("__TITLE__", htmlmod.escape(args.title)).replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    Path(args.out).write_text(page, encoding="utf-8")
    log(f"dashboard written: {args.out} ({len(page) / 1e6:.1f} MB)")
    for nme in names:
        r = ev.get(nme)
        if not r:
            continue
        mt = r["metrics"].get("holdout", {})
        for a in r["algos"]:
            x = mt.get(a)
            if x:
                log(f"  {nme:10s} {a}: hold-out lift {x['lift'] * 1000:+.2f}e-3 [{x['lo'] * 1000:+.2f}, {x['hi'] * 1000:+.2f}]"
                    + (f"  placebo {x['pl_lift'] * 1000:+.2f}e-3" if x.get("pl_lift") is not None else ""))


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
<style>
.verdict{display:inline-block;font-size:11px;padding:1px 8px;border-radius:99px;border:1px solid var(--ring)}
.verdict.real{background:var(--goodbg);color:var(--good);border-color:transparent}.verdict.weak{color:var(--ink2)}.verdict.none{color:var(--muted)}
.pbar{position:relative;height:8px;background:var(--grid);border-radius:4px;min-width:70px}.pbar i{position:absolute;top:-2px;width:3px;height:12px;border-radius:2px;background:var(--s1)}
.pbar b{position:absolute;left:50%;top:-1px;width:1px;height:10px;background:var(--axis)}
td.wrap{white-space:normal;text-align:left;min-width:220px}
@media (max-width:600px){.g2,.g3{grid-template-columns:1fr}.big{font-size:38px}}
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub" id="sub"></p>
<div class="badges" id="badges"></div>

<div class="controls">
  <label>Period<select id="period"><option value="holdout">Hold-out (honest)</option><option value="discovery">Discovery</option><option value="all">All</option></select></label>
  <label>Model family<select id="algo"><option value="glm">Ridge GLM</option><option value="gbm">Gradient boosting</option></select></label>
  <label>Target in focus<select id="focus"></select></label>
  <button id="theme" type="button">Theme</button>
</div>

<div class="card hero" id="hero"></div>
<div class="grid g5" id="tiles" style="margin-top:14px"></div>

<div class="section"><h2>Does knowing the referee improve the forecast?</h2>
<p class="note">Same matches, same algorithm, same pre-match context (market, both teams' discipline, league level, fixture context). The only difference is the referee features. Lift = context-only loss minus context+referee loss (log loss; Poisson deviance for card counts; squared error for card difference and stoppage time), ×1000. Bars: real referee lift with 95% interval (weekly block bootstrap). Orange dot: the same model fed <i>placebo</i> referees shuffled within league-season; real information must clear it.</p></div>
<div class="grid">
  <div class="card"><h2>Referee lift by target</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>Real referee, interval above zero</span><span class="key"><span class="sw" style="background:var(--ctrl)"></span>Could be chance</span><span class="key"><span class="sw" style="background:var(--s2);border-radius:99px"></span>Placebo referee</span></div><div id="c-lift"></div></div>
  <div class="card"><h2>Detail</h2><div class="wide"><table id="t-lift"></table></div><p class="note" id="lift-note"></p></div>
</div>

<div class="section"><h2>Do referees genuinely differ?</h2>
<p class="note">Every trait is a residual against a pre-match expectation, so referees are not credited for the games they are given. <b>Real share</b>: part of the between-referee spread that is not sampling noise. <b>Reliability</b>: correlation between a referee's alternate matches (Spearman–Brown corrected); below ~0.2 a "profile" is mostly noise. <b>p</b>: permutation test, referees shuffled within league-season. <b>P10–P90</b>: spread of shrunk referee estimates. Whole sample, descriptive only: the forecasting models never see this table.</p></div>
<div class="card"><div class="tabs" id="tr-tabs"></div><div class="scroll" style="max-height:620px"><table id="t-traits"></table></div></div>

<div class="section"><h2>Home bias, pressure and game management: league-wide effects</h2>
<p class="note">Average residual over all matches with a 95% interval from a bootstrap over referees. Home-bias rows show how much decisions lean towards the home side <i>beyond</i> what the two teams' own histories predict. Slopes measure pressure: how far the decision index moves per unit of crowd size, favourite strength or a team's record of winning decisions. A population effect says officials as a group respond; the reliability column above says whether some referees respond more than others.</p></div>
<div class="grid g2">
  <div class="card"><h2>Home and pressure</h2><div id="c-pop1"></div></div>
  <div class="card"><h2>Game flow</h2><div id="c-pop2"></div></div>
</div>

<div class="section"><h2>Focus: <span id="fname"></span></h2><p class="note" id="fdesc"></p></div>
<div class="grid g2">
  <div class="card"><h2>Outcome vs expectation by referee quintile</h2><p class="note" id="q-note"></p><div id="c-q"></div></div>
  <div class="card"><h2>What drives it (hold-out permutation importance)</h2><p class="note">Boosted model with referee features trained on discovery matches; increase in hold-out loss (×1000) when a group of inputs is scrambled. Bigger = matters more.</p><div id="c-grp"></div><h3>Top individual inputs</h3><div id="c-feat"></div></div>
  <div class="card"><h2>Referee coefficients (ridge GLM, averaged over every refit)</h2><p class="note">Standardised: effect of +1 SD of the input on the log-odds / log-rate / value. "Same sign" = share of refits keeping that direction.</p><div class="scroll" style="max-height:420px"><table id="t-coef"></table></div></div>
  <div class="card"><h2>Calibration (referee model, hold-out)</h2><div id="c-cal"></div></div>
</div>
<div class="grid g2" style="margin-top:14px" id="bet-wrap">
  <div class="card"><h2>Betting vs the market</h2><p class="note">Back the side with the largest edge when edge ≥ τ (τ chosen on discovery). Average and best closing prices.</p><div class="wide"><table id="t-bets"></table></div></div>
  <div class="card"><h2>Cumulative profit (1-unit stakes, average prices)</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Profit</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Hold-out starts</span></div><div id="c-cum"></div></div>
</div>

<div class="section"><h2>Referee angles</h2><p class="note">Pre-declared rules. Cut-offs are quantiles of each trait over discovery matches only (referees with 15+ earlier matches) and are applied unchanged to the hold-out.</p></div>
<div class="card"><div class="scroll" style="max-height:620px"><table id="t-angles"></table></div></div>

<div class="section"><h2>Referee explorer</h2><p class="note">Current shrunk profile of every referee with enough matches (end of data). Cells show the percentile among listed referees; hover for the value. Click a referee for the full profile.</p></div>
<div class="card">
  <div class="controls" style="position:static;margin-bottom:10px"><label>Search<input id="r-q" placeholder="name or league"></label><label>Columns<select id="r-grp"></select></label></div>
  <div class="scroll" style="max-height:520px"><table id="t-refs"></table></div>
</div>
<div class="card" style="margin-top:14px" id="r-prof-card"><h2 id="r-prof-name">Referee profile</h2><p class="note">Percentile among listed referees (50 = typical). Traits with low reliability are faded: a referee's value there is mostly noise.</p><div id="c-prof"></div></div>

<div class="section"><h2>Upcoming fixtures</h2><p class="note" id="up-note"></p></div>
<div class="card"><div class="scroll" style="max-height:620px"><table id="t-up"></table></div></div>

<div class="section"><h2>Method and leakage checks</h2></div>
<div class="grid g2">
  <div class="card"><h2>How look-ahead is prevented</h2><ul class="m" id="safeguards"></ul><h3>Time-travel tests</h3><table id="t-leak"></table></div>
  <div class="card"><h2>Coverage</h2><table id="t-cov"></table><h3>Shrinkage strength k per trait (latest)</h3><div class="kv" id="kv-k"></div></div>
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
const TL = Object.fromEntries(R.traits.map(t => [t.key, t]));
const perSel = $('#period'), algSel = $('#algo'), focSel = $('#focus');
const P = () => perSel.value;
const tgt = n => R.targets[n];
const LOSS = {multi:'log loss', bin:'log loss', count:'Poisson deviance', gauss:'squared error'};
function lift(n){ const t = tgt(n); const m = t && t.metrics[P()]; if (!m) return null; return m[algSel.value] || m.glm || null; }
function verdict(x){ if (!x) return ['none','–'];
  const realPl = x.pl_lift == null || x.vs_pl_lo == null || x.vs_pl_lo > 0;
  if (x.lo > 0 && realPl) return ['real','Referee adds information'];
  if (x.lift > 0) return ['weak','Positive, could be chance'];
  return ['none','No measurable effect']; }
function traitName(c){ const k = c.replace(/^(pl_)?r_/, ''); return TL[k] ? TL[k].label : c.replace(/^c_/, '').replace(/_/g,' '); }

function header(){
  const c = R.coverage;
  $('#sub').innerHTML = `${int(c.eval)} matches in ${c.leagues} competitions, ${c.from} to ${c.to}. Hold-out from <b>${c.holdout_from}</b> (${int(c.n_holdout)} matches): every model, threshold and cut-off was fixed before it. ${int(c.n_refs)} referees seen, ${int(c.n_refs_listed)} with ${R.params.min_ref_matches}+ matches. Generated ${R.generated} from '${esc(R.db)}'.`;
  const lk = R.leak, b = [];
  b.push(lk.skipped ? '<span class="badge">Time-travel test skipped</span>' : lk.passed ? '<span class="badge ok">✓ Time-travel test passed</span>' : '<span class="badge bad">✗ Time-travel test found differences</span>');
  b.push(c.placebo ? '<span class="badge ok">✓ Placebo referees run</span>' : '<span class="badge">Placebo off</span>');
  b.push(lk.alarms && lk.alarms.length ? `<span class="badge bad">⚠ ${lk.alarms.length} implausible-gain alarm(s)</span>` : '<span class="badge ok">✓ No implausible-gain alarms</span>');
  b.push(`<span class="badge">Referee known for ${pctu(c.with_ref,0)} of matches</span><span class="badge">Events ${pctu(c.with_events,0)} · stoppage time ${pctu(c.with_added,0)} · BTTS prices ${pctu(c.with_btts,0)}</span>`);
  $('#badges').innerHTML = b.join('');
  focSel.innerHTML = R.order.filter(n => R.targets[n]).map(n => `<option value="${n}">${esc(R.targets[n].label)}</option>`).join('');
}

function hero(){
  const rows = R.order.filter(n => R.targets[n]).map(n => ({n, x: lift(n), t: tgt(n)}));
  const real = rows.filter(r => verdict(r.x)[0] === 'real');
  const best = rows.filter(r => r.x && r.x.lift_pct != null).sort((a, b) => b.x.lift_pct - a.x.lift_pct)[0];
  $('#hero').innerHTML = `<div><div class="lbl">Targets where the referee adds information (${P()})</div><div class="big">${real.length}<span style="font-size:22px;color:var(--muted)"> / ${rows.length}</span></div></div>
    <div style="max-width:720px" class="lbl">${real.length ? 'Clear referee signal: <b>' + real.map(r => esc(r.t.label)).join('</b>, <b>') + '</b>.' : 'No target shows a referee effect that clears both zero and the placebo in this period.'}
    ${best ? ` Largest relative gain: <b>${esc(best.t.label)}</b>, ${pct(best.x.lift_pct, 2)} of the context model's loss.` : ''} Effects on cards and stoppage time are expected to be larger than effects on goals and results, which the market already prices.</div>`;
  $('#tiles').innerHTML = rows.map(r => { const v = verdict(r.x); const x = r.x;
    return `<div class="card tile"><div class="lbl">${esc(r.t.label)}</div><div class="val ${x ? cls(x.lift) : ''}">${x ? sgn(x.lift*1000, 2) : '–'}<span style="font-size:13px;color:var(--muted)"> ×10⁻³</span></div>
      <div class="ci">${x ? `[${sgn(x.lo*1000,2)}, ${sgn(x.hi*1000,2)}] · ${pct(x.lift_pct,2)} of loss` : 'not enough data'}</div><div style="margin-top:6px"><span class="verdict ${v[0]}">${v[1]}</span></div></div>`; }).join('');
}

function liftSection(){
  const rows = R.order.filter(n => R.targets[n] && lift(n)).map(n => { const x = lift(n), v = verdict(x);
    return {label: tgt(n).label.length > 26 ? n : tgt(n).label, v: x.lift*1000, lo: x.lo*1000, hi: x.hi*1000, v2: x.pl_lift != null ? x.pl_lift*1000 : null,
      color: v[0] === 'real' ? css('--s1') : css('--ctrl'), onclick: () => { focSel.value = n; focus(); $('#fname').scrollIntoView({behavior:'smooth'}); },
      tip: `<b>${esc(tgt(n).label)}</b><br>Lift ${sgn(x.lift*1000,2)}×10⁻³ [${sgn(x.lo*1000,2)}, ${sgn(x.hi*1000,2)}]<br>Placebo ${x.pl_lift != null ? sgn(x.pl_lift*1000,2) : '–'}×10⁻³<br>${LOSS[tgt(n).kind]} ${num(x.base_loss,4)} → ${num(x.ref_loss,4)}`}; });
  hbars($('#c-lift'), rows, v => sgn(v, 1), {left: 190, W: 1100, rowH: 28});
  const head = [{h:'Target', c:'l'}, {h:'n'}, {h:'Loss'}, {h:'Context'}, {h:'+ Referee'}, {h:'Market', t:'Closing market alone, where priced'}, {h:'Lift ×10⁻³'}, {h:'95% CI'}, {h:'Placebo ×10⁻³'}, {h:'Real − placebo', t:'Lift of the real referee features over the placebo ones, with 95% CI'}, {h:'Verdict', c:'l'}];
  const tr = R.order.filter(n => R.targets[n]).map(n => { const t = tgt(n), m = t.metrics[P()] || {}, x = lift(n), v = verdict(x);
    return {k:n, v:[t.label, m.n, null, x?.base_loss, x?.ref_loss, m.market_loss, x?.lift, x?.lo, x?.pl_lift, x?.vs_pl, v[1]],
      c:[esc(t.label), int(m.n), LOSS[t.kind], num(x?.base_loss,4), num(x?.ref_loss,4), m.market_loss != null ? num(m.market_loss,4) : '–',
         `<span class="${cls(x?.lift)}">${x ? sgn(x.lift*1000,2) : '–'}</span>`, x ? `${sgn(x.lo*1000,2)} … ${sgn(x.hi*1000,2)}` : '–',
         x?.pl_lift != null ? sgn(x.pl_lift*1000,2) : '–', x?.vs_pl != null ? `${sgn(x.vs_pl*1000,2)} [${sgn(x.vs_pl_lo*1000,2)}, ${sgn(x.vs_pl_hi*1000,2)}]` : '–',
         `<span class="verdict ${v[0]}">${v[1]}</span>`]}; });
  sortable($('#t-lift'), head, tr, {onRow: k => { focSel.value = k; focus(); }});
  $('#lift-note').textContent = 'Where the market column is filled, compare it with "+ Referee": a referee model that beats the closing market there is a betting candidate (see Focus). Click a row to focus it.';
}

let trGroup = 'All';
function traitsSection(){
  const groups = ['All', ...new Set(R.traits.map(t => t.group))];
  $('#tr-tabs').innerHTML = groups.map(g => `<button type="button" class="${g===trGroup?'on':''}" data-g="${g}">${g}</button>`).join('');
  $('#tr-tabs').querySelectorAll('button').forEach(b => b.onclick = () => { trGroup = b.dataset.g; traitsSection(); });
  const head = [{h:'Trait', c:'l'}, {h:'Group', c:'l'}, {h:'Referees'}, {h:'Matches'}, {h:'Real share', t:'Share of between-referee variance that is not sampling noise'}, {h:'Reliability'}, {h:'p (perm.)'}, {h:'P10 … P90'}, {h:'League-wide mean [95% CI]'}, {h:'What + means', c:'l wrap'}];
  const rows = R.traits.filter(t => trGroup === 'All' || t.group === trGroup).map(t => { const s = R.trait_stats[t.key] || {};
    const relc = s.reliability == null ? '' : s.reliability >= 0.4 ? 'pos' : s.reliability < 0.15 ? 'neg' : '';
    return {k:t.key, v:[t.label, t.group, s.n_refs, s.n_matches, s.share_real, s.reliability, s.p, s.p90 - s.p10, s.pop_mean, null],
      c:[esc(t.label), t.group, int(s.n_refs), int(s.n_matches), s.share_real != null ? pctu(s.share_real,0) : '–', `<span class="${relc}">${num(s.reliability,2)}</span>`,
         s.p != null ? (s.p < 0.005 ? '<0.005' : num(s.p,3)) : '–', s.p10 != null ? `${num(s.p10,3)} … ${num(s.p90,3)}` : '–',
         s.pop_mean != null ? `${num(s.pop_mean,3)} [${num(s.pop_lo,3)}, ${num(s.pop_hi,3)}]` : '–', `<span class="why">${esc(t.desc)}</span>`]}; });
  sortable($('#t-traits'), head, rows, {sortKey: 5});
}

function popSection(){
  const mk = keys => keys.filter(k => R.trait_stats[k]).map(k => { const s = R.trait_stats[k];
    const sig = s.pop_lo > 0 || s.pop_hi < 0;
    return {label: TL[k].label.length > 30 ? TL[k].label.slice(0, 29) + '…' : TL[k].label, v: s.pop_mean, lo: s.pop_lo, hi: s.pop_hi, color: sig ? css('--s1') : css('--ctrl'),
      tip: `<b>${esc(TL[k].label)}</b><br>${esc(TL[k].desc)}<br>Mean ${num(s.pop_mean,4)} [${num(s.pop_lo,4)}, ${num(s.pop_hi,4)}]<br>${int(s.n_matches)} matches`}; });
  hbars($('#c-pop1'), mk(['home_pts','home_cards','home_fouls','home_pens','home_dec','fergie','crowd_slope','fav_slope','rep_slope','fav_pts']), v => num(v, 2), {left: 200});
  hbars($('#c-pop2'), mk(['goals','btts','after_lead','trailer_scores','alternation','lead_cards','pen_trailer']), v => num(v, 2), {left: 200});
}

function focus(){
  const n = focSel.value, t = tgt(n); if (!t) return;
  $('#fname').textContent = t.label;
  const x = lift(n);
  $('#fdesc').innerHTML = x ? `${LOSS[t.kind]}: context ${num(x.base_loss,4)} → with referee ${num(x.ref_loss,4)} (${pct(x.lift_pct,2)}). Placebo lift ${x.pl_lift != null ? sgn(x.pl_lift*1000,2)+'×10⁻³' : '–'}.` : 'Not enough data in this period.';
  // quintiles
  const q = t.quintiles;
  if (q){ $('#q-note').innerHTML = `Hold-out matches grouped by the referee's <b>${esc(traitName(q.trait))}</b> trait (quintiles fixed on discovery; referees with 10+ earlier matches). Bar = observed ${esc(q.what)} minus what the context-only model expected. A real referee effect rises from Q1 to Q5 (or falls).`;
    cols($('#c-q'), q.rows.map(r => ({label:'Q'+r.q, n:r.n, v: r.diff, lo:r.lo, hi:r.hi, small: r.lo != null && r.lo <= 0 && r.hi >= 0,
      tip: r.obs != null ? `Q${r.q}: observed ${num(r.obs,3)}, expected ${num(r.exp,3)}<br>difference ${sgn(r.diff,3)} [${sgn(r.lo,3)}, ${sgn(r.hi,3)}]<br>${int(r.n)} matches` : 'too few matches'})), v => sgn(v, 2), 'Referee quintile (low → high)'); }
  else { $('#q-note').textContent = ''; empty($('#c-q'), 'Not enough seasoned-referee matches'); }
  // importance
  const im = t.importance;
  if (im){ hbars($('#c-grp'), Object.entries(im.groups).sort((a,b)=>b[1]-a[1]).map(([g,v]) => ({label:g.length>30?g.slice(0,29)+'…':g, v:v*1000, color: g.startsWith('Referee') ? css('--s1') : css('--ctrl'), tip:`<b>${esc(g)}</b><br>+${num(v*1000,3)}×10⁻³ loss when scrambled`})), v => num(v,2), {left: 210});
    hbars($('#c-feat'), Object.entries(im.features).slice(0, 15).map(([f,v]) => ({label: traitName(f).slice(0, 30), v:v*1000, color: /^(r_|rt_)/.test(f) ? css('--s1') : css('--ctrl'), tip:`<b>${esc(f)}</b><br>+${num(v*1000,3)}×10⁻³`})), v => num(v,2), {left: 210, rowH: 24}); }
  else { empty($('#c-grp'), 'Importance needs gradient boosting and enough discovery matches'); $('#c-feat').innerHTML = ''; }
  // coefficients
  const co = t.coefs || [];
  sortable($('#t-coef'), [{h:'Input', c:'l'}, {h:'Mean effect'}, {h:'SD over refits'}, {h:'Same sign'}],
    co.map(c => ({v:[traitName(c.f), c.mean, c.sd, c.same_sign], c:[esc(traitName(c.f)) + `<span class="note"> ${esc(c.f)}</span>`, `<span class="${cls(c.mean)}">${sgn(c.mean,3)}</span>`, num(c.sd,3), pctu(c.same_sign,0)]})), {sortKey: null});
  if (t.calibration) calChart($('#c-cal'), t.calibration); else empty($('#c-cal'), 'Calibration is shown for yes/no targets');
  // bets
  const bets = t.bets || {}, keys = Object.keys(bets);
  $('#bet-wrap').style.display = keys.length ? '' : 'none';
  if (keys.length){
    const head = [{h:'Model / price', c:'l'}, {h:'τ'}, {h:'Bets'}, {h:'ROI'}, {h:'95% CI'}, {h:'p'}, {h:'Avg price'}];
    sortable($('#t-bets'), head, keys.map(k => { const b = bets[k], p = b.periods[P()] || {};
      return {v:[k, b.tau, p.n, p.roi, p.lo, p.p, p.avg_price], c:[k.replace('_', ' · '), pctu(b.tau,0), int(p.n), `<span class="${cls(p.roi)}">${pct(p.roi)}</span>`, p.lo != null ? `${pct(p.lo)} … ${pct(p.hi)}` : '–', num(p.p,3), num(p.avg_price,2)]}; }), {sortKey: null});
    const k = `${algSel.value}_avg` in bets ? `${algSel.value}_avg` : keys[0];
    lines($('#c-cum'), [{name:'Profit', color:css('--s1'), pts: bets[k].cum || [], endLabel: bets[k].cum?.length ? sgn(bets[k].cum[bets[k].cum.length-1][1], 1) : ''}], {yref:0, vline: Date.parse(R.coverage.holdout_from), vlabel:'hold-out', fmt: v => sgn(v, 0)});
  }
}

function anglesSection(){
  const head = [{h:'Angle', c:'l'}, {h:'Disc. bets'}, {h:'Disc. ROI'}, {h:'Hold-out bets'}, {h:'Hold-out ROI'}, {h:'95% CI'}, {h:'p'}, {h:'Win vs implied'}, {h:'Best-price ROI'}, {h:'Why', c:'l wrap'}];
  sortable($('#t-angles'), head, R.angles.map(a => { const d = a.periods.discovery.avg, h = a.periods.holdout.avg, hb = a.periods.holdout.best;
    return {v:[a.title, d.n, d.roi, h.n, h.roi, h.lo, h.p, h.win - h.implied, hb.roi, null],
      c:[`<b>${esc(a.title)}</b>`, int(d.n), `<span class="${cls(d.roi)}">${pct(d.roi)}</span>`, int(h.n), `<span class="${cls(h.roi)}">${pct(h.roi)}</span>`,
         h.lo != null ? `${pct(h.lo)} … ${pct(h.hi)}` : '–', num(h.p,3), h.win != null ? `${pctu(h.win)} vs ${pctu(h.implied)}` : '–', `<span class="${cls(hb.roi)}">${pct(hb.roi)}</span>`, `<span class="why">${esc(a.why)}</span>`]}; }), {sortKey: 4});
}

let refSel = null;
function refsSection(){
  const groups = ['Key traits', ...new Set(R.traits.map(t => t.group))];
  if (!$('#r-grp').options.length) $('#r-grp').innerHTML = groups.map(g => `<option>${g}</option>`).join('');
  const g = $('#r-grp').value, q = $('#r-q').value.trim().toLowerCase();
  const KEY = ['cards','strict','reds','pens','home_dec','fergie','fav_slope','btts','after_lead','added2'];
  const ks = g === 'Key traits' ? KEY : R.traits.filter(t => t.group === g).map(t => t.key);
  const head = [{h:'Referee', c:'l'}, {h:'League', c:'l'}, {h:'Matches'}, ...ks.map(k => ({h: TL[k].label.length > 16 ? TL[k].label.slice(0, 15) + '…' : TL[k].label, t: TL[k].desc}))];
  const rows = R.referees.filter(r => !q || (r.name + ' ' + r.league).toLowerCase().includes(q)).map(r => ({k: String(r.id), sel: r.id === refSel,
    v:[r.name, r.league, r.n, ...ks.map(k => r[k + '_pct'])],
    c:[esc(r.name), esc(r.league), int(r.n), ...ks.map(k => { const p = r[k + '_pct']; if (p == null) return '–';
      return `<span title="${esc(TL[k].label)}: ${num(r[k],3)} (${int(r[k+'_n'])} obs)"><span class="pbar" style="display:inline-block;width:70px;vertical-align:middle"><b></b><i style="left:${(p*100).toFixed(0)}%"></i></span> ${Math.round(p*100)}</span>`; })]}));
  sortable($('#t-refs'), head, rows, {sortKey: 2, onRow: k => { refSel = +k; profile(); refsSection(); }});
}
function profile(){
  const r = R.referees.find(x => x.id === refSel) || R.referees.slice().sort((a,b) => b.n - a.n)[0];
  if (!r){ $('#r-prof-card').style.display = 'none'; return; }
  $('#r-prof-name').textContent = `${r.name} · ${r.league} · ${r.n} matches · last ${r.last}`;
  hbars($('#c-prof'), R.traits.filter(t => r[t.key + '_pct'] != null).map(t => { const rel = (R.trait_stats[t.key] || {}).reliability;
    return {label: t.label.length > 30 ? t.label.slice(0, 29) + '…' : t.label, v: (r[t.key + '_pct'] - 0.5) * 100,
      color: rel != null && rel >= 0.2 ? css('--s1') : css('--ctrl'), tip: `<b>${esc(t.label)}</b><br>${esc(t.desc)}<br>Value ${num(r[t.key],4)} · percentile ${Math.round(r[t.key+'_pct']*100)}<br>${int(r[t.key+'_n'])} observations · trait reliability ${num(rel,2)}`}; }),
    v => (v >= 0 ? '+' : '') + Math.round(v), {left: 230, rowH: 22, W: 1150});
}

function upcomingSection(){
  const U = R.upcoming || [];
  $('#up-note').textContent = U.length ? `Fixtures from the end of the data up to ${R.params.upcoming_days} days ahead, priced with models trained on the latest window. "Ref" = referee GLM; "ctx" = same model without referee features; market = de-vigged closing price where available.` : 'No upcoming fixtures with kickoff after the last result were found (use --upcoming-days).';
  if (!U.length){ $('#t-up').innerHTML = ''; return; }
  const g = (u, n, k='ref') => u.pred[n] ? u.pred[n][k] : null;
  const pr = (u, n, i) => { const v = g(u, n); return v == null ? null : Array.isArray(v) ? v[i] : v; };
  const head = [{h:'Date', c:'l'}, {h:'Match', c:'l'}, {h:'Referee', c:'l'}, {h:'Ref n'}, {h:'Home win ref / mkt'}, {h:'Over 2.5 ref / mkt'}, {h:'BTTS ref / mkt'}, {h:'Cards ref / ctx'}, {h:'Red'}, {h:'Penalty'}, {h:'Away−home cards'}];
  sortable($('#t-up'), head, U.map(u => ({v:[u.date, u.home, u.ref, u.ref_n, pr(u,'home',0), pr(u,'over25'), pr(u,'btts'), pr(u,'cards'), pr(u,'red'), pr(u,'pen'), pr(u,'card_diff')],
    c:[u.date, `${esc(u.home)} – ${esc(u.away)}<span class="note"> ${esc(u.league)}</span>`, esc(u.ref || '–'), int(u.ref_n),
       `${pctu(pr(u,'home',0))} / ${pctu(u.market.p_h)}`, `${pctu(pr(u,'over25'))} / ${pctu(u.market.ou_p_o)}`, `${pctu(pr(u,'btts'))} / ${pctu(u.market.btts_p_y)}`,
       `${num(pr(u,'cards'),1)} / ${num(g(u,'cards','base'),1)}`, pctu(pr(u,'red')), pctu(pr(u,'pen')), sgn(pr(u,'card_diff'),2)]})), {sortKey: 0, sortDir: 1});
}

function methodSection(){
  const S = ['Features come from a day-by-day state machine: all of a day\'s rows are written before any of that day\'s results enter the state (referee accumulators, team and league rates, online regressions).',
    'Pre-match inputs only: referee appointment, past fixtures, venue, weather, and odds whose last update is not after kickoff. In-play targets add only the half-time score.',
    'Season tables (referee_season_stats, referee_season_summary, standings, season statistics) are never read: they contain the season\'s future matches.',
    'Every referee trait is a residual against a pre-match expectation, league-centred, time-decayed and shrunk by empirical Bayes; shrinkage strength is re-estimated every 14 days from past matches only.',
    `Walk-forward: models retrained every ${R.params.retrain_months} month(s) on the previous ${R.params.train_window_days} days; ridge strength and boosting shrinkage chosen on the latest 20% of each training window.`,
    'Placebo: the whole engine is re-run with referees shuffled within league-season. Real referee lift has to beat the placebo lift.',
    'Bet thresholds and angle cut-offs are fixed on the discovery period only. Trait reliability tables use the whole sample and are descriptive: models never see them.'];
  $('#safeguards').innerHTML = S.map(s => `<li>${s}</li>`).join('');
  const lk = R.leak;
  $('#t-leak').innerHTML = lk.skipped ? '<tr><td>Skipped (--skip-leak-test)</td></tr>' : '<thead><tr><th class="l">Cutoff</th><th>Matches</th><th>Features</th><th>Mismatches</th></tr></thead><tbody>' +
    lk.tests.map(t => `<tr><td class="l">${t.day}</td><td>${t.matches}</td><td>${t.features}</td><td class="${t.mismatches ? 'neg' : 'pos'}">${t.mismatches}${t.columns && t.columns.length ? ' (' + esc(t.columns.join(', ')) + ')' : ''}</td></tr>`).join('') + '</tbody>';
  const c = R.coverage;
  const cv = [['Matches in evaluation window', int(c.eval)], ['Discovery / hold-out', `${int(c.n_discovery)} / ${int(c.n_holdout)}`], ['Referee appointment known', pctu(c.with_ref)],
    ['Match events (cards, goals, subs)', pctu(c.with_events)], ['Team fouls', pctu(c.with_fouls)], ['Stoppage time', pctu(c.with_added)], ['1X2 prices', pctu(c.with_1x2)], ['O/U 2.5 prices', pctu(c.with_ou)],
    ['BTTS prices', pctu(c.with_btts)], ['Card lines', pctu(c.with_cards_line)], ['Referees seen / listed', `${int(c.n_refs)} / ${int(c.n_refs_listed)}`], ['Run time', `${Math.round(c.runtime_s/60)} min`]];
  $('#t-cov').innerHTML = '<tbody>' + cv.map(r => `<tr><td class="l">${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</tbody>';
  $('#kv-k').innerHTML = R.traits.map(t => `<div><span>${esc(t.label)}</span> ${R.k_hist[t.key] != null ? num(R.k_hist[t.key], 0) : '–'}</div>`).join('');
}

function all(){ hero(); liftSection(); focus(); }
perSel.onchange = all; algSel.onchange = all; focSel.onchange = focus;
$('#r-q').oninput = refsSection; $('#r-grp').onchange = refsSection;
$('#theme').onclick = () => { const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; all(); traitsSection(); popSection(); profile(); };
if (!R.targets[R.order[0]]?.algos?.includes('gbm')) algSel.querySelector('[value=gbm]').disabled = true;
header(); all(); traitsSection(); popSection(); anglesSection(); refsSection(); profile(); upcomingSection(); methodSection();
</script>
</body>
</html>
'''

if __name__ == "__main__":
    main()
