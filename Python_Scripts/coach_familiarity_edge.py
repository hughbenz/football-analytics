#!/usr/bin/env python3
"""
coach_familiarity_edge.py
=========================

Question
--------
Does a team beat the betting market when its coach has inside knowledge of the
opponent, because he

  (a) recently coached the opposition club ("club familiarity"), or
  (b) has recently coached many of the opposition's players ("player familiarity")?

Data
----
Reads the Sportmonks-derived PostgreSQL database built from
claude/sportmonks_schema.sql. Tables used:

  football.fixtures, fixture_participants, fixture_scores    results (90-minute)
  football.fixture_coaches, football.coach_teams             who coached whom, when
  football.fixture_lineups, fixture_lineup_details           who played under whom (minutes)
  odds.prematch_odds, odds.premium_odds                      closing 1X2 prices, every bookmaker
  odds.premium_odds_history                                  opening prices (line movement)
  football.predictions                                       Sportmonks model probabilities
  football.fixture_xg, football.fixture_statistics           xG, shots on target, possession
  core.types, core.states, football.leagues/seasons/teams/coaches

Method (one row per team per finished match)
--------------------------------------------
  club_days      days since our coach last took charge of the opponent club
  club_matches   matches he coached them for (match evidence; coach_teams fills gaps)
  n_former       opponent players in this match (starters by default) who made
                 >= --min-apps appearances under our coach in the previous
                 --player-window-days, at any club
  opp_*          the same measures for the opposition coach against us
  Market         de-vigged consensus closing probability across all bookmakers,
                 plus average / best / sharp (Pinnacle) / opening prices for ROI

Treatments (pick in the dashboard):
  club     club_days <= --club-window-days and club_matches >= --min-spell-matches
  players  n_former >= --player-threshold
  any      club or players
  net      any, and the opposition coach is NOT familiar with us

Outcomes: flat-stake ROI backing the team to win (and draw-no-bet), excess win
rate vs market probability, excess points, xG and shots-on-target difference vs
what the market price implies. Uncertainty: bootstrap resampling coaches
(clustered), and a logistic regression of win on market log-odds plus
familiarity and controls (home, new-coach bounce, rest days) with
coach-clustered standard errors.

Usage
-----
  pip install pandas numpy psycopg[binary]     # psycopg2 also works; psql CLI is a fallback
  Put this file next to load_league.py and it uses the same connection (load_league.DATABASE):
      python coach_familiarity_edge.py --export-csv team_matches.csv
  or point it at the database explicitly:
      python coach_familiarity_edge.py --dsn "postgresql://user:pass@host:5432/football"

  Useful options: --leagues 8,82  --seasons 23614,23744  --include-subs
                  --club-window-days 730  --player-window-days 730  --min-apps 5
                  --player-threshold 3  --sharp-bookmaker Pinnacle  --cache .cache
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd



# =============================================================================
# Database access: psycopg (3) -> psycopg2 -> psql CLI
# =============================================================================
def default_dsn() -> tuple[str, str]:
    """Connection string, in order: DATABASE_URL, then load_league.DATABASE
    (the loader that sits in the same folder), then libpq defaults (PG* env vars)."""
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


class DB:
    def __init__(self, dsn: str | None):
        source = "--dsn"
        if not dsn:
            dsn, source = default_dsn()
        self.dsn = dsn
        self.kind, self.conn = None, None
        try:
            import psycopg  # type: ignore
            self.conn = psycopg.connect(self.dsn)
            self.kind = "psycopg"
        except ImportError:
            try:
                import psycopg2  # type: ignore
                self.conn = psycopg2.connect(self.dsn)
                self.kind = "psycopg2"
            except ImportError:
                self.kind = "psql"
        name = self.query("SELECT current_database() AS db").db.iloc[0]
        log(f"database driver: {self.kind} · connected to '{name}' (settings from {source})")
        check = self.query("SELECT to_regclass('football.fixtures') IS NOT NULL AS ok").ok.iloc[0]
        if not check or str(check).lower() in ("f", "false"):
            sys.exit(f"Database '{name}' has no football.fixtures table - this is not the Sportmonks database.\n"
                     "Pass --dsn \"postgresql://user:pass@host:5432/<your db>\", set DATABASE_URL, "
                     "or put this script next to load_league.py.")

    def query(self, sql: str) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        if self.kind == "psql":
            cmd = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1"] + ([self.dsn] if self.dsn else [])
            cmd += ["-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode())
            data = res.stdout.decode()
            if not data.strip():
                return pd.DataFrame()
            return _coerce(pd.read_csv(io.StringIO(data)))
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        finally:
            self.conn.rollback()  # read-only; also clears a failed transaction so later queries still run
        return _coerce(pd.DataFrame(rows, columns=cols))


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            if c in ("starting_at", "start_date", "end_date", "update_ts"):
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
                continue
            vals = set(df[c].dropna().astype(str).str.lower().unique())
            if vals and vals <= {"t", "f", "true", "false"}:  # psql CSV booleans
                df[c] = df[c].astype(str).str.lower().map({"t": True, "true": True, "f": False, "false": False})
                continue
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() == df[c].notna().sum() and df[c].notna().any():
                df[c] = conv
        if c in ("starting_at", "start_date", "end_date", "update_ts") and not str(df[c].dtype).startswith("datetime"):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


def ints(csv_arg: str | None) -> list[int]:
    return [int(x) for x in csv_arg.split(",") if x.strip()] if csv_arg else []


# =============================================================================
# SQL extracts
# =============================================================================
def scope_sql(alias: str, args) -> str:
    parts = []
    if ints(args.leagues):
        parts.append(f"{alias}.league_id IN ({','.join(map(str, ints(args.leagues)))})")
    if ints(args.seasons):
        parts.append(f"{alias}.season_id IN ({','.join(map(str, ints(args.seasons)))})")
    return (" AND " + " AND ".join(parts)) if parts else ""


def sql_fixtures(args) -> str:
    # 1X2 settles on 90 minutes: matches decided in extra time / penalties use
    # the 2ND_HALF score (cumulative at 90'), everything else uses CURRENT.
    return f"""
    WITH sc AS (
        SELECT fixture_id,
               max(goals) FILTER (WHERE location = 'home' AND description = 'CURRENT')  AS h_cur,
               max(goals) FILTER (WHERE location = 'away' AND description = 'CURRENT')  AS a_cur,
               max(goals) FILTER (WHERE location = 'home' AND description = '2ND_HALF') AS h_90,
               max(goals) FILTER (WHERE location = 'away' AND description = '2ND_HALF') AS a_90
        FROM football.fixture_scores
        WHERE description IN ('CURRENT', '2ND_HALF')
        GROUP BY fixture_id
    )
    SELECT f.fixture_id, f.league_id, l.name AS league, f.season_id, s.name AS season,
           f.starting_at,
           COALESCE(f.home_team_id, hp.team_id) AS home_team_id,
           COALESCE(f.away_team_id, ap.team_id) AS away_team_id,
           ht.name AS home_team, at.name AS away_team, st.developer_name AS state,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN sc.h_90 ELSE sc.h_cur END AS home_goals,
           CASE WHEN st.developer_name IN ('AET', 'FT_PEN') THEN sc.a_90 ELSE sc.a_cur END AS away_goals,
           hp.position AS home_position, ap.position AS away_position
    FROM football.fixtures f
    JOIN core.states st ON st.state_id = f.state_id
    JOIN sc ON sc.fixture_id = f.fixture_id
    LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
    LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'
    LEFT JOIN football.leagues l ON l.league_id = f.league_id
    LEFT JOIN football.seasons s ON s.season_id = f.season_id
    LEFT JOIN football.teams ht ON ht.team_id = COALESCE(f.home_team_id, hp.team_id)
    LEFT JOIN football.teams at ON at.team_id = COALESCE(f.away_team_id, ap.team_id)
    WHERE st.developer_name IN ('FT', 'AET', 'FT_PEN') {scope_sql('f', args)}
    """


# Every match a team played (all competitions) -> rest days. Not scoped.
SQL_TEAM_DATES = """
    SELECT p.team_id, f.starting_at
    FROM football.fixture_participants p
    JOIN football.fixtures f ON f.fixture_id = p.fixture_id
    WHERE f.starting_at < now()
"""

# Coach in charge per team per match (all competitions, not scoped: history matters)
SQL_FIXTURE_COACHES = """
    SELECT fc.fixture_id, fc.team_id, fc.coach_id, f.starting_at
    FROM football.fixture_coaches fc
    JOIN football.fixtures f ON f.fixture_id = fc.fixture_id
    WHERE fc.team_id IS NOT NULL AND f.starting_at < now()
"""

SQL_COACH_NAMES = """
    SELECT coach_id, COALESCE(display_name, common_name, name) AS coach FROM football.coaches
"""

# Coaching spells (fills gaps where fixture_coaches is missing). Assistants excluded.
SQL_COACH_SPELLS = """
    SELECT ct.coach_id, ct.team_id, ct.start_date, ct.end_date, ct.temporary,
           t.developer_name AS position
    FROM football.coach_teams ct
    LEFT JOIN core.types t ON t.type_id = ct.position_id
    WHERE ct.start_date IS NOT NULL
      AND COALESCE(t.developer_name, '') !~* '(ASSIST|GOALKEEP|FITNESS|ANALYST|SCOUT|DOCTOR|PHYSIO)'
"""

# Every appearance (all competitions): starter, or sub with minutes
SQL_APPEARANCES = """
    WITH mins AS (
        SELECT ld.lineup_id, max(ld.value) AS minutes
        FROM football.fixture_lineup_details ld
        WHERE ld.type_id IN (SELECT type_id FROM core.types WHERE developer_name = 'MINUTES_PLAYED')
        GROUP BY ld.lineup_id
    )
    SELECT l.fixture_id, l.team_id, l.player_id,
           (lt.developer_name = 'LINEUP')::int AS started,
           m.minutes
    FROM football.fixture_lineups l
    JOIN football.fixtures f ON f.fixture_id = l.fixture_id
    LEFT JOIN core.types lt ON lt.type_id = l.type_id
    LEFT JOIN mins m ON m.lineup_id = l.lineup_id
    WHERE l.player_id IS NOT NULL AND l.team_id IS NOT NULL AND f.starting_at < now()
"""


def market_cte(args) -> str:
    name = args.market_name.replace("'", "''")
    return f"""mk AS (
        SELECT market_id FROM odds.markets
        WHERE market_id = {int(args.market_id)} OR developer_name = '{name}'
    )"""


SIDE_CASE = """CASE WHEN lower(coalesce(label, '')) IN ('home', '1') OR {ol} = '1' THEN 'H'
                    WHEN lower(coalesce(label, '')) IN ('draw', 'x') OR {ol} = 'x' THEN 'D'
                    WHEN lower(coalesce(label, '')) IN ('away', '2') OR {ol} = '2' THEN 'A' END"""


def sql_closing_odds(args) -> str:
    """Latest (closing) 1X2 price per fixture x bookmaker, both odds feeds."""
    return f"""
    WITH {market_cte(args)},
    o AS (
        SELECT fixture_id, bookmaker_id, decimal_odds,
               {SIDE_CASE.format(ol="lower(coalesce(original_label, ''))")} AS side
        FROM odds.prematch_odds
        WHERE market_id IN (SELECT market_id FROM mk) AND decimal_odds > 1 {scope_sql('prematch_odds', args)}
        UNION ALL
        SELECT fixture_id, bookmaker_id, decimal_odds,
               {SIDE_CASE.format(ol="''")} AS side
        FROM odds.premium_odds
        WHERE market_id IN (SELECT market_id FROM mk) AND decimal_odds > 1 {scope_sql('premium_odds', args)}
    )
    SELECT o.fixture_id, o.bookmaker_id, b.name AS bookmaker,
           avg(decimal_odds) FILTER (WHERE side = 'H') AS o_h,
           avg(decimal_odds) FILTER (WHERE side = 'D') AS o_d,
           avg(decimal_odds) FILTER (WHERE side = 'A') AS o_a
    FROM o LEFT JOIN odds.bookmakers b ON b.bookmaker_id = o.bookmaker_id
    WHERE side IS NOT NULL
    GROUP BY o.fixture_id, o.bookmaker_id, b.name
    """


def sql_opening_odds(args) -> str:
    """Earliest pre-kickoff price per premium odd (opening line)."""
    return f"""
    WITH {market_cte(args)},
    po AS (
        SELECT odd_id, fixture_id, bookmaker_id, {SIDE_CASE.format(ol="''")} AS side
        FROM odds.premium_odds
        WHERE market_id IN (SELECT market_id FROM mk) {scope_sql('premium_odds', args)}
    ),
    h AS (
        SELECT DISTINCT ON (h.odd_id) h.odd_id, h.decimal_odds
        FROM odds.premium_odds_history h
        JOIN po ON po.odd_id = h.odd_id
        JOIN football.fixtures f ON f.fixture_id = po.fixture_id
        WHERE h.decimal_odds > 1 AND h.bookmaker_update < f.starting_at
        ORDER BY h.odd_id, h.bookmaker_update
    )
    SELECT po.fixture_id, po.bookmaker_id,
           avg(h.decimal_odds) FILTER (WHERE po.side = 'H') AS o_h,
           avg(h.decimal_odds) FILTER (WHERE po.side = 'D') AS o_d,
           avg(h.decimal_odds) FILTER (WHERE po.side = 'A') AS o_a
    FROM h JOIN po ON po.odd_id = h.odd_id
    WHERE po.side IS NOT NULL
    GROUP BY po.fixture_id, po.bookmaker_id
    """


SQL_PREDICTIONS = """
    SELECT p.fixture_id,
           avg(core.num(p.predictions ->> 'home')) AS pred_h,
           avg(core.num(p.predictions ->> 'draw')) AS pred_d,
           avg(core.num(p.predictions ->> 'away')) AS pred_a
    FROM football.predictions p
    JOIN core.types t ON t.type_id = p.type_id
    WHERE t.developer_name = 'FULLTIME_RESULT_PROBABILITY'
    GROUP BY p.fixture_id
"""

SQL_PERFORMANCE = """
    WITH x AS (
        SELECT x.fixture_id, x.team_id, max(x.value) AS xg
        FROM football.fixture_xg x JOIN core.types t ON t.type_id = x.type_id
        WHERE t.developer_name = 'EXPECTED_GOALS'
        GROUP BY 1, 2
    ),
    s AS (
        SELECT s.fixture_id, s.team_id,
               max(s.value) FILTER (WHERE t.developer_name = 'SHOTS_ON_TARGET')   AS sot,
               max(s.value) FILTER (WHERE t.developer_name = 'BALL_POSSESSION')   AS possession,
               max(s.value) FILTER (WHERE t.developer_name = 'DANGEROUS_ATTACKS') AS dangerous_attacks
        FROM football.fixture_statistics s JOIN core.types t ON t.type_id = s.type_id
        WHERE t.developer_name IN ('SHOTS_ON_TARGET', 'BALL_POSSESSION', 'DANGEROUS_ATTACKS')
        GROUP BY 1, 2
    )
    SELECT COALESCE(x.fixture_id, s.fixture_id) AS fixture_id,
           COALESCE(x.team_id, s.team_id) AS team_id,
           x.xg, s.sot, s.possession, s.dangerous_attacks
    FROM x FULL JOIN s ON s.fixture_id = x.fixture_id AND s.team_id = x.team_id
"""


def extract(db: DB, args) -> dict[str, pd.DataFrame]:
    queries = {
        "fixtures": sql_fixtures(args),
        "team_dates": SQL_TEAM_DATES,
        "fixture_coaches": SQL_FIXTURE_COACHES,
        "coach_names": SQL_COACH_NAMES,
        "coach_spells": SQL_COACH_SPELLS,
        "appearances": SQL_APPEARANCES,
        "closing": sql_closing_odds(args),
        "opening": sql_opening_odds(args),
        "predictions": SQL_PREDICTIONS,
        "performance": SQL_PERFORMANCE,
    }
    cache = Path(args.cache) if args.cache else None
    out = {}
    for name, sql in queries.items():
        path = cache / f"{name}.pkl" if cache else None
        if path and path.exists() and not args.refresh:
            out[name] = pd.read_pickle(path)
            log(f"{name:16s} {len(out[name]):>10,} rows (cache)")
            continue
        try:
            out[name] = db.query(sql)
        except Exception as e:  # optional tables may be empty / missing
            if name in ("fixtures", "fixture_coaches", "appearances", "closing"):
                sys.exit(f"Required query '{name}' failed: {e}")
            log(f"{name}: query failed ({str(e).strip().splitlines()[0][:120]}) - continuing without it")
            out[name] = pd.DataFrame()
        log(f"{name:16s} {len(out[name]):>10,} rows")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            out[name].to_pickle(path)
    return out


# =============================================================================
# Market: de-vigged consensus probabilities and prices
# =============================================================================
def market_table(closing: pd.DataFrame, opening: pd.DataFrame, sharp: str) -> pd.DataFrame:
    c = closing.dropna(subset=["o_h", "o_d", "o_a"])
    c = c[(c[["o_h", "o_d", "o_a"]] > 1).all(axis=1)]
    inv = 1 / c[["o_h", "o_d", "o_a"]].to_numpy(float)
    over = inv.sum(axis=1)
    c = c.assign(margin=over - 1, q_h=inv[:, 0] / over, q_d=inv[:, 1] / over, q_a=inv[:, 2] / over)
    c = c[(c.margin > -0.02) & (c.margin < 0.30)]  # drop broken books
    g = c.groupby("fixture_id")
    m = pd.DataFrame({
        "n_books": g.size(),
        "p_h": g.q_h.mean(), "p_d": g.q_d.mean(), "p_a": g.q_a.mean(),
        "avg_h": g.o_h.mean(), "avg_d": g.o_d.mean(), "avg_a": g.o_a.mean(),
        "best_h": g.o_h.max(), "best_d": g.o_d.max(), "best_a": g.o_a.max(),
        "margin": g.margin.mean(),
    })
    s = m[["p_h", "p_d", "p_a"]].sum(axis=1)
    for k in ("h", "d", "a"):
        m[f"p_{k}"] = m[f"p_{k}"] / s
    sh = c[c.bookmaker.fillna("").str.lower().str.contains(sharp.lower())] if sharp else c.iloc[0:0]
    if len(sh):
        sh = sh.groupby("fixture_id")[["o_h", "o_d", "o_a"]].mean()
        m = m.join(sh.rename(columns={"o_h": "sharp_h", "o_d": "sharp_d", "o_a": "sharp_a"}))
    else:
        m[["sharp_h", "sharp_d", "sharp_a"]] = np.nan
    if len(opening):
        o = opening.dropna(subset=["o_h", "o_d", "o_a"])
        o = o[(o[["o_h", "o_d", "o_a"]] > 1).all(axis=1)]
        inv = 1 / o[["o_h", "o_d", "o_a"]].to_numpy(float)
        over = inv.sum(axis=1)
        o = o.assign(q_h=inv[:, 0] / over, q_d=inv[:, 1] / over, q_a=inv[:, 2] / over)
        go = o.groupby("fixture_id")
        m = m.join(pd.DataFrame({"open_p_h": go.q_h.mean(), "open_p_a": go.q_a.mean(),
                                 "open_h": go.o_h.mean(), "open_d": go.o_d.mean(), "open_a": go.o_a.mean()}))
    else:
        m[["open_p_h", "open_p_a", "open_h", "open_d", "open_a"]] = np.nan
    return m.reset_index()


# =============================================================================
# Team-match rows and familiarity features
# =============================================================================
def team_rows(fx: pd.DataFrame) -> pd.DataFrame:
    fx = fx.dropna(subset=["home_team_id", "away_team_id", "home_goals", "away_goals"])
    base = ["fixture_id", "league_id", "league", "season_id", "season", "starting_at"]
    h = fx[base].assign(team_id=fx.home_team_id, opp_id=fx.away_team_id, team=fx.home_team, opp=fx.away_team,
                        is_home=1, gf=fx.home_goals, ga=fx.away_goals, pos=fx.home_position, opp_pos=fx.away_position)
    a = fx[base].assign(team_id=fx.away_team_id, opp_id=fx.home_team_id, team=fx.away_team, opp=fx.home_team,
                        is_home=0, gf=fx.away_goals, ga=fx.home_goals, pos=fx.away_position, opp_pos=fx.home_position)
    r = pd.concat([h, a], ignore_index=True)
    r[["team_id", "opp_id"]] = r[["team_id", "opp_id"]].astype("int64")
    r["win"] = (r.gf > r.ga).astype(int)
    r["draw"] = (r.gf == r.ga).astype(int)
    r["points"] = 3 * r.win + r.draw
    r["date"] = r.starting_at.dt.tz_convert(None).dt.normalize()
    return r


def main_coach(fc: pd.DataFrame) -> pd.DataFrame:
    """One coach per (fixture, team): if several are listed, the one who has the
    most matches for that team (the head coach rather than an assistant)."""
    fc = fc.dropna(subset=["team_id", "coach_id"])
    fc[["team_id", "coach_id"]] = fc[["team_id", "coach_id"]].astype("int64")
    n = fc.groupby(["team_id", "coach_id"]).size().rename("n_for_team").reset_index()
    fc = fc.merge(n, on=["team_id", "coach_id"]).sort_values("n_for_team", ascending=False)
    fc = fc.drop_duplicates(["fixture_id", "team_id"])
    fc["date"] = fc.starting_at.dt.tz_convert(None).dt.normalize()
    return fc[["fixture_id", "team_id", "coach_id", "date"]]


def club_familiarity(rows: pd.DataFrame, mc: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """For each row (coach c of team T vs opp O on date d): last date c coached O
    before d, and how many matches he took charge of O."""
    key = rows[["row_id", "coach_id", "opp_id", "date"]].dropna(subset=["coach_id"])
    hist = mc.rename(columns={"team_id": "opp_id", "date": "h_date"})[["coach_id", "opp_id", "h_date"]]
    m = key.merge(hist, on=["coach_id", "opp_id"])
    m = m[m.h_date < m.date]
    ev = m.groupby("row_id").agg(club_last=("h_date", "max"), club_matches=("h_date", "size"))

    # Spell evidence from coach_teams (ended before this match)
    sp = spells.dropna(subset=["coach_id", "team_id", "start_date"]).copy()
    if len(sp):
        sp["start"] = sp.start_date.dt.tz_convert(None)
        sp["end"] = sp.end_date.dt.tz_convert(None)
        sp = sp[sp.temporary.fillna(False).astype(bool) == False]  # noqa: E712 - interim spells excluded
        sp = sp.rename(columns={"team_id": "opp_id"})[["coach_id", "opp_id", "start", "end"]]
        sp[["coach_id", "opp_id"]] = sp[["coach_id", "opp_id"]].astype("int64")
        s = key.merge(sp, on=["coach_id", "opp_id"])
        s = s[s.end.notna() & (s.end < s.date)]
        s["est_matches"] = ((s.end - s.start).dt.days / 7).clip(lower=0).round()
        se = s.groupby("row_id").agg(spell_last=("end", "max"), spell_est=("est_matches", "sum"))
    else:
        se = pd.DataFrame(columns=["spell_last", "spell_est"])
    out = ev.join(se, how="outer")
    out["club_last"] = out[["club_last", "spell_last"]].max(axis=1)
    out["club_matches"] = out.club_matches.fillna(0).combine(out.spell_est.fillna(0), max)
    return out[["club_last", "club_matches"]]


def player_familiarity(rows: pd.DataFrame, mc: pd.DataFrame, apps: pd.DataFrame, args) -> pd.DataFrame:
    """n_former: opponent players in this match with >= min_apps appearances under
    our coach within the look-back window (any club)."""
    apps = apps.dropna(subset=["player_id", "team_id"])
    apps[["player_id", "team_id"]] = apps[["player_id", "team_id"]].astype("int64")
    played = (apps.started.fillna(0) == 1) | (apps.minutes.fillna(0) > 0)
    exp = apps[played].merge(mc, on=["fixture_id", "team_id"])[["player_id", "coach_id", "date"]]

    # opposition players in the target match
    sel = apps.started.fillna(0) == 1
    if args.include_subs:
        sel = sel | (apps.minutes.fillna(0) > 0)
    opp_pl = apps[sel][["fixture_id", "team_id", "player_id"]].rename(columns={"team_id": "opp_id"})
    tgt = rows[["row_id", "fixture_id", "opp_id", "coach_id", "date"]].dropna(subset=["coach_id"])
    tgt = tgt.merge(opp_pl, on=["fixture_id", "opp_id"])
    n_lineup = tgt.groupby("row_id").size().rename("opp_lineup_n")

    m = tgt[["row_id", "player_id", "coach_id", "date"]].merge(
        exp.rename(columns={"date": "e_date"}), on=["player_id", "coach_id"])
    win = pd.Timedelta(days=args.player_window_days)
    m = m[(m.e_date < m.date) & (m.e_date >= m.date - win)]
    per = m.groupby(["row_id", "player_id"]).size()
    per = per[per >= args.min_apps].reset_index(name="apps_under_coach")
    out = per.groupby("row_id").agg(n_former=("player_id", "size"), former_apps=("apps_under_coach", "sum"))
    return out.join(n_lineup, how="outer")


def build_dataset(d: dict[str, pd.DataFrame], args) -> pd.DataFrame:
    rows = team_rows(d["fixtures"])
    rows["row_id"] = np.arange(len(rows))
    log(f"team-match rows: {len(rows):,}")

    mc = main_coach(d["fixture_coaches"])
    rows = rows.merge(mc[["fixture_id", "team_id", "coach_id"]], on=["fixture_id", "team_id"], how="left")

    # tenure: matches in charge of this team before this match (new-coach bounce control)
    mc_sorted = mc.sort_values("date")
    mc_sorted["tenure"] = mc_sorted.groupby(["coach_id", "team_id"]).cumcount()
    rows = rows.merge(mc_sorted[["fixture_id", "team_id", "tenure"]], on=["fixture_id", "team_id"], how="left")

    # rest days (all competitions)
    td = d["team_dates"].dropna()
    if len(td):
        td["date"] = td.starting_at.dt.tz_convert(None).dt.normalize()
        td = td[["team_id", "date"]].drop_duplicates().sort_values(["team_id", "date"])
        td["rest_days"] = td.groupby("team_id").date.diff().dt.days
        rows = rows.merge(td, on=["team_id", "date"], how="left")
    else:
        rows["rest_days"] = np.nan

    log("club familiarity ...")
    cf = club_familiarity(rows, mc, d["coach_spells"])
    rows = rows.merge(cf, left_on="row_id", right_index=True, how="left")
    rows["club_days"] = (rows.date - rows.club_last).dt.days
    log("player familiarity ...")
    pf = player_familiarity(rows, mc, d["appearances"], args)
    rows = rows.merge(pf, left_on="row_id", right_index=True, how="left")
    rows["n_former"] = rows.n_former.fillna(0).astype(int)
    rows["former_apps"] = rows.former_apps.fillna(0)
    rows["has_coach"] = rows.coach_id.notna()
    rows["has_lineup"] = rows.opp_lineup_n.fillna(0) > 0

    rows["club_recent"] = ((rows.club_days <= args.club_window_days)
                           & (rows.club_matches >= args.min_spell_matches)).astype(int)
    rows["players_fam"] = (rows.n_former >= args.player_threshold).astype(int)
    rows["any_fam"] = ((rows.club_recent == 1) | (rows.players_fam == 1)).astype(int)

    # mirror: the opposition coach's familiarity with us
    mir = rows[["fixture_id", "team_id", "club_recent", "n_former", "players_fam", "any_fam", "tenure",
                "rest_days", "coach_id", "club_days"]]
    mir = mir.rename(columns={c: "opp_" + c for c in mir.columns if c not in ("fixture_id",)})
    mir = mir.rename(columns={"opp_team_id": "opp_id"})
    rows = rows.merge(mir, on=["fixture_id", "opp_id"], how="left")
    rows["net_fam"] = ((rows.any_fam == 1) & (rows.opp_any_fam.fillna(0) == 0)).astype(int)

    # market
    mk = market_table(d["closing"], d["opening"], args.sharp_bookmaker)
    rows = rows.merge(mk, on="fixture_id", how="left")
    h = rows.is_home == 1
    pick = lambda hc, ac: np.where(h, rows[hc], rows[ac])  # noqa: E731
    rows["p_win"], rows["p_draw"], rows["p_loss"] = pick("p_h", "p_a"), rows.p_d, pick("p_a", "p_h")
    rows["odds_avg"], rows["odds_best"], rows["odds_sharp"] = pick("avg_h", "avg_a"), pick("best_h", "best_a"), pick("sharp_h", "sharp_a")
    rows["odds_open"], rows["p_open"] = pick("open_h", "open_a"), pick("open_p_h", "open_p_a")
    rows["odds_dnb"] = rows.odds_avg * (rows.avg_d - 1) / rows.avg_d  # synthetic draw-no-bet from 1X2
    pr = d["predictions"]
    if len(pr):
        rows = rows.merge(pr, on="fixture_id", how="left")
        s = rows[["pred_h", "pred_d", "pred_a"]].sum(axis=1).replace(0, np.nan)
        rows["pred_win"] = np.where(h, rows.pred_h, rows.pred_a) / s
    else:
        rows["pred_win"] = np.nan

    # performance (xG / shots on target) and opposition's
    pf = d["performance"]
    if len(pf):
        pf = pf.dropna(subset=["team_id"])
        pf["team_id"] = pf.team_id.astype("int64")
        rows = rows.merge(pf, on=["fixture_id", "team_id"], how="left")
        rows = rows.merge(pf.rename(columns={"team_id": "opp_id", "xg": "xg_against", "sot": "sot_against",
                                             "possession": "opp_possession", "dangerous_attacks": "opp_dangerous_attacks"}),
                          on=["fixture_id", "opp_id"], how="left")
    for c in ("xg", "xg_against", "sot", "sot_against"):
        if c not in rows:
            rows[c] = np.nan

    # outcomes vs market
    rows["exp_points"] = 3 * rows.p_win + rows.p_draw
    rows["ex_win"] = rows.win - rows.p_win
    rows["ex_points"] = rows.points - rows.exp_points
    for k in ("avg", "best", "sharp", "open"):
        rows[f"pl_{k}"] = np.where(rows.win == 1, rows[f"odds_{k}"] - 1, -1.0)
        rows.loc[rows[f"odds_{k}"].isna(), f"pl_{k}"] = np.nan
    rows["pl_dnb"] = np.select([rows.win == 1, rows.draw == 1], [rows.odds_dnb - 1, 0.0], -1.0)
    rows.loc[rows.odds_dnb.isna(), "pl_dnb"] = np.nan
    rows["line_move"] = rows.p_win - rows.p_open  # >0: market moved towards this team
    rows["ex_pred"] = rows.win - rows.pred_win

    # performance residual vs market: regress diff on market strength, keep residual
    strength = rows.p_win - rows.p_loss
    for col, a, b in (("xgd_resid", "xg", "xg_against"), ("sotd_resid", "sot", "sot_against")):
        diff = rows[a] - rows[b]
        ok = diff.notna() & strength.notna()
        rows[col] = np.nan
        if ok.sum() > 50:
            X = np.c_[np.ones(ok.sum()), strength[ok]]
            beta = np.linalg.lstsq(X, diff[ok].to_numpy(float), rcond=None)[0]
            rows.loc[ok, col] = diff[ok] - X @ beta

    names = d["coach_names"]
    if len(names):
        rows = rows.merge(names, on="coach_id", how="left")
    else:
        rows["coach"] = None
    rows["coach"] = rows.coach.fillna(rows.coach_id.map(lambda x: f"coach {int(x)}" if pd.notna(x) else "unknown"))

    ok = rows.p_win.notna() & rows.odds_avg.between(args.min_odds, args.max_odds) & rows.has_coach
    log(f"rows with odds + coach: {ok.sum():,} of {len(rows):,}")
    rows["in_sample"] = ok
    return rows


# =============================================================================
# Statistics
# =============================================================================
DEFS = {
    "club": ("Coached the opposition club", "club_recent"),
    "players": ("Coached many opposition players", "players_fam"),
    "any": ("Either signal", "any_fam"),
    "net": ("Either signal, opposition coach not familiar", "net_fam"),
}
PRICES = {
    "avg": "Average closing price",
    "best": "Best closing price",
    "sharp": "Sharp closing price",
    "open": "Average opening price",
    "dnb": "Draw-no-bet (average)",
}


class Boot:
    """Cluster bootstrap over coaches: ratio-of-sums statistics and treated-control differences."""

    def __init__(self, B: int, seed: int = 11):
        self.B, self.rng = B, np.random.default_rng(seed)

    def ratio(self, num: pd.Series, den: pd.Series, cl: pd.Series) -> tuple[float, float, float]:
        g = pd.DataFrame({"n": num.to_numpy(float), "d": den.to_numpy(float), "c": cl.to_numpy()}).groupby("c").sum()
        if len(g) < 2 or g.d.sum() == 0:
            return (np.nan, np.nan, np.nan)
        idx = self.rng.integers(0, len(g), (self.B, len(g)))
        n, dd = g.n.to_numpy()[idx].sum(axis=1), g.d.to_numpy()[idx].sum(axis=1)
        est = g.n.sum() / g.d.sum()
        bs = n / np.where(dd == 0, np.nan, dd)
        return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

    def diff(self, val: pd.Series, treat: pd.Series, cl: pd.Series) -> tuple[float, float, float, float]:
        df = pd.DataFrame({"v": val.to_numpy(float), "t": treat.to_numpy(int), "c": cl.to_numpy()}).dropna()
        g = df.assign(vt=df.v * df.t, nt=df.t, vc=df.v * (1 - df.t), nc=1 - df.t).groupby("c")[["vt", "nt", "vc", "nc"]].sum()
        if g.nt.sum() == 0 or g.nc.sum() == 0:
            return (np.nan,) * 4
        a = g.to_numpy()
        est = a[:, 0].sum() / a[:, 1].sum() - a[:, 2].sum() / a[:, 3].sum()
        idx = self.rng.integers(0, len(g), (self.B, len(g)))
        s = a[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
        bs = bs[np.isfinite(bs)]
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()) if len(bs) else np.nan
        return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), min(p, 1.0)


def summarise(df: pd.DataFrame, pl: str, boot: Boot) -> dict:
    d = df.dropna(subset=[pl])
    if len(d) == 0:
        return {"n": 0}
    roi, lo, hi = boot.ratio(d[pl], pd.Series(1.0, index=d.index), d.coach_id)
    ew, ewlo, ewhi = boot.ratio(d.ex_win, pd.Series(1.0, index=d.index), d.coach_id)
    return {
        "n": int(len(d)), "coaches": int(d.coach_id.nunique()),
        "win_rate": float(d.win.mean()), "p_win": float(d.p_win.mean()),
        "avg_odds": float(d[pl.replace("pl_", "odds_")].mean()) if pl.replace("pl_", "odds_") in d else None,
        "roi": roi, "roi_lo": lo, "roi_hi": hi,
        "ex_win": ew, "ex_win_lo": ewlo, "ex_win_hi": ewhi,
        "ex_points": float(d.ex_points.mean()),
        "xgd_resid": float(d.xgd_resid.mean()) if d.xgd_resid.notna().any() else None,
        "sotd_resid": float(d.sotd_resid.mean()) if d.sotd_resid.notna().any() else None,
        "line_move": float(d.line_move.mean()) if d.line_move.notna().any() else None,
        "ex_pred": float(d.ex_pred.mean()) if d.ex_pred.notna().any() else None,
        "profit": float(d[pl].sum()),
    }


def bucket_stats(df: pd.DataFrame, col: str, labels: list[str], pl: str, boot: Boot) -> list[dict]:
    out = []
    for lab in labels:
        d = df[(df[col] == lab)].dropna(subset=[pl])
        if len(d) == 0:
            out.append({"label": lab, "n": 0})
            continue
        roi, lo, hi = boot.ratio(d[pl], pd.Series(1.0, index=d.index), d.coach_id)
        ew, ewlo, ewhi = boot.ratio(d.ex_win, pd.Series(1.0, index=d.index), d.coach_id)
        out.append({"label": lab, "n": int(len(d)), "roi": roi, "roi_lo": lo, "roi_hi": hi,
                    "ex_win": ew, "ex_win_lo": ewlo, "ex_win_hi": ewhi, "win_rate": float(d.win.mean()),
                    "p_win": float(d.p_win.mean())})
    return out


def cluster_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, logistic: bool):
    """Logistic (IRLS) or OLS with coach-clustered sandwich standard errors."""
    b = np.zeros(X.shape[1])
    if logistic:
        for _ in range(100):
            p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
            w = p * (1 - p)
            H = X.T @ (X * w[:, None])
            step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), X.T @ (y - p))
            b += step
            if np.abs(step).max() < 1e-8:
                break
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        resid, bread = y - p, np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    else:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid, bread = y - X @ b, np.linalg.inv(X.T @ X)
    u = pd.DataFrame(X * resid[:, None]).groupby(groups).sum().to_numpy()
    G, N, K = len(u), len(y), X.shape[1]
    corr = G / (G - 1) * ((N - 1) / (N - K) if not logistic else 1)
    V = bread @ (u.T @ u) @ bread * corr
    se = np.sqrt(np.diag(V))
    return b, se


def regressions(df: pd.DataFrame) -> dict:
    d = df[df.in_sample].copy()
    d["mkt_logit"] = np.log(d.p_win.clip(1e-4, 1 - 1e-4) / (1 - d.p_win.clip(1e-4, 1 - 1e-4)))
    d["n_former_c"] = d.n_former.clip(upper=6)
    d["opp_n_former_c"] = d.opp_n_former.fillna(0).clip(upper=6)
    d["opp_club_recent"] = d.opp_club_recent.fillna(0)
    d["new_coach"] = (d.tenure.fillna(99) < 10).astype(int)
    d["opp_new_coach"] = (d.opp_tenure.fillna(99) < 10).astype(int)
    d["rest_diff"] = (d.rest_days.clip(upper=14) - d.opp_rest_days.clip(upper=14)).fillna(0)
    terms = [("mkt_logit", "Market log-odds (should be ~1)"), ("is_home", "Home"),
             ("club_recent", "Coached opposition club (recent)"), ("n_former_c", "Per former player in opp XI"),
             ("opp_club_recent", "Opp coach coached our club"), ("opp_n_former_c", "Per our player opp coach had"),
             ("new_coach", "Our coach new (<10 matches)"), ("opp_new_coach", "Opp coach new (<10 matches)"),
             ("rest_diff", "Rest-day difference (per day)")]
    cols = [t for t, _ in terms if d[t].std() > 0]
    X = np.c_[np.ones(len(d)), d[cols].to_numpy(float)]
    out = {"n": int(len(d)), "logit": [], "ols": []}
    if len(d) < 200:
        return out
    b, se = cluster_fit(X, d.win.to_numpy(float), d.coach_id.to_numpy(), logistic=True)
    lab = dict(terms)
    for i, c in enumerate(cols, start=1):
        out["logit"].append({"term": c, "label": lab[c], "coef": b[i], "lo": b[i] - 1.96 * se[i], "hi": b[i] + 1.96 * se[i],
                             "z": b[i] / se[i], "p": float(math.erfc(abs(b[i] / se[i]) / math.sqrt(2)))})
    cols2 = [c for c in cols if c != "mkt_logit"]
    X2 = np.c_[np.ones(len(d)), d[cols2].to_numpy(float)]
    b, se = cluster_fit(X2, d.ex_points.to_numpy(float), d.coach_id.to_numpy(), logistic=False)
    for i, c in enumerate(cols2, start=1):
        out["ols"].append({"term": c, "label": lab[c], "coef": b[i], "lo": b[i] - 1.96 * se[i], "hi": b[i] + 1.96 * se[i],
                           "p": float(math.erfc(abs(b[i] / se[i]) / math.sqrt(2)))})
    return out


def calibration(df: pd.DataFrame, flag: str) -> list[dict]:
    d = df[df.in_sample]
    bins = np.linspace(0, 1, 11)
    d = d.assign(bin=pd.cut(d.p_win, bins, include_lowest=True, labels=False))
    out = []
    for t in (1, 0):
        g = d[d[flag] == t].groupby("bin").agg(p=("p_win", "mean"), w=("win", "mean"), n=("win", "size"))
        g = g[g.n >= 15]
        out.append([{"p": float(r.p), "w": float(r.w), "n": int(r.n)} for r in g.itertuples()])
    return out


def cumulative(df: pd.DataFrame, flag: str, pl: str, base_roi: float) -> list[dict]:
    d = df[(df[flag] == 1)].dropna(subset=[pl]).sort_values("starting_at")
    if len(d) == 0:
        return []
    cum, stake = d[pl].cumsum().to_numpy(), np.arange(1, len(d) + 1)
    step = max(1, len(d) // 300)
    idx = list(range(0, len(d), step)) + [len(d) - 1]
    dates = d.starting_at.dt.strftime("%Y-%m-%d").to_numpy()
    return [{"i": int(stake[i]), "date": dates[i], "pl": float(cum[i]), "base": float(stake[i] * base_roi)} for i in idx]


def nan_to_none(o):
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def analyse(rows: pd.DataFrame, args) -> dict:
    boot = Boot(args.bootstrap)
    s = rows[rows.in_sample].copy()
    s["coach_id"] = s.coach_id.astype("int64")

    # buckets
    s["dose"] = pd.cut(s.n_former, [-1, 0, 1, 2, 3, 4, 99], labels=["0", "1", "2", "3", "4", "5+"]).astype(str)
    cd = s.club_days.where(s.club_matches >= args.min_spell_matches)
    s["recency"] = np.select(
        [cd.isna(), cd <= 180, cd <= 365, cd <= 730, cd <= 1460],
        ["Never", "≤ 6 months", "6–12 months", "1–2 years", "2–4 years"], "4+ years")
    s["fav"] = np.select([s.p_win >= 0.55, s.p_win >= 0.35], ["Favourite (≥55%)", "Middle (35–55%)"], "Underdog (<35%)")
    s["venue"] = np.where(s.is_home == 1, "Home", "Away")

    prices = [p for p in PRICES if s[f"pl_{p}"].notna().sum() >= 50]
    res = {"defs": {}, "dose": {}, "recency": {}}
    for p in prices:
        pl = f"pl_{p}"
        res["dose"][p] = bucket_stats(s, "dose", ["0", "1", "2", "3", "4", "5+"], pl, boot)
        res["recency"][p] = bucket_stats(s, "recency", ["≤ 6 months", "6–12 months", "1–2 years", "2–4 years", "4+ years", "Never"], pl, boot)

    for key, (label, flag) in DEFS.items():
        out = {"label": label, "prices": {}}
        for p in prices:
            pl = f"pl_{p}"
            all_ = summarise(s, pl, boot)
            tr = summarise(s[s[flag] == 1], pl, boot)
            ct = summarise(s[s[flag] == 0], pl, boot)
            droi = boot.diff(s[pl], s[flag], s.coach_id)
            dwin = boot.diff(s.ex_win, s[flag], s.coach_id)
            splits = []
            for dim in ("venue", "fav", "league", "season"):
                for val in sorted(s[dim].dropna().unique()):
                    sub = s[s[dim] == val]
                    t = sub[sub[flag] == 1].dropna(subset=[pl])
                    c = sub[sub[flag] == 0].dropna(subset=[pl])
                    if len(t) < 5:
                        continue
                    splits.append({"dim": dim, "value": str(val), "n": int(len(t)), "roi": float(t[pl].mean()),
                                   "ctrl_roi": float(c[pl].mean()) if len(c) else None,
                                   "ex_win": float(t.ex_win.mean()), "win_rate": float(t.win.mean()),
                                   "p_win": float(t.p_win.mean())})
            tc = s[s[flag] == 1].dropna(subset=[pl]).groupby(["coach_id", "coach"]).agg(
                n=(pl, "size"), roi=(pl, "mean"), ex_win=("ex_win", "mean"), wins=("win", "sum"),
                club=("club_recent", "sum"), players=("players_fam", "sum")).reset_index()
            tc = tc[tc.n >= 3].sort_values("n", ascending=False).head(15)
            out["prices"][p] = {
                "all": all_, "treated": tr, "control": ct,
                "diff_roi": dict(zip(["est", "lo", "hi", "p"], droi)),
                "diff_win": dict(zip(["est", "lo", "hi", "p"], dwin)),
                "splits": splits,
                "cum": cumulative(s, flag, pl, ct.get("roi", 0) if ct.get("n") else 0),
                "coaches": tc.drop(columns=["coach_id"]).to_dict("records"),
            }
        out["calibration"] = calibration(s, flag)
        ex = s[s[flag] == 1].sort_values("starting_at", ascending=False).head(40)
        out["examples"] = [{
            "date": r.starting_at.strftime("%Y-%m-%d"), "league": r.league, "team": r.team, "opp": r.opp,
            "coach": r.coach, "home": int(r.is_home), "club_days": None if pd.isna(r.club_days) else int(r.club_days),
            "n_former": int(r.n_former), "p_win": r.p_win, "odds": r.odds_avg,
            "score": f"{int(r.gf)}–{int(r.ga)}", "pl": r.pl_avg} for r in ex.itertuples()]
        res["defs"][key] = out
    res["regression"] = regressions(rows)
    res["prices"] = {p: PRICES[p] for p in prices}
    res["coverage"] = {
        "team_rows": int(len(rows)), "fixtures": int(rows.fixture_id.nunique()),
        "with_coach": int(rows.has_coach.sum()), "with_lineup": int(rows.has_lineup.sum()),
        "with_odds": int(rows.p_win.notna().sum()), "in_sample": int(rows.in_sample.sum()),
        "with_xg": int(rows.xg.notna().sum()), "with_open": int(rows.p_open.notna().sum()),
        "with_sharp": int(rows.odds_sharp.notna().sum()), "with_pred": int(rows.pred_win.notna().sum()),
        "leagues": int(rows.league_id.nunique()), "seasons": int(rows.season_id.nunique()),
        "coaches": int(rows.coach_id.nunique()),
        "date_min": rows.starting_at.min().strftime("%Y-%m-%d"), "date_max": rows.starting_at.max().strftime("%Y-%m-%d"),
        "avg_margin": float(rows.margin.mean()) if rows.margin.notna().any() else None,
        "n_club": int(s.club_recent.sum()), "n_players": int(s.players_fam.sum()),
        "n_any": int(s.any_fam.sum()), "n_net": int(s.net_fam.sum()),
    }
    res["params"] = {k: v for k, v in vars(args).items() if k not in ("dsn",)}
    res["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return nan_to_none(res)


# =============================================================================
# Dashboard
# =============================================================================
def render(results: dict, path: str, title: str) -> None:
    tpl = DASHBOARD_HTML.replace("__TITLE__", html.escape(title))
    tpl = tpl.replace("__DATA__", json.dumps(results, ensure_ascii=False).replace("</", "<\\/"))
    Path(path).write_text(tpl, encoding="utf-8")
    log(f"dashboard written: {path}")


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root{color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--ctrl:#a9a79f;--good:#006300;--bad:#d03b3b;--wash:rgba(42,120,214,.10)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--wash:rgba(57,135,229,.12)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--ctrl:#6b6a64;--good:#0ca30c;--bad:#e66767;--wash:rgba(57,135,229,.12)}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh;padding:28px clamp(16px,4vw,48px) 60px}
.wrap{max-width:1240px;margin:0 auto}
h1{font-size:24px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:15px;font-weight:600;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 20px;max-width:880px}
.note{color:var(--muted);font-size:12px;margin:2px 0 12px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:end;margin:0 0 20px;padding:12px 14px;background:var(--surface);border:1px solid var(--ring);border-radius:10px;position:sticky;top:8px;z-index:5}
.controls label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--ink2)}
select,button{font:inherit;color:var(--ink);background:var(--page);border:1px solid var(--ring);border-radius:7px;padding:6px 10px}
button{cursor:pointer}
.grid{display:grid;gap:16px}
.g4{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(480px,1fr))}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px}
.hero{display:flex;flex-wrap:wrap;gap:28px;align-items:flex-end}
.hero .big{font-size:52px;font-weight:650;line-height:1;letter-spacing:-.02em}
.hero .lbl{color:var(--ink2);font-size:13px;margin-top:6px}
.verdict{font-size:15px;max-width:620px;color:var(--ink)}
.tile .lbl{color:var(--ink2);font-size:12.5px}
.tile .val{font-size:26px;font-weight:620;margin:4px 0 2px}
.tile .ci{color:var(--muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin:4px 0 6px;flex-wrap:wrap}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
.ln{width:14px;height:2px;display:inline-block;border-radius:1px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font-family:inherit;font-size:11px;fill:var(--muted)}
svg .lab{fill:var(--ink2);font-size:11.5px}
svg .val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
td:first-child,th:first-child,td.l,th.l{text-align:left}
td{font-variant-numeric:tabular-nums}
.scroll{max-height:420px;overflow:auto}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);box-shadow:0 4px 18px rgba(0,0,0,.14);border-radius:8px;padding:8px 10px;font-size:12px;opacity:0;transition:opacity .08s;z-index:20;max-width:280px}
.tip b{font-weight:600}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:12.5px}
.section{margin:30px 0 10px}
.pill{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:99px;border:1px solid var(--ring);color:var(--ink2)}
.method li{margin:3px 0;color:var(--ink2)}
.hit{fill:transparent}
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">Teams whose coach recently managed the opposition club, or has coached many of the opposition's players, compared with the bookmakers' closing 1X2 prices. Flat 1-unit stakes on the team to win.</p>

  <div class="controls">
    <label>Signal<select id="def"></select></label>
    <label>Price used for returns<select id="price"></select></label>
    <label>Split by<select id="dim">
      <option value="venue">Home / away</option><option value="fav">Market expectation</option>
      <option value="league">League</option><option value="season">Season</option></select></label>
    <button id="theme" type="button">Theme</button>
    <span class="note" id="gen" style="margin-left:auto"></span>
  </div>

  <div class="card hero" id="hero"></div>
  <div class="grid g4" style="margin-top:16px" id="tiles"></div>

  <div class="section"><h2>Return by group</h2><p class="note">Return per unit staked, with 95% interval from resampling coaches. Everyone pays the bookmaker margin, so compare the signal with the control group, not with zero.</p></div>
  <div class="grid g2">
    <div class="card"><h2>Return on investment</h2><div class="legend" id="lg-roi"></div><div id="c-roi"></div></div>
    <div class="card"><h2>Wins above market expectation</h2><p class="note">Actual win rate minus the de-vigged market probability, in percentage points.</p><div id="c-exwin"></div></div>
  </div>

  <div class="section"><h2>Does more familiarity mean more edge?</h2></div>
  <div class="grid g2">
    <div class="card"><h2>By number of former players in the opposition line-up</h2><p class="note" id="dose-note"></p><div id="c-dose"></div></div>
    <div class="card"><h2>By time since the coach left the opposition club</h2><p class="note">Coaches who managed the opponent for at least the minimum number of matches.</p><div id="c-rec"></div></div>
  </div>

  <div class="section"><h2>Over time and calibration</h2></div>
  <div class="grid g2">
    <div class="card"><h2>Cumulative profit, signal bets</h2><div class="legend"><span class="key"><span class="ln" style="background:var(--s1)"></span>Signal bets</span><span class="key"><span class="ln" style="background:var(--ctrl)"></span>Expected at control-group return</span></div><div id="c-cum"></div></div>
    <div class="card"><h2>Market probability vs actual win rate</h2><div class="legend"><span class="key"><span class="sw" style="background:var(--s1)"></span>Signal</span><span class="key"><span class="sw" style="background:var(--ctrl)"></span>Control</span><span class="key"><span class="ln" style="background:var(--axis)"></span>Perfect calibration</span></div><div id="c-cal"></div></div>
  </div>

  <div class="section"><h2>Controlling for everything else</h2><p class="note">Logistic regression of winning on the market's log-odds plus familiarity and controls; coefficients are log-odds shifts (0 = the market already prices it). Coach-clustered 95% intervals.</p></div>
  <div class="grid g2">
    <div class="card"><h2>Win model: effect beyond the market price</h2><div id="c-forest"></div></div>
    <div class="card"><h2>Points above expectation (linear model)</h2><p class="note">Points won minus market-expected points (3×P(win) + P(draw)).</p><div id="c-forest2"></div></div>
  </div>

  <div class="section"><h2>Breakdowns</h2></div>
  <div class="grid g2">
    <div class="card"><h2>Signal bets by <span id="dimname"></span></h2><div class="scroll"><table id="t-split"></table></div></div>
    <div class="card"><h2>Coaches with the most signal matches</h2><div class="scroll"><table id="t-coach"></table></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h2>Most recent signal matches</h2><div class="scroll"><table id="t-ex"></table></div></div>

  <div class="section"><h2>Data and method</h2></div>
  <div class="grid g2">
    <div class="card"><h2>Coverage</h2><table id="t-cov"></table></div>
    <div class="card method"><h2>How the signals are built</h2><ul id="method"></ul></div>
  </div>
</div></div>
<div class="tip" id="tip"></div>
<script>
const R = __DATA__;
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const pct = (v, d=1) => v == null ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(d) + '%';
const pp = (v, d=1) => v == null ? '–' : (v >= 0 ? '+' : '') + (v*100).toFixed(d) + ' pp';
const num = (v, d=2) => v == null ? '–' : (+v).toFixed(d);
const int = v => v == null ? '–' : Math.round(v).toLocaleString();
const cls = v => v == null ? '' : v >= 0 ? 'pos' : 'neg';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tip = $('#tip');
function showTip(e, h){ tip.innerHTML = h; tip.style.opacity = 1; const w = tip.offsetWidth, x = e.clientX + 14 + w > innerWidth ? e.clientX - w - 14 : e.clientX + 14; tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px'; }
function hideTip(){ tip.style.opacity = 0; }
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent){ const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
function svg(host, w, h){ host.innerHTML = ''; return el('svg', {viewBox:`0 0 ${w} ${h}`, role:'img'}, host); }
function ticks(lo, hi, n=5){ const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0))); const step = [1,2,2.5,5,10].map(m=>m*mag).find(s => span/s <= n) || mag*10; const out=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) out.push(+v.toFixed(10)); return out; }
function text(p, x, y, s, a={}){ const t = el('text', Object.assign({x, y}, a), p); t.textContent = s; return t; }
// rounded data-end, square at baseline (horizontal bar)
function hbarPath(x0, x1, y, h){ const r = Math.min(4, Math.abs(x1-x0)/2, h/2); if (x1 >= x0) return `M${x0},${y}H${x1-r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1-r},${y+h}H${x0}Z`; return `M${x0},${y}H${x1+r}Q${x1},${y} ${x1},${y+r}V${y+h-r}Q${x1},${y+h} ${x1+r},${y+h}H${x0}Z`; }
function vbarPath(x, w, y0, y1){ const r = Math.min(4, Math.abs(y1-y0)/2, w/2); if (y1 <= y0) return `M${x},${y0}V${y1+r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1+r}V${y0}Z`; return `M${x},${y0}V${y1-r}Q${x},${y1} ${x+r},${y1}H${x+w-r}Q${x+w},${y1} ${x+w},${y1-r}V${y0}Z`; }

// ---- horizontal bars with CI (rows: {label, v, lo, hi, color, tip}) ----
function hbars(host, rows, fmt){
  const W = 560, rowH = 40, top = 6, left = 150, right = 70, H = top + rows.length*rowH + 26;
  const s = svg(host, W, H); const vals = rows.flatMap(r => [r.v, r.lo, r.hi, 0]).filter(v => v != null);
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi-lo)*0.08 || 0.01; lo -= pad; hi += pad;
  const x = v => left + (v - lo) / (hi - lo) * (W - left - right);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-22, stroke:css('--grid')}, s); text(s, x(t), H-6, fmt(t), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:top, y2:H-22, stroke:css('--axis')}, s);
  rows.forEach((r, i) => {
    const y = top + i*rowH + 8, h = 20;
    text(s, left - 10, y + h/2 + 4, r.label, {'text-anchor':'end', class:'lab'});
    if (r.v == null) return;
    el('path', {d: hbarPath(x(0), x(r.v), y, h), fill: r.color}, s);
    if (r.lo != null) { el('line', {x1:x(r.lo), x2:x(r.hi), y1:y+h/2, y2:y+h/2, stroke:css('--ink2'), 'stroke-width':1.5}, s);
      for (const e of [r.lo, r.hi]) el('line', {x1:x(e), x2:x(e), y1:y+h/2-5, y2:y+h/2+5, stroke:css('--ink2'), 'stroke-width':1.5}, s); }
    const vx = Math.max(x(r.v), r.hi != null ? x(r.hi) : 0, x(0)) + 6;
    text(s, vx, y + h/2 + 4, fmt(r.v), {class:'val'});
    const hit = el('rect', {x:left-140, y:y-6, width:W-left+140, height:h+12, class:'hit'}, s);
    hit.addEventListener('mousemove', e => showTip(e, r.tip)); hit.addEventListener('mouseleave', hideTip);
  });
}

// ---- columns with CI (rows: {label, v, lo, hi, n}) ----
function cols(host, rows, fmt, color, xTitle){
  const W = 560, H = 270, top = 14, bottom = 58, left = 56, right = 12;
  const s = svg(host, W, H); const vals = rows.flatMap(r => [r.v, r.lo, r.hi, 0]).filter(v => v != null && isFinite(v));
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi-lo)*0.08 || 0.01; lo -= pad; hi += pad;
  const y = v => top + (hi - v) / (hi - lo) * (H - top - bottom);
  const bw = (W - left - right) / rows.length, w = Math.min(24, bw*0.5);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-8, y(t)+4, fmt(t), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis')}, s);
  rows.forEach((r, i) => {
    const cx = left + bw*i + bw/2;
    text(s, cx, H - bottom + 18, r.label, {'text-anchor':'middle', class:'lab'});
    text(s, cx, H - bottom + 32, r.n ? `n=${r.n.toLocaleString()}` : 'n=0', {'text-anchor':'middle'});
    if (r.v == null || !r.n) return;
    el('path', {d: vbarPath(cx - w/2, w, y(0), y(r.v)), fill: r.small ? css('--ctrl') : color}, s);
    if (r.lo != null) { el('line', {x1:cx, x2:cx, y1:y(r.lo), y2:y(r.hi), stroke:css('--ink2'), 'stroke-width':1.5}, s);
      for (const e of [r.lo, r.hi]) el('line', {x1:cx-5, x2:cx+5, y1:y(e), y2:y(e), stroke:css('--ink2'), 'stroke-width':1.5}, s); }
    const hit = el('rect', {x:cx-bw/2, y:top, width:bw, height:H-top-bottom, class:'hit'}, s);
    hit.addEventListener('mousemove', e => showTip(e, r.tip)); hit.addEventListener('mouseleave', hideTip);
  });
  if (xTitle) text(s, left + (W-left-right)/2, H-2, xTitle, {'text-anchor':'middle'});
}

// ---- cumulative P/L line with crosshair ----
function cumChart(host, pts){
  const W = 560, H = 260, top = 14, bottom = 30, left = 52, right = 64;
  const s = svg(host, W, H);
  if (!pts.length) { text(s, W/2, H/2, 'No signal bets for this selection', {'text-anchor':'middle'}); return; }
  const n = pts[pts.length-1].i, vals = pts.flatMap(p => [p.pl, p.base, 0]);
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi-lo)*0.06 || 1; lo -= pad; hi += pad;
  const x = i => left + (i-1)/(Math.max(n-1,1))*(W-left-right), y = v => top + (hi-v)/(hi-lo)*(H-top-bottom);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s); text(s, left-8, y(t)+4, t.toLocaleString(), {'text-anchor':'end'}); }
  el('line', {x1:left, x2:W-right, y1:y(0), y2:y(0), stroke:css('--axis')}, s);
  for (const t of ticks(1, n, 5)) text(s, x(t), H-10, t.toLocaleString(), {'text-anchor':'middle'});
  text(s, W-right, H-10, 'bets', {'text-anchor':'start', dx:6});
  const path = k => pts.map((p, j) => (j?'L':'M') + x(p.i).toFixed(1) + ',' + y(p[k]).toFixed(1)).join('');
  el('path', {d:path('base'), fill:'none', stroke:css('--ctrl'), 'stroke-width':2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}, s);
  el('path', {d:path('pl'), fill:'none', stroke:css('--s1'), 'stroke-width':2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}, s);
  const last = pts[pts.length-1];
  el('circle', {cx:x(last.i), cy:y(last.pl), r:4, fill:css('--s1'), stroke:css('--surface'), 'stroke-width':2}, s);
  text(s, x(last.i)+8, y(last.pl)+4, (last.pl>=0?'+':'') + last.pl.toFixed(1) + 'u', {class:'val'});
  const cross = el('line', {y1:top, y2:H-bottom, stroke:css('--axis'), opacity:0}, s);
  const dot = el('circle', {r:4, fill:css('--s1'), stroke:css('--surface'), 'stroke-width':2, opacity:0}, s);
  const hit = el('rect', {x:left, y:top, width:W-left-right, height:H-top-bottom, class:'hit'}, s);
  hit.addEventListener('mousemove', e => {
    const b = s.getBoundingClientRect(), mx = (e.clientX - b.left) / b.width * W;
    const i = 1 + (mx-left)/(W-left-right)*(n-1); let best = pts[0]; for (const p of pts) if (Math.abs(p.i-i) < Math.abs(best.i-i)) best = p;
    cross.setAttribute('x1', x(best.i)); cross.setAttribute('x2', x(best.i)); cross.setAttribute('opacity', 1);
    dot.setAttribute('cx', x(best.i)); dot.setAttribute('cy', y(best.pl)); dot.setAttribute('opacity', 1);
    showTip(e, `<b>Bet ${best.i.toLocaleString()}</b> · ${best.date}<br>Profit: <b>${best.pl.toFixed(1)}u</b><br>At control return: ${best.base.toFixed(1)}u`);
  });
  hit.addEventListener('mouseleave', () => { hideTip(); cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); });
}

// ---- calibration ----
function calChart(host, sets){
  const W = 560, H = 280, top = 10, bottom = 34, left = 46, right = 14;
  const s = svg(host, W, H), x = v => left + v*(W-left-right), y = v => top + (1-v)*(H-top-bottom);
  for (const t of [0, .2, .4, .6, .8, 1]) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-bottom, stroke:css('--grid')}, s); el('line', {x1:left, x2:W-right, y1:y(t), y2:y(t), stroke:css('--grid')}, s);
    text(s, x(t), H-bottom+16, (t*100)+'%', {'text-anchor':'middle'}); text(s, left-8, y(t)+4, (t*100)+'%', {'text-anchor':'end'}); }
  text(s, left+(W-left-right)/2, H-2, 'Market win probability', {'text-anchor':'middle'});
  el('line', {x1:x(0), y1:y(0), x2:x(1), y2:y(1), stroke:css('--axis'), 'stroke-width':1.5}, s);
  [[sets[1], '--ctrl', 'Control'], [sets[0], '--s1', 'Signal']].forEach(([pts, c, name]) => {
    if (!pts || !pts.length) return;
    el('path', {d: pts.map((p,j)=>(j?'L':'M')+x(p.p)+','+y(p.w)).join(''), fill:'none', stroke:css(c), 'stroke-width':2, 'stroke-linejoin':'round'}, s);
    pts.forEach(p => { const r = Math.max(4, Math.min(9, Math.sqrt(p.n)/3));
      el('circle', {cx:x(p.p), cy:y(p.w), r, fill:css(c), stroke:css('--surface'), 'stroke-width':2}, s);
      const h = el('circle', {cx:x(p.p), cy:y(p.w), r:r+8, class:'hit'}, s);
      h.addEventListener('mousemove', e => showTip(e, `<b>${name}</b><br>Market: ${(p.p*100).toFixed(1)}%<br>Actual: ${(p.w*100).toFixed(1)}%<br>${p.n.toLocaleString()} matches`)); h.addEventListener('mouseleave', hideTip); });
  });
}

// ---- forest plot ----
function forest(host, rows, xlabel){
  if (!rows || !rows.length) { host.innerHTML = '<p class="note">Not enough data for the model.</p>'; return; }
  const W = 560, rowH = 30, top = 6, left = 210, right = 70, H = top + rows.length*rowH + 30;
  const s = svg(host, W, H), vals = rows.flatMap(r => [r.lo, r.hi, 0]);
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi-lo)*.06; lo -= pad; hi += pad;
  const x = v => left + (v-lo)/(hi-lo)*(W-left-right);
  for (const t of ticks(lo, hi, 5)) { el('line', {x1:x(t), x2:x(t), y1:top, y2:H-26, stroke:css('--grid')}, s); text(s, x(t), H-12, t.toFixed(2), {'text-anchor':'middle'}); }
  el('line', {x1:x(0), x2:x(0), y1:top, y2:H-26, stroke:css('--axis'), 'stroke-width':1.5}, s);
  text(s, left + (W-left-right)/2, H, xlabel, {'text-anchor':'middle'});
  rows.forEach((r, i) => {
    const cy = top + i*rowH + rowH/2, fam = /club_recent|n_former/.test(r.term) && !/^opp/.test(r.term);
    text(s, left-10, cy+4, r.label, {'text-anchor':'end', class:'lab'});
    const c = fam ? css('--s1') : css('--ctrl');
    el('line', {x1:x(r.lo), x2:x(r.hi), y1:cy, y2:cy, stroke:c, 'stroke-width':2, 'stroke-linecap':'round'}, s);
    el('circle', {cx:x(r.coef), cy, r:5, fill:c, stroke:css('--surface'), 'stroke-width':2}, s);
    text(s, x(Math.max(r.hi, 0))+8, cy+4, (r.coef>=0?'+':'') + r.coef.toFixed(3) + (r.p < .05 ? ' *' : ''), {class:'val'});
    const h = el('rect', {x:0, y:cy-rowH/2, width:W, height:rowH, class:'hit'}, s);
    h.addEventListener('mousemove', e => showTip(e, `<b>${esc(r.label)}</b><br>Estimate: ${r.coef.toFixed(3)}<br>95% CI: ${r.lo.toFixed(3)} to ${r.hi.toFixed(3)}<br>p = ${r.p.toFixed(3)}`)); h.addEventListener('mouseleave', hideTip);
  });
}

function table(t, head, rows){ t.innerHTML = '<thead><tr>' + head.map(h => `<th class="${h[1]||''}">${h[0]}</th>`).join('') + '</tr></thead><tbody>' + rows.map(r => '<tr>' + r.map((c, i) => `<td class="${head[i][1]||''}">${c}</td>`).join('') + '</tr>').join('') + '</tbody>'; }

// ---- controls ----
const defSel = $('#def'), priceSel = $('#price'), dimSel = $('#dim');
for (const [k, v] of Object.entries(R.defs)) defSel.add(new Option(v.label, k));
for (const [k, v] of Object.entries(R.prices)) priceSel.add(new Option(v, k));
defSel.value = 'any';
const DIMS = {venue:'home / away', fav:'market expectation', league:'league', season:'season'};
$('#theme').onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; r.dataset.theme = dark ? 'light' : 'dark'; draw(); };
$('#gen').textContent = 'Generated ' + R.generated;

function draw(){
  const D = R.defs[defSel.value], P = D.prices[priceSel.value], pr = priceSel.value, T = P.treated, C = P.control, A = P.all;
  const dr = P.diff_roi, dw = P.diff_win;
  const sig = dr.p != null && dr.p < 0.05;
  const verdict = !T.n ? 'No matches carry this signal with the current settings.' :
    `Backing the ${T.n.toLocaleString()} teams with this signal returned <b class="${cls(T.roi)}">${pct(T.roi)}</b> per bet, against <b>${pct(C.roi)}</b> for all other teams. ` +
    (sig ? `The ${pct(dr.est).replace('+','')} gap is unlikely to be chance (p = ${num(dr.p,3)}).` : `The gap of ${pct(dr.est)} could be chance (p = ${num(dr.p,2)}).`);
  $('#hero').innerHTML = `<div><div class="big ${cls(dr.est)}">${pct(dr.est)}</div><div class="lbl">Extra return per bet vs control · ${esc(R.prices[pr])}</div></div><div class="verdict">${verdict}<div class="note" style="margin-top:6px">95% interval for the gap: ${pct(dr.lo)} to ${pct(dr.hi)} · ${T.coaches ?? 0} coaches</div></div>`;
  const tiles = [
    ['Signal matches', int(T.n), `${((T.n||0)/(A.n||1)*100).toFixed(1)}% of ${int(A.n)} matches`],
    ['Return per bet, signal', `<span class="${cls(T.roi)}">${pct(T.roi)}</span>`, `95% CI ${pct(T.roi_lo)} to ${pct(T.roi_hi)}`],
    ['Wins above market', `<span class="${cls(T.ex_win)}">${pp(T.ex_win)}</span>`, `Won ${pct(T.win_rate).replace('+','')} vs ${pct(T.p_win).replace('+','')} implied · gap to control ${pp(dw.est)}`],
    ['Points above market', `<span class="${cls(T.ex_points)}">${T.ex_points==null?'–':(T.ex_points>=0?'+':'')+T.ex_points.toFixed(3)}</span>`, `per match · control ${C.ex_points==null?'–':(C.ex_points>=0?'+':'')+C.ex_points.toFixed(3)}`],
    ['xG difference vs market', T.xgd_resid==null?'–':`<span class="${cls(T.xgd_resid)}">${(T.xgd_resid>=0?'+':'')+T.xgd_resid.toFixed(3)}</span>`, 'per match, after allowing for market strength'],
    ['Line movement to team', T.line_move==null?'–':pp(T.line_move,2), `opening → closing · control ${C.line_move==null?'–':pp(C.line_move,2)}`],
    ['Vs Sportmonks model', T.ex_pred==null?'–':`<span class="${cls(T.ex_pred)}">${pp(T.ex_pred)}</span>`, `wins above predicted · control ${C.ex_pred==null?'–':pp(C.ex_pred)}`],
    ['Average price', num(T.avg_odds), `control ${num(C.avg_odds)}`],
  ];
  $('#tiles').innerHTML = tiles.map(t => `<div class="card tile"><div class="lbl">${t[0]}</div><div class="val">${t[1]}</div><div class="ci">${t[2]}</div></div>`).join('');

  const tipG = (name, g) => `<b>${name}</b><br>${int(g.n)} bets · ${g.coaches ?? '–'} coaches<br>Return: ${pct(g.roi)} (${pct(g.roi_lo)} to ${pct(g.roi_hi)})<br>Won ${pct(g.win_rate).replace('+','')} vs ${pct(g.p_win).replace('+','')} implied<br>Profit: ${num(g.profit,1)}u`;
  $('#lg-roi').innerHTML = `<span class="key"><span class="sw" style="background:var(--s1)"></span>Signal</span><span class="key"><span class="sw" style="background:var(--ctrl)"></span>Comparison groups</span>`;
  hbars($('#c-roi'), [
    {label:'Signal', v:T.roi, lo:T.roi_lo, hi:T.roi_hi, color:css('--s1'), tip:tipG('Signal', T)},
    {label:'No signal (control)', v:C.roi, lo:C.roi_lo, hi:C.roi_hi, color:css('--ctrl'), tip:tipG('Control', C)},
    {label:'All matches', v:A.roi, lo:A.roi_lo, hi:A.roi_hi, color:css('--ctrl'), tip:tipG('All matches', A)},
  ], v => pct(v));
  hbars($('#c-exwin'), [
    {label:'Signal', v:T.ex_win, lo:T.ex_win_lo, hi:T.ex_win_hi, color:css('--s1'), tip:`<b>Signal</b><br>${pp(T.ex_win)} (${pp(T.ex_win_lo)} to ${pp(T.ex_win_hi)})`},
    {label:'No signal (control)', v:C.ex_win, lo:C.ex_win_lo, hi:C.ex_win_hi, color:css('--ctrl'), tip:`<b>Control</b><br>${pp(C.ex_win)} (${pp(C.ex_win_lo)} to ${pp(C.ex_win_hi)})`},
  ], v => pp(v));

  const bucketTip = (r, name) => `<b>${name}</b><br>${int(r.n)} bets<br>Return: ${pct(r.roi)} (${pct(r.roi_lo)} to ${pct(r.roi_hi)})<br>Wins above market: ${pp(r.ex_win)}`;
  $('#dose-note').textContent = `Players who made at least ${R.params.min_apps} appearances under our coach in the previous ${R.params.player_window_days} days. Bars in grey have fewer than 100 bets.`;
  cols($('#c-dose'), R.dose[pr].map(r => ({label:r.label, v:r.roi, lo:r.roi_lo, hi:r.roi_hi, n:r.n, small:r.n<100, tip:bucketTip(r, r.label + ' former players')})), v => pct(v,0), css('--s1'), 'Former players of our coach in opposition line-up · return per bet');
  cols($('#c-rec'), R.recency[pr].map(r => ({label:r.label, v:r.roi, lo:r.roi_lo, hi:r.roi_hi, n:r.n, small:r.n<100, tip:bucketTip(r, r.label)})), v => pct(v,0), css('--s1'), 'Time since our coach last managed the opposition · return per bet');
  cumChart($('#c-cum'), P.cum);
  calChart($('#c-cal'), D.calibration);
  forest($('#c-forest'), R.regression.logit, 'Change in log-odds of winning');
  forest($('#c-forest2'), R.regression.ols, 'Points per match above market expectation');

  const dim = dimSel.value; $('#dimname').textContent = DIMS[dim];
  table($('#t-split'), [['Group','l'],['Bets'],['Won'],['Implied'],['Return'],['Control return']],
    P.splits.filter(r => r.dim === dim).map(r => [esc(r.value), int(r.n), pct(r.win_rate).replace('+',''), pct(r.p_win).replace('+',''), `<span class="${cls(r.roi)}">${pct(r.roi)}</span>`, pct(r.ctrl_roi)]));
  table($('#t-coach'), [['Coach','l'],['Bets'],['Wins'],['Club'],['Players'],['Wins vs mkt'],['Return']],
    P.coaches.map(r => [esc(r.coach), int(r.n), int(r.wins), int(r.club), int(r.players), pp(r.ex_win), `<span class="${cls(r.roi)}">${pct(r.roi)}</span>`]));
  table($('#t-ex'), [['Date','l'],['League','l'],['Team','l'],['Opponent','l'],['Coach','l'],['Days since at opp'],['Former players'],['Market P(win)'],['Price'],['Score'],['P/L']],
    D.examples.map(r => [r.date, esc(r.league), esc(r.team) + (r.home ? '' : ' <span class="pill">away</span>'), esc(r.opp), esc(r.coach), r.club_days==null?'–':int(r.club_days), r.n_former, pct(r.p_win).replace('+',''), num(r.odds), r.score, `<span class="${cls(r.pl)}">${r.pl==null?'–':(r.pl>=0?'+':'')+r.pl.toFixed(2)}</span>`]));
}

const C = R.coverage, Pm = R.params;
table($('#t-cov'), [['Item','l'],['Count']], [
  ['Finished matches', int(C.fixtures)], ['Team-match rows', int(C.team_rows)], ['Leagues · league-seasons', `${C.leagues} · ${C.seasons}`],
  ['Date range', `${C.date_min} → ${C.date_max}`], ['Rows with coach identified', int(C.with_coach)], ['Rows with opposition line-up', int(C.with_lineup)],
  ['Rows with 1X2 odds', int(C.with_odds)], ['Rows with sharp price', int(C.with_sharp)], ['Rows with opening price', int(C.with_open)],
  ['Rows with xG', int(C.with_xg)], ['Rows with Sportmonks prediction', int(C.with_pred)], ['Analysed (odds + coach)', int(C.in_sample)],
  ['Average bookmaker margin', C.avg_margin==null?'–':(C.avg_margin*100).toFixed(2)+'%'], ['Coaches', int(C.coaches)],
  ['Signal rows: club · players · either · net', `${int(C.n_club)} · ${int(C.n_players)} · ${int(C.n_any)} · ${int(C.n_net)}`]]);
$('#method').innerHTML = [
  `<b>Coached the opposition club</b>: our coach took charge of the opponent within the last ${Pm.club_window_days} days, for at least ${Pm.min_spell_matches} matches (from match-by-match coach data; non-interim coach_teams spells fill gaps).`,
  `<b>Coached many opposition players</b>: at least ${Pm.player_threshold} of the opponent's ${Pm.include_subs ? 'players used' : 'starters'} made ${Pm.min_apps}+ appearances under our coach in the previous ${Pm.player_window_days} days, at any club.`,
  `<b>Net</b>: either signal applies to us and neither applies to the opposition coach.`,
  `<b>Market</b>: closing 1X2 prices from every bookmaker in odds.prematch_odds and odds.premium_odds; each book is de-vigged proportionally, then averaged. Result settles at 90 minutes.`,
  `<b>Returns</b>: 1 unit on the team to win at the chosen price. Draw-no-bet price is derived from the 1X2 prices. Intervals and p-values resample coaches (${Pm.bootstrap.toLocaleString()} draws), so one coach's run cannot dominate.`,
  `<b>Model</b>: logistic regression of the win on market log-odds plus both coaches' familiarity, home, new-coach (<10 matches) and rest days, with coach-clustered errors.`,
  `<b>Odds filter</b>: prices between ${Pm.min_odds} and ${Pm.max_odds}.`].map(s => `<li>${s}</li>`).join('');

for (const s of [defSel, priceSel, dimSel]) s.onchange = draw;
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
draw();
</script>
</body>
</html>
"""


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", help="PostgreSQL connection string. Default: DATABASE_URL, else load_league.DATABASE from the same folder")
    ap.add_argument("--out", default="coach_familiarity_dashboard.html")
    ap.add_argument("--export-csv", help="write the team-match dataset to this CSV")
    ap.add_argument("--title", default="Does inside knowledge of the opponent beat the market?")
    ap.add_argument("--leagues", help="comma-separated league_ids to analyse (history still uses all)")
    ap.add_argument("--seasons", help="comma-separated season_ids to analyse")
    ap.add_argument("--club-window-days", type=int, default=730)
    ap.add_argument("--min-spell-matches", type=int, default=5)
    ap.add_argument("--player-window-days", type=int, default=730)
    ap.add_argument("--min-apps", type=int, default=5)
    ap.add_argument("--player-threshold", type=int, default=3)
    ap.add_argument("--include-subs", action="store_true", help="count opposition subs who played, not just starters")
    ap.add_argument("--market-id", type=int, default=1, help="Sportmonks 1X2 market id (Fulltime Result = 1)")
    ap.add_argument("--market-name", default="FULLTIME_RESULT")
    ap.add_argument("--sharp-bookmaker", default="Pinnacle")
    ap.add_argument("--min-odds", type=float, default=1.01)
    ap.add_argument("--max-odds", type=float, default=15.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cache", help="directory to cache query results between runs")
    ap.add_argument("--refresh", action="store_true", help="ignore cached query results")
    args = ap.parse_args(argv)

    db = DB(args.dsn)
    data = extract(db, args)
    if data["fixtures"].empty:
        sys.exit("No finished fixtures found - check --dsn / --leagues / --seasons.")
    rows = build_dataset(data, args)
    if args.export_csv:
        keep = [c for c in rows.columns if c not in ("starting_at",)]
        rows[keep].to_csv(args.export_csv, index=False)
        log(f"dataset written: {args.export_csv}")
    log("analysing ...")
    res = analyse(rows, args)
    render(res, args.out, args.title)
    a = res["defs"]["any"]["prices"]["avg"]
    log(f"either signal: {a['treated']['n']} bets, ROI {a['treated']['roi']:+.3%} vs control "
        f"{a['control']['roi']:+.3%} (gap {a['diff_roi']['est']:+.3%}, p={a['diff_roi']['p']})")


if __name__ == "__main__":
    main()
