#!/usr/bin/env python3
"""
FOUL MAPS - where each team commits and wins fouls, and where each referee
gives them, compared with the league average.

Uses data already in the football database (no API calls):
  * text commentary   -> every foul: who committed it, who was fouled, and the
                         zone of the free kick ("offensive position",
                         "in defence", "right wing", "left flank" ...)
  * ball coordinates  -> an estimated exact spot for fouls and cards: while play
                         is stopped the ball sits still at the free-kick spot
  * goals / zones     -> which way each team was attacking in each half

Writes two tables you can query yourself:
    analysis.fouls           one row per foul
    analysis.card_locations  one row per card with its estimated spot
and an HTML report (foul_maps.html) with pitch maps per team and referee.

RUN (same folder as load_league.py, which it reads the database settings from)
    python3 foul_maps.py               # process new matches, then build the report
    python3 foul_maps.py --rebuild     # re-process every match
    python3 foul_maps.py --report-only # just rebuild the report
    python3 foul_maps.py --leagues 8 564   # choose leagues (default: top 5 European leagues)
    python3 foul_maps.py --all-leagues     # everything
Then open foul_maps.html in your browser. Menus: competition, season, view
(teams / referees / players), the team/referee/player in focus, a referee (or
team) filter, score state, players on the pitch, home/away and match period.

COORDINATES IN THE TABLES
  x, y run 0..1. x = 1 is the goal the team attacks.
  analysis.fouls:          seen from the team that COMMITTED the foul
  analysis.card_locations: seen from the team that received the card
  zone (analysis.fouls) is the commentary zone from the FOULED team's view.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

GRID_X, GRID_Y = 12, 8          # heat-map cells (length x width)
MIN_FOULS = 60                  # entities with fewer parsed fouls are listed as "low sample"
CHUNK = 150                     # matches processed per database round-trip
STILL = 0.012                   # ball moving less than this (share of pitch) = stationary
MIN_STOP_SECONDS = 8            # shortest stoppage counted
REPORT_FILE = "foul_maps.html"
TOP5 = [8, 564, 82, 384, 301]   # Premier League, La Liga, Bundesliga, Serie A, Ligue 1
VERSION = 2                     # bump to make already-processed matches be processed again

SETUP_SQL = """
CREATE TABLE IF NOT EXISTS analysis.fouls (
    foul_id                 text PRIMARY KEY,   -- commentary id of the "commits a foul" line
    fixture_id              bigint NOT NULL,
    league_id               bigint,
    season_id               bigint,
    referee_id              bigint,
    minute                  int,
    extra_minute            int,
    committed_by_team_id    bigint,
    committed_by_player     text,
    committed_by_player_id  bigint,
    fouled_team_id          bigint,
    fouled_player           text,
    fouled_player_id        bigint,
    fouled_team_is_home     boolean,
    zone                    text,      -- attacking / defensive / left / right, FOULED team's view
    zone_text               text,      -- the commentary line the zone came from
    x                       numeric,   -- 0..1, committing team attacking towards x = 1
    y                       numeric,
    card                    text       -- YELLOWCARD / REDCARD / YELLOWREDCARD for the fouler, same minute
);
CREATE INDEX IF NOT EXISTS ix_fouls_fixture ON analysis.fouls (fixture_id);
CREATE INDEX IF NOT EXISTS ix_fouls_league_team ON analysis.fouls (league_id, committed_by_team_id);
CREATE INDEX IF NOT EXISTS ix_fouls_league_fouled ON analysis.fouls (league_id, fouled_team_id);
CREATE INDEX IF NOT EXISTS ix_fouls_referee ON analysis.fouls (league_id, referee_id);

CREATE TABLE IF NOT EXISTS analysis.card_locations (
    event_id      bigint PRIMARY KEY,
    fixture_id    bigint NOT NULL,
    league_id     bigint,
    season_id     bigint,
    referee_id    bigint,
    team_id       bigint,
    team_is_home  boolean,
    player_id     bigint,
    player_name   text,
    card          text,
    minute        int,
    extra_minute  int,
    x             numeric,   -- 0..1, carded team attacking towards x = 1 (NULL if no spot found)
    y             numeric
);
CREATE INDEX IF NOT EXISTS ix_card_locations_fixture ON analysis.card_locations (fixture_id);
CREATE INDEX IF NOT EXISTS ix_card_locations_referee ON analysis.card_locations (league_id, referee_id);

-- game state at the moment of the foul / card (added in version 2)
ALTER TABLE analysis.fouls ADD COLUMN IF NOT EXISTS score_diff int;   -- committing team's goals minus opponent's
ALTER TABLE analysis.fouls ADD COLUMN IF NOT EXISTS men_diff int;     -- committing team's players minus opponent's
ALTER TABLE analysis.card_locations ADD COLUMN IF NOT EXISTS score_diff int;   -- carded team's view, before the card
ALTER TABLE analysis.card_locations ADD COLUMN IF NOT EXISTS men_diff int;
"""


# =============================================================== PARSING ===
FOUL_RE = re.compile(r"\bcommit(?:s|ted)?\s+a\s+foul\b|^\s*foul(?:ed)?\s+by\b", re.I)
WON_RE = re.compile(r"\b(?:won|wins|win|earned|earns|secured|secures|awarded|draws|drew|gets|got|"
                    r"wins?|winning)\b.*\bfree[\s-]?kick\b|\bfree[\s-]?kick\b.*\b(?:won|awarded|earned)\s+by\b",
                    re.I)
CARD_TYPES = ("YELLOWCARD", "REDCARD", "YELLOWREDCARD")


def zone_of(text):
    t = text.lower()
    if "penalty" in t:
        return "attacking"
    if re.search(r"offensive|attacking|in attack|final third|opposition half|opponent'?s half", t):
        return "attacking"
    if re.search(r"defensive|in defen[cs]e|own half|defending", t):
        return "defensive"
    if re.search(r"right[\s-]?(?:wing|flank|side)|on the right", t):
        return "right"
    if re.search(r"left[\s-]?(?:wing|flank|side)|on the left", t):
        return "left"
    return None


def player_of(text):
    t = text.strip().rstrip(".").strip()
    m = re.match(r"^foul(?:ed)?\s+by\s+(.+?)\s*(?:[-–(]|$)", t, re.I)
    if m:
        return m.group(1).strip()
    cuts = [t.find(s) for s in (" of ", " - ", " – ", " (", " commits", " committed", " has ",
                                " won", " wins", " earned", " secured")]
    cuts = [c for c in cuts if c > 0]
    return t[:min(cuts)].strip() if cuts else None


def team_of(text, teams):
    """teams: [(team_id, name, short_code)] for the two sides of the match."""
    low = text.lower()
    hits = [(len(name), tid) for tid, name, short in teams if name and name.lower() in low]
    if hits:
        return max(hits)[1]
    words = set(re.findall(r"[a-zà-ÿ0-9]+", low))
    best, best_score = None, 0
    for tid, name, short in teams:
        toks = set(re.findall(r"[a-zà-ÿ0-9]+", (name or "").lower())) - {"fc", "cf", "sc", "ac", "club", "de"}
        score = len(toks & words) / max(1, len(toks))
        if short and re.search(rf"\b{re.escape(short.lower())}\b", low):
            score = max(score, 0.6)
        if score > best_score:
            best, best_score = tid, score
    return best if best_score >= 0.5 else None


def norm_name(s):
    return re.sub(r"[^a-zà-ÿ ]", "", (s or "").lower()).strip()


def match_player(name, roster):
    """roster: list of (player_id, player_name, team_id)."""
    n = norm_name(name)
    if not n:
        return None
    exact = [pid for pid, pname, _ in roster if norm_name(pname) == n]
    if len(exact) == 1:
        return exact[0]
    last = n.split()[-1]
    cands = [pid for pid, pname, _ in roster if norm_name(pname).split()[-1:] == [last]]
    return cands[0] if len(set(cands)) == 1 else None


def parse_commentary(lines, teams, roster):
    """lines: [(commentary_id, minute, extra, sort_order, comment)] for one match.
    Returns foul dicts and stats about unparsed foul-like lines."""
    order = sorted(lines, key=lambda r: (r[1] or 0, r[2] or 0, r[3] if r[3] is not None else 0, str(r[0])))
    fouls, wons, unparsed = [], [], []
    for pos, (cid, minute, extra, _, text) in enumerate(order):
        if not text:
            continue
        if FOUL_RE.search(text):
            fouls.append({"pos": pos, "id": str(cid), "minute": minute, "extra": extra or 0,
                          "team": team_of(text, teams), "player": player_of(text)})
        elif WON_RE.search(text):
            wons.append({"pos": pos, "minute": minute, "extra": extra or 0, "team": team_of(text, teams),
                         "player": player_of(text), "zone": zone_of(text), "text": text, "used": False})
        elif re.search(r"\bfoul", text, re.I):
            unparsed.append(text)

    for f in fouls:
        best = None
        for w in wons:
            if w["used"] or abs((w["minute"] or 0) - (f["minute"] or 0)) > 1 or abs(w["pos"] - f["pos"]) > 3:
                continue
            if f["team"] and w["team"] and f["team"] == w["team"]:
                continue
            if best is None or abs(w["pos"] - f["pos"]) < abs(best["pos"] - f["pos"]):
                best = w
        team_ids = [t[0] for t in teams]
        if best:
            best["used"] = True
            f["fouled_team"] = best["team"] or next((t for t in team_ids if t != f["team"]), None)
            f["fouled_player"], f["zone"], f["zone_text"] = best["player"], best["zone"], best["text"]
        else:
            f["fouled_team"] = next((t for t in team_ids if t != f["team"]), None) if f["team"] else None
            f["fouled_player"] = f["zone"] = f["zone_text"] = None
        if not f["team"] and f["fouled_team"]:
            f["team"] = next((t for t in team_ids if t != f["fouled_team"]), None)
        f["player_id"] = match_player(f["player"], [r for r in roster if r[2] == f["team"]])
        f["fouled_player_id"] = match_player(f["fouled_player"], [r for r in roster if r[2] == f["fouled_team"]])
    return fouls, unparsed


# ============================================================== LOCATIONS ===
def period_of_minute(minute):
    m = minute or 0
    return 1 if m <= 45 else 2 if m <= 90 else 3 if m <= 105 else 4


def build_stoppages(samples):
    """samples: [(period_index, abs_seconds, x, y)] sorted. Returns stoppages:
    (period, start, end, x, y) where the ball stayed (almost) still."""
    stops, run = [], []

    def close_run():
        if len(run) >= 2 and run[-1][1] - run[0][1] >= MIN_STOP_SECONDS:
            xs = [r[2] for r in run]
            ys = [r[3] for r in run]
            stops.append((run[0][0], run[0][1], run[-1][1], sum(xs) / len(xs), sum(ys) / len(ys)))

    for s in samples:
        if run and s[0] == run[0][0] and math.hypot(s[2] - run[0][2], s[3] - run[0][3]) <= STILL:
            run.append(s)
        else:
            close_run()
            run = [s]
    close_run()
    return stops


def window(minute, extra):
    m = (minute or 0) + (extra or 0)
    return (m - 1) * 60 - 20, m * 60 + 45


def candidates(stops, period, minute, extra):
    lo, hi = window(minute, extra)
    return [s for s in stops if s[0] == period and s[2] >= lo and s[1] <= hi]


def fit(stop, minute, extra):
    """0 when the stoppage starts inside the event's minute, else seconds away."""
    m = (minute or 0) + (extra or 0)
    lo, hi = (m - 1) * 60, m * 60
    return 0 if lo <= stop[1] < hi else min(abs(stop[1] - lo), abs(stop[1] - hi))


def assign(items, stops, ok=lambda item, stop: True):
    """Match events to stoppages one-to-one, best fits first.
    items: [(key, period, minute, extra)] -> {key: stop}"""
    pairs = []
    for key, period, minute, extra in items:
        for s in candidates(stops, period, minute, extra):
            if ok(key, s):
                pairs.append((fit(s, minute, extra), -(s[2] - s[1]), key, s))
    pairs.sort(key=lambda p: (p[0], p[1], str(p[2]), p[3]))
    out, used = {}, set()
    for _, _, key, s in pairs:
        if key not in out and s not in used:
            out[key] = s
            used.add(s)
    return out


def timeline(rows):
    """rows: [(period_id, sort_order, timer, x, y)] for one match ->
    sorted samples [(period_index, abs_seconds, x, y)] (handles clocks that
    restart at 00:00 or carry on from 45:00 in the second half)."""
    by_period = defaultdict(list)
    order_key = {}
    for pid, sort_order, timer, x, y in rows:
        if timer is None or x is None or y is None:
            continue
        try:
            mm, ss = str(timer).split(":")[:2]
            secs = int(mm) * 60 + int(ss)
        except ValueError:
            continue
        by_period[pid].append((secs, float(x), float(y)))
        order_key[pid] = (sort_order if sort_order is not None else 99, min(order_key.get(pid, (0, 1e9))[1], secs))
    periods = sorted(by_period, key=lambda p: order_key[p])
    out = []
    starts = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
    for idx, pid in enumerate(periods, 1):
        pts = sorted(by_period[pid])
        restart = idx > 1 and pts and pts[0][0] < starts.get(idx, 0) - 120   # clock began near 00:00
        offset = starts.get(idx, 0) if restart else 0
        out.extend((idx, s + offset, x, y) for s, x, y in pts)
    return out


def orientation(stops, goals, fouls, home_id, away_id):
    """Which way the HOME team attacks in each period: +1 (towards x=1) or -1.
    Votes from goals (ball resting near a goal) and from fouls whose free-kick
    zone says which half they were in."""
    score = defaultdict(float)
    for g in goals:                                # (period, minute, extra, team_id)
        cands = [s for s in candidates(stops, g[0], g[1], g[2]) if abs(s[3] - 0.5) >= 0.38]
        if cands:
            s = max(cands, key=lambda c: abs(c[3] - 0.5))
            d = 1 if s[3] > 0.5 else -1
            score[g[0]] += 3 * (d if g[3] == home_id else -d)
    for f in fouls:
        if f["zone"] not in ("attacking", "defensive") or not f["fouled_team"]:
            continue
        p = period_of_minute(f["minute"])
        cands = [s for s in candidates(stops, p, f["minute"], f["extra"]) if abs(s[3] - 0.5) >= 0.1]
        if not cands:
            continue
        s = max(cands, key=lambda c: c[2] - c[1])
        d = (1 if s[3] > 0.5 else -1) * (1 if f["zone"] == "attacking" else -1)   # fouled team's direction
        score[p] += d if f["fouled_team"] == home_id else -d
    dirs = {p: (1 if v > 0 else -1) for p, v in score.items() if v != 0}
    for p in (1, 2, 3, 4):                         # teams swap ends at half-time
        if p not in dirs:
            other = {1: 2, 2: 1, 3: 4, 4: 3}[p]
            if other in dirs:
                dirs[p] = -dirs[other]
    return dirs


def to_frame(x, y, d):
    """Rotate so the team attacking in direction d attacks towards x = 1."""
    return (x, y) if d == 1 else (1 - x, 1 - y)


def game_state(evts, home_id, away_id):
    """evts: [(minute, extra, team_id, developer_name)]. Returns at(team, minute, extra)
    -> (goal difference, player difference) for that team just BEFORE that moment."""
    goals, reds = [], []
    for minute, extra, tid, dev in evts:
        t = (minute or 0) * 100 + (extra or 0)
        if dev in ("GOAL", "PENALTY"):
            goals.append((t, tid))
        elif dev == "OWNGOAL":                       # counts for the other side
            goals.append((t, away_id if tid == home_id else home_id))
        elif dev in ("REDCARD", "YELLOWREDCARD"):
            reds.append((t, tid))

    def at(team, minute, extra):
        if team is None:
            return None, None
        t = (minute or 0) * 100 + (extra or 0)
        gd = sum(1 if tm == team else -1 for tt, tm in goals if tt < t)
        md = sum(1 if tm != team else -1 for tt, tm in reds if tt < t)
        return gd, md
    return at


# ============================================================ PROCESSING ===
def process(db, leagues=None, rebuild=False):
    played = [r[0] for r in db.query(
        "SELECT state_id FROM core.states WHERE developer_name = ANY(%s)", (["FT", "AET", "FT_PEN"],))]
    sql = ("SELECT DISTINCT c.fixture_id FROM football.fixture_commentaries c "
           "JOIN football.fixtures f ON f.fixture_id = c.fixture_id WHERE f.state_id = ANY(%s)")
    params = [played]
    if leagues:
        sql += " AND f.league_id = ANY(%s)"
        params.append(list(leagues))
    with_text = [r[0] for r in db.query(sql, tuple(params))]
    sync = {} if rebuild else db.get_sync("foul_map")
    done = {k for k, v in sync.items() if (v[2] or {}).get("v") == VERSION}
    todo = sorted(fx for fx in with_text if str(fx) not in done)
    print(f"Matches with commentary: {len(with_text):,}; already processed: {len(with_text) - len(todo):,}; "
          f"to process: {len(todo):,}")
    stats = Counter()
    unparsed_examples = Counter()
    started = time.time()

    for start in range(0, len(todo), CHUNK):
        ids = todo[start:start + CHUNK]
        fx = {r[0]: r[1:] for r in db.query(
            "SELECT f.fixture_id, f.league_id, f.season_id, f.home_team_id, f.away_team_id, mr.referee_id "
            "FROM football.fixtures f LEFT JOIN analysis.main_referee mr ON mr.fixture_id = f.fixture_id "
            "WHERE f.fixture_id = ANY(%s)", (ids,))}
        teams = defaultdict(list)
        for fid, tid, name, short in db.query(
                "SELECT p.fixture_id, p.team_id, t.name, t.short_code FROM football.fixture_participants p "
                "JOIN football.teams t ON t.team_id = p.team_id WHERE p.fixture_id = ANY(%s)", (ids,)):
            teams[fid].append((tid, name, short))
        roster = defaultdict(list)
        for fid, pid, pname, tid in db.query(
                "SELECT fixture_id, player_id, player_name, team_id FROM football.fixture_lineups "
                "WHERE fixture_id = ANY(%s)", (ids,)):
            roster[fid].append((pid, pname, tid))
        comments = defaultdict(list)
        for fid, cid, minute, extra, order, text in db.query(
                "SELECT fixture_id, commentary_id, minute, extra_minute, sort_order, comment "
                "FROM football.fixture_commentaries WHERE fixture_id = ANY(%s) "
                "AND comment ~* '(foul|free.?kick)'", (ids,)):
            comments[fid].append((cid, minute, extra, order, text))
        events = defaultdict(list)
        for fid, eid, minute, extra, tid, pid, pname, dev in db.query(
                "SELECT e.fixture_id, e.event_id, e.minute, e.extra_minute, e.team_id, e.player_id, "
                "e.player_name, t.developer_name FROM football.fixture_events e "
                "JOIN core.types t ON t.type_id = e.type_id WHERE e.fixture_id = ANY(%s) "
                "AND e.rescinded IS NOT TRUE AND t.developer_name = ANY(%s)",
                (ids, list(CARD_TYPES) + ["GOAL", "PENALTY", "OWNGOAL"])):
            events[fid].append((eid, minute, extra, tid, pid, pname, dev))
        balls = defaultdict(list)
        for fid, pid, sort_order, timer, x, y in db.query(
                "SELECT b.fixture_id, b.period_id, p.sort_order, b.timer, b.x, b.y "
                "FROM football.fixture_ball_coordinates b "
                "LEFT JOIN football.fixture_periods p ON p.period_id = b.period_id "
                "WHERE b.fixture_id = ANY(%s)", (ids,)):
            balls[fid].append((pid, sort_order, timer, x, y))

        foul_rows, card_rows = [], []
        for fid in ids:
            if fid not in fx:
                continue
            league_id, season_id, home_id, away_id, referee_id = fx[fid]
            fouls, unparsed = parse_commentary(comments.get(fid, []), teams.get(fid, []), roster.get(fid, []))
            for u in unparsed:
                unparsed_examples[re.sub(r"[A-ZÀ-Ý][\wÀ-ÿ'.-]+", "X", u)[:90]] += 1
            stats["fouls"] += len(fouls)
            stats["unparsed"] += len(unparsed)
            stats["with_zone"] += sum(1 for f in fouls if f["zone"])

            stops = build_stoppages(timeline(balls.get(fid, [])))
            goals = [(period_of_minute(m), m, ex, t) for _, m, ex, t, _, _, dev in events.get(fid, [])
                     if dev in ("GOAL", "PENALTY")]
            state = game_state([(m, ex, t, dev) for _, m, ex, t, _, _, dev in events.get(fid, [])],
                               home_id, away_id)
            dirs = orientation(stops, goals, fouls, home_id, away_id) if stops else {}
            if stops:
                stats["matches_ball"] += 1
                stats["matches_oriented"] += bool(dirs)

            def team_dir(team_id, period):
                d = dirs.get(period)
                if d is None or team_id is None:
                    return None
                return d if team_id == home_id else -d

            cards = [e for e in events.get(fid, []) if e[6] in CARD_TYPES]

            def zone_ok(i, stop):
                f = fouls[i]
                d = team_dir(f["fouled_team"], period_of_minute(f["minute"]))
                if d is None:
                    return False
                if f["zone"] in ("attacking", "defensive"):
                    return (to_frame(stop[3], stop[4], d)[0] > 0.5) == (f["zone"] == "attacking")
                return True

            spots = assign([(i, period_of_minute(f["minute"]), f["minute"], f["extra"])
                            for i, f in enumerate(fouls)], stops, zone_ok) if dirs else {}

            # each card goes to one foul: same player, nearest minute
            card_of = {}
            for e in cards:
                options = [(abs((e[1] or 0) - (f["minute"] or 0)), i) for i, f in enumerate(fouls)
                           if e[4] and e[4] == f["player_id"] and abs((e[1] or 0) - (f["minute"] or 0)) <= 1
                           and i not in card_of]
                if options:
                    card_of[min(options)[1]] = e[6]

            for i, f in enumerate(fouls):
                x = y = None
                if i in spots:
                    s_ = spots[i]
                    d_fouled = team_dir(f["fouled_team"], period_of_minute(f["minute"]))
                    x, y = to_frame(s_[3], s_[4], -d_fouled)            # committing team's view
                    stats["located"] += 1
                card = card_of.get(i)
                foul_rows.append({
                    "foul_id": f["id"], "fixture_id": fid, "league_id": league_id, "season_id": season_id,
                    "referee_id": referee_id, "minute": f["minute"], "extra_minute": f["extra"],
                    "committed_by_team_id": f["team"], "committed_by_player": f["player"],
                    "committed_by_player_id": f["player_id"], "fouled_team_id": f["fouled_team"],
                    "fouled_player": f["fouled_player"], "fouled_player_id": f["fouled_player_id"],
                    "fouled_team_is_home": (f["fouled_team"] == home_id) if f["fouled_team"] else None,
                    "zone": f["zone"], "zone_text": f["zone_text"],
                    "x": round(x, 4) if x is not None else None, "y": round(y, 4) if y is not None else None,
                    "card": card,
                    **dict(zip(("score_diff", "men_diff"), state(f["team"], f["minute"], f["extra"])))})

            card_spots = assign([(e[0], period_of_minute(e[1]), e[1], e[2]) for e in cards], stops) if dirs else {}
            for eid, minute, extra, tid, pid, pname, dev in cards:
                p = period_of_minute(minute)
                d = team_dir(tid, p)
                x = y = None
                if d is not None and eid in card_spots:
                    s_ = card_spots[eid]
                    x, y = to_frame(s_[3], s_[4], d)
                    stats["cards_located"] += 1
                stats["cards"] += 1
                card_rows.append({
                    "event_id": eid, "fixture_id": fid, "league_id": league_id, "season_id": season_id,
                    "referee_id": referee_id, "team_id": tid, "team_is_home": tid == home_id if tid else None,
                    "player_id": pid, "player_name": pname, "card": dev, "minute": minute,
                    "extra_minute": extra, "x": round(x, 4) if x is not None else None,
                    "y": round(y, 4) if y is not None else None,
                    **dict(zip(("score_diff", "men_diff"), state(tid, minute, extra)))})

        # replace this batch's rows (they may be from an older version)
        db.execute("DELETE FROM analysis.fouls WHERE fixture_id = ANY(%s)", (ids,))
        db.execute("DELETE FROM analysis.card_locations WHERE fixture_id = ANY(%s)", (ids,))
        db.upsert("analysis.fouls", foul_rows)
        db.upsert("analysis.card_locations", card_rows)
        for fid in ids:
            db.set_sync("foul_map", fid, "ok", {"v": VERSION})
        db.commit()
        done_n = min(start + CHUNK, len(todo))
        rate = done_n / max(time.time() - started, 1e-9)
        print(f"  {done_n:,}/{len(todo):,} matches  ({stats['fouls']:,} fouls so far, "
              f"~{(len(todo) - done_n) / max(rate, 1e-9) / 60:.0f} min left)")

    if stats["fouls"]:
        print(f"\nParsed {stats['fouls']:,} fouls; {stats['with_zone'] / stats['fouls']:.0%} have a zone; "
              f"{stats['located'] / stats['fouls']:.0%} have an estimated spot")
    if stats["cards"]:
        print(f"Cards: {stats['cards']:,}; {stats['cards_located'] / stats['cards']:.0%} have an estimated spot")
    if stats["matches_ball"]:
        print(f"Direction of play worked out for {stats['matches_oriented']:,} of "
              f"{stats['matches_ball']:,} matches with ball data")
    if stats["unparsed"]:
        share = stats["unparsed"] / (stats["unparsed"] + stats["fouls"])
        print(f"\n{stats['unparsed']:,} lines mention a foul but weren't recognised ({share:.1%}). Most common:")
        for text, n in unparsed_examples.most_common(8):
            print(f"  {n:>6,}  {text}")
        if share > 0.05:
            print("  -> paste these to Claude to add the missing wording patterns.")


# ================================================================ REPORT ===
def build_report(db, leagues=None):
    """Exports every foul and card (compact columns) for the chosen leagues;
    the page itself does the counting, so any combination of filters works."""
    print("\nBuilding report...")
    lim = " WHERE league_id = ANY(%s)" if leagues else ""
    args = (list(leagues),) if leagues else None
    fouls = db.query(
        "SELECT league_id, season_id, fixture_id, minute, referee_id, committed_by_team_id, "
        "committed_by_player_id, fouled_team_id, fouled_player_id, fouled_team_is_home::int, zone, "
        "round(x * 100)::int, round(y * 100)::int, card, score_diff, men_diff, "
        "committed_by_player, fouled_player FROM analysis.fouls" + lim, args)
    cards = db.query(
        "SELECT league_id, season_id, fixture_id, minute, referee_id, team_id, player_id, team_is_home::int, "
        "card, round(x * 100)::int, round(y * 100)::int, score_diff, men_diff, player_name "
        "FROM analysis.card_locations" + lim, args)
    if not fouls:
        print("No fouls to report for these leagues yet - run without --report-only first.")
        return

    class Index:                                  # compact ids for the page
        def __init__(self):
            self.ids, self.pos = [], {}

        def __call__(self, key):
            if key is None:
                return -1
            if key not in self.pos:
                self.pos[key] = len(self.ids)
                self.ids.append(key)
            return self.pos[key]

    LG, SE, FX, RF, TM, PL = Index(), Index(), Index(), Index(), Index(), Index()
    ZONE = {"attacking": 1, "defensive": 2, "left": 3, "right": 4}
    CARD = {"YELLOWCARD": 1, "YELLOWREDCARD": 2, "REDCARD": 3}
    player_fallback = {}
    clamp = lambda v, lo, hi: -9 if v is None else max(lo, min(hi, int(v)))
    nz = lambda v: -1 if v is None else int(v)

    F = {k: [] for k in ("lg", "se", "fx", "mi", "rf", "ct", "cp", "ft", "fp", "fh", "zo", "x", "y", "cd", "sd", "md")}
    for (lid, sid, fid, minute, rid, ct, cp, ft, fp, fh, zone, x, y, card, sd, md, cpn, fpn) in fouls:
        for col, val in (("lg", LG(lid)), ("se", SE(sid)), ("fx", FX(fid)), ("mi", nz(minute)), ("rf", RF(rid)),
                         ("ct", TM(ct)), ("cp", PL(cp)), ("ft", TM(ft)), ("fp", PL(fp)), ("fh", nz(fh)),
                         ("zo", ZONE.get(zone, 0)), ("x", nz(x)), ("y", nz(y)), ("cd", CARD.get(card, 0)),
                         ("sd", clamp(sd, -3, 3)), ("md", clamp(md, -2, 2))):
            F[col].append(val)
        if cp and cpn:
            player_fallback.setdefault(cp, (cpn, ct))
        if fp and fpn:
            player_fallback.setdefault(fp, (fpn, ft))
    C = {k: [] for k in ("lg", "se", "fx", "mi", "rf", "tm", "pl", "hm", "cd", "x", "y", "sd", "md")}
    for (lid, sid, fid, minute, rid, tid, pid, hm, card, x, y, sd, md, pname) in cards:
        for col, val in (("lg", LG(lid)), ("se", SE(sid)), ("fx", FX(fid)), ("mi", nz(minute)), ("rf", RF(rid)),
                         ("tm", TM(tid)), ("pl", PL(pid)), ("hm", nz(hm)), ("cd", CARD.get(card, 0)),
                         ("x", nz(x)), ("y", nz(y)), ("sd", clamp(sd, -3, 3)), ("md", clamp(md, -2, 2))):
            C[col].append(val)
        if pid and pname:
            player_fallback.setdefault(pid, (pname, tid))

    def names(sql, ids):
        return dict(db.query(sql, (list(ids),))) if ids else {}
    league_names = {r[0]: ((r[2] + " · ") if r[2] else "") + (r[1] or "") for r in db.query(
        "SELECT l.league_id, l.name, c.name FROM football.leagues l "
        "LEFT JOIN core.countries c ON c.country_id = l.country_id WHERE l.league_id = ANY(%s)", (LG.ids,))}
    season_names = names("SELECT season_id, name FROM football.seasons WHERE season_id = ANY(%s)", SE.ids)
    season_order = names("SELECT season_id, coalesce(starting_at::text, name) FROM football.seasons "
                         "WHERE season_id = ANY(%s)", SE.ids)
    ref_names = names("SELECT referee_id, coalesce(display_name, common_name, name) FROM football.referees "
                      "WHERE referee_id = ANY(%s)", RF.ids)
    team_names = names("SELECT team_id, name FROM football.teams WHERE team_id = ANY(%s)", TM.ids)
    player_names = names("SELECT player_id, coalesce(display_name, common_name, name) FROM football.players "
                         "WHERE player_id = ANY(%s)", PL.ids)

    side = {(z, int(h)): n for z, h, n in db.query(
        "SELECT zone, ((1 - y) > 0.5)::int, count(*) FROM analysis.fouls "
        "WHERE zone IN ('left', 'right') AND y IS NOT NULL" + (" AND league_id = ANY(%s)" if leagues else "") +
        " GROUP BY 1, 2", args)}
    left_high = side.get(("left", 1), 0) + side.get(("right", 0), 0)
    left_low = side.get(("left", 0), 0) + side.get(("right", 1), 0)

    data = {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "grid": [GRID_X, GRID_Y],
                 "flipY": left_high >= left_low, "minFouls": MIN_FOULS},
        "leagues": [league_names.get(i, f"league {i}") for i in LG.ids],
        "seasons": [season_names.get(i, f"season {i}") for i in SE.ids],
        "seasonOrder": [str(season_order.get(i, "")) for i in SE.ids],
        "refs": [ref_names.get(i, f"referee {i}") for i in RF.ids],
        "teams": [team_names.get(i, f"team {i}") for i in TM.ids],
        "players": [player_names.get(i) or player_fallback.get(i, (f"player {i}",))[0] for i in PL.ids],
        "F": F, "C": C}
    path = os.path.abspath(REPORT_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"Report written: {path}  ({os.path.getsize(path) / 1e6:.1f} MB, {len(fouls):,} fouls, "
          f"{len(cards):,} cards, {len(LG.ids)} competitions)")
    print("Open it in your browser (on a Mac:  open foul_maps.html)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Foul Maps</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#677084;--line:#d9dde5;--pitch:#e9f2ea;--pline:#9fb3a3;
--accent:#2d6cdf;--hi:#c0392b;--lo:#2c6db5;--mid:#f4f4f2}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e6e9ef;--muted:#99a2b3;--line:#2c3340;
--pitch:#17261c;--pline:#3d5a45;--accent:#6aa0ff;--hi:#e0675a;--lo:#5b9bea;--mid:#2a2f38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 6px}h1{font-size:20px;margin:0 0 4px}header p{margin:0;color:var(--muted)}
.bar{display:flex;flex-wrap:wrap;gap:10px;padding:8px 20px;align-items:end}
.bar.filters{padding-top:0;border-bottom:1px solid var(--line);padding-bottom:14px}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
select,button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 10px;max-width:260px}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;align-self:end}.seg button{border:0;border-radius:0}
.seg button.on{background:var(--accent);color:#fff}
button.reset{align-self:end;color:var(--muted)}
main{padding:12px 20px 30px;display:grid;gap:14px}
#compare{color:var(--muted);font-size:13px}#compare b{color:var(--ink);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:20px}.stat span{color:var(--muted);font-size:12px}.stat i{font-style:normal;color:var(--muted);font-size:12px}
.maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.map{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.map h3{margin:0 0 2px;font-size:15px}.map .sub{color:var(--muted);font-size:12px;margin-bottom:8px}
svg{width:100%;height:auto;display:block}
svg text{paint-order:stroke;stroke:var(--card);stroke-width:.5px;stroke-linejoin:round}
.legend{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;margin-top:6px}
.legend .grad{height:10px;flex:0 0 160px;border-radius:5px}
.note{color:var(--muted);font-size:12px;margin:0}.warn{color:var(--hi)}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;display:none;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.15)}
</style></head><body>
<header><h1>Foul Maps</h1>
<p>Where fouls are committed, won and given. Every map is drawn with the team in focus attacking left&nbsp;→&nbsp;right, and compared with the rest of the competition under the same filters.</p></header>
<div class="bar">
 <label>Competition<select id="lg"></select></label>
 <label>Season<select id="se"></select></label>
 <div class="seg" id="view"><button data-v="teams" class="on">Teams</button><button data-v="refs">Referees</button><button data-v="players">Players</button></div>
 <label id="focusLabel">Team<select id="focus"></select></label>
 <div class="seg" id="mode"><button data-m="vs" class="on">vs comparison</button><button data-m="raw">raw share</button></div>
</div>
<div class="bar filters">
 <label id="crossLabel">Referee<select id="cross"></select></label>
 <label id="sdLabel">Score (team in focus)<select id="sd">
   <option value="all">Any score</option><option value="w">Winning</option><option value="w2">Winning by 2+</option>
   <option value="d">Drawing</option><option value="l">Losing</option><option value="l2">Losing by 2+</option></select></label>
 <label>Players on pitch<select id="md">
   <option value="all">Any</option><option value="even">Equal numbers</option><option value="up">A player up</option><option value="down">A player down</option></select></label>
 <label id="venueLabel">Venue<select id="ve"><option value="all">Home or away</option><option value="h">Home</option><option value="a">Away</option></select></label>
 <label>Match period<select id="pe">
   <option value="all">Whole match</option><option value="0-15">0–15'</option><option value="16-30">16–30'</option><option value="31-45">31–45+'</option>
   <option value="46-60">46–60'</option><option value="61-75">61–75'</option><option value="76-200">76–90+'</option></select></label>
 <button class="reset" id="reset">Reset filters</button>
</div>
<main><p id="compare"></p><div class="stats" id="stats"></div><p class="note warn" id="warn"></p><div class="maps" id="maps"></div>
<p class="note" id="foot"></p></main><div id="tip"></div>
<script>
const D = __DATA__;
const F = D.F, C = D.C, NF = F.lg.length, NC = C.lg.length;
const [GX, GY] = D.meta.grid;
let view = "teams", mode = "vs";
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (n, d = 0) => (+n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
const pct = v => isFinite(v) ? fmt(100 * v, 0) + "%" : "–";
const MIRROR = [0, 2, 1, 4, 3];             // zone codes: 1 attacking 2 defensive 3 left 4 right

function hex2rgb(h){h=h.replace("#","");if(h.length==3)h=h.split("").map(c=>c+c).join("");return [0,2,4].map(i=>parseInt(h.substr(i,2),16))}
function mix(a,b,t){const A=hex2rgb(a),B=hex2rgb(b);return `rgb(${A.map((v,i)=>Math.round(v+(B[i]-v)*t)).join(",")})`}
function divColor(r){ if(r===0) return css("--lo"); if(!isFinite(r)||r<0) return css("--mid"); const v=Math.max(-1,Math.min(1,Math.log2(r)));
  return v>=0?mix(css("--mid"),css("--hi"),v):mix(css("--mid"),css("--lo"),-v)}
function seqColor(s,max){ if(!max) return css("--mid"); return mix(css("--mid"),css("--accent"),Math.min(1,s/max))}

// ------------------------------------------------------------- filters
function opt(sel, items, keep){const prev=keep?sel.value:null; sel.innerHTML=items.map(([v,t,g])=>
  `<option value="${v}">${t}</option>`).join(""); if(prev!==null&&[...sel.options].some(o=>o.value==prev)) sel.value=prev}
function fillLeagues(){const n=D.leagues.map(()=>0); F.lg.forEach(l=>n[l]++);
  opt($("#lg"), D.leagues.map((name,i)=>[i,`${name} (${fmt(n[i])} fouls)`]).sort((a,b)=>n[b[0]]-n[a[0]]))}
function fillSeasons(){const lg=+$("#lg").value, seen=new Set(); for(let i=0;i<NF;i++) if(F.lg[i]==lg) seen.add(F.se[i]);
  const list=[...seen].sort((a,b)=>D.seasonOrder[b].localeCompare(D.seasonOrder[a]));
  opt($("#se"), [["all","All seasons"], ...list.map(s=>[s,D.seasons[s]])], true)}
let Q = {};
function readFilters(){const pe=$("#pe").value;
  Q={lg:+$("#lg").value, se:$("#se").value=="all"?null:+$("#se").value, sd:$("#sd").value, md:$("#md").value,
     ve:view=="refs"?"all":$("#ve").value, pe:pe=="all"?null:pe.split("-").map(Number),
     cross:$("#cross").value=="all"?null:+$("#cross").value}}
function seasonOk(i, S){const s=$("#se").value; return s=="all"||S.se[i]==+s}
function fillFocus(){const lg=+$("#lg").value, cnt=new Map(), team=new Map();
  for(let i=0;i<NF;i++){ if(F.lg[i]!=lg||!seasonOk(i,F)) continue;
    if(view=="teams"){[F.ct[i],F.ft[i]].forEach(t=>t>=0&&cnt.set(t,(cnt.get(t)||0)+1))}
    else if(view=="refs"){F.rf[i]>=0&&cnt.set(F.rf[i],(cnt.get(F.rf[i])||0)+1)}
    else {[[F.cp[i],F.ct[i]],[F.fp[i],F.ft[i]]].forEach(([p,t])=>{if(p>=0){cnt.set(p,(cnt.get(p)||0)+1);team.set(p,t)}})}}
  const sel=$("#focus"); $("#focusLabel").firstChild.textContent=view=="teams"?"Team":view=="refs"?"Referee":"Player";
  const names=view=="teams"?D.teams:view=="refs"?D.refs:D.players;
  if(view=="players"){ // group players by team
    const groups=new Map(); [...cnt.keys()].forEach(p=>{const t=team.get(p);if(!groups.has(t))groups.set(t,[]);groups.get(t).push(p)});
    const prev=sel.value; sel.innerHTML=[...groups.entries()].sort((a,b)=>(D.teams[a[0]]||"").localeCompare(D.teams[b[0]]||"")).map(([t,ps])=>
      `<optgroup label="${D.teams[t]||"Unknown team"}">`+ps.sort((a,b)=>cnt.get(b)-cnt.get(a)).map(p=>
      `<option value="${p}">${names[p]} (${cnt.get(p)})</option>`).join("")+"</optgroup>").join("");
    if([...sel.options].some(o=>o.value==prev)) sel.value=prev;
  } else opt(sel,[...cnt.entries()].sort((a,b)=>b[1]-a[1]).map(([k,n])=>[k,`${names[k]} (${fmt(n)}${n<D.meta.minFouls?", low sample":""})`]),true);
  // cross filter: referee (teams/players view) or team (referees view)
  const cross=new Map(); for(let i=0;i<NF;i++){ if(F.lg[i]!=lg||!seasonOk(i,F)) continue;
    if(view=="refs"){[F.ct[i],F.ft[i]].forEach(t=>t>=0&&cross.set(t,(cross.get(t)||0)+1))} else if(F.rf[i]>=0) cross.set(F.rf[i],(cross.get(F.rf[i])||0)+1)}
  $("#crossLabel").firstChild.textContent=view=="refs"?"Matches involving":"Referee";
  const cn=view=="refs"?D.teams:D.refs;
  opt($("#cross"),[["all",view=="refs"?"Any team":"Any referee"],...[...cross.keys()].sort((a,b)=>cn[a].localeCompare(cn[b])).map(k=>[k,cn[k]])],true);
  $("#venueLabel").style.display=view=="refs"?"none":"";
  $("#sdLabel").firstChild.textContent=view=="refs"?"Score (team awarded the free kick)":view=="players"?"Score (player's team)":"Score (team in focus)";
}

// does row i pass the context filters, seen from the subject team?  side: 'c' committer, 'f' fouled, 'k' card
function context(i, S, side){
  if(S.lg[i]!=Q.lg || (Q.se!==null&&S.se[i]!=Q.se)) return false;
  let sd, md, home;
  if(S===F){ sd=side=="c"?F.sd[i]:(F.sd[i]==-9?-9:-F.sd[i]); md=side=="c"?F.md[i]:(F.md[i]==-9?-9:-F.md[i]); home=F.fh[i]<0?-1:(side=="c"?1-F.fh[i]:F.fh[i]) }
  else { sd=C.sd[i]; md=C.md[i]; home=C.hm[i] }
  const fs=Q.sd; if(fs!="all"){ if(sd==-9) return false;
    if(fs=="w"&&sd<=0||fs=="w2"&&sd<2||fs=="d"&&sd!=0||fs=="l"&&sd>=0||fs=="l2"&&sd>-2) return false }
  const fm=Q.md; if(fm!="all"){ if(md==-9) return false;
    if(fm=="even"&&md!=0||fm=="up"&&md<=0||fm=="down"&&md>=0) return false }
  const fv=Q.ve; if(fv!="all"){ if(home<0||(fv=="h")!=(home==1)) return false }
  if(Q.pe){ if(S.mi[i]<Q.pe[0]||S.mi[i]>Q.pe[1]) return false }
  if(Q.cross!==null){ const c=Q.cross;
    if(view=="refs"){ if(S===F? (F.ct[i]!=c&&F.ft[i]!=c) : C.tm[i]!=c) return false }
    else if(S.rf[i]!=c) return false }
  return true }

// ------------------------------------------------------------- counting
function blank(){return {n:0, zones:[0,0,0,0,0], grid:new Array(GX*GY).fill(0), located:0, carded:0, fx:new Set()}}
function cellOf(x,y){return Math.min(GY-1,Math.max(0,Math.floor(y/100*GY)))*GX+Math.min(GX-1,Math.max(0,Math.floor(x/100*GX)))}
// spec: {S, side, who(i)->bool (focus filter), home: undefined|0|1 (referee maps)}
function count(spec){const a=blank(), b=blank(); const S=spec.S;
  for(let i=0;i<S.lg.length;i++){
    if(spec.home!==undefined && (S===F?F.fh[i]:C.hm[i])!=spec.home) continue;
    if(!context(i,S,spec.side)) continue;
    const mine=spec.who(i); const targets=mine?[a,b]:[b];
    let zone=0,x=-1,y=-1;
    if(S===F){ zone=spec.side=="c"?MIRROR[F.zo[i]]:F.zo[i]; if(F.x[i]>=0){x=spec.side=="c"?F.x[i]:100-F.x[i]; y=spec.side=="c"?F.y[i]:100-F.y[i]} }
    else { x=C.x[i]; y=C.y[i] }
    for(const t of targets){ t.n++; t.zones[zone]++; t.fx.add(S.fx[i]); if(S===F&&F.cd[i]) t.carded++; if(x>=0){t.grid[cellOf(x,y)]++; t.located++} } }
  return {me:a, all:b}}

// ------------------------------------------------------------- drawing
function pitch(svg){const P=css("--pline");return `<rect x="0" y="0" width="105" height="68" fill="${css("--pitch")}"/>
 ${svg}<g fill="none" stroke="${P}" stroke-width=".4"><rect x="0" y="0" width="105" height="68"/><line x1="52.5" y1="0" x2="52.5" y2="68"/>
 <circle cx="52.5" cy="34" r="9.15"/><rect x="0" y="13.84" width="16.5" height="40.32"/><rect x="88.5" y="13.84" width="16.5" height="40.32"/>
 <rect x="0" y="24.84" width="5.5" height="18.32"/><rect x="99.5" y="24.84" width="5.5" height="18.32"/></g>
 <path d="M44 72 L61 72" stroke="#8a93a3" stroke-width=".6" marker-end="url(#arr)"/>
 <text x="52.5" y="77" text-anchor="middle" font-size="3" fill="${css("--muted")}">attacking direction</text>`}
const defs=`<defs><marker id="arr" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L6 3L0 6z" fill="#8a93a3"/></marker></defs>`;
function gridSVG(m, base){const g=m.grid, b=base.grid, t=m.located, bt=base.located; let max=0; g.forEach(v=>max=Math.max(max,t?v/t:0));
  const cw=105/GX, ch=68/GY, A=GX*GY; let out="";
  for(let iy=0;iy<GY;iy++)for(let ix=0;ix<GX;ix++){const k=iy*GX+ix, row=D.meta.flipY?GY-1-iy:iy;
    const s=t?g[k]/t:0, bs=bt?b[k]/bt:0, r=bs?((g[k]+A*bs)/(t+A))/bs:NaN;
    const fill=mode=="vs"?(t&&bt?divColor(r):css("--mid")):seqColor(s,max);
    out+=`<rect x="${ix*cw}" y="${row*ch}" width="${cw}" height="${ch}" fill="${fill}" fill-opacity=".9"
      data-t="${fmt(g[k])} here · ${pct(s)} of this map · comparison ${pct(bs)}"/>`}
  return out}
function zonesSVG(m, base){const z=m.zones, bz=base.zones, tot=z[1]+z[2]+z[3]+z[4], btot=bz[1]+bz[2]+bz[3]+bz[4];
  const R={3:[0,0,105,15,"Left wing"],4:[0,53,105,15,"Right wing"],2:[0,15,52.5,38,"Own half"],1:[52.5,15,52.5,38,"Opposition half"]};
  let out="";for(const k in R){const [x,y,w,h,name]=R[k], s=tot?z[k]/tot:0, b=btot?bz[k]/btot:0;
    const fill=mode=="vs"?(tot&&btot?divColor(s/b):css("--mid")):seqColor(s,1);
    out+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" fill-opacity=".9" stroke="${css("--card")}" stroke-width=".6" data-t="${name}: ${fmt(z[k])} (${pct(s)}) · comparison ${pct(b)}"/>
      <text x="${x+w/2}" y="${y+h/2-1}" text-anchor="middle" font-size="3.6" font-weight="600" fill="${css("--ink")}">${name}</text>
      <text x="${x+w/2}" y="${y+h/2+3.8}" text-anchor="middle" font-size="3.4" fill="${css("--ink")}">${pct(s)} <tspan fill="${css("--muted")}">(${pct(b)})</tspan></text>`}
  return {svg:out,tot}}
function legend(){return mode=="vs"?`<div class="legend">less <div class="grad" style="background:linear-gradient(90deg,${css("--lo")},${css("--mid")},${css("--hi")})"></div> more than comparison (½×–2×)</div>`
  :`<div class="legend">share of this map <div class="grad" style="background:linear-gradient(90deg,${css("--mid")},${css("--accent")})"></div></div>`}
function panel(title, sub, inner){return `<div class="map"><h3>${title}</h3><div class="sub">${sub}</div><svg viewBox="0 0 105 79">${defs}${pitch(inner)}</svg>${legend()}</div>`}
function zoneMap(title, sub, r){const z=zonesSVG(r.me, r.all); return panel(title, `${sub} · ${fmt(z.tot)} with a zone`, z.svg)}
function spotMap(title, sub, r){return panel(title, `${sub} · ${fmt(r.me.located)} with an estimated spot`,
  r.me.located?gridSVG(r.me,r.all):`<text x="52.5" y="36" text-anchor="middle" font-size="4" fill="${css("--muted")}">Nothing located with these filters</text>`)}
function stat(v,label,cmp){return `<div class="stat"><b>${v}</b><span>${label}</span>${cmp!==undefined?`<br><i>comparison ${cmp}</i>`:""}</div>`}
const oppHalf=m=>{const t=m.zones[1]+m.zones[2];return t?m.zones[1]/t:NaN};

function describe(){const parts=[D.leagues[+$("#lg").value]]; const s=$("#se").value; parts.push(s=="all"?"all seasons":D.seasons[+s]);
  const pick=id=>{const e=$(id);return e.options[e.selectedIndex]?.text};
  if($("#cross").value!="all") parts.push((view=="refs"?"matches involving ":"referee ")+pick("#cross"));
  if($("#sd").value!="all") parts.push(pick("#sd").toLowerCase()); if($("#md").value!="all") parts.push(pick("#md").toLowerCase());
  if(view!="refs"&&$("#ve").value!="all") parts.push(pick("#ve").toLowerCase()); if($("#pe").value!="all") parts.push(pick("#pe"));
  return parts.join(" · ")}

function render(){readFilters(); const f=$("#focus").value; if(f===""){$("#stats").innerHTML="";$("#maps").innerHTML="<p class='note'>No data with these filters.</p>";$("#compare").textContent="";return}
  const id=+f; let S="",M="",n=0;
  const who=view=="teams"?{c:i=>F.ct[i]==id,f:i=>F.ft[i]==id,k:i=>C.tm[i]==id}
          :view=="players"?{c:i=>F.cp[i]==id,f:i=>F.fp[i]==id,k:i=>C.pl[i]==id}
          :{c:i=>F.rf[i]==id,f:i=>F.rf[i]==id,k:i=>C.rf[i]==id};
  const label=view=="teams"?D.teams[id]:view=="players"?D.players[id]:D.refs[id];
  const others=view=="teams"?"all teams":view=="players"?"all players":"all referees";
  $("#compare").innerHTML=`<b>${label}</b> compared with ${others} · ${describe()}`;
  if(view!="refs"){
    const com=count({S:F,side:"c",who:who.c}), won=count({S:F,side:"f",who:who.f}), crd=count({S:C,side:"k",who:who.k});
    n=com.me.n+won.me.n; const matches=new Set([...com.me.fx,...won.me.fx]).size;
    const perEnt=view=="teams"?2:22;   // comparison per team (2 per match) / per player (~22 per match)
    const bm=Math.max(1,com.all.fx.size);
    S+=stat(fmt(matches),"matches");
    S+=stat(fmt(com.me.n/Math.max(1,matches),view=="teams"?1:2),"fouls committed / match",fmt(com.all.n/bm/perEnt,view=="teams"?1:2));
    S+=stat(fmt(won.me.n/Math.max(1,matches),view=="teams"?1:2),"fouls won / match",fmt(won.all.n/bm/perEnt,view=="teams"?1:2));
    S+=stat(fmt(crd.me.n),"cards"); S+=stat(pct(com.me.carded/Math.max(1,com.me.n)),"of fouls committed carded",pct(com.all.carded/Math.max(1,com.all.n)));
    S+=stat(pct(oppHalf(com.me)),"of fouls committed in opposition half",pct(oppHalf(com.all)));
    M+=zoneMap("Fouls committed — zones","from this "+(view=="teams"?"team":"player")+"'s side",com);
    M+=spotMap("Fouls committed — spots","estimated from where the ball stopped",com);
    M+=zoneMap("Fouls won — zones","free kicks won",won);
    M+=spotMap("Fouls won — spots","estimated",won);
    M+=spotMap("Cards received","estimated spot of each card",crd);
  } else {
    const home=count({S:F,side:"f",who:who.f,home:1}), away=count({S:F,side:"f",who:who.f,home:0}), crd=count({S:C,side:"k",who:who.k});
    const all=count({S:F,side:"f",who:who.f}); n=all.me.n; const matches=Math.max(1,all.me.fx.size), bm=Math.max(1,all.all.fx.size);
    S+=stat(fmt(all.me.fx.size),"matches");
    S+=stat(fmt(all.me.n/matches,1),"fouls given / match",fmt(all.all.n/bm,1));
    S+=stat(fmt(crd.me.n/matches,2),"cards / match",fmt(crd.all.n/Math.max(1,crd.all.fx.size),2));
    S+=stat(pct(all.me.carded/Math.max(1,all.me.n)),"of fouls carded",pct(all.all.carded/Math.max(1,all.all.n)));
    S+=stat(pct(home.me.n/Math.max(1,home.me.n+away.me.n)),"of free kicks to the home team",pct(home.all.n/Math.max(1,home.all.n+away.all.n)));
    M+=zoneMap("Free kicks to the HOME team — zones","from the home team's side",home);
    M+=zoneMap("Free kicks to the AWAY team — zones","from the away team's side",away);
    M+=spotMap("Free kicks to the home team — spots","estimated",home);
    M+=spotMap("Free kicks to the away team — spots","estimated",away);
    M+=spotMap("Cards given","from the carded team's side",crd);
  }
  $("#stats").innerHTML=S;
  $("#warn").textContent=n<D.meta.minFouls?`Small sample with these filters (${fmt(n)} fouls) — treat differences as noise.`:"";
  $("#warn").style.display=n<D.meta.minFouls?"":"none";
  $("#maps").innerHTML=M;
  $("#foot").textContent=`Built ${D.meta.generated}. Score and players-on-pitch are from the team in focus's side at the moment of the foul (for referees: the team awarded the free kick or given the card). Zones come from commentary wording (wings don't say which half). Spots are estimates from where the ball rested during the stoppage. Only matches with commentary are included.`}

document.addEventListener("mousemove",ev=>{const t=ev.target.getAttribute&&ev.target.getAttribute("data-t"),tip=$("#tip");
  if(t){tip.textContent=t;tip.style.display="block";tip.style.left=Math.min(ev.clientX+12,innerWidth-280)+"px";tip.style.top=(ev.clientY+12)+"px"}else tip.style.display="none"});
$("#lg").onchange=()=>{fillSeasons();fillFocus();render()}; $("#se").onchange=()=>{fillFocus();render()};
$("#focus").onchange=render; ["#cross","#sd","#md","#ve","#pe"].forEach(s=>$(s).onchange=render);
$("#reset").onclick=()=>{["#cross","#sd","#md","#ve","#pe"].forEach(s=>$(s).value="all");render()};
document.querySelectorAll("#view button").forEach(b=>b.onclick=()=>{view=b.dataset.v;document.querySelectorAll("#view button").forEach(x=>x.classList.toggle("on",x==b));$("#cross").value="all";fillFocus();render()});
document.querySelectorAll("#mode button").forEach(b=>b.onclick=()=>{mode=b.dataset.m;document.querySelectorAll("#mode button").forEach(x=>x.classList.toggle("on",x==b));render()});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
fillLeagues();fillSeasons();fillFocus();render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Build foul maps from commentary and ball positions")
    ap.add_argument("--rebuild", action="store_true", help="re-process every match")
    ap.add_argument("--report-only", action="store_true", help="only rebuild the HTML report")
    ap.add_argument("--leagues", nargs="+", type=int, default=TOP5,
                    help="league ids to include (default: top 5 European leagues)")
    ap.add_argument("--all-leagues", action="store_true", help="include every league")
    args = ap.parse_args()
    leagues = None if args.all_leagues else args.leagues
    db = L.Db(L.DATABASE)
    db.columns("football.fixtures")
    db.execute(SETUP_SQL)
    db.commit()
    if not args.report_only:
        process(db, leagues, rebuild=args.rebuild)
    build_report(db, leagues)


if __name__ == "__main__":
    main()
