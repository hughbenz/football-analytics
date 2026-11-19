#!/usr/bin/env python3
"""
Load the last N Bundesliga seasons from Sportmonks into the football database
(the PostgreSQL schema from schema.sql).

LOAD ORDER (each step builds the IDs the next one depends on)
  1. Core reference data   types, states, continents, countries, bookmakers, markets
  2. League + seasons      Bundesliga and the seasons to load
  3. Season structure      stages, rounds, groups, season statistics, fixture list
  4. Teams                 teams, venues, team season stats, squads, injuries,
                           rivals, trophies, coaching spells, transfers
  5. Fixtures              everything inside each match, in passes:
                             main   participants, scores, events, lineups + player
                                    stats, team stats, officials, coaches, xG,
                                    weather, injuries, predictions, news ...
                             live   trends, pressure index, commentary, match facts
                             odds   pre-match odds, premium odds + price history
                             ball   ball coordinates (large)
  6. Standings + topscorers
  7. People                players, referees, coaches with season statistics
  8. Refresh analysis views

SETUP (once)
  pip install "psycopg[binary]"
  Fill in the DATABASE settings below (password etc.).

RUN
  python load_bundesliga.py

  Re-running is safe. Finished matches already downloaded are skipped;
  unfinished/upcoming ones are refreshed. If the run stops (internet drops,
  computer sleeps), run it again and it continues where it left off.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:
    sys.exit('Missing package. Run:  pip install "psycopg[binary]"')


# =============================================================== SETTINGS ===
API_TOKEN = os.environ.get("SPORTMONKS_API_TOKEN") or \
    "KisK8kQ5z9fz0hRsxYIVraixeiwMQgxBhXAvfmwEDCzOQwQZCngnOj5Hj074"

DATABASE = {
    "dbname":   os.environ.get("FOOTBALL_DB_NAME", "football"),
    "user":     os.environ.get("FOOTBALL_DB_USER", "postgres"),
    "password": os.environ.get("FOOTBALL_DB_PASSWORD", ""),   # your PostgreSQL password
    "host":     os.environ.get("FOOTBALL_DB_HOST", "localhost"),
    "port":     os.environ.get("FOOTBALL_DB_PORT", "5432"),
}

LEAGUE_ID       = 82      # Bundesliga
NUM_SEASONS     = 3       # most recent seasons
INCLUDE_CURRENT = True    # True: 2024/25, 2025/26, 2026/27 (in progress)
                          # False: last three finished seasons

FETCH_ODDS              = True
FETCH_BALL_COORDINATES  = True   # largest data set; set False to skip
STORE_RAW               = True   # keep raw API responses (not odds / ball data)
PEOPLE_REFRESH_DAYS     = 7      # re-download player/referee/coach stats after N days

API_BASE = "https://api.sportmonks.com/v3"

# Fixture include passes. "a|b" = try a, fall back to b if a isn't available.
FIXTURE_PASSES = {
    "main": {
        "batch": 10,
        "raw": True,
        "includes": [
            "participants", "scores", "periods", "events", "venue",
            "lineups.details", "lineups.xGLineup", "formations", "statistics",
            "referees", "coaches", "tvStations", "weatherReport",
            "sidelined.sideline", "xGFixture", "expectedLineups", "metadata",
            "predictions", "prematchNews.lines", "postmatchNews.lines",
        ],
    },
    "live": {
        "batch": 10,
        "raw": True,
        "includes": ["trends", "pressure", "comments", "matchfacts|matchFacts"],
    },
    "odds": {
        "batch": 3,
        "raw": False,
        "includes": ["odds", "premiumOdds.history|premiumOdds"],
    },
    "ball": {
        "batch": 2,
        "raw": False,
        "includes": ["ballCoordinates"],
    },
}

FINAL_STATES = {"FT", "AET", "FT_PEN", "AWARDED", "CANCELLED", "ABANDONED", "DELETED"}

WARNINGS = []


def warn(msg):
    print("  ! " + msg)
    WARNINGS.append(msg)


# ==================================================================== API ===
class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class Api:
    def __init__(self, token, db):
        self.token = token
        self.db = db
        self.calls = 0
        self.include_cache = {}

    def get(self, path, params=None, store=None, retries=6):
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        url = API_BASE + path + ("?" + urllib.parse.urlencode(params, safe=";,:|.") if params else "")
        req = urllib.request.Request(url, headers={"Authorization": self.token, "Accept": "application/json"})

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.load(resp)
                self.calls += 1
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                if e.code == 429:
                    wait = 60
                    try:
                        wait = int(json.loads(text)["rate_limit"]["resets_in_seconds"]) + 5
                    except Exception:
                        pass
                    print(f"  rate limit reached, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if e.code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                try:
                    message = json.loads(text).get("message", text)
                except Exception:
                    message = text
                if e.code == 401:
                    sys.exit(f"Sportmonks rejected the API token (401): {message}")
                raise ApiError(e.code, str(message)[:300])
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  network problem ({e}), retrying")
                time.sleep(2 ** attempt)
                continue

            rl = body.get("rate_limit") or {}
            if isinstance(rl.get("remaining"), int) and rl["remaining"] < 5:
                wait = int(rl.get("resets_in_seconds", 60)) + 5
                print(f"  {rl.get('requested_entity')} limit nearly used, waiting {wait}s")
                time.sleep(wait)

            if store and STORE_RAW:
                self.db.store_raw(path, params, store[0], store[1], body)
            return body

        raise ApiError(0, f"gave up on {path} after {retries} attempts")

    def pages(self, path, params=None, store=None):
        params = dict(params or {})
        page, items = 1, []
        while True:
            params["page"] = page
            body = self.get(path, params, store)
            data = body.get("data") or []
            items.extend(data if isinstance(data, list) else [data])
            if not (body.get("pagination") or {}).get("has_more"):
                return items
            page += 1

    def get_inc(self, kind, path, includes, params=None, store=None):
        """GET with includes. If Sportmonks rejects an include (not on your
        plan / different name), find which ones work, remember, and continue."""
        params = dict(params or {})
        accepted = self.include_cache.get(kind)
        if accepted is None:
            first_choice = [spec.split("|")[0] for spec in includes]
            try:
                body = self.get(path, {**params, "include": ";".join(first_choice)}, store)
                self.include_cache[kind] = first_choice
                return body
            except ApiError as err:
                print(f"  {kind}: request failed ({err}); checking includes one by one")
                accepted, dropped = [], []
                for spec in includes:
                    for alt in spec.split("|"):
                        try:
                            self.get(path, {**params, "include": alt})
                            accepted.append(alt)
                            break
                        except ApiError:
                            dropped.append(alt)
                if not accepted:
                    raise err
                if dropped:
                    warn(f"{kind}: not available / skipped includes: {', '.join(dropped)}")
                self.include_cache[kind] = accepted
        return self.get(path, {**params, "include": ";".join(accepted)}, store)


# =============================================================== DATABASE ===
def to_num(v):
    if v is None or isinstance(v, (dict, list)):
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, Decimal)):
        return v
    s = str(v).replace("%", "").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def to_int(v):
    n = to_num(v)
    try:
        return int(n) if n is not None else None
    except (ValueError, OverflowError):
        return None


def to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def to_ts(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00").replace("T", " "))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def to_date(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc).date()
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


class Db:
    def __init__(self, settings):
        try:
            self.conn = psycopg.connect(**settings)
        except Exception as e:
            sys.exit(f"Could not connect to PostgreSQL: {e}\n"
                     f"Check the DATABASE settings at the top of this file.")
        self._cols, self._pk = {}, {}

    def query(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else []

    def execute(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)

    def commit(self):
        self.conn.commit()

    def columns(self, table):
        if table not in self._cols:
            schema, name = table.split(".")
            rows = self.query(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s AND is_generated = 'NEVER' "
                "ORDER BY ordinal_position", (schema, name))
            if not rows:
                sys.exit(f"Table {table} not found - has schema.sql been loaded into this database?")
            self._cols[table] = {r[0]: r[1] for r in rows}
            self._pk[table] = [r[0] for r in self.query(
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = %s::regclass AND i.indisprimary", (table,))]
        return self._cols[table]

    @staticmethod
    def adapt(dtype, v):
        if v is None:
            return None
        if dtype in ("bigint", "integer", "smallint"):
            return to_int(v)
        if dtype in ("numeric", "double precision", "real"):
            return to_num(v)
        if dtype == "boolean":
            return to_bool(v)
        if dtype == "date":
            return to_date(v)
        if dtype.startswith("timestamp"):
            return to_ts(v)
        if dtype == "jsonb":
            return Jsonb(v)
        if dtype == "ARRAY":
            return list(v) if isinstance(v, (list, tuple)) else None
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    def upsert(self, table, rows):
        """Insert rows; update existing ones without blanking known values."""
        cols_meta = self.columns(table)
        pk = self._pk[table]
        rows = [r for r in rows if r and all(r.get(k) is not None for k in pk)]
        if not rows:
            return 0
        # one row per key (last wins)
        unique = {}
        for r in rows:
            unique[tuple(r[k] for k in pk)] = r
        rows = list(unique.values())

        cols = [c for c in cols_meta if any(c in r for r in rows)]
        non_pk = [c for c in cols if c not in pk]
        sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               f"ON CONFLICT ({', '.join(pk)}) ")
        if non_pk:
            sql += "DO UPDATE SET " + ", ".join(
                f"{c} = COALESCE(EXCLUDED.{c}, {table}.{c})" for c in non_pk)
        else:
            sql += "DO NOTHING"
        data = [tuple(self.adapt(cols_meta[c], r.get(c)) for c in cols) for r in rows]
        with self.conn.cursor() as cur:
            cur.executemany(sql, data)
        return len(rows)

    def store_raw(self, endpoint, params, entity, entity_id, body):
        self.execute(
            "INSERT INTO raw.api_responses (endpoint, params, entity, entity_id, payload) "
            "VALUES (%s, %s, %s, %s, %s)",
            (endpoint, Jsonb(params), entity, entity_id, Jsonb(body)))

    def set_sync(self, entity, key, status="ok", detail=None):
        self.execute(
            "INSERT INTO raw.sync_state (entity, scope_key, last_synced_at, status, detail) "
            "VALUES (%s, %s, now(), %s, %s) ON CONFLICT (entity, scope_key) DO UPDATE SET "
            "last_synced_at = now(), status = EXCLUDED.status, detail = EXCLUDED.detail",
            (entity, str(key), status, Jsonb(detail or {})))

    def get_sync(self, entity):
        return {str(r[0]): (r[1], r[2], r[3]) for r in self.query(
            "SELECT scope_key, last_synced_at, status, detail FROM raw.sync_state WHERE entity = %s",
            (entity,))}


# ================================================================ MAPPERS ===
# Each turns a Sportmonks object into a row dict for one table.

def g(o, *path):
    for p in path:
        if not isinstance(o, dict):
            return None
        o = o.get(p)
    return o


def person_row(o, id_col):
    return {id_col: o.get("id"), "country_id": o.get("country_id"),
            "nationality_id": o.get("nationality_id"), "city_id": o.get("city_id"),
            "common_name": o.get("common_name"), "firstname": o.get("firstname"),
            "lastname": o.get("lastname"), "name": o.get("name"),
            "display_name": o.get("display_name"), "height": o.get("height"),
            "weight": o.get("weight"), "date_of_birth": o.get("date_of_birth"),
            "gender": o.get("gender"), "image_path": o.get("image_path")}


def team_row(o):
    return {"team_id": o.get("id"), "country_id": o.get("country_id"), "venue_id": o.get("venue_id"),
            "name": o.get("name"), "short_code": o.get("short_code"), "gender": o.get("gender"),
            "type": o.get("type"), "founded": o.get("founded"), "placeholder": o.get("placeholder"),
            "last_played_at": o.get("last_played_at"), "image_path": o.get("image_path")}


def venue_row(o):
    return {"venue_id": o.get("id"), "country_id": o.get("country_id"), "city_id": o.get("city_id"),
            "name": o.get("name"), "address": o.get("address"), "zipcode": o.get("zipcode"),
            "state": o.get("state"), "city_name": o.get("city_name"), "latitude": o.get("latitude"),
            "longitude": o.get("longitude"), "capacity": o.get("capacity"), "surface": o.get("surface"),
            "national_team": o.get("national_team"), "image_path": o.get("image_path")}


def fixture_row(o):
    return {"fixture_id": o.get("id"), "league_id": o.get("league_id"), "season_id": o.get("season_id"),
            "stage_id": o.get("stage_id"), "round_id": o.get("round_id"), "group_id": o.get("group_id"),
            "aggregate_id": o.get("aggregate_id"), "state_id": o.get("state_id"),
            "venue_id": o.get("venue_id"), "name": o.get("name"),
            "starting_at": o.get("starting_at_timestamp") or o.get("starting_at"),
            "result_info": o.get("result_info"), "leg": o.get("leg"), "details": o.get("details"),
            "length": o.get("length"), "placeholder": o.get("placeholder"),
            "has_odds": o.get("has_odds"), "has_premium_odds": o.get("has_premium_odds")}


def structure_row(o, id_col, extra=()):
    row = {id_col: o.get("id"), "league_id": o.get("league_id"), "season_id": o.get("season_id"),
           "name": o.get("name"), "finished": o.get("finished"), "is_current": o.get("is_current"),
           "starting_at": o.get("starting_at"), "ending_at": o.get("ending_at")}
    for k in extra:
        row[k] = o.get(k)
    return row


def odd_row(o, fid):
    return {"odd_id": o.get("id"), "fixture_id": fid, "market_id": o.get("market_id"),
            "bookmaker_id": o.get("bookmaker_id"), "label": o.get("label"),
            "original_label": o.get("original_label"), "name": o.get("name"),
            "market_description": o.get("market_description"), "value": o.get("value"),
            "probability": o.get("probability"), "dp3": o.get("dp3"),
            "fractional": o.get("fractional"), "american": o.get("american"),
            "winning": o.get("winning"), "stopped": o.get("stopped"), "total": o.get("total"),
            "handicap": o.get("handicap"), "participants": o.get("participants"),
            "latest_bookmaker_update": o.get("latest_bookmaker_update")}


def stat_rows(items, parent_table, parent_id_col, owner_col, detail_table, detail_id_col):
    """Season statistics: parent rows + detail rows (team/player/coach/referee)."""
    parents, details = [], []
    for s in items or []:
        parents.append({parent_id_col: s.get("id"), owner_col: s.get(owner_col),
                        "team_id": s.get("team_id"), "season_id": s.get("season_id"),
                        "position_id": s.get("position_id"), "jersey_number": s.get("jersey_number"),
                        "has_values": s.get("has_values")})
        for d in s.get("details") or []:
            details.append({detail_id_col: d.get("id"), parent_id_col: s.get("id"),
                            "type_id": d.get("type_id"), "data": d.get("value")})
    return parents, details


# ================================================================== STEPS ===
def load_core(api, db):
    print("\n[1/8] Core reference data")
    jobs = [
        ("core.types", "/core/types", lambda o: {
            "type_id": o["id"], "name": o.get("name"), "code": o.get("code"),
            "developer_name": o.get("developer_name"), "model_type": o.get("model_type"),
            "stat_group": o.get("stat_group")}),
        ("core.states", "/football/states", lambda o: {
            "state_id": o["id"], "state": o.get("state"), "name": o.get("name"),
            "short_name": o.get("short_name"), "developer_name": o.get("developer_name")}),
        ("core.continents", "/core/continents", lambda o: {
            "continent_id": o["id"], "name": o.get("name"), "code": o.get("code")}),
        ("core.countries", "/core/countries", lambda o: {
            "country_id": o["id"], "continent_id": o.get("continent_id"), "name": o.get("name"),
            "official_name": o.get("official_name"), "fifa_name": o.get("fifa_name"),
            "iso2": o.get("iso2"), "iso3": o.get("iso3"), "latitude": o.get("latitude"),
            "longitude": o.get("longitude"), "borders": o.get("borders"),
            "image_path": o.get("image_path")}),
        ("odds.bookmakers", "/odds/bookmakers", lambda o: {
            "bookmaker_id": o["id"], "legacy_id": o.get("legacy_id"), "name": o.get("name")}),
        ("odds.markets", "/odds/markets", lambda o: {
            "market_id": o["id"], "legacy_id": o.get("legacy_id"), "name": o.get("name"),
            "developer_name": o.get("developer_name"),
            "has_winning_calculations": o.get("has_winning_calculations")}),
    ]
    for table, path, mapper in jobs:
        existing = db.query(f"SELECT count(*) FROM {table}")[0][0]
        if existing:
            print(f"  {table}: {existing} rows already loaded")
            continue
        try:
            items = api.pages(path, {"per_page": 50})
        except ApiError as e:
            if path == "/football/states":     # older docs place states under /core
                try:
                    items = api.pages("/core/states", {"per_page": 50})
                except ApiError:
                    warn(f"{table}: {e}")
                    continue
            else:
                warn(f"{table}: {e}")
                continue
        n = db.upsert(table, [mapper(o) for o in items if o.get("id") is not None])
        db.commit()
        print(f"  {table}: {n} rows")


def load_league_and_seasons(api, db):
    print("\n[2/8] League and seasons")
    body = api.get(f"/football/leagues/{LEAGUE_ID}", {"include": "seasons"}, store=("league", LEAGUE_ID))
    lg = body["data"]
    db.upsert("football.leagues", [{
        "league_id": lg["id"], "country_id": lg.get("country_id"), "name": lg.get("name"),
        "short_code": lg.get("short_code"), "type": lg.get("type"), "sub_type": lg.get("sub_type"),
        "category": lg.get("category"), "active": lg.get("active"), "has_jerseys": lg.get("has_jerseys"),
        "last_played_at": lg.get("last_played_at"), "image_path": lg.get("image_path")}])

    seasons = lg.get("seasons") or []
    db.upsert("football.seasons", [{
        "season_id": s["id"], "league_id": lg["id"], "name": s.get("name"),
        "is_current": s.get("is_current"), "finished": s.get("finished"), "pending": s.get("pending"),
        "starting_at": s.get("starting_at"), "ending_at": s.get("ending_at"),
        "tie_breaker_rule_id": s.get("tie_breaker_rule_id"),
        "standings_recalculated_at": s.get("standings_recalculated_at"),
        "games_in_current_week": s.get("games_in_current_week")} for s in seasons])
    db.commit()

    today = date.today()
    started = [s for s in seasons if to_date(s.get("starting_at")) and to_date(s["starting_at"]) <= today]
    if not INCLUDE_CURRENT:
        started = [s for s in started if s.get("finished")]
    started.sort(key=lambda s: to_date(s["starting_at"]), reverse=True)
    chosen = started[:NUM_SEASONS]
    if not chosen:
        sys.exit("No seasons found for this league - check LEAGUE_ID and your plan's leagues.")
    print(f"  {lg.get('name')}: loading " + ", ".join(f"{s['name']} (id {s['id']})" for s in chosen))
    return [s["id"] for s in sorted(chosen, key=lambda s: to_date(s["starting_at"]))]


def load_season_structure(api, db, season_id):
    """Stages, rounds, groups, season statistics and the fixture list."""
    body = api.get_inc("season", f"/football/seasons/{season_id}",
                       ["stages", "rounds", "groups", "statistics", "fixtures"],
                       store=("season", season_id))
    s = body["data"]
    db.upsert("football.stages", [structure_row(o, "stage_id", ("type_id", "sort_order", "tie_breaker_rule_id"))
                                  for o in s.get("stages") or []])
    db.upsert("football.rounds", [structure_row(o, "round_id", ("stage_id",)) for o in s.get("rounds") or []])
    db.upsert("football.groups", [structure_row(o, "group_id", ("stage_id", "pending"))
                                  for o in s.get("groups") or []])
    db.upsert("football.season_statistics", [{
        "season_statistic_id": o.get("id"), "season_id": season_id, "type_id": o.get("type_id"),
        "relation_type": o.get("relation_type"), "relation_id": o.get("relation_id"),
        "data": o.get("value")} for o in s.get("statistics") or []])
    fixtures = s.get("fixtures") or []
    db.upsert("football.fixtures", [fixture_row(f) for f in fixtures])
    db.commit()
    return fixtures


def load_teams(api, db, season_ids):
    print("\n[4/8] Teams, squads, injuries, transfers")
    season_filter = ",".join(map(str, season_ids))
    team_seasons = {}
    for sid in season_ids:
        try:
            teams = api.pages(f"/football/teams/seasons/{sid}", {"include": "venue"})
        except ApiError as e:
            warn(f"teams for season {sid}: {e}")
            teams = []
        db.upsert("football.venues", [venue_row(t["venue"]) for t in teams if isinstance(t.get("venue"), dict)])
        db.upsert("football.teams", [team_row(t) for t in teams])
        for t in teams:
            team_seasons.setdefault(t["id"], []).append(sid)
        db.commit()
    print(f"  {len(team_seasons)} teams across the selected seasons")

    for i, (tid, sids) in enumerate(sorted(team_seasons.items()), 1):
        try:
            body = api.get_inc("team", f"/football/teams/{tid}",
                               ["venue", "statistics.details", "sidelined", "rivals",
                                "trophies.trophy|trophies", "coaches"],
                               {"filters": f"teamStatisticSeasons:{season_filter}"}, store=("team", tid))
            t = body["data"]
            db.upsert("football.teams", [team_row(t)])
            if isinstance(t.get("venue"), dict):
                db.upsert("football.venues", [venue_row(t["venue"])])
            parents, details = stat_rows(t.get("statistics"), None, "team_statistic_id", "team_id",
                                         None, "team_statistic_detail_id")
            db.upsert("football.team_season_stats", parents)
            db.upsert("football.team_season_stat_details", details)
            db.upsert("football.sidelined", [{
                "sidelined_id": o.get("id"), "player_id": o.get("player_id"), "team_id": o.get("team_id") or tid,
                "season_id": o.get("season_id"), "type_id": o.get("type_id"), "category": o.get("category"),
                "start_date": o.get("start_date"), "end_date": o.get("end_date"),
                "games_missed": o.get("games_missed"), "completed": o.get("completed")}
                for o in t.get("sidelined") or []])
            db.upsert("football.rivals", [{"team_id": tid, "rival_team_id": o.get("rival_id") or o.get("id")}
                                          for o in t.get("rivals") or []])
            db.upsert("football.trophies", [{
                "trophy_id": o["trophy"].get("id"), "position": o["trophy"].get("position"),
                "name": o["trophy"].get("name")} for o in t.get("trophies") or [] if isinstance(o.get("trophy"), dict)])
            db.upsert("football.team_trophies", [{
                "team_trophy_id": o.get("id"), "team_id": o.get("team_id") or o.get("participant_id") or tid,
                "league_id": o.get("league_id"), "season_id": o.get("season_id"),
                "trophy_id": o.get("trophy_id")} for o in t.get("trophies") or []])
            db.upsert("football.coach_teams", [{
                "coach_team_id": o.get("id"), "coach_id": o.get("coach_id"), "team_id": o.get("team_id") or tid,
                "position_id": o.get("position_id"), "active": o.get("active"), "temporary": o.get("temporary"),
                "start_date": o.get("start"), "end_date": o.get("end")} for o in t.get("coaches") or []])
        except ApiError as e:
            warn(f"team {tid}: {e}")

        for sid in sids:
            try:
                squad = api.pages(f"/football/squads/seasons/{sid}/teams/{tid}")
                db.upsert("football.team_squads", [{
                    "squad_id": o.get("id"), "team_id": o.get("team_id") or tid, "player_id": o.get("player_id"),
                    "season_id": sid, "transfer_id": o.get("transfer_id"), "position_id": o.get("position_id"),
                    "detailed_position_id": o.get("detailed_position_id"),
                    "jersey_number": o.get("jersey_number"), "captain": o.get("captain"),
                    "start_date": o.get("start"), "end_date": o.get("end")} for o in squad])
            except ApiError as e:
                warn(f"squad team {tid} season {sid}: {e}")

        try:
            transfers = api.pages(f"/football/transfers/teams/{tid}")
            db.upsert("football.transfers", [{
                "transfer_id": o.get("id"), "player_id": o.get("player_id"), "type_id": o.get("type_id"),
                "from_team_id": o.get("from_team_id"), "to_team_id": o.get("to_team_id"),
                "position_id": o.get("position_id"), "detailed_position_id": o.get("detailed_position_id"),
                "date": o.get("date"), "career_ended": o.get("career_ended"), "completed": o.get("completed"),
                "completed_at": o.get("completed_at"), "amount": o.get("amount")} for o in transfers])
        except ApiError as e:
            warn(f"transfers team {tid}: {e}")
        db.commit()
        print(f"  team {i}/{len(team_seasons)} done")


def save_fixture(db, f, pass_name):
    """Write one fixture's included data into the tables."""
    fid = f["id"]
    if pass_name == "main":
        db.upsert("football.fixtures", [fixture_row(f)])
        if isinstance(f.get("venue"), dict):
            db.upsert("football.venues", [venue_row(f["venue"])])

        parts = f.get("participants") or []
        db.upsert("football.teams", [team_row(p) for p in parts])
        db.upsert("football.fixture_participants", [{
            "fixture_id": fid, "team_id": p.get("id"), "location": g(p, "meta", "location"),
            "winner": g(p, "meta", "winner"), "position": g(p, "meta", "position")} for p in parts])

        db.upsert("football.fixture_scores", [{
            "score_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "type_id": o.get("type_id"), "goals": g(o, "score", "goals"),
            "location": g(o, "score", "participant"), "description": o.get("description")}
            for o in f.get("scores") or []])

        db.upsert("football.fixture_periods", [{
            "period_id": o.get("id"), "fixture_id": fid, "type_id": o.get("type_id"),
            "started": o.get("started"), "ended": o.get("ended"), "counts_from": o.get("counts_from"),
            "ticking": o.get("ticking"), "sort_order": o.get("sort_order"), "description": o.get("description"),
            "time_added": o.get("time_added"), "period_length": o.get("period_length"),
            "minutes": o.get("minutes"), "seconds": o.get("seconds"), "has_timer": o.get("has_timer")}
            for o in f.get("periods") or []])

        db.upsert("football.fixture_events", [{
            "event_id": o.get("id"), "fixture_id": fid, "period_id": o.get("period_id"),
            "detailed_period_id": o.get("detailed_period_id"), "team_id": o.get("participant_id"),
            "type_id": o.get("type_id"), "sub_type_id": o.get("sub_type_id"), "section": o.get("section"),
            "player_id": o.get("player_id"), "related_player_id": o.get("related_player_id"),
            "coach_id": o.get("coach_id"), "player_name": o.get("player_name"),
            "related_player_name": o.get("related_player_name"), "result": o.get("result"),
            "info": o.get("info"), "addition": o.get("addition"), "minute": o.get("minute"),
            "extra_minute": o.get("extra_minute"), "injured": o.get("injured"), "on_bench": o.get("on_bench"),
            "rescinded": o.get("rescinded"), "sort_order": o.get("sort_order")}
            for o in f.get("events") or []])

        lineups, details, xg_players = [], [], []
        for o in f.get("lineups") or []:
            lineups.append({
                "lineup_id": o.get("id"), "fixture_id": fid, "team_id": o.get("team_id"),
                "player_id": o.get("player_id"), "type_id": o.get("type_id"), "position_id": o.get("position_id"),
                "detailed_position_id": o.get("detailed_position_id"), "formation_field": o.get("formation_field"),
                "formation_position": o.get("formation_position"), "jersey_number": o.get("jersey_number"),
                "player_name": o.get("player_name")})
            for d in o.get("details") or []:
                details.append({
                    "lineup_detail_id": d.get("id"), "lineup_id": o.get("id"), "fixture_id": fid,
                    "team_id": d.get("team_id") or o.get("team_id"),
                    "player_id": d.get("player_id") or o.get("player_id"),
                    "type_id": d.get("type_id"), "data": d.get("data")})
            for x in o.get("xgLineup") or o.get("xglineup") or o.get("xGLineup") or []:
                xg_players.append({
                    "xg_player_id": x.get("id"), "fixture_id": fid, "team_id": x.get("team_id") or o.get("team_id"),
                    "player_id": x.get("player_id") or o.get("player_id"), "lineup_id": o.get("id"),
                    "type_id": x.get("type_id"), "data": x.get("data")})
        db.upsert("football.fixture_lineups", lineups)
        db.upsert("football.fixture_lineup_details", details)
        db.upsert("football.fixture_xg_players", xg_players)

        db.upsert("football.fixture_statistics", [{
            "statistic_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "type_id": o.get("type_id"), "location": o.get("location"), "data": o.get("data")}
            for o in f.get("statistics") or []])

        db.upsert("football.fixture_formations", [{
            "formation_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "formation": o.get("formation"), "location": o.get("location")}
            for o in f.get("formations") or []])

        db.upsert("football.fixture_officials", [{
            "fixture_official_id": o.get("id"), "fixture_id": fid, "referee_id": o.get("referee_id"),
            "type_id": o.get("type_id")} for o in f.get("referees") or [] if o.get("referee_id")])

        coaches = f.get("coaches") or []
        db.upsert("football.coaches", [{**person_row(c, "coach_id"), "player_id": c.get("player_id")}
                                       for c in coaches])
        db.upsert("football.fixture_coaches", [{
            "fixture_id": fid, "coach_id": c.get("id"),
            "team_id": g(c, "meta", "participant_id") or c.get("participant_id")} for c in coaches])

        tv = f.get("tvStations") or f.get("tvstations") or []
        db.upsert("football.tv_stations", [{
            "tv_station_id": g(o, "tvstation", "id"), "name": g(o, "tvstation", "name"),
            "url": g(o, "tvstation", "url"), "image_path": g(o, "tvstation", "image_path")} for o in tv])
        db.upsert("football.fixture_tv_stations", [{
            "fixture_tv_station_id": o.get("id"), "fixture_id": fid,
            "tv_station_id": o.get("tvstation_id") or o.get("tv_station_id"),
            "country_id": o.get("country_id")} for o in tv if o.get("tvstation_id") or o.get("tv_station_id")])

        w = f.get("weatherReport") or f.get("weatherreport")
        if isinstance(w, dict):
            db.upsert("football.fixture_weather", [{
                "weather_id": w.get("id"), "fixture_id": fid, "venue_id": w.get("venue_id"),
                "temperature_day": g(w, "temperature", "day"), "temperature_morning": g(w, "temperature", "morning"),
                "temperature_evening": g(w, "temperature", "evening"),
                "temperature_night": g(w, "temperature", "night"), "feels_like_day": g(w, "feels_like", "day"),
                "wind_speed": g(w, "wind", "speed"), "wind_direction": g(w, "wind", "direction"),
                "humidity": w.get("humidity"), "pressure": w.get("pressure"), "clouds": w.get("clouds"),
                "description": w.get("description"), "icon": w.get("icon"), "type": w.get("type"),
                "metric": w.get("metric"), "data": w}])

        side = f.get("sidelined") or []
        db.upsert("football.sidelined", [{
            "sidelined_id": s.get("id"), "player_id": s.get("player_id"), "team_id": s.get("team_id"),
            "season_id": s.get("season_id"), "type_id": s.get("type_id"), "category": s.get("category"),
            "start_date": s.get("start_date"), "end_date": s.get("end_date"),
            "games_missed": s.get("games_missed"), "completed": s.get("completed")}
            for s in (o.get("sideline") for o in side) if isinstance(s, dict)])
        db.upsert("football.fixture_sidelined", [{
            "fixture_sidelined_id": o.get("id"), "fixture_id": fid,
            "sidelined_id": o.get("sideline_id") or g(o, "sideline", "id"),
            "team_id": o.get("participant_id")} for o in side])

        db.upsert("football.fixture_xg", [{
            "xg_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "type_id": o.get("type_id"), "location": o.get("location"), "data": o.get("data")}
            for o in f.get("xgfixture") or f.get("xGFixture") or []])

        db.upsert("football.fixture_expected_lineups", [{
            "expected_lineup_id": o.get("id"), "fixture_id": fid,
            "team_id": o.get("team_id") or o.get("participant_id"), "player_id": o.get("player_id"),
            "type_id": o.get("type_id"), "position_id": o.get("position_id"),
            "formation_field": o.get("formation_field"), "formation_position": o.get("formation_position"),
            "jersey_number": o.get("jersey_number"), "data": o}
            for o in f.get("expectedLineups") or f.get("expectedlineups") or []])

        db.upsert("football.metadata", [{
            "metadata_id": o.get("id"), "entity_type": o.get("metadatable_type") or "fixture",
            "entity_id": o.get("metadatable_id") or fid, "type_id": o.get("type_id"),
            "value_type": o.get("value_type"), "value": o.get("values", o.get("value"))}
            for o in f.get("metadata") or []])

        db.upsert("football.predictions", [{
            "prediction_id": o.get("id"), "fixture_id": fid, "type_id": o.get("type_id"),
            "predictions": o.get("predictions")} for o in f.get("predictions") or []])

        news, lines = [], []
        for key, kind in (("prematchNews", "prematch"), ("prematchnews", "prematch"),
                          ("postmatchNews", "postmatch"), ("postmatchnews", "postmatch")):
            for n in f.get(key) or []:
                news.append({"news_id": n.get("id"), "fixture_id": fid, "league_id": n.get("league_id"),
                             "title": n.get("title"), "type": n.get("type") or kind})
                for ln in n.get("lines") or []:
                    lines.append({"news_line_id": ln.get("id"), "news_id": n.get("id"),
                                  "text": ln.get("text"), "type": ln.get("type")})
        db.upsert("football.fixture_news", news)
        db.upsert("football.fixture_news_lines", lines)

    elif pass_name == "live":
        db.upsert("football.fixture_trends", [{
            "trend_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "type_id": o.get("type_id"), "period_id": o.get("period_id"), "minute": o.get("minute"),
            "value": o.get("value")} for o in f.get("trends") or []])
        db.upsert("football.fixture_pressure", [{
            "pressure_id": o.get("id"), "fixture_id": fid, "team_id": o.get("participant_id"),
            "minute": o.get("minute"), "pressure": o.get("pressure")} for o in f.get("pressure") or []])
        db.upsert("football.fixture_commentaries", [{
            "commentary_id": o.get("id"), "fixture_id": fid, "player_id": o.get("player_id"),
            "related_player_id": o.get("related_player_id"), "comment": o.get("comment"),
            "minute": o.get("minute"), "extra_minute": o.get("extra_minute"), "is_goal": o.get("is_goal"),
            "is_important": o.get("is_important"), "sort_order": o.get("order")}
            for o in f.get("comments") or []])
        db.upsert("football.fixture_match_facts", [{
            "match_fact_id": o.get("id"), "fixture_id": fid,
            "team_id": o.get("participant_id") or o.get("team_id"), "type_id": o.get("type_id"), "data": o}
            for o in f.get("matchfacts") or f.get("matchFacts") or []])

    elif pass_name == "odds":
        db.upsert("odds.prematch_odds", [odd_row(o, fid) for o in f.get("odds") or []])
        premium = f.get("premiumOdds") or f.get("premiumodds") or []
        db.upsert("odds.premium_odds", [odd_row(o, fid) for o in premium])
        db.upsert("odds.premium_odds_history", [{
            "history_id": h.get("id"), "odd_id": h.get("odd_id") or o.get("id"), "value": h.get("value"),
            "probability": h.get("probability"), "dp3": h.get("dp3"), "fractional": h.get("fractional"),
            "american": h.get("american"), "bookmaker_update": h.get("bookmaker_update")}
            for o in premium for h in o.get("history") or []])

    elif pass_name == "ball":
        db.upsert("football.fixture_ball_coordinates", [{
            "ball_coordinate_id": o.get("id"), "fixture_id": fid, "period_id": o.get("period_id"),
            "timer": o.get("timer"), "x": o.get("x"), "y": o.get("y")}
            for o in f.get("ballCoordinates") or f.get("ballcoordinates") or []])


def load_fixtures(api, db, fixtures):
    print(f"\n[5/8] Fixtures ({len(fixtures)} matches)")
    final_ids = {r[0] for r in db.query(
        "SELECT state_id FROM core.states WHERE developer_name = ANY(%s)", (list(FINAL_STATES),))}
    state_of = {r[0]: r[1] for r in db.query(
        "SELECT fixture_id, state_id FROM football.fixtures WHERE fixture_id = ANY(%s)",
        ([f["id"] for f in fixtures],))}

    for pass_name, cfg in FIXTURE_PASSES.items():
        if pass_name == "odds" and not FETCH_ODDS:
            continue
        if pass_name == "ball" and not FETCH_BALL_COORDINATES:
            continue
        done = db.get_sync(f"fixture_{pass_name}")
        todo = [f["id"] for f in fixtures
                if not (str(f["id"]) in done and done[str(f["id"])][1] == "ok"
                        and (done[str(f["id"])][2] or {}).get("final"))]
        print(f"  pass '{pass_name}': {len(fixtures) - len(todo)} already complete, {len(todo)} to fetch")
        for start in range(0, len(todo), cfg["batch"]):
            batch = todo[start:start + cfg["batch"]]
            path = "/football/fixtures/multi/" + ",".join(map(str, batch))
            try:
                body = api.get_inc(f"fixtures_{pass_name}", path, cfg["includes"],
                                   store=("fixtures", None) if cfg["raw"] else None)
            except ApiError as e:
                warn(f"fixtures {batch} pass {pass_name}: {e}")
                continue
            for f in body.get("data") or []:
                save_fixture(db, f, pass_name)
                state = f.get("state_id") or state_of.get(f["id"])
                db.set_sync(f"fixture_{pass_name}", f["id"], "ok", {"final": state in final_ids})
            db.commit()
            print(f"    {min(start + cfg['batch'], len(todo))}/{len(todo)}")


def load_standings(api, db, season_id):
    try:
        body = api.get_inc("standings", f"/football/standings/seasons/{season_id}", ["details", "form"],
                           store=("standings", season_id))
        rows = body.get("data") or []
        db.upsert("football.standings", [{
            "standing_id": o.get("id"), "league_id": o.get("league_id"), "season_id": o.get("season_id") or season_id,
            "stage_id": o.get("stage_id"), "group_id": o.get("group_id"), "round_id": o.get("round_id"),
            "team_id": o.get("participant_id"), "standing_rule_id": o.get("standing_rule_id"),
            "position": o.get("position"), "result": o.get("result"), "points": o.get("points")} for o in rows])
        db.upsert("football.standing_details", [{
            "standing_detail_id": d.get("id"), "standing_id": o.get("id"), "team_id": o.get("participant_id"),
            "type_id": d.get("type_id"), "value": d.get("value")} for o in rows for d in o.get("details") or []])
        db.upsert("football.standing_form", [{
            "standing_form_id": d.get("id"), "standing_id": o.get("id"), "team_id": o.get("participant_id"),
            "fixture_id": d.get("fixture_id"), "form": d.get("form"), "sort_order": d.get("sort_order")}
            for o in rows for d in o.get("form") or []])
    except ApiError as e:
        warn(f"standings season {season_id}: {e}")
    try:
        tops = api.pages(f"/football/topscorers/seasons/{season_id}")
        db.upsert("football.topscorers", [{
            "topscorer_id": o.get("id"), "season_id": o.get("season_id") or season_id, "stage_id": o.get("stage_id"),
            "player_id": o.get("player_id"), "team_id": o.get("participant_id"), "type_id": o.get("type_id"),
            "position": o.get("position"), "total": o.get("total")} for o in tops])
    except ApiError as e:
        warn(f"topscorers season {season_id}: {e}")
    db.commit()


def load_people(api, db, season_ids):
    print("\n[7/8] Players, referees, coaches + season statistics")
    seasons = ",".join(map(str, season_ids))
    groups = [
        ("player", "football.players", "player_id",
         "SELECT player_id FROM football.fixture_lineups WHERE season_id = ANY(%(s)s) "
         "UNION SELECT player_id FROM football.fixture_events WHERE season_id = ANY(%(s)s) "
         "UNION SELECT related_player_id FROM football.fixture_events WHERE season_id = ANY(%(s)s) "
         "UNION SELECT player_id FROM football.team_squads WHERE season_id = ANY(%(s)s) "
         "UNION SELECT player_id FROM football.sidelined WHERE season_id = ANY(%(s)s) "
         "UNION SELECT player_id FROM football.topscorers WHERE season_id = ANY(%(s)s)",
         "/football/players/{}", "playerStatisticSeasons"),
        ("referee", "football.referees", "referee_id",
         "SELECT referee_id FROM football.fixture_officials WHERE season_id = ANY(%(s)s)",
         "/football/referees/{}", "refereeStatisticSeasons"),
        ("coach", "football.coaches", "coach_id",
         "SELECT coach_id FROM football.fixture_coaches WHERE season_id = ANY(%(s)s) "
         "UNION SELECT coach_id FROM football.coach_teams WHERE end_date IS NULL OR end_date >= "
         "(SELECT min(starting_at) FROM football.seasons WHERE season_id = ANY(%(s)s))",
         "/football/coaches/{}", "coachStatisticSeasons"),
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(days=PEOPLE_REFRESH_DAYS)
    for kind, table, id_col, id_sql, path, filt in groups:
        ids = sorted({int(r[0]) for r in db.query(id_sql, {"s": season_ids}) if r[0]})
        done = db.get_sync(kind)
        todo = []
        for pid in ids:
            d = done.get(str(pid))
            if d and d[1] == "ok" and (d[2] or {}).get("seasons") == seasons \
                    and to_ts(d[0]) and to_ts(d[0]) > cutoff:
                continue
            todo.append(pid)
        label = {"player": "players", "referee": "referees", "coach": "coaches"}[kind]
        print(f"  {label}: {len(ids) - len(todo)} up to date, {len(todo)} to fetch")

        for i, pid in enumerate(todo, 1):
            try:
                body = api.get_inc(kind, path.format(pid), ["statistics.details"],
                                   {"filters": f"{filt}:{seasons}"}, store=(kind, pid))
            except ApiError as e:
                warn(f"{kind} {pid}: {e}")
                db.set_sync(kind, pid, "error", {"error": str(e)})
                continue
            o = body.get("data") or {}
            row = person_row(o, id_col)
            if kind == "player":
                row.update({"position_id": o.get("position_id"),
                            "detailed_position_id": o.get("detailed_position_id"), "type_id": o.get("type_id")})
            if kind == "coach":
                row["player_id"] = o.get("player_id")
            db.upsert(table, [row])

            stats_table = {"player": "player", "referee": "referee", "coach": "coach"}[kind]
            parents, details = stat_rows(o.get("statistics"), None, f"{stats_table}_statistic_id", id_col,
                                         None, f"{stats_table}_statistic_detail_id")
            db.upsert(f"football.{stats_table}_season_stats", parents)
            db.upsert(f"football.{stats_table}_season_stat_details", details)
            db.set_sync(kind, pid, "ok", {"seasons": seasons})
            if i % 25 == 0 or i == len(todo):
                db.commit()
                print(f"    {i}/{len(todo)}")
        db.commit()


# =================================================================== MAIN ===
def main():
    parser = argparse.ArgumentParser(description="Load Bundesliga data from Sportmonks")
    parser.add_argument("--skip-teams", action="store_true", help="skip step 4")
    parser.add_argument("--skip-people", action="store_true", help="skip step 7")
    args = parser.parse_args()

    started = time.time()
    db = Db(DATABASE)
    db.columns("football.fixtures")      # confirms the schema is loaded
    api = Api(API_TOKEN, db)

    load_core(api, db)
    season_ids = load_league_and_seasons(api, db)

    print("\n[3/8] Season structure and fixture lists")
    fixtures = []
    for sid in season_ids:
        fx = load_season_structure(api, db, sid)
        print(f"  season {sid}: {len(fx)} fixtures")
        fixtures.extend(fx)

    if not args.skip_teams:
        load_teams(api, db, season_ids)

    load_fixtures(api, db, fixtures)

    print("\n[6/8] Standings and top scorers")
    for sid in season_ids:
        load_standings(api, db, sid)
    print("  done")

    if not args.skip_people:
        load_people(api, db, season_ids)

    print("\n[8/8] Refreshing analysis views")
    db.execute("SELECT analysis.refresh_all()")
    db.commit()

    print("\nSUMMARY")
    for table in ("football.fixtures", "football.fixture_events", "football.fixture_lineup_details",
                  "football.fixture_statistics", "football.fixture_officials", "football.players",
                  "football.referees", "odds.prematch_odds", "odds.premium_odds",
                  "football.fixture_ball_coordinates", "analysis.referee_match_discipline"):
        print(f"  {table:<40}{db.query(f'SELECT count(*) FROM {table}')[0][0]:>12,}")
    missing = db.query("SELECT ref_tbl, count(DISTINCT missing_id) FROM raw.missing_references() GROUP BY 1")
    if missing:
        print("  IDs referenced but not downloaded: " + ", ".join(f"{t} {n}" for t, n in missing))
    print(f"\n  {api.calls} API calls in {(time.time() - started) / 60:.1f} minutes")
    if WARNINGS:
        print(f"  {len(WARNINGS)} warnings (shown above with '!')")


if __name__ == "__main__":
    main()
