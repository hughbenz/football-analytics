#!/usr/bin/env python3
"""
TEMPLATE: load the last N seasons of one or more leagues from Sportmonks into
the football database (the PostgreSQL schema from schema.sql).

CHOOSE LEAGUES
  Edit the LEAGUES list below, or give league ids on the command line:
      python load_league.py              # every league in LEAGUES
      python load_league.py 8 384        # just Premier League + Serie A

LOAD ORDER (each step builds the IDs the next one depends on)
  1. Core reference data   (once) types, states, continents, countries,
                           bookmakers, markets
  then for each league:
  2. League + seasons
  3. Season structure      stages, rounds, groups, season statistics, fixture list
  4. Teams                 teams, venues, team season stats, squads, injuries,
                           rivals, trophies, coaching spells, transfers
  5. Fixtures              separate passes, so includes Sportmonks won't combine
                           (e.g. odds + premiumOdds) never share a request:
                             main          participants, scores, events, lineups +
                                           player stats, team stats, officials,
                                           coaches, xG, weather, injuries,
                                           predictions, news, TV stations ...
                             live          trends, pressure index, commentary,
                                           match facts
                             odds          standard pre-match odds
                             premium_odds  premium odds + price history
                             inplay_odds   in-play odds
                             ball          ball coordinates (large)
  6. Standings + top scorers
  7. People                players (+ transfers), referees, coaches (+ clubs),
                           with season statistics
  8. Refresh analysis views (once, at the end)

  If any other includes turn out not to combine, the script notices, fetches
  them in separate requests and merges the results automatically.

SETUP (once)
  pip install "psycopg[binary]"
  Fill in the DATABASE settings below (password etc.).

RUN
  python load_league.py

  Re-running is safe: finished matches already downloaded are skipped and
  unfinished/upcoming ones are refreshed. If a run stops, run it again.

  Re-download one pass for matches already loaded (league ids first):
      python load_league.py 82 --refetch main

API CALL BUDGET
  Every call is counted against CALL_BUDGET (stored in the database, so it
  carries across runs). The script stops cleanly before the budget runs out
  and resumes next time. See how many calls a run will need first with:
      python load_league.py 82 564 8 --estimate
  Ways it keeps calls down: large batches (shrunk automatically if Sportmonks
  refuses), player season stats fetched per squad instead of per player,
  data types that turn out to be empty for a league are skipped, finished
  seasons are never re-downloaded, and current-season data refreshes at most
  every REFRESH_DAYS.
"""

import argparse
import http.client
import json
import ssl
import math
import threading
import os
import shutil
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

# Sportmonks league ids to load. Remove the # to add a league. Each league's
# name is printed when it loads, so you can confirm the id. Your Sportmonks
# plan must include the league.
LEAGUES = [
    82,      # Germany      Bundesliga
    564,     # Spain        La Liga
    # 8,     # England      Premier League
    # 9,     # England      Championship
    # 384,   # Italy        Serie A
    # 301,   # France       Ligue 1
    # 72,    # Netherlands  Eredivisie
    # 462,   # Portugal     Liga Portugal
    # 501,   # Scotland     Premiership
    # 208,   # Belgium      Pro League
    # 600,   # Turkey       Super Lig
    # 271,   # Denmark      Superliga
    # 2,     # Europe       Champions League
    # 5,     # Europe       Europa League
]

NUM_SEASONS     = 3       # most recent seasons
INCLUDE_CURRENT = True    # True: 2024/25, 2025/26, 2026/27 (in progress)
                          # False: last three finished seasons

FETCH_ODDS              = True   # standard pre-match odds
FETCH_PREMIUM_ODDS      = True   # premium odds + price history
FETCH_INPLAY_ODDS       = True   # in-play odds
FETCH_BALL_COORDINATES  = True   # largest data set; set False to skip
STORE_RAW               = True   # keep raw API responses (not odds / ball data)
FETCH_TEAM_TRANSFERS    = True   # each club's transfer history (once per club)

# ---- API call budget -------------------------------------------------------
CALL_BUDGET     = 50_000     # calls your Sportmonks plan allows per period
BUDGET_PERIOD   = "hour"     # "hour"  = allowance resets every hour: when it's used up
                             #           the run PAUSES until the next hour, then carries on
                             # "month" = resets each calendar month (run stops when used up)
                             # "total" = one allowance that never resets
BUDGET_RESERVE  = 300        # keep this many calls spare in each period
# ---- Storage safety ------------------------------------------------------
MAX_DB_SIZE_GB   = 70        # stop safely when the database reaches this size
MIN_FREE_DISK_GB = 10        # ...or when the disk it lives on has less free space
SIZE_CHECK_EVERY = 50        # API calls between size checks

NETWORK_WAIT_MINUTES = 30   # if the internet drops, wait this long for it to come back
REFRESH_DAYS    = 7          # current-season squads/stats/referees refresh at
                             # most this often (finished seasons: never)

# Download IDs that other data points to but that weren't loaded.
# Cheap ones on; the big ones (from clubs' full transfer histories) off.
FILL_MISSING = {
    "coaches": True,     # ~20-30 per league
    "venues":  True,     # a handful
    "leagues": True,     # leagues named in trophy records (+ their seasons)
    "teams":   False,    # ~800+ per league: every club in transfer histories
    "players": False,    # ~1,700+ per league: every player in transfer histories
}

API_BASE = "https://api.sportmonks.com/v3"

# Fixture include passes. Each pass is a separate request per batch.
# "a|b" = try a, fall back to b if a isn't available.
FIXTURE_PASSES = {
    "main": {
        "enabled": True, "batch": 25, "raw": True,
        "includes": [
            "participants", "scores", "periods", "events", "venue",
            "lineups.details", "lineups.xGLineup", "formations", "statistics",
            "referees", "coaches", "tvStations.tvStation|tvStations", "weatherReport",
            "sidelined.sideline", "xGFixture", "expectedLineups", "metadata",
            "predictions", "prematchNews.lines", "postmatchNews.lines",
        ],
    },
    "live": {
        "enabled": True, "batch": 25, "raw": True,
        "includes": ["trends", "pressure", "comments", "matchfacts|matchFacts"],
    },
    "odds": {
        "enabled": FETCH_ODDS, "batch": 10, "raw": False,
        "includes": ["odds"],
    },
    "premium_odds": {
        "enabled": FETCH_PREMIUM_ODDS, "batch": 5, "raw": False,
        "includes": ["premiumOdds.history|premiumOdds"],
    },
    "inplay_odds": {
        "enabled": FETCH_INPLAY_ODDS, "batch": 10, "raw": False,
        "includes": ["inplayOdds"],
    },
    "ball": {
        "enabled": FETCH_BALL_COORDINATES, "batch": 25, "raw": False,
        "includes": ["ballCoordinates"],
    },
}

# Where each optional pass puts its data, to spot passes that come back empty.
PASS_KEYS = {
    "live":         ["trends", "pressure", "comments", "matchfacts", "matchFacts"],
    "odds":         ["odds"],
    "premium_odds": ["premiumOdds", "premiumodds"],
    "inplay_odds":  ["inplayOdds", "inplayodds"],
    "ball":         ["ballCoordinates", "ballcoordinates"],
}
EMPTY_CHECK = 30     # finished matches checked before an empty pass is skipped

PEOPLE_INCLUDES = {
    "player":  ["statistics.details", "transfers"],
    "referee": ["statistics.details"],
    "coach":   ["statistics.details", "teams"],
}

FINAL_STATES = {"FT", "AET", "FT_PEN", "AWARDED", "CANCELLED", "ABANDONED", "DELETED"}

WARNINGS = []


def warn(msg):
    print("  ! " + msg)
    WARNINGS.append(msg)


class BudgetExhausted(Exception):
    """Stops the run cleanly; progress is saved and the next run continues."""


class NetworkDown(BudgetExhausted):
    """No internet connection for NETWORK_WAIT_MINUTES - stop cleanly."""


class StorageLimitReached(BudgetExhausted):
    """Database reached MAX_DB_SIZE_GB or the disk is nearly full - stop cleanly."""


class SizeGuard:
    """Checks database size (and free disk space where PostgreSQL lets us see
    its data folder) every SIZE_CHECK_EVERY API calls. Once a limit is hit,
    every worker stops at its next call and nothing more is downloaded."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tripped = None
        self.calls = 0
        self.data_dir = None
        self.last_gb = None

    def check(self, db, force=False):
        with self.lock:
            if self.tripped:
                raise StorageLimitReached(self.tripped)
            self.calls += 1
            if not force and self.calls % SIZE_CHECK_EVERY:
                return
        gb = db.query("SELECT pg_database_size(current_database())")[0][0] / 1024 ** 3
        free = self._free_gb(db)
        msg = None
        if MAX_DB_SIZE_GB and gb >= MAX_DB_SIZE_GB:
            msg = f"database size limit reached: {gb:.1f} GB (limit {MAX_DB_SIZE_GB} GB)"
        elif MIN_FREE_DISK_GB and free is not None and free < MIN_FREE_DISK_GB:
            msg = f"disk nearly full: {free:.1f} GB free (minimum {MIN_FREE_DISK_GB} GB)"
        with self.lock:
            self.last_gb = gb
            if msg and not self.tripped:
                self.tripped = msg
        if msg:
            raise StorageLimitReached(msg)

    def _free_gb(self, db):
        if self.data_dir is None:
            try:
                self.data_dir = db.query("SELECT current_setting('data_directory')")[0][0] or ""
            except Exception:                       # not allowed to see it: size limit only
                try:
                    db.conn.rollback()
                except Exception:
                    pass
                self.data_dir = ""
        if not self.data_dir:
            return None
        try:
            return shutil.disk_usage(self.data_dir).free / 1024 ** 3
        except OSError:
            return None


SIZE_GUARD = SizeGuard()


class Budget:
    """Counts every API request against CALL_BUDGET. The count is stored in
    raw.sync_state so it carries across runs (and is shared by threads)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.used = None
        self.session = 0
        self.key = self.period_key()

    @staticmethod
    def period_key():
        if BUDGET_PERIOD == "hour":
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        if BUDGET_PERIOD == "month":
            return date.today().strftime("%Y-%m")
        return "all-time"

    def load(self, db):
        with self.lock:
            if self.used is None:
                rows = db.query("SELECT detail FROM raw.sync_state WHERE entity = 'api_budget' "
                                "AND scope_key = %s", (self.key,))
                self.used = int(((rows[0][0] if rows else None) or {}).get("calls", 0))

    @property
    def remaining(self):
        return CALL_BUDGET - BUDGET_RESERVE - (self.used or 0)

    def spend(self):
        while True:
            with self.lock:
                key = self.period_key()
                if key != self.key:                     # new hour/month: fresh allowance
                    self.key, self.used = key, 0
                if self.used is None:
                    self.used = 0
                if self.used + 1 <= CALL_BUDGET - BUDGET_RESERVE:
                    self.used += 1
                    self.session += 1
                    return self.session
                if BUDGET_PERIOD != "hour":
                    raise BudgetExhausted(
                        f"API call budget reached ({self.used:,} of {CALL_BUDGET:,} used, "
                        f"{BUDGET_RESERVE} kept in reserve)")
                now = datetime.now(timezone.utc)
                wait = 3600 - (now.minute * 60 + now.second) + 5
            print(f"  hourly API allowance used ({CALL_BUDGET - BUDGET_RESERVE:,} calls) - "
                  f"pausing {wait // 60} min until the next hour, then continuing")
            time.sleep(wait)

    def refund(self):
        """A request that never reached Sportmonks (no network) isn't counted."""
        with self.lock:
            if self.used:
                self.used -= 1
            if self.session:
                self.session -= 1

    def save(self, db, force=False):
        """Store the count. Normally never lowers a stored count (so parallel
        workers can't overwrite each other with an older number); force=True
        sets it exactly, used by --used-so-far."""
        with self.lock:
            used = self.used or 0
        keep_higher = ("CASE WHEN coalesce((raw.sync_state.detail->>'calls')::int, 0) "
                       "> (EXCLUDED.detail->>'calls')::int THEN raw.sync_state.detail ELSE EXCLUDED.detail END")
        db.execute(
            "INSERT INTO raw.sync_state (entity, scope_key, last_synced_at, status, detail) "
            "VALUES ('api_budget', %s, now(), 'ok', %s) ON CONFLICT (entity, scope_key) DO UPDATE SET "
            "last_synced_at = now(), detail = " + ("EXCLUDED.detail" if force else keep_higher),
            (self.key, Jsonb({"calls": used, "budget": CALL_BUDGET})))
        db.commit()


BUDGET = Budget()


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
        self.split = {}

    def get(self, path, params=None, store=None, retries=6):
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        url = API_BASE + path + ("?" + urllib.parse.urlencode(params, safe=";,:|.") if params else "")
        req = urllib.request.Request(url, headers={"Authorization": self.token, "Accept": "application/json"})

        server_errors = transient = attempt = 0
        offline_since = None
        while attempt < retries:
            SIZE_GUARD.check(self.db)
            if BUDGET.spend() % 25 == 0:
                BUDGET.save(self.db)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.load(resp)
                self.calls += 1
                if offline_since is not None:
                    print(f"  connection is back after {int(time.time() - offline_since)}s - continuing")
                    offline_since = None
            except urllib.error.HTTPError as e:
                attempt += 1
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
                    server_errors += 1
                    if server_errors >= 3:        # likely too much data in one request
                        raise ApiError(e.code, "server error (request may be too large)")
                    time.sleep(2 ** attempt)
                    continue
                try:
                    message = json.loads(text).get("message", text)
                except Exception:
                    message = text
                if e.code == 401:
                    sys.exit(f"Sportmonks rejected the API token (401): {message}")
                raise ApiError(e.code, str(message)[:300])
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, ValueError) as e:
                BUDGET.refund()                         # never reached Sportmonks / no usable answer
                reason = getattr(e, "reason", e)
                if isinstance(reason, ssl.SSLError):
                    sys.exit(f"Secure connection failed ({reason}).\nOn a Mac with Homebrew Python run:\n"
                             '  export SSL_CERT_FILE="$(brew --prefix)/etc/ca-certificates/cert.pem"')
                timed_out = isinstance(e, (TimeoutError, http.client.HTTPException, ValueError)) \
                    or isinstance(reason, TimeoutError) or "timed out" in str(reason)
                if timed_out:                           # slow or cut-off response: retry a few times
                    transient += 1
                    if transient >= 3:
                        raise ApiError(0, "response timed out or was cut off (request may be too large)")
                    print(f"  slow/cut-off response ({reason}), retrying")
                    time.sleep(5)
                    continue
                # no connection at all: wait for the internet to come back
                if offline_since is None:
                    offline_since = time.time()
                    print(f"  network problem ({reason}) - waiting for the connection to come back "
                          f"(up to {NETWORK_WAIT_MINUTES} min; nothing is lost)")
                if time.time() - offline_since > NETWORK_WAIT_MINUTES * 60:
                    raise NetworkDown(f"no internet connection for {NETWORK_WAIT_MINUTES} minutes")
                time.sleep(15)
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
        """GET with includes.
        - If Sportmonks rejects an include (not on your plan / different name),
          find which ones work, remember that, and continue without the rest.
        - If the working includes can't be combined in one request (like odds
          + premiumOdds), fetch each separately and merge the results."""
        params = dict(params or {})
        accepted = self.include_cache.get(kind)
        if accepted is None:
            first_choice = [spec.split("|")[0] for spec in includes]
            try:
                body = self.get(path, {**params, "include": ";".join(first_choice)}, store)
                self.include_cache[kind] = first_choice
                return body
            except ApiError as err:
                if err.status == 0 or err.status >= 500:
                    raise                               # connection/server trouble, not a bad include
                print(f"  {kind}: request failed ({err}); checking includes one by one")
                accepted, dropped = [], []
                for spec in includes:
                    for alt in spec.split("|"):
                        try:
                            self.get(path, {**params, "include": alt})
                            accepted.append(alt)
                            break
                        except ApiError as probe_err:
                            if probe_err.status == 0 or probe_err.status >= 500:
                                raise                   # can't judge includes without a working connection
                            dropped.append(alt)
                if not accepted:
                    raise err
                if dropped:
                    warn(f"{kind}: not available / skipped includes: {', '.join(dropped)}")
                self.include_cache[kind] = accepted

        if not self.split.get(kind):
            try:
                return self.get(path, {**params, "include": ";".join(accepted)}, store)
            except ApiError as err:
                if len(accepted) < 2 or err.status == 0 or err.status >= 500:
                    raise
                warn(f"{kind}: includes can't be combined ({err}); fetching them separately from now on")
                self.split[kind] = True

        merged = None
        for inc in accepted:
            merged = merge_bodies(merged, self.get(path, {**params, "include": inc}, store))
        return merged


def merge_bodies(a, b):
    """Merge two responses for the same request made with different includes."""
    if a is None:
        return b
    da, db_ = a.get("data"), b.get("data")
    if isinstance(da, list) and isinstance(db_, list):
        by_id = {o.get("id"): o for o in da if isinstance(o, dict)}
        for o in db_:
            if isinstance(o, dict) and o.get("id") in by_id:
                by_id[o["id"]].update(o)
            else:
                da.append(o)
    elif isinstance(da, dict) and isinstance(db_, dict):
        da.update(db_)
    return a


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


def player_row(o):
    row = person_row(o, "player_id")
    row.update({"position_id": o.get("position_id"), "detailed_position_id": o.get("detailed_position_id"),
                "type_id": o.get("type_id")})
    return row


def squad_row(o, team_id, season_id):
    return {"squad_id": o.get("id"), "team_id": o.get("team_id") or team_id, "player_id": o.get("player_id"),
            "season_id": season_id, "transfer_id": o.get("transfer_id"), "position_id": o.get("position_id"),
            "detailed_position_id": o.get("detailed_position_id"), "jersey_number": o.get("jersey_number"),
            "captain": o.get("captain"), "start_date": o.get("start"), "end_date": o.get("end")}


def needs_refresh(rec, finished):
    """rec = (last_synced_at, status, detail) from raw.sync_state, or None."""
    if not rec or rec[1] != "ok":
        return True
    if finished:
        return False
    synced = to_ts(rec[0])
    return not synced or synced < datetime.now(timezone.utc) - timedelta(days=REFRESH_DAYS)


def season_finished(db, season_ids):
    return {r[0]: bool(r[1]) for r in db.query(
        "SELECT season_id, coalesce(finished, false) FROM football.seasons WHERE season_id = ANY(%s)",
        (list(season_ids),))}


def transfer_row(o):
    return {"transfer_id": o.get("id"), "player_id": o.get("player_id"), "type_id": o.get("type_id"),
            "from_team_id": o.get("from_team_id"), "to_team_id": o.get("to_team_id"),
            "position_id": o.get("position_id"), "detailed_position_id": o.get("detailed_position_id"),
            "date": o.get("date"), "career_ended": o.get("career_ended"), "completed": o.get("completed"),
            "completed_at": o.get("completed_at"), "amount": o.get("amount")}


def coach_team_row(o, coach_id=None, team_id=None):
    return {"coach_team_id": o.get("id"), "coach_id": o.get("coach_id") or coach_id,
            "team_id": o.get("team_id") or team_id, "position_id": o.get("position_id"),
            "active": o.get("active"), "temporary": o.get("temporary"),
            "start_date": o.get("start"), "end_date": o.get("end")}


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


def load_league_and_seasons(api, db, league_id):
    print("\n[2/8] League and seasons")
    try:
        body = api.get(f"/football/leagues/{league_id}", {"include": "seasons"}, store=("league", league_id))
    except ApiError as e:
        warn(f"league {league_id}: {e} - is it on your plan?")
        return []
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
        warn(f"league {league_id}: no seasons found - check the id and your plan")
        return []
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
    print("\n[4/8] Teams, squads + player season stats, injuries, transfers")
    season_filter = ",".join(map(str, season_ids))
    finished = season_finished(db, season_ids)
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

    done_team, done_transfers = db.get_sync("team"), db.get_sync("team_transfers")
    todo = [tid for tid, sids in sorted(team_seasons.items())
            if needs_refresh(done_team.get(str(tid)), all(finished.get(s) for s in sids))]
    print(f"  {len(team_seasons)} teams; {len(team_seasons) - len(todo)} up to date, {len(todo)} to fetch")
    for i, tid in enumerate(todo, 1):
        try:
            body = api.get_inc("team", f"/football/teams/{tid}",
                               ["venue", "statistics.details", "sidelined", "rivals",
                                "trophies.trophy|trophies", "coaches"],
                               {"filters": f"teamStatisticSeasons:{season_filter}"}, store=("team", tid))
        except ApiError as e:
            warn(f"team {tid}: {e}")
            continue
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
        db.upsert("football.coach_teams", [coach_team_row(o, team_id=tid) for o in t.get("coaches") or []])
        db.set_sync("team", tid)
        db.commit()
        print(f"    team {i}/{len(todo)}")

    if FETCH_TEAM_TRANSFERS:
        national = {r[0] for r in db.query(
            "SELECT team_id FROM football.teams WHERE team_id = ANY(%s) AND lower(type) = 'national'",
            (list(team_seasons),))}
        todo = [tid for tid in sorted(team_seasons)
                if tid not in national and needs_refresh(done_transfers.get(str(tid)), True)]
        print(f"  transfer histories: {len(team_seasons) - len(todo)} done, {len(todo)} to fetch")
        for tid in todo:
            try:
                db.upsert("football.transfers", [transfer_row(o) for o in
                                                 api.pages(f"/football/transfers/teams/{tid}")])
                db.set_sync("team_transfers", tid)
            except ApiError as e:
                warn(f"transfers team {tid}: {e}")
            db.commit()

    load_squads(api, db, team_seasons, finished)


def load_squads(api, db, team_seasons, finished):
    """One call per team-season returns the whole squad with each player's
    profile and season statistics (instead of one call per player)."""
    done = db.get_sync("squad")
    jobs = sorted((tid, sid) for tid, sids in team_seasons.items() for sid in sids)
    todo = [(tid, sid) for tid, sid in jobs if needs_refresh(done.get(f"{tid}:{sid}"), finished.get(sid))]
    print(f"  squads: {len(jobs) - len(todo)} up to date, {len(todo)} to fetch")
    for i, (tid, sid) in enumerate(todo, 1):
        items, page = [], 1
        try:
            while True:
                body = api.get_inc("squad", f"/football/squads/seasons/{sid}/teams/{tid}",
                                   ["player.statistics.details|player"],
                                   {"filters": f"playerStatisticSeasons:{sid}", "page": page},
                                   store=("squad", tid))
                data = body.get("data") or []
                items.extend(data if isinstance(data, list) else [data])
                if not (body.get("pagination") or {}).get("has_more"):
                    break
                page += 1
        except ApiError as e:
            warn(f"squad team {tid} season {sid}: {e}")
            continue
        players = [o["player"] for o in items if isinstance(o.get("player"), dict)]
        db.upsert("football.players", [player_row(p) for p in players])
        parents, details = stat_rows([s for p in players for s in p.get("statistics") or []], None,
                                     "player_statistic_id", "player_id", None, "player_statistic_detail_id")
        db.upsert("football.player_season_stats", parents)
        db.upsert("football.player_season_stat_details", details)
        db.upsert("football.team_squads", [squad_row(o, tid, sid) for o in items])
        with_stats = "player.statistics.details" in (api.include_cache.get("squad") or [])
        db.set_sync("squad", f"{tid}:{sid}", "ok", {"stats": with_stats})
        db.commit()
        if i % 10 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)}")


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
        stations = [o.get("tvstation") or o.get("tvStation") for o in tv]
        db.upsert("football.tv_stations", [{
            "tv_station_id": st.get("id"), "name": st.get("name"), "url": st.get("url"),
            "type": st.get("type"), "related_id": st.get("related_id"), "image_path": st.get("image_path")}
            for st in stations if isinstance(st, dict)])
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

    elif pass_name == "premium_odds":
        premium = f.get("premiumOdds") or f.get("premiumodds") or []
        db.upsert("odds.premium_odds", [odd_row(o, fid) for o in premium])
        db.upsert("odds.premium_odds_history", [{
            "history_id": h.get("id"), "odd_id": h.get("odd_id") or o.get("id"), "value": h.get("value"),
            "probability": h.get("probability"), "dp3": h.get("dp3"), "fractional": h.get("fractional"),
            "american": h.get("american"), "bookmaker_update": h.get("bookmaker_update")}
            for o in premium for h in o.get("history") or []])

    elif pass_name == "inplay_odds":
        inplay = f.get("inplayOdds") or f.get("inplayodds") or []
        db.upsert("odds.inplay_odds", [{**odd_row(o, fid), "external_id": o.get("external_id"),
                                        "suspended": o.get("suspended")} for o in inplay])

    elif pass_name == "ball":
        db.upsert("football.fixture_ball_coordinates", [{
            "ball_coordinate_id": o.get("id"), "fixture_id": fid, "period_id": o.get("period_id"),
            "timer": o.get("timer"), "x": o.get("x"), "y": o.get("y")}
            for o in f.get("ballCoordinates") or f.get("ballcoordinates") or []])


# Tables each pass fills, used to spot matches marked done but missing data.
GAP_TABLES = {
    "main": ["football.fixture_events", "football.fixture_lineups", "football.fixture_lineup_details",
             "football.fixture_statistics", "football.fixture_officials", "football.fixture_xg"],
    "live": ["football.fixture_pressure", "football.fixture_trends", "football.fixture_commentaries"],
    "odds": ["odds.prematch_odds"],
    "premium_odds": ["odds.premium_odds"],
    "inplay_odds": ["odds.inplay_odds"],
    "ball": ["football.fixture_ball_coordinates"],
}


def repair_gaps(db, fixture_ids, played_ids):
    """A finished match marked done in a pass but with no rows in a table that
    most of this league's finished matches DO have (e.g. cut off by a network
    drop) is queued to download again. Each match is retried at most once per
    table, so matches that genuinely lack that data aren't re-fetched forever."""
    if not fixture_ids or not played_ids:
        return
    repaired = db.get_sync("fixture_repair")
    total = 0
    for pass_name, tables in GAP_TABLES.items():
        for table in tables:
            rows = db.query(f"""
                WITH played AS (
                    SELECT f.fixture_id FROM football.fixtures f
                    JOIN raw.sync_state s ON s.entity = %(entity)s AND s.scope_key = f.fixture_id::text
                                          AND s.status = 'ok'
                    WHERE f.fixture_id = ANY(%(ids)s) AND f.state_id = ANY(%(played)s)),
                have AS (SELECT DISTINCT t.fixture_id FROM {table} t JOIN played USING (fixture_id))
                SELECT p.fixture_id, (SELECT count(*) FROM have)::float / NULLIF((SELECT count(*) FROM played), 0)
                FROM played p WHERE NOT EXISTS (SELECT 1 FROM have h WHERE h.fixture_id = p.fixture_id)""",
                {"entity": f"fixture_{pass_name}", "ids": list(fixture_ids), "played": list(played_ids)})
            gaps = [fid for fid, coverage in rows
                    if coverage is not None and float(coverage) >= 0.5 and f"{table}:{fid}" not in repaired]
            for fid in gaps:
                db.execute("DELETE FROM raw.sync_state WHERE entity = %s AND scope_key = %s",
                           (f"fixture_{pass_name}", str(fid)))
                db.set_sync("fixture_repair", f"{table}:{fid}")
            total += len(gaps)
    db.commit()
    if total:
        print(f"  repair: {total} match/data gaps found (e.g. from a dropped connection) - re-fetching them")


def load_fixtures(api, db, fixtures, refetch=()):
    print(f"\n[5/8] Fixtures ({len(fixtures)} matches)")
    if not fixtures:
        return
    league_id = fixtures[0].get("league_id")
    states = {r[0]: r[1] for r in db.query("SELECT state_id, developer_name FROM core.states")}
    final_ids = {sid for sid, name in states.items() if name in FINAL_STATES}
    played_ids = {sid for sid, name in states.items() if name in ("FT", "AET", "FT_PEN")}
    skipped = db.get_sync("pass_skip")
    repair_gaps(db, [f["id"] for f in fixtures], played_ids)

    for pass_name, cfg in FIXTURE_PASSES.items():
        if not cfg["enabled"]:
            continue
        skip_key = f"{league_id}:{pass_name}"
        if skip_key in skipped and pass_name not in refetch:
            print(f"  pass '{pass_name}': skipped (no data for this league on an earlier run; "
                  f"use --refetch {pass_name} to try again)")
            continue
        state_of = {r[0]: r[1] for r in db.query(
            "SELECT fixture_id, state_id FROM football.fixtures WHERE fixture_id = ANY(%s)",
            ([f["id"] for f in fixtures],))}
        done = {} if pass_name in refetch else db.get_sync(f"fixture_{pass_name}")
        todo = [f["id"] for f in fixtures
                if not (str(f["id"]) in done and done[str(f["id"])][1] == "ok"
                        and (done[str(f["id"])][2] or {}).get("final"))]
        todo.sort(key=lambda fid: (state_of.get(fid) not in played_ids, fid))   # played matches first
        print(f"  pass '{pass_name}': {len(fixtures) - len(todo)} already complete, {len(todo)} to fetch")

        size, pos, checked, found = cfg["batch"], 0, 0, 0
        while pos < len(todo):
            batch = todo[pos:pos + size]
            path = "/football/fixtures/multi/" + ",".join(map(str, batch))
            try:
                body = api.get_inc(f"fixtures_{pass_name}", path, cfg["includes"],
                                   store=("fixtures", None) if cfg["raw"] else None)
            except ApiError as e:
                if len(batch) > 1:
                    size = max(1, len(batch) // 2)
                    warn(f"pass '{pass_name}': batch refused ({e}); trying {size} matches per request")
                    continue
                warn(f"fixture {batch[0]} pass '{pass_name}': {e}")
                pos += 1
                continue
            for f in body.get("data") or []:
                save_fixture(db, f, pass_name)
                state = f.get("state_id") or state_of.get(f["id"])
                db.set_sync(f"fixture_{pass_name}", f["id"], "ok", {"final": state in final_ids})
                if pass_name in PASS_KEYS and state in played_ids:
                    checked += 1
                    found += any(f.get(k) for k in PASS_KEYS[pass_name])
            db.commit()
            pos += len(batch)
            print(f"    {pos}/{len(todo)}")
            if pass_name in PASS_KEYS and checked >= EMPTY_CHECK and found == 0:
                warn(f"pass '{pass_name}': nothing returned for {checked} finished matches - not covered "
                     f"for this league or plan; skipping it from now on")
                db.set_sync("pass_skip", skip_key, "ok", {"checked": checked})
                db.commit()
                break


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
         # players seen in these seasons, minus those already covered by a squad call
         "(SELECT player_id FROM football.fixture_lineups WHERE season_id = ANY(%(s)s) "
         " UNION SELECT player_id FROM football.fixture_events WHERE season_id = ANY(%(s)s) "
         " UNION SELECT related_player_id FROM football.fixture_events WHERE season_id = ANY(%(s)s) "
         " UNION SELECT player_id FROM football.sidelined WHERE season_id = ANY(%(s)s) "
         " UNION SELECT player_id FROM football.topscorers WHERE season_id = ANY(%(s)s)) "
         "EXCEPT SELECT ts.player_id FROM football.team_squads ts "
         " JOIN raw.sync_state ss ON ss.entity = 'squad' "
         "  AND ss.scope_key = ts.team_id::text || ':' || ts.season_id::text "
         " WHERE ts.season_id = ANY(%(s)s) AND (ss.detail->>'stats')::boolean",
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
    finished = season_finished(db, season_ids)
    current = [sid for sid in season_ids if not finished.get(sid)]
    for kind, table, id_col, id_sql, path, filt in groups:
        ids = sorted({int(r[0]) for r in db.query(id_sql, {"s": season_ids}) if r[0]})
        active_now = {int(r[0]) for r in db.query(id_sql, {"s": current}) if r[0]} if current else set()
        done = db.get_sync(kind)
        todo = []
        for pid in ids:
            d = done.get(f"{pid}@{seasons}") or done.get(str(pid))
            if d and (d[2] or {}).get("seasons") != seasons:
                d = None
            if needs_refresh(d, pid not in active_now):
                todo.append(pid)
        label = {"player": "players", "referee": "referees", "coach": "coaches"}[kind]
        print(f"  {label}: {len(ids) - len(todo)} up to date, {len(todo)} to fetch")

        for i, pid in enumerate(todo, 1):
            try:
                body = api.get_inc(kind, path.format(pid), PEOPLE_INCLUDES[kind],
                                   {"filters": f"{filt}:{seasons}"}, store=(kind, pid))
            except ApiError as e:
                warn(f"{kind} {pid}: {e}")
                db.set_sync(kind, f"{pid}@{seasons}", "error", {"error": str(e)})
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
            if kind == "player":
                db.upsert("football.transfers", [transfer_row(t) for t in o.get("transfers") or []])
            if kind == "coach":
                db.upsert("football.coach_teams", [coach_team_row(t, coach_id=pid) for t in o.get("teams") or []])
            db.set_sync(kind, f"{pid}@{seasons}", "ok", {"seasons": seasons})
            if i % 25 == 0 or i == len(todo):
                db.commit()
                print(f"    {i}/{len(todo)}")
        db.commit()


def fill_missing(api, db):
    """Download IDs that other tables point to but that weren't loaded."""
    jobs = {
        "coaches": ("football.coaches", "/football/coaches/{}", None),
        "venues":  ("football.venues", "/football/venues/{}", None),
        "leagues": ("football.leagues", "/football/leagues/{}", "seasons"),
        "teams":   ("football.teams", "/football/teams/{}", None),
        "players": ("football.players", "/football/players/{}", None),
    }
    wanted = {jobs[k][0]: k for k, on in FILL_MISSING.items() if on and k in jobs}
    if not wanted:
        return
    print("\n[+] Filling in referenced IDs that weren't downloaded")
    missing = {}
    for tbl, col, ref_tbl, ref_col in db.query(
            "SELECT tbl, col, ref_tbl, ref_col FROM raw.relationships "
            "WHERE NOT enforced AND ref_tbl = ANY(%s)", (list(wanted),)):
        for (mid,) in db.query(
                f"SELECT DISTINCT s.{col} FROM {tbl} s WHERE s.{col} IS NOT NULL "
                f"AND NOT EXISTS (SELECT 1 FROM {ref_tbl} t WHERE t.{ref_col} = s.{col})"):
            missing.setdefault(ref_tbl, set()).add(int(mid))
    failed_before = db.get_sync("fill_failed")
    for tbl, name in wanted.items():
        ids = sorted(i for i in missing.get(tbl, ()) if f"{tbl}:{i}" not in failed_before)
        if not ids:
            continue
        _, path, include = jobs[name]
        loaded = failed = 0
        for mid in ids:
            try:
                o = api.get(path.format(mid), {"include": include} if include else None).get("data") or {}
            except ApiError as e:
                if 400 <= e.status < 500:               # Sportmonks says no: don't ask again
                    failed += 1
                    db.set_sync("fill_failed", f"{tbl}:{mid}", "error")
                else:
                    warn(f"{name} {mid}: {e} (will retry next run)")
                continue
            if name == "coaches":
                db.upsert(tbl, [{**person_row(o, "coach_id"), "player_id": o.get("player_id")}])
            elif name == "venues":
                db.upsert(tbl, [venue_row(o)])
            elif name == "teams":
                db.upsert(tbl, [team_row(o)])
            elif name == "players":
                db.upsert(tbl, [player_row(o)])
            elif name == "leagues":
                db.upsert(tbl, [{"league_id": o.get("id"), "country_id": o.get("country_id"), "name": o.get("name"),
                                 "short_code": o.get("short_code"), "type": o.get("type"),
                                 "sub_type": o.get("sub_type"), "image_path": o.get("image_path")}])
                db.upsert("football.seasons", [{
                    "season_id": x.get("id"), "league_id": o.get("id"), "name": x.get("name"),
                    "finished": x.get("finished"), "is_current": x.get("is_current"),
                    "starting_at": x.get("starting_at"), "ending_at": x.get("ending_at")}
                    for x in o.get("seasons") or []])
            loaded += 1
            db.commit()
        db.commit()
        print(f"  {name}: {loaded} loaded" + (f", {failed} not available (won't retry)" if failed else ""))


def estimate_league(api, db, league_id, refetch=()):
    """Rough count of the API calls a full run would make for one league.
    Costs about 4 calls (league + season lists), which the real run reuses."""
    season_ids = load_league_and_seasons(api, db, league_id)
    if not season_ids:
        return None
    fixtures = []
    for sid in season_ids:
        try:
            fixtures.extend(load_season_structure(api, db, sid))
        except ApiError as e:
            warn(f"season {sid}: {e}")
    finished = season_finished(db, season_ids)
    est = {}

    skipped = db.get_sync("pass_skip")
    for pass_name, cfg in FIXTURE_PASSES.items():
        if not cfg["enabled"] or (f"{league_id}:{pass_name}" in skipped and pass_name not in refetch):
            continue
        done = {} if pass_name in refetch else db.get_sync(f"fixture_{pass_name}")
        todo = sum(1 for f in fixtures if not (str(f["id"]) in done and done[str(f["id"])][1] == "ok"
                                               and (done[str(f["id"])][2] or {}).get("final")))
        est[f"matches: {pass_name}"] = math.ceil(todo / cfg["batch"])

    # teams per season from match count (double round robin: n*(n-1) matches)
    per_season = {}
    for f in fixtures:
        per_season[f.get("season_id")] = per_season.get(f.get("season_id"), 0) + 1
    team_counts = {sid: max(2, round((1 + math.sqrt(1 + 4 * m)) / 2)) for sid, m in per_season.items()}
    unique_teams = round(max(team_counts.values(), default=0) * (1 + 0.15 * (len(season_ids) - 1)))
    done_squads = db.get_sync("squad")
    squads_done = sum(1 for k, v in done_squads.items()
                      if int(k.split(":")[1]) in finished and needs_refresh(v, finished[int(k.split(":")[1])]) is False)
    est["teams + squads"] = len(season_ids) + unique_teams + max(0, sum(team_counts.values()) - squads_done)
    if FETCH_TEAM_TRANSFERS:
        done_tr = db.get_sync("team_transfers")
        est["transfer histories (once)"] = 0 if len(done_tr) >= unique_teams else unique_teams * 6
    est["referees + coaches + extra players"] = round(len(fixtures) * 0.15) + unique_teams * 2
    est["standings + top scorers"] = len(season_ids) * 2
    return est


# =================================================================== MAIN ===
def print_estimates(estimates):
    total = 0
    for league_id, est in estimates:
        if est is None:
            print(f"  league {league_id}: not available")
            continue
        sub = sum(est.values())
        total += sub
        print(f"\n  league {league_id}: ~{sub:,} calls")
        for k, v in est.items():
            print(f"      {k:<40}{v:>8,}")
    print(f"\n  TOTAL (rough): ~{total:,} calls; budget left: {BUDGET.remaining:,} "
          f"(of {CALL_BUDGET:,} per {BUDGET_PERIOD})")
    if BUDGET_PERIOD == "hour":
        per_hour = CALL_BUDGET - BUDGET_RESERVE
        print(f"  That's about {max(1, math.ceil(total / per_hour))} hour(s) of allowance; if a run "
              f"uses up an hour's calls it pauses until the next hour and carries on.")
    elif total > BUDGET.remaining:
        print("  This is more than the budget left. The run will stop cleanly when it runs out and\n"
              "  carry on next time - or load fewer leagues / switch off FETCH_ options.")


def main():
    parser = argparse.ArgumentParser(description="Load leagues from Sportmonks into the football database")
    parser.add_argument("leagues", nargs="*", type=int,
                        help="Sportmonks league ids (default: the LEAGUES list in this file)")
    parser.add_argument("--refetch", nargs="+", default=[], choices=list(FIXTURE_PASSES),
                        help="re-download these fixture passes for matches already loaded")
    parser.add_argument("--estimate", action="store_true", help="only estimate the API calls needed")
    parser.add_argument("--used-so-far", type=int, metavar="N",
                        help="set how many API calls you've already used this period (e.g. from earlier runs)")
    parser.add_argument("--skip-teams", action="store_true", help="skip step 4")
    parser.add_argument("--skip-people", action="store_true", help="skip step 7")
    args = parser.parse_args()
    leagues = list(dict.fromkeys(args.leagues or LEAGUES))

    started = time.time()
    db = Db(DATABASE)
    db.columns("football.fixtures")      # confirms the schema is loaded
    BUDGET.load(db)
    if args.used_so_far is not None:
        BUDGET.used = args.used_so_far
        BUDGET.save(db, force=True)
    print(f"API budget: {BUDGET.remaining:,} calls left this {BUDGET_PERIOD} "
          f"(of {CALL_BUDGET:,}, {BUDGET_RESERVE} in reserve)")
    api = Api(API_TOKEN, db)

    try:
        SIZE_GUARD.check(db, force=True)
        print(f"Database size: {SIZE_GUARD.last_gb:.1f} GB (limit {MAX_DB_SIZE_GB} GB)")
        load_core(api, db)
        if args.estimate:
            print("\nESTIMATE")
            print_estimates([(lid, estimate_league(api, db, lid, args.refetch)) for lid in leagues])
            return

        for n, league_id in enumerate(leagues, 1):
            print(f"\n{'=' * 70}\nLEAGUE {n}/{len(leagues)}  (id {league_id})   "
                  f"budget left: {BUDGET.remaining:,}\n{'=' * 70}")
            season_ids = load_league_and_seasons(api, db, league_id)
            if not season_ids:
                continue

            print("\n[3/8] Season structure and fixture lists")
            fixtures = []
            for sid in season_ids:
                try:
                    fx = load_season_structure(api, db, sid)
                except ApiError as e:
                    warn(f"season {sid}: {e}")
                    continue
                print(f"  season {sid}: {len(fx)} fixtures")
                fixtures.extend(fx)

            if not args.skip_teams:
                load_teams(api, db, season_ids)

            load_fixtures(api, db, fixtures, args.refetch)

            print("\n[6/8] Standings and top scorers")
            for sid in season_ids:
                load_standings(api, db, sid)
            print("  done")

            if not args.skip_people:
                load_people(api, db, season_ids)

        fill_missing(api, db)
    except NetworkDown as e:
        db.commit()
        print(f"\nSTOPPED: {e}. Everything so far is saved; run again once you're back online.")
    except StorageLimitReached as e:
        db.commit()
        print(f"\nSTOPPED SAFELY: {e}. Everything so far is saved. Free up space or raise "
              f"MAX_DB_SIZE_GB / lower MIN_FREE_DISK_GB in load_league.py, then run again.")
    except BudgetExhausted as e:
        db.commit()
        print(f"\nSTOPPED: {e}. Everything so far is saved; run again when the allowance resets.")
    finally:
        db.commit()
        BUDGET.save(db)

    if SIZE_GUARD.tripped:
        print("\n[8/8] Skipping the analysis-view refresh (it needs extra space). Once there's room run:\n"
              "      SELECT analysis.refresh_all();")
    else:
        print("\n[8/8] Refreshing analysis views")
        db.execute("SELECT analysis.refresh_all()")
        db.commit()
    print_summary(db, leagues)
    print(f"\n  {BUDGET.session:,} API calls this run in {(time.time() - started) / 60:.1f} minutes; "
          f"budget left: {BUDGET.remaining:,}")
    if WARNINGS:
        print(f"  {len(WARNINGS)} warnings (shown above with '!')")


def print_summary(db, leagues):
    print("\nSUMMARY")
    for name, matches in db.query(
            "SELECT l.name, count(*) FROM football.fixtures f JOIN football.leagues l USING (league_id) "
            "WHERE f.league_id = ANY(%s) GROUP BY 1 ORDER BY 1", (list(leagues),)):
        print(f"  {name:<40}{matches:>12,} matches")
    for table in ("football.fixture_events", "football.fixture_lineup_details",
                  "football.fixture_statistics", "football.fixture_officials", "football.players",
                  "football.player_season_stats", "football.referees", "odds.prematch_odds",
                  "odds.premium_odds", "odds.inplay_odds", "football.fixture_ball_coordinates",
                  "analysis.referee_match_discipline"):
        print(f"  {table:<40}{db.query(f'SELECT count(*) FROM {table}')[0][0]:>12,}")
    missing = db.query("SELECT ref_tbl, count(DISTINCT missing_id) FROM raw.missing_references() GROUP BY 1")
    if missing:
        print("  IDs referenced but not downloaded (mostly clubs/players from transfer histories; "
              "see FILL_MISSING): " + ", ".join(f"{t} {n}" for t, n in missing))


if __name__ == "__main__":
    main()
