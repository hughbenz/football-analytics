#!/usr/bin/env python3
"""
Load several leagues AT THE SAME TIME, one worker thread per league.

Uses load_league.py for all the actual loading - keep both files in the same
folder. API key, database settings, number of seasons and the FETCH_ options
are all read from load_league.py, so set them there.

RUN
  python3 load_leagues_parallel.py                    # leagues in LEAGUES below
  python3 load_leagues_parallel.py 82 564 8 384       # these leagues
  python3 load_leagues_parallel.py 82 564 --workers 2
  python3 load_leagues_parallel.py 82 --refetch main  # same options as load_league.py
  python3 load_leagues_parallel.py 82 564 8 --estimate   # calls needed, before running

  The API call budget (CALL_BUDGET in load_league.py) is shared by all
  workers; when it runs out every worker stops cleanly and the next run
  carries on.

  Ctrl+C stops every worker after its current request; progress is saved
  and the next run continues where it left off.

  "WC_QUALIFIERS" in the LEAGUES list is replaced at start-up by every
  men's World Cup qualifying competition Sportmonks has (looked up by name,
  ~2-3 calls), and the names and ids found are printed.

HOW THE WORKERS SHARE THINGS SAFELY
  * Sportmonks' hourly limits are per account, not per worker. All workers
    watch the same counters: when one sees the allowance for a type of data
    (fixtures, players ...) running low, every worker pauses that type until
    the hour resets, instead of all of them hitting the limit.
  * Each worker has its own database connection and saves after every write,
    so two workers writing the same row (a player who changed league, say)
    just wait for each other; a rare deadlock is retried automatically.
  * Output lines are tagged with the league they belong to.

SPEED
  More workers keep the hourly allowance fully used and overlap different
  types of data, but can't exceed the allowance itself. Expect roughly
  2-3x faster than one league at a time; beyond 3-4 workers it rarely helps.
"""

import argparse
import builtins
import random
import re
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

from psycopg import errors as pg_errors


# =============================================================== SETTINGS ===
# The 50 most notable competitions on your plan, MOST IMPORTANT FIRST.
# Workers take them in this order, so if the API budget runs out, the ones at
# the bottom are what wait for next month. Add/remove a # to change the set.
LEAGUES = [
    # --- Top 5 European leagues
    8,       # England       Premier League
    564,     # Spain         La Liga
    82,      # Germany       Bundesliga
    384,     # Italy         Serie A
    301,     # France        Ligue 1
    # --- UEFA club competitions + World Cup
    2,       # Europe        UEFA Champions League
    5,       # Europe        UEFA Europa League
    2286,    # Europe        UEFA Europa Conference League
    732,     # World         FIFA World Cup
    1328,    # Europe        UEFA Super Cup
    # --- Next tier Europe
    72,      # Netherlands   Eredivisie
    462,     # Portugal      Liga Portugal
    9,       # England       Championship
    208,     # Belgium       Pro League
    600,     # Turkey        Super Lig
    501,     # Scotland      Premiership
    # --- Americas + major club competitions
    648,     # Brazil        Serie A
    636,     # Argentina     Liga Profesional
    779,     # USA           Major League Soccer
    743,     # Mexico        Liga MX
    1122,    # South America Copa Libertadores
    1116,    # South America Copa Sudamericana
    3412,    # World         FIFA Club World Cup
    "WC_QUALIFIERS",   # World Cup qualifying, every confederation (found by name at start-up)
    # --- Second divisions of the top 5
    85,      # Germany       2. Bundesliga
    387,     # Italy         Serie B
    567,     # Spain         La Liga 2
    304,     # France        Ligue 2
    # --- Strong European leagues
    181,     # Austria       Admiral Bundesliga
    591,     # Switzerland   Super League
    325,     # Greece        Super League
    271,     # Denmark       Superliga
    262,     # Czech Rep.    Chance Liga
    244,     # Croatia       1. HNL
    609,     # Ukraine       Premier League
    486,     # Russia        Premier League
    453,     # Poland        Ekstraklasa
    444,     # Norway        Eliteserien
    573,     # Sweden        Allsvenskan
    531,     # Serbia        Super Liga
    474,     # Romania       Superliga
    # --- Asia / Middle East / Oceania
    944,     # Saudi Arabia  Pro League
    968,     # Japan         J1 League
    1034,    # South Korea   K League 1
    1085,    # Asia          AFC Champions League Elite
    1356,    # Australia     A-League Men
    989,     # China         Super League
    1111,    # N&C America   CONCACAF Champions Cup
    # --- More South America
    672,     # Colombia      Liga BetPlay
    663,     # Chile         Primera Division
    # --- Domestic cups (top 5 countries)
    24,      # England       FA Cup
    570,     # Spain         Copa Del Rey
    109,     # Germany       DFB Pokal
    390,     # Italy         Coppa Italia
    307,     # France        Coupe de France
    12,      # England       League One
]
MAX_WORKERS = 6          # leagues loading at the same time
PAUSE_BELOW = 20         # pause a data type when fewer calls than this remain


# ========================================================== TAGGED OUTPUT ===
_print_lock = threading.Lock()
_tag = threading.local()


def tprint(*args, **kwargs):
    prefix = getattr(_tag, "label", "")
    text = " ".join(str(a) for a in args)
    with _print_lock:
        for line in text.split("\n"):
            builtins.print(f"{prefix}{line}" if line.strip() else line, **kwargs)


L.print = tprint          # every message from load_league.py gets the league tag


# ================================================================ CTRL+C ===
STOP = threading.Event()


class StopRequested(L.BudgetExhausted):
    """Raised inside workers after Ctrl+C so they stop like a budget stop."""


def interruptible_sleep(seconds):
    end = time.time() + seconds
    while True:
        if STOP.is_set():
            raise StopRequested("stopped with Ctrl+C")
        left = end - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 1.0))


class _Clock:                       # load_league.py's waits become interruptible too
    time = staticmethod(time.time)
    sleep = staticmethod(interruptible_sleep)


L.time = _Clock


# ===================================================== SHARED RATE LIMITS ===
class RateGate:
    """Shared across all workers. Learns which Sportmonks rate-limit bucket
    each kind of request belongs to, and pauses that bucket for everyone
    when it is nearly used up."""

    def __init__(self):
        self.lock = threading.Lock()
        self.bucket_of = {}       # request kind -> entity name ('Fixture', ...)
        self.paused_until = {}    # entity name -> unix time

    @staticmethod
    def kind(path):
        parts = [p for p in path.split("/") if p and not re.fullmatch(r"[\d,]+", p)]
        return "/".join(parts[:2])

    def wait(self, path):
        with self.lock:
            entity = self.bucket_of.get(self.kind(path))
            until = self.paused_until.get(entity, 0)
        delay = until - time.time()
        if delay > 0:
            tprint(f"  waiting {int(delay)}s for the shared {entity} limit to reset")
            interruptible_sleep(delay)

    def update(self, path, rate_limit):
        if not isinstance(rate_limit, dict):
            return
        entity = rate_limit.get("requested_entity")
        remaining = rate_limit.get("remaining")
        with self.lock:
            if entity:
                self.bucket_of[self.kind(path)] = entity
            if isinstance(remaining, int) and remaining < PAUSE_BELOW:
                resume = time.time() + int(rate_limit.get("resets_in_seconds", 60)) + 5
                if resume > self.paused_until.get(entity, 0):
                    self.paused_until[entity] = resume


GATE = RateGate()
_original_get = L.Api.get


def gated_get(self, path, params=None, store=None, retries=6):
    if STOP.is_set():
        raise StopRequested("stopped with Ctrl+C")
    GATE.wait(path)
    body = _original_get(self, path, params, store, retries)
    GATE.update(path, body.get("rate_limit"))
    return body


L.Api.get = gated_get

# Include discoveries are shared, so only one worker probes each request type.
SHARED_INCLUDE_CACHE, SHARED_SPLIT = {}, {}


# ==================================================== THREAD-SAFE DATABASE ===
RETRYABLE = (pg_errors.DeadlockDetected, pg_errors.SerializationFailure)


class ThreadDb(L.Db):
    """One connection per worker. Every write is its own short transaction,
    retried if PostgreSQL reports a deadlock with another worker."""

    def _retry(self, fn):
        for attempt in range(8):
            try:
                result = fn()
                self.conn.commit()
                return result
            except RETRYABLE:
                self.conn.rollback()
                time.sleep(0.2 * (attempt + 1) + random.random() * 0.5)
        result = fn()
        self.conn.commit()
        return result

    def upsert(self, table, rows):
        return self._retry(lambda: L.Db.upsert(self, table, rows))

    def execute(self, sql, params=None):
        return self._retry(lambda: L.Db.execute(self, sql, params))

    def query(self, sql, params=None):
        rows = L.Db.query(self, sql, params)
        self.conn.commit()          # don't hold a transaction open between calls
        return rows


# ===================================================== WORLD CUP QUALIFIERS ===
WCQ_SEARCHES = ["WC Qualification", "World Cup Qualification", "World Cup Qualifying"]
WCQ_EXCLUDE = ("women", "u17", "u19", "u20", "u21", "u23", "youth", "olympic", "futsal", "beach")


def find_wc_qualifiers(api):
    found = {}
    for term in WCQ_SEARCHES:
        try:
            for lg in api.pages("/football/leagues/search/" + urllib.parse.quote(term)):
                name = (lg.get("name") or "").lower()
                is_wc = "wc " in f"{name} " or "world cup" in name
                if is_wc and "qualif" in name and not any(x in name for x in WCQ_EXCLUDE):
                    found[lg["id"]] = lg.get("name")
        except L.ApiError as e:
            L.warn(f"league search '{term}': {e}")
    return found


def resolve_leagues(api, leagues):
    out = []
    for item in leagues:
        if item == "WC_QUALIFIERS":
            found = find_wc_qualifiers(api)
            if found:
                tprint("World Cup qualifying competitions found:")
                for lid, name in sorted(found.items(), key=lambda kv: kv[1]):
                    tprint(f"  {lid:>6}  {name}")
            else:
                L.warn("no World Cup qualifying competitions found by name search")
            out.extend(sorted(found))
        else:
            out.append(int(item))
    return list(dict.fromkeys(out))


# ============================================================= ONE LEAGUE ===
def run_league(league_id, args):
    _tag.label = f"[{league_id}] "
    started = time.time()
    db = ThreadDb(L.DATABASE)
    api = L.Api(L.API_TOKEN, db)
    api.include_cache = SHARED_INCLUDE_CACHE
    api.split = SHARED_SPLIT
    try:
        season_ids = L.load_league_and_seasons(api, db, league_id)
        if not season_ids:
            return league_id, "skipped", api.calls, time.time() - started
        name = db.query("SELECT name FROM football.leagues WHERE league_id = %s", (league_id,))
        _tag.label = f"[{name[0][0] if name else league_id}] "

        tprint("\n[3/8] Season structure and fixture lists")
        fixtures = []
        for sid in season_ids:
            try:
                fx = L.load_season_structure(api, db, sid)
            except L.ApiError as e:
                L.warn(f"season {sid}: {e}")
                continue
            tprint(f"  season {sid}: {len(fx)} fixtures")
            fixtures.extend(fx)

        if not args.skip_teams:
            L.load_teams(api, db, season_ids)

        L.load_fixtures(api, db, fixtures, args.refetch)

        tprint("\n[6/8] Standings and top scorers")
        for sid in season_ids:
            L.load_standings(api, db, sid)

        if not args.skip_people:
            L.load_people(api, db, season_ids)

        tprint("\nFINISHED this league")
        return league_id, "ok", api.calls, time.time() - started
    except StopRequested:
        tprint("\nSTOPPED (Ctrl+C). Progress is saved; run again to continue.")
        return league_id, "stopped", api.calls, time.time() - started
    except L.StorageLimitReached as e:
        tprint(f"\nSTOPPED SAFELY: {e}. Progress is saved.")
        return league_id, "storage limit", api.calls, time.time() - started
    except L.NetworkDown as e:
        tprint(f"\nSTOPPED: {e}. Progress is saved; run again once you're back online.")
        return league_id, "no network", api.calls, time.time() - started
    except L.BudgetExhausted:
        tprint("\nSTOPPED: API call budget reached. Progress is saved; run again later to continue.")
        return league_id, "budget reached", api.calls, time.time() - started
    except Exception as e:                      # one league failing doesn't stop the others
        tprint(f"\nSTOPPED with an error: {type(e).__name__}: {e}")
        tprint("Run the script again for this league; it will resume.")
        return league_id, f"error: {type(e).__name__}", api.calls, time.time() - started
    finally:
        try:
            db.conn.close()
        except Exception:
            pass


# =================================================================== MAIN ===
def main():
    parser = argparse.ArgumentParser(description="Load several leagues from Sportmonks in parallel")
    parser.add_argument("leagues", nargs="*", type=int, help="Sportmonks league ids (default: LEAGUES)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="leagues at the same time")
    parser.add_argument("--refetch", nargs="+", default=[], choices=list(L.FIXTURE_PASSES),
                        help="re-download these fixture passes for matches already loaded")
    parser.add_argument("--estimate", action="store_true", help="only estimate the API calls needed")
    parser.add_argument("--used-so-far", type=int, metavar="N",
                        help="set how many API calls you've already used this period (e.g. from earlier runs)")
    parser.add_argument("--skip-teams", action="store_true", help="skip step 4")
    parser.add_argument("--skip-people", action="store_true", help="skip step 7")
    args = parser.parse_args()
    started = time.time()

    # Reference data first, once, before the workers start.
    db = ThreadDb(L.DATABASE)
    db.columns("football.fixtures")
    L.BUDGET.load(db)
    if args.used_so_far is not None:
        L.BUDGET.used = args.used_so_far
        L.BUDGET.save(db, force=True)
    tprint(f"API budget: {L.BUDGET.remaining:,} calls left this {L.BUDGET_PERIOD} (of {L.CALL_BUDGET:,})")
    api = L.Api(L.API_TOKEN, db)
    api.include_cache, api.split = SHARED_INCLUDE_CACHE, SHARED_SPLIT
    try:
        L.SIZE_GUARD.check(db, force=True)
        tprint(f"Database size: {L.SIZE_GUARD.last_gb:.1f} GB (limit {L.MAX_DB_SIZE_GB} GB)")
        L.load_core(api, db)
        leagues = resolve_leagues(api, args.leagues or LEAGUES)
        if args.estimate:
            tprint("\nESTIMATE")
            L.print_estimates([(lid, L.estimate_league(api, db, lid, args.refetch)) for lid in leagues])
            L.BUDGET.save(db)
            return
    except L.BudgetExhausted as e:
        tprint(f"\nSTOPPED: {e}")
        L.BUDGET.save(db)
        return

    workers = max(1, min(args.workers, len(leagues)))
    tprint(f"\nLoading {len(leagues)} leagues with {workers} workers: {', '.join(map(str, leagues))}")
    results = []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(run_league, lid, args) for lid in leagues]
    try:
        for fut in as_completed(futures):
            results.append(fut.result())
    except KeyboardInterrupt:
        STOP.set()
        tprint("\nCtrl+C - stopping all workers after their current request (a few seconds)...")
        pool.shutdown(wait=True, cancel_futures=True)
        results = [f.result() for f in futures if f.done() and not f.cancelled()]
    finally:
        pool.shutdown(wait=False)

    if not STOP.is_set():
        try:
            L.fill_missing(api, db)
        except L.BudgetExhausted:
            tprint("  stopped filling missing IDs (budget/storage limit)")
    L.BUDGET.save(db)

    if L.SIZE_GUARD.tripped:
        tprint(f"\nSTOPPED SAFELY: {L.SIZE_GUARD.tripped}. Progress is saved.")
        tprint("Skipping the analysis-view refresh (it needs extra space). Once there's room run:\n"
               "      SELECT analysis.refresh_all();")
    else:
        tprint("\n[8/8] Refreshing analysis views")
        db.execute("SELECT analysis.refresh_all()")

    tprint("\nSUMMARY")
    names = dict(db.query("SELECT league_id, name FROM football.leagues WHERE league_id = ANY(%s)", (leagues,)))
    counts = dict(db.query("SELECT league_id, count(*) FROM football.fixtures WHERE league_id = ANY(%s) "
                           "GROUP BY 1", (leagues,)))
    for lid, status, calls, secs in sorted(results, key=lambda r: leagues.index(r[0])):
        tprint(f"  {str(names.get(lid, lid)):<28} {status:<22} {counts.get(lid, 0):>7,} matches "
               f"{calls:>7,} calls {secs / 60:>6.1f} min")
    tprint(f"\n  {L.BUDGET.session:,} API calls this run in {(time.time() - started) / 60:.1f} minutes; "
           f"budget left: {L.BUDGET.remaining:,} of {L.CALL_BUDGET:,}")
    if L.WARNINGS:
        tprint(f"  {len(L.WARNINGS)} warnings (lines marked '!' above)")
    db.conn.close()


if __name__ == "__main__":
    main()
