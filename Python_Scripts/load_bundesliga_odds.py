#!/usr/bin/env python3
"""
Load ONLY the pre-match odds and premium odds (+ price history) for the
Bundesliga seasons set in load_league.py.

Uses load_league.py for everything (API key, database, seasons, call budget),
so keep both files in the same folder.

RUN
  python3 load_bundesliga_odds.py
  python3 load_bundesliga_odds.py --league 564     # same thing for La Liga
  python3 load_bundesliga_odds.py --refetch        # re-download odds already loaded

Calls: about 4 to list the matches, then ~1 per 10 matches for odds and
~1 per 5 matches for premium odds (~280 for 3 Bundesliga seasons).
Matches whose odds are already loaded are skipped; re-running is safe.
"""

import argparse
import sys
import time

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

ODDS_PASSES = ("odds", "premium_odds")


def main():
    parser = argparse.ArgumentParser(description="Load odds + premium odds for one league")
    parser.add_argument("--league", type=int, default=82, help="Sportmonks league id (default 82 = Bundesliga)")
    parser.add_argument("--refetch", action="store_true", help="re-download odds already loaded")
    args = parser.parse_args()

    # Only the two odds passes run.
    for name, cfg in L.FIXTURE_PASSES.items():
        cfg["enabled"] = name in ODDS_PASSES

    started = time.time()
    db = L.Db(L.DATABASE)
    db.columns("football.fixtures")
    L.BUDGET.load(db)
    print(f"API budget: {L.BUDGET.remaining:,} calls left this {L.BUDGET_PERIOD} (of {L.CALL_BUDGET:,})")
    api = L.Api(L.API_TOKEN, db)

    try:
        L.load_core(api, db)                       # skipped if already loaded (needs bookmakers/markets)
        season_ids = L.load_league_and_seasons(api, db, args.league)
        fixtures = []
        for sid in season_ids:
            fixtures.extend(L.load_season_structure(api, db, sid))
        L.load_fixtures(api, db, fixtures, ODDS_PASSES if args.refetch else ())
    except L.BudgetExhausted as e:
        print(f"\nSTOPPED: {e}. Progress is saved; run again later to continue.")
    finally:
        db.commit()
        L.BUDGET.save(db)

    print("\nSUMMARY")
    for label, sql in (
        ("matches with odds", "SELECT count(DISTINCT fixture_id) FROM odds.prematch_odds WHERE league_id = %s"),
        ("odds rows", "SELECT count(*) FROM odds.prematch_odds WHERE league_id = %s"),
        ("matches with premium odds", "SELECT count(DISTINCT fixture_id) FROM odds.premium_odds WHERE league_id = %s"),
        ("premium odds rows", "SELECT count(*) FROM odds.premium_odds WHERE league_id = %s"),
        ("premium price-history rows", "SELECT count(*) FROM odds.premium_odds_history h "
                                       "JOIN odds.premium_odds o USING (odd_id) WHERE o.league_id = %s"),
    ):
        print(f"  {label:<30}{db.query(sql, (args.league,))[0][0]:>12,}")
    print(f"\n  {L.BUDGET.session:,} API calls in {(time.time() - started) / 60:.1f} minutes; "
          f"budget left: {L.BUDGET.remaining:,}")
    if L.WARNINGS:
        print(f"  {len(L.WARNINGS)} warnings (shown above with '!')")


if __name__ == "__main__":
    main()
