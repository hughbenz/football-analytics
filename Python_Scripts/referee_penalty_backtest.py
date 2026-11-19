#!/usr/bin/env python3
"""
referee_penalty_backtest.py - walk-forward backtest of "penalty in the match" markets against referees
======================================================================================================

Same method and page as referee_ou_backtest.py / referee_cards_backtest.py (keep all of them, and
backtest_models.py, in one folder), applied to penalties awarded:

  * Market: penalty yes/no (or penalties over/under 0.5). Found in odds.markets by name (contains 'penalt',
    excludes shoot-outs, team, player, half, scored/missed); candidates are printed, the one with most matches is
    used unless you pass --market-id. Default reference book: bookmaker id 2.
  * Expected: the reference book's P(a penalty), margin removed ('power' method by default: these markets carry
    big margins, mostly loaded onto the long 'yes' price).
  * Actual: penalties AWARDED in 90 minutes (stoppage time included, extra time and shoot-outs excluded):
    scored penalties plus missed/saved ones. Check how your bookmaker settles (awarded vs taken, extra time).
  * Margin: a bet is placed only when (fair chance + league tendency + referee effect) x the price actually paid
    leaves the chosen expected profit, so every result is after margin.
  * Everything else - earlier-days-only referee records, daily multiple-testing correction, league adjustment,
    effect-size staking, discovery / hold-out split, time-travel test - is identical to the other two.

    python3 referee_penalty_backtest.py
    python3 referee_penalty_backtest.py --market-id 123 --bookmaker-id 2 --out penalties.html
"""
import argparse
import html as htmlmod
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import backtest_models as bt  # noqa: E402
from backtest_models import log  # noqa: E402
import referee_ou_backtest as base  # noqa: E402

MATCHES = """
WITH ref AS (SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
        FROM football.fixture_officials fo LEFT JOIN core.types t ON t.type_id = fo.type_id
        WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
        ORDER BY fo.fixture_id, fo.fixture_official_id),
ev AS (SELECT e.fixture_id, count(*) AS pens
       FROM football.fixture_events e JOIN core.types t ON t.type_id = e.type_id
       WHERE upper(t.developer_name) IN ('PENALTY','MISSED_PENALTY','PENALTY_MISSED','PENALTY_SAVED')
         AND e.rescinded IS NOT TRUE AND e.minute <= 90
       GROUP BY 1),
has_ev AS (SELECT DISTINCT fixture_id FROM football.fixture_events)
SELECT f.fixture_id, f.starting_at, f.league_id, ref.referee_id,
       COALESCE(f.home_team_id, hp.team_id) AS home_id, COALESCE(f.away_team_id, ap.team_id) AS away_id,
       COALESCE(ev.pens, 0) AS pens
FROM football.fixtures f
JOIN core.states st ON st.state_id = f.state_id AND st.developer_name IN ('FT','AET','FT_PEN')
JOIN ref ON ref.fixture_id = f.fixture_id
JOIN has_ev he ON he.fixture_id = f.fixture_id          -- matches with no events at all are left out (no card data, not 0 cards)
LEFT JOIN ev ON ev.fixture_id = f.fixture_id
LEFT JOIN football.fixture_participants hp ON hp.fixture_id = f.fixture_id AND hp.location = 'home'
LEFT JOIN football.fixture_participants ap ON ap.fixture_id = f.fixture_id AND ap.location = 'away'"""

MARKETS = """
SELECT market_id, name, developer_name FROM odds.markets
WHERE (upper(coalesce(name,'')) LIKE '%PENALT%' OR upper(coalesce(developer_name,'')) LIKE '%PENALT%')
  AND upper(coalesce(name,'') || ' ' || coalesce(developer_name,'')) !~ '(SHOOT|TEAM|HOME|AWAY|HALF|PLAYER|SCORE|CONVERT|MISS|SAVE|FIRST|LAST|1ST|2ND|METHOD|WINNER|QUALIF)'"""


def odds_sql(mid):
    side = ("CASE WHEN lower(coalesce(o.label,'')) LIKE 'over%' OR lower(coalesce(o.label,'')) = 'yes' THEN 'O' "
            "WHEN lower(coalesce(o.label,'')) LIKE 'under%' OR lower(coalesce(o.label,'')) = 'no' THEN 'U' END")
    return f"""
    WITH o AS (
        SELECT o.fixture_id, o.bookmaker_id, core_num AS line, o.decimal_odds AS price, {side} AS side
        FROM (SELECT o.*, COALESCE(NULLIF(regexp_replace(coalesce(o.total, ''), '[^0-9.]', '', 'g'), '')::numeric, 0.5) AS core_num
              FROM odds.prematch_odds o WHERE o.market_id = {mid} AND o.decimal_odds > 1) o
        JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at
        UNION ALL
        SELECT o.fixture_id, o.bookmaker_id, core_num, o.decimal_odds, {side}
        FROM (SELECT o.*, COALESCE(NULLIF(regexp_replace(coalesce(o.total, ''), '[^0-9.]', '', 'g'), '')::numeric, 0.5) AS core_num
              FROM odds.premium_odds o WHERE o.market_id = {mid} AND o.decimal_odds > 1) o
        JOIN football.fixtures f ON f.fixture_id = o.fixture_id
        WHERE o.latest_bookmaker_update IS NULL OR o.latest_bookmaker_update <= f.starting_at)
    SELECT o.fixture_id, o.bookmaker_id, b.name AS bookmaker, o.line,
           avg(price) FILTER (WHERE side = 'O') AS o_o, avg(price) FILTER (WHERE side = 'U') AS o_u
    FROM o LEFT JOIN odds.bookmakers b ON b.bookmaker_id = o.bookmaker_id
    WHERE side IS NOT NULL AND o.line IS NOT NULL GROUP BY 1, 2, 3, 4"""


def devig(o, u, method):
    """Fair P(over) from the two prices. 'power' solves (1/o)^k + (1/u)^k = 1 (bisection), which puts more of the margin on the longer price."""
    a, b = 1 / o, 1 / u
    if method == "proportional":
        return a / (a + b)
    lo, hi = np.full(len(a), 1.0), np.full(len(a), 4.0)
    for _ in range(50):
        k = (lo + hi) / 2
        big = a ** k + b ** k > 1
        lo, hi = np.where(big, k, lo), np.where(big, hi, k)
    return a ** ((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--out", default="referee_cards_backtest.html")
    ap.add_argument("--market-id", type=int, default=0, help="odds market id for 'penalty in the match' (0 = auto-detect)")
    ap.add_argument("--bookmaker-id", type=int, default=2, help="reference bookmaker id (its prices define 'expected' and are the default betting price)")
    ap.add_argument("--bookmaker", help="reference bookmaker by name instead of id, or 'consensus' for the average of all books")
    ap.add_argument("--devig", choices=["power", "proportional"], default="power",
                    help="how the margin is removed: 'power' loads more of it onto the longer price (fits card markets' longshot bias), "
                         "'proportional' spreads it evenly")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--bh-min", type=int, default=30)
    ap.add_argument("--shrink", type=float, default=30)
    ap.add_argument("--league-k", type=float, default=100)
    ap.add_argument("--max-p", type=float, default=0.25)
    ap.add_argument("--holdout-share", type=float, default=0.4)
    ap.add_argument("--leak-tests", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    db = bt.DB(args.dsn)

    # ---- market
    if not args.market_id:
        cand = db.query(MARKETS)
        if cand.empty:
            sys.exit("No penalty market found in odds.markets - pass --market-id.")
        ids = ",".join(str(int(x)) for x in cand.market_id)
        cov = db.query(f"SELECT market_id, count(DISTINCT fixture_id) AS n FROM (SELECT market_id, fixture_id FROM odds.prematch_odds WHERE market_id IN ({ids}) "
                       f"UNION ALL SELECT market_id, fixture_id FROM odds.premium_odds WHERE market_id IN ({ids})) x GROUP BY 1")
        cand = cand.merge(cov, on="market_id", how="left").fillna({"n": 0}).sort_values("n", ascending=False)
        cand["yellow"] = (cand.name.fillna("") + cand.developer_name.fillna("")).str.upper().str.contains("MATCH|AWARD|YES")
        log("penalty markets found:\n" + "\n".join(f"    id {int(r.market_id):>6}  {int(r.n):>8,} matches  {r.name}" for r in cand.itertuples()))
        pick = cand[cand.yellow & (cand.n > 0)]
        pick = pick if len(pick) else cand[cand.n > 0]
        if pick.empty:
            sys.exit("Penalty markets exist but have no prices.")
        args.market_id = int(pick.market_id.iloc[0])
        log(f"using market {args.market_id} ({pick.name.iloc[0]}) - override with --market-id")
    mname = db.query(f"SELECT name FROM odds.markets WHERE market_id = {args.market_id}")
    mname = str(mname.name.iloc[0]) if len(mname) else f"market {args.market_id}"

    # ---- matches and prices
    m = db.query(MATCHES).dropna(subset=["referee_id", "starting_at"])
    m["cards"] = m.pens                                               # 'cards' = the counted stat below (penalties here)
    o = db.query(odds_sql(args.market_id)).dropna(subset=["o_o", "o_u"])
    o = o[(o.line * 2) % 2 == 1]                                       # .5 lines only
    o["pf"] = devig(o.o_o.to_numpy(float), o.o_u.to_numpy(float), args.devig)
    o["margin"] = 1 / o.o_o + 1 / o.o_u - 1
    if args.bookmaker and args.bookmaker.lower() == "consensus":
        ref = o.groupby(["fixture_id", "line"]).agg(pf=("pf", "mean"), o_o=("o_o", "mean"), o_u=("o_u", "mean")).reset_index()
        book = "Consensus"
    else:
        sel = o.bookmaker.fillna("").str.lower().str.contains(args.bookmaker.lower()) if args.bookmaker else (o.bookmaker_id == args.bookmaker_id)
        ref = o[sel].groupby(["fixture_id", "line"])[["pf", "o_o", "o_u"]].mean().reset_index()
        book = args.bookmaker.title() if args.bookmaker else (o.loc[sel, "bookmaker"].dropna().iloc[0] if o.loc[sel, "bookmaker"].notna().any() else f"Bookmaker {args.bookmaker_id}")
    if ref.empty:
        sys.exit(f"No {args.bookmaker or 'bookmaker id ' + str(args.bookmaker_id)} prices for market {args.market_id} (try --bookmaker consensus).")
    ref["dist"] = (ref.pf - 0.5).abs()
    ref = ref.sort_values(["fixture_id", "dist", "line"]).drop_duplicates("fixture_id")   # main line: closest to 50/50 (pre-match prices only)
    agg = o.merge(ref[["fixture_id", "line"]], on=["fixture_id", "line"]).groupby("fixture_id").agg(
        avg_o=("o_o", "mean"), avg_u=("o_u", "mean"), best_o=("o_o", "max"), best_u=("o_u", "max"))
    d = m.merge(ref.rename(columns={"o_o": "pin_o", "o_u": "pin_u", "pf": "p"})[["fixture_id", "line", "p", "pin_o", "pin_u"]], on="fixture_id")
    d = d.merge(agg, left_on="fixture_id", right_index=True, how="left")
    d["day"] = pd.to_datetime(d.starting_at, utc=True).dt.tz_convert(None).dt.normalize()
    d["goals"] = d.cards                                              # the shared engine calls the counted stat 'goals'
    d["y"] = (d.cards > d.line).astype(float)
    d = d.sort_values(["day", "fixture_id"]).reset_index(drop=True)
    if d.empty:
        sys.exit("No matches with a referee, events and penalty prices.")
    d["ref_i"], refs = pd.factorize(d.referee_id)
    d["lg_i"], lgs = pd.factorize(d.league_id)
    log(f"{len(d):,} matches with {book} '{mname}' prices, {d.y.mean():.1%} had a penalty · {len(refs):,} referees · {len(lgs)} competitions · "
        f"most common line {d.line.mode().iloc[0]} · average margin {(1 / d.pin_o + 1 / d.pin_u - 1).mean():.1%} (removed with the {args.devig} method)")

    def names(sql, key):
        try:
            x = db.query(sql)
            return dict(zip(x[key].astype("int64"), x.name))
        except Exception:  # noqa: BLE001
            return {}
    rname = names("SELECT referee_id, COALESCE(display_name, common_name, name) AS name FROM football.referees", "referee_id")
    tname = names("SELECT team_id, name FROM football.teams", "team_id")
    lgq = None
    for sql in ("SELECT league_id, name, type FROM football.leagues", "SELECT league_id, name, NULL AS type FROM football.leagues"):
        try:
            lgq = db.query(sql)
            break
        except Exception:  # noqa: BLE001
            continue
    lname = dict(zip(lgq.league_id.astype("int64"), lgq.name)) if lgq is not None else {}
    ltype = dict(zip(lgq.league_id.astype("int64"), lgq["type"])) if lgq is not None else {}

    log("walking forward day by day ...")
    W = base.walk_forward(d, args)
    leak = {"tests": [], "passed": True, "skipped": True}
    if args.leak_tests:
        log("time-travel test: re-running with the future deleted ...")
        leak = {**base.leak_test(d, W, args, args.leak_tests), "skipped": False}

    keep = np.isfinite(W["p"]) & (W["p"] <= args.max_p)
    hold_from = d.day[np.isfinite(W["p"])].quantile(1 - args.holdout_share).normalize() if np.isfinite(W["p"]).any() else d.day.max()
    teams = sorted({int(x) for x in pd.concat([d.home_id, d.away_id]).dropna()})
    tix = {t: i for i, t in enumerate(teams)}
    ix = np.where(keep)[0]
    r4 = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(a, float)]  # noqa: E731
    rows = {"d": d.day.dt.strftime("%Y-%m-%d").to_numpy()[ix].tolist(), "r": d.ref_i.to_numpy()[ix].tolist(), "l": d.lg_i.to_numpy()[ix].tolist(),
            "h": [tix.get(int(x), -1) if pd.notna(x) else -1 for x in d.home_id.to_numpy()[ix]],
            "a": [tix.get(int(x), -1) if pd.notna(x) else -1 for x in d.away_id.to_numpy()[ix]],
            "n": W["n"][ix].astype(int).tolist(), "e": r4(W["eff"][ix]), "p": r4(W["p"][ix]), "q": r4(W["q"][ix]), "s": W["steady"][ix].astype(int).tolist(),
            "y": d.y.to_numpy()[ix].astype(int).tolist(), "g": d.cards.to_numpy()[ix].astype(int).tolist(), "ln": r4(d.line.to_numpy()[ix]),
            "pp": r4(d.p.to_numpy()[ix]), "la": r4(W["ladj"][ix]), "po": r4(d.pin_o.to_numpy()[ix]), "pu": r4(d.pin_u.to_numpy()[ix]), "ao": r4(d.avg_o.to_numpy()[ix]),
            "au": r4(d.avg_u.to_numpy()[ix]), "bo": r4(d.best_o.to_numpy()[ix]), "bu": r4(d.best_u.to_numpy()[ix])}
    d["adj"] = W["adj"]
    d["year"] = d.day.dt.year
    summ = []
    for j, g in d.groupby("ref_i"):
        nn, var = len(g), float((g.p * (1 - g.p)).sum())
        z = g.adj.sum() / math.sqrt(var) if var > 0 else np.nan
        h1, h2 = g.adj.iloc[: nn // 2].mean(), g.adj.iloc[nn // 2:].mean()
        yrs = g.groupby("year").agg(n=("y", "size"), over=("y", "mean"), exp=("p", "mean"), adj=("adj", "mean")).reset_index()
        summ.append({"name": rname.get(int(refs[j])) or f"Referee {int(refs[j])}", "lg": int(g.lg_i.mode().iloc[0]), "n": nn, "over": g.y.mean(), "exp": g.p.mean(),
                     "eff": g.adj.sum() / (nn + args.shrink), "z": z, "p": float(base.pval(np.array([z]))[0]) if np.isfinite(z) else None,
                     "steady": bool(nn >= 10 and np.sign(h1) == np.sign(h2) != 0), "leagues": int(g.lg_i.nunique()), "last": str(g.day.max().date()),
                     "cards": float(g.cards.mean()), "rate": float((g.cards > 0).mean()), "years": yrs[["year", "n", "over", "exp", "adj"]].round(4).values.tolist()})
    qz = base.bh(np.array([s["p"] if s["p"] is not None and s["n"] >= args.bh_min else np.nan for s in summ], float))
    for s, q in zip(summ, qz):
        s["q"] = None if not np.isfinite(q) else float(q)
    payload = bt.clean({
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "db": db.name, "book": book, "market": mname, "devig": args.devig,
        "params": {"min_matches": args.min_matches, "bh_min": args.bh_min, "shrink": args.shrink, "league_k": args.league_k, "max_p": args.max_p},
        "hold_from": str(hold_from.date()), "from": str(d.day.min().date()), "to": str(d.day.max().date()),
        "n_matches": len(d), "over_rate": float(d.y.mean()), "exp_rate": float(d.p.mean()),
        "margin": float((1 / d.pin_o + 1 / d.pin_u - 1).mean()), "with_avg": float(d.avg_o.notna().mean()),
        "leagues": [lname.get(int(x)) or f"Competition {int(x)}" for x in lgs],
        "ltypes": [str(ltype.get(int(x)) or "").strip().lower() or "unknown" for x in lgs],
        "teams": [tname.get(t) or str(t) for t in teams], "refs": summ, "rows": rows, "leak": leak, "runtime": round(time.time() - t0)})
    page = page_template().replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__TITLE__", htmlmod.escape("Referee penalties backtest"))
    Path(args.out).write_text(page, encoding="utf-8")
    log(f"{len(ix):,} bettable match rows exported · page written: {args.out} ({len(page) / 1e6:.1f} MB)")
    if not leak["skipped"] and not leak["passed"]:
        log("!!! TIME-TRAVEL TEST FAILED - see the page")


def page_template():
    """The goals page, reworded for penalties ('over' = a penalty is awarded)."""
    t = base.TEMPLATE
    rep = [
        ('<span class="pill o">Over</span> more goals than priced', '<span class="pill o">More</span> more penalties than priced'),
        ('<span class="pill u">Under</span> fewer goals than priced', '<span class="pill u">Fewer</span> fewer penalties than priced'),
        ("${esc(R.book)} Over/Under 2.5 prices", "${esc(R.book)} '${esc(R.market)}' prices"),
        ("went over vs ${pctu(R.exp_rate)} priced.", "had a penalty vs ${pctu(R.exp_rate)} priced."),
        ("[['both', 'Overs & unders'], ['over', 'Overs'], ['under', 'Unders']]", "[['both', 'Yes & no'], ['over', 'Penalty: yes'], ['under', 'Penalty: no']]"),
        ("<th>Goals</th>", "<th>Penalties</th>"),
        ("${b.over ? 'Over' : 'Under'}</span></td><td>${b.price", "${b.over ? 'Yes' : 'No'}</span></td><td>${b.price"),
        ("[['all', 'Both'], ['over', 'Over'], ['under', 'Under']]", "[['all', 'Both'], ['over', 'More'], ['under', 'Fewer']]"),
        ("r.eff > 0 ? 'Over' : 'Under'}", "r.eff > 0 ? 'More' : 'Fewer'}"),
        ("['over', 'Went over',", "['over', 'Had a penalty',"),
        ("${int(s.over)} overs · ${int(s.n - s.over)} unders", "${int(s.over)} yes · ${int(s.n - s.over)} no"),
        ("${pctu(f.over)} over vs ${pctu(f.exp)} priced, league-adjusted effect", "${pctu(f.over)} with a penalty vs ${pctu(f.exp)} priced, ${f.cards.toFixed(2)} penalties per match, league-adjusted effect"),
        ("<h3>Each year: went over vs priced</h3>", "<h3>Each year: matches with a penalty vs priced</h3>"),
        ("></span>Went over</span>", "></span>Had a penalty</span>"),
        ("<br>Went over ${pctu(r[2])}", "<br>Had a penalty ${pctu(r[2])}"),
        ("gives the chance of 3+ goals (margin removed", "gives the chance of a penalty being awarded (margin removed"),
        ('A referee "leans over" when his matches go over more often than those prices said', 'A referee gives "more penalties" when his matches have one more often than those prices said'),
        ("of his matches went over than priced", "of his matches had a penalty than priced"),
        ("Bets settle on 90 minutes, like the market.", "Penalties are counted when awarded (scored or missed) up to 90 minutes including stoppage time; extra time and shoot-outs are excluded. Check how your bookmaker settles."),
    ]
    for a, b in rep:
        if a not in t:
            raise RuntimeError(f"template text not found: {a[:50]}")
        t = t.replace(a, b)
    return t


if __name__ == "__main__":
    main()
