#!/usr/bin/env python3
"""
referee_ou.py - do any referees' matches go over / under 2.5 goals more than Pinnacle expects?

For every finished match with a referee and Pinnacle's last pre-kickoff Over/Under 2.5 prices:
  expected  = Pinnacle's probability of over 2.5, margin removed (1/over / (1/over + 1/under))
  actual    = 1 if 3+ goals in 90 minutes, else 0
  residual  = actual - expected            (positive = more goals than Pinnacle priced)
By default each residual is first adjusted for its league-season (league-season average residual
subtracted), so a referee is not credited with his league's pricing bias.

Per referee: overs vs expected, z-score (sum of residuals / sqrt(sum p(1-p))), two-sided p-value, and
Benjamini-Hochberg q-value across all referees tested. "Consistent" = the same direction in the first
and second half of his matches (in date order) and in most of his seasons.
A persistence check at the end asks whether referees' first-half residuals predict their second half at all.

Same folder as backtest_models.py (for the database connection), then:
    python3 referee_ou.py
    python3 referee_ou.py --min-matches 40 --csv referees_ou.csv --no-league-adjust
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import backtest_models as bt  # noqa: E402

MATCHES = """
WITH sc AS (SELECT fixture_id,
       max(goals) FILTER (WHERE location='home' AND description='CURRENT')  AS h_cur, max(goals) FILTER (WHERE location='away' AND description='CURRENT')  AS a_cur,
       max(goals) FILTER (WHERE location='home' AND description='2ND_HALF') AS h_90,  max(goals) FILTER (WHERE location='away' AND description='2ND_HALF') AS a_90
       FROM football.fixture_scores WHERE description IN ('CURRENT','2ND_HALF') GROUP BY fixture_id),
ref AS (SELECT DISTINCT ON (fo.fixture_id) fo.fixture_id, fo.referee_id
        FROM football.fixture_officials fo LEFT JOIN core.types t ON t.type_id = fo.type_id
        WHERE t.developer_name = 'REFEREE' OR lower(t.name) = 'referee'
        ORDER BY fo.fixture_id, fo.fixture_official_id)
SELECT f.fixture_id, f.starting_at, f.league_id, l.name AS league, f.season_id, ref.referee_id,
       CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.h_90, sc.h_cur) ELSE sc.h_cur END
     + CASE WHEN st.developer_name IN ('AET','FT_PEN') THEN COALESCE(sc.a_90, sc.a_cur) ELSE sc.a_cur END AS goals
FROM football.fixtures f
JOIN core.states st ON st.state_id = f.state_id AND st.developer_name IN ('FT','AET','FT_PEN')
JOIN ref ON ref.fixture_id = f.fixture_id
JOIN sc ON sc.fixture_id = f.fixture_id
LEFT JOIN football.leagues l ON l.league_id = f.league_id"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--market-ou", type=int, default=80, help="Sportmonks market id for Over/Under (same default as the other scripts)")
    ap.add_argument("--bookmaker", default="pinnacle", help="bookmaker name to match (case-insensitive)")
    ap.add_argument("--min-matches", type=int, default=30)
    ap.add_argument("--q", type=float, default=0.05, help="false-discovery rate for 'significant'")
    ap.add_argument("--no-league-adjust", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args()

    db = bt.DB(args.dsn)
    m = db.query(MATCHES).dropna(subset=["goals", "referee_id"])
    names = db.query("SELECT referee_id, COALESCE(display_name, common_name, name) AS referee FROM football.referees") \
        if db.exists("football.referees") else pd.DataFrame(columns=["referee_id", "referee"])
    m = m.merge(names, on="referee_id", how="left")
    o = db.query(bt.odds_sql(args, "ou", False))
    o = o[o.bookmaker.fillna("").str.lower().str.contains(args.bookmaker.lower())].dropna(subset=["o_o", "o_u"])
    o = o.groupby("fixture_id")[["o_o", "o_u"]].mean().reset_index()
    d = m.merge(o, on="fixture_id")
    if d.empty:
        sys.exit(f"No matches with a referee, a score and {args.bookmaker} O/U 2.5 prices (try --market-ou or --bookmaker).")
    d["p"] = (1 / d.o_o) / (1 / d.o_o + 1 / d.o_u)
    d["y"] = (d.goals >= 3).astype(float)
    d["r"] = d.y - d.p
    if not args.no_league_adjust:
        g = d.groupby(["league_id", "season_id"]).r
        d["r"] = d.r - (g.transform("sum") - d.r) / (g.transform("size") - 1).clip(lower=1)   # league-season mean, excluding the match itself
    d["season"] = d.season_id
    d = d.sort_values("starting_at")
    print(f"{len(d):,} matches, {d.referee_id.nunique():,} referees, {d.league_id.nunique()} leagues. "
          f"Overall: {d.y.mean():.1%} went over vs {d.p.mean():.1%} priced by {args.bookmaker.title()}.\n")

    rows = []
    for rid, g in d.groupby("referee_id"):
        n = len(g)
        if n < args.min_matches:
            continue
        var = (g.p * (1 - g.p)).sum()
        z = g.r.sum() / math.sqrt(var)
        half = n // 2
        a, b = g.r.iloc[:half].mean(), g.r.iloc[half:].mean()
        ss = g.groupby("season").r.agg(["mean", "size"])
        ss = ss[ss["size"] >= 5]
        same = (np.sign(ss["mean"]) == np.sign(g.r.mean())).mean() if len(ss) else np.nan
        rows.append({"referee": g.referee.iloc[0] if pd.notna(g.referee.iloc[0]) else f"referee {int(rid)}", "league": g.league.mode().iloc[0], "matches": n,
                     "over_rate": g.y.mean(), "pinnacle_over": g.p.mean(), "resid": g.r.mean(), "z": z,
                     "p": math.erfc(abs(z) / math.sqrt(2)), "first_half": a, "second_half": b,
                     "seasons": len(ss), "seasons_same_dir": same})
    t = pd.DataFrame(rows)
    if t.empty:
        sys.exit(f"No referee has {args.min_matches}+ matches.")
    # Benjamini-Hochberg
    o_ = t.p.sort_values()
    q = (o_ * len(o_) / np.arange(1, len(o_) + 1))[::-1].cummin()[::-1].clip(upper=1)
    t["q"] = q.reindex(t.index)
    t["direction"] = np.where(t.resid > 0, "OVER", "UNDER")
    t["consistent"] = (np.sign(t.first_half) == np.sign(t.second_half)) & (t.seasons_same_dir.fillna(0) >= 0.66)
    t = t.sort_values("p")

    sig = t[t.q < args.q]
    pd.set_option("display.width", 200)
    fmt = lambda x: x.assign(over_rate=x.over_rate.map("{:.1%}".format), pinnacle_over=x.pinnacle_over.map("{:.1%}".format),  # noqa: E731
                             resid=x.resid.map("{:+.3f}".format), z=x.z.map("{:+.2f}".format), p=x.p.map("{:.4f}".format), q=x.q.map("{:.3f}".format),
                             first_half=x.first_half.map("{:+.3f}".format), second_half=x.second_half.map("{:+.3f}".format),
                             seasons_same_dir=x.seasons_same_dir.map(lambda v: "-" if pd.isna(v) else f"{v:.0%}"))
    cols = ["referee", "league", "matches", "direction", "over_rate", "pinnacle_over", "resid", "z", "p", "q", "first_half", "second_half", "seasons", "seasons_same_dir", "consistent"]
    print(f"{len(t)} referees with {args.min_matches}+ matches tested"
          f"{'' if args.no_league_adjust else ' (residuals adjusted for league-season)'}; "
          f"by chance alone ~{0.05 * len(t):.0f} would have raw p < 0.05 (actual: {(t.p < 0.05).sum()}).\n")
    if len(sig):
        print(f"SIGNIFICANT after correcting for {len(t)} tests (q < {args.q}):")
        print(fmt(sig)[cols].to_string(index=False))
    else:
        print(f"No referee is significant after correcting for {len(t)} tests (q < {args.q}). Closest 15:")
        print(fmt(t.head(15))[cols].to_string(index=False))
    # does a referee's tendency persist at all?
    rr = t[["first_half", "second_half"]].astype(float)
    r = rr.first_half.corr(rr.second_half)
    se = 1 / math.sqrt(max(len(rr) - 3, 1))
    print(f"\nPersistence across all {len(rr)} referees: first-half vs second-half residual correlation r = {r:+.3f} "
          f"(95% CI {math.tanh(math.atanh(r) - 1.96 * se):+.3f} to {math.tanh(math.atanh(r) + 1.96 * se):+.3f}). "
          "Near zero = referee tendencies vs Pinnacle are mostly noise.")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
