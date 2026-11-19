#!/usr/bin/env python3
"""
REFEREE DECISION IMPACT LEDGER
What were the referees' big decisions actually WORTH - and to whom?

1. A win-probability model for every match
   Each side's scoring rate comes from the closing 1X2 odds (de-margined, averaged
   over bookmakers; team ratings from that season's results when odds are
   missing). Through the match it updates for the score, the time left and
   players sent off.

2. Every penalty and red card is valued in expected league points
   value = expected points just after the decision - just before it.
   A penalty at 0-0 in the 88th minute is worth far more than at 4-0.

3. Outputs (interactive dashboard: decision_impact.html)
   * Teams: points each team gained/lost from decisions, and the league table
     with those points removed ("table without the referees").
   * Referees: expected points per match their decisions hand the HOME side
     and the FAVOURITE, relative to peers in the same competition, with 90%
     intervals (empirical-Bayes shrinkage).
   * Make-up calls: after a card to one team, is the next card more likely to
     go to the other team than the two teams' usual card rates predict?
     > 1 = balancing, < 1 = piling on.
   * Biggest calls: the decisions that moved win probability most.

RUN (same folder as load_league.py; no API calls)
    python3 decision_impact.py                     # top 5 European leagues
    python3 decision_impact.py --leagues 8 2 5     # any league ids
    python3 decision_impact.py --all-leagues
Then open decision_impact.html.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

TOP5 = [8, 564, 82, 384, 301]
REPORT_FILE = "decision_impact.html"
MATCH_MINUTES = 95          # typical length including stoppage time
RED_OWN = 0.68              # scoring rate multiplier for the side a player down
RED_OPP = 1.22              # ... and for their opponents
MAXG = 10                   # goals per side considered in the probability grid
FINISHED = ("FT", "AET", "FT_PEN")
GOAL_TYPES = ("GOAL", "PENALTY", "OWNGOAL")
RED_TYPES = ("REDCARD", "YELLOWREDCARD")
CARD_TYPES = ("YELLOWCARD", "REDCARD", "YELLOWREDCARD")
PEN_TYPES = ("PENALTY", "MISSED_PENALTY")


# ================================================================== MODEL ===
def poisson(lam):
    p = [math.exp(-lam)]
    for k in range(1, MAXG + 1):
        p.append(p[-1] * lam / k)
    return p


_cache = {}


def outcome(lh, la, diff):
    """P(home win, draw, away win) given remaining-goal rates and current goal difference."""
    key = (round(lh, 4), round(la, 4), diff)
    if key in _cache:
        return _cache[key]
    ph, pa = poisson(lh), poisson(la)
    w = d = 0.0
    for i, a in enumerate(ph):
        for j, b in enumerate(pa):
            x = diff + i - j
            if x > 0:
                w += a * b
            elif x == 0:
                d += a * b
    res = (w, d, max(0.0, 1 - w - d))
    if len(_cache) < 2_000_000:
        _cache[key] = res
    return res


def solve_rates(p_home, p_draw, p_away):
    """Full-match scoring rates (home, away) whose Poisson 1X2 matches the odds."""
    lh, la = 1.5, 1.1
    def err(a, b):
        w, d, l_ = outcome(a, b, 0)
        return (w - p_home) ** 2 + (d - p_draw) ** 2 + (l_ - p_away) ** 2
    best = err(lh, la)
    step = 0.4
    while step > 0.002:
        improved = False
        for da, db in ((step, 0), (-step, 0), (0, step), (0, -step), (step, -step), (-step, step)):
            a, b = max(0.05, lh + da), max(0.05, la + db)
            e = err(a, b)
            if e < best:
                lh, la, best, improved = a, b, e, True
        if not improved:
            step /= 2
    return lh, la


class MatchModel:
    def __init__(self, lh, la):
        self.lh, self.la = lh, la

    def probs(self, t, diff, men):
        """t = minutes played, diff = home goals - away goals, men = home players - away players."""
        rem = max(0.0, (MATCH_MINUTES - t) / MATCH_MINUTES)
        fh = fa = 1.0
        if men < 0:
            fh, fa = RED_OWN ** -men, RED_OPP ** -men
        elif men > 0:
            fh, fa = RED_OPP ** men, RED_OWN ** men
        return outcome(self.lh * rem * fh, self.la * rem * fa, diff)

    def ep(self, t, diff, men):
        w, d, l_ = self.probs(t, diff, men)
        return 3 * w + d, 3 * l_ + d, w, l_


# ================================================================== DATA ===
def implied_1x2(db, match_ids):
    """Closing 1X2 probabilities per match (margin removed, averaged over bookmakers)."""
    markets = [r[0] for r in db.query(
        "SELECT market_id FROM odds.markets WHERE coalesce(developer_name, '') ~* "
        "'(FULL.?TIME.?RESULT|MATCH.?WINNER|^1X2$|3.?WAY)' OR coalesce(name, '') ~* "
        "'(full.?time.?result|match winner|^1x2$|3.?way result)'")] or [1]
    out = {}
    for table in ("odds.premium_odds", "odds.prematch_odds"):
        try:
            rows = db.query(
                f"SELECT fixture_id, bookmaker_id, lower(coalesce(label, name, '')), decimal_odds, "
                f"latest_bookmaker_update FROM {table} WHERE market_id = ANY(%s) AND fixture_id = ANY(%s) "
                f"AND decimal_odds > 1", (markets, list(match_ids)))
        except Exception:
            db.conn.rollback()
            continue
        latest = {}
        for fid, bm, label, odd, upd in rows:
            side = "h" if label in ("1", "home") else "d" if label in ("x", "draw") else \
                "a" if label in ("2", "away") else None
            if side is None or fid in out:
                continue
            key = (fid, bm, side)
            stamp = str(upd or "")
            if key not in latest or stamp >= latest[key][1]:
                latest[key] = (float(odd), stamp)
        books = defaultdict(dict)
        for (fid, bm, side), (odd, _) in latest.items():
            books[(fid, bm)][side] = 1 / odd
        per_match = defaultdict(list)
        for (fid, bm), sides in books.items():
            if len(sides) == 3:
                tot = sum(sides.values())
                per_match[fid].append((sides["h"] / tot, sides["d"] / tot, sides["a"] / tot))
        for fid, lst in per_match.items():
            if fid not in out:
                out[fid] = tuple(sum(x[i] for x in lst) / len(lst) for i in range(3))
    return out


def rating_rates(matches):
    """Fallback scoring rates from each team's season goals for/against."""
    tot = defaultdict(lambda: [0.0, 0.0, 0])
    team = defaultdict(lambda: [0.0, 0.0, 0])
    for m in matches.values():
        if m["hg"] is None:
            continue
        k = (m["lg"], m["se"])
        tot[k][0] += m["hg"]; tot[k][1] += m["ag"]; tot[k][2] += 1
        for t, gf, ga in ((m["h"], m["hg"], m["ag"]), (m["a"], m["ag"], m["hg"])):
            team[(t, k)][0] += gf; team[(t, k)][1] += ga; team[(t, k)][2] += 1
    rates = {}
    for fid, m in matches.items():
        k = (m["lg"], m["se"])
        n = tot[k][2]
        if not n:
            rates[fid] = (1.5, 1.15)
            continue
        hm, am = tot[k][0] / n, tot[k][1] / n
        avg = (hm + am) / 2 or 1.3
        def r(t):
            gf, ga, c = team[(t, k)]
            return ((gf + 3 * avg) / (c + 3)) / avg, ((ga + 3 * avg) / (c + 3)) / avg
        (ah, dh), (aa, da) = r(m["h"]), r(m["a"])
        rates[fid] = (max(0.2, hm * ah * da), max(0.2, am * aa * dh))
    return rates


def main():
    ap = argparse.ArgumentParser(description="Referee decision impact ledger")
    ap.add_argument("--leagues", nargs="+", type=int, default=TOP5)
    ap.add_argument("--all-leagues", action="store_true")
    args = ap.parse_args()
    leagues = None if args.all_leagues else args.leagues
    db = L.Db(L.DATABASE)
    db.columns("football.fixtures")
    q = db.query
    lim = " AND f.league_id = ANY(%(lg)s)" if leagues else ""
    P = {"lg": leagues, "fin": list(FINISHED)}

    matches = {}
    for fid, lg, se, date, h, a, ref in q(
            "SELECT f.fixture_id, f.league_id, f.season_id, f.starting_at::date::text, f.home_team_id, "
            "f.away_team_id, mr.referee_id FROM football.fixtures f "
            "LEFT JOIN analysis.main_referee mr ON mr.fixture_id = f.fixture_id "
            "WHERE f.state_id IN (SELECT state_id FROM core.states WHERE developer_name = ANY(%(fin)s))"
            " AND f.home_team_id IS NOT NULL AND f.away_team_id IS NOT NULL" + lim, P):
        matches[fid] = {"lg": lg, "se": se, "date": date, "h": h, "a": a, "ref": ref, "hg": None, "ag": None}
    if not matches:
        sys.exit("No finished matches for these leagues.")
    ids = list(matches)
    print(f"Finished matches: {len(matches):,} ({sum(1 for m in matches.values() if m['ref']):,} with a referee)")

    for fid, loc, goals in q("SELECT fixture_id, location, goals FROM football.fixture_scores "
                             "WHERE description = 'CURRENT' AND fixture_id = ANY(%(ids)s)", {"ids": ids}):
        if fid in matches and goals is not None:
            matches[fid]["hg" if loc == "home" else "ag"] = int(goals)

    events = defaultdict(list)
    for fid, eid, minute, extra, order, tid, pid, coach, dev, section in q(
            "SELECT e.fixture_id, e.event_id, e.minute, e.extra_minute, e.sort_order, e.team_id, e.player_id, "
            "e.coach_id, t.developer_name, e.section FROM football.fixture_events e "
            "JOIN core.types t ON t.type_id = e.type_id WHERE e.fixture_id = ANY(%(ids)s) "
            "AND e.rescinded IS NOT TRUE AND t.developer_name = ANY(%(types)s)",
            {"ids": ids, "types": list(set(GOAL_TYPES + CARD_TYPES + PEN_TYPES))}):
        if (minute or 0) > 125 or "shootout" in str(section or "").lower():
            continue
        events[fid].append(((minute or 0) + (extra or 0) * 0.01, order if order is not None else 0, eid,
                            minute or 0, extra or 0, tid, pid, coach, dev))
    for fid in events:
        events[fid].sort()

    # final scores from goal events where the score table is missing
    for fid, m in matches.items():
        if m["hg"] is None:
            hg = sum(1 for e in events[fid] if (e[8] in ("GOAL", "PENALTY") and e[5] == m["h"])
                     or (e[8] == "OWNGOAL" and e[5] == m["a"]))
            ag = sum(1 for e in events[fid] if (e[8] in ("GOAL", "PENALTY") and e[5] == m["a"])
                     or (e[8] == "OWNGOAL" and e[5] == m["h"]))
            if events[fid]:
                m["hg"], m["ag"] = hg, ag

    odds = implied_1x2(db, ids)
    fallback = rating_rates(matches)
    print(f"Closing odds found for {len(odds):,} matches; team ratings used for the rest")
    total_pens = sum(1 for evs in events.values() for e in evs if e[8] in PEN_TYPES)
    scored = sum(1 for evs in events.values() for e in evs if e[8] == "PENALTY")
    p_conv = scored / total_pens if total_pens >= 30 else 0.77
    print(f"Penalties: {total_pens:,} (conversion {p_conv:.0%}); "
          f"red cards: {sum(1 for evs in events.values() for e in evs if e[8] in RED_TYPES):,}")

    # team-season card habits for the make-up-call test
    cards_per = defaultdict(lambda: [0, 0])
    for fid, m in matches.items():
        for side in ("h", "a"):
            n = sum(1 for e in events[fid] if e[8] in CARD_TYPES and e[5] == m[side] and e[6])
            cards_per[(m[side], m["lg"], m["se"])][0] += n
            cards_per[(m[side], m["lg"], m["se"])][1] += 1
    league_cards = defaultdict(lambda: [0, 0])
    for (t, lg, se), (n, c) in cards_per.items():
        league_cards[(lg, se)][0] += n; league_cards[(lg, se)][1] += c

    def card_rate(t, lg, se):
        n, c = cards_per[(t, lg, se)]
        ln, lc = league_cards[(lg, se)]
        base = ln / lc if lc else 2.0
        return (n + 5 * base) / (c + 5)                 # shrunk towards the league

    decisions, match_rows, pairs = [], [], defaultdict(lambda: [0, 0, 0.0])
    done = 0
    for fid, m in matches.items():
        probs = odds.get(fid)
        if probs:
            lh, la = solve_rates(*probs)
            src = 1
        else:
            lh, la = fallback[fid]
            src = 0
            w, d, l_ = outcome(lh, la, 0)
            probs = (w, d, l_)
        model = MatchModel(lh, la)
        diff = men = 0
        ep_h = ep_a = 0.0
        n_dec = 0
        for (_, _, eid, minute, extra, tid, pid, coach, dev) in events[fid]:
            t = minute + extra
            side = "h" if tid == m["h"] else "a" if tid == m["a"] else None
            if side is None:
                continue
            if dev in PEN_TYPES:
                b_h, b_a, w0, l0 = model.ep(t, diff, men)
                step = 1 if side == "h" else -1
                g_h, g_a, wg, lg_ = model.ep(t, diff + step, men)
                a_h = p_conv * g_h + (1 - p_conv) * b_h
                a_a = p_conv * g_a + (1 - p_conv) * b_a
                wa = p_conv * wg + (1 - p_conv) * w0
                la_ = p_conv * lg_ + (1 - p_conv) * l0
                dh, da = a_h - b_h, a_a - b_a
                kind, benef = 0, side
            elif dev in RED_TYPES:
                b_h, b_a, w0, l0 = model.ep(t, diff, men)
                nm = men - 1 if side == "h" else men + 1
                a_h, a_a, wa, la_ = model.ep(t, diff, nm)
                dh, da = a_h - b_h, a_a - b_a
                kind, benef = 1, "a" if side == "h" else "h"
            else:
                kind = None
            if kind is not None and m["ref"]:
                wp_b = w0 if benef == "h" else l0
                wp_a = wa if benef == "h" else la_
                decisions.append([len(match_rows), minute, extra, kind, 0 if benef == "h" else 1,
                                  round(dh, 4), round(da, 4), round(wp_b, 4), round(wp_a, 4), diff, men,
                                  1 if dev == "PENALTY" else 0])
                ep_h += dh; ep_a += da; n_dec += 1
            # update the state AFTER the event
            if dev in ("GOAL", "PENALTY"):
                diff += 1 if side == "h" else -1
            elif dev == "OWNGOAL":
                diff += -1 if side == "h" else 1
            elif dev in RED_TYPES:
                men += -1 if side == "h" else 1

        # make-up-call test: consecutive player cards in this match
        if m["ref"]:
            seq = [e[5] for e in events[fid] if e[8] in CARD_TYPES and e[6] and e[5] in (m["h"], m["a"])]
            if len(seq) >= 2:
                rh, ra = card_rate(m["h"], m["lg"], m["se"]), card_rate(m["a"], m["lg"], m["se"])
                p = rh / (rh + ra)
                agg = pairs[(m["ref"], m["lg"], m["se"])]
                for x, y in zip(seq, seq[1:]):
                    agg[0] += 1
                    agg[1] += x != y
                    agg[2] += 2 * p * (1 - p)

        fav = 1 if probs[0] > probs[2] + 0.02 else -1 if probs[2] > probs[0] + 0.02 else 0
        hp = None if m["hg"] is None else (3 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 0)
        ap_ = None if m["hg"] is None else (3 if m["ag"] > m["hg"] else 1 if m["hg"] == m["ag"] else 0)
        match_rows.append([fid, m["lg"], m["se"], m["date"], m["h"], m["a"], m["ref"], m["hg"], m["ag"],
                           hp, ap_, round(ep_h, 4), round(ep_a, 4), n_dec, fav, src,
                           round(probs[0], 3), round(probs[2], 3)])
        done += 1
        if done % 2000 == 0:
            print(f"  {done:,}/{len(matches):,} matches modelled")

    # names + compact ids
    def idx(values):
        vals = sorted({v for v in values if v is not None})
        return vals, {v: i for i, v in enumerate(vals)}
    lgs, lgi = idx(r[1] for r in match_rows)
    ses, sei = idx(r[2] for r in match_rows)
    tms, tmi = idx([r[4] for r in match_rows] + [r[5] for r in match_rows])
    rfs, rfi = idx(r[6] for r in match_rows)
    one = lambda sql, ids_: dict(q(sql, {"ids": list(ids_)})) if ids_ else {}
    lg_names = {r[0]: ((r[2] + " · ") if r[2] else "") + (r[1] or "") for r in q(
        "SELECT l.league_id, l.name, c.name FROM football.leagues l LEFT JOIN core.countries c "
        "ON c.country_id = l.country_id WHERE l.league_id = ANY(%(ids)s)", {"ids": lgs})}
    se_names = one("SELECT season_id, name FROM football.seasons WHERE season_id = ANY(%(ids)s)", ses)
    se_order = one("SELECT season_id, coalesce(starting_at::text, name) FROM football.seasons "
                   "WHERE season_id = ANY(%(ids)s)", ses)
    tm_names = one("SELECT team_id, name FROM football.teams WHERE team_id = ANY(%(ids)s)", tms)
    rf_names = one("SELECT referee_id, coalesce(display_name, common_name, name) FROM football.referees "
                   "WHERE referee_id = ANY(%(ids)s)", rfs)

    M = [[lgi[r[1]], sei[r[2]], r[3], tmi[r[4]], tmi[r[5]], -1 if r[6] is None else rfi[r[6]],
          -1 if r[7] is None else r[7], -1 if r[8] is None else r[8], -1 if r[9] is None else r[9],
          -1 if r[10] is None else r[10]] + r[11:] for r in match_rows]
    C = [[rfi[k[0]], lgi[k[1]], sei[k[2]], v[0], v[1], round(v[2], 3)]
         for k, v in pairs.items() if k[0] in rfi and k[1] in lgi and k[2] in sei]
    data = {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "pConv": round(p_conv, 3),
                 "redOwn": RED_OWN, "redOpp": RED_OPP, "withOdds": len(odds)},
        "leagues": [lg_names.get(x, f"league {x}") for x in lgs],
        "seasons": [se_names.get(x, f"season {x}") for x in ses],
        "seasonOrder": [str(se_order.get(x, "")) for x in ses],
        "teams": [tm_names.get(x, f"team {x}") for x in tms],
        "refs": [rf_names.get(x, f"referee {x}") for x in rfs],
        "M": M, "D": decisions, "C": C}
    path = os.path.abspath(REPORT_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"\nDashboard written: {path}  ({len(decisions):,} decisions in {len(M):,} matches)")
    print("Open it in your browser (on a Mac:  open decision_impact.html)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decision Impact</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#677084;--line:#d9dde5;--grid:#eceff3;--accent:#2d6cdf;
--pos:#1f8a5b;--neg:#c0392b;--home:#2d6cdf;--away:#d4892a;--neutral:#9aa3b2}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e6e9ef;--muted:#99a2b3;--line:#2c3340;--grid:#242a35;
--accent:#6aa0ff;--pos:#4cc38a;--neg:#e0675a;--home:#6aa0ff;--away:#e6a650;--neutral:#6f7888}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 4px}h1{font-size:20px;margin:0 0 4px}header p{margin:0;color:var(--muted);max-width:920px}
.bar{display:flex;flex-wrap:wrap;gap:10px;padding:10px 20px;align-items:end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
select,input{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
.tabs{display:flex;gap:4px;padding:0 20px;border-bottom:1px solid var(--line);overflow-x:auto}
.tabs button{font:inherit;background:none;border:0;border-bottom:2px solid transparent;padding:10px 12px;color:var(--muted);cursor:pointer;white-space:nowrap}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
main{padding:16px 20px 30px;display:grid;gap:16px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.grid2.wide{grid-template-columns:repeat(auto-fit,minmax(560px,1fr))}
@media (max-width:640px){.grid2,.grid2.wide{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}
.card h2{font-size:15px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin-bottom:10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
svg{display:block;width:100%;height:auto}svg text{font-size:11px;fill:var(--ink)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:500;cursor:pointer;user-select:none}th:first-child,td:first-child,th.l,td.l{text-align:left}
.scroll{overflow:auto;max-height:640px}.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;font-weight:600;color:#fff}
.note{color:var(--muted);font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;display:none;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.15);max-width:320px}
.method p{max-width:860px;margin:0 0 10px}
</style></head><body>
<header><h1>Referee Decision Impact</h1>
<p>Every penalty and red card valued in <b>expected league points</b>: how much it changed each team's chances of winning, given the score, the minute, players on the pitch and the pre-match odds.</p></header>
<div class="bar">
 <label>Competition<select id="lg"></select></label>
 <label>Season<select id="se"></select></label>
 <label>Min. matches (referees)<input type="number" id="minm" value="15" min="1" style="width:90px"></label>
</div>
<nav class="tabs" id="tabs"><button data-tab="overview" class="on">Overview</button><button data-tab="teams">Teams</button><button data-tab="refs">Referees</button><button data-tab="calls">Biggest calls</button><button data-tab="method">How it works</button></nav>
<main id="main"></main><div id="tip"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (n,d=2) => isFinite(n) ? (+n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : "–";
const sgn = (n,d=2) => isFinite(n) ? (n>0?"+":"")+fmt(n,d) : "–";
// M: [lg, se, date, home, away, ref, hg, ag, hpts, apts, epH, epA, nDec, fav(1 home,-1 away,0), oddsSource, pHome, pAway]
// D: [matchIdx, minute, extra, kind(0 pen,1 red), benef(0 home,1 away), dEPhome, dEPaway, wpBefore, wpAfter, diff, men, scored]
let tab = "overview", sortTeams = {k:"net", dir:-1};

function filtered(){const lg=$("#lg").value, se=$("#se").value; const keep=new Set();
  D.M.forEach((m,i)=>{ if((lg=="all"||m[0]==+lg)&&(se=="all"||D.seasons[m[1]]==se)) keep.add(i)});
  return {keep, dec:D.D.filter(d=>keep.has(d[0]))}}

// ---------------------------------------------------------------- stats helpers
function ebNormal(groups, all){ // groups: [{key, vals}] -> shrunk means relative to the pooled mean
  const mu=all.length?all.reduce((a,b)=>a+b,0)/all.length:0;
  const st=groups.map(g=>{const n=g.vals.length, m=g.vals.reduce((a,b)=>a+b,0)/n, v=g.vals.reduce((a,b)=>a+(b-m)**2,0)/Math.max(1,n-1);
    return {...g, n, mean:m, se2:Math.max(v,1e-6)/n}});
  const between=st.length>1?st.reduce((a,g)=>a+(g.mean-mu)**2,0)/(st.length-1):0, noise=st.length?st.reduce((a,g)=>a+g.se2,0)/st.length:0;
  const tau2=Math.max(between-noise, 1e-5);
  return {mu, rows:st.map(g=>{const w=tau2/(tau2+g.se2), est=w*(g.mean-mu), sd=Math.sqrt(w*g.se2); return {...g, est, lo:est-1.645*sd, hi:est+1.645*sd}})}}
function ebShare(items){ // items [{key,name,k,n}] -> shrunk share k/n, centred on the pooled share
  const K0=items.reduce((s,i)=>s+i.k,0), N0=items.reduce((s,i)=>s+i.n,0); if(!N0) return {p:NaN, rows:[]};
  const p=K0/N0, live=items.filter(i=>i.n>0); const mean=live.reduce((s,i)=>s+i.n*(i.k/i.n-p)**2,0)/N0;
  const tau2=Math.max(mean-p*(1-p)*live.length/N0, 1e-4), K=Math.max(2, p*(1-p)/tau2-1);
  return {p, rows:items.map(i=>{const est=(i.k+K*p)/(i.n+K), sd=Math.sqrt(est*(1-est)/(i.n+K)); return {...i, est, lo:est-1.645*sd, hi:est+1.645*sd}})}}
function gq(a,b,z){const c=1/(9*a); return a*Math.pow(Math.max(0,1-c+z*Math.sqrt(c)),3)/b}
function ebGamma(items){ // items [{key, O, E}] -> ratio O/E shrunk (1 = expected)
  const So=items.reduce((s,i)=>s+i.O,0), Se=items.reduce((s,i)=>s+i.E,0); if(!Se) return {pooled:NaN, rows:[]};
  const m=So/Se, live=items.filter(i=>i.E>0); let v=0; live.forEach(i=>v+=i.E*(i.O/i.E-m)**2); v/=Se;
  const tau2=Math.max(v-m*live.length/Se, m*m*0.002), a0=m*m/tau2, b0=m/tau2;
  return {pooled:m, rows:items.map(i=>{const a=a0+i.O, b=b0+i.E; return {...i, est:a/b, lo:gq(a,b,-1.645), hi:gq(a,b,1.645)}})}}

// ---------------------------------------------------------------- charts
function forest(rows, opts){ // rows: [{name, est, lo, hi, n, flag}] ; opts.center, opts.fmt
  const W=640,rowH=20,left=180,right=70,top=6,H=top+rows.length*rowH+30; if(!rows.length) return "<p class='note'>Not enough data with these filters.</p>";
  const lo=Math.min(opts.center-opts.min, ...rows.map(r=>r.lo)), hi=Math.max(opts.center+opts.min, ...rows.map(r=>r.hi));
  const X=v=>left+(v-lo)/(hi-lo)*(W-left-right); let s="";
  const ticks=opts.ticks(lo,hi); ticks.forEach(t=>s+=`<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${top+rows.length*rowH}" stroke="${css(t==opts.center?"--muted":"--grid")}" ${t==opts.center?'stroke-dasharray="3 3"':""}/>
    <text x="${X(t)}" y="${top+rows.length*rowH+14}" text-anchor="middle" fill="${css("--muted")}">${opts.tick(t)}</text>`);
  rows.forEach((r,i)=>{const y=top+i*rowH+rowH/2, c=r.lo>opts.center?css(opts.upColor):r.hi<opts.center?css(opts.downColor):css("--neutral");
    s+=`<g data-t="${r.name}: ${opts.label(r)}"><rect x="0" y="${y-rowH/2}" width="${W}" height="${rowH}" fill="transparent"/>
    <text x="${left-8}" y="${y+4}" text-anchor="end">${r.name.length>26?r.name.slice(0,25)+"…":r.name}</text>
    <line x1="${X(r.lo)}" x2="${X(r.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="${X(r.est)}" cy="${y}" r="${3+Math.min(4,Math.sqrt(r.n)/4)}" fill="${c}" stroke="${css("--card")}" stroke-width="1.5"/>
    <text x="${W-right+8}" y="${y+4}" fill="${css("--muted")}">${opts.tick(r.est)}</text></g>`});
  return `<svg viewBox="0 0 ${W} ${H}">${s}</svg>`}
const linTicks=()=>(lo,hi)=>{const step=[0.01,0.02,0.05,0.1,0.2,0.25,0.5,1,2].find(s=>(hi-lo)/s<=7)||5, t=[];
  for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)t.push(+v.toFixed(4)); return t};

function bars(items, opts){ // items [{label, value, color, title}] vertical bars
  const W=520,H=220,pad=40,bw=(W-pad-10)/items.length, mx=Math.max(...items.map(i=>i.value),1e-9);
  let s=`<line x1="${pad}" x2="${W-10}" y1="${H-28}" y2="${H-28}" stroke="${css("--line")}"/>`;
  items.forEach((it,i)=>{const h=(H-44)*it.value/mx, x=pad+i*bw+6;
    s+=`<rect x="${x}" y="${H-28-h}" width="${bw-12}" height="${h}" rx="3" fill="${it.color}" data-t="${it.title}"/>
      <text x="${x+(bw-12)/2}" y="${H-12}" text-anchor="middle" fill="${css("--muted")}">${it.label}</text>
      <text x="${x+(bw-12)/2}" y="${H-32-h}" text-anchor="middle">${opts.fmt(it.value)}</text>`});
  return `<svg viewBox="0 0 ${W} ${H}">${s}</svg>`}

// ---------------------------------------------------------------- views
function overview(){const {keep,dec}=filtered(); const ms=[...keep].map(i=>D.M[i]).filter(m=>m[5]>=0);
  const pens=dec.filter(d=>d[3]==0), reds=dec.filter(d=>d[3]==1);
  const swing=d=>d[4]==0?d[5]-d[6]:d[6]-d[5];                         // points swing to the beneficiary
  const avgSwing=dec.length?dec.reduce((s,d)=>s+swing(d),0)/dec.length:NaN;
  const netHome=ms.map(m=>m[10]-m[11]), homeMean=netHome.reduce((a,b)=>a+b,0)/Math.max(1,netHome.length);
  const homeSe=Math.sqrt(netHome.reduce((a,b)=>a+(b-homeMean)**2,0)/Math.max(1,netHome.length-1)/Math.max(1,netHome.length));
  const favMs=ms.filter(m=>m[14]!=0), netFav=favMs.map(m=>m[14]==1?m[10]-m[11]:m[11]-m[10]), favMean=netFav.reduce((a,b)=>a+b,0)/Math.max(1,netFav.length);
  const cards=D.C.filter(c=>($("#lg").value=="all"||c[1]==+$("#lg").value)&&($("#se").value=="all"||D.seasons[c[2]]==$("#se").value));
  const alt=cards.reduce((s,c)=>s+c[4],0), exp=cards.reduce((s,c)=>s+c[5],0), npairs=cards.reduce((s,c)=>s+c[3],0);
  const bands=[[0,15],[16,30],[31,45],[46,60],[61,75],[76,200]];
  const byBand=bands.map(([a,b])=>{const ds=dec.filter(d=>d[1]>=a&&d[1]<=b); return {n:ds.length, avg:ds.length?ds.reduce((s,d)=>s+swing(d),0)/ds.length:0}});
  const homeP=pens.filter(d=>d[4]==0).length, homeR=reds.filter(d=>d[4]==0).length;
  let h=`<div class="kpis">
   <div class="kpi"><b>${fmt(ms.length,0)}</b><span>matches with a referee · ${fmt(pens.length,0)} penalties · ${fmt(reds.length,0)} red cards</span></div>
   <div class="kpi"><b>${fmt(avgSwing)}</b><span>average points swing per decision (to the side it favours)</span></div>
   <div class="kpi"><b class="${homeMean>0?"pos":"neg"}">${sgn(homeMean,3)}</b><span>net expected points per match to the HOME side from decisions (90%: ${sgn(homeMean-1.645*homeSe,3)} to ${sgn(homeMean+1.645*homeSe,3)})</span></div>
   <div class="kpi"><b class="${favMean>0?"pos":"neg"}">${sgn(favMean,3)}</b><span>net expected points per match to the pre-match FAVOURITE</span></div>
   <div class="kpi"><b>${fmt(exp?alt/exp:NaN)}</b><span>make-up-call index: next card to the other team ${fmt(alt,0)} times vs ${fmt(exp,0)} expected (${fmt(npairs,0)} card pairs). &gt;1 balancing, &lt;1 piling on</span></div></div>
  <div class="grid2">
   <section class="card"><h2>Points swing by match period</h2><div class="sub">Average points swing per decision, by match period (bar label = swing; hover for counts)</div>
    ${bars(byBand.map((b,i)=>({label:["0–15","16–30","31–45+","46–60","61–75","76–90+"][i],value:b.avg,color:css("--accent"),title:`${b.n} decisions, average swing ${fmt(b.avg)} points`})),{fmt:v=>fmt(v)})}</section>
   <section class="card"><h2>Who the decisions favour</h2><div class="sub">Count of decisions favouring each side (penalties awarded; red cards shown to the opponent)</div>
    ${bars([{label:"Pens → home",value:homeP,color:css("--home"),title:`${homeP} penalties to the home side`},{label:"Pens → away",value:pens.length-homeP,color:css("--away"),title:`${pens.length-homeP} penalties to the away side`},
            {label:"Reds → helps home",value:homeR,color:css("--home"),title:`${homeR} red cards shown to the away side`},{label:"Reds → helps away",value:reds.length-homeR,color:css("--away"),title:`${reds.length-homeR} red cards shown to the home side`}],{fmt:v=>fmt(v,0)})}</section>
  </div>
  <p class="note">Win probability from ${fmt(D.meta.withOdds,0)} matches with closing odds; team ratings elsewhere. Penalty conversion ${fmt(D.meta.pConv*100,0)}%. Home/favourite effects include legitimate reasons (home and stronger sides attack more) — the Referees tab compares referees with each other, which removes that.</p>`;
  $("#main").innerHTML=h}

function teams(){const {keep,dec}=filtered(); const t=new Map();
  const get=(id)=>{if(!t.has(id))t.set(id,{id,name:D.teams[id],p:0,pts:0,net:0,penF:0,penA:0,redF:0,redA:0,gain:0,loss:0});return t.get(id)};
  keep.forEach(i=>{const m=D.M[i]; if(m[8]<0) return; const H=get(m[3]),A=get(m[4]); H.p++;A.p++;H.pts+=m[8];A.pts+=m[9]; H.net+=m[10]; A.net+=m[11]});
  dec.forEach(d=>{const m=D.M[d[0]]; const ben=get(d[4]==0?m[3]:m[4]), oth=get(d[4]==0?m[4]:m[3]);
    if(d[3]==0){ben.penF++;oth.penA++} else {ben.redF++;oth.redA++}
    const db=d[4]==0?d[5]:d[6], dob=d[4]==0?d[6]:d[5]; ben.gain+=db; oth.loss+=dob});
  let rows=[...t.values()].filter(r=>r.p>0); rows.forEach(r=>r.adj=r.pts-r.net);
  const rank=(arr,k)=>{const s=[...arr].sort((a,b)=>b[k]-a[k]||b.net-a.net);s.forEach((r,i)=>r[k+"Rank"]=i+1)}; rank(rows,"pts"); rank(rows,"adj");
  rows.forEach(r=>r.move=r.ptsRank-r.adjRank);
  const k=sortTeams.k; rows.sort((a,b)=>sortTeams.dir*((typeof a[k]=="string")?a[k].localeCompare(b[k]):(a[k]-b[k])));
  const mx=Math.max(0.5,...rows.map(r=>Math.abs(r.net)));
  const cols=[["name","Team","l"],["p","P"],["pts","Pts"],["ptsRank","Pos"],["penF","Pens for"],["penA","Pens agst"],["redF","Opp. reds"],["redA","Own reds"],["net","Points from decisions"],["adj","Pts without decisions"],["adjRank","Pos without"],["move","Move"]];
  const one=$("#se").value!="all"&&$("#lg").value!="all";
  let h=`<section class="card"><h2>The table without the referees</h2><div class="sub">${one?"":"Tip: pick one competition and one season for a true league table. "}
   "Points from decisions" = expected points each team's penalties and red cards gave (+) or cost (−) them. "Pts without decisions" removes them. Click a column to sort.</div>
   <div class="scroll"><table><tr>${cols.map(([c,lab,cl])=>`<th class="${cl||""}" data-k="${c}">${lab}${sortTeams.k==c?(sortTeams.dir<0?" ▾":" ▴"):""}</th>`).join("")}</tr>`+
   rows.map(r=>`<tr><td class="l">${r.name}</td><td>${r.p}</td><td>${r.pts}</td><td>${r.ptsRank}</td><td>${r.penF}</td><td>${r.penA}</td><td>${r.redF}</td><td>${r.redA}</td>
     <td><svg viewBox="0 0 120 14" style="width:120px;display:inline-block;vertical-align:middle"><line x1="60" x2="60" y1="0" y2="14" stroke="${css("--line")}"/>
       <rect x="${r.net>=0?60:60-56*Math.abs(r.net)/mx}" y="3" width="${56*Math.abs(r.net)/mx}" height="8" rx="2" fill="${css(r.net>=0?"--pos":"--neg")}"/></svg>
       <span class="${r.net>=0?"pos":"neg"}">${sgn(r.net,1)}</span></td>
     <td>${fmt(r.adj,1)}</td><td>${r.adjRank}</td><td class="${r.move>0?"pos":r.move<0?"neg":"muted"}">${r.move>0?"▲"+r.move:r.move<0?"▼"+(-r.move):"–"}</td></tr>`).join("")+
   `</table></div></section>`;
  $("#main").innerHTML=h;
  document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{const k=th.dataset.k; sortTeams={k,dir:sortTeams.k==k?-sortTeams.dir:(k=="name"?1:-1)}; teams()})}

function refs(){const {keep}=filtered(); const minm=+$("#minm").value; const by=new Map(), allHome=[], allFav=[];
  keep.forEach(i=>{const m=D.M[i]; if(m[5]<0) return; if(!by.has(m[5])) by.set(m[5],{home:[],fav:[],dec:0});
    const g=by.get(m[5]), nh=m[10]-m[11]; g.home.push(nh); allHome.push(nh); g.dec+=m[12];
    if(m[14]!=0){const nf=m[14]==1?nh:-nh; g.fav.push(nf); allFav.push(nf)} });
  const groups=[...by.entries()].filter(([r,g])=>g.home.length>=minm);
  const H=ebNormal(groups.map(([r,g])=>({key:r,name:D.refs[r],vals:g.home,dec:g.dec})),allHome);
  const F=ebNormal(groups.filter(([r,g])=>g.fav.length>=3).map(([r,g])=>({key:r,name:D.refs[r],vals:g.fav})),allFav);
  const cards=new Map(); D.C.forEach(c=>{ if(($("#lg").value=="all"||c[1]==+$("#lg").value)&&($("#se").value=="all"||D.seasons[c[2]]==$("#se").value)){
    if(!cards.has(c[0])) cards.set(c[0],{O:0,E:0,n:0}); const x=cards.get(c[0]); x.n+=c[3]; x.O+=c[4]; x.E+=c[5]}});
  const okRefs=new Set(groups.map(([r])=>r));
  const pens=new Map(); filtered().dec.forEach(d=>{ if(d[3]!=0) return; const r=D.M[d[0]][5]; if(!okRefs.has(r)) return;
    if(!pens.has(r)) pens.set(r,{k:0,n:0}); const x=pens.get(r); x.n++; if(d[4]==0) x.k++ });
  const S=ebShare([...okRefs].map(r=>({key:r,name:D.refs[r],k:pens.get(r)?.k||0,n:pens.get(r)?.n||0})).filter(x=>x.n>0));
  const B=ebGamma([...cards.entries()].filter(([r])=>okRefs.has(r)).map(([r,x])=>({key:r,name:D.refs[r],...x})));
  const hRows=H.rows.sort((a,b)=>b.est-a.est), fRows=F.rows.sort((a,b)=>b.est-a.est), bRows=B.rows.map(r=>({...r,n:r.n})).sort((a,b)=>b.est-a.est);
  const pts=v=>sgn(v,2);
  let h=`<div class="grid2 wide">
   <section class="card"><h2>Home advantage from decisions, vs other referees</h2><div class="sub">Net expected points per match his decisions give the home side, minus the average referee (${pts(H.mu)} per match here). Right = more home-friendly.</div>
    ${forest(hRows.map(r=>({...r})),{center:0,min:0.1,ticks:linTicks(0.1),tick:v=>pts(v),upColor:"--home",downColor:"--away",label:r=>`${pts(r.est)} pts/match vs peers (90%: ${pts(r.lo)} to ${pts(r.hi)}) · ${r.n} matches · ${r.dec} decisions`})}
    <div class="legend"><span><i class="dot" style="background:var(--home)"></i>clearly more home-friendly</span><span><i class="dot" style="background:var(--away)"></i>clearly more away-friendly</span><span><i class="dot" style="background:var(--neutral)"></i>within normal range</span></div></section>
   <section class="card"><h2>Share of his penalties given to the home side</h2><div class="sub">Counting test — sharper than points for spotting home bias, because penalties are rare and red cards add noise. Pooled: ${fmt(S.p*100,0)}% to the home side. Right = more home-friendly.</div>
    ${forest(S.rows.sort((a,b)=>b.est-a.est).map(r=>({...r,n:r.n})),{center:S.p,min:0.1,ticks:linTicks(0.1),tick:v=>fmt(v*100,0)+"%",upColor:"--home",downColor:"--away",label:r=>`${fmt(r.est*100,0)}% to home (90%: ${fmt(r.lo*100,0)}–${fmt(r.hi*100,0)}%) · ${r.k} of ${r.n} penalties to the home side`})}</section>
   <section class="card"><h2>Favourite vs underdog, vs other referees</h2><div class="sub">Net expected points per match his decisions give the pre-match favourite, minus the average referee (${pts(F.mu)}). Right = favours favourites.</div>
    ${forest(fRows,{center:0,min:0.1,ticks:linTicks(0.1),tick:v=>pts(v),upColor:"--pos",downColor:"--neg",label:r=>`${pts(r.est)} pts/match vs peers (90%: ${pts(r.lo)} to ${pts(r.hi)}) · ${r.n} matches`})}</section>
   <section class="card"><h2>Make-up calls: does the next card go to the other team?</h2><div class="sub">Observed ÷ expected share of consecutive cards that switch team, given both teams' usual card rates. Pooled: ${fmt(B.pooled)}. &gt;1 balancing, &lt;1 piling on.</div>
    ${forest(bRows,{center:1,min:0.15,ticks:linTicks(0.1),tick:v=>fmt(v,2),upColor:"--home",downColor:"--neg",label:r=>`index ${fmt(r.est)} (90%: ${fmt(r.lo)}–${fmt(r.hi)}) · switched ${fmt(r.O,0)} of ${fmt(r.n,0)} pairs vs ${fmt(r.E,1)} expected`})}
    <div class="legend"><span><i class="dot" style="background:var(--home)"></i>balances the cards</span><span><i class="dot" style="background:var(--neg)"></i>piles on</span></div></section>
  </div>`;
  $("#main").innerHTML=h}

function calls(){const {dec}=filtered(); const swing=d=>d[4]==0?d[5]-d[6]:d[6]-d[5];
  const top=[...dec].sort((a,b)=>Math.abs(b[8]-b[7])-Math.abs(a[8]-a[7])).slice(0,40);
  let h=`<section class="card"><h2>The 40 decisions that moved win probability most</h2><div class="sub">Win probability of the side the decision favoured, just before → just after (penalties at the expected conversion rate, not the actual kick).</div>
   <div class="scroll"><table><tr><th class="l">Date</th><th class="l">Match</th><th class="l">Referee</th><th>Min</th><th class="l">Decision</th><th>Score</th><th>Win prob.</th><th>Points swing</th></tr>`+
   top.map(d=>{const m=D.M[d[0]], ben=d[4]==0?D.teams[m[3]]:D.teams[m[4]], oth=d[4]==0?D.teams[m[4]]:D.teams[m[3]];
     const what=d[3]==0?`Penalty to ${ben} <span class="muted">(${d[11]?"scored":"missed"})</span>`:`Red card to ${oth}`;
     const sc=d[9]>0?`home +${d[9]}`:d[9]<0?`away +${-d[9]}`:"level";
     return `<tr><td class="l">${m[2]||""}</td><td class="l">${D.teams[m[3]]} ${m[6]>=0?m[6]+"–"+m[7]:""} ${D.teams[m[4]]}</td><td class="l">${D.refs[m[5]]||""}</td>
       <td>${d[1]}${d[2]?"+"+d[2]:""}'</td><td class="l">${what}</td><td>${sc}${d[10]?` · ${d[10]>0?"home":"away"} +${Math.abs(d[10])} man`:""}</td>
       <td>${fmt(d[7]*100,0)}% → <b>${fmt(d[8]*100,0)}%</b></td><td class="pos">+${fmt(swing(d))}</td></tr>`}).join("")+`</table></div></section>`;
  $("#main").innerHTML=h}

function method(){$("#main").innerHTML=`<section class="card method"><h2>How it works</h2>
 <p><b>Scoring rates.</b> For each match, the closing 1X2 odds (margin removed, averaged across bookmakers) are turned into the two teams' expected goals, using a Poisson model that reproduces the home/draw/away probabilities. Without odds, team attack/defence ratings from that season's results are used.</p>
 <p><b>In-play win probability.</b> At any minute, the remaining expected goals are the full-match rate × the share of the ${95} minutes left. A side a player down scores at ${D.meta.redOwn}× its rate and concedes at ${D.meta.redOpp}×. The current goal difference is added to the simulated remaining goals.</p>
 <p><b>Decision value.</b> Expected points (3 × win + 1 × draw) just after the decision minus just before. A penalty is valued at the average conversion rate (${fmt(D.meta.pConv*100,0)}%), not at what actually happened, so a missed penalty still counts as a decision in that team's favour. A red card changes the player counts for the rest of the match.</p>
 <p><b>Table without the referees.</b> Actual points minus each team's net expected points from decisions. It doesn't say decisions were wrong — only how much they were worth.</p>
 <p><b>Referee comparisons</b> use each match's net points to the home side (and to the favourite), relative to the average referee in the same filter, with empirical-Bayes shrinkage so small samples are pulled towards average; bars are 90% intervals.</p>
 <p><b>Make-up calls.</b> For consecutive cards in a match, the chance the second goes to the other team if cards were independent is 2p(1−p), where p is the home side's share of the two teams' usual card rates that season. Observed ÷ expected above 1 suggests balancing.</p>
 <p class="note">Limits: decisions not given (penalties waved away) can't be valued; VAR overturns aren't included; the red-card multipliers are fixed approximations from published research rather than estimated per league.</p></section>`}

function render(){({overview,teams,refs,calls,method})[tab]()}
function init(){const n=new Map(); D.M.forEach(m=>n.set(m[0],(n.get(m[0])||0)+1));
  $("#lg").innerHTML=`<option value="all">All competitions</option>`+[...n.entries()].sort((a,b)=>b[1]-a[1]).map(([l,c])=>`<option value="${l}">${D.leagues[l]} (${fmt(c,0)})</option>`).join("");
  fillSeasons()}
function fillSeasons(){const lg=$("#lg").value, s=new Map(); D.M.forEach(m=>{if(lg=="all"||m[0]==+lg) s.set(D.seasons[m[1]],D.seasonOrder[m[1]])});
  const prev=$("#se").value; $("#se").innerHTML=`<option value="all">All seasons</option>`+[...s.entries()].sort((a,b)=>b[1].localeCompare(a[1])).map(([x])=>`<option>${x}</option>`).join("");
  if([...$("#se").options].some(o=>o.value==prev)) $("#se").value=prev}
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>{tab=b.dataset.tab;document.querySelectorAll("#tabs button").forEach(x=>x.classList.toggle("on",x==b));render()});
$("#lg").onchange=()=>{fillSeasons();render()}; $("#se").onchange=render; $("#minm").oninput=render;
document.addEventListener("mousemove",ev=>{const t=ev.target.closest&&ev.target.closest("[data-t]"),tip=$("#tip");
  if(t){tip.textContent=t.getAttribute("data-t");tip.style.display="block";tip.style.left=Math.min(ev.clientX+12,innerWidth-330)+"px";tip.style.top=(ev.clientY+12)+"px"}else tip.style.display="none"});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
init(); render();
</script></body></html>"""


if __name__ == "__main__":
    main()
