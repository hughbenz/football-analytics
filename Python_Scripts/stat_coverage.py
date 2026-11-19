#!/usr/bin/env python3
"""
STAT COVERAGE TABLE - % of finished fixtures that have each stat, for every league.

For every stat type in
    team match stats    football.fixture_statistics
    player match stats  football.fixture_lineup_details
    team xG             football.fixture_xg
    player xG           football.fixture_xg_players
it counts, per league (or per league-season with --by-season), how many FINISHED
fixtures (FT / AET / FT_PEN) have at least one row of that stat type, and divides by
the league's finished fixtures.

Outputs (current folder, prefix changeable with --out):
    stat_coverage.html       heatmap table: stats down the side, leagues across the top,
                             with a search box. Open in a browser.
    stat_coverage_wide.csv   same matrix as CSV (opens in Excel / pandas)
    stat_coverage_long.csv   one row per league x stat, with the counts behind each %

Read-only: no API calls, nothing is written to the database.
Every aggregation runs inside PostgreSQL, so Python memory stays small.

RUN (same folder as load_league.py):
    python3 stat_coverage.py
    python3 stat_coverage.py --by-season              # one column per league-season
    python3 stat_coverage.py --leagues 8 82 564       # only some leagues
    python3 stat_coverage.py --sources team player    # only some stat sources

Note: Sportmonks sometimes omits a stat row when its value is 0, so a rare stat
(e.g. penalties) can look less covered than it really is.
"""

import argparse
import csv
import html
import sys
import time
from collections import defaultdict
from datetime import datetime

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

FINISHED = ("FT", "AET", "FT_PEN")
SOURCES = {                      # key -> (label, table)
    "team":   ("Team match stats",   "football.fixture_statistics"),
    "player": ("Player match stats", "football.fixture_lineup_details"),
    "xg":     ("Team xG",            "football.fixture_xg"),
    "xgp":    ("Player xG",          "football.fixture_xg_players"),
}


def short_season(name):
    """'2025/2026' -> '25/26'."""
    p = (name or "").split("/")
    return f"{p[0][-2:]}/{p[1][-2:]}" if len(p) == 2 else (name or "?")


# ================================================================= COLLECT ===
def collect(db, leagues, by_season, sources):
    params = {"fin": list(FINISHED), "lg": leagues}
    lim = " AND f.league_id = ANY(%(lg)s)" if leagues else ""
    season_col = "f.season_id" if by_season else "NULL::bigint"
    fin_cte = (f"WITH fin AS (SELECT f.fixture_id, f.league_id, {season_col} AS season_id "
               f"FROM football.fixtures f JOIN core.states st ON st.state_id = f.state_id "
               f"WHERE st.developer_name = ANY(%(fin)s){lim}) ")

    def run(label, sql):
        t = time.time()
        try:
            rows = db.query(sql, params)
        except Exception as e:                       # table missing / not loaded
            db.conn.rollback()
            print(f"  {label:<34} skipped ({str(e).splitlines()[0]})")
            return []
        print(f"  {label:<34} {time.time() - t:6.1f}s")
        return rows

    print("Scanning the database (read-only)...")
    denom = {(lg, se): n for lg, se, n in
             run("finished fixtures", fin_cte + "SELECT league_id, season_id, count(*) FROM fin GROUP BY 1, 2")}
    if not denom:
        sys.exit("No finished fixtures found.")

    hits = {}                                        # (source, type_id) -> {(lg, se): n}
    for key in sources:
        label, table = SOURCES[key]
        rows = run(label, fin_cte +
                   f"SELECT fin.league_id, fin.season_id, x.type_id, count(*) "
                   f"FROM (SELECT DISTINCT s.fixture_id, s.type_id FROM {table} s "
                   f"      WHERE s.fixture_id IN (SELECT fixture_id FROM fin)) x "
                   f"JOIN fin ON fin.fixture_id = x.fixture_id GROUP BY 1, 2, 3")
        for lg, se, tid, n in rows:
            hits.setdefault((key, tid), {})[(lg, se)] = n

    types = {tid: (dev, name) for tid, dev, name in db.query(
        "SELECT type_id, coalesce(developer_name, name, type_id::text), coalesce(name, developer_name, '') FROM core.types")}
    lg_names = {lg: (name, country) for lg, name, country in db.query(
        "SELECT l.league_id, l.name, c.name FROM football.leagues l "
        "LEFT JOIN core.countries c ON c.country_id = l.country_id")}
    se_names = {se: name for se, name in db.query("SELECT season_id, name FROM football.seasons")} if by_season else {}
    return denom, hits, types, lg_names, se_names


# ================================================================== SHAPE ===
def build(denom, hits, types, lg_names, se_names, sources):
    # columns: leagues by finished fixtures (desc); seasons inside a league oldest first
    lg_total = defaultdict(int)
    for (lg, _), n in denom.items():
        lg_total[lg] += n
    cols = sorted(denom, key=lambda k: (-lg_total[k[0]], lg_names.get(k[0], ("",))[0] or "",
                                        se_names.get(k[1], "") if k[1] else ""))
    col_meta = []
    for lg, se in cols:
        name, country = lg_names.get(lg, (f"league {lg}", None))
        col_meta.append({"league_id": lg, "league": name, "country": country or "",
                         "season_id": se, "season": se_names.get(se, "") if se else "",
                         "label": name + (f" {short_season(se_names.get(se))}" if se else ""),
                         "fixtures": denom[(lg, se)]})
    total_fin = sum(denom.values())

    rows = []
    for key in sources:
        stats = [(tid, cnt) for (src, tid), cnt in hits.items() if src == key]
        block = []
        for tid, cnt in stats:
            dev, name = types.get(tid, (str(tid), ""))
            overall = sum(cnt.values()) / total_fin
            cells = [(cnt.get(c, 0), denom[c]) for c in cols]
            block.append({"source": key, "source_label": SOURCES[key][0], "type_id": tid, "stat": dev,
                          "name": name, "overall": overall, "cells": cells,
                          "leagues_full": sum(1 for h, d in cells if d and h / d >= 0.95)})
        block.sort(key=lambda r: (-r["overall"], r["stat"]))
        rows += block
    return col_meta, rows, total_fin


# ================================================================= OUTPUTS ===
def write_csvs(prefix, col_meta, rows, total_fin):
    with open(f"{prefix}_wide.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "stat", "type_id", "name", "all_leagues_pct"] + [c["label"] for c in col_meta])
        w.writerow(["", "finished fixtures", "", "", total_fin] + [c["fixtures"] for c in col_meta])
        w.writerow(["", "league_id", "", "", ""] + [c["league_id"] for c in col_meta])
        for r in rows:
            w.writerow([r["source_label"], r["stat"], r["type_id"], r["name"], round(r["overall"] * 100, 1)]
                       + [round(h / d * 100, 1) if d else "" for h, d in r["cells"]])
    with open(f"{prefix}_long.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["league_id", "league", "country", "season_id", "season", "source", "stat", "type_id",
                    "fixtures_with_stat", "finished_fixtures", "coverage_pct"])
        for r in rows:
            for c, (h, d) in zip(col_meta, r["cells"]):
                w.writerow([c["league_id"], c["league"], c["country"], c["season_id"] or "", c["season"],
                            r["source_label"], r["stat"], r["type_id"], h, d, round(h / d * 100, 1) if d else ""])


def cell_style(p):
    """White -> teal by coverage; dark text on light cells, white on dark."""
    lo, hi = (255, 255, 255), (15, 110, 86)
    rgb = tuple(round(a + (b - a) * p) for a, b in zip(lo, hi))
    return f"background:rgb{rgb};color:{'#fff' if p > 0.55 else '#2c2c2a'}"


def write_html(prefix, col_meta, rows, total_fin, by_season):
    e = html.escape
    head = "".join(
        f'<th class="col" title="{e(c["league"])} ({e(c["country"])}) id {c["league_id"]}'
        f'{" · " + e(c["season"]) if c["season"] else ""} · {c["fixtures"]:,} finished">'
        f'<div><span>{e(c["label"])}</span></div></th>' for c in col_meta)
    fixtures_row = "".join(f'<td class="n">{c["fixtures"]:,}</td>' for c in col_meta)
    body, current = [], None
    for r in rows:
        if r["source"] != current:
            current = r["source"]
            body.append(f'<tr class="grp"><th colspan="{len(col_meta) + 3}">{e(r["source_label"])}</th></tr>')
        cells = "".join(
            f'<td style="{cell_style(h / d if d else 0)}" title="{h:,} of {d:,}">{(h / d * 100 if d else 0):.0f}</td>'
            for h, d in r["cells"])
        body.append(f'<tr class="s" data-q="{e((r["stat"] + " " + r["name"]).lower())}">'
                    f'<th class="stat" title="{e(r["name"])} · type_id {r["type_id"]}">{e(r["stat"])}</th>'
                    f'<td class="all" style="{cell_style(r["overall"])}">{r["overall"] * 100:.0f}</td>'
                    f'<td class="n">{r["leagues_full"]}</td>{cells}</tr>')
    unit = "league-season" if by_season else "league"
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Stat coverage by {unit}</title>
<style>
body{{font:13px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#2c2c2a;margin:20px}}
h1{{font-size:20px;font-weight:600;margin:0 0 4px}} p{{color:#5f5e5a;margin:0 0 12px}}
input{{font:inherit;padding:6px 10px;border:1px solid #b4b2a9;border-radius:6px;width:260px;margin-bottom:12px}}
.wrap{{overflow:auto;max-height:82vh;border:1px solid #d3d1c7;border-radius:8px}}
table{{border-collapse:separate;border-spacing:0}}
th,td{{padding:3px 6px;border-bottom:1px solid #eee;white-space:nowrap}}
td{{text-align:center;min-width:30px;font-variant-numeric:tabular-nums}}
thead th{{position:sticky;top:0;background:#fff;z-index:2;vertical-align:bottom;border-bottom:1px solid #b4b2a9}}
th.col{{height:150px;padding:0 0 6px}} th.col div{{width:30px;display:flex;justify-content:center}}
th.col span{{writing-mode:vertical-rl;transform:rotate(180deg);font-weight:500}}
th.stat,thead th.left{{position:sticky;left:0;background:#fff;text-align:left;z-index:1;font-weight:500}}
thead th.left{{z-index:3}} tr.grp th{{background:#f1efe8;text-align:left;position:sticky;left:0}}
td.n{{color:#5f5e5a}} td.all{{font-weight:600}} tr.fx td,tr.fx th{{background:#faf9f5}}
</style></head><body>
<h1>Stat coverage by {unit}</h1>
<p>% of finished fixtures (FT, AET, FT_PEN) with at least one row of each stat. {len(col_meta)} {unit}s,
{total_fin:,} finished fixtures, {len(rows)} stats. Generated {datetime.now():%Y-%m-%d %H:%M}.
Hover a cell for counts. "Full" = leagues at 95% or more.</p>
<input id="q" placeholder="Filter stats, e.g. foul, shots, xg" oninput="f(this.value)">
<div class="wrap"><table>
<thead><tr><th class="left">Stat</th><th class="col"><div><span>All {unit}s</span></div></th>
<th class="col"><div><span>Full ({unit}s)</span></div></th>{head}</tr></thead>
<tbody><tr class="fx"><th class="stat">Finished fixtures</th><td class="n">{total_fin:,}</td><td></td>{fixtures_row}</tr>
{''.join(body)}</tbody></table></div>
<script>
function f(v){{v=v.toLowerCase().trim();document.querySelectorAll('tr.s').forEach(r=>{{
r.style.display=!v||r.dataset.q.includes(v)?'':'none'}});}}
</script></body></html>"""
    with open(f"{prefix}.html", "w", encoding="utf-8") as fh:
        fh.write(doc)


def main():
    ap = argparse.ArgumentParser(description="% coverage by stat for every league")
    ap.add_argument("--leagues", nargs="+", type=int, help="only these league_ids (default: all)")
    ap.add_argument("--by-season", action="store_true", help="one column per league-season")
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                    help="stat sources to include (default: all)")
    ap.add_argument("--out", default="stat_coverage", help="output file prefix")
    args = ap.parse_args()

    db = L.Db(L.DATABASE)
    denom, hits, types, lg_names, se_names = collect(db, args.leagues, args.by_season, args.sources)
    col_meta, rows, total_fin = build(denom, hits, types, lg_names, se_names, args.sources)
    write_csvs(args.out, col_meta, rows, total_fin)
    write_html(args.out, col_meta, rows, total_fin, args.by_season)
    print(f"\n{len(rows)} stats x {len(col_meta)} columns, {total_fin:,} finished fixtures.")
    print(f"Wrote {args.out}.html, {args.out}_wide.csv, {args.out}_long.csv")


if __name__ == "__main__":
    main()
