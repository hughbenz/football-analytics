#!/usr/bin/env python3
"""
PENALTY EAGERNESS - is a referee quicker or slower to give a penalty than he is
to give a foul anywhere else on the pitch?

For every referee it compares, with the same matches:
  BOX       penalties given  vs  penalties expected from how much attacking
            happened in the box (shots from inside the box, at that league-
            season's rate of penalties per box shot)
  ELSEWHERE fouls given (excluding penalties)  vs  fouls expected from the two
            teams' usual foul counts that season
  EAGERNESS = BOX / ELSEWHERE
            > 1  relatively eager to give penalties
            < 1  relatively hesitant in the box
Small samples are pulled towards the average (empirical-Bayes gamma-Poisson),
and every figure has a 90% interval: a referee is only labelled eager or
hesitant when the whole interval is above or below 1.

If foul_maps.py has been run, a pitch-band profile is added: fouls given in the
attacking team's own third, middle third, final third and the box (penalties),
relative to the league - showing where along the pitch the whistle dries up.

Needs: matches with a referee, team statistics (fouls, shots inside the box)
and penalty events. No API calls.

RUN (same folder as load_league.py, which it reads the database settings from)
    python3 penalty_eagerness.py                   # top 5 European leagues
    python3 penalty_eagerness.py --leagues 8 2 5   # any league ids
    python3 penalty_eagerness.py --all-leagues
Then open penalty_eagerness.html in your browser.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

try:
    import load_league as L
except ImportError:
    sys.exit("load_league.py must be in the same folder as this script.")

TOP5 = [8, 564, 82, 384, 301]
REPORT_FILE = "penalty_eagerness.html"
FOUL_STATS = ["FOULS"]
BOX_STATS = ["SHOTS_INSIDEBOX", "SHOTS_INSIDE_BOX", "SHOTS_INSIDE_THE_BOX"]
FALLBACK_OPPORTUNITY = ["DANGEROUS_ATTACKS"]
MIN_BAND_FOULS = 10        # a match needs this many located fouls to join the pitch-band profile


def main():
    ap = argparse.ArgumentParser(description="Referee penalty eagerness vs the rest of the pitch")
    ap.add_argument("--leagues", nargs="+", type=int, default=TOP5, help="league ids (default: top 5)")
    ap.add_argument("--all-leagues", action="store_true", help="include every league")
    args = ap.parse_args()
    leagues = None if args.all_leagues else args.leagues

    db = L.Db(L.DATABASE)
    db.columns("football.fixtures")
    q = db.query
    lim = " AND f.league_id = ANY(%(lg)s)" if leagues else ""
    P = {"lg": leagues}

    # ---- which event types are penalties (not shoot-outs, not VAR checks)
    ptypes = q("SELECT type_id, developer_name FROM core.types WHERE model_type IS DISTINCT FROM 'statistic' "
               "AND developer_name ~ 'PENALTY' AND developer_name !~ '(SHOOTOUT|^VAR|SAVE)'")
    if not ptypes:
        sys.exit("No penalty event types found in core.types - has the loader's reference data been loaded?")
    print("Penalty events counted:", ", ".join(sorted({d for _, d in ptypes})))
    P["ptypes"] = [t for t, _ in ptypes]

    # ---- matches with a referee
    matches = {r[0]: {"lg": r[1], "se": r[2], "date": r[3], "h": r[4], "a": r[5], "ref": r[6]} for r in q(
        "SELECT f.fixture_id, f.league_id, f.season_id, f.starting_at::date::text, f.home_team_id, "
        "f.away_team_id, mr.referee_id FROM football.fixtures f "
        "JOIN analysis.main_referee mr ON mr.fixture_id = f.fixture_id "
        "WHERE f.state_id IN (SELECT state_id FROM core.states WHERE developer_name IN ('FT','AET','FT_PEN'))"
        + lim, P)}
    if not matches:
        sys.exit("No finished matches with a referee for these leagues.")
    print(f"Finished matches with a referee: {len(matches):,}")

    # ---- team statistics: fouls and box shots
    avail = {r[0] for r in q("SELECT DISTINCT t.developer_name FROM football.fixture_statistics s "
                             "JOIN core.types t ON t.type_id = s.type_id JOIN football.fixtures f "
                             "ON f.fixture_id = s.fixture_id WHERE t.developer_name = ANY(%(names)s)" + lim,
                             {**P, "names": FOUL_STATS + BOX_STATS + FALLBACK_OPPORTUNITY})}
    box_stat = next((n for n in BOX_STATS if n in avail), None) or next(
        (n for n in FALLBACK_OPPORTUNITY if n in avail), None)
    print(f"Penalty opportunity measured by: {box_stat or 'matches played (no box-shot statistic found)'}")
    stat = defaultdict(dict)          # (fixture, team) -> {"fouls": n, "box": n}
    for fid, tid, name, value in q(
            "SELECT s.fixture_id, s.team_id, t.developer_name, s.value FROM football.fixture_statistics s "
            "JOIN core.types t ON t.type_id = s.type_id JOIN football.fixtures f ON f.fixture_id = s.fixture_id "
            "WHERE t.developer_name = ANY(%(names)s)" + lim,
            {**P, "names": FOUL_STATS + ([box_stat] if box_stat else [])}):
        if value is not None and fid in matches:
            stat[(fid, tid)]["fouls" if name in FOUL_STATS else "box"] = float(value)

    # ---- penalties
    pens = defaultdict(list)          # fixture -> [(team, minute, scored)]
    for fid, tid, minute, dev in q(
            "SELECT e.fixture_id, e.team_id, e.minute, t.developer_name FROM football.fixture_events e "
            "JOIN core.types t ON t.type_id = e.type_id JOIN football.fixtures f ON f.fixture_id = e.fixture_id "
            "WHERE e.type_id = ANY(%(ptypes)s) AND e.rescinded IS NOT TRUE AND coalesce(e.minute, 0) <= 125 "
            "AND coalesce(e.section, '') !~* 'shootout'" + lim, P):
        if fid in matches:
            pens[fid].append((tid, minute, 0 if "MISS" in dev else 1))
    print(f"Penalties found: {sum(len(v) for v in pens.values()):,}")

    # ---- league-season baselines
    ls = defaultdict(lambda: {"pens": 0, "tm": 0, "box_pens": 0, "box": 0.0})
    for fid, m in matches.items():
        b = ls[(m["lg"], m["se"])]
        n_p = len(pens.get(fid, []))
        b["pens"] += n_p
        b["tm"] += 2
        hb, ab = stat[(fid, m["h"])].get("box"), stat[(fid, m["a"])].get("box")
        if hb is not None and ab is not None:
            b["box_pens"] += n_p
            b["box"] += hb + ab
    # team-season foul habits (committed / suffered per match)
    habit = defaultdict(lambda: {"c": 0.0, "s": 0.0, "n": 0})
    for fid, m in matches.items():
        fh, fa = stat[(fid, m["h"])].get("fouls"), stat[(fid, m["a"])].get("fouls")
        if fh is None or fa is None:
            continue
        for team, committed, suffered in ((m["h"], fh, fa), (m["a"], fa, fh)):
            h = habit[(team, m["lg"], m["se"])]
            h["c"] += committed
            h["s"] += suffered
            h["n"] += 1

    def avg(team, lg, se, key):
        h = habit.get((team, lg, se))
        return h[key] / h["n"] if h and h["n"] else None

    # ---- pitch bands from foul_maps.py output, if present
    bands = {}
    has_fouls = q("SELECT to_regclass('analysis.fouls') IS NOT NULL")[0][0] in (True, "t", 1)
    if has_fouls:
        for fid, own, mid, fin, n in q(
                "SELECT fo.fixture_id, "
                "count(*) FILTER (WHERE 1 - fo.x < 1/3.0), "
                "count(*) FILTER (WHERE 1 - fo.x >= 1/3.0 AND 1 - fo.x < 2/3.0), "
                "count(*) FILTER (WHERE 1 - fo.x >= 2/3.0), count(*) "
                "FROM analysis.fouls fo JOIN football.fixtures f ON f.fixture_id = fo.fixture_id "
                "WHERE fo.x IS NOT NULL" + lim + " GROUP BY 1", P):
            if fid in matches and int(n) >= MIN_BAND_FOULS:
                bands[fid] = (int(own), int(mid), int(fin))
        print(f"Matches with located fouls for the pitch-band profile: {len(bands):,}")
    else:
        print("analysis.fouls not found - run foul_maps.py first to add the pitch-band profile.")

    # ---- per match observed vs expected, summed per (referee, league, season)
    agg = defaultdict(lambda: defaultdict(float))
    pen_list = []
    for fid, m in matches.items():
        b = ls[(m["lg"], m["se"])]
        rate_box = b["box_pens"] / b["box"] if b["box"] else None
        rate_tm = b["pens"] / b["tm"] if b["tm"] else 0
        exp = {}
        for side, team in (("h", m["h"]), ("a", m["a"])):
            box = stat[(fid, team)].get("box")
            exp[side] = rate_box * box if (rate_box is not None and box is not None) else rate_tm
        p_list = pens.get(fid, [])
        obs = {"h": sum(1 for t, _, _ in p_list if t == m["h"]), "a": sum(1 for t, _, _ in p_list if t == m["a"])}

        a = agg[(m["ref"], m["lg"], m["se"])]
        a["m"] += 1
        a["P"] += len(p_list)
        a["EP"] += exp["h"] + exp["a"]
        a["PH"] += obs["h"]
        a["EPH"] += exp["h"]
        a["PA"] += obs["a"]
        a["EPA"] += exp["a"]

        fh, fa = stat[(fid, m["h"])].get("fouls"), stat[(fid, m["a"])].get("fouls")
        ch, sh = avg(m["h"], m["lg"], m["se"], "c"), avg(m["h"], m["lg"], m["se"], "s")
        ca, sa = avg(m["a"], m["lg"], m["se"], "c"), avg(m["a"], m["lg"], m["se"], "s")
        if None not in (fh, fa, ch, sh, ca, sa):
            a["F"] += max(0.0, fh + fa - len(p_list))          # fouls away from the penalty spot
            a["EF"] += (ch + sa) / 2 + (ca + sh) / 2
            a["fm"] += 1

        if fid in bands:
            own, mid, fin = bands[fid]
            a["b_own"] += own
            a["b_mid"] += mid
            a["b_fin"] += fin
            a["b_box"] += len(p_list)
            a["bm"] += 1

        for team, minute, scored in p_list:
            pen_list.append((m["ref"], m["lg"], m["se"], m["date"], m["h"], m["a"], minute,
                             0 if team == m["h"] else 1, scored))

    # ---- names
    refs, lgs, ses, teams = sorted({k[0] for k in agg}), sorted({k[1] for k in agg}), \
        sorted({k[2] for k in agg}), sorted({t for m in matches.values() for t in (m["h"], m["a"])})
    rid, lid, sid, tid = ({v: i for i, v in enumerate(x)} for x in (refs, lgs, ses, teams))
    one = lambda sql, ids: dict(q(sql, {"ids": list(ids)})) if ids else {}
    ref_names = one("SELECT referee_id, coalesce(display_name, common_name, name) FROM football.referees "
                    "WHERE referee_id = ANY(%(ids)s)", refs)
    team_names = one("SELECT team_id, name FROM football.teams WHERE team_id = ANY(%(ids)s)", teams)
    lg_names = {r[0]: ((r[2] + " · ") if r[2] else "") + (r[1] or "") for r in q(
        "SELECT l.league_id, l.name, c.name FROM football.leagues l LEFT JOIN core.countries c "
        "ON c.country_id = l.country_id WHERE l.league_id = ANY(%(ids)s)", {"ids": lgs})}
    se_names = one("SELECT season_id, name FROM football.seasons WHERE season_id = ANY(%(ids)s)", ses)
    se_order = one("SELECT season_id, coalesce(starting_at::text, name) FROM football.seasons "
                   "WHERE season_id = ANY(%(ids)s)", ses)

    cols = ["m", "P", "EP", "PH", "EPH", "PA", "EPA", "F", "EF", "fm", "b_own", "b_mid", "b_fin", "b_box", "bm"]
    rows = [[rid[r], lid[lg], sid[se]] + [round(a[c], 3) for c in cols] for (r, lg, se), a in agg.items()]
    data = {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "opportunity": box_stat,
                 "bands": bool(bands), "penaltyTypes": sorted({d for _, d in ptypes})},
        "cols": ["r", "l", "s"] + cols, "rows": rows,
        "refs": [ref_names.get(r, f"referee {r}") for r in refs],
        "leagues": [lg_names.get(x, f"league {x}") for x in lgs],
        "seasons": [se_names.get(x, f"season {x}") for x in ses],
        "seasonOrder": [str(se_order.get(x, "")) for x in ses],
        "teams": [team_names.get(t, f"team {t}") for t in teams],
        "pens": [[rid[r], lid[lg], sid[se], d, tid[h], tid[a], mi, side, sc]
                 for r, lg, se, d, h, a, mi, side, sc in pen_list],
    }
    path = os.path.abspath(REPORT_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"\nReport written: {path}  ({len(refs)} referees, {len(matches):,} matches)")
    print("Open it in your browser (on a Mac:  open penalty_eagerness.html)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Penalty Eagerness</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#677084;--line:#d9dde5;--grid:#eceff3;
--accent:#2d6cdf;--eager:#c0392b;--hes:#2c6db5;--neutral:#9aa3b2}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e6e9ef;--muted:#99a2b3;--line:#2c3340;--grid:#242a35;
--accent:#6aa0ff;--eager:#e0675a;--hes:#5b9bea;--neutral:#6f7888}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 20px 4px}h1{font-size:20px;margin:0 0 4px}header p{margin:0;color:var(--muted);max-width:900px}
.bar{display:flex;flex-wrap:wrap;gap:10px;padding:10px 20px;align-items:end;border-bottom:1px solid var(--line)}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:4px}
select,input[type=search]{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
input[type=range]{width:140px}
main{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:16px;padding:16px 20px 30px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h2{font-size:15px;margin:0 0 2px}.card .sub{color:var(--muted);font-size:12px;margin-bottom:10px}
svg{display:block;width:100%;height:auto}svg text{font-size:11px;fill:var(--ink)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:12px}
.stat{border:1px solid var(--line);border-radius:10px;padding:8px 10px}.stat b{display:block;font-size:18px}.stat span{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}
.note{color:var(--muted);font-size:12px}.row{cursor:pointer}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;display:none;z-index:5;box-shadow:0 4px 14px rgba(0,0,0,.15);max-width:300px}
</style></head><body>
<header><h1>Penalty Eagerness</h1>
<p>Is a referee quicker or slower to give a penalty than to give a foul elsewhere on the pitch? <b>Box</b> = penalties vs expected from shots inside the box. <b>Elsewhere</b> = other fouls vs expected from the two teams' usual foul counts. <b>Eagerness</b> = Box ÷ Elsewhere. Bars are 90% intervals; small samples are pulled towards average.</p></header>
<div class="bar">
 <label>Competition<select id="lg"></select></label>
 <label>Season<select id="se"></select></label>
 <label>Minimum matches <span><input type="range" id="minm" min="1" max="80" value="15"> <b id="minmv">15</b></span></label>
 <label>Sort by<select id="sort"><option value="eager">Eagerness</option><option value="box">Box</option><option value="else">Elsewhere</option><option value="m">Matches</option><option value="name">Name</option></select></label>
 <label>Find referee<input type="search" id="find" placeholder="Name…"></label>
</div>
<main>
 <section class="card"><h2>Eagerness to give penalties, relative to the rest of the pitch</h2>
  <div class="sub" id="forestSub"></div><div id="forest"></div>
  <div class="legend"><span><i class="dot" style="background:var(--eager)"></i>eager (interval above 1)</span><span><i class="dot" style="background:var(--hes)"></i>hesitant (interval below 1)</span><span><i class="dot" style="background:var(--neutral)"></i>not distinguishable from average</span></div></section>
 <div style="display:grid;gap:16px;align-content:start">
  <section class="card"><h2>Box vs elsewhere</h2><div class="sub">Each dot is a referee. Above the diagonal: relatively keener in the box than elsewhere.</div><div id="scatter"></div></section>
  <section class="card" id="detail"></section>
 </div>
</main>
<div id="tip"></div>
<script>
const D = __DATA__;
const IX = Object.fromEntries(D.cols.map((c,i)=>[c,i]));
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (n,d=2) => isFinite(n) ? (+n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}) : "–";
let selected = null, current = [];

// ------------------------------------------------------------ statistics
// Wilson-Hilferty quantile of a gamma(shape a, rate b)
function gq(a,b,z){const c=1/(9*a); return a*Math.pow(Math.max(0,1-c+z*Math.sqrt(c)),3)/b}
// empirical-Bayes gamma prior from the population: mean m, between-referee variance tau2
function prior(items){const So=items.reduce((s,i)=>s+i.O,0), Se=items.reduce((s,i)=>s+i.E,0);
  if(!Se) return null; const m=So/Se, live=items.filter(i=>i.E>0);
  let v=0; live.forEach(i=>{v+=i.E*(i.O/i.E-m)**2}); v/=Se;
  const tau2=Math.max(v-m*live.length/Se, m*m*0.0025);      // at least ~5% spread between referees
  return {m, a:m*m/tau2, b:m/tau2}}
function post(p,O,E){ if(!p) return null; const a=p.a+O, b=p.b+E;
  return {est:a/b/p.m, lo:gq(a,b,-1.645)/p.m, hi:gq(a,b,1.645)/p.m, a}}
function ratio(x,y){ if(!x||!y) return null; const est=x.est/y.est, se=Math.sqrt(1/x.a+1/y.a);
  return {est, lo:est*Math.exp(-1.645*se), hi:est*Math.exp(1.645*se)}}
const verdict = r => !r ? "neutral" : r.lo>1 ? "eager" : r.hi<1 ? "hesitant" : "neutral";
const colour = v => css(v=="eager"?"--eager":v=="hesitant"?"--hes":"--neutral");

// ------------------------------------------------------------ data per filter
function compute(){const lg=$("#lg").value, se=$("#se").value, by=new Map();
  for(const row of D.rows){ if(lg!="all"&&row[IX.l]!=+lg) continue; if(se!="all"&&D.seasons[row[IX.s]]!=se) continue;
    const r=row[IX.r]; if(!by.has(r)) by.set(r,Object.fromEntries(D.cols.slice(3).map(c=>[c,0])));
    const a=by.get(r); D.cols.slice(3).forEach(c=>a[c]+=row[IX[c]]) }
  const all=[...by.entries()].map(([r,a])=>({r,name:D.refs[r],...a}));
  const pBox=prior(all.map(x=>({O:x.P,E:x.EP}))), pElse=prior(all.filter(x=>x.fm>0).map(x=>({O:x.F,E:x.EF})));
  // pitch-band population shares
  const pop={own:0,mid:0,fin:0,box:0}; all.forEach(x=>{pop.own+=x.b_own;pop.mid+=x.b_mid;pop.fin+=x.b_fin;pop.box+=x.b_box});
  const popN=pop.own+pop.mid+pop.fin+pop.box;
  all.forEach(x=>{x.box=post(pBox,x.P,x.EP); x.else=x.fm>0?post(pElse,x.F,x.EF):null; x.eager=ratio(x.box,x.else);
    x.home=post(pBox,x.PH,x.EPH); x.away=post(pBox,x.PA,x.EPA); x.v=verdict(x.eager);
    const N=x.b_own+x.b_mid+x.b_fin+x.b_box, K=150;
    x.bandIdx=popN&&N?["own","mid","fin","box"].map(k=>{const p=pop[k]/popN; return ((x["b_"+k]+K*p)/(N+K))/p}):null;
    const q=pop.box/Math.max(1,pop.box+pop.fin), n2=x.b_box+x.b_fin, K2=40;
    x.boxPerFinal=popN&&n2?((x.b_box+K2*q)/(n2+K2))/q:null });
  return {all, pBox, pElse}}

// ------------------------------------------------------------ charts
const LOG = v => Math.log2(Math.max(1/8,Math.min(8,v)));
function axisX(x0,x1,ticks,X,y,h){return ticks.map(t=>`<line x1="${X(t)}" x2="${X(t)}" y1="${y}" y2="${y+h}" stroke="${css(t==1?"--muted":"--grid")}" stroke-width="${t==1?1:1}" ${t==1?'stroke-dasharray="3 3"':""}/>
  <text x="${X(t)}" y="${y+h+14}" text-anchor="middle" fill="${css("--muted")}">${t==0.5?"½":t==0.25?"¼":t==0.125?"⅛":t}×</text>`).join("")}

function forest(list){const W=640, rowH=20, top=8, left=170, right=60, H=top+list.length*rowH+28;
  if(!list.length){$("#forest").innerHTML="<p class='note'>No referees with this many matches.</p>";return}
  const lo=Math.min(0.5,...list.map(x=>x.eager?x.eager.lo:1)), hi=Math.max(2,...list.map(x=>x.eager?x.eager.hi:1));
  const a=LOG(Math.max(0.125,lo)), b=LOG(Math.min(8,hi)), X=v=>left+(LOG(v)-a)/(b-a)*(W-left-right);
  const ticks=[0.125,0.25,0.5,1,2,4,8].filter(t=>LOG(t)>=a-1e-9&&LOG(t)<=b+1e-9);
  let s=axisX(0,0,ticks,X,top,list.length*rowH);
  list.forEach((x,i)=>{const y=top+i*rowH+rowH/2, e=x.eager, c=colour(x.v), sel=selected===x.r;
    s+=`<g class="row" data-r="${x.r}" data-t="${x.name}: eagerness ${fmt(e?.est)} (90%: ${fmt(e?.lo)}–${fmt(e?.hi)}) · ${x.m} matches · ${x.P} penalties">
      <rect x="0" y="${y-rowH/2}" width="${W}" height="${rowH}" fill="${sel?css("--grid"):"transparent"}"/>
      <text x="${left-8}" y="${y+4}" text-anchor="end" font-weight="${sel?700:400}">${x.name.length>24?x.name.slice(0,23)+"…":x.name}</text>
      ${e?`<line x1="${X(e.lo)}" x2="${X(e.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="${X(e.est)}" cy="${y}" r="${3+Math.min(4,Math.sqrt(x.m)/3)}" fill="${c}" stroke="${css("--card")}" stroke-width="1.5"/>`:""}
      <text x="${W-right+8}" y="${y+4}" fill="${css("--muted")}">${fmt(e?.est)}</text></g>`});
  $("#forest").innerHTML=`<svg viewBox="0 0 ${W} ${H}">${s}</svg>`;
  document.querySelectorAll("#forest .row").forEach(g=>g.onclick=()=>{selected=+g.dataset.r;render()})}

function scatter(list){const W=420,H=360,pad=44; const pts=list.filter(x=>x.box&&x.else);
  if(!pts.length){$("#scatter").innerHTML="<p class='note'>Needs foul statistics.</p>";return}
  const vals=pts.flatMap(x=>[x.box.est,x.else.est]); const lo=Math.min(0.6,...vals), hi=Math.max(1/0.6,...vals);
  const a=LOG(lo)-0.1, b=LOG(hi)+0.1, X=v=>pad+(LOG(v)-a)/(b-a)*(W-pad-10), Y=v=>H-pad-(LOG(v)-a)/(b-a)*(H-pad-10);
  const ticks=[0.5,0.67,0.8,1,1.25,1.5,2].filter(t=>LOG(t)>=a&&LOG(t)<=b);
  let s=ticks.map(t=>`<line x1="${X(t)}" x2="${X(t)}" y1="10" y2="${H-pad}" stroke="${css("--grid")}"/><line x1="${pad}" x2="${W-10}" y1="${Y(t)}" y2="${Y(t)}" stroke="${css("--grid")}"/>
    <text x="${X(t)}" y="${H-pad+14}" text-anchor="middle" fill="${css("--muted")}">${t}</text><text x="${pad-6}" y="${Y(t)+4}" text-anchor="end" fill="${css("--muted")}">${t}</text>`).join("");
  s+=`<line x1="${X(Math.pow(2,a))}" y1="${Y(Math.pow(2,a))}" x2="${X(Math.pow(2,b))}" y2="${Y(Math.pow(2,b))}" stroke="${css("--muted")}" stroke-dasharray="4 4"/>
   <text x="${W-14}" y="22" text-anchor="end" fill="${css("--eager")}">eager in the box</text>
   <text x="${W-14}" y="${H-pad-8}" text-anchor="end" fill="${css("--hes")}">hesitant in the box</text>
   <text x="${(pad+W)/2}" y="${H-8}" text-anchor="middle" fill="${css("--muted")}">Elsewhere: fouls vs expected →</text>
   <text transform="translate(12 ${(H-pad)/2}) rotate(-90)" text-anchor="middle" fill="${css("--muted")}">Box: penalties vs expected →</text>`;
  pts.forEach(x=>{const sel=selected===x.r; s+=`<circle class="row" data-r="${x.r}" cx="${X(x.else.est)}" cy="${Y(x.box.est)}" r="${sel?7:3+Math.min(4,Math.sqrt(x.m)/3)}"
    fill="${colour(x.v)}" fill-opacity="${sel?1:.75}" stroke="${sel?css("--ink"):css("--card")}" stroke-width="${sel?2:1}"
    data-t="${x.name}: box ${fmt(x.box.est)} · elsewhere ${fmt(x.else.est)} · eagerness ${fmt(x.eager?.est)}"/>`});
  $("#scatter").innerHTML=`<svg viewBox="0 0 ${W} ${H}">${s}</svg>`;
  document.querySelectorAll("#scatter .row").forEach(g=>g.onclick=()=>{selected=+g.dataset.r;render()})}

function detail(x, ctx){const el=$("#detail"); if(!x){el.innerHTML="<p class='note'>Click a referee to see the details.</p>";return}
  const v=x.v, tag=`<span class="tag" style="background:${colour(v)};color:#fff">${v=="eager"?"Eager in the box":v=="hesitant"?"Hesitant in the box":"No clear tendency"}</span>`;
  let h=`<h2>${x.name} ${tag}</h2><div class="sub">${x.m} matches · ${D.leagues[+$("#lg").value]||"all competitions"} · ${$("#se").value=="all"?"all seasons":$("#se").value}</div>
  <div class="stats">
   <div class="stat"><b>${fmt(x.eager?.est)}</b><span>eagerness (90%: ${fmt(x.eager?.lo)}–${fmt(x.eager?.hi)})</span></div>
   <div class="stat"><b>${x.P} / ${fmt(x.EP,1)}</b><span>penalties given / expected → box ${fmt(x.box?.est)}</span></div>
   <div class="stat"><b>${fmt(x.F/Math.max(1,x.fm),1)} / ${fmt(x.EF/Math.max(1,x.fm),1)}</b><span>fouls per match given / expected → elsewhere ${fmt(x.else?.est)}</span></div>
   <div class="stat"><b>${fmt(x.home?.est)} · ${fmt(x.away?.est)}</b><span>box index: home · away team (${x.PH}/${fmt(x.EPH,1)} · ${x.PA}/${fmt(x.EPA,1)})</span></div></div>`;
  if(D.meta.bands&&x.bandIdx){const labels=["Own third","Middle third","Final third","Penalty box"], W=420, H=150, pad=36, bw=(W-pad-10)/4;
    const vals=x.bandIdx, mx=Math.max(2,...vals), Y=v=>H-24-(LOG(v)-LOG(1/mx))/(LOG(mx)-LOG(1/mx))*(H-34);
    let s=`<line x1="${pad}" x2="${W-10}" y1="${Y(1)}" y2="${Y(1)}" stroke="${css("--muted")}" stroke-dasharray="3 3"/><text x="${pad-6}" y="${Y(1)+4}" text-anchor="end" fill="${css("--muted")}">1×</text>`;
    vals.forEach((val,i)=>{const x0=pad+i*bw+8, y0=Y(1), y1=Y(val), c=val>=1?css("--eager"):css("--hes");
      s+=`<rect x="${x0}" y="${Math.min(y0,y1)}" width="${bw-16}" height="${Math.abs(y1-y0)||1}" fill="${c}" fill-opacity="${i==3?1:.6}" rx="3" data-t="${labels[i]}: ${fmt(val)}× the league share"/>
       <text x="${x0+(bw-16)/2}" y="${H-8}" text-anchor="middle" fill="${css("--muted")}">${labels[i]}</text>
       <text x="${x0+(bw-16)/2}" y="${(val>=1?y1-4:y1+12)}" text-anchor="middle">${fmt(val)}×</text>`});
    h+=`<h2 style="margin-top:6px">Where on the pitch he gives fouls</h2><div class="sub">Share of fouls in each band (from the attacking team's side) vs the league, ${x.bm} matches with located fouls. Penalties per final-third foul: ${fmt(x.boxPerFinal)}× the league.</div><svg viewBox="0 0 ${W} ${H}">${s}</svg>`}
  const pens=D.pens.filter(p=>p[0]==x.r&&($("#lg").value=="all"||p[1]==+$("#lg").value)&&($("#se").value=="all"||D.seasons[p[2]]==$("#se").value)).sort((a,b)=>b[3].localeCompare(a[3]));
  h+=`<h2 style="margin-top:12px">Penalties given (${pens.length})</h2><div style="max-height:300px;overflow:auto;border:1px solid var(--line);border-radius:8px"><table><tr><th>Date</th><th>Match</th><th>Min</th><th>To</th><th></th></tr>`+
    pens.slice(0,500).map(p=>`<tr><td>${p[3]||""}</td><td>${D.teams[p[4]]} – ${D.teams[p[5]]}</td><td>${p[6]??""}'</td><td>${p[7]?"away":"home"}</td><td>${p[8]?"scored":"missed"}</td></tr>`).join("")+
    `</table></div>`;
  el.innerHTML=h}

function render(){const ctx=compute(); const minm=+$("#minm").value, find=$("#find").value.trim().toLowerCase();
  $("#minmv").textContent=minm;
  let list=ctx.all.filter(x=>x.m>=minm&&(!find||x.name.toLowerCase().includes(find)));
  const key=$("#sort").value, val=x=>key=="eager"?(x.eager?.est??0):key=="box"?(x.box?.est??0):key=="else"?(x.else?.est??0):x.m;
  list.sort(key=="name"?(a,b)=>a.name.localeCompare(b.name):(a,b)=>val(b)-val(a));
  current=list;
  const tP=ctx.all.reduce((s,x)=>s+x.P,0), tM=ctx.all.reduce((s,x)=>s+x.m,0);
  $("#forestSub").textContent=`${list.length} referees with ≥${minm} matches · ${fmt(tP,0)} penalties in ${fmt(tM,0)} matches (${fmt(tP/Math.max(1,tM),2)} per match) · opportunity measured by ${D.meta.opportunity||"matches played"}`;
  forest(list); scatter(list);
  if(selected===null||!ctx.all.some(x=>x.r===selected)) selected=list[0]?.r??null;
  detail(ctx.all.find(x=>x.r===selected), ctx)}

function init(){const counts=new Map(); D.rows.forEach(r=>counts.set(r[IX.l],(counts.get(r[IX.l])||0)+r[IX.m]));
  $("#lg").innerHTML=`<option value="all">All competitions</option>`+[...counts.entries()].sort((a,b)=>b[1]-a[1]).map(([l,n])=>`<option value="${l}">${D.leagues[l]} (${fmt(n,0)} matches)</option>`).join("");
  fillSeasons()}
function fillSeasons(){const lg=$("#lg").value, names=new Map();
  D.rows.forEach(r=>{ if(lg=="all"||r[IX.l]==+lg) names.set(D.seasons[r[IX.s]],D.seasonOrder[r[IX.s]])});
  const prev=$("#se").value; $("#se").innerHTML=`<option value="all">All seasons</option>`+[...names.entries()].sort((a,b)=>b[1].localeCompare(a[1])).map(([n])=>`<option>${n}</option>`).join("");
  if([...$("#se").options].some(o=>o.value==prev)) $("#se").value=prev}
document.addEventListener("mousemove",ev=>{const t=ev.target.closest&&ev.target.closest("[data-t]"),tip=$("#tip");
  if(t){tip.textContent=t.getAttribute("data-t");tip.style.display="block";tip.style.left=Math.min(ev.clientX+12,innerWidth-310)+"px";tip.style.top=(ev.clientY+12)+"px"}else tip.style.display="none"});
$("#lg").onchange=()=>{fillSeasons();render()}; ["#se","#sort"].forEach(s=>$(s).onchange=render); ["#minm","#find"].forEach(s=>$(s).oninput=render);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
init(); render();
</script></body></html>"""


if __name__ == "__main__":
    main()
