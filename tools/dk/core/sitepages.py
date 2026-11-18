"""dk site, part 2: HTML for every page. Pure functions of the gathered data, so a build is deterministic."""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
from pathlib import Path

from . import sitemd

E = html.escape

NAV = [("index.html", "Status"), ("timeline.html", "Timeline"), ("code/index.html", "Source"), ("tours/index.html", "Tours"),
       ("ideas.html", "Roadmap"), ("failures.html", "Lessons"), ("future.html", "Next"), ("library/index.html", "Library")]

CLASS_WHY = {
    "near-register": "Right instructions; the compiler picked other register numbers.",
    "near-scheduling": "Right instructions in a different order.",
    "near-shape": "80% or more alike, but the code takes another shape somewhere.",
    "unit-data": "The code is right; only data addresses differ until the unit's data is written.",
    "sinit": "Static initialisation (__sinit) still differs.",
    "compiler-version": "Probably built with another compiler version or flags.",
    "far": "Under 80% alike: needs a real rewrite.",
    "fresh": "Written but never worked on by a matcher yet.",
    "giant": "Very large functions (several KB) that need to be cut into pieces first.",
}
WORKER_WHY = {
    "cpu": "deterministic search on this PC (dk close)",
    "lora": "the local fine-tuned model",
    "claude": "a Claude agent (the hard ones)",
    "blocked": "waiting on something else first",
    "frozen": "held out as a test set for the models; nobody touches them",
}

BRAND = "decomp"            # the project's name ([site] name): set by site.gather before any page renders


def set_brand(name: str) -> None:
    global BRAND
    BRAND = name or "decomp"


BRAND_SVG = """<svg viewBox="0 0 40 40" aria-hidden="true"><defs><linearGradient id="bg1" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#7d5cff"/><stop offset="1" stop-color="#2a168a"/></linearGradient></defs>
<rect x="2" y="2" width="36" height="36" rx="11" fill="url(#bg1)"/><path d="M8 27 A13 13 0 0 1 32 27" fill="none" stroke="#38f2c4" stroke-width="3.2" stroke-linecap="round"/>
<path d="M20 27 L28.5 15.5" stroke="#ffb547" stroke-width="3" stroke-linecap="round"/><circle cx="20" cy="27" r="3" fill="#fff"/>
<rect x="5" y="4" width="30" height="12" rx="6" fill="#fff" opacity=".13"/></svg>"""

HERO_ART = """<svg class="art" viewBox="0 0 1200 300" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<defs><linearGradient id="road" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#38f2c4" stop-opacity=".0"/><stop offset="1" stop-color="#38f2c4" stop-opacity=".35"/></linearGradient>
<radialGradient id="glow" cx=".78" cy=".55" r=".5"><stop offset="0" stop-color="#ffb547" stop-opacity=".55"/><stop offset="1" stop-color="#ffb547" stop-opacity="0"/></radialGradient></defs>
<rect width="1200" height="300" fill="url(#glow)"/>
<g stroke="#9f8cff" stroke-opacity=".28" fill="none">
<path d="M940 170 L600 300"/><path d="M940 170 L760 300"/><path d="M940 170 L900 300"/><path d="M940 170 L1040 300"/><path d="M940 170 L1200 290"/>
<path d="M700 230 H1200"/><path d="M780 200 H1200"/><path d="M850 184 H1200"/><path d="M600 268 H1200"/></g>
<path d="M940 170 L820 300 L1060 300 Z" fill="url(#road)"/>
<path d="M940 170 L935 300" stroke="#b8f53c" stroke-width="3" stroke-dasharray="14 12" stroke-opacity=".7"/>
<path d="M-40 250 C 260 150, 520 120, 1240 60" stroke="#38f2c4" stroke-opacity=".35" stroke-width="2" fill="none"/>
<path d="M-40 275 C 300 180, 560 150, 1240 95" stroke="#a98bff" stroke-opacity=".35" stroke-width="10" fill="none"/>
<path d="M-40 290 C 320 210, 600 185, 1240 130" stroke="#ffffff" stroke-opacity=".08" stroke-width="26" fill="none"/>
<circle cx="940" cy="170" r="5" fill="#fff" opacity=".8"/></svg>"""

TOUR_ICONS = {
    "world": '<path d="M8 40 L24 14 L40 40 Z" fill="none" stroke="currentColor" stroke-width="3"/><path d="M24 14 V40" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>',
    "camera": '<rect x="7" y="16" width="26" height="18" rx="4" fill="none" stroke="currentColor" stroke-width="3"/><path d="M33 22 L42 17 V33 L33 28" fill="none" stroke="currentColor" stroke-width="3"/>',
    "physics": '<circle cx="24" cy="24" r="15" fill="none" stroke="currentColor" stroke-width="3"/><circle cx="24" cy="24" r="4" fill="currentColor"/><path d="M24 9 V18 M24 30 V39 M9 24 H18 M30 24 H39" stroke="currentColor" stroke-width="3"/>',
    "renderer": '<path d="M24 7 L41 37 H7 Z" fill="none" stroke="currentColor" stroke-width="3"/><path d="M24 7 L24 37 M15.5 22 L32.5 22" stroke="currentColor" stroke-width="2"/>',
    "ai": '<rect x="9" y="18" width="30" height="14" rx="5" fill="none" stroke="currentColor" stroke-width="3"/><path d="M16 18 L20 10 H28 L32 18" fill="none" stroke="currentColor" stroke-width="3"/><circle cx="17" cy="34" r="3" fill="currentColor"/><circle cx="31" cy="34" r="3" fill="currentColor"/>',
    "sound": '<path d="M9 19 H16 L26 10 V38 L16 29 H9 Z" fill="none" stroke="currentColor" stroke-width="3"/><path d="M31 17 Q36 24 31 31 M35 12 Q43 24 35 36" fill="none" stroke="currentColor" stroke-width="3"/>',
}


def rel(href: str, here: str) -> str:
    """Site-relative `href` as seen from the page at site path `here`."""
    if href.startswith(("#", "http://", "https://")):
        return href
    path, _, frag = href.partition("#")
    r = os.path.relpath(path, os.path.dirname(here) or ".")
    return r + (f"#{frag}" if frag else "")


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}"


def kb(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.2f} MB"


def day(t: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t))


def stamp(t: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def meter(parts: list[tuple[str, float]], cls: str = "") -> str:
    spans = "".join(f'<span class="{c}" style="width:{max(0.0, min(100.0, v)):.2f}%"></span>' for c, v in parts if v > 0)
    return f'<div class="meter {cls}">{spans}</div>'


def chip(cls: str, text: str | None = None) -> str:
    return f'<span class="chip {E(cls)}">{E(text or cls)}</span>'


def page(here: str, title: str, body: str, active: str | None, hero: str = "", foot: str = "") -> str:
    nav = "".join(f'<a href="{rel(h, here)}"{" class=on" if h == active else ""}>{E(t)}</a>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title + " · " + BRAND)}</title><meta name="color-scheme" content="dark light">
<script>(function(){{var t=(location.search.match(/[?&]theme=(dark|light)/)||[])[1]||null;if(!t){{try{{t=localStorage.getItem("dk-theme")}}catch(e){{}}}}if(!t)t=matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";document.documentElement.setAttribute("data-theme",t)}})();</script>
<link rel="stylesheet" href="{rel('assets/site.css', here)}"><link rel="icon" href="{rel('assets/icon.svg', here)}">
</head><body>
<header class="topbar"><div class="wrap"><a class="brand" href="{rel('index.html', here)}">{BRAND_SVG}<span class="bt">{E(BRAND)}<small>STATUS BOARD</small></span></a>
<nav class="menu">{nav}</nav><button class="themebtn" id="themebtn" title="Light or dark">&#9728;</button></div></header>
{hero}<main class="wrap">{body}</main>
<footer class="footer wrap">{foot}Local only: this site shows the game's source and is never published.</footer>
<script src="{rel('assets/site.js', here)}"></script></body></html>
"""


def small_hero(kicker: str, title: str, lead: str = "", crumbs: str = "") -> str:
    return (f'<section class="hero small">{HERO_ART}<div class="wrap">'
            f'{f"<div class=crumbs>{crumbs}</div>" if crumbs else ""}<div class="kicker">{E(kicker)}</div><h1>{title}</h1>'
            f'{f"<p class=lead>{lead}</p>" if lead else ""}</div></section>')


def stat(label: str, num: str, unit: str, sub: str, cls: str = "", bar: str = "") -> str:
    return (f'<div class="stat {cls}"><div class="label">{E(label)}</div><div class="num">{num}<small>{E(unit)}</small></div>'
            f'<div class="sub">{sub}</div>{bar}</div>')


# ---- status --------------------------------------------------------------------------------------------

def status_page(d: dict) -> str:
    here = "index.html"
    m = d["measures"]
    tc, td_ = m["total_code"], m["total_data"]
    matched = pct(m["matched_code"], tc)
    linked = pct(m["complete_code"], tc)
    data = pct(m["matched_data"], td_)
    fnp = pct(m["matched_functions"], m["total_functions"])
    hero = f"""<section class="hero">{HERO_ART}<div class="wrap">
<div class="kicker">GameCube &middot; matching decompilation &middot; {E(d['game_id'])}</div>
<h1>{E(d['opts'].get('headline', ''))}</h1>
<p class="lead">Every function counted here is C++ that compiles to exactly the bytes of the original game.
{m['matched_functions']:,} of {m['total_functions']:,} functions match, and {fmt_pct(linked)}% of the code is already linked into a rebuilt game that is identical to the original.</p>
<div class="bigstats">
{stat('Code matched', fmt_pct(matched, 2), '%', f"{kb(m['matched_code'])} of {kb(tc)}", 'exact', meter([('x', matched)]))}
{stat('Code linked', fmt_pct(linked, 2), '%', 'in units that build the real game', 'lime', meter([('l', linked)]))}
{stat('Functions exact', f"{m['matched_functions']:,}", f"/{m['total_functions']:,}", f"{fmt_pct(fnp)}% of all functions", '', meter([('x', fnp)]))}
{stat('Units linked', f"{m['complete_units']}", f"/{m['total_units']}", 'source files done end to end', 'exact', meter([('x', pct(m['complete_units'], m['total_units']))]))}
{stat('Data matched', fmt_pct(data, 1), '%', f"{kb(m['matched_data'])} of {kb(td_)}", 'amber', meter([('n', data)]))}
</div></div></section>"""
    out = []
    if d.get("stale"):
        out.append(f'<div class="panel warn"><b>Numbers may be a little old.</b> {E(d["stale"])}. Run <code>./dk build</code>, then the site rebuilds itself.</div>')

    # categories
    rows = "".join(
        f"<tr><td><b>{E(c['id'])}</b> <span class=muted>{E(c.get('name', ''))}</span></td>"
        f"<td class=num>{fmt_pct(pct(c['matched_code'], c['total_code']), 2)}%</td><td class=num>{fmt_pct(pct(c['complete_code'], c['total_code']), 2)}%</td>"
        f"<td class=num>{fmt_pct(pct(c['matched_data'], c['total_data']), 1)}%</td><td class=num>{c['matched_functions']:,}/{c['total_functions']:,}</td>"
        f"<td class=num>{c['complete_units']}/{c['total_units']}</td><td>{meter([('x', pct(c['complete_code'], c['total_code'])), ('n', pct(c['matched_code'] - c['complete_code'], c['total_code']))])}</td></tr>"
        for c in d["categories"])
    out.append(f"""<div class="panel"><div class="panel-title"><h2>The game code and the system libraries</h2>
<span class="hint">{E(d['opts'].get('categories_hint', ''))}</span></div>
<div class="tablewrap"><table><thead><tr><th>Part</th><th class=num>Matched</th><th class=num>Linked</th><th class=num>Data</th><th class=num>Functions</th><th class=num>Units</th><th>Linked + matched</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="legend" style="margin-top:8px"><span><i style="background:var(--exact)"></i>linked</span><span><i style="background:var(--amber)"></i>matched but not linked yet</span></div></div>""")

    # systems
    cards = []
    for s in d["systems"]:
        ex = pct(s["exact"], s["code"])
        nr = pct(s["near"], s["code"])
        tour = d["tour_for_system"].get(s["name"])
        cards.append(f"""<a class="panel syscard" href="code/index.html?sys={E(s['name'])}"><div class="top"><span class="name">{E(s['name'])}</span><span class="pct">{ex:.0f}%</span></div>
{meter([('x', ex), ('n', nr)])}<div class="facts"><span><b>{s['units']}</b> units</span><span><b>{s['linked']}</b> linked</span><span><b>{s['open']}</b> open</span><span><b>{s['unwritten']}</b> unwritten</span><span><b>{kb(s['code'])}</b></span></div>
{f'<div class="facts"><span>Guided tour: <b>{E(tour)}</b></span></div>' if tour else ''}</a>""")
    out.append(f"""<h2>Progress per game system</h2><p class="muted">Share of each system's code that matches exactly (green) or is a near miss at 80% or more (amber). A function can count in two systems. Click a card to see its units.</p>
<div class="grid g3">{''.join(cards)}</div>""")

    # open by kind
    r = d.get("route") or {}
    if r.get("classes"):
        total_b = max(1, r.get("bytes", 1))
        rows = []
        for k, v in sorted(r["classes"].items(), key=lambda kv: -kv[1]["bytes"]):
            rows.append(f"""<div class="row"><b>{E(k)}</b>{meter([('n', pct(v['bytes'], total_b))])}<span class="v">{v['functions']} fn &middot; {kb(v['bytes'])}</span>
<div class="why">{E(CLASS_WHY.get(k, ''))}</div></div>""")
        wrows = []
        for k, v in sorted((r.get("workers") or {}).items(), key=lambda kv: -kv[1]["bytes"]):
            wrows.append(f"""<div class="row"><b>{E(k)}</b>{meter([('x', pct(v['bytes'], total_b))])}<span class="v">{v['functions']} fn &middot; {kb(v['bytes'])}</span>
<div class="why">{E(WORKER_WHY.get(k, ''))}</div></div>""")
        out.append(f"""<div class="grid g2" style="margin-top:22px"><div class="panel"><div class="panel-title"><h2>What is still open, by kind</h2><span class="hint">{r.get('functions', 0)} functions, {kb(r.get('bytes', 0))}</span></div>
<div class="bars">{''.join(rows)}</div></div>
<div class="panel"><div class="panel-title"><h2>Who works on it</h2><span class="hint">from <code>./dk route</code></span></div><div class="bars">{''.join(wrows)}</div>
{_frozen_block(d)}</div></div>""")

    # closest to linking
    near = d["one_from_link"]
    if near:
        rows = "".join(
            f"<tr><td class=unitname><a href=\"code/{E(u['slug'])}.html\">{E(u['short'])}</a></td><td class=fnname><a href=\"code/{E(u['slug'])}.html{('#L' + str(u['line'])) if u['line'] else ''}\">{E(u['fn'])}</a></td>"
            f"<td class=num>{u['pct']:.2f}%</td><td>{chip(u['status'])} <span class=muted>{E(u['cls'])}</span></td><td class=num>{kb(u['code'])}</td></tr>"
            for u in near)
        out.append(f"""<div class="panel"><div class="panel-title"><h2>{len(near)} units are one function from done</h2><span class="hint">Finish the one open function (and the unit's data) and the whole file links. Biggest first.</span></div>
<div class="tablewrap"><table><thead><tr><th>Unit</th><th>Last open function</th><th class=num>Match</th><th>Status</th><th class=num>Unit code</th></tr></thead><tbody>{rows}</tbody></table></div></div>""")

    # recent
    rows = "".join(f"<tr><td class=mono>{E(stamp(c['t']))}</td><td class=mono>{E(c['rev'])}</td><td>{E(c['subject'][:160])}</td></tr>" for c in d["recent"])
    out.append(f"""<div class="panel"><div class="panel-title"><h2>Latest landings</h2><span class="hint">the last {len(d['recent'])} commits</span></div>
<div class="tablewrap"><table><tbody>{rows}</tbody></table></div></div>""")
    return page(here, "Status", "".join(out), "index.html", hero, _foot(d))


def _frozen_block(d: dict) -> str:
    fr = d.get("frozen") or {}
    rel_ = d.get("released") or {}
    recent = sorted(rel_.items(), key=lambda kv: (kv[1].get("date", ""), kv[0]), reverse=True)[:5]
    items = "".join(f"<li><code>{E(k)}</code> <span class=muted>{E(v.get('date', ''))}: {E(v.get('reason', ''))}</span></li>" for k, v in recent)
    sets = ", ".join(f"{E(k)} {v}" for k, v in sorted(fr.items(), key=lambda kv: -kv[1]))
    return f"""<h3>Frozen and released</h3><p class="muted"><b style="color:var(--ink)">{sum(fr.values())}</b> open functions are frozen as test sets for the models ({sets or 'none'}).
<b style="color:var(--ink)">{len(rel_)}</b> functions have been released from the sets so far.</p>{f'<ul class="muted" style="font-size:.85rem">{items}</ul>' if items else ''}"""


def _foot(d: dict) -> str:
    h = d["head"]
    return f'Built from commit <code>{E(h["rev"])}</code> ({E(stamp(h["t"]))}) with <code>./dk site build</code>. '


# ---- timeline ------------------------------------------------------------------------------------------

def timeline_page(d: dict) -> str:
    here = "timeline.html"
    hist, events = d["history"], d["events"]
    W, H, L, R, T, B = 1000, 380, 46, 46, 18, 64
    body = []
    if hist:
        t0, t1 = hist[0]["t"], max(hist[-1]["t"], d["head"]["t"])
        tc, tf, tu = d["measures"]["total_code"], d["measures"]["total_functions"], d["measures"]["total_units"]
        span = max(1, t1 - t0)

        def X(t): return L + (W - L - R) * (t - t0) / span
        def Y(p): return T + (H - T - B) * (1 - p / 100.0)
        lk = [(X(h["t"]), Y(pct(h["complete_code"], tc))) for h in hist]
        mf = [(X(h["t"]), Y(pct(h["matched_functions"], tf))) for h in hist]
        un = [(X(h["t"]), Y(pct(h["complete_units"], tu))) for h in hist]

        def path(pts): return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        area = path(lk) + f" L{lk[-1][0]:.1f},{Y(0):.1f} L{lk[0][0]:.1f},{Y(0):.1f} Z"
        grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{Y(p):.1f}" y2="{Y(p):.1f}"/><text x="{L - 8}" y="{Y(p) + 4:.1f}" text-anchor="end">{p}%</text>' for p in range(0, 101, 20))
        days, t = [], t0 - (t0 % 86400) + 86400
        while t <= t1:
            days.append(t)
            t += 86400
        step = max(1, len(days) // 8)
        ticks = "".join(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H - B}" opacity=".5"/><text x="{X(t):.1f}" y="{H - B + 16}" text-anchor="middle">{time.strftime("%b %d", time.localtime(t))}</text>'
                        for t in days[::step])
        evs = []
        colors = {"start": "var(--exact)", "link": "var(--exact)", "tool": "var(--violet)", "run": "var(--sky)", "overnight": "var(--sky)",
                  "lora": "var(--amber)", "burn": "var(--amber)", "decision": "var(--lime)", "setback": "var(--red)"}
        for i, e in enumerate(events):
            if e["t"] < t0 - 86400 or e["t"] > t1 + 86400:
                continue
            x = X(min(max(e["t"], t0), t1))
            row = i % 3
            evs.append(f'<line class="ev" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{H - B + 24 + row * 10}"/>'
                       f'<circle class="evdot" cx="{x:.1f}" cy="{H - B + 26 + row * 10}" r="5.5" fill="{colors.get(e["kind"], "var(--muted)")}" data-id="ev{i}" '
                       f'data-tip="{E("<b>" + E(e["date"]) + "</b><br>" + E(e["title"]))}"/>')
        # thinned hover data: [x, y, label, linked, fn%, fn, units, subject]
        keep = max(1, len(hist) // 500)
        pts = [[round(X(h["t"]), 1), 0, stamp(h["t"]), round(pct(h["complete_code"], tc), 2), round(pct(h["matched_functions"], tf), 2),
                f"{h['matched_functions']:,} fn", h["complete_units"], h["subject"][:90]] for h in hist[::keep]] + \
              [[round(X(hist[-1]["t"]), 1), 0, stamp(hist[-1]["t"]), round(pct(hist[-1]["complete_code"], tc), 2),
                round(pct(hist[-1]["matched_functions"], tf), 2), f"{hist[-1]['matched_functions']:,} fn", hist[-1]["complete_units"], hist[-1]["subject"][:90]]]
        svg = f"""<svg class="chart" id="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Linked code and exact functions over time">
<g class="grid">{grid}{ticks}</g><path class="linked-area" d="{area}"/><path class="units" d="{path(un)}"/><path class="matched" d="{path(mf)}"/><path class="linked" d="{path(lk)}"/>
<line id="cross" class="cross" x1="0" x2="0" y1="{T}" y2="{H - B}" style="display:none"/>{''.join(evs)}</svg>"""
        first, last = hist[0], hist[-1]
        days_n = max(1, (last["t"] - first["t"]) / 86400)
        body.append(f"""<div class="panel"><div class="panel-title"><h2>Progress over time</h2>
<div class="legend"><span><i style="background:var(--exact)"></i>linked code %</span><span><i style="background:var(--amber)"></i>exact functions %</span><span><i style="background:var(--violet)"></i>linked units %</span><span>dots: milestones (hover)</span></div></div>
<div class="chartbox">{svg}<div class="tip" id="tip"></div></div>
<script type="application/json" id="chartdata">{json.dumps({"w": W, "p": pts}).replace("</", "<\\/")}</script>
<p class="muted" style="margin:6px 0 0">From {E(day(first['t']))} to {E(day(last['t']))}: {last['matched_functions'] - first['matched_functions']:+,} exact functions
({(last['matched_functions'] - first['matched_functions']) / days_n:.0f} a day) and {last['complete_units'] - first['complete_units']:+} linked units, read from the progress floor that every landing raises
({len(hist):,} landings). The floor can only go up, so the lines never fall.</p></div>""")
    if d["per_day"]:
        mx = max(n for _, n in d["per_day"]) or 1
        bars = "".join(f'<span style="height:{100 * n / mx:.1f}%" title="{E(k)}: {n} commits"></span>' for k, n in d["per_day"])
        body.append(f"""<div class="panel"><div class="panel-title"><h2>Commits per day</h2><span class="hint">{sum(n for _, n in d['per_day']):,} commits since {E(d['per_day'][0][0])}; busiest day {max(n for _, n in d['per_day'])}</span></div>
<div class="daybars">{bars}</div></div>""")
    items = []
    for i, e in enumerate(events):
        items.append(f"""<li class="k-{E(e['kind'])}" id="ev{i}"><span class="when">{E(e['date'])}</span>{chip('k-' + e['kind'], e['kind'])}
<div class="t">{E(e['title'])}</div>{f'<div class="d">{sitemd.inline(e["detail"])}</div>' if e.get('detail') else ''}</li>""")
    body.append(f"""<div class="panel"><div class="panel-title"><h2>Milestones</h2><span class="hint">Hand-written in <code>{E(d['content_rel'])}/milestones.toml</code>; "auto" ones come from the floor.</span></div>
<ul class="events">{''.join(items) or '<li>No milestones yet.</li>'}</ul></div>""")
    return page(here, "Timeline", "".join(body), "timeline.html",
                small_hero("Progress timeline", "How we got here", "Every exact function and every linked unit, from the first commit to now."), _foot(d))


# ---- source browser -------------------------------------------------------------------------------------

def code_index(d: dict) -> str:
    here = "code/index.html"
    rows = []
    for u in d["units"]:
        st = u.status
        open_n = len(u.open_functions)
        keys = " ".join([u.short.lower(), (u.source or "").lower(), " ".join(c.lower() for c, _ in u.classes[:6])])
        bar = meter([("x", u.code_pct)], "thin") if u.functions else ""
        rows.append(f"""<tr data-k="{E(keys)}" data-st="{st}" data-sys="{E(' '.join(u.systems))}" data-open="{open_n}">
<td class="unitname"><a href="{E(u.slug)}.html">{E(u.short)}</a></td><td>{''.join(chip('sys', s) + ' ' for s in u.systems[:2])}</td>
<td>{chip(st, {"wip": f"{u.code_pct:.0f}%", "exact": "exact, not linked", "data": "data only"}.get(st, st))}</td><td class="num">{u.exact_count}/{len(u.functions)}{bar}</td><td class="num">{kb(u.total_code)}</td></tr>""")
    syss = "".join(f'<button data-f="sys:{E(s)}">{E(s)}</button>' for s in d["system_names"])
    body = f"""<div class="panel"><div class="toolbar"><input id="q" class="search" type="search" placeholder="Find a unit, class or function (3+ letters searches functions too)" autocomplete="off">
<span class="count"><b id="shown">{len(d['units'])}</b> of {len(d['units'])} units</span></div>
<div class="filters"><button data-f="all" class="on">All</button><button data-f="linked">Linked</button><button data-f="wip">In progress</button><button data-f="exact">Exact, not linked</button><button data-f="data">Data only</button>{syss}</div>
<div id="fnresults" class="fnsearch tablewrap"></div></div>
<div class="panel"><div class="tablewrap"><table id="units" class="units-table"><thead><tr><th>Unit</th><th>System</th><th>Status</th><th class=num>Functions exact</th><th class=num>Code</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>"""
    return page(here, "Source", body, "code/index.html",
                small_hero("Source browser", "Every unit, every function",
                           "A unit is one source file the compiler builds on its own. Linked units are finished; the rest show which functions are exact, close or not written yet."), _foot(d))


def functions_json(d: dict) -> str:
    rows = []
    for u in d["units"]:
        for f in u.functions:
            cls = (f.route or {}).get("class") or ""
            rows.append([f.name, f.demangled, u.short, u.slug, f.size, f.status, f.line or 0, f"{f.pct:.2f}% {cls}".strip()])
    return json.dumps(rows, separators=(",", ":"))


def unit_page(d: dict, u, text: str | None, header: tuple[str, str] | None, mentions: list[dict], tours: list[tuple[str, str]]) -> str:
    here = f"code/{u.slug}.html"
    st = u.status
    n, ex = len(u.functions), u.exact_count
    opens = u.open_functions
    near = sum(1 for f in opens if f.status == "near")
    unwritten = sum(1 for f in opens if f.status == "unwritten")
    frozen = sum(1 for f in opens if f.status == "frozen")
    sysname = ", ".join(u.systems) if u.systems else "no named system yet"
    classes = u.classes
    cls_txt = ", ".join(f"<code>{E(c)}</code>" for c, _ in classes[:8])
    what = [f"<p><b>What it is.</b> One source file of the game{' (' + E(sysname) + ')' if u.systems else ''}, compiled on its own into "
            f"{n} function{'s' if n != 1 else ''} and {kb(u.total_code)} of code"
            + (f", mostly for the class{'es' if len(classes) > 1 else ''} {cls_txt}." if classes else ".") + "</p>"]
    if st == "linked":
        what.append("<p><b>Where it stands.</b> Linked: every function and every byte of its data match, and the file is part of the rebuilt game.</p>")
    elif st == "data":
        what.append("<p><b>Where it stands.</b> This unit holds data only (no functions).</p>")
    else:
        parts = [f"{ex} of {n} functions are exact"]
        if opens:
            bits = [f"{near} near miss{'es' if near != 1 else ''}" if near else "", f"{unwritten} not written yet" if unwritten else "",
                    f"{frozen} frozen as a model test" if frozen else ""]
            far = len(opens) - near - unwritten - frozen
            if far:
                bits.append(f"{far} under 80%")
            parts.append(f"{len(opens)} open ({', '.join(b for b in bits if b)})")
        what.append(f"<p><b>Where it stands.</b> {'; '.join(parts)}. Code matched: {u.code_pct:.2f}%.</p>")
    hc = header_text = ""
    if text:
        from .sitedata import header_comment
        header_text = header_comment(text)
    if header_text:
        hc = f'<div class="mention"><div class="src">comment at the top of the file</div><div class="body">{E(header_text)}</div></div>'
    tour_html = ""
    if tours:
        tour_html = "<p><b>Explained in:</b> " + ", ".join(f'<a href="{rel(h, here)}">{E(t)}</a>' for t, h in tours) + "</p>"
    ment = ""
    if mentions:
        ment = "<h3>What the notes say about it</h3>" + "".join(
            f'<div class="mention"><div class="src">{f"<a href={chr(34)}{rel(m["href"], here)}{chr(34)}>" if m.get("href") else ""}{E(m["doc"])}{"</a>" if m.get("href") else ""}'
            f'{" &rsaquo; " + E(m["heading"]) if m["heading"] else ""}</div><div class="body">{sitemd.inline(m["text"])}</div></div>' for m in mentions)
    secs = "".join(f"<tr><td class=mono>{E(s['name'])}</td><td class=num>{int(s.get('size', 0) or 0):,}</td><td class=num>{float(s.get('fuzzy_match_percent', 0) or 0):.1f}%</td>"
                   f"<td>{meter([('x', float(s.get('fuzzy_match_percent', 0) or 0))], 'thin')}</td></tr>" for s in u.sections)
    frows = []
    for f in u.functions:
        r = f.route or {}
        line = f'<a href="#L{f.line}">line {f.line}</a>' if f.line else '<span class="muted" title="defined in a header, by a macro, or made by the compiler">elsewhere</span>'
        block = ""
        if r:
            block = f"{E(r.get('class') or '')}<br><span class=muted>{E(r.get('why') or '')}{(' &middot; ' + E(r.get('worker'))) if r.get('worker') else ''}{(' &middot; set ' + E(r['frozen'])) if r.get('frozen') else ''}</span>"
        frows.append(f"""<tr><td class="fnname">{E(f.demangled)}{f'<small>{E(f.name)}</small>' if f.demangled != f.name else ''}</td>
<td class="num mono">{f.address:08X}</td><td class="num">{f.size:,}</td><td class="num">{f.pct:.2f}%{meter([('x' if f.pct >= 100 else 'n', f.pct)], 'thin')}</td>
<td>{chip(f.status)}</td><td style="font-size:.82rem">{block}</td><td>{line}</td></tr>""")
    src_html = ""
    if text is not None:
        marks = {f.line: f.status for f in u.functions if f.line}
        src_html = f"""<div class="srcwrap" id="source" style="margin-top:18px"><div class="srchead"><span>{E(u.source or '')}</span><span>{text.count(chr(10)) + 1:,} lines &middot; bar colour = function status</span></div>
<pre class="src" data-src="1" data-marks='{E(json.dumps(marks))}'>{E(text, quote=False)}</pre></div>"""
    else:
        src_html = '<div class="panel"><p class="muted">No source file for this unit yet.</p></div>'
    if header:
        src_html += f"""<details class="srcwrap" style="margin-top:14px"><summary class="srchead" style="cursor:pointer"><span>{E(header[0])}</span><span>its header</span></summary>
<pre class="src" data-src="1">{E(header[1], quote=False)}</pre></details>"""
    crumbs = f'<a href="index.html">source</a> / {E(u.short)}'
    hero = small_hero(f"{st} unit" if st != "wip" else f"in progress: {u.code_pct:.1f}%", E(u.short.rsplit('/', 1)[-1]),
                      f"{ex} of {n} functions exact &middot; {kb(u.total_code)} code" if n else "data only", crumbs)
    body = f"""<div class="grid g2" style="margin-top:18px"><div class="panel explain"><h2>About this unit</h2>{''.join(what)}{tour_html}{hc}</div>
<div class="panel"><h2>Sections</h2><p class="muted" style="margin-top:0">How much of each part of the object file matches the original.</p><div class="tablewrap"><table><thead><tr><th>Section</th><th class=num>Bytes</th><th class=num>Match</th><th></th></tr></thead><tbody>{secs}</tbody></table></div></div></div>
{f'<div class="panel">{ment}</div>' if ment else ''}
<div class="panel"><div class="panel-title"><h2>Functions</h2><span class="hint">target = the original game; ours = this source compiled. Exact means byte for byte. <a href="#source">Jump to the source</a></span></div>
<div class="tablewrap fns"><table><thead><tr><th>Function</th><th class=num>Address</th><th class=num>Bytes</th><th class=num>Match</th><th>Status</th><th>Blocker</th><th>Source</th></tr></thead><tbody>{''.join(frows) or '<tr><td colspan=7 class=muted>none</td></tr>'}</tbody></table></div></div>
{src_html}"""
    return page(here, f"{u.short}", body, "code/index.html", hero, _foot(d))


# ---- markdown pages ------------------------------------------------------------------------------------

def live_panel(d: dict, here: str, systems: list[str]) -> str:
    """Today's numbers for the tour's systems, regenerated on every build (the prose may age; this does not)."""
    rows = [s for s in d["systems"] if s["name"] in systems]
    if not rows:
        return ""
    out = []
    for s in rows:
        ex, nr = pct(s["exact"], s["code"]), pct(s["near"], s["code"])
        units = [u for u in d["units"] if s["name"] in u.systems]
        units.sort(key=lambda u: (u.status != "wip", -u.total_code))
        top = "".join(f'<a class="pill" href="{rel("code/" + u.slug + ".html", here)}">{E(u.short.rsplit("/", 1)[-1])} '
                      f'<b>{"linked" if u.complete else f"{u.code_pct:.0f}%"}</b></a>' for u in units[:14])
        out.append(f"""<div class="panel"><div class="panel-title"><h2>Live: <span style="text-transform:capitalize">{E(s['name'])}</span> is {ex:.1f}% exact</h2>
<a class="hint" href="{rel('code/index.html', here)}?sys={E(s['name'])}">all {s['units']} units</a></div>{meter([('x', ex), ('n', nr)])}
<p class="muted" style="margin:8px 0">{s['linked']} of {s['units']} units linked, {s['open']} functions open ({s['unwritten']} not written), {kb(s['code'])} of code. Updated on every build.</p>{top}</div>""")
    return "".join(out)


def doc_page(d: dict, here: str, title: str, kicker: str, md_html: str, toc: list, active: str, lead: str = "", source: str = "",
             pre: str = "") -> str:
    toc_html = "".join(f'<a class="l{lv}" href="#{E(i)}">{E(t)}</a>' for lv, i, t in toc if lv in (2, 3))
    side = f'<nav class="panel toc"><div class="tt">On this page</div>{toc_html}</nav>' if toc_html else "<div></div>"
    src = f'<p class="muted" style="font-size:.8rem">Source: <code>{E(source)}</code></p>' if source else ""
    body = f'{pre}<div class="doclayout" style="margin-top:22px">{side}<article class="panel prose">{md_html}{src}</article></div>'
    return page(here, f"{title}", body, active, small_hero(kicker, E(title), lead), _foot(d))


def card_page(d: dict, here: str, title: str, kicker: str, text: str, link, active: str, kind: str, lead: str = "", source: str = "") -> str:
    """A markdown file whose `## ` sections become cards (failures, ideas)."""
    parts = re.split(r"(?m)^(?=## )", text)
    intro_md = re.sub(r"(?m)^#\s+.*$", "", parts[0]).strip()
    intro, _ = sitemd.render(intro_md, link)
    cards = []
    for p in parts[1:]:
        h, _ = sitemd.render(p, link)
        cards.append(f'<section class="panel card {kind}">{h}</section>')
    src = f'<p class="muted" style="font-size:.8rem">Source: <code>{E(source)}</code></p>' if source else ""
    body = f'<div class="prose" style="margin-top:20px">{intro}</div><div class="cards">{"".join(cards)}</div>{src}'
    return page(here, f"{title}", body, active, small_hero(kicker, E(title), lead), _foot(d))


def ranked_page(d: dict, here: str, title: str, kicker: str, text: str, link, active: str, lead: str = "", source: str = "") -> str:
    md = re.sub(r"(?m)^#\s+.*$", "", text, count=1)
    h, toc = sitemd.render(md, link)
    h = h.replace("<ol>", '<ol class="rank">', 1)
    src = f'<p class="muted" style="font-size:.8rem">Source: <code>{E(source)}</code></p>' if source else ""
    body = f'<div class="prose" style="margin-top:20px;max-width:980px">{h}{src}</div>'
    return page(here, f"{title}", body, active, small_hero(kicker, E(title), lead), _foot(d))


def tours_index(d: dict, tours: list[dict]) -> str:
    here = "tours/index.html"
    cards = "".join(f"""<a class="panel tourcard" href="{E(t['slug'])}.html"><span style="color:var(--exact)"><svg viewBox="0 0 48 48">{TOUR_ICONS.get(t['slug'], TOUR_ICONS['world'])}</svg></span>
<h3>{E(t['title'])}</h3><p>{sitemd.inline(t['blurb'])}</p></a>""" for t in tours)
    body = f'<div class="grid g3" style="margin-top:22px">{cards or "<p>No tours written yet.</p>"}</div>'
    return page(here, "Guided tours", body, "tours/index.html",
                small_hero("Guided tours", "How the game works, part by part",
                           "Plain-English walks through the systems that matter for exploring the maps on foot and for building a new engine, with links into the real source."), _foot(d))


def library_index(d: dict, sections: list[tuple[str, list[dict]]], experiments: list[dict]) -> str:
    here = "library/index.html"
    out = []
    for name, docs in sections:
        rows = "".join(f'<tr><td><a href="{E(x["page"].split("/", 1)[1])}">{E(x["title"])}</a></td><td class="mono muted" style="font-size:.78rem">{E(x["label"])}</td></tr>' for x in docs)
        out.append(f'<div class="panel"><h2>{E(name)}</h2><div class="tablewrap"><table><tbody>{rows}</tbody></table></div></div>')
    if experiments:
        rows = "".join(
            f"<tr><td class=mono><b>{E(x['id'])}</b></td><td>{E(x['title'])}</td><td>{' '.join(f'<a href={chr(34)}{E(p.split(chr(47), 1)[1])}{chr(34)}>{E(k)}</a>' for k, p in x['pages'])}</td></tr>"
            for x in experiments)
        out.append(f"""<div class="panel"><div class="panel-title"><h2>Council experiments</h2><span class="hint">Each test was pre-registered (pass and kill marks written first), then run.</span></div>
<div class="tablewrap"><table><thead><tr><th>Test</th><th>Title</th><th>Files</th></tr></thead><tbody>{rows}</tbody></table></div></div>""")
    return page(here, "Library", "".join(out), "library/index.html",
                small_hero("Library", "Every doc, note and experiment", "The project's own writing, rendered. The source of truth stays in the repo and beside it."), _foot(d))
