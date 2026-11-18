"""dk site: the project's local status website, generated from the repo's own data.

  ./dk site build    write the site to [site] out (default build/dk/site), in seconds, deterministically
  ./dk site serve    serve it on 127.0.0.1:[site] port (default 8003); rebuilds when HEAD, the report or the
                     hand-written content change (checked every 30 s; --no-watch turns that off)
  ./dk site shots    screenshot each main view with headless Chrome into [site] shots (build/dk/site-shots)
  ./dk site stop     stop the server started by serve/open (pid in build/dk/site-serve.pid)
  ./dk site open     start `serve` in the background when nothing answers on the port, then open the browser

The site shows the game's source, so it binds to 127.0.0.1 only and loads nothing from the internet.
Pages: status, timeline, source browser (a page per unit), guided tours, roadmap, lessons, next steps, library.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import sitedata as sd
from . import sitemd
from . import sitepages as sp
from .command import OK, USAGE, Command, hint
from .config import Config
from .progress import load_report, stale

ASSETS = Path(__file__).with_name("site_assets")
ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect x="2" y="2" width="36" height="36" rx="11" fill="#3b22a8"/>
<path d="M8 27 A13 13 0 0 1 32 27" fill="none" stroke="#38f2c4" stroke-width="3.2" stroke-linecap="round"/>
<path d="M20 27 L28.5 15.5" stroke="#ffb547" stroke-width="3" stroke-linecap="round"/><circle cx="20" cy="27" r="3" fill="#fff"/></svg>"""
THRESHOLDS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def _measures(m: dict) -> dict:
    keys = ("total_code", "matched_code", "complete_code", "total_data", "matched_data", "total_functions",
            "matched_functions", "total_units", "complete_units")
    return {k: sd._n(m, k) for k in keys}


def _slug(path: Path, home: Path) -> str:
    try:
        r = str(path.relative_to(home))
    except ValueError:
        r = str(path).lstrip("/")
    r = r.replace("../", "").removesuffix(".md")
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in r.replace("/", "--"))


def label(p: Path, cfg: Config, home: Path) -> str:
    """A short readable path: relative to the checkout, else the main checkout, else the folder beside it."""
    for base in (cfg.root, home, home.parent):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(p).replace(str(Path.home()), "~")


def gather(cfg: Config, route: bool = True) -> dict:
    opts = sd.site_options(cfg)
    sp.set_brand(opts["name"])
    home = sd.main_root(cfg.root)
    report = load_report(cfg) or {"units": [], "measures": {}, "categories": []}
    systems = dict(cfg.platform_options.get("systems", {}) or {})
    r = sd.route_json(cfg) if route else {}
    units = sd.load_units(report, systems, r.get("rows", []))
    content = sd.resolve_dir(cfg, opts["content"], home)
    d: dict = {
        "opts": opts, "home": home, "game_id": cfg.game_id or cfg.name, "measures": _measures(report.get("measures", {})),
        "categories": [{"id": c.get("id", "?"), "name": c.get("name", ""), **_measures(c.get("measures", {}))}
                       for c in report.get("categories", []) if sd._n(c.get("measures", {}), "total_code")],
        "units": units, "route": r, "systems": sd.system_rows(units, systems), "system_names": list(systems),
        "head": sd.head_info(cfg.root), "recent": sd.commits(cfg.root, 15), "stale": stale(cfg) if cfg.report.is_file() else None,
        "content": content, "content_rel": opts["content"],
    }
    # frozen labels of open functions, and released functions
    fr: dict[str, int] = {}
    for row in r.get("rows", []):
        if row.get("frozen"):
            fr[row["frozen"]] = fr.get(row["frozen"], 0) + 1
    d["frozen"] = fr
    rel_files = sd.resolve(cfg, opts["released"], home) if opts["released"] else []
    try:
        doc = json.loads(rel_files[0].read_text()) if rel_files else {}
        d["released"] = {k: v for k, v in doc.items() if isinstance(v, dict)} if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        d["released"] = {}
    # units one function from linking
    near = []
    for u in units:
        if u.complete or len(u.open_functions) != 1 or u.short.startswith("auto_"):
            continue
        f = u.open_functions[0]
        near.append({"slug": u.slug, "short": u.short, "fn": f.demangled, "pct": f.pct, "status": f.status,
                     "cls": (f.route or {}).get("class") or "", "code": u.total_code, "fnobj": f, "unit": u, "line": None})
    near.sort(key=lambda x: (-x["code"], x["short"]))
    d["one_from_link"] = near
    # history and events
    hist = sd.floor_history(cfg.root, opts["floor"])
    d["history"] = hist
    per: dict[str, int] = {}
    for c in sd.commits(cfg.root):
        k = sp.day(c["t"])
        per[k] = per.get(k, 0) + 1
    d["per_day"] = sorted(per.items())
    d["events"] = events(content, hist, d["measures"])
    return d


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    import re as _r
    m = _r.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:240]


def _event_time(date: str) -> int:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(time.mktime(time.strptime(date, fmt))) + (12 * 3600 if fmt == "%Y-%m-%d" else 0)
        except ValueError:
            continue
    return 0


def events(content: Path | None, hist: list[dict], m: dict) -> list[dict]:
    out = []
    for e in sd.load_milestones(content / "milestones.toml" if content else None):
        date = str(e.get("date"))
        out.append({"date": date.replace("T", " "), "t": _event_time(date), "kind": str(e.get("kind", "decision")),
                    "title": str(e.get("title", "")), "detail": str(e.get("detail", ""))})
    tc = m.get("total_code") or 0
    seen = set()
    for h in hist:
        p = sp.pct(h["complete_code"], tc)
        for th in THRESHOLDS:
            if p >= th and th not in seen and tc:
                seen.add(th)
                if h is hist[0]:
                    continue
                out.append({"date": sp.stamp(h["t"]), "t": h["t"], "kind": "auto", "title": f"Linked code passed {th}%",
                            "detail": f"`{h['rev']}` {h['subject'][:120]}"})
        if h is hist[0]:
            seen.update(th for th in THRESHOLDS if p >= th)
    out.sort(key=lambda e: (e["t"], e["title"]))
    return out


# ---- build ------------------------------------------------------------------------------------------------

class Linker:
    """Rewrites links in rendered markdown: docs that have a page, sources that belong to a unit."""

    def __init__(self, pages: dict[Path, str], sources: dict[Path, str], roots: tuple[Path, ...] = ()):
        self.pages, self.sources, self.roots = pages, sources, roots

    def for_doc(self, doc: Path, here: str):
        def link(target: str) -> str | None:
            if target.startswith(("http://", "https://", "mailto:")):
                return None
            if target.startswith("#"):
                return target
            path, _, frag = target.partition("#")
            dest = Path(os.path.normpath(doc.parent / path))
            if dest in self.pages:
                return sp.rel(self.pages[dest] + (f"#{frag}" if frag else ""), here)
            if dest in self.sources:
                return sp.rel(self.sources[dest], here)
            for root in self.roots:                      # a header: its unit's page; a source folder: the filtered list
                try:
                    r = dest.relative_to(root)
                except ValueError:
                    continue
                if r.parts and r.parts[0] == "include" and dest.suffix in (".h", ".hpp"):
                    for ext in (".cpp", ".c"):
                        src = root / "src" / Path(*r.parts[1:]).with_suffix(ext)
                        if src in self.sources:
                            return sp.rel(self.sources[src], here)
                if r.parts and r.parts[0] == "src" and dest.is_dir():
                    return sp.rel("code/index.html", here) + "?q=" + "/".join(r.parts[1:]).lower()
            return None
        return link


def build(cfg: Config, route: bool = True, out: Path | None = None) -> tuple[Path, dict]:
    t_start = time.time()
    d = gather(cfg, route)
    opts, home = d["opts"], d["home"]
    out = out or (cfg.root / opts["out"])
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "assets").mkdir(parents=True)
    (tmp / "code").mkdir()
    (tmp / "tours").mkdir()
    (tmp / "library").mkdir()
    for a in sorted(ASSETS.iterdir()):
        shutil.copyfile(a, tmp / "assets" / a.name)
    (tmp / "assets" / "icon.svg").write_text(ICON)

    def write(rel_: str, text: str) -> None:
        p = tmp / rel_
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    # sources and their units
    sources: dict[Path, str] = {}
    texts: dict[str, str | None] = {}
    for u in d["units"]:
        if not u.source:
            texts[u.name] = None
            continue
        p = Path(os.path.normpath(cfg.root / u.source))
        sources[p] = f"code/{u.slug}.html"
        try:
            texts[u.name] = p.read_text(errors="replace")
        except OSError:
            texts[u.name] = None
        if texts[u.name]:
            sd.definition_lines(texts[u.name], u.functions)
    for x in d["one_from_link"]:
        x["line"] = x["fnobj"].line

    # library documents
    lib_sections: list[tuple[str, list[dict]]] = []
    pages: dict[Path, str] = {}
    docs_text: dict[Path, str] = {}
    for name, specs in (opts.get("library") or {}).items():
        items = []
        for spec in specs if isinstance(specs, list) else [specs]:
            for p in sd.resolve(cfg, spec, home):
                if p in pages:
                    continue
                try:
                    text = p.read_text(errors="replace")
                except OSError:
                    continue
                page_ = f"library/{_slug(p, home)}.html"
                pages[p] = page_
                docs_text[p] = text
                items.append({"path": p, "page": page_, "title": sitemd.first_heading(text) or p.stem, "label": label(p, cfg, home)})
        if items:
            lib_sections.append((name, items))
    experiments = []
    exp_dir = sd.resolve_dir(cfg, opts.get("experiments", ""), home)
    if exp_dir:
        for sub in sorted(exp_dir.iterdir(), key=lambda p: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", p.name)]):
            if not sub.is_dir():
                continue
            files = [sub / n for n in ("RESULT.md", "PREREG.md", "PLAN.md") if (sub / n).is_file()]
            if not files:
                continue
            ps = []
            title = ""
            for f in files:
                text = f.read_text(errors="replace")
                page_ = f"library/{_slug(f, home)}.html"
                pages[f] = page_
                docs_text[f] = text
                ps.append((f.stem.lower(), page_))
                title = title or (sitemd.first_heading(text) or "")
            experiments.append({"id": sub.name, "title": title, "pages": ps})
    # content pages: tours and story
    content = d["content"]
    tours = []
    tour_files = sorted((content / "tours").glob("*.md")) if content and (content / "tours").is_dir() else []
    order = list(opts.get("tours", {}).get("order", [])) if isinstance(opts.get("tours"), dict) else []
    tour_files.sort(key=lambda p: (order.index(p.stem) if p.stem in order else 99, p.stem))
    for p in tour_files:
        text = p.read_text()
        paras = [b for b in text.split("\n\n") if b.strip() and not b.lstrip().startswith("#")]
        tours.append({"slug": p.stem, "path": p, "title": sitemd.first_heading(text) or p.stem, "text": text,
                      "blurb": _first_sentence(paras[0] if paras else "")})
        pages[p] = f"tours/{p.stem}.html"
    story = {}
    for stem in ("ideas", "failures", "future"):
        p = content / f"{stem}.md" if content else None
        if p and p.is_file():
            story[stem] = (p, p.read_text())
            pages[p] = f"{stem}.html"
    linker = Linker(pages, sources, (Path(os.path.normpath(cfg.root.absolute())),))
    systems_tour = (opts.get("tours", {}) or {}).get("systems", {}) if isinstance(opts.get("tours"), dict) else {}
    titles = {t["slug"]: t["title"] for t in tours}
    d["tour_for_system"] = {s: titles[t] for s, t in systems_tour.items() if t in titles}

    # which tours link each source
    tour_links: dict[str, list[tuple[str, str]]] = {}
    import re as _re
    for t in tours:
        for target in _re.findall(r"\]\(([^)\s]+)\)", t["text"]):
            dest = Path(os.path.normpath(t["path"].parent / target.split("#")[0]))
            if dest in sources:
                lst = tour_links.setdefault(sources[dest], [])
                if (t["title"], f"tours/{t['slug']}.html") not in lst:
                    lst.append((t["title"], f"tours/{t['slug']}.html"))

    # mentions: notes, faithfulness and every library doc
    mention_docs = [(label(p, cfg, home), p, docs_text[p]) for p in sorted(docs_text)]
    mentions = sd.find_mentions(d["units"], mention_docs)

    # pages
    write("index.html", sp.status_page(d))
    write("timeline.html", sp.timeline_page(d))
    write("code/index.html", sp.code_index(d))
    write("code/functions.json", sp.functions_json(d))
    for u in d["units"]:
        here = f"code/{u.slug}.html"
        ms = [{**m, "href": pages.get(m["path"])} for m in mentions.get(u.name, [])]
        header = None
        if u.source:
            src = Path(u.source)
            for cand in (Path("include") / src.relative_to("src").with_suffix(".h") if src.parts and src.parts[0] == "src" else None,
                         src.with_suffix(".h")):
                if cand and (cfg.root / cand).is_file():
                    header = (str(cand), (cfg.root / cand).read_text(errors="replace"))
                    break
        write(here, sp.unit_page(d, u, texts.get(u.name), header, ms, tour_links.get(here, [])))
    write("tours/index.html", sp.tours_index(d, tours))
    for t in tours:
        here = f"tours/{t['slug']}.html"
        h, toc = sitemd.render(_re.sub(r"(?m)^#\s+.*$", "", t["text"], count=1), linker.for_doc(t["path"], here))
        mine = [sysname for sysname, slug in systems_tour.items() if slug == t["slug"]]
        write(here, sp.doc_page(d, here, t["title"], "Guided tour", h, toc, "tours/index.html", t["blurb"],
                                os.path.relpath(t["path"], cfg.root), pre=sp.live_panel(d, here, mine)))
    meta = {"ideas": ("Ideas and roadmap", "Where this is going", "ideas.html"),
            "failures": ("What didn't work", "Failures and lessons", "failures.html"),
            "future": ("What we could do next", "Next steps, ranked", "future.html")}
    for stem, (title, kicker, here) in meta.items():
        if stem not in story:
            write(here, sp.doc_page(d, here, title, kicker, f"<p>No <code>{stem}.md</code> in the content folder yet.</p>", [], here))
            continue
        p, text = story[stem]
        link = linker.for_doc(p, here)
        heading = sitemd.first_heading(text) or title
        srcrel = os.path.relpath(p, cfg.root)
        if stem == "failures":
            write(here, sp.card_page(d, here, heading, kicker, text, link, here, "fail", source=srcrel))
        elif stem == "future":
            write(here, sp.ranked_page(d, here, heading, kicker, text, link, here, source=srcrel))
        else:
            h, toc = sitemd.render(_re.sub(r"(?m)^#\s+.*$", "", text, count=1), link)
            write(here, sp.doc_page(d, here, heading, kicker, h, toc, here, source=srcrel))
    write("library/index.html", sp.library_index(d, lib_sections, experiments))
    for p, page_ in pages.items():
        if not page_.startswith("library/"):
            continue
        text = docs_text[p]
        h, toc = sitemd.render(_re.sub(r"(?m)^#\s+.*$", "", text, count=1), linker.for_doc(p, page_))
        write(page_, sp.doc_page(d, page_, sitemd.first_heading(text) or p.stem, "Library", h, toc, "library/index.html",
                                 source=label(p, cfg, home)))

    if out.exists():
        old = out.with_name(out.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        out.rename(old)
        tmp.rename(out)
        shutil.rmtree(old)
    else:
        tmp.rename(out)
    stats = {"units": len(d["units"]), "pages": sum(1 for _ in out.rglob("*.html")), "tours": len(tours),
             "docs": len(pages), "events": len(d["events"]), "seconds": round(time.time() - t_start, 1)}
    return out, stats


# ---- serve -------------------------------------------------------------------------------------------------

def _signature(cfg: Config, opts: dict) -> tuple:
    head = sd.head_info(cfg.root)["rev"]
    rep = cfg.report.stat().st_mtime if cfg.report.is_file() else 0
    content = cfg.root / opts["content"]
    newest = max((p.stat().st_mtime for p in content.rglob("*") if p.is_file()), default=0) if content.is_dir() else 0
    return head, rep, newest


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def serve(cfg: Config, port: int, watch: bool, rebuild: bool, route: bool) -> int:
    opts = sd.site_options(cfg)
    out = cfg.root / opts["out"]
    if rebuild or not (out / "index.html").is_file():     # always, unless --no-rebuild: the old build may predate the code
        out, stats = build(cfg, route)
        print(f"built {cfg.rel(out)}: {stats}", flush=True)
    handler = functools.partial(_Quiet, directory=str(out))
    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"error: cannot listen on 127.0.0.1:{port}: {e}")
        return USAGE
    pidfile = cfg.root / "build/dk/site-serve.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(f"{os.getpid()}\n")
    print(f"serving http://127.0.0.1:{port}/ (local only; Ctrl-C or ./dk site stop ends it)", flush=True)
    if watch:
        def loop():
            sig = _signature(cfg, opts)
            while True:
                time.sleep(30)
                try:
                    now = _signature(cfg, opts)
                    if now != sig:
                        sig = now
                        _, stats = build(cfg, route)
                        print(f"{time.strftime('%H:%M:%S')} rebuilt: {stats}", flush=True)
                except Exception as e:  # keep serving the last good build
                    print(f"rebuild failed: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return OK


SHOTS = [("status", "index.html", 1300, 2600), ("status-dark", "index.html?theme=dark", 1300, 2600),
         ("timeline", "timeline.html", 1300, 2400), ("source-index", "code/index.html", 1300, 1800),
         ("unit", "code/{unit}.html", 1300, 3000), ("unit-dark", "code/{unit}.html?theme=dark", 1300, 3000),
         ("tour", "tours/{tour}.html", 1300, 2600), ("roadmap", "ideas.html", 1300, 2400), ("lessons", "failures.html", 1300, 2600),
         ("next", "future.html", 1300, 2400), ("library", "library/index.html", 1300, 2000),
         ("phone-status", "index.html", 390, 2400), ("phone-unit", "code/{unit}.html?theme=dark", 390, 2400)]


def shots(cfg: Config, port: int, unit: str, tour: str) -> int:
    """Screenshot each main view with headless Chrome (to check the layout by eye)."""
    opts = sd.site_options(cfg)
    chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        print("error: no google-chrome or chromium on PATH")
        return USAGE
    if not _listening(port):
        print(f"error: nothing serves 127.0.0.1:{port}; run ./dk site serve first")
        return USAGE
    unit, tour = unit or str(opts.get("shot_unit", "")), tour or str(opts.get("shot_tour", ""))
    if not unit or not tour:
        print("error: name the pages to capture: --unit U --tour T, or dk.toml [site] shot_unit / shot_tour")
        return USAGE
    out = cfg.root / opts["shots"]
    out.mkdir(parents=True, exist_ok=True)
    for name, path, w, h in SHOTS:
        url = f"http://127.0.0.1:{port}/" + path.format(unit=unit, tour=tour)
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                        f"--screenshot={out / (name + '.png')}", url], capture_output=True, timeout=120)
        print(f"{cfg.rel(out / (name + '.png'))}  <- {url}")
    return OK


def run(cfg: Config, args: argparse.Namespace) -> int:
    opts = sd.site_options(cfg)
    port = args.port or int(opts["port"])
    route = not args.no_route
    if args.action == "build":
        out, stats = build(cfg, route)
        print(f"wrote {cfg.rel(out)}: {stats['pages']} pages ({stats['units']} units, {stats['tours']} tours, "
              f"{stats['docs']} docs, {stats['events']} events) in {stats['seconds']} s")
        hint("site serve", "site open")
        return OK
    if args.action == "stop":
        pidfile = cfg.root / "build/dk/site-serve.pid"
        try:
            pid = int(pidfile.read_text().split()[0])
            os.kill(pid, 15)
            pidfile.unlink()
            print(f"stopped the site server (pid {pid})")
            return OK
        except (OSError, ValueError, IndexError):
            print("no running site server recorded in build/dk/site-serve.pid")
            return USAGE
    if args.action == "shots":
        return shots(cfg, port, args.unit, args.tour)
    if args.action == "serve":
        return serve(cfg, port, not args.no_watch, not args.no_rebuild, route)
    if args.action == "open":
        url = f"http://127.0.0.1:{port}/"
        if not _listening(port):
            log = cfg.root / "build/dk/site-serve.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            subprocess.Popen([str(cfg.root / "dk"), "site", "serve", "--port", str(port)], cwd=cfg.root,
                             stdout=log.open("a"), stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(240):
                if _listening(port):
                    break
                time.sleep(0.5)
            print(f"started the server in the background (log: {cfg.rel(log)})")
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener and not args.no_browser:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(url)
        return OK
    return USAGE


def configure(p: argparse.ArgumentParser) -> None:
    p.add_argument("action", choices=["build", "serve", "open", "stop", "shots"])
    p.add_argument("--port", type=int, default=0, help="default: dk.toml [site] port (8003)")
    p.add_argument("--no-route", action="store_true", help="skip `dk route --json` (blocker classes; ~8 s)")
    p.add_argument("--no-watch", action="store_true", help="serve: do not rebuild when HEAD, the report or the content change")
    p.add_argument("--rebuild", action="store_true", help=argparse.SUPPRESS)   # the default now
    p.add_argument("--no-rebuild", action="store_true", help="serve: serve the existing build without rebuilding first")
    p.add_argument("--unit", default="", help="shots: the unit page to capture (default: dk.toml [site] shot_unit)")
    p.add_argument("--tour", default="", help="shots: the tour page to capture (default: dk.toml [site] shot_tour)")
    p.add_argument("--no-browser", action="store_true", help="open: start the server and print the URL only")


COMMAND = Command("site", "docs", "local status website: build / serve on 127.0.0.1 / open", run, configure)
