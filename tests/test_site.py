"""dk site: markdown rendering, data gathering and a full build on a synthetic project (no game data)."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_project
from tools.dk.core import site, sitedata, sitemd

REPORT = {
    "measures": {"total_code": "300", "matched_code": "200", "complete_code": "100", "total_data": "40", "matched_data": "20",
                 "total_functions": 3, "matched_functions": 2, "total_units": 2, "complete_units": 1},
    "categories": [{"id": "game", "name": "Game", "measures": {"total_code": "300", "matched_code": "200"}}],
    "units": [
        {"name": "main/game/cam/Cam", "metadata": {"complete": True, "source_path": "src/game/cam/Cam.cpp", "progress_categories": ["game"]},
         "measures": {"total_code": "100", "matched_code_percent": 100.0}, "sections": [{"name": ".text", "size": "100", "fuzzy_match_percent": 100.0}],
         "functions": [{"name": "update__3CamFv", "size": "100", "fuzzy_match_percent": 100.0,
                        "metadata": {"demangled_name": "Cam::update()", "virtual_address": "2147483904"}}]},
        {"name": "main/game/car/Car", "metadata": {"complete": False, "source_path": "src/game/car/Car.cpp"},
         "measures": {"total_code": "200", "matched_code_percent": 50.0}, "sections": [],
         "functions": [{"name": "fn_80000200", "size": "120", "fuzzy_match_percent": 91.5, "metadata": {"virtual_address": "2147484160"}},
                       {"name": "drive__3CarFf", "size": "80", "fuzzy_match_percent": 100.0,
                        "metadata": {"demangled_name": "Car::drive(float)", "virtual_address": "2147484280"}}]},
    ],
}

SOURCES = {
    "src/game/cam/Cam.cpp": "// The demo camera.\n#include \"Cam.h\"\n\nvoid Cam::update() {\n    m0 = 1;\n}\n",
    "src/game/car/Car.cpp": "/* Car <driving> */\nextern \"C\" void fn_80000200(int a) {\n    a++;\n}\n\nvoid Car::drive(float t) {\n}\n",
    "docs/site/tours/cam.md": "# The camera\n\nThe camera follows the car. See [Cam.cpp](../../../src/game/cam/Cam.cpp).\n\n## Parts\n\n- one\n- two\n",
    "docs/site/failures.md": "# What failed\n\nIntro.\n\n## Model A matched 0 of 27\n**Lesson:** test first.\n",
    "docs/site/future.md": "# Next\n\n1. **Do X** - because.\n2. **Do Y** - later.\n",
    "docs/site/milestones.toml": "[[event]]\ndate = \"2026-01-02\"\nkind = \"start\"\ntitle = \"Started\"\ndetail = \"First commit.\"\n",
    "notes/n1.md": "# Notes\n\n## Open\n- game/car/Car: fn_80000200 needs a loop.\n",
}

SITE_TOML = """
[platform.systems]
camera = "game/cam/"
cars = "game/car/|Car"

[site]
content = "docs/site"
out = "build/site"
[site.library]
"Notes" = ["notes/*.md"]
"""


def project(tmp: Path):
    cfg = make_project(tmp, files=SOURCES)
    text = (tmp / "dk.toml").read_text().replace("[platform]\n", "[platform]\n", 1)
    # [platform.systems] must follow [platform]; append the site tables at the end
    (tmp / "dk.toml").write_text(text + SITE_TOML)
    (tmp / "build").mkdir(exist_ok=True)
    (tmp / "build/report.json").write_text(json.dumps(REPORT))
    from tools.dk.core.config import load
    return load(tmp)


class MarkdownTest(unittest.TestCase):
    def test_blocks_and_inline(self):
        h, toc = sitemd.render("# T\n\nA **b** `c<d>` [x](y.md) http://e.com/x\n\n- a\n  - b\n1. c\n\n| h | i |\n|---|---|\n| 1 | 2 |\n\n```cpp\nint a<b>;\n```",
                               link=lambda t: None)
        self.assertIn("<strong>b</strong>", h)
        self.assertIn("<code>c&lt;d&gt;</code>", h)
        self.assertIn('<span class="deadlink">x</span>', h)
        self.assertNotIn("href=\"http", h)            # local site: no outbound links
        self.assertIn("<ul><li>a<ul><li>b", h)
        self.assertIn("<table>", h)
        self.assertIn("int a&lt;b&gt;;", h)
        self.assertEqual(toc[0][:2], (1, "t"))


class DataTest(unittest.TestCase):
    def test_units_status_lines_and_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = project(Path(tmp))
            systems = cfg.platform_options["systems"]
            units = sitedata.load_units(REPORT, systems, [{"unit": "main/game/car/Car", "function": "fn_80000200", "class": "near-shape", "frozen": "bench"}])
            cam, car = units
            self.assertEqual((cam.status, car.status), ("linked", "wip"))
            self.assertEqual(car.systems, ["cars"])
            self.assertEqual(car.functions[0].status, "frozen")
            sitedata.definition_lines(SOURCES["src/game/car/Car.cpp"], car.functions)
            self.assertEqual([f.line for f in car.functions], [2, 6])
            self.assertEqual(sitedata.header_comment(SOURCES["src/game/car/Car.cpp"]), "Car <driving>")
            m = sitedata.find_mentions(units, [("n1", Path("n1.md"), SOURCES["notes/n1.md"])])
            self.assertIn("main/game/car/Car", m)
            self.assertNotIn("main/game/cam/Cam", m)


class BuildTest(unittest.TestCase):
    def test_build_is_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = project(Path(tmp))
            out, stats = site.build(cfg, route=False)
            self.assertEqual(stats["units"], 2)
            for p in ("index.html", "timeline.html", "code/index.html", "code/game__car__Car.html", "tours/cam.html",
                      "failures.html", "future.html", "library/index.html", "assets/site.css", "code/functions.json"):
                self.assertTrue((out / p).is_file(), p)
            car = (out / "code/game__car__Car.html").read_text()
            self.assertIn("/* Car &lt;driving&gt; */", car)                  # source escaped
            self.assertIn('data-marks=\'{&quot;2&quot;: &quot;near&quot;', car)
            self.assertIn("n1", car)                                          # note mention
            tour = (out / "tours/cam.html").read_text()
            self.assertIn('href="../code/game__cam__Cam.html"', tour)        # source link -> unit page
            cam = (out / "code/game__cam__Cam.html").read_text()
            self.assertIn("The camera", cam)                                  # tour back-link
            self.assertIn("Model A matched 0 of 27", (out / "failures.html").read_text())
            self.assertIn('class="rank"', (out / "future.html").read_text())
            self.assertIn("Started", (out / "timeline.html").read_text())
            for f in out.rglob("*.html"):
                self.assertNotRegex(f.read_text(), r'(src|href)="https?://', f.name)   # nothing fetched online
            first = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
            out2, _ = site.build(cfg, route=False)
            second = {p.relative_to(out2): p.read_bytes() for p in out2.rglob("*") if p.is_file()}
            self.assertEqual(first, second)

    def test_wording_comes_from_dk_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = project(Path(tmp))
            out, _ = site.build(cfg, route=False)
            index = (out / "index.html").read_text()
            self.assertIn(f"<title>Status · {cfg.name}</title>", index)      # default brand: [project] name
            self.assertIn("<h1>Rebuilt byte for byte</h1>", index)
            text = (Path(tmp) / "dk.toml").read_text().replace(
                "[site]\n", '[site]\nname = "Demo RE"\nheadline = "Demo, rebuilt"\ncategories_hint = "game: ours"\n', 1)
            (Path(tmp) / "dk.toml").write_text(text)
            out, _ = site.build(cfg, route=False)
            index = (out / "index.html").read_text()
            self.assertIn("<title>Status · Demo RE</title>", index)
            self.assertIn('class="bt">Demo RE<small>', index)
            self.assertIn("<h1>Demo, rebuilt</h1>", index)
            self.assertIn("game: ours", index)


if __name__ == "__main__":
    unittest.main()
