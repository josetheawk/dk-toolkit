// dk site: theme, C/C++ highlighting, unit filter, function search, chart tooltips. No libraries.
(function () {
  "use strict";
  var root = document.documentElement;
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  var on = document.querySelector(".menu a.on");
  if (on && on.scrollIntoView) { var m = on.parentNode; m.scrollLeft = on.offsetLeft - 16; }

  // ---- theme ----
  var btn = document.getElementById("themebtn");
  function paint() { if (btn) btn.textContent = root.getAttribute("data-theme") === "light" ? "☾" : "☀"; }
  paint();
  if (btn) btn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next); store("dk-theme", next); paint();
  });

  // ---- C/C++ highlighting with line numbers ----
  var KW = "alignas alignof asm auto break case catch class const const_cast constexpr continue default delete do dynamic_cast else enum explicit extern false for friend goto if inline mutable namespace new nullptr operator private protected public register reinterpret_cast return sizeof static static_cast struct switch template this throw true try typedef typename union using virtual volatile while __attribute__ __declspec";
  var TY = "void bool char short int long float double signed unsigned size_t u8 u16 u32 u64 s8 s16 s32 s64 f32 f64 BOOL";
  var kw = {}, ty = {};
  KW.split(" ").forEach(function (w) { kw[w] = 1; });
  TY.split(" ").forEach(function (w) { ty[w] = 1; });
  var TOK = /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|("(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(^[ \t]*#[^\n]*)|(\b0[xX][0-9a-fA-F]+[uUlL]*\b|\b\d+\.?\d*(?:[eE][-+]?\d+)?[fFuUlL]*\b)|([A-Za-z_]\w*)(?=\s*\()|([A-Za-z_]\w*)/gm;
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function highlight(text) {
    var out = [], last = 0, m;
    TOK.lastIndex = 0;
    while ((m = TOK.exec(text))) {
      out.push([null, text.slice(last, m.index)]);
      var cls = null, s = m[0];
      if (m[1]) cls = "com"; else if (m[2]) cls = "str"; else if (m[3]) cls = "pre"; else if (m[4]) cls = "num";
      else if (m[5]) cls = kw[s] ? "kw" : ty[s] ? "ty" : "fn";
      else if (m[6]) cls = kw[s] ? "kw" : (ty[s] || /^[A-Z][a-z]\w*[a-z]\w*$/.test(s) && s.length > 3) ? "ty" : null;
      out.push([cls, s]);
      last = TOK.lastIndex;
    }
    out.push([null, text.slice(last)]);
    // split into lines, re-opening spans that cross a newline
    var lines = [""];
    out.forEach(function (p) {
      var parts = p[1].split("\n");
      parts.forEach(function (piece, i) {
        if (i > 0) lines.push("");
        if (!piece) return;
        lines[lines.length - 1] += p[0] ? '<span class="tk-' + p[0] + '">' + esc(piece) + "</span>" : esc(piece);
      });
    });
    return lines;
  }
  document.querySelectorAll("pre.src[data-src]").forEach(function (pre) {
    var text = pre.textContent;
    var marks = {};
    try { marks = JSON.parse(pre.getAttribute("data-marks") || "{}"); } catch (e) {}
    var lines = highlight(text), html = [], st = null;
    for (var i = 0; i < lines.length; i++) {
      var n = i + 1, mk = marks[n];
      if (mk) st = mk;
      html.push('<span class="ln' + (mk ? " fnstart" : "") + '" id="L' + n + '" data-n="' + n + '"' + (st ? ' data-st="' + st + '"' : "") + ">" + (lines[i] || " ") + "</span>");
      if (st && /^\}/.test(text.split("\n")[i] || "")) st = null;
    }
    pre.innerHTML = html.join("");
    function hit() {
      document.querySelectorAll(".ln.hit").forEach(function (e) { e.classList.remove("hit"); });
      var m = /^#L(\d+)$/.exec(location.hash);
      if (m) { var el = document.getElementById("L" + m[1]); if (el) { el.classList.add("hit"); el.scrollIntoView({ block: "center" }); } }
    }
    window.addEventListener("hashchange", hit); hit();
  });

  // ---- unit filter ----
  var table = document.getElementById("units");
  if (table) {
    var q = document.getElementById("q"), rows = Array.prototype.slice.call(table.tBodies[0].rows), counter = document.getElementById("shown");
    var mode = "all";
    var params = new URLSearchParams(location.search);
    if (params.get("sys")) mode = "sys:" + params.get("sys");
    if (params.get("q")) q.value = params.get("q");
    function apply() {
      var needle = q.value.trim().toLowerCase(), n = 0;
      rows.forEach(function (r) {
        var ok = !needle || r.getAttribute("data-k").indexOf(needle) >= 0;
        if (ok && mode !== "all") {
          if (mode.indexOf("sys:") === 0) ok = (" " + r.getAttribute("data-sys") + " ").indexOf(" " + mode.slice(4) + " ") >= 0;
          else ok = r.getAttribute("data-st") === mode || (mode === "open" && r.getAttribute("data-open") !== "0");
        }
        r.style.display = ok ? "" : "none"; if (ok) n++;
      });
      if (counter) counter.textContent = n;
      document.querySelectorAll(".filters button").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-f") === mode); });
      fnsearch(needle);
    }
    document.querySelectorAll(".filters button").forEach(function (b) {
      b.addEventListener("click", function () { mode = b.getAttribute("data-f"); apply(); });
    });
    q.addEventListener("input", apply);
    // function search (loaded on first use)
    var fns = null, loading = false, box = document.getElementById("fnresults");
    function fnsearch(needle) {
      if (!box) return;
      if (needle.length < 3) { box.classList.remove("show"); return; }
      if (!fns) {
        if (!loading) { loading = true; fetch("functions.json").then(function (r) { return r.json(); }).then(function (d) { fns = d; apply(); }).catch(function () {}); }
        return;
      }
      var hits = [];
      for (var i = 0; i < fns.length && hits.length < 40; i++) {
        var f = fns[i];
        if ((f[0] + " " + f[1]).toLowerCase().indexOf(needle) >= 0) hits.push(f);
      }
      if (!hits.length) { box.classList.remove("show"); return; }
      box.innerHTML = '<table><thead><tr><th>Function</th><th>Unit</th><th class="num">Size</th><th>Status</th></tr></thead><tbody>' +
        hits.map(function (f) {
          return '<tr><td class="fnname"><a href="' + f[3] + '.html' + (f[6] ? "#L" + f[6] : "") + '">' + esc(f[1]) + "</a>" +
            (f[1] !== f[0] ? "<small>" + esc(f[0]) + "</small>" : "") + "</td><td class=unitname>" + esc(f[2]) + '</td><td class="num">' + f[4] +
            '</td><td><span class="chip ' + f[5] + '">' + f[5] + "</span> " + f[7] + "</td></tr>";
        }).join("") + "</tbody></table>";
      box.classList.add("show");
    }
    apply();
  }

  // ---- timeline chart tooltip ----
  var chart = document.getElementById("chart");
  if (chart) {
    var data = JSON.parse(document.getElementById("chartdata").textContent);
    var tip = document.getElementById("tip"), cross = document.getElementById("cross"), box2 = chart.parentNode;
    function show(html, cx, cy) {
      tip.innerHTML = html; tip.style.display = "block";
      var r = box2.getBoundingClientRect(), w = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(r.width - w, cx - r.left + 12)) + "px";
      tip.style.top = Math.max(0, cy - r.top - 70) + "px";
    }
    chart.addEventListener("mousemove", function (e) {
      if (e.target.classList && e.target.classList.contains("evdot")) return;
      var r = chart.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * data.w;
      var best = null, bd = 1e9;
      data.p.forEach(function (p) { var d = Math.abs(p[0] - x); if (d < bd) { bd = d; best = p; } });
      if (!best) return;
      cross.setAttribute("x1", best[0]); cross.setAttribute("x2", best[0]); cross.style.display = "";
      show("<b>" + best[2] + "</b><br>Linked code: <b>" + best[3] + "%</b><br>Exact functions: <b>" + best[4] + "%</b> (" + best[5] +
           ")<br>Linked units: <b>" + best[6] + "</b><br><span class=muted>" + esc(best[7]) + "</span>", e.clientX, e.clientY);
    });
    chart.addEventListener("mouseleave", function () { tip.style.display = "none"; cross.style.display = "none"; });
    chart.querySelectorAll(".evdot").forEach(function (d) {
      d.addEventListener("mouseenter", function (e) { show(d.getAttribute("data-tip"), e.clientX, e.clientY); });
      d.addEventListener("click", function () { var t = document.getElementById(d.getAttribute("data-id")); if (t) t.scrollIntoView({ block: "center" }); });
    });
  }
})();
