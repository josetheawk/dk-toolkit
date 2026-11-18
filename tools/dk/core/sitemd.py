"""A small Markdown to HTML renderer for `dk site` (stdlib only, deterministic).

Covers what the project's docs use: ATX headings, paragraphs, bullet and numbered lists (nested by indent),
fenced code, tables, block quotes, rules, and inline code, bold, italic, links and images-as-links.
Links go through `link(target) -> href | None`; None drops the link and keeps its text.
"""

from __future__ import annotations

import html
import re
from typing import Callable

LinkFn = Callable[[str], "str | None"]

_INLINE = re.compile(
    r"(?P<code>`+)(?P<codetext>.+?)(?P=code)"
    r"|!?\[(?P<ltext>[^\]]*)\]\((?P<href>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
    r"|\*\*(?P<b>.+?)\*\*"
    r"|__(?P<b2>.+?)__"
    r"|(?<![\w*])\*(?P<i>[^*\s][^*]*?)\*(?![\w*])"
    r"|(?<![\w_])_(?P<i2>[^_\s][^_]*?)_(?![\w_])"
    r"|(?P<url>https?://[^\s<>()]+[^\s<>().,;:])"
)


def slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text).lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[\s]+", "-", s.strip()) or "section"


def inline(text: str, link: LinkFn | None = None) -> str:
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(html.escape(text[pos:m.start()], quote=False))
        pos = m.end()
        if m.group("code"):
            out.append(f"<code>{html.escape(m.group('codetext').strip(), quote=False)}</code>")
        elif m.group("href") is not None:
            label = inline(m.group("ltext"), link)
            href = link(m.group("href")) if link else m.group("href")
            out.append(f'<a href="{html.escape(href)}">{label}</a>' if href else f'<span class="deadlink">{label}</span>')
        elif m.group("b") or m.group("b2"):
            out.append(f"<strong>{inline(m.group('b') or m.group('b2'), link)}</strong>")
        elif m.group("i") or m.group("i2"):
            out.append(f"<em>{inline(m.group('i') or m.group('i2'), link)}</em>")
        elif m.group("url"):
            u = m.group("url")
            out.append(f'<span class="url">{html.escape(u)}</span>')   # local site: no outbound links
    out.append(html.escape(text[pos:], quote=False))
    return "".join(out)


def _table(rows: list[str], link: LinkFn | None) -> str:
    def cells(r: str) -> list[str]:
        r = r.strip()
        if r.startswith("|"):
            r = r[1:]
        if r.endswith("|") and not r.endswith("\\|"):
            r = r[:-1]
        return [c.strip() for c in re.split(r"(?<!\\)\|", r)]
    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    h = "".join(f"<th>{inline(c, link)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c, link)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")


def _list(lines: list[str], link: LinkFn | None) -> str:
    """Nested list from consecutive item/continuation lines."""
    items: list[tuple[int, bool, str]] = []   # (indent, ordered, text)
    for ln in lines:
        m = _LIST.match(ln)
        if m:
            items.append((len(m.group(1).expandtabs(4)), m.group(2)[0].isdigit(), m.group(3)))
        elif items:
            ind, o, t = items[-1]
            items[-1] = (ind, o, t + " " + ln.strip())
    out: list[str] = []
    stack: list[tuple[int, str]] = []
    for ind, ordered, text in items:
        while stack and ind < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if not stack or ind > stack[-1][0]:
            tag = "ol" if ordered else "ul"
            stack.append((ind, tag))
            out.append(f"<{tag}><li>")
        else:
            out.append("</li><li>")
        out.append(inline(text, link))
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def render(text: str, link: LinkFn | None = None, shift: int = 0) -> tuple[str, list[tuple[int, str, str]]]:
    """(html, headings as (level, id, text)). `shift` demotes headings (shift=1: # -> h2)."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}
    para: list[str] = []
    i = 0

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(s.strip() for s in para), link)}</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("<!--"):
            flush()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if s.startswith("```") or s.startswith("~~~"):
            flush()
            fence, lang = s[:3], s[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f'<pre class="block"><code{cls}>{html.escape(chr(10).join(body), quote=False)}</code></pre>')
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", s)
        if m:
            flush()
            level = min(6, len(m.group(1)) + shift)
            body = inline(m.group(2), link)
            hid = slug(m.group(2))
            if hid in seen:
                seen[hid] += 1
                hid = f"{hid}-{seen[hid]}"
            else:
                seen[hid] = 0
            toc.append((level, hid, re.sub(r"<[^>]+>", "", body)))
            out.append(f'<h{level} id="{hid}">{body}</h{level}>')
            i += 1
            continue
        if re.match(r"^(\*\s*){3,}$|^(-\s*){3,}$|^(_\s*){3,}$", s):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_table(rows, link))
            continue
        if s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner, _ = render("\n".join(quote), link, shift)
            out.append(f"<blockquote>{inner}</blockquote>")
            continue
        if _LIST.match(ln):
            flush()
            block = []
            while i < len(lines):
                cur = lines[i]
                if _LIST.match(cur) or (cur.strip() and cur.startswith((" ", "\t")) and block):
                    block.append(cur)
                    i += 1
                elif not cur.strip() and i + 1 < len(lines) and _LIST.match(lines[i + 1]) and \
                        (lines[i + 1].startswith((" ", "\t")) or block):
                    i += 1
                else:
                    break
            out.append(_list(block, link))
            continue
        if not s:
            flush()
            i += 1
            continue
        para.append(ln)
        if ln.endswith("  "):
            para[-1] = ln.rstrip() + "\ue000"
        i += 1
    flush()
    return "\n".join(out).replace("\ue000", "<br>"), toc


def first_heading(text: str) -> str | None:
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    return m.group(1) if m else None
