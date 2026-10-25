"""dtk's config.yml: the `force_active` list (symbols the linker must keep).

The original link kept some functions nothing references (an out-of-line constructor whose
only caller inlined it). A linked unit compiled from source marks its functions removable,
so the linker strips them and the binary shrinks; dtk's `force_active` list puts them in the
linker script's FORCEACTIVE. (The symbols.txt attribute of that name does not reach it.)
"""

from __future__ import annotations

import re

from ...core.config import Config

COMMENT = "# Unreferenced symbols the original link kept: ./dk symbol force-active NAME\n"
_BLOCK = re.compile(r"^(?:" + re.escape(COMMENT) + r")?force_active:[^\n]*\n(?:[ \t]+-[^\n]*\n)*", re.M)


def path(cfg: Config) -> str:
    return cfg.platform_options.get("dtk_config", f"config/{cfg.game_id}/config.yml")


def force_active_names(text: str) -> list[str]:
    m = _BLOCK.search(text)
    if not m:
        return []
    lines = m.group(0).splitlines()
    head = next(l for l in lines if l.startswith("force_active:"))
    inline = re.match(r"force_active:\s*\[(.*)\]", head)
    names = [x.strip() for x in inline.group(1).split(",") if x.strip()] if inline else []
    return names + [l.strip()[1:].strip() for l in lines if l.strip().startswith("-")]


def set_force_active(text: str, names) -> str:
    names = sorted(set(names))
    block = COMMENT + "force_active:\n" + "".join(f"  - {n}\n" for n in names) if names else ""
    m = _BLOCK.search(text)
    if m:
        return text[:m.start()] + block + text[m.end():]
    if not names:
        return text
    return text + ("" if text.endswith("\n") else "\n") + block


def compose(head_text: str, work_text: str, in_unit) -> str:
    """HEAD's config.yml with the force_active entries of one unit (`in_unit(name)`) taken
    from the working copy: other agents' entries stay out of this unit's commit."""
    head, work = force_active_names(head_text), force_active_names(work_text)
    names = [n for n in head if not in_unit(n)] + [n for n in work if in_unit(n)]
    return set_force_active(head_text, names) if sorted(set(names)) != sorted(set(head)) else head_text
