"""dk llm-agent --menu: the legal-move menu and grammar-constrained replies (council T43 idea 1, T44 C).

Before each turn the toolkit lists every rewrite-table move that applies to the current definition: each
(transform, anchor line, pick) whose rewrite exists there, found with llm-agent's own rule (`rewrites.candidates`
at that line, the Site llm-agent's `structured` builds), so a menu move applies exactly as the agent applies it.
Moves that give the same source are listed once. The menu goes into the state after the numbered definition, each
move written as the reply that selects it (the SFT data's structured form, `{"transform": "<id>", "anchor": N}`).

Grammar: the reply is constrained to the menu's moves, a STOP label or (optionally) an edit:
- Ollama (`format`): a JSON schema whose alternatives are exactly the menu's moves (`const` transform, anchor,
  pick), `{"stop": <label>}` and `{"search": ..., "replace": ...}`; Ollama hands it to llama.cpp's converter;
- llama-server (`--backend openai`): a GBNF grammar with the trained text forms (the move lines verbatim,
  `STOP: <label>`, the SEARCH/REPLACE block).
`parse_reply` maps the JSON forms to llmagent.Action and leaves everything else to `llmagent.parse_action`, so an
unconstrained reply is judged by the same rules as without the menu.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DEFAULT_LIMIT = 40
HEADER = ("Legal transforms for this definition (each one has a rewrite at its line; reply with one of these "
          "lines exactly, or an edit, or STOP):")
HEADER_NO_EDIT = ("Legal transforms for this definition (each one has a rewrite at its line; reply with one of "
                  "these lines exactly, or STOP):")
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass(frozen=True)
class Move:
    transform: str
    anchor: int                 # 1-based line of the definition (llm-agent's anchor)
    pick: int                   # index among the rule's candidates at that line (llm-agent's "pick")
    label: str = field(default="", compare=False)
    new: str = field(default="", compare=False, repr=False)

    def reply(self) -> str:
        d: dict = {"transform": self.transform, "anchor": self.anchor}
        if self.pick:
            d["pick"] = self.pick
        return json.dumps(d)


def enumerate_moves(text: str, span: tuple[int, int], at, rules) -> list[Move]:
    """Every (transform, anchor, pick) with a rewrite in the definition text[span], by line then table order;
    `at(rule, file_line0)` -> [(label, new source)] is the agent's rule at one line. A source two moves give is
    listed once (its first move); a move that changes nothing is dropped."""
    first = text.count("\n", 0, span[0])
    n = text.count("\n", span[0], span[1]) + 1
    seen = {text}
    out: list[Move] = []
    for k in range(1, n + 1):
        for name in rules:
            for pick, (label, new) in enumerate(at(name, first + k - 1)):
                if new in seen:
                    continue
                seen.add(new)
                out.append(Move(name, k, pick, label, new))
    return out


def site_at(text: str, body, cplusplus: bool, members=None, accessors=None):
    """llm-agent's structured() per line, keeping labels: at(rule, line0) -> [(label, new)]."""
    from . import rewrites

    def at(rule: str, line0: int) -> list[tuple[str, str]]:
        if body is None or rule not in rewrites.BY_NAME:
            return []
        site = rewrites.Site(text, body, {line0}, cplusplus, members=members, accessors=accessors)
        try:
            return [(label, new) for _r, label, new in rewrites.candidates(site, [rewrites.BY_NAME[rule]], groups=False)]
        except Exception:                                # noqa: BLE001 (a rule that fails on a body applies nowhere)
            return []
    return at


def env_moves(env, text: str, score, rules=None) -> list[Move]:
    """The menu for llm-agent's CompilerEnv state (the same body and Site its `structured` uses)."""
    from . import rewrites
    span = env.definition(text)
    if span is None or score is None or not getattr(score, "compiled", False):
        return []
    body = env.closer.body(text, getattr(score.handle, "first_line", None))
    names = [r.name for r in rewrites.RULES] if rules is None else list(rules)
    return enumerate_moves(text, span, site_at(text, body, not env.closer.c, members=env.closer.members), names)


def turn_menu(env, text: str, score, limit: int = DEFAULT_LIMIT) -> tuple[list[Move], int]:
    """(the moves shown, all legal moves) for one agent turn. CompilerEnv: its own body and Site (labels kept);
    any other env: its `structured(text, score, rule, line0)`, so the menu is exactly what the agent applies."""
    from . import rewrites
    span = env.definition(text)
    if span is None or not getattr(score, "compiled", False):
        return [], 0
    if hasattr(env, "closer"):
        moves = env_moves(env, text, score)
    else:
        def at(rule, line0):
            return [("", new) for new in env.structured(text, score, rule, line0)]
        moves = enumerate_moves(text, span, at, [r.name for r in rewrites.RULES])
    sym = getattr(getattr(score, "handle", None), "sym", None)
    return rank(moves, text, span, sym, limit), len(moves)


def rank(moves: list[Move], text: str, span: tuple[int, int], sym=None, limit: int = DEFAULT_LIMIT) -> list[Move]:
    """At most `limit` moves for the prompt: moves at lines the differing rows come from first (sym.lines,
    1-based file lines, +-1), then the symptom table's rule order, then line order; shown in line order."""
    if limit <= 0 or len(moves) <= limit:
        return list(moves)
    from . import rewrites
    first = text.count("\n", 0, span[0])
    hot = set()
    order = {r.name: i for i, r in enumerate(rewrites.RULES)}
    if sym is not None:
        hot = {ln - first + d for ln in getattr(sym, "lines", {}) for d in (-1, 0, 1)}
        order = {r.name: i for i, r in enumerate(rewrites.ranked(sym))}
    key = lambda m: (m.anchor not in hot, order.get(m.transform, 99), m.anchor, m.pick)
    keep = set(sorted(moves, key=key)[:limit])
    return [m for m in moves if m in keep]


def menu_block(moves: list[Move], edits: bool = True, width: int = 70) -> str:
    """The menu as prompt text: one reply line per move with a short note of what it rewrites."""
    if not moves:
        return "No transform from the table applies to this definition (an edit or STOP)." if edits else \
            "No transform from the table applies to this definition (STOP)."
    lines = [HEADER if edits else HEADER_NO_EDIT]
    for m in moves:
        note = " ".join(m.label.split())
        lines.append(f"  {m.reply()}" + (f"   {note[:width]}" if note else ""))
    return "\n".join(lines)


def add_menu(messages: list[dict], block: str) -> list[dict]:
    """The state with the menu after the numbered definition (before the diff rows); other parts untouched."""
    out = [dict(m) for m in messages]
    u = out[-1]["content"]
    i = u.find("\nDiff (target | ours")
    if i < 0:
        i = u.rfind("\nYour one action:")
    u = (u[:i + 1] + block + "\n" + u[i + 1:]) if i >= 0 else u + "\n" + block + "\n"
    out[-1]["content"] = u
    return out


# ---- grammars -------------------------------------------------------------------------------------------------

def json_schema(moves: list[Move], labels, edits: bool = True) -> dict:
    """Ollama `format`: one alternative per menu move, a STOP label, and (edits) a search/replace pair."""
    alts: list[dict] = []
    for m in moves:
        props: dict = {"transform": {"const": m.transform}, "anchor": {"const": m.anchor}}
        req = ["transform", "anchor"]
        if m.pick:
            props["pick"] = {"const": m.pick}
            req.append("pick")
        alts.append({"type": "object", "properties": props, "required": req, "additionalProperties": False})
    alts.append({"type": "object", "properties": {"stop": {"enum": sorted(labels)}}, "required": ["stop"],
                 "additionalProperties": False})
    if edits:
        alts.append({"type": "object", "properties": {"search": {"type": "string", "minLength": 1},
                                                      "replace": {"type": "string"}},
                     "required": ["search", "replace"], "additionalProperties": False})
    return {"anyOf": alts}


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def gbnf(moves: list[Move], labels, edits: bool = True) -> str:
    """llama-server `grammar`: the trained text forms (move lines verbatim, STOP: <label>, SEARCH/REPLACE)."""
    roots = (["move"] if moves else []) + ["stop"] + (["edit"] if edits else [])
    rules = ["root ::= " + " | ".join(roots)]
    if moves:
        rules.append("move ::= " + " | ".join(_lit(m.reply()) for m in moves))
    rules.append('stop ::= "STOP: " (' + " | ".join(_lit(x) for x in sorted(labels)) + ")")
    if edits:
        rules.append('edit ::= "<<<<<<< SEARCH\\n" line line* "=======\\n" line* ">>>>>>> REPLACE"')
        rules.append('line ::= [^\\n]* "\\n"')
    return "\n".join(rules) + "\n"


# ---- replies ---------------------------------------------------------------------------------------------------

def parse_reply(reply: str, allowed: str = "both"):
    """llmagent.Action for a menu-mode reply: the JSON stop and search/replace forms the schema allows, else
    llmagent.parse_action (structured JSON, SEARCH/REPLACE text, STOP text) with its usual errors."""
    from .llmagent import BLOCKERS, Action, parse_action
    text = _THINK.sub("", reply or "").strip()
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict) and "transform" not in obj:
        if set(obj) == {"stop"}:
            label = str(obj["stop"]).lower()
            return Action("stop", label=label) if label in BLOCKERS else Action(None, error=f"unknown STOP label {label!r}")
        if set(obj) == {"search", "replace"} and isinstance(obj["search"], str) and isinstance(obj["replace"], str):
            if allowed == "structured":
                return Action(None, error="a edit action, but only structured actions are allowed")
            if not obj["search"].strip():
                return Action(None, error="empty SEARCH")
            if obj["search"] == obj["replace"]:
                return Action(None, error="SEARCH and REPLACE are the same")
            return Action("edit", old=obj["search"], new=obj["replace"])
    return parse_action(reply, allowed)


def on_menu(act, moves: list[Move]) -> bool:
    """A structured action is one of the menu's moves (an int anchor, as the menu writes it)."""
    if act.kind != "structured":
        return False
    try:
        k = int(act.anchor)
    except (TypeError, ValueError):
        return False
    return Move(act.transform, k, act.pick) in set(moves)


def constrained_chat(client, model: str, messages: list[dict], options: dict, moves: list[Move], labels,
                     edits: bool = True, keep_alive: str = "15m") -> dict:
    """One reply under the menu's grammar: Ollama's /api/chat with `format` (JSON schema), or llama-server's
    /v1/chat/completions with `grammar` (GBNF); answered in Ollama's reply shape like llmtry's clients."""
    from .llmtry import OpenAICompat
    if isinstance(client, OpenAICompat):
        body = {"model": model, "messages": messages, "stream": False, "max_tokens": options.get("num_predict"),
                "repeat_penalty": client.REPEAT_PENALTY, "grammar": gbnf(moves, labels, edits),
                **{k: options[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "stop") if k in options}}
        r = client._call("/v1/chat/completions", body)
        choice = r["choices"][0]
        return {"message": {"role": "assistant", "content": choice["message"].get("content") or ""},
                "done_reason": choice.get("finish_reason"), "eval_count": r.get("usage", {}).get("completion_tokens", 0),
                "prompt_eval_count": r.get("usage", {}).get("prompt_tokens", 0)}
    body = {"model": model, "messages": messages, "stream": False, "options": options, "keep_alive": keep_alive,
            "format": json_schema(moves, labels, edits)}
    return client._call("/api/chat", body)
