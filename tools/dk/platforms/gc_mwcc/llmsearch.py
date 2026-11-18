"""Search policies for llm-try, llm-bench run and llm-grind (inference only; opt-in, the default is
llmtry.search's greedy loop: every request of a round from the best variant so far).

  --search tree --beam K   keep the top K device-free variants after each round (by score, then distinct
                           asm: two variants with the same score and diff rows are one node) and branch
                           the next round's N requests across them, round-robin from the best; the start
                           stays a node (a repair's current definition), an unwritten start (no code) only
                           while nothing compiled
  --adaptive               a function whose round brought no progress (no better device-free score)
                           stops there and banks its unused rounds' attempts; a function still improving
                           after its planned rounds draws a round from the bank (up to twice --rounds).
                           The bank is per run (bench, grind), in queue order, never borrowed ahead.

Every attempt keeps llm-try's log schema; in these modes it also carries `parent` (the attempt id
`r<round>v<k>` or `start` its prompt was built from) and `search`, and each function ends with a
`kind: search` row (rounds run, attempts, bank).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

MODES = ("greedy", "tree")


@dataclass
class Node:
    id: str                         # "start" or the attempt's "r<round>v<k>"
    code: str
    pct: float
    rows: list = field(default_factory=list)
    key: str = ""                   # what makes two variants the same node (their asm, as far as the diff shows)
    order: int = 0                  # creation order (ties keep the earlier)


def node_key(res: dict) -> str:
    """Distinct asm: the scorer's asm hash when it gives one, else the score and the diff rows (identical
    rows at the same score are the same instructions where they differ from the target)."""
    if res.get("asm_hash"):
        return str(res["asm_hash"])
    text = f"{res.get('percent', 0.0):.4f}\n" + "\n".join(res.get("rows") or [])
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def select(nodes: list[Node], k: int) -> list[Node]:
    """The beam: the top `k` nodes by score (earlier first on ties), one per distinct asm; an empty-code
    node (an unwritten start) only when no node has code."""
    with_code = [n for n in nodes if n.code]
    pool = with_code or nodes
    out, seen = [], set()
    for n in sorted(pool, key=lambda n: (-n.pct, n.order)):
        if n.key in seen:
            continue
        seen.add(n.key)
        out.append(n)
        if len(out) >= max(1, k):
            break
    return out


def allocate(beam: list[Node], n: int) -> list[Node]:
    """Request k of a round goes to beam[k % len(beam)]: the best gets ceil(n / K), every node gets
    several of the rewrite hints (they cycle with k)."""
    return [beam[k % len(beam)] for k in range(n)] if beam else []


@dataclass
class Budget:
    """The adaptive bank, shared by every function of one run."""
    bank: int = 0                   # attempts saved and not yet spent
    saved: int = 0                  # all attempts ever banked
    spent: int = 0                  # all attempts drawn for extra rounds

    def deposit(self, n: int) -> None:
        if n > 0:
            self.bank += n
            self.saved += n

    def draw(self, n: int) -> bool:
        if n <= 0 or self.bank < n:
            return False
        self.bank -= n
        self.spent += n
        return True


def tag(st) -> dict:
    """The run-row fields of a non-default search ({} for the default, so its logs stay as they were)."""
    mode = getattr(st, "search", "greedy")
    if mode == "greedy" and not getattr(st, "adaptive", False):
        return {}
    return {"search": mode, "beam": getattr(st, "beam", 1) if mode == "tree" else 1,
            "adaptive": bool(getattr(st, "adaptive", False))}


def add_args(p) -> None:
    p.add_argument("--search", choices=MODES, default="greedy", help="greedy (default): every request from the best "
                   "variant so far; tree: branch each round's requests across the top --beam variants (llmsearch.py)")
    p.add_argument("--beam", type=int, default=3, help="with --search tree: variants kept per round (default 3)")
    p.add_argument("--adaptive", action="store_true", help="stop a function after a round with no progress and bank "
                   "its unused attempts; give improving functions extra rounds from the bank (up to 2x --rounds)")


def configure_settings(st, args) -> None:
    """Copy the search options onto llm-try's Settings (a fresh bank per run)."""
    st.search = getattr(args, "search", "greedy") or "greedy"
    st.beam = max(1, int(getattr(args, "beam", 3) or 1))
    st.adaptive = bool(getattr(args, "adaptive", False))
    st.budget = Budget() if st.adaptive else None
