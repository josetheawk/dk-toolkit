import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tools.dk.platforms.gc_mwcc.llmtry import (Context, Ingredients, Ollama, Settings, build_messages, compact_asm,
                                               compiler_errors, declarations, devices, extract, pair, search, summarize,
                                               clean_draft, asm_shape, heldout_pick, _until, draft_function,
                                               paired, heldout_split, gate_verdict, relative_rows, shift_rows)


class FakeOllama:
    """A stdlib server speaking the two Ollama calls llm-try makes; replies come from a script."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, doc):
                body = json.dumps(doc).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send({"models": [{"name": "fake", "size": 2**30, "size_vram": 2**30}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                text = outer.replies.pop(0) if outer.replies else "no code"
                self._send({"message": {"role": "assistant", "content": text}, "eval_count": 50,
                            "eval_duration": 10**9, "prompt_eval_count": 400, "prompt_eval_duration": 10**9 // 2,
                            "load_duration": 0, "done_reason": "stop"})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def fence(code):
    return f"Here it is:\n```cpp\n{code}\n```\n"


A = "int f(int a) {\n    return a + 1;\n}"
B = "int f(int a) {\n    volatile int x = a;\n    return x + 1;\n}"
EXACT = "int f(int a) {\n    return 1 + a;\n}"
SRC = "int g;\nint f(int a) {\n    return a;\n}\n"


def fake_evaluate(code, slot):
    found = devices(SRC, SRC.replace("int f(int a) {\n    return a;\n}", code.strip()), code, "f", [])
    if "volatile" in code:
        return {"compiled": True, "percent": 100.0, "devices": found, "exact": False, "rows": [], "judge": found}
    if "1 + a" in code:
        return {"compiled": True, "percent": 100.0, "devices": found, "exact": not found, "rows": [], "text": "whole"}
    if "a + 1" in code:
        return {"compiled": True, "percent": 90.0, "devices": found, "exact": False, "rows": ["    4  addi r3, r3, 1  arg  addi r3, r4, 1"]}
    return {"compiled": False, "error": "line 2: x | undefined identifier 'x'", "devices": found}


class ExtractTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(extract(fence(A)), (A + "\n", "fenced"))
        self.assertEqual(extract("<think>maybe ```cpp\nint q() {}\n```</think>" + fence(A)), (A + "\n", "fenced"))
        self.assertEqual(extract(fence("int h();") + fence(A))[1], "fenced")       # a block with no body is skipped
        self.assertEqual(extract(fence(B) + fence(A)), (A + "\n", "several"))
        self.assertEqual(extract(A), (A + "\n", "bare"))
        self.assertEqual(extract("```cpp\nint f(int a) {\n    return"), (None, "truncated"))
        self.assertEqual(extract("<think>still going"), (None, "truncated"))
        self.assertEqual(extract("I cannot do that."), (None, "none"))

    def test_compiler_errors(self):
        out = ("### mwcceppc.exe Compiler:\n#    File: src\\a.cpp\n# ----\n#     141: if (p == nullptr) {\n"
               "#   Error:          ^^^^^^^\n#   undefined identifier 'nullptr'\n#   Too many errors printed\n")
        self.assertEqual(compiler_errors(out), "line 141: if (p == nullptr) { | undefined identifier 'nullptr'")


class PromptTest(unittest.TestCase):
    def test_compact_asm_uses_function_offsets(self):
        asm = (".fn f, global\n/* 80401000 00000100  38 63 00 01 */\taddi r3, r3, 0x1\n.L_80401004:\n"
               "/* 80401004 00000104  4E 80 00 20 */\tblr\n.endfn f\n# .text:0x8 | 0x80401008 | size: 0x4")
        self.assertEqual(compact_asm(asm), ".fn f, global\n    0  addi r3, r3, 0x1\n.L_80401004:\n    4  blr\n.endfn f")

    def test_declarations_cut_bodies(self):
        src = ("struct S { int m; int get() { return m; } };\nextern int lbl;\n"
               "void S_helper(S* s) {\n" + "    s->m = 1;\n" * 40 + "}\n"
               "int S::other() {\n    return 2;\n}\nint f(int a) {\n    return a;\n}\n")
        at = src.index("int f(")
        d = declarations(src, (at, len(src) - 1), set())
        self.assertIn("int get() { return m; }", d)            # short inline body kept
        self.assertIn("void S_helper(S* s);", d)                 # long body cut to a prototype
        self.assertNotIn("S::other", d)                          # out-of-class member dropped
        self.assertIn("(the function being matched)", d)
        self.assertNotIn("return a;", d)

    def test_trimmed_declarations_keep_only_what_the_function_uses(self):
        src = ('#include "a.h"\nstruct Used {\n    int m;\n};\nstruct Other {\n    int n;\n};\n'
               "extern int gUsed;\nextern int gOther;\nint f(Used* u) {\n    return u->m + gUsed;\n}\n")
        at = src.index("int f(")
        d = declarations(src, (at, len(src) - 1), {"Used", "gUsed"}, trim=True)
        self.assertIn("struct Used {\n    int m;\n};", d)
        self.assertIn("extern int gUsed;", d)
        self.assertIn('#include "a.h"', d)
        self.assertNotIn("Other", d)
        self.assertNotIn("gOther", d)
        self.assertIn("gOther", declarations(src, (at, len(src) - 1), {"Used"}))      # untrimmed under budget
        self.assertTrue(declarations(src, (at, len(src) - 1), {"nothing"}, trim=True).strip())  # never empty

    def test_the_draft_gets_types_the_unit_can_see(self):
        draft = "s32 f(u8 *p, ? *q) {\n    if (p == NULL) return 0;\n    return (f32) *p;\n}"
        self.assertEqual(clean_draft(draft, "int x;"),
                         "int f(unsigned char *p, int *q) {\n    if (p == 0) return 0;\n    return (float) *p;\n}")
        seen = clean_draft(draft, "typedef signed long s32;\n#define NULL 0\n")
        self.assertTrue(seen.startswith("s32 f(unsigned char *p"))
        self.assertIn("p == NULL", seen)

    def test_the_draft_is_the_function_alone_in_cpp_names(self):
        draft = ("typedef struct Bomb {\n    int unk4;\n} Bomb;\n\nBoom *__dt__4BombFv(short x); /* static */\n"
                 "/* Bomb::~Bomb (void) */\nBoom *__dt__4BombFv(Bomb *this, short destroyFlag) {\n"
                 "    fn_8061F2EC__3WndFv(this->unk4);\n}")
        self.assertEqual(draft_function(draft, "__dt__4BombFv"), "Bomb::~Bomb() {\n    Wnd::fn_8061F2EC(this->unk4);\n}")
        self.assertEqual(draft_function("int fn_1__3FooFi(Foo *this, int a) {\n    return a;\n}", "fn_1__3FooFi"),
                         "int Foo::fn_1(int a) {\n    return a;\n}")
        self.assertEqual(draft_function("void fn_80401000(void) {\n}", "fn_80401000"), "void fn_80401000(void) {\n}")
        self.assertEqual(draft_function("int fn_1__3FooFi(Foo *this, int a) {\n    return fn_2__3WndFv(g) + fn_3__3FooFi(this, a);\n}",
                                        "fn_1__3FooFi"), "int Foo::fn_1(int a) {\n    return g->fn_2() + this->fn_3(a);\n}")

    def test_messages_hold_the_ingredients(self):
        ctx = Context("f", "    0  blr", "int g;\n", "int f(int a) { return 0; }", "")
        msgs = build_messages(ctx, "", 0.0, [], Ingredients(), True)
        user = msgs[1]["content"]
        self.assertIn("Target assembly", user)
        self.assertIn("Machine draft", user)                       # unwritten: m2c shown
        self.assertIn("volatile", msgs[0]["content"])              # the rules
        ctx.current = A
        user = build_messages(ctx, A, 90.0, ["row"], Ingredients(asm=False), True)[1]["content"]
        self.assertNotIn("Target assembly", user)
        self.assertNotIn("Machine draft", user)                    # written: m2c only with --m2c always
        self.assertIn("scores 90.00%", user)
        self.assertIn("row", user)


class PairTest(unittest.TestCase):
    def test_a_training_pair_is_the_unwritten_prompt_and_the_exact_source(self):
        rec = pair(Context("f", "    0  blr", "int g;\n", "draft", A), A)
        roles = [m["role"] for m in rec["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        user = rec["messages"][1]["content"]
        self.assertIn("Target assembly", user)
        self.assertNotIn("draft", user)                  # no m2c, no current source: the model must write it
        self.assertNotIn("return a + 1", user)
        self.assertEqual(rec["messages"][2]["content"], "```cpp\n" + A + "\n```")


class GateTest(unittest.TestCase):
    def test_devices(self):
        after = SRC.replace("return a;", "volatile int x = a;\n    return x;")
        self.assertEqual(devices(SRC, after, "", "f", []), ["adds 1 volatile"])
        code = "static int helper(int a) {\n    return a;\n}\nint f(int a) {\n    return helper(a);\n}\n"
        self.assertEqual(devices(SRC, SRC, code, "f", []), ["adds function helper"])
        self.assertEqual(devices(SRC, SRC.replace("#", ""), A, "f", []), [])
        mangled = "int f__Fi(int a) {\n    return a;\n}\n"
        self.assertEqual(devices(SRC, SRC, mangled, "f__Fi", []), ["writes the mangled name f__Fi"])


class SearchTest(unittest.TestCase):
    def test_rounds_feed_back_the_best_and_stop_at_an_exact_one(self):
        fake = FakeOllama([fence(A), fence(B), fence(A), "no code here",      # round 0
                           fence(EXACT), fence(A), fence(A), fence(A)])        # round 1
        rows = []
        try:
            st = Settings(model="fake", n=4, rounds=3, think=False, num_ctx=8192, prefill=False)
            ctx = Context("f", "    0  blr", "int g;\n", "", "int f(int a) {\n    return a;\n}\n")
            out = search(ctx, ctx.current, 50.0, [], fake_evaluate, Ollama(fake.url), st, rows.append, jobs=2,
                         say=lambda *_: None)
        finally:
            fake.close()
        att = [r for r in rows if r["kind"] == "attempt"]
        self.assertEqual(len(att), 8)                              # stopped after round 1 (exact found)
        self.assertEqual(out.exact_code.strip(), EXACT)
        self.assertEqual(out.best, 100.0)
        self.assertEqual(att[1]["devices"], ["adds 1 volatile"])   # the 100% volatile one is not the best
        self.assertFalse(att[1]["exact"])
        self.assertEqual(att[2]["duplicate_of"], 0)
        self.assertEqual(att[3]["format"], "none")
        self.assertEqual(att[0]["delta"], 40.0)
        self.assertEqual(att[0]["gen_tps"], 50.0)
        self.assertEqual(att[0]["prompt_tokens"], 400)
        req = fake.requests
        self.assertEqual(req[0]["options"]["num_ctx"], 8192)
        self.assertIs(req[0]["think"], False)
        self.assertEqual([r["options"]["seed"] for r in req[:2]], [1, 2])
        self.assertEqual(req[4]["options"]["seed"], 1001)
        self.assertIn("scores 50.00%", req[0]["messages"][1]["content"])
        self.assertIn("Best attempt so far (scores 90.00%)", req[4]["messages"][1]["content"])   # A fed back
        self.assertIn("return a + 1;", req[4]["messages"][1]["content"])
        self.assertNotIn("Try", req[0]["messages"][1]["content"])          # variant 0: no hint
        self.assertIn("Try", req[1]["messages"][1]["content"])             # variant 1: a rewrite hint
        prompts = [r for r in rows if r["kind"] == "prompt"]
        self.assertEqual([p["round"] for p in prompts], [0, 1])

    def test_prefill_opens_the_fence_and_stops_at_the_closing_one(self):
        fake = FakeOllama([A + "\n", "int f(int a) {\n    return"])           # the server stopped at ```
        rows = []
        try:
            st = Settings(model="fake", n=2, rounds=1, think=False)
            ctx = Context("f", "    0  blr", "", "", "int f(int a) {\n    return a;\n}\n")
            search(ctx, ctx.current, 50.0, [], fake_evaluate, Ollama(fake.url), st, rows.append, say=lambda *_: None)
        finally:
            fake.close()
        req = fake.requests[0]
        self.assertEqual(req["messages"][-1], {"role": "assistant", "content": "```cpp\n"})
        self.assertEqual(req["options"]["stop"], ["```"])
        att = [r for r in rows if r["kind"] == "attempt"]
        self.assertEqual((att[0]["format"], att[0]["percent"]), ("prefill", 90.0))
        thinking = Settings(think=True)
        self.assertFalse(thinking.prefilled())                   # the answer comes after the thinking
        self.assertNotIn("stop", thinking.options(1))

    def test_a_template_that_wants_its_think_block_gets_it_in_the_prefill(self):
        fake = FakeOllama(["</think>\n\n", A + "\n", A.replace("1", "2") + "\n"])
        rows = []
        try:
            st = Settings(model="fake", n=2, rounds=1, think=False)
            ctx = Context("f", "    0  blr", "", "", "int f(int a) {\n    return a;\n}\n")
            search(ctx, ctx.current, 50.0, [], fake_evaluate, Ollama(fake.url), st, rows.append, say=lambda *_: None)
        finally:
            fake.close()
        self.assertEqual(len(fake.requests), 3)                  # the first variant asked again, once
        self.assertEqual(fake.requests[1]["messages"][-1]["content"], "<think>\n\n</think>\n\n```cpp\n")
        self.assertEqual(fake.requests[2]["messages"][-1]["content"], "<think>\n\n</think>\n\n```cpp\n")
        att = [r for r in rows if r["kind"] == "attempt"]
        self.assertTrue(att[0]["prefill_think_block"])
        self.assertEqual(att[0]["percent"], 90.0)

    def test_a_failed_compile_is_shown_when_nothing_compiled(self):
        fake = FakeOllama([fence("int f(int a) {\n    return x;\n}"), fence(A)])
        try:
            st = Settings(model="fake", n=1, rounds=2, think=None, prefill=False)
            ctx = Context("f", "    0  blr", "", "int f(int a) { return 0; }", "")
            search(ctx, "", 0.0, [], fake_evaluate, Ollama(fake.url), st, lambda r: None, say=lambda *_: None)
        finally:
            fake.close()
        self.assertNotIn("think", fake.requests[0])
        second = fake.requests[1]["messages"][1]["content"]
        self.assertIn("did not compile", second)
        self.assertIn("undefined identifier 'x'", second)

    def test_summary(self):
        rows = [{"kind": "run", "model": "m", "think": False, "ingredients": "m2c=auto", "n": 2, "rounds": 1},
                {"kind": "attempt", "format": "fenced", "variant": "x", "compiled": True, "devices": [], "secs": 2,
                 "gen_tps": 10, "prompt_tokens": 300, "gen_tokens": 40},
                {"kind": "attempt", "format": "none", "variant": None, "compiled": False, "secs": 4, "gen_tps": 20,
                 "prompt_tokens": 300, "gen_tokens": 60},
                {"kind": "function", "bench_kind": "register", "start": 98.0, "best": 99.0, "exact": False},
                {"kind": "function", "bench_kind": "unwritten", "start": 0.0, "best": 100.0, "exact": True}]
        s = summarize(rows)
        self.assertEqual((s["format_ok"], s["compile"], s["devices"]), (50.0, 50.0, 0.0))
        self.assertEqual((s["improved"], s["mean_gain_near"], s["exact_unwritten"]), ("1/1", 1.0, "1/1"))
        self.assertEqual(s["s_per_variant"], 3.0)

    def test_paired(self):
        def run(results):
            return [{"kind": "function", "function": f, "best": b, "exact": b == 100.0} for f, b in results]
        base = run([("a", 100.0), ("b", 50.0), ("c", 20.0), ("d", 10.0)])
        new = run([("a", 100.0), ("b", 100.0), ("c", 100.0), ("d", 5.0), ("e", 100.0)])     # e: not in both
        first, second = paired([base, new])
        self.assertEqual((first["paired"], first["exact"], first["sign_p"], first["best_diff"]), (4, 1, 1.0, 0.0))
        self.assertEqual((second["exact"], second["exact_only_here"], second["exact_only_first"]), (3, 2, 0))
        self.assertEqual((second["best_up"], second["best_down"], second["best_diff"], second["sign_p"]), (2, 1, 31.25, 0.5))

    def test_paired_compile_counts_skip_interrupted_attempts(self):
        def run(compiled, interrupted=0):
            rows = []
            for f, n in compiled.items():
                rows += [{"kind": "prompt", "function": f, "round": 0}] + \
                    [{"kind": "attempt", "function": f, "compiled": True}] * interrupted     # killed, then resumed
                rows += [{"kind": "prompt", "function": f, "round": 0}] + \
                    [{"kind": "attempt", "function": f, "compiled": True}] * n + \
                    [{"kind": "attempt", "function": f, "compiled": False}] * (16 - n) + \
                    [{"kind": "function", "function": f, "best": 50.0, "exact": False}]
            return rows
        first, second = paired([run({"a": 2, "b": 5, "c": 3}), run({"a": 4, "b": 5, "c": 1}, interrupted=9)])
        self.assertEqual((second["compile_up"], second["compile_down"], second["compile_p"]), (1, 1, 1.0))
        self.assertEqual((first["compile_up"], first["compile_down"]), (0, 0))


class SplitAndGateTest(unittest.TestCase):
    def test_split_dev_every_third_small_final_rest_minus_used(self):
        fns = [{"function": f"f{i}", "size": 0x40 * (1 + i % 12), "area": f"a{i % 3}"} for i in range(60)]
        sp = heldout_split(fns, used={"f1", "f2"})
        small = [r for r in fns if r["size"] <= 0x200]
        self.assertEqual(len(sp["dev"]), -(-len(small) // 3))
        self.assertTrue(all(next(r for r in fns if r["function"] == f)["size"] <= 0x200 for f in sp["dev"]))
        self.assertFalse(set(sp["dev"]) & set(sp["final"]))
        self.assertFalse({"f1", "f2"} & set(sp["final"]))
        self.assertEqual(set(sp["excluded"]), {"f1", "f2"} - set(sp["dev"]))
        rest = {r["function"] for r in fns if r["size"] <= 0x300} - set(sp["dev"]) - {"f1", "f2"}
        self.assertEqual(set(sp["final"]), rest)
        self.assertEqual(sp, heldout_split(list(reversed(fns)), used={"f1", "f2"}))     # stable

    @staticmethod
    def log(exact, seed=1, devices=0, n=8, skip=(), rid=False):
        rows = [{"kind": "run", "n": n, "rounds": 2, "prompt_version": 6, "examples": 4, "options": {"seed": seed}}]
        for i, (f, x) in enumerate(exact.items()):
            fn = f.split("#")[0]
            rows += [{"kind": "attempt", "function": fn, "variant": "v", "compiled": True,
                      "devices": ["volatile"] if i < devices else []},
                     {"kind": "function", "function": fn, "exact": x, "best": 100.0 if x else 50.0,
                      **({"rid": f} if rid else {})}]
        rows += [{"kind": "skip", "function": f} for f in skip]
        return rows

    def test_gate_pass_and_each_failure(self):
        final = {f"f{i}" for i in range(10)}
        rep = {"r0#0", "r0#1", "r1#2"}
        base = {f: i < 2 for i, f in enumerate(sorted(final))}                  # 2 exact
        cand = {f: i < 4 for i, f in enumerate(sorted(final))}                  # 4 exact, 2 only here
        rb = {k: k == "r0#0" for k in rep}
        rc = {k: k in ("r0#0", "r1#2") for k in rep}
        args = lambda c=cand, b=base, r_c=rc, dev=0: ([self.log(b)], [self.log(c, devices=dev)], final,
                                                     [self.log(rb, rid=True)], [self.log(r_c, rid=True)], rep)
        v = gate_verdict(*args())
        self.assertTrue(v["pass"], v["checks"])
        four = {f: i < 4 for i, f in enumerate(sorted(final))}
        five = {f: i < 5 for i, f in enumerate(sorted(final))}                  # favored 1-0, but 5 < 1.5 x 4
        v = gate_verdict(*args(c=five, b=four))
        self.assertFalse(v["pass"])
        self.assertEqual([n for n, ok, _w in v["checks"] if not ok], ["exact >= 1.5x baseline"])
        v = gate_verdict(*args(), alpha=0.1)                                     # 2-0 favors it, but p 0.5
        self.assertEqual([n for n, ok, _w in v["checks"] if not ok], ["sign test favors the candidate"])
        self.assertFalse(gate_verdict(*args(dev=3))["pass"])                     # more devices
        self.assertFalse(gate_verdict(*args(r_c={k: False for k in rep}))["pass"])   # repair down
        missing = dict(list(cand.items())[:9])
        v = gate_verdict(*args(c=missing))
        self.assertFalse(v["pass"])
        self.assertIn(("held-out complete", False, "paired 9/10"), v["checks"])
        v = gate_verdict([self.log(base)], [self.log(cand, seed=2)], final, [self.log(rb, rid=True)],
                         [self.log(rc, rid=True)], rep)
        self.assertFalse(dict((n, ok) for n, ok, _w in v["checks"])["held-out comparable"])
        self.assertFalse(gate_verdict([self.log(base)], [self.log(cand)], final, [], [], rep)["pass"])   # no repair logs

    def test_gate_skips_what_both_arms_skipped(self):
        final = {"a", "b", "c"}
        b = self.log({"a": False, "b": False}, skip=["c"])
        c = self.log({"a": True, "b": True}, skip=["c"])
        v = gate_verdict([b], [c], final, [b], [c], {"a"})
        self.assertIn(("held-out complete", True, "paired 2/2"), v["checks"])

    def test_paired_keys_repairs_by_rid(self):
        base = self.log({"f#0": False, "f#1": True}, rid=True)
        new = self.log({"f#0": True, "f#1": True}, rid=True)
        self.assertEqual((paired([base, new])[1]["paired"], paired([base, new])[1]["exact_only_here"]), (2, 1))


class HeldOutTest(unittest.TestCase):
    def test_asm_shape_ignores_registers_immediates_and_symbols(self):
        a = ("/* 80401000 00000100  38 63 00 01 */\taddi r3, r3, 0x1\n"
             "/* 80401004 00000104  48 00 00 01 */\tbl fn_80402000\n/* 80401008 00000108  4E 80 00 20 */\tblr\n")
        b = a.replace("r3, r3, 0x1", "r4, r4, 0x8").replace("fn_80402000", "Foo__3BarFv").replace("80001", "80005")
        self.assertEqual(asm_shape(a), asm_shape(b))
        self.assertNotEqual(asm_shape(a), asm_shape(a.replace("addi", "subi")))

    def test_the_pick_is_stable_and_about_one_in_twenty(self):
        names = [f"main/game/x/U{i}" for i in range(4000)]
        share = sum(heldout_pick(n) for n in names) / len(names)
        self.assertTrue(0.035 < share < 0.065, share)
        self.assertEqual([heldout_pick(n) for n in names[:50]], [heldout_pick(n) for n in names[:50]])

    def test_until_is_the_next_such_time(self):
        import time
        t = _until("07:00")
        self.assertTrue(time.time() < t <= time.time() + 86400)
        self.assertEqual(time.localtime(t).tm_hour, 7)
        self.assertEqual(_until(""), 0.0)


def _insn(text, address=None, kind=None):
    """One row of objdiff's JSON: a zero address is left out, as objdiff does."""
    row = {"instruction": {"formatted": text}}
    if address is not None:
        row["instruction"]["address"] = str(address)
    if kind:
        row["diff_kind"] = kind
    return row


class RelativeRowsTest(unittest.TestCase):
    ARG = "DIFF_ARG_MISMATCH"

    def pair(self, at: int = 0):
        """A function at section offset `at` (objdiff's JSON has no address for offset 0), `beq` to its last word."""
        a = lambda n: None if at + n == 0 else at + n
        left = {"instructions": [_insn("li r3, 0x0", a(0), self.ARG), _insn("cmpwi r3, 0x0", a(4)),
                                 _insn(f"beq 0x{at + 16:x}", a(8)), _insn("li r4, 0x1", a(12), self.ARG), _insn("blr", a(16))]}
        right = {"instructions": [_insn("li r5, 0x0", a(0), self.ARG), _insn("cmpwi r3, 0x0", a(4)),
                                  _insn(f"beq 0x{at + 16:x}", a(8)), _insn("li r4, 0x2", a(12), self.ARG), _insn("blr", a(16))]}
        return left, right

    def test_a_function_at_offset_zero_has_its_true_offsets(self):
        rows = relative_rows(*self.pair(0))
        self.assertEqual([r[:5] for r in rows[1:]], ["    0", "    4", "    8", "    C", "   10"])
        self.assertIn("beq 10", rows[3])                          # the branch target, relative to the start
        self.assertEqual(rows[1].split()[:2], ["0", "li"])

    def test_the_offsets_do_not_depend_on_where_the_function_is(self):
        self.assertEqual(relative_rows(*self.pair(0)), relative_rows(*self.pair(0x120)))

    def test_a_row_without_a_target_instruction_has_no_offset(self):
        left, right = self.pair(0)
        left["instructions"][3] = {"diff_kind": "DIFF_INSERT"}
        rows = relative_rows(left, right)
        self.assertTrue(rows[4].startswith("     "))
        self.assertEqual(rows[1][:5], "    0")

    def test_shift_rows_puts_the_rows_made_before_the_fix_right(self):
        old = ["  ...",                                           # relative_rows before the fix, the function at offset 0
               "       li r3, 0x0                           arg  li r5, 0x0",
               "    0  cmpwi r3, 0x0                             cmpwi r3, 0x0",
               "    4  beq C                                     beq C",
               "    8  li r4, 0x1                           arg  li r4, 0x2",
               "    C  blr                                       blr", "  … (more differences)"]
        self.assertEqual(shift_rows(old), relative_rows(*self.pair(0)) + ["  … (more differences)"])
        self.assertEqual(shift_rows(old, False, False), old)       # parses back to itself
        self.assertIsNone(shift_rows(["junk"]))
        self.assertIsNone(shift_rows(["    4  nop    but not a row of ours"]))


if __name__ == "__main__":
    unittest.main()
