import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.dk.platforms.gc_mwcc import llmreview, llmtrain, llmtry
try:
    from tools.llm_train import sftdata, train
except ImportError:                      # the project's tools/llm_train/, not in the toolkit repo
    raise unittest.SkipTest("tools/llm_train is not in this checkout")

try:
    import jinja2.sandbox
    import regex
except ImportError:                 # the golden test needs both (system python has them)
    jinja2 = regex = None


class CharTokenizer:
    """One token per character, ChatML rendering: enough to check the loss mask and lengths."""
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def rec(unit, answer="```cpp\nint f() {}\n```", size=0x40, weight=1, source="base", prompt="asm"):
    return {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "tags": {"unit": unit, "function": f"f_{unit}", "size": size, "weight": weight, "source": source}}


def review_rec(unit, verdict="reject", rank=5, flags=True):
    """An overseer row as `dataset export --with-review` writes it (llmreview.train_row, synthetic)."""
    r = {"unit": unit, "function": f"g_{unit}", "path": "src/game/a.cpp", "size": 0x30, "asm_shape": "s",
         "asm": ".fn g, global\n    0  li r3, 0\n    4  blr\n", "decls": "int g(void);\n",
         "before_def": "", "after_def": "int g(void) {\n    volatile int x = 0;\n    return x;\n}",
         "exact": True, "before": 0.0, "after": 100.0, "flags": ["volatile"], "rid": f"r-{unit}",
         "source": "synthetic", "label_source": "construction", "verdict": verdict, "rank": rank,
         "why": "device: volatile added to a local without evidence"}
    return r, llmreview.train_row(r, 0.0 if flags else 1.0, 2.2)


class SftDataTest(unittest.TestCase):
    def test_mask_covers_the_prompt_only(self):
        tok = CharTokenizer()
        e = sftdata.encode(tok, rec("u/a"), 4096)
        prompt = tok.apply_chat_template(rec("u/a")["messages"][:-1], add_generation_prompt=True)
        self.assertTrue(all(x == sftdata.IGNORE for x in e["labels"][: len(prompt)]))
        answer = "".join(chr(x) for x in e["labels"][len(prompt):])
        self.assertEqual(answer, "```cpp\nint f() {}\n```<|im_end|>\n")     # ends like the template
        self.assertEqual(len(e["input_ids"]), len(e["labels"]))
        self.assertIsNone(sftdata.encode(tok, rec("u/a", prompt="x" * 5000), 4096))   # never truncated

    def test_template_fallback_when_full_does_not_extend_prompt(self):
        class Odd(CharTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
                text = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
                return text + "<think></think>" if add_generation_prompt else text   # only the prompt has it
        p, c, exact = sftdata.render(Odd(), rec("u/a")["messages"])
        self.assertFalse(exact)
        self.assertTrue(c.endswith("<eos>"))

    def test_eval_split_keeps_units_whole_and_is_stable(self):
        recs = [rec(f"u/{i % 40}") for i in range(400)]
        train, held = sftdata.split_by_unit(recs, 0.2)
        self.assertTrue(held)
        self.assertFalse({sftdata.unit_of(r) for r in train} & {sftdata.unit_of(r) for r in held})
        self.assertEqual(held, sftdata.split_by_unit(recs, 0.2)[1])
        self.assertEqual(sftdata.split_by_unit(recs, 0)[1], [])

    def test_weights_repeat_records_per_epoch(self):
        order = sftdata.epoch_order([1, 3, 0, 1.5], [0, 0, 0, 0], 0, 1)
        self.assertEqual(order.count(1), 3)
        self.assertEqual(order.count(2), 0)
        self.assertIn(order.count(3), (1, 2))

    def test_weight_is_the_exports_sample_weight(self):
        r = rec("u/a")
        del r["tags"]["weight"]
        self.assertEqual(sftdata.weight_of(r), 1.0)
        r["tags"]["sample_weight"] = 2.5
        self.assertEqual(sftdata.weight_of(r), 2.5)
        self.assertEqual(sftdata.weight_of(rec("u/b", weight=3)), 3.0)          # an older file's tag

    def test_subset_by_unit_is_whole_units_and_stable(self):
        recs = [rec(f"u/{i % 50}") for i in range(500)]
        sub = sftdata.subset_by_unit(recs, 0.2)
        units = {sftdata.unit_of(r) for r in sub}
        self.assertTrue(3 <= len(units) <= 20, len(units))
        self.assertEqual(len(sub), 10 * len(units))                           # every record of a picked unit
        self.assertEqual(sub, sftdata.subset_by_unit(recs, 0.2))
        self.assertEqual(sftdata.subset_by_unit(recs, 1.0), recs)
        _, held = sftdata.split_by_unit(recs, 0.2)
        self.assertNotEqual(units, {sftdata.unit_of(r) for r in held})          # its own salt, not the eval split

    def test_eval_sample_spreads_over_the_file(self):
        recs = [rec(f"u/{i}") for i in range(100)]
        got = sftdata.eval_sample(recs, 10)
        self.assertEqual([r["tags"]["unit"] for r in got], [f"u/{i}" for i in range(0, 100, 10)])
        self.assertEqual(sftdata.eval_sample(recs[:5], 10), recs[:5])

    def test_sweep_order_budget_and_pick(self):
        lengths = [10, 50, 20, 30]
        self.assertEqual(sftdata.sweep_order(lengths, [0, 2, 1, 3, 0]), [1, 0, 2, 3, 0])
        self.assertEqual(sftdata.steps_for_budget([10] * 100, 4, 125, least=2), 3)
        self.assertEqual(sftdata.steps_for_budget([10] * 100, 4, 1, least=2), 2)
        self.assertEqual(sftdata.steps_for_budget([10] * 8, 4, 10**9), 2)
        res = [{"lr": 5e-5, "eval_loss": 0.700, "first_loss": 1.0, "last_loss": 0.8},
               {"lr": 1e-4, "eval_loss": 0.695, "first_loss": 1.0, "last_loss": 0.7},
               {"lr": 2e-4, "eval_loss": 0.650, "first_loss": 1.0, "last_loss": 0.6}]
        self.assertEqual(sftdata.pick_lr(res), 2e-4)
        res[2]["eval_loss"] = 0.690                                           # within 1% of 1e-4: the lower rate
        self.assertEqual(sftdata.pick_lr(res), 5e-5 if 0.700 <= 0.690 * 1.01 else 1e-4)
        res[0]["last_loss"] = 1.2                                             # diverged: never picked
        self.assertEqual(sftdata.pick_lr(res), 1e-4)
        self.assertIsNone(sftdata.pick_lr([{"lr": 1e-4, "eval_loss": float("nan")}]))

    def test_curriculum_goes_mostly_easy_to_hard_and_epochs_differ(self):
        recs = [rec(f"u/{i}", size=16 << (i % 10)) for i in range(200)]
        order = sftdata.plan(recs, 2, 7, "curriculum", 0.35)
        self.assertEqual(len(order), 400)
        first, second = order[:200], order[200:]
        sizes = [recs[i]["tags"]["size"] for i in first]
        self.assertLess(sum(sizes[:50]), sum(sizes[-50:]))
        self.assertGreater(max(sizes[:50]), min(sizes[-50:]))      # some hard ones early
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(first), sorted(second))
        with self.assertRaises(ValueError):
            sftdata.plan(recs, 1, 7, "random", 0.1)


class LlmTrainTest(unittest.TestCase):
    def _dataset(self, d: Path, **card):
        d.mkdir(parents=True)
        (d / "train.jsonl").write_text(json.dumps(rec("u/a")) + "\n")
        base = {"records": 1, "tokens_per_epoch": 100, "leakage_check": {"passed": True}, "upsampled": False,
                "effective_tokens_per_epoch": 120}
        (d / "dataset.json").write_text(json.dumps({**base, **card}))
        return d

    def test_bundle_has_data_scripts_and_a_matching_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ds = self._dataset(t / "mixed")
            tools = t / "tools"
            tools.mkdir()
            for n in ("train.py", "sftdata.py", "config.yaml"):
                (tools / n).write_text(f"# {n}\n")
            (ds / "dpo.jsonl").write_text("{}\n")                              # the export writes it beside
            path, man = llmtrain.make_bundle(ds, tools, t / "out", head="abc")
            with tarfile.open(path) as tar:
                names = set(tar.getnames())
                self.assertIn(f"{llmtrain.top()}/data/train.jsonl", names)
                self.assertFalse([n for n in names if "dpo" in n])            # SFT only: no DPO stage
                self.assertIn("SFT only", man["stages"])
                self.assertIn(f"{llmtrain.top()}/llm_train/train.py", names)
                got = json.loads(tar.extractfile(f"{llmtrain.top()}/MANIFEST.json").read())
            self.assertEqual(got["files"]["data/train.jsonl"], llmtrain.sha256_file(ds / "train.jsonl"))
            self.assertEqual(got["head"], "abc")

    def test_bundle_refuses_unchecked_or_upsampled_exports(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(ValueError):
                llmtrain.make_bundle(self._dataset(t / "a", leakage_check={}), t, t / "out")
            with self.assertRaises(ValueError):
                llmtrain.make_bundle(self._dataset(t / "b", upsampled=True), t, t / "out")
            old = self._dataset(t / "c")
            card = json.loads((old / "dataset.json").read_text())
            del card["effective_tokens_per_epoch"]                            # before the trained-token weights
            (old / "dataset.json").write_text(json.dumps(card))
            with self.assertRaises(ValueError):
                llmtrain.make_bundle(old, t, t / "out")

    def test_joint_card_shows_both_tasks_and_its_cap_is_checked(self):
        tasks = {"worker": {"records": 10, "effective_tokens": 950, "effective_share": 0.95},
                 "review": {"records": 3, "effective_tokens": 50, "effective_share": 0.05}}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ds = self._dataset(t / "joint", tasks=tasks, review={"share_cap": 0.05}, review_prompt="r1", prompt_version=7)
            _, man = llmtrain.make_bundle(ds, t, t / "out")
            self.assertEqual((man["tasks"], man["review_prompt"], man["prompt_version"]), (tasks, "r1", 7))
            over = dict(tasks, review={**tasks["review"], "effective_share": 0.08})
            with self.assertRaises(ValueError):
                llmtrain.make_bundle(self._dataset(t / "over", tasks=over, review={"share_cap": 0.05}), t, t / "out")
            _, man = llmtrain.make_bundle(self._dataset(t / "worker"), t, t / "out")
            self.assertEqual(list(man["tasks"]), ["worker"])
        line = llmtrain.task_line({"tasks": tasks})
        self.assertIn("worker 10 records ~950 trained tokens (95.0%)", line)
        self.assertIn("review 3 records ~50 trained tokens (5.0%)", line)
        self.assertEqual(llmtrain.task_line({}), "")

    def test_estimate_weights_sources_and_scales(self):
        card = {"records": 100, "sources": {"base": {"tokens": 1_000_000, "weight": 1},
                                            "history": {"tokens": 500_000, "weight": 3}}}
        r = llmtrain.estimate(card, 0, 2, 1000, 1.0, 1.0, {"h100": 4.0})
        self.assertEqual(r["tokens_per_epoch"], 2_500_000)
        self.assertAlmostEqual(r["train_h"], 1.4, places=1)
        self.assertEqual(r["cost"]["h100"], round(4.0 * (5_000_000 / 1000 / 3600 + 1), 2))
        self.assertEqual(llmtrain.estimate(card, 400, 2, 1000, 1.0, 0, {})["tokens_per_epoch"], 10_000_000)
        card["effective_tokens_per_epoch"] = 3_000_000                       # the card's own trained tokens win
        self.assertEqual(llmtrain.estimate(card, 0, 2, 1000, 0.5, 0, {})["tokens_per_epoch"], 1_500_000)

    def test_gpu_rows_price_each_gpu_at_the_length_it_fits(self):
        card = {"records": 100, "effective_tokens_per_epoch": 10_000_000,
                "effective_tokens_by_cap": {"8192": 6_000_000, "16384": 10_000_000}}
        rows = llmtrain.gpu_rows(card, ["h200-secure", "a100-sxm", "l40s"], [1, 2], 0, 1.0, 1.0)
        h200 = next(r for r in rows if r["gpu"] == "h200-secure" and r["epochs"] == 1)
        a100 = next(r for r in rows if r["gpu"] == "a100-sxm" and r["epochs"] == 2)
        self.assertEqual((h200["tokens_per_epoch"], h200["kept"]), (10_000_000, 1.0))
        self.assertEqual((a100["tokens_per_epoch"], a100["kept"]), (6_000_000, 0.6))       # 8K only
        self.assertAlmostEqual(h200["hours"][0], round(10_000_000 / 4000 / 3600 + 1, 1))
        self.assertLess(h200["hours"][0], h200["hours"][1])                                  # planned < pessimistic
        self.assertEqual(h200["cost"][0], round(5.29 * (10_000_000 / 4000 / 3600 + 1), 2))
        self.assertFalse(next(r for r in rows if r["gpu"] == "l40s")["fits"])

    def test_parse_modelfile(self):
        mf = ("FROM /x/blobs/sha256-1\nTEMPLATE {{ .Prompt }}\nRENDERER qwen3-coder\nPARSER qwen3-coder\n"
              "PARAMETER top_k 20\nPARAMETER top_p 0.8\nPARAMETER stop <|im_start|>\nPARAMETER stop \"<|im_end|>\"\n"
              'LICENSE """Apache\nmore"""\n')
        got = llmtrain.parse_modelfile(mf)
        self.assertEqual(got["renderer"], "qwen3-coder")
        self.assertEqual(got["parser"], "qwen3-coder")
        self.assertEqual(got["template"], "{{ .Prompt }}")
        self.assertEqual(got["parameters"], {"top_k": 20, "top_p": 0.8, "stop": ["<|im_start|>", "<|im_end|>"]})
        multi = llmtrain.parse_modelfile('TEMPLATE """{{ .System }}\n{{ .Prompt }}"""\n')
        self.assertEqual(multi["template"], "{{ .System }}\n{{ .Prompt }}")

    def test_check_sums(self):
        with tempfile.TemporaryDirectory() as t:
            g = Path(t) / "m.gguf"
            g.write_bytes(b"gguf")
            sums = Path(t) / "SHA256SUMS"
            sums.write_text(f"{llmtrain.sha256_file(g)}  m.gguf\n")
            self.assertIsNone(llmtrain.check_sums(g, sums))
            sums.write_text("00  m.gguf\n")
            self.assertIn("mismatch", llmtrain.check_sums(g, sums))
            sums.write_text("00  other.gguf\n")
            self.assertIn("not listed", llmtrain.check_sums(g, sums))


class TrainPyTest(unittest.TestCase):
    """train.py's own guards (no torch: config, the format check in prepare, the merge check's picks)."""

    def test_set_values_are_typed_like_their_defaults(self):
        cfg = train.load_config("", ["learning_rate=5e-05", "epochs=1", "max_minutes=90", "keep_at=[0.5]"])
        self.assertEqual((cfg["learning_rate"], cfg["epochs"], cfg["max_minutes"]), (5e-05, 1, 90))   # YAML 1.1
        self.assertIsInstance(cfg["learning_rate"], float)                        # reads 5e-05 as a string
        self.assertEqual(cfg["keep_at"], [0.5])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.yaml"
            p.write_text("learning_rate: 5e-5\ngradient_checkpointing: false\n")
            got = train.load_config(str(p), [])
            self.assertEqual((got["learning_rate"], got["gradient_checkpointing"]), (5e-5, False))
        with self.assertRaises(SystemExit):
            train.load_config("", ["learning_rate=fast"])
        with self.assertRaises(SystemExit):
            train.load_config("", ["no_such_key=1"])

    def test_throughput_check_fits_a_short_pilot(self):
        self.assertEqual(sftdata.check_step(1304, 20), 20)                        # the real run
        self.assertEqual(sftdata.check_step(11, 20), 8)                           # the sweep: timing starts
        self.assertEqual(sftdata.check_step(2, 20), 1)                            # after step 2, so 2 + 8 <= 11

    def test_prepare_runs_the_golden_check_with_the_training_tokenizer(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "train.jsonl"
            p.write_text("".join(json.dumps(rec(f"u{i}")) + "\n" for i in range(6)))
            cfg = dict(train.DEFAULTS, dataset=str(p), eval_frac=0.0, epochs=1, max_seq_len=4096)
            info = train.prepare(cfg, CharTokenizer())["info"]
            self.assertEqual(info["golden"]["ok"], info["golden"]["records_checked"])
            self.assertFalse(info["golden"]["template_is_qwen3_coder"])

            class Drift(CharTokenizer):                         # a template that adds its own system turn
                def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
                    return super().apply_chat_template([{"role": "system", "content": "You are Qwen"}] + messages,
                                                       tokenize, add_generation_prompt)
            with self.assertRaises(SystemExit):
                train.prepare(cfg, Drift())

    def test_prepare_checks_and_counts_each_task(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "train.jsonl"
            rows = [rec(f"u{i}") for i in range(40)] + [review_rec(f"u{i}")[1] for i in range(4)]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            cfg = dict(train.DEFAULTS, dataset=str(p), eval_frac=0.0, epochs=1, max_seq_len=8192)
            data = train.prepare(cfg, CharTokenizer())
            info = data["info"]
            self.assertEqual(info["golden"]["by_task"], {"review": {"records_checked": 4, "ok": 4},
                                                         "worker": {"records_checked": 40, "ok": 40}})
            self.assertEqual((info["tasks"]["worker"]["records"], info["tasks"]["review"]["records"]), (40, 4))
            self.assertAlmostEqual(sum(v["share_of_run_tokens"] for v in info["tasks"].values()), 1.0, places=3)
            self.assertEqual(data["eval_tasks"], {})                           # no eval split here
            cfg["eval_frac"] = 0.5
            data = train.prepare(cfg, CharTokenizer())
            got = data["info"]["eval_by_task"]
            self.assertEqual(set(got), {"worker", "review"})                  # each task its own eval loss
            self.assertGreater(got["review"], 0)
            bad = [rec("u1"), {**review_rec("u2")[1], "messages": review_rec("u2")[1]["messages"][:-1]
                               + [{"role": "assistant", "content": "```cpp\nVERDICT: clean\n```"}]}]
            p.write_text("".join(json.dumps(r) + "\n" for r in bad))
            with self.assertRaises(SystemExit):                                # a review answer in the wrong form
                train.prepare(dict(cfg, eval_frac=0.0), CharTokenizer())

    def test_merge_check_needs_an_expert_tensor_on_a_moe(self):
        dense = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight"]
        self.assertEqual(train.merge_picks(dense, dense), {k: False for k in dense})
        moe = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate.weight",
               "model.layers.0.mlp.experts.0.down_proj.weight", "model.layers.0.mlp.experts.1.down_proj.weight"]
        picks = train.merge_picks(moe, moe)
        self.assertEqual(picks["model.layers.0.mlp.gate.weight"], True)                      # the router: unchanged
        self.assertIn("model.layers.0.mlp.experts.0.down_proj.weight", picks)
        fused = ["model.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.gate.weight"]   # transformers 5 in memory
        self.assertIn("model.layers.0.mlp.experts.down_proj", train.merge_picks(fused, fused))
        with self.assertRaises(ValueError):                     # a merge written fused against a per-expert base,
            train.merge_picks(fused, moe)                       # or one that lost the experts: nothing to compare
        with self.assertRaises(ValueError):
            train.merge_picks(moe[:2], moe)


QWEN_TEMPLATE =Path(sftdata.__file__).parent / "templates" / "qwen3-coder.jinja"
# Qwen2/Qwen3's pre-tokenizer (tokenizer.json): byte-level BPE merges never cross these pieces, so
# equal pieces up to a boundary mean equal token ids up to it
PRETOKEN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
            r"|\s*[\r\n]+|\s+(?!\S)|\s+")


class QwenLikeTokenizer:
    """Qwen3-Coder's own chat template (vendored; `train.py stats` checks the real tokenizer still
    has it) rendered the way transformers does, and its pre-tokenizer with the special tokens kept
    whole: a stand-in that sees every boundary where the real tokens could differ."""
    eos_token = "<|im_end|>"

    def __init__(self):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, **kw)
        self.template = env.from_string(QWEN_TEMPLATE.read_text())

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in regex.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            ids += [part] if part.startswith("<|im_") else regex.findall(PRETOKEN, part)
        return {"input_ids": ids}


@unittest.skipUnless(jinja2 and regex, "needs jinja2 and regex")
class GoldenFormatTest(unittest.TestCase):
    """Train/inference drift (the LLM plan, risk 3): a record as `dk dataset export` writes it
    (llmtry.pair), rendered for training, against what llm-try sends Ollama for the same prompt."""

    def record(self, code):
        ctx = llmtry.Context("fn_80001234", "/* 80001234 */  li r3, 0\n/* 80001238 */  blr",
                             "struct Foo {\n    int x;\n};\n", "", "")
        return llmtry.pair(ctx, code)

    def test_training_and_ollama_render_the_same_assistant_turn(self):
        tok = QwenLikeTokenizer()
        self.assertEqual(sftdata.PREFILL, llmtry.PREFILL)
        for code in ("int fn_80001234(void) {\n    return 0;\n}",
                     "void Foo::Bar(int a) {\n    if (a) {\n        x = a;\n    }\n}",
                     "    static int s;\n// note\nint f() { return s; }\n\n"):
            g = sftdata.golden(tok, self.record(code))
            self.assertTrue(g["ok"], (code, g))
        prompt, completion, _ = sftdata.render(tok, self.record("int f();")["messages"])
        self.assertTrue(prompt.endswith("<|im_end|>\n<|im_start|>assistant\n"))
        self.assertEqual(completion, "```cpp\nint f();\n```<|im_end|>\n")

    def test_options_stop_where_the_answer_ends(self):
        st = llmtry.Settings(think=False)
        self.assertEqual(st.options(1)["stop"], [sftdata.STOP])
        code = "int f() {\n    return 1;\n}"
        _, completion, _ = sftdata.render(QwenLikeTokenizer(), self.record(code)["messages"])
        reply = completion[len(sftdata.PREFILL):].split(sftdata.STOP)[0]          # what Ollama returns
        self.assertEqual(llmtry.extract_prefilled(reply, "stop")[0], code + "\n")

    def test_overseer_rows_render_as_llm_review_asks(self):
        tok = QwenLikeTokenizer()
        for verdict, rank, flags in (("reject", 5, True), ("clean", 100, False), ("needs-judgment", 60, True)):
            r, row = review_rec("u", verdict, rank, flags)
            g = sftdata.golden(tok, row)
            self.assertTrue(g["ok"], (verdict, g))
            prompt, completion, _ = sftdata.render(tok, row["messages"])
            sent = sftdata.ollama_qwen3_coder(llmreview.build_review_messages(r, flags))   # ask(): no prefill
            self.assertEqual(sent, prompt)
            self.assertEqual(completion, llmreview.format_verdict(verdict, rank, r["why"]) + "<|im_end|>\n")
            self.assertEqual(llmreview.parse_verdict(completion.split(sftdata.END)[0])["verdict"], verdict)
            enc = sftdata.encode(tok, row, 1 << 20)
            n = enc["labels"].count(sftdata.IGNORE)                     # the loss: the answer and the end only
            self.assertEqual(enc["labels"][n:], tok(completion)["input_ids"])
            self.assertEqual(enc["input_ids"][:n], tok(prompt)["input_ids"])

    def test_overseer_drift_is_caught(self):
        tok = QwenLikeTokenizer()
        _, row = review_rec("u")
        for bad in ("VERDICT: Reject\nRANK: 5\nWHY: x",                     # the parser's spelling only
                    "```cpp\nVERDICT: reject\nRANK: 5\nWHY: x\n```",       # a worker fence
                    "VERDICT: reject\nRANK: 5\nWHY: x\nmore",                 # a fourth line
                    "VERDICT: reject\nRANK: 500\nWHY: x", "VERDICT: reject\nRANK: 5\nWHY: "):
            row["messages"][-1]["content"] = bad
            self.assertFalse(sftdata.golden(tok, row)["answer_format"], bad)
        _, row = review_rec("u")
        worker_tagged = {**row, "tags": {**row["tags"], "task": "worker"}}    # a review answer as a worker row
        self.assertFalse(sftdata.golden(tok, worker_tagged)["ok"])
        self.assertTrue(sftdata.golden(tok, row)["ok"])
        for v in llmreview.VERDICTS:                       # sftdata's copy of the format (no dk in the bundle)
            for rank in (0, 9, 10, 60, 99, 100):
                self.assertTrue(sftdata.REVIEW_ANSWER.fullmatch(llmreview.format_verdict(v, rank, " a  reason ")))

    def test_drift_is_caught(self):
        tok = QwenLikeTokenizer()
        rec = self.record("int f() {}")
        rec["messages"][-1]["content"] = "```c\nint f() {}\n```"                # another fence
        self.assertFalse(sftdata.golden(tok, rec)["ok"])
        rec = self.record("int f() {}")
        rec["messages"][-1]["content"] = "cpp\nint f() {}"                       # no fence at all
        self.assertFalse(sftdata.golden(tok, rec)["ok"])
        rec = self.record("int f() {}")
        rec["messages"][0]["content"] += " "                                       # Ollama keeps it, so does Qwen
        self.assertTrue(sftdata.golden(tok, rec)["ok"])
        sysless = {"messages": rec["messages"][1:]}              # no default system text without tools
        self.assertTrue(sftdata.golden(tok, sysless)["ok"])
        rec["messages"].insert(1, {"role": "system", "content": "late"})          # Ollama hoists system messages
        self.assertFalse(sftdata.golden(tok, rec)["prompt_matches"])


class OpenAIBackendTest(unittest.TestCase):
    def test_request_and_reply_mapping(self):
        sent = []

        class Fake(llmtry.OpenAICompat):
            def _call(self, path, body=None):
                sent.append((path, body))
                return {"choices": [{"message": {"content": "int f();\n"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 400, "completion_tokens": 9},
                        "timings": {"prompt_ms": 500.0, "predicted_ms": 90.0}}

        c = Fake("http://x")
        opts = llmtry.Settings(think=False).options(7)
        r = c.chat("lora=2", [{"role": "user", "content": "q"}], opts)
        path, body = sent[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((body["max_tokens"], body["seed"], body["stop"], body["lora"]),
                         (opts["num_predict"], 7, ["```"], [{"id": 2, "scale": 1.0}]))
        self.assertEqual((r["message"]["content"], r["done_reason"], r["eval_count"], r["prompt_eval_duration"]),
                         ("int f();\n", "stop", 9, 500_000_000))
        c.chat("base", [], opts)
        self.assertEqual(sent[-1][1]["lora"], [])
        c.chat("game", [], opts)
        self.assertNotIn("lora", sent[-1][1])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(jinja2 and regex, "needs jinja2 and regex")
class GoldenRound2Test(unittest.TestCase):
    """Round 2: a prompt v8 record (annotated asm) whose repair answer opens with a DIAG line: training
    renders it as llm-try --diag prompts and parses it (prefill DIAG:, the code from the fence after)."""

    def record(self, code):
        from tools.dk.platforms.gc_mwcc import llmdata
        rec = {"unit": "u", "function": "fn_80001234", "asm": "    0  lwz r3, 0x4(r3)  ; this->m4 (int)\n    4  blr",
               "decls": "struct Foo {\n    int m4;\n};\n", "wrong": "int fn_80001234(void) {\n    int t;\n    t = 1;\n}",
               "wrong_percent": 50.0, "rows": [], "exact": code, "diff_class": "shape"}
        return llmdata.render(rec, llmdata.builder(8), None, "", "DIAG: at 0 the target has `lwz r3, 0x4(r3)` → "
                              "declare `t` where it is first assigned")

    def test_diag_answers_render_and_parse_like_inference(self):
        tok = QwenLikeTokenizer()
        self.assertEqual(sftdata.DIAG_PREFILL, llmtry.DIAG_PREFILL)
        for code in ("int fn_80001234(void) {\n    int t = 1;\n}", "void Foo::Bar(int a) {\n    x = a;\n}"):
            rec = self.record(code)
            g = sftdata.golden(tok, rec)
            self.assertTrue(g["ok"], (code, g))
            _, completion, _ = sftdata.render(tok, rec["messages"])
            reply = completion[len(sftdata.DIAG_PREFILL):].split("<|im_end|>")[0]      # what Ollama returns
            got, fmt, diag = llmtry.extract_diag(reply, "stop")
            self.assertEqual((got, fmt), (code.strip() + "\n", "diag"))
            self.assertTrue(diag.startswith("DIAG: at 0"))

    def test_diag_drift_is_caught(self):
        tok = QwenLikeTokenizer()
        rec = self.record("int f() {}")
        rec["messages"][-1]["content"] = rec["messages"][-1]["content"].replace("\n```cpp\n", " ```cpp\n")   # no line
        self.assertFalse(sftdata.golden(tok, rec)["ok"])
        plain = self.record("int f() {}")
        plain["messages"][-1]["content"] = "```cpp\nint f() {}\n```"
        self.assertTrue(sftdata.golden(tok, plain)["ok"])        # without the line: the usual prefill
