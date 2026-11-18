"""tools/llm_train/prune_experts.py: sizes, calibration strata, REAP scores, the tensor plan and the
checkpoint rewrite (synthetic; the rewrite test needs torch and safetensors, else it skips)."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from tools.llm_train import prune_experts as pe
except ImportError:                      # the project's tools/llm_train/, not in the toolkit repo
    raise unittest.SkipTest("tools/llm_train is not in this checkout")

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    torch = None


def stats(counts, reap):
    return {"experts": len(counts[0]), "layers": [{"count": c, "reap_sum": r, "gate_sum": [0] * len(c)}
                                                  for c, r in zip(counts, reap)]}


class SizeTest(unittest.TestCase):
    sizes = {"layers": 2, "experts": 8, "expert_bytes": 8_000_000_000, "other_bytes": 1_000_000_000,
             "kv_bytes_per_token": 100_000}

    def test_projection_scales_only_the_experts(self):
        p = pe.project(self.sizes, 4, ctx=10_000, overhead_gb=0.5)
        self.assertEqual((p["weights_gb"], p["kv_gb"], p["total_gb"], p["pruned_frac"]), (5.0, 1.0, 6.5, 0.5))

    def test_keep_from_count_fraction_or_fit(self):
        self.assertEqual(pe.keep_for(self.sizes, keep=6), 6)
        self.assertEqual(pe.keep_for(self.sizes, prune=0.25), 6)
        # 1 + k + 1 (KV) + 0.5 <= 7.6  ->  k = 5
        self.assertEqual(pe.keep_for(self.sizes, fit_gb=7.6, ctx=10_000, overhead_gb=0.5), 5)
        with self.assertRaises(ValueError):
            pe.keep_for(self.sizes, keep=9)

    def test_the_measured_30b_target_is_in_the_planned_range(self):
        k = pe.keep_for(pe.QWEN3_CODER_30B_Q4KM, fit_gb=15.0)
        self.assertTrue(0.30 <= 1 - k / 128 <= 0.40, k)


class PlanTest(unittest.TestCase):
    def test_reap_averages_over_routed_tokens_and_differs_from_frequency(self):
        s = stats([[10, 1, 0, 5]], [[10.0, 3.0, 0.0, 10.0]])
        self.assertEqual(pe.scores(s["layers"][0], "reap"), [1.0, 3.0, 0.0, 2.0])
        self.assertEqual(pe.pick(s, 2, "reap"), [[1, 3]])          # rare but strong expert 1 kept
        self.assertEqual(pe.pick(s, 2, "freq"), [[0, 3]])
        self.assertEqual(pe.pick(s, 2, "ean"), [[0, 3]])

    def test_ties_keep_the_busier_expert_and_order_is_original(self):
        s = stats([[1, 5, 5, 0], [0, 0, 0, 0]], [[1.0, 5.0, 5.0, 0.0], [0.0] * 4])
        self.assertEqual(pe.pick(s, 3), [[0, 1, 2], [0, 1, 2]])

    def test_tensor_plan(self):
        kept = [[1, 3], [0, 2]]
        self.assertEqual(pe.plan_tensor("model.layers.0.mlp.experts.3.up_proj.weight", kept),
                         ("rename", "model.layers.0.mlp.experts.1.up_proj.weight"))
        self.assertEqual(pe.plan_tensor("model.layers.0.mlp.experts.2.up_proj.weight", kept), ("drop",))
        self.assertEqual(pe.plan_tensor("model.layers.1.mlp.gate.weight", kept), ("slice0", [0, 2]))
        self.assertEqual(pe.plan_tensor("model.layers.1.mlp.experts.gate_up_proj", kept), ("slice0", [0, 2]))
        for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight",
                     "lm_head.weight", "model.layers.0.mlp.shared_expert_gate.weight"):
            self.assertEqual(pe.plan_tensor(name, kept), ("copy",), name)

    def test_config_sets_every_expert_count_key_and_refuses_too_few(self):
        self.assertEqual(pe.pruned_config({"num_experts": 128, "num_experts_per_tok": 8}, 84)["num_experts"], 84)
        c = pe.pruned_config({"num_local_experts": 8, "num_experts_per_tok": 2}, 5)
        self.assertEqual((c["num_local_experts"], "num_experts" in c), (5, False))
        with self.assertRaises(ValueError):
            pe.pruned_config({"num_experts": 8, "num_experts_per_tok": 4}, 3)


class CalibrationTest(unittest.TestCase):
    def test_every_stratum_is_sampled_in_proportion(self):
        recs = [{"tags": {"record_prompt_version": 6, "difficulty": "easy", "size": 0x20, "unit": f"e{i}"}}
                for i in range(80)]
        recs += [{"tags": {"record_prompt_version": 7, "difficulty": "hard", "size": 0x800, "unit": f"h{i}"}}
                 for i in range(18)]
        recs += [{"tags": {"task": "review", "unit": f"r{i}"}} for i in range(2)]
        got = pe.stratified(recs, 50, seed=1)
        by = {}
        for r in got:
            by[pe.stratum(r)] = by.get(pe.stratum(r), 0) + 1
        self.assertEqual(by, {("worker", 6, "easy", "<64"): 40, ("worker", 7, "hard", "1k+"): 9, ("review",): 1})
        self.assertEqual([r["tags"]["unit"] for r in got], [r["tags"]["unit"] for r in pe.stratified(recs, 50, 1)])


@unittest.skipIf(torch is None, "needs torch and safetensors")
class RewriteTest(unittest.TestCase):
    def test_pruned_checkpoint_slices_router_and_experts(self):
        with tempfile.TemporaryDirectory() as t:
            src, out = Path(t) / "src", Path(t) / "out"
            src.mkdir()
            (src / "config.json").write_text(json.dumps({"num_experts": 4, "num_experts_per_tok": 1}))
            (src / "tokenizer.json").write_text("{}")
            t1 = {"model.layers.0.mlp.gate.weight": torch.arange(8.).reshape(4, 2),
                  "model.layers.0.mlp.experts.gate_up_proj": torch.arange(4.).reshape(4, 1, 1)}
            t2 = {f"model.layers.1.mlp.experts.{j}.down_proj.weight": torch.full((1, 1), float(j)) for j in range(4)}
            t2["model.layers.1.mlp.gate.weight"] = torch.arange(4.).reshape(4, 1)
            t2["model.norm.weight"] = torch.ones(2)
            save_file(t1, str(src / "model-1.safetensors"))
            save_file(t2, str(src / "model-2.safetensors"))
            (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                **{k: "model-1.safetensors" for k in t1}, **{k: "model-2.safetensors" for k in t2}}}))
            n = pe.write_pruned(src, out, [[0, 3], [1, 2]])
            self.assertEqual(n, {"copy": 1, "drop": 2, "rename": 2, "slice0": 3})
            self.assertEqual(json.loads((out / "config.json").read_text())["num_experts"], 2)
            self.assertTrue((out / "tokenizer.json").exists())
            with safe_open(str(out / "model-1.safetensors"), "pt") as h:
                self.assertEqual(h.get_tensor("model.layers.0.mlp.gate.weight").tolist(), [[0, 1], [6, 7]])
                self.assertEqual(h.get_tensor("model.layers.0.mlp.experts.gate_up_proj").flatten().tolist(), [0, 3])
            with safe_open(str(out / "model-2.safetensors"), "pt") as h:
                self.assertEqual(h.get_tensor("model.layers.1.mlp.experts.0.down_proj.weight").item(), 1)
                self.assertEqual(h.get_tensor("model.layers.1.mlp.experts.1.down_proj.weight").item(), 2)
                self.assertNotIn("model.layers.1.mlp.experts.2.down_proj.weight", h.keys())
            idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(len(idx), 6)
            with self.assertRaises(FileExistsError):
                pe.write_pruned(src, out, [[0, 3], [1, 2]])               # never overwrites


if __name__ == "__main__":
    unittest.main()
