"""dk llm-train: the cloud LoRA run's local ends (the project's LLM plan, P5; the runbook is
tools/llm_train/README.md).

- `bundle`: the exported dataset (`dk dataset export`, leakage check passed, not upsampled: train.py
  applies the weights) and tools/llm_train in one tarball under llm-data/train-bundles/, with a
  manifest of sha256s, for scp to the rented machine. Private: it holds game-derived code. SFT only:
  train.jsonl and its card, never dpo.jsonl (DPO is a later, separate run).
- `estimate`: GPU-hours and cost from the dataset card's trained tokens (each record's tokens x its
  sample weight, source x difficulty), the epochs and a throughput, optionally scaled to a target
  record count.
- `import`: a downloaded GGUF checked against its SHA256SUMS, imported through NeuronPoke (`np
  import`), then given the chat renderer, parser, template and sampling parameters of the model it
  was trained from (`--like`), so llm-try talks to it exactly as to the base.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from ...core.command import MISMATCH, OK, USAGE, Command, hint
from ...core.config import Config
from .llmtry import DEFAULT_URL, Ollama, data_dir, llm_option

BUNDLE_FILES = ("train.py", "sftdata.py", "config.yaml", "smoke.yaml", "merge_and_convert.sh", "setup.sh",
                "requirements.txt", "README.md", "templates/qwen3-coder.jinja", "prune_experts.py")
TOP = "llm-train"                   # the bundle's top folder; [platform.llm] train_top overrides it


def top() -> str:
    return llm_option("train_top", TOP)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def card_problems(card: dict) -> list[str]:
    out = []
    if not (card.get("leakage_check") or {}).get("passed"):
        out.append("the dataset card has no passed leakage check (re-run `./dk dataset export`)")
    if card.get("upsampled"):
        out.append("the export is upsampled; train.py applies the weights itself (export without --upsample)")
    if "effective_tokens_per_epoch" not in card:
        out.append("the card has no trained-token shares (an export from before 2026-11-02's weights; re-run "
                   "`./dk dataset export`)")
    rev = (card.get("tasks") or {}).get("review")
    if rev and rev.get("effective_share", 0) > (card.get("review") or {}).get("share_cap", 1) + 1e-3:
        out.append(f"the review rows are {rev['effective_share']:.1%} of the trained tokens, over the export's cap")
    return out


def task_line(card: dict) -> str:
    """'worker N (x%), review M (y%)' from a joint export's card (dk dataset export --with-review), else ''."""
    tasks = card.get("tasks") or {}
    return ", ".join(f"{t} {v.get('records', 0):,} records ~{v.get('effective_tokens', 0):,} trained tokens "
                     f"({v.get('effective_share', 0):.1%})" for t, v in tasks.items())


def make_bundle(dataset: Path, tools: Path, out_dir: Path, head: str = "", name: str = "") -> tuple[Path, dict]:
    """Tar dataset (train.jsonl + dataset.json) and the training scripts; return (path, manifest)."""
    card = json.loads((dataset / "dataset.json").read_text())
    probs = card_problems(card)
    if probs:
        raise ValueError("; ".join(probs))
    files = {f"data/{n}": dataset / n for n in ("train.jsonl", "dataset.json")}
    files.update({f"llm_train/{n}": tools / n for n in BUNDLE_FILES if (tools / n).exists()})
    manifest = {"made": time.strftime("%Y-%m-%d %H:%M"), "head": head, "dataset": str(dataset),
                "records": card.get("records"), "tokens_per_epoch_approx": card.get("tokens_per_epoch"),
                "trained_tokens_per_epoch_approx": card.get("effective_tokens_per_epoch"),
                "generated_trained_share": card.get("generated_effective_share"), "weights": card.get("weights"),
                "prompt_version": card.get("prompt_version"), "review_prompt": card.get("review_prompt"),
                "tasks": card.get("tasks") or {"worker": {"records": card.get("records"),
                                                          "effective_tokens": card.get("effective_tokens_per_epoch")}},
                "stages": "SFT only (train.jsonl); no DPO in this bundle",
                "files": {k: sha256_file(p) for k, p in sorted(files.items())},
                "private": "Game-derived code: private instance only, delete it and its disks afterwards"}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or dataset.name}-{time.strftime('%Y%m%d-%H%M')}.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for arc, p in sorted(files.items()):
            tar.add(p, arcname=f"{top()}/{arc}")
        blob = json.dumps(manifest, indent=1).encode()
        ti = tarfile.TarInfo(f"{top()}/MANIFEST.json")
        ti.size, ti.mtime = len(blob), int(time.time())
        tar.addfile(ti, io.BytesIO(blob))
    return path, manifest


def estimate(card: dict, records_target: int, epochs: float, tps: float, real_ratio: float,
             overhead_h: float, prices: dict[str, float]) -> dict:
    """Trained tokens per epoch x epochs / throughput, plus fixed overhead (setup, the 61 GB download,
    the learning-rate sweep, merge, convert, upload/download). Trained tokens: the card's
    effective_tokens_per_epoch (every record's tokens x its sample weight, what train.py repeats), or
    for an older card the sources' tokens x their source weights."""
    src = card.get("sources") or {}
    weighted = (card.get("effective_tokens_per_epoch")
                or sum(s.get("tokens", 0) * s.get("weight", 1) for s in src.values()) or card.get("tokens_per_epoch", 0))
    scale = records_target / card["records"] if records_target and card.get("records") else 1.0
    tokens = weighted * real_ratio * scale
    train_h = tokens * epochs / tps / 3600
    hours = train_h + overhead_h
    return {"records": round(card.get("records", 0) * scale), "tokens_per_epoch": round(tokens),
            "epochs": epochs, "tokens_per_sec": tps, "train_h": round(train_h, 1), "total_h": round(hours, 1),
            "cost": {k: round(v * hours, 2) for k, v in prices.items()}}


# RunPod, 1 GPU (H200/H100 prices checked 2026-11-02; A100, L40S and Community from the 2026-11-02 listing).
# tokens/s: (planned, pessimistic) for Qwen3-Coder-30B-A3B bf16 LoRA with Unsloth (the LLM plan, P5);
# the sweep's throughput check measures the real one in the first minutes. max_seq: the longest
# sequence that fits (Unsloth: 80.9 GB up to 8K, 85.5 GB at 16K on top of the 61 GB base); 0 = doesn't fit.
GPUS = {
    "h200-secure": {"usd_h": 5.29, "tps": (4000, 2000), "max_seq": 16384,
                    "note": "141 GB, Secure Cloud (datacenter, single tenant): the plan of record"},
    "h200-community": {"usd_h": 3.59, "tps": (4000, 2000), "max_seq": 16384,
                       "note": "Community Cloud: a third party's machine; the data rules say Secure only"},
    "h100-sxm": {"usd_h": 3.99, "tps": (3800, 1900), "max_seq": 8192,
                 "note": "80 GB: 16K doesn't fit, so 8K (the long tail is left out)"},
    "a100-sxm": {"usd_h": 1.39, "tps": (1500, 900), "max_seq": 8192,
                 "note": "80 GB: 8K only; no Hopper grouped-GEMM kernels or FP8, so slower (a guess: measure first)"},
    "l40s": {"usd_h": 0.79, "tps": (0, 0), "max_seq": 0,
             "note": "48 GB: doesn't fit. MoE experts can't be 4-bit on the current stack (Unsloth: bitsandbytes "
                     "doesn't quantize their fused parameters), and the bf16 base alone is 61 GB"},
}


def gpu_rows(card: dict, gpus: list[str], epochs: list[float], records_target: int, real_ratio: float,
             overhead_h: float) -> list[dict]:
    """GPU x epochs: hours and cost at the planned and the pessimistic rate, at the longest sequence the
    GPU fits (the card's trained tokens of records up to that length)."""
    by_cap = card.get("effective_tokens_by_cap") or {}
    full = card.get("effective_tokens_per_epoch") or card.get("tokens_per_epoch", 0)
    rows = []
    for g in gpus:
        spec = GPUS[g]
        for e in epochs:
            if not spec["max_seq"]:
                rows.append({"gpu": g, "max_seq": 0, "epochs": e, "fits": False, "note": spec["note"]})
                continue
            known = str(spec["max_seq"]) in by_cap or spec["max_seq"] >= 16384
            tokens = by_cap.get(str(spec["max_seq"]), full)
            c = dict(card, effective_tokens_per_epoch=tokens)
            plan, pess = (estimate(c, records_target, e, t, real_ratio, overhead_h, {g: spec["usd_h"]})
                          for t in spec["tps"])
            rows.append({"gpu": g, "max_seq": spec["max_seq"], "epochs": e, "fits": True,
                         "kept": round(tokens / max(1, full), 3) if known else None, "records": plan["records"],
                         "tokens_per_epoch": plan["tokens_per_epoch"], "hours": (plan["total_h"], pess["total_h"]),
                         "cost": (plan["cost"][g], pess["cost"][g]), "note": spec["note"]})
    return rows


def parse_modelfile(text: str) -> dict:
    """The renderer, parser, template and parameters of an `ollama show` Modelfile, as /api/create
    fields (multi-line TEMPLATE in triple quotes included)."""
    out: dict = {"parameters": {}}
    lines, i = text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        word, _, rest = line.partition(" ")
        word = word.upper()
        if word == "TEMPLATE":
            if rest.startswith('"""'):
                body = rest[3:]
                while '"""' not in body and i + 1 < len(lines):
                    i += 1
                    body += "\n" + lines[i]
                rest = body[: body.index('"""')] if '"""' in body else body
            out["template"] = rest
        elif word in ("RENDERER", "PARSER"):
            out[word.lower()] = rest.strip()
        elif word == "PARAMETER":
            k, _, v = rest.strip().partition(" ")
            v = v.strip().strip('"')
            try:
                val: object = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
            if k == "stop":
                out["parameters"].setdefault("stop", []).append(val)
            else:
                out["parameters"][k] = val
        i += 1
    return out


def check_sums(gguf: Path, sums: Path) -> str | None:
    """None if gguf's sha256 is listed in sums, else the problem."""
    want = {}
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            want[parts[1].lstrip("*")] = parts[0]
    if gguf.name not in want:
        return f"{gguf.name} is not listed in {sums}"
    got = sha256_file(gguf)
    return None if got == want[gguf.name] else f"sha256 mismatch: {got} != {want[gguf.name]} (download again)"


def np_path() -> Path:
    return Path(os.environ.get("DK_NP", str(Path.home() / "Projects" / "NeuronPoke" / "np")))


def imports_dir() -> Path:
    """NeuronPoke imports only from here (a btrfs reflink copy costs no space)."""
    return Path(os.environ.get("DK_NP_IMPORTS", str(Path.home() / "Models" / "imports")))


# ---- actions -----------------------------------------------------------------------------------------

def bundle_cmd(cfg: Config, args: argparse.Namespace) -> int:
    data = data_dir(cfg)
    ds = Path(args.dataset) if "/" in args.dataset else data / "dataset" / args.dataset
    if not (ds / "train.jsonl").exists() or not (ds / "dataset.json").exists():
        print(f"no train.jsonl + dataset.json in {ds} (run `./dk dataset export`)")
        return USAGE
    from ...core.gitops import git
    head = git(cfg.root, "rev-parse", "HEAD").stdout.strip()
    try:
        path, manifest = make_bundle(ds, cfg.root / "tools" / "llm_train", Path(args.out) if args.out
                                     else data / "train-bundles", head)
    except ValueError as e:
        print(f"refused: {e}")
        return USAGE
    digest = sha256_file(path)
    print(f"{path}  {path.stat().st_size / 2**20:.1f} MiB  {manifest['records']} records"
          + (f" ({task_line(json.loads((ds / 'dataset.json').read_text()))})" if len(manifest["tasks"]) > 1 else ""))
    print(f"sha256 {digest}")
    print(f"upload: scp {path} root@HOST:/workspace/   (then: tar xzf {path.name} && cd {top()})")
    hint("llm-train estimate")
    return OK


def estimate_cmd(cfg: Config, args: argparse.Namespace) -> int:
    ds = Path(args.dataset) if "/" in args.dataset else data_dir(cfg) / "dataset" / args.dataset
    card = json.loads((ds / "dataset.json").read_text())
    src = card.get("sources") or {}
    if card.get("effective_tokens_per_epoch"):
        print(f"trained tokens per epoch (card): ~{card['effective_tokens_per_epoch']:,}; generated "
              f"{card.get('generated_effective_share', 0):.1%}; " + ", ".join(
                  f"{k} {v.get('effective_share', 0):.1%}" for k, v in sorted(src.items(), key=lambda kv: -kv[1].get("effective_tokens", 0)))
              + (" (shares of the worker's tokens)" if card.get("tasks") else ""))
    if card.get("tasks"):
        print(f"tasks (joint LoRA): {task_line(card)}")
    if args.price:                                    # a custom rate and prices: the old table
        prices = {k: float(v) for k, v in (p.split("=", 1) for p in args.price)}
        rows = [estimate(card, n, e, args.tps, args.real_ratio, args.overhead, prices)
                for n in [0] + list(args.records) for e in args.epochs]
        print(f"{'records':>8} {'tok/epoch':>11} {'epochs':>6} {'train h':>7} {'total h':>7}  " +
              "  ".join(f"{k:>10}" for k in prices))
        for r in rows:
            print(f"{r['records']:>8} {r['tokens_per_epoch']:>11,} {r['epochs']:>6} {r['train_h']:>7} {r['total_h']:>7}  "
                  + "  ".join(f"{'$' + format(v, '.2f'):>10}" for v in r["cost"].values()))
        print(f"(at {args.tps:.0f} tokens/s, real/approx token ratio {args.real_ratio}, +{args.overhead} h overhead)")
        return OK
    gpus = args.gpu or list(GPUS)
    for n in [0] + list(args.records):
        rows = gpu_rows(card, gpus, args.epochs, n, args.real_ratio, args.overhead)
        fit = [r for r in rows if r["fits"]]
        print(f"\n{fit[0]['records'] if fit else card.get('records')} records "
              f"({'this export' if not n else 'scaled'}); hours and cost at the planned / pessimistic rate, "
              f"+{args.overhead} h setup, sweep, merge and download")
        print("| GPU | $/h | seq | trained tokens kept | epochs | tokens/epoch | hours | cost |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if not r["fits"]:
                print(f"| {r['gpu']} | {GPUS[r['gpu']]['usd_h']} | - | - | {r['epochs']:g} | - | doesn't fit | - |")
                continue
            kept = "?" if r["kept"] is None else f"{r['kept']:.0%}"
            print(f"| {r['gpu']} | {GPUS[r['gpu']]['usd_h']} | {r['max_seq'] // 1024}k | {kept} | {r['epochs']:g} | "
                  f"{r['tokens_per_epoch'] / 1e6:.1f}M | {r['hours'][0]}-{r['hours'][1]} | ${r['cost'][0]:.0f}-${r['cost'][1]:.0f} |")
        for e in args.epochs:
            ok = [r for r in fit if r["epochs"] == e]
            long = [r for r in ok if r["max_seq"] >= 16384 and "Secure" in r["note"]]
            if ok:
                best = min(ok, key=lambda r: r["cost"][1])
                line = f"{e:g} epoch(s): lowest cost {best['gpu']} ${best['cost'][0]:.0f}-${best['cost'][1]:.0f}"
                if long and long[0] is not best:
                    line += f"; at 16k on Secure Cloud {long[0]['gpu']} ${long[0]['cost'][0]:.0f}-${long[0]['cost'][1]:.0f}"
                print(line)
        for g in gpus:
            print(f"  {g}: {GPUS[g]['note']}")
    return OK


def import_cmd(cfg: Config, args: argparse.Namespace) -> int:
    gguf = Path(args.gguf).expanduser().resolve()
    sums = Path(args.sums) if args.sums else gguf.parent / "SHA256SUMS"
    if sums.exists():
        bad = check_sums(gguf, sums)
        if bad:
            print(bad)
            return MISMATCH
        print(f"sha256 ok ({sums.name})")
    elif not args.no_verify:
        print(f"no {sums}; pass --sums FILE, or --no-verify to skip the check")
        return USAGE
    client = Ollama(args.url)
    like = parse_modelfile(client._call("/api/show", {"model": args.like}).get("modelfile", ""))
    src, extra = gguf, []
    if not gguf.is_relative_to(imports_dir()):
        src = imports_dir() / gguf.name
        subprocess.run(["cp", "--reflink=auto", str(gguf), str(src)], check=True)
        extra = ["--delete-original"]                             # the copy, never the download
    r = subprocess.run([str(np_path()), "import", str(src), "--name", args.name, "--wait", *extra,
                        "--origin", args.origin or f"{cfg.name} LoRA, {gguf.name}", "--tag", llm_option("model_tag", "dk-lora")])
    if r.returncode:
        print("np import failed")
        return r.returncode
    body = {"model": args.name, "from": args.name, "stream": False, **like}
    if not body["parameters"]:
        del body["parameters"]
    res = client._call("/api/create", body)
    print(f"{args.name}: {res.get('status', res)}; renderer {like.get('renderer')}, parser {like.get('parser')}, "
          f"parameters like {args.like}")
    hint(f"llm-bench run --model {args.name}", f"llm-bench run --set heldout --model {args.name}")
    return OK


def run(cfg: Config, args: argparse.Namespace) -> int:
    return {"bundle": bundle_cmd, "estimate": estimate_cmd, "import": import_cmd}[args.action](cfg, args)


def _configure(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("bundle", help="dataset + tools/llm_train in one tarball for the rented GPU")
    b.add_argument("--dataset", default="mixed", help="llm-data/dataset/NAME or a path (default mixed)")
    b.add_argument("--out", default="", help="directory (default llm-data/train-bundles)")
    e = sub.add_parser("estimate", help="GPU-hours and cost of a run from the dataset card")
    e.add_argument("--dataset", default="mixed")
    e.add_argument("--gpu", nargs="*", default=[], choices=list(GPUS), help="GPUs to price (default all)")
    e.add_argument("--tps", type=float, default=4000, help="with --price: training tokens/s (default 4000)")
    e.add_argument("--epochs", type=float, nargs="+", default=[1, 2])
    e.add_argument("--records", type=int, nargs="*", default=[], help="also scale to these record counts")
    e.add_argument("--real-ratio", type=float, default=1.0,
                   help="real tokenizer tokens per card token (train.py stats measures it)")
    e.add_argument("--overhead", type=float, default=2.0, help="hours of setup, download, the learning-rate sweep "
                   "(~0.5), merge, convert (default 2.0)")
    e.add_argument("--price", nargs="*", default=[], metavar="NAME=USD_PER_H", help="a custom price at --tps "
                   "instead of the GPU table")
    i = sub.add_parser("import", help="verify a downloaded GGUF, import it via NeuronPoke, copy the base's chat setup")
    i.add_argument("gguf")
    i.add_argument("--name", required=True, help="Ollama name, e.g. game-coder:lora1")
    i.add_argument("--like", default="qwen3-coder:30b", help="copy renderer/parser/template/parameters from this model")
    i.add_argument("--sums", default="", help="SHA256SUMS (default: next to the GGUF)")
    i.add_argument("--no-verify", action="store_true")
    i.add_argument("--origin", default="")
    i.add_argument("--url", default=DEFAULT_URL)


COMMAND = Command("llm-train", "matching", "cloud LoRA run: bundle the dataset and scripts, estimate the cost, "
                  "import the trained GGUF like its base", run, _configure)
