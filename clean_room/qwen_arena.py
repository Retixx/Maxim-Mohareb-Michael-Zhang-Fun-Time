#!/usr/bin/env python3
"""
Clean-room Qwen3 vs Qwen2.5 arena.

This file deliberately imports NOTHING from this repository. That is the whole
point: if the Qwen3 regression reproduces here, it is the model or the models'
interaction with the hardware. If it does not reproduce here, it was the
harness. There is no third option, because nothing in src/ is on the import
path.

Three tiers, run in order. Tier 0 is model-agnostic and must pass before any
model comparison is meaningful.

  Tier 0  harness self-test
          - decoder-only left-padding assertion
          - batch-size invariance under greedy decoding
          - thinking-mode suppression actually took effect
          - no NaN/inf in logits, no silent truncation
  Tier 1  schema adherence (can the model emit the JSON the pipeline needs)
  Tier 2  task accuracy: HotpotQA token F1 / EM, paired, with bootstrap CI
          and McNemar

Retrieval is removed as a variable on purpose. Tier 2 uses the HotpotQA
`distractor` config, which ships 10 paragraphs per question (2 gold, 8
distractors), and hands all 10 to every model. This is a pure reading-
comprehension comparison. It answers "is Qwen3 better than Qwen2.5 at the job
the pipeline asks of it", and nothing about BM25.

Usage
-----
    python qwen_arena.py --tier 0                      # harness only, ~2 min
    python qwen_arena.py --tier all --n 300            # full local run
    python qwen_arena.py --models qwen3-0.6b-fp16,qwen2.5-0.5b-fp16 --n 200

On a 4GB card, stick to the 0.6B/0.5B pairs at fp16 or anything at 4bit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SEED = 20260807
MAX_NEW_TOKENS = 64
MAX_PROMPT_TOKENS = 3072


# --------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    precision: str          # fp16 | bf16 | 8bit | 4bit
    family: str             # qwen2.5 | qwen3
    params_b: float

    @property
    def approx_vram_gb(self) -> float:
        per_param = {"fp16": 2.0, "bf16": 2.0, "8bit": 1.0, "4bit": 0.55}[self.precision]
        return self.params_b * per_param


def _registry() -> dict[str, ModelSpec]:
    pairs = [
        ("qwen2.5", "Qwen/Qwen2.5-0.5B-Instruct", 0.5),
        ("qwen2.5", "Qwen/Qwen2.5-1.5B-Instruct", 1.5),
        ("qwen2.5", "Qwen/Qwen2.5-3B-Instruct", 3.0),
        ("qwen2.5", "Qwen/Qwen2.5-7B-Instruct", 7.0),
        ("qwen2.5", "Qwen/Qwen2.5-14B-Instruct", 14.0),
        ("qwen3", "Qwen/Qwen3-0.6B", 0.6),
        ("qwen3", "Qwen/Qwen3-1.7B", 1.7),
        ("qwen3", "Qwen/Qwen3-4B", 4.0),
        ("qwen3", "Qwen/Qwen3-8B", 8.0),
        ("qwen3", "Qwen/Qwen3-14B", 14.0),
    ]
    out: dict[str, ModelSpec] = {}
    for family, repo, params in pairs:
        size = repo.split("-")[1] if family == "qwen3" else repo.split("-")[1]
        for precision in ("fp16", "bf16", "8bit", "4bit"):
            key = f"{family}-{size.lower()}-{precision}"
            out[key] = ModelSpec(key, repo, precision, family, params)
    return out


REGISTRY = _registry()

# Matched-size pairs. Qwen3 has no 0.5B and Qwen2.5 has no 0.6B/1.7B, so the
# comparison is nearest-neighbour by parameter count. State this in any writeup;
# Qwen3-0.6B has ~20% more parameters than Qwen2.5-0.5B and that is not nothing
# at this scale.
MATCHED_PAIRS = [
    ("qwen2.5-0.5b", "qwen3-0.6b"),
    ("qwen2.5-1.5b", "qwen3-1.7b"),
    ("qwen2.5-3b", "qwen3-4b"),
    ("qwen2.5-7b", "qwen3-8b"),
    ("qwen2.5-14b", "qwen3-14b"),
]


# --------------------------------------------------------------------------
# HotpotQA metrics -- canonical implementation, transcribed from the official
# evaluation script so the numbers are comparable to published results
# --------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    # yes/no/noanswer must match exactly under the official script
    if normalize_answer(pred) in {"yes", "no", "noanswer"} or normalize_answer(gold) in {"yes", "no", "noanswer"}:
        return float(normalize_answer(pred) == normalize_answer(gold))
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


# --------------------------------------------------------------------------
# the single generation path -- every model in every tier goes through this
# --------------------------------------------------------------------------

@dataclass
class GenRecord:
    text: str
    prompt_tokens: int
    output_tokens: int
    truncated: bool
    has_think_tag: bool
    nonfinite_logits: bool


class Runner:
    """One model, loaded. The only family-conditional logic in this class is
    the enable_thinking probe, and it is a capability probe rather than a
    hardcoded branch on the model name."""

    def __init__(self, spec: ModelSpec, device: str = "cuda", verbose: bool = True):
        self.spec = spec
        self.device = device
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation is silently wrong with right padding.
        # This is the single most common cause of "model X is mysteriously bad".
        assert self.tokenizer.padding_side == "left", "decoder-only models require left padding"

        kwargs: dict[str, Any] = {"device_map": device if device != "cpu" else None}
        if spec.precision in ("fp16", "bf16"):
            dtype = torch.float16 if spec.precision == "fp16" else torch.bfloat16
            kwargs["dtype"] = dtype
        elif spec.precision == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif spec.precision == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.repo, **kwargs)
        except TypeError:
            # older transformers wants torch_dtype
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(spec.repo, **kwargs)
        self.model.eval()

        self.supports_enable_thinking = self._probe_enable_thinking()
        if verbose:
            print(f"    loaded {spec.key:24s} enable_thinking_supported={self.supports_enable_thinking}")

    def _probe_enable_thinking(self) -> bool:
        msgs = [{"role": "user", "content": "hi"}]
        try:
            with_flag = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return False
        without = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # If the flag changes nothing the template is ignoring it (Qwen2.5).
        return with_flag != without

    def build_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.supports_enable_thinking:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts: list[str], batch_size: int = 8,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> list[GenRecord]:
        out: list[GenRecord] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_PROMPT_TOKENS,
            ).to(self.model.device)

            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # greedy: no sampling noise anywhere
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            seqs = gen.sequences
            in_len = enc["input_ids"].shape[1]
            new_tokens = seqs[:, in_len:]

            nonfinite = False
            if gen.scores:
                stacked = torch.stack(gen.scores, dim=0)
                nonfinite = bool((~torch.isfinite(stacked)).any().item())

            for row, new in zip(range(len(chunk)), new_tokens):
                text = self.tokenizer.decode(new, skip_special_tokens=True)
                n_new = int((new != self.tokenizer.pad_token_id).sum().item())
                out.append(GenRecord(
                    text=text,
                    prompt_tokens=int(enc["attention_mask"][row].sum().item()),
                    output_tokens=n_new,
                    truncated=n_new >= max_new_tokens,
                    has_think_tag=("<think>" in text or "</think>" in text),
                    nonfinite_logits=nonfinite,
                ))
        return out

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_hotpot(n: int, seed: int = SEED) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]

    rows = []
    for i in idx:
        r = ds[i]
        ctx = r["context"]
        titles, sents = ctx["title"], ctx["sentences"]
        paragraphs = [f"[{t}] " + " ".join(s) for t, s in zip(titles, sents)]
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "level": r.get("level", "?"),
            "type": r.get("type", "?"),
            "paragraphs": paragraphs,
        })
    return rows


ANSWER_SYSTEM = (
    "You answer questions from provided passages. "
    "Reply with JSON only, exactly: {\"answer\": \"...\"}. "
    "The answer must be the shortest exact span that answers the question, "
    "usually one to four words. Do not explain."
)


def answer_user_prompt(row: dict[str, Any]) -> str:
    body = "\n".join(row["paragraphs"])
    return f"Passages:\n{body}\n\nQuestion: {row['question']}\n\nJSON:"


def parse_answer(text: str) -> tuple[str, str]:
    """Returns (answer, parse_status). Lenient on purpose -- we want to measure
    model knowledge, not JSON pedantry, but we still record how often strict
    parsing would have failed."""
    t = text.strip()
    m = re.search(r'\{.*?"answer"\s*:\s*"(.*?)".*?\}', t, re.DOTALL)
    if m:
        return m.group(1).strip(), "ok"
    m = re.search(r'"answer"\s*:\s*"(.*?)"', t, re.DOTALL)
    if m:
        return m.group(1).strip(), "loose_json"
    if not t:
        return "", "empty"
    # last resort: first line, stripped of fencing
    first = t.split("\n")[0].strip().strip("`").strip()
    return first, "fallback_text"


# --------------------------------------------------------------------------
# Tier 0 -- harness self-test
# --------------------------------------------------------------------------

def tier0(spec: ModelSpec, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    print(f"\n[tier0] {spec.key}")
    r = Runner(spec, device=device)
    probe_rows = rows[:16]
    prompts = [r.build_prompt(ANSWER_SYSTEM, answer_user_prompt(x)) for x in probe_rows]

    t = time.time()
    b1 = r.generate(prompts, batch_size=1)
    b8 = r.generate(prompts, batch_size=8)
    elapsed = time.time() - t

    identical = sum(1 for a, b in zip(b1, b8) if a.text == b.text)
    agree_rate = identical / len(prompts)

    think_leak = sum(1 for g in b1 if g.has_think_tag)
    trunc = sum(1 for g in b1 if g.truncated)
    nonfinite = any(g.nonfinite_logits for g in b1 + b8)
    empty = sum(1 for g in b1 if not g.text.strip())

    # Greedy decoding is mathematically batch-invariant. Floating point
    # reduction order is not, so a healthy harness lands high but not always
    # at 1.00. A padding-side bug collapses this number.
    verdict = "PASS" if agree_rate >= 0.90 else "FAIL"

    res = {
        "model": spec.key,
        "batch_invariance": round(agree_rate, 3),
        "batch_invariance_verdict": verdict,
        "enable_thinking_supported": r.supports_enable_thinking,
        "think_tag_leaked": think_leak,
        "truncated": trunc,
        "empty_output": empty,
        "nonfinite_logits": nonfinite,
        "probe_seconds": round(elapsed, 1),
        "sample_output": b1[0].text[:160],
    }
    for k, v in res.items():
        print(f"    {k:28s} {v}")
    if verdict == "FAIL":
        print("    >> batch invariance broken. Any accuracy number from this "
              "harness is meaningless until fixed.")
    if think_leak:
        print("    >> <think> tags reaching output. Thinking mode is NOT off.")
    if nonfinite:
        print("    >> non-finite logits. Very likely fp16 overflow; try bf16.")
    r.close()
    return res


# --------------------------------------------------------------------------
# Tier 1 / Tier 2
# --------------------------------------------------------------------------

def evaluate(spec: ModelSpec, rows: list[dict[str, Any]], device: str,
             batch_size: int) -> dict[str, Any]:
    print(f"\n[eval] {spec.key}  n={len(rows)}")
    r = Runner(spec, device=device)
    prompts = [r.build_prompt(ANSWER_SYSTEM, answer_user_prompt(x)) for x in rows]

    t = time.time()
    gens = r.generate(prompts, batch_size=batch_size)
    elapsed = time.time() - t

    per_q = []
    status = Counter()
    for row, g in zip(rows, gens):
        ans, st = parse_answer(g.text)
        status[st] += 1
        per_q.append({
            "id": row["id"],
            "gold": row["answer"],
            "pred": ans,
            "f1": token_f1(ans, row["answer"]),
            "em": exact_match(ans, row["answer"]),
            "parse_status": st,
            "think_tag": g.has_think_tag,
            "truncated": g.truncated,
            "output_tokens": g.output_tokens,
            "raw": g.text[:400],
        })

    f1 = float(np.mean([q["f1"] for q in per_q]))
    em = float(np.mean([q["em"] for q in per_q]))
    strict = status["ok"] / len(per_q)
    print(f"    F1={f1:.4f}  EM={em:.4f}  strict_json={strict:.3f}  "
          f"think_leak={sum(q['think_tag'] for q in per_q)}  "
          f"trunc={sum(q['truncated'] for q in per_q)}  {elapsed:.0f}s")

    r.close()
    return {
        "model": spec.key,
        "family": spec.family,
        "precision": spec.precision,
        "params_b": spec.params_b,
        "n": len(per_q),
        "f1": f1,
        "em": em,
        "strict_json_rate": strict,
        "parse_status": dict(status),
        "think_leak": sum(q["think_tag"] for q in per_q),
        "truncated": sum(q["truncated"] for q in per_q),
        "wall_s": round(elapsed, 1),
        "per_question": per_q,
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def paired_stats(a: dict[str, Any], b: dict[str, Any], reps: int = 10000,
                 seed: int = SEED) -> dict[str, Any]:
    """a and b evaluated on the same ids. Returns delta with bootstrap CI and
    a McNemar test on EM."""
    ai = {q["id"]: q for q in a["per_question"]}
    bi = {q["id"]: q for q in b["per_question"]}
    ids = sorted(set(ai) & set(bi))
    if not ids:
        return {"error": "no shared ids"}

    da = np.array([ai[i]["f1"] for i in ids])
    db = np.array([bi[i]["f1"] for i in ids])
    diff = db - da

    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    n = len(ids)
    for k in range(reps):
        idx = rng.integers(0, n, n)
        boot[k] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # McNemar on EM
    b_only = sum(1 for i in ids if bi[i]["em"] > ai[i]["em"])
    a_only = sum(1 for i in ids if ai[i]["em"] > bi[i]["em"])
    disc = a_only + b_only
    if disc:
        from math import comb
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(comb(disc, j) for j in range(k + 1)) / (2 ** disc))
    else:
        p = 1.0

    return {
        "a": a["model"], "b": b["model"], "n_paired": n,
        "f1_a": round(float(da.mean()), 4),
        "f1_b": round(float(db.mean()), 4),
        "delta_f1": round(float(diff.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "significant": bool(lo > 0 or hi < 0),
        "em_b_only_wins": b_only, "em_a_only_wins": a_only,
        "mcnemar_p": round(float(p), 5),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def default_models(vram_gb: float) -> list[str]:
    """Pick the biggest matched pair set that fits, in fp16 plus a 4bit echo."""
    picks: list[str] = []
    for q25, q3 in MATCHED_PAIRS:
        for stem in (q25, q3):
            fp16 = REGISTRY[f"{stem}-fp16"]
            if fp16.approx_vram_gb + 1.2 <= vram_gb:
                picks.append(fp16.key)
            else:
                q4 = REGISTRY[f"{stem}-4bit"]
                if q4.approx_vram_gb + 1.0 <= vram_gb:
                    picks.append(q4.key)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["0", "1", "2", "all"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="arena_results.json")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024 ** 3
        print(f"device: {props.name}  {vram:.1f} GiB  sm_{props.major}{props.minor}  "
              f"native_bf16={torch.cuda.is_bf16_supported()}")
        if not torch.cuda.is_bf16_supported():
            print("  !! no native bf16. Qwen3 is a bf16-native family; running it "
                  "in fp16 here risks activation overflow. Watch nonfinite_logits.")
    else:
        vram = 0.0
        print("device: cpu (this will be slow)")

    keys = [k.strip() for k in args.models.split(",") if k.strip()] or default_models(vram)
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"unknown model keys: {unknown}\navailable: {sorted(REGISTRY)}")
        return 2
    specs = [REGISTRY[k] for k in keys]
    print(f"models: {[s.key for s in specs]}")

    print("\nloading HotpotQA distractor validation ...")
    rows = load_hotpot(args.n)
    print(f"  {len(rows)} questions, levels={Counter(r['level'] for r in rows)}")

    report: dict[str, Any] = {"seed": SEED, "n": len(rows), "tier0": [], "eval": [], "paired": []}

    if args.tier in ("0", "all"):
        for s in specs:
            try:
                report["tier0"].append(tier0(s, rows, args.device))
            except Exception as e:
                print(f"    tier0 FAILED for {s.key}: {type(e).__name__}: {e}")
                report["tier0"].append({"model": s.key, "error": f"{type(e).__name__}: {e}"})
        failed = [t for t in report["tier0"] if t.get("batch_invariance_verdict") == "FAIL"]
        if failed and args.tier == "all":
            print("\n!! tier0 failed for: " + ", ".join(t["model"] for t in failed))
            print("!! refusing to report accuracy on a broken harness.")
            _dump(report, args.out)
            return 1

    if args.tier in ("1", "2", "all"):
        for s in specs:
            try:
                report["eval"].append(evaluate(s, rows, args.device, args.batch_size))
            except Exception as e:
                print(f"    eval FAILED for {s.key}: {type(e).__name__}: {e}")

        by_key = {e["model"]: e for e in report["eval"]}
        for q25, q3 in MATCHED_PAIRS:
            for precision in ("fp16", "bf16", "8bit", "4bit"):
                ka, kb = f"{q25}-{precision}", f"{q3}-{precision}"
                if ka in by_key and kb in by_key:
                    report["paired"].append(paired_stats(by_key[ka], by_key[kb]))

        print("\n" + "=" * 78)
        print(f"{'model':28s} {'F1':>7s} {'EM':>7s} {'json':>7s} {'think':>6s} {'trunc':>6s}")
        print("-" * 78)
        for e in sorted(report["eval"], key=lambda x: -x["f1"]):
            print(f"{e['model']:28s} {e['f1']:7.4f} {e['em']:7.4f} "
                  f"{e['strict_json_rate']:7.3f} {e['think_leak']:6d} {e['truncated']:6d}")

        if report["paired"]:
            print("\npaired comparisons (positive delta = Qwen3 better)")
            print("-" * 78)
            for p in report["paired"]:
                if "error" in p:
                    continue
                flag = "SIGNIFICANT" if p["significant"] else "not significant"
                print(f"  {p['a']} -> {p['b']}  n={p['n_paired']}  "
                      f"dF1={p['delta_f1']:+.4f}  95%CI={p['ci95']}  "
                      f"McNemar p={p['mcnemar_p']}  [{flag}]")

    _dump(report, args.out)
    return 0


def _dump(report: dict[str, Any], path: str) -> None:
    slim = dict(report)
    slim["eval"] = [{k: v for k, v in e.items() if k != "per_question"} for e in report.get("eval", [])]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(path.replace(".json", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path} and {path.replace('.json', '_summary.json')}")


if __name__ == "__main__":
    sys.exit(main())
