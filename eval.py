"""
Zero-shot evaluation of a pretrained base model across commonsense + math datasets.

Usage:
    python zero_shot_eval.py \
        --model meta-llama/Meta-Llama-3-8B \
        --datasets arc_c arc_e boolq hellaswag openbookqa \
                   addsub multiarith svamp singleeq gsm8k mawps \
        --output base_zeroshot.json

Output: JSON with {dataset: accuracy} — use these numbers as the "Base" row
in the forgetting table.

Commonsense datasets use log-likelihood scoring over answer choices (MCQ).
Math datasets use generation + numeric-answer extraction.
"""

import argparse
import json
import re
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


# =============================================================================
#  DATASET LOADERS — return list of examples in a unified schema
# =============================================================================

def load_arc(subset: str, split: str = "test", limit=None):
    """ARC-C or ARC-E from allenai/ai2_arc."""
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    out = []
    for ex in ds:
        choices = ex["choices"]["text"]
        labels  = ex["choices"]["label"]
        ans_idx = labels.index(ex["answerKey"]) if ex["answerKey"] in labels else None
        if ans_idx is None: continue
        out.append({
            "type":     "mcq",
            "question": ex["question"],
            "choices":  choices,
            "answer":   ans_idx,
        })
        if limit and len(out) >= limit: break
    return out


def load_boolq(split: str = "validation", limit=None):
    ds = load_dataset("google/boolq", split=split)
    out = []
    for ex in ds:
        out.append({
            "type":     "mcq",
            "question": f"Passage: {ex['passage']}\nQuestion: {ex['question']}",
            "choices":  ["No", "Yes"],
            "answer":   1 if ex["answer"] else 0,
        })
        if limit and len(out) >= limit: break
    return out


def load_hellaswag(split: str = "validation", limit=None):
    ds = load_dataset("Rowan/hellaswag", split=split)
    out = []
    for ex in ds:
        out.append({
            "type":     "mcq",
            "question": ex["ctx"],
            "choices":  ex["endings"],
            "answer":   int(ex["label"]),
        })
        if limit and len(out) >= limit: break
    return out


def load_openbookqa(split: str = "test", limit=None):
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    out = []
    for ex in ds:
        choices = ex["choices"]["text"]
        labels  = ex["choices"]["label"]
        ans_idx = labels.index(ex["answerKey"])
        out.append({
            "type":     "mcq",
            "question": ex["question_stem"],
            "choices":  choices,
            "answer":   ans_idx,
        })
        if limit and len(out) >= limit: break
    return out


def load_math_dataset(name: str, split: str = "test", limit=None):
    """
    Unified loader for AddSub / MultiArith / SVAMP / SingleEq / GSM8K / MAWPS.
    Returns {question, answer_numeric}.
    """
    repo_map = {
        "addsub":     ("allenai/lila", "addsub"),
        "multiarith": ("ChilleD/MultiArith", None),
        "svamp":      ("ChilleD/SVAMP", None),
        "singleeq":   ("allenai/lila", "single_eq"),
        "gsm8k":      ("openai/gsm8k", "main"),
        "mawps":      ("MU-NLPC/Calc-mawps", None),
    }
    if name not in repo_map:
        raise ValueError(f"Unknown math dataset: {name}")

    repo, cfg = repo_map[name]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)

    out = []
    for ex in ds:
        # Normalize field names across repos
        q = ex.get("question") or ex.get("Body") or ex.get("input_text") or ex.get("problem")
        a = ex.get("answer")   or ex.get("Answer") or ex.get("output_answer") or ex.get("final_answer")
        if q is None or a is None:
            continue
        # GSM8K answers look like "...\n#### 42" — strip to number
        if isinstance(a, str):
            m = re.search(r"####\s*(-?[\d.,]+)", a)
            if m: a = m.group(1)
            a = a.replace(",", "").strip()
        try:
            a_num = float(a)
        except (ValueError, TypeError):
            continue
        out.append({"type": "math", "question": q, "answer": a_num})
        if limit and len(out) >= limit: break
    return out


DATASET_LOADERS = {
    "arc_c":       lambda lim: load_arc("ARC-Challenge", limit=lim),
    "arc_e":       lambda lim: load_arc("ARC-Easy",      limit=lim),
    "boolq":       lambda lim: load_boolq(limit=lim),
    "hellaswag":   lambda lim: load_hellaswag(limit=lim),
    "openbookqa":  lambda lim: load_openbookqa(limit=lim),
    "addsub":      lambda lim: load_math_dataset("addsub",     limit=lim),
    "multiarith":  lambda lim: load_math_dataset("multiarith", limit=lim),
    "svamp":       lambda lim: load_math_dataset("svamp",      limit=lim),
    "singleeq":    lambda lim: load_math_dataset("singleeq",   limit=lim),
    "gsm8k":       lambda lim: load_math_dataset("gsm8k",      limit=lim),
    "mawps":       lambda lim: load_math_dataset("mawps",      limit=lim),
}


# =============================================================================
#  SCORING — MCQ via log-likelihood, math via generation
# =============================================================================

@torch.no_grad()
def score_choice_loglik(model, tokenizer, context: str, choice: str, device) -> float:
    """
    Length-normalized log-likelihood of `choice` given `context`.
    Returns sum(log p(choice tokens | context)) / num_choice_tokens.
    """
    ctx_ids     = tokenizer(context, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    full_ids    = tokenizer(context + " " + choice, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    n_ctx       = ctx_ids.shape[1]
    n_choice    = full_ids.shape[1] - n_ctx
    if n_choice <= 0:
        return float("-inf")

    logits = model(full_ids).logits[0]                     # [T, V]
    # log p(token_i | tokens_<i) — shift by 1
    log_probs   = torch.log_softmax(logits[:-1], dim=-1)   # [T-1, V]
    target_ids  = full_ids[0, 1:]                          # [T-1]
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    choice_logp = token_logps[n_ctx - 1:].sum().item()
    return choice_logp / n_choice


@torch.no_grad()
def generate_math_answer(model, tokenizer, question: str, device,
                          max_new_tokens: int = 256) -> str:
    prompt = (
        "Solve the following math problem. "
        "Show your reasoning, then give the final numeric answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def extract_numeric(text: str):
    """Extract final numeric answer from generated text."""
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        try:    return float(m.group(1).replace(",", ""))
        except ValueError: pass
    # Fallback — last number in the output
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if nums:
        try:    return float(nums[-1])
        except ValueError: pass
    return None


# =============================================================================
#  EVAL LOOP
# =============================================================================

def eval_mcq(model, tokenizer, examples, device):
    correct = 0
    for ex in tqdm(examples, desc="  MCQ", leave=False):
        scores = [
            score_choice_loglik(model, tokenizer, ex["question"], c, device)
            for c in ex["choices"]
        ]
        if scores.index(max(scores)) == ex["answer"]:
            correct += 1
    return correct / len(examples) if examples else 0.0


def eval_math(model, tokenizer, examples, device):
    correct = 0
    for ex in tqdm(examples, desc="  MATH", leave=False):
        gen  = generate_math_answer(model, tokenizer, ex["question"], device)
        pred = extract_numeric(gen)
        if pred is not None and abs(pred - ex["answer"]) < 1e-3:
            correct += 1
    return correct / len(examples) if examples else 0.0


# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, help="HF model id or local path")
    ap.add_argument("--datasets", nargs="+", required=True,
                    choices=list(DATASET_LOADERS.keys()) + ["all"])
    ap.add_argument("--output",   default="base_zeroshot.json")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Cap examples per dataset (debug)")
    ap.add_argument("--dtype",    default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    if "all" in args.datasets:
        args.datasets = list(DATASET_LOADERS.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]

    print(f"▶ Loading {args.model} in {args.dtype} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    results = {}
    for name in args.datasets:
        print(f"\n▶ Loading {name} ...")
        examples = DATASET_LOADERS[name](args.limit)
        print(f"  {len(examples)} examples")
        if not examples:
            print(f"  ⚠ skipping — empty")
            continue

        if examples[0]["type"] == "mcq":
            acc = eval_mcq(model, tokenizer, examples, device)
        else:
            acc = eval_math(model, tokenizer, examples, device)

        results[name] = round(acc, 4)
        print(f"  {name:12s}: {acc:.4f}")

        # Write incrementally so partial runs aren't lost
        Path(args.output).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 50)
    print("  ZERO-SHOT BASE RESULTS")
    print("=" * 50)
    for name, acc in results.items():
        print(f"  {name:12s}: {acc:.4f}")
    print(f"\n  saved to: {args.output}")


if __name__ == "__main__":
    main()