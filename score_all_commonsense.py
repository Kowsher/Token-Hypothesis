"""
Score all commonsense reasoning datasets for selection="attention".

Three TokenSeek scoring stages per dataset:
  Stage A — forward attention scores  (Eq. 3)
  Stage B — backward gradient scores  (Eq. 4)
  Stage C — combined I_1 + I_2        (Eq. 5)

Output: one combined .pt file per dataset.

Usage:
    # one dataset with progress bars
    python score_all_commonsense.py --dataset arc_challenge

    # all six (use --raise_on_error to abort on any failure rather than skip)
    python score_all_commonsense.py --dataset all --raise_on_error

    # smoke test: 5 samples, full pipeline, fast
    python score_all_commonsense.py --dataset arc_challenge --limit 5

Auth:
    huggingface-cli login
"""

import os
import argparse
import hashlib
import time
import gc

os.environ["HF_DATASETS_CACHE"]  = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["HF_HOME"]            = "/ocean/projects/cis250258p/mkowsher/hf_cache"

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "arc_challenge": {"load_train": "load_arc_train",        "dataset_class": "ARCDataset"},
    "arc_easy":      {"load_train": "load_arceasy_train",    "dataset_class": "ARCEasyDataset"},
    "boolq":         {"load_train": "load_boolq_train",      "dataset_class": "BoolQDataset"},
    "hellaswag":     {"load_train": "load_hellaswag_train",  "dataset_class": "HellaSwagDataset"},
    "openbookqa":    {"load_train": "load_openbookqa_train", "dataset_class": "OpenBookQADataset"},
    "winogrande":    {"load_train": "load_winogrande_train", "dataset_class": "WinoGrandeDataset"},
}


def get_dataset(dataset_name, tokenizer, max_length):
    cfg = DATASET_CONFIGS[dataset_name]
    import dataset as ds_module
    load_train   = getattr(ds_module, cfg["load_train"])
    DatasetClass = getattr(ds_module, cfg["dataset_class"])
    raw = load_train()
    return DatasetClass(raw, tokenizer, max_length=max_length, is_test=False)


# =============================================================================
#  HELPERS
# =============================================================================

def generate_identifier(input_ids) -> str:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() > 1:
            input_ids = input_ids.flatten()
        input_list = input_ids.tolist()
    else:
        input_list = list(input_ids)
    return hashlib.sha256(",".join(map(str, input_list)).encode()).hexdigest()[:16]


def to_device(t, device):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.to(device)


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def min_max_normalize(x):
    x = x.float()
    lo, hi = x.min(), x.max()
    rng = hi - lo
    return torch.zeros_like(x) if rng < 1e-12 else (x - lo) / rng


# =============================================================================
#  STAGE A — forward attention  (with tqdm)
# =============================================================================

def stage_attention(model, train_dataset, device, n_samples):
    print(f"\n  [Stage A] forward attention scores")
    model.eval()
    all_scores = {}

    bar = tqdm(range(n_samples), desc="    attention", unit="sample",
               dynamic_ncols=True, mininterval=0.5)
    for idx in bar:
        sample = train_dataset[idx]
        input_ids_t = to_device(sample["input_ids"], device)
        attn_mask = to_device(sample["attention_mask"], device) \
                    if sample.get("attention_mask") is not None else None

        with torch.no_grad():
            out = model(
                input_ids=input_ids_t,
                attention_mask=attn_mask,
                output_attentions=True,
                use_cache=False,
            )
            attn_last = out.attentions[-1]
            attn = attn_last.mean(dim=1).squeeze(0)

        attn_np = attn.detach().cpu().to(torch.float32).numpy()
        col_sums = attn_np.sum(axis=0)
        col_min, col_max = col_sums.min(), col_sums.max()
        if col_max - col_min > 0:
            scores = (col_sums - col_min) / (col_max - col_min)
        else:
            scores = np.zeros_like(col_sums)

        scores_t = torch.tensor(scores, dtype=torch.float32)
        key = generate_identifier(sample["input_ids"])
        all_scores[key] = {"len": len(scores_t), "scores": scores_t}

        if idx % 50 == 0:
            bar.set_postfix(T=len(scores_t))

    return all_scores


# =============================================================================
#  STAGE B — backward gradient  (with tqdm)
# =============================================================================

def stage_gradient(model, train_dataset, device, n_samples):
    print(f"\n  [Stage B] backward gradient scores")
    model.train()
    captured = {}

    def hook(module, inputs, output):
        h = inputs[0]
        h.retain_grad()
        captured["activation"] = h

    handle = model.model.layers[-1].register_forward_hook(hook)
    loss_fct = CrossEntropyLoss()
    all_grads = {}

    bar = tqdm(range(n_samples), desc="    gradient ", unit="sample",
               dynamic_ncols=True, mininterval=0.5)
    n_zero_loss = 0
    for idx in bar:
        sample = train_dataset[idx]
        input_ids_t = to_device(sample["input_ids"], device)
        attn_mask = to_device(sample["attention_mask"], device) \
                    if sample.get("attention_mask") is not None else None
        labels_t = to_device(sample["labels"], device)

        model.zero_grad(set_to_none=True)
        captured.clear()

        out = model(input_ids=input_ids_t, attention_mask=attn_mask, use_cache=False)
        logits = out.logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_t[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        T = input_ids_t.shape[1]
        key = generate_identifier(sample["input_ids"])

        if torch.isnan(loss) or loss.item() == 0.0:
            all_grads[key] = {"len": T, "gradient": torch.zeros(T, dtype=torch.float32)}
            n_zero_loss += 1
            continue

        loss.backward()
        h = captured["activation"]
        if h.grad is None:
            handle.remove()
            raise RuntimeError("No gradient on captured activation — hook not firing.")

        g_score = h.grad.detach().abs().sum(dim=-1).squeeze(0).float().cpu()
        all_grads[key] = {"len": g_score.shape[0], "gradient": g_score}

        if idx % 25 == 0:
            bar.set_postfix(loss=f"{loss.item():.2f}", zero=n_zero_loss)

    handle.remove()
    if n_zero_loss:
        print(f"    {n_zero_loss}/{n_samples} samples had zero loss "
              f"(all labels masked) — kept as zero gradients")
    return all_grads


# =============================================================================
#  STAGE C — combine
# =============================================================================

def stage_combine(attn_data, grad_data, alpha, beta, num_trainable_list):
    print(f"\n  [Stage C] combining (α={alpha}, β={beta})")
    common = set(attn_data) & set(grad_data)
    print(f"    common keys: {len(common):,}")
    if not common:
        raise RuntimeError("No common keys between attention and gradient files.")

    combined = {}
    for key in tqdm(common, desc="    combine  ", unit="sample",
                    dynamic_ncols=True, mininterval=0.5):
        a = attn_data[key]
        g = grad_data[key]
        if a["len"] != g["len"]:
            continue
        T = a["len"]
        I1 = a["scores"].float()
        I2 = g["gradient"].float()
        I = alpha * log_transform(I1) + beta * min_max_normalize(I2)
        sorted_idx = torch.argsort(I, descending=True).tolist()

        entry = {"len": T, "scores": I}
        for n in num_trainable_list:
            entry[str(n)] = sorted_idx[: min(n, T)]
        combined[key] = entry

    print(f"    combined: {len(combined):,} entries")
    return combined


# =============================================================================
#  ONE DATASET END-TO-END
# =============================================================================

def score_one_dataset(args, dataset_name):
    print(f"\n{'='*60}")
    print(f"  Scoring: {dataset_name}")
    print(f"{'='*60}")

    output_path = os.path.join(args.output_dir, f"{dataset_name}_combined.pt")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"  ✓ Already exists: {output_path}  (pass --overwrite to redo)")
        return

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print(f"  Loading {dataset_name}...")
    train_dataset = get_dataset(dataset_name, tokenizer, args.max_length)
    n_samples = len(train_dataset) if args.limit is None else min(args.limit, len(train_dataset))
    print(f"  Train samples: {len(train_dataset):,}  (using {n_samples:,})")

    # Model
    print(f"  Loading model on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    print(f"  Model loaded.")

    # Stage A
    t_a = time.time()
    attn_data = stage_attention(model, train_dataset, args.device, n_samples)
    print(f"  [Stage A] {(time.time()-t_a)/60:.1f} min, {len(attn_data):,} entries")

    # Stage B
    t_b = time.time()
    grad_data = stage_gradient(model, train_dataset, args.device, n_samples)
    print(f"  [Stage B] {(time.time()-t_b)/60:.1f} min, {len(grad_data):,} entries")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Stage C
    combined = stage_combine(attn_data, grad_data, args.alpha, args.beta,
                             args.num_trainable)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(combined, output_path)
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"\n  ✓ Saved: {output_path}  ({size_mb:.1f} MB, {len(combined):,} entries)")

    # Verification
    if combined:
        key0 = next(iter(combined))
        e0 = combined[key0]
        T = e0["len"]
        n0 = args.num_trainable[0]
        if str(n0) in e0:
            avg_pos = sum(e0[str(n0)]) / len(e0[str(n0)])
            pct = 100 * avg_pos / T
            print(f"  Sample 0 (T={T}): N={n0} avg position = "
                  f"{avg_pos:.0f}/{T} ({pct:.1f}%)  "
                  f"{'✓ near answer' if pct > 85 else '⚠ NOT near answer'}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit per-dataset to first N samples (debugging)")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta",  type=float, default=0.7)
    parser.add_argument("--num_trainable", type=int, nargs="+",
                        default=[10, 25, 50, 100, 300, 500, 700, 900])
    parser.add_argument("--output_dir", type=str,
                        default="/ocean/projects/cis250258p/mkowsher/scores")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score even if output file exists")
    parser.add_argument("--raise_on_error", action="store_true",
                        help="Abort on first error instead of skipping the dataset. "
                             "Use this if you suspect a setup problem (e.g. cache "
                             "permissions, missing auth) — it makes the failure visible.")
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASET_CONFIGS.keys():
            try:
                score_one_dataset(args, name)
            except Exception as e:
                if args.raise_on_error:
                    raise
                print(f"\n!!! {name} failed: {type(e).__name__}: {e}")
                print(f"    (use --raise_on_error to debug)")
                gc.collect()
                torch.cuda.empty_cache()
    else:
        score_one_dataset(args, args.dataset)

    print("\n" + "=" * 60)
    print("All scoring done.")
    print("=" * 60)


if __name__ == "__main__":
    main()