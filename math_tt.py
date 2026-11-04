"""
Math Reasoning — Token Theory full FT — LLaMA-3-8B (last-N selection by default).

Trains on the math_10k composite train set, evaluates on all 7 math test sets:
  AddSub, MultiArith, SVAMP, SingleEq, GSM8K, MathQA, MAWPS

Uses llama_tt.py / llama_token.py with LlamaPrefixForCausalLM. Disk-conscious:
NO model checkpoints are saved (save_strategy="no", no save_pretrained calls).

Includes the balanced-device-map fix to prevent the GPU 0 collapse / OOM at
N >= 100 that we saw on commonsense.

Usage:
    # Default: last-N sweep, no baseline
    python math_tt.py

    # With baseline
    python math_tt.py --include_baseline

    # Random-K instead of last-N
    python math_tt.py --selection random

    # Custom N values
    python math_tt.py --num_trainable 10 25 50 100

Auth (do this BEFORE running):
    huggingface-cli login
"""

import os
import json
import time
import csv
import gc
import argparse

# Adjust if your project allocation moves
os.environ["HF_DATASETS_CACHE"]  = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["HF_HOME"]            = "/ocean/projects/cis250258p/mkowsher/hf_cache"

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)

from llama_tt import (
    LlamaPrefixForCausalLM,
    LlamaConfigPrefix,
    DataCollatorWithPaddingForPrefix,
)

from dataset import load_math_train, MathDataset
from evaluator import MathMultiEvaluator


# =============================================================================
#  HELPERS
# =============================================================================

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Frozen parameters    : {total - trainable:,}")


def get_gpu_memory_stats():
    stats = {}
    if not torch.cuda.is_available():
        return stats
    for i in range(torch.cuda.device_count()):
        try:
            peak_alloc    = torch.cuda.max_memory_allocated(i) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(i)  / 1024**3
            stats[f"gpu{i}_peak_alloc_gb"]    = round(peak_alloc, 3)
            stats[f"gpu{i}_peak_reserved_gb"] = round(peak_reserved, 3)
            print(f"  GPU {i}: peak alloc={peak_alloc:.2f} GB  "
                  f"peak reserved={peak_reserved:.2f} GB")
        except RuntimeError:
            pass
    return stats


def reset_gpu_memory_stats():
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except RuntimeError:
            pass


def build_balanced_device_map(num_layers: int, num_gpus: int) -> dict:
    """Manual balanced device map for LlamaPrefixForCausalLM.

    Layout (e.g. 32 layers, 2 GPUs):
      GPU 0: embed_tokens, layers 0..15
      GPU 1: layers 16..31, norm, lm_head

    This avoids accelerate's auto-balancer mis-estimating layer sizes for
    custom subclasses, which collapsed the model onto GPU 0 at N >= 100
    in our commonsense runs."""
    if num_gpus < 1:
        raise ValueError(f"need at least 1 GPU, got {num_gpus}")

    device_map = {"model.embed_tokens": 0}

    layers_per_gpu = [num_layers // num_gpus] * num_gpus
    for i in range(num_layers % num_gpus):
        layers_per_gpu[i] += 1

    layer_idx = 0
    for gpu in range(num_gpus):
        for _ in range(layers_per_gpu[gpu]):
            device_map[f"model.layers.{layer_idx}"] = gpu
            layer_idx += 1

    last_gpu = num_gpus - 1
    device_map["model.norm"] = last_gpu
    device_map["lm_head"]    = last_gpu

    return device_map


def transfer_prefix_to_standard(prefix_model, model_name, dtype=torch.bfloat16):
    """Copy state_dict from LlamaPrefixForCausalLM to standard
    LlamaForCausalLM. Needed because the prefix model can't .generate()."""
    trained_state = {k: v.detach().cpu() for k, v in prefix_model.state_dict().items()}
    del prefix_model
    gc.collect()
    torch.cuda.empty_cache()

    standard = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
    )
    missing, unexpected = standard.load_state_dict(trained_state, strict=False)
    if missing or unexpected:
        print(f"  ! state_dict transfer — missing={len(missing)}, "
              f"unexpected={len(unexpected)}")
    else:
        print(f"  ✓ state_dict transferred cleanly ({len(trained_state)} tensors)")
    del trained_state
    gc.collect()
    return standard


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Math sweep — Token Theory (last-N default) — LLaMA-3-8B")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--num_trainable", type=int, nargs="+", default=None,
                        help="Override default sweep list")
    parser.add_argument("--selection", type=str, default="last",
                        choices=["last", "random", "attention"],
                        help="Selection strategy (default: last)")
    parser.add_argument("--attention_map_path", type=str, default=None,
                        help="Required when --selection attention")
    parser.add_argument("--include_baseline", action="store_true",
                        help="Include full FT baseline (None) in sweep")
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--max_length",    type=int,   default=1024)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--logging_steps", type=int,   default=50)
    parser.add_argument("--tolerance",     type=float, default=0.1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens",  type=int, default=512)
    parser.add_argument("--output_root",   type=str,
                        default=None)
    args = parser.parse_args()

    if args.selection == "attention" and not args.attention_map_path:
        parser.error("--selection attention requires --attention_map_path")

    output_root = args.output_root or f"math_tt_{args.selection}"
    os.makedirs(output_root, exist_ok=True)

    DEFAULT_NS = [10, 25, 50, 100, 300, 500, 700, 900]
    n_list = args.num_trainable or DEFAULT_NS
    num_trainable_list = ([None] + n_list) if args.include_baseline else n_list

    print(f"\n{'='*60}")
    print(f"  Math Reasoning — Token Theory ({args.selection})")
    print(f"  Model        : {args.model_name}")
    print(f"  Batch size   : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR           : {args.lr}  Epochs: {args.epochs}")
    print(f"  Selection    : {args.selection}")
    if args.selection == "attention":
        print(f"  Score file   : {args.attention_map_path}")
    print(f"  Baseline     : {'YES' if args.include_baseline else 'NO'}")
    print(f"  N values     : {num_trainable_list}")
    print(f"  Output       : {output_root}")
    print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Train data ────────────────────────────────────────────────────────────
    print("Loading math_10k train data...")
    raw_train = load_math_train()
    train_dataset = MathDataset(
        raw_train, tokenizer, max_length=args.max_length, is_test=False)
    print(f"Train samples: {len(train_dataset):,}\n")

    # ── Build balanced device map (computed once) ────────────────────────────
    base_cfg_for_layers = AutoConfig.from_pretrained(args.model_name)
    num_layers = base_cfg_for_layers.num_hidden_layers
    num_gpus   = max(1, torch.cuda.device_count())
    balanced_dmap = build_balanced_device_map(num_layers, num_gpus)
    print(f"Balanced device map: {num_layers} layers across {num_gpus} GPU(s)")
    counts = {g: sum(1 for v in balanced_dmap.values() if v == g)
              for g in range(num_gpus)}
    print(f"  modules per GPU: {counts}\n")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    DATASET_NAMES = ["AddSub", "MultiArith", "SVAMP", "SingleEq",
                     "GSM8K", "MathQA", "MAWPS"]
    acc_fields = ["acc_" + n.lower() for n in DATASET_NAMES]

    csv_fields = (
        ["num_trainable", "train_time_min", "average_accuracy"]
        + acc_fields + gpu_fields
    )
    CSV_PATH  = os.path.join(output_root, "math_tt.csv")
    JSON_PATH = os.path.join(output_root, "math_tt.json")
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    all_results = {}

    # ── Sweep ─────────────────────────────────────────────────────────────────
    for num_trainable in num_trainable_list:
        is_baseline = (num_trainable is None)
        run_label   = "baseline_full" if is_baseline else f"trainable_{num_trainable}"

        print(f"\n{'='*60}")
        print(f"  RUN: {run_label}")
        print(f"{'='*60}\n")

        torch.cuda.empty_cache()
        reset_gpu_memory_stats()

        # ── Build model + collator ───────────────────────────────────────────
        if is_baseline:
            print("  Mode: baseline — standard LlamaForCausalLM (device_map=auto)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
                label_pad_token_id=-100,
            )
        else:
            print(f"  Mode: TokenTune {args.selection} — LlamaPrefixForCausalLM, "
                  f"prefix_length={num_trainable}")
            print(f"  Using BALANCED device map (not auto)")

            base_cfg = AutoConfig.from_pretrained(args.model_name)
            prefix_cfg = LlamaConfigPrefix(**base_cfg.to_dict())
            prefix_cfg.prefix_length = num_trainable

            model = LlamaPrefixForCausalLM.from_pretrained(
                args.model_name,
                config=prefix_cfg,
                torch_dtype=torch.bfloat16,
                device_map=balanced_dmap,
            )

            collator_kwargs = dict(
                tokenizer=tokenizer,
                padding=True,
                max_length=args.max_length,
                pad_to_multiple_of=8,
                prefix_length=num_trainable,
                selection=args.selection,
            )
            if args.selection == "attention":
                collator_kwargs["attention_map_path"] = args.attention_map_path
            collator = DataCollatorWithPaddingForPrefix(**collator_kwargs)

        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config = GenerationConfig(
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        count_parameters(model)

        # ── Train ────────────────────────────────────────────────────────────
        run_dir = os.path.join(output_root, run_label)
        training_args = TrainingArguments(
            output_dir                  = run_dir,
            learning_rate               = args.lr,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs            = args.epochs,
            weight_decay                = 0.01,
            save_strategy               = "no",
            logging_steps               = args.logging_steps,
            lr_scheduler_type           = "cosine",
            warmup_steps                = args.warmup_steps,
            bf16                        = True,
            remove_unused_columns       = False,
            report_to                   = "none",
        )

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = train_dataset,
            data_collator = collator,
        )

        train_start = time.time()
        trainer.train()
        train_time  = time.time() - train_start

        print(f"\n  Training time: {train_time:.1f}s  ({train_time/60:.2f} min)")
        print("  GPU memory (training peak):")
        gpu_stats = get_gpu_memory_stats()
        reset_gpu_memory_stats()

        # ── Convert prefix → standard for eval (no save_pretrained) ─────────
        if is_baseline:
            eval_model = model
            trainer.optimizer    = None
            trainer.lr_scheduler = None
            del trainer
            trainer = None
        else:
            trainer.optimizer    = None
            trainer.lr_scheduler = None
            del trainer
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()

            print("  Converting LlamaPrefixForCausalLM → LlamaForCausalLM for eval...")
            eval_model = transfer_prefix_to_standard(model, args.model_name)
            eval_model.config.pad_token_id = tokenizer.pad_token_id
            eval_model.generation_config = GenerationConfig(
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ── Evaluate on all 7 math sets ─────────────────────────────────────
        multi_eval = MathMultiEvaluator(
            model           = eval_model,
            tokenizer       = tokenizer,
            device          = "cuda",
            batch_size      = args.eval_batch_size,
            max_new_tokens  = args.max_new_tokens,
            max_length      = args.max_length,
            tolerance       = args.tolerance,
        )
        eval_results = multi_eval.evaluate_all()
        eval_dict    = multi_eval.results_to_dict(eval_results)

        # ── Build row ────────────────────────────────────────────────────────
        per_ds_accs = {
            "acc_" + n.lower(): eval_dict[n]["accuracy"]
            for n in DATASET_NAMES
        }
        row = {
            "num_trainable":    "baseline" if is_baseline else num_trainable,
            "train_time_min":   round(train_time / 60, 2),
            "average_accuracy": round(eval_dict["average_accuracy"], 4),
            **per_ds_accs,
            **gpu_stats,
        }
        all_results[run_label] = row

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        print(f"  ✓ Row saved → {CSV_PATH}")

        # ── Cleanup (no save_pretrained! disk stays clean) ──────────────────
        del eval_model, multi_eval
        gc.collect()
        torch.cuda.empty_cache()

        try:
            import shutil
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

        print(f"\n✓ Done: {run_label}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    print("\n" + "=" * 100)
    print(f"  SUMMARY  (Math / LLaMA-3-8B / {args.selection})")
    print("=" * 100)

    short_names = ["AddSub", "MulAri", "SVAMP", "SingEq", "GSM8K", "MathQA", "MAWPS"]
    header_acc  = "  ".join(f"{n:>7}" for n in short_names)
    gpu_header  = "  ".join(f"gpu{i}_alloc_gb" for i in range(num_gpus))
    print(f"  {'run':>20}  {'avg_acc':>8}  {header_acc}  "
          f"{'time_min':>10}  {gpu_header}")
    print("  " + "-" * 98)

    for label, row in all_results.items():
        per_ds = "  ".join(
            f"{row['acc_' + n.lower()]:>7.4f}"
            for n in DATASET_NAMES
        )
        gpu_vals = "  ".join(
            str(row.get(f"gpu{i}_peak_alloc_gb", "N/A"))
            for i in range(num_gpus)
        )
        print(f"  {label:>20}  {row['average_accuracy']:>8.4f}  {per_ds}  "
              f"{row['train_time_min']:>10.2f}  {gpu_vals}")

    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved:")
    print(f"  CSV  → {CSV_PATH}")
    print(f"  JSON → {JSON_PATH}")


if __name__ == "__main__":
    main()