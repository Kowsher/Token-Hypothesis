"""
Commonsense Reasoning — Full Fine-tuning with Token Theory — LLaMA-3-8B
Unified script across all commonsense reasoning datasets.

Usage:
    python commonsense_full.py --dataset arc_challenge
    python commonsense_full.py --dataset arc_easy
    python commonsense_full.py --dataset boolq
    python commonsense_full.py --dataset hellaswag
    python commonsense_full.py --dataset openbookqa
    python commonsense_full.py --dataset winogrande
    python commonsense_full.py --dataset arc_challenge --num_trainable 10 25 50
"""

import os
import json
import time
import csv
import gc
import argparse

os.environ["HF_DATASETS_CACHE"] = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["HF_HOME"]            = "/ocean/projects/cis250089p/nprottasha/hf"

from huggingface_hub import login
login(token="")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)

from causal import full, full_baseline, merge


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "arc_challenge": {
        "load_train":    "load_arc_train",
        "load_test":     "load_arc_test",
        "dataset_class": "ARCDataset",
        "eval_class":    "ARCEvaluator",
        "output_root":   "full_arc_challange",
        "csv_name":      "full_arc_challange",
    },
    "arc_easy": {
        "load_train":    "load_arceasy_train",
        "load_test":     "load_arceasy_test",
        "dataset_class": "ARCEasyDataset",
        "eval_class":    "ARCEasyEvaluator",
        "output_root":   "arceasy_full",
        "csv_name":      "arceasy_full",
    },
    "boolq": {
        "load_train":    "load_boolq_train",
        "load_test":     "load_boolq_test",
        "dataset_class": "BoolQDataset",
        "eval_class":    "BoolQEvaluator",
        "output_root":   "boolq_full",
        "csv_name":      "boolq_full",
    },
    "hellaswag": {
        "load_train":    "load_hellaswag_train",
        "load_test":     "load_hellaswag_test",
        "dataset_class": "HellaSwagDataset",
        "eval_class":    "HellaSwagEvaluator",
        "output_root":   "hellaswag_full",
        "csv_name":      "hellaswag_full",
    },
    "openbookqa": {
        "load_train":    "load_openbookqa_train",
        "load_test":     "load_openbookqa_test",
        "dataset_class": "OpenBookQADataset",
        "eval_class":    "OpenBookQAEvaluator",
        "output_root":   "openbookqa_full",
        "csv_name":      "openbookqa_full",
    },
    "winogrande": {
        "load_train":    "load_winogrande_train",
        "load_test":     "load_winogrande_test",
        "dataset_class": "WinoGrandeDataset",
        "eval_class":    "WinoGrandeEvaluator",
        "output_root":   "winogrande_full",
        "csv_name":      "winogrande_full",
    },
}


# =============================================================================
#  DYNAMIC IMPORTS
# =============================================================================

def get_dataset_and_evaluator(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]

    # Dynamic imports from dataset.py
    import dataset as ds_module
    import evaluator as ev_module

    load_train    = getattr(ds_module, cfg["load_train"])
    load_test     = getattr(ds_module, cfg["load_test"])
    DatasetClass  = getattr(ds_module, cfg["dataset_class"])
    EvaluatorClass= getattr(ev_module, cfg["eval_class"])

    return load_train, load_test, DatasetClass, EvaluatorClass


# =============================================================================
#  HELPERS
# =============================================================================

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")


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
            print(f"  GPU {i}: peak allocated={peak_alloc:.2f} GB  "
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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Commonsense Reasoning — Token Theory Full Fine-tuning")
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name",    type=str,
                        default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--num_trainable", type=int, nargs="+", default=None,
                        help="Override default sweep list")
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--max_length",    type=int,   default=1000)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--logging_steps", type=int,   default=50)
    parser.add_argument("--output_root",   type=str,   default=None)
    args = parser.parse_args()

    cfg         = DATASET_CONFIGS[args.dataset]
    model_name  = args.model_name
    output_root = args.output_root or cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)

    NUM_TRAINABLE_LIST = [None, 10, 25, 50, 100, 300, 500, 700, 900]
    num_trainable_list = [None] + (args.num_trainable or NUM_TRAINABLE_LIST[1:])

    print(f"\n{'='*60}")
    print(f"  Commonsense Reasoning — Token Theory Full Fine-tuning")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Model        : {model_name}")
    print(f"  Batch size   : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR           : {args.lr}  Epochs: {args.epochs}")
    print(f"  N values     : {num_trainable_list}")
    print(f"  Output       : {output_root}")
    print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────────────
    load_train, load_test, DatasetClass, EvaluatorClass = \
        get_dataset_and_evaluator(args.dataset)

    print(f"\nLoading {args.dataset} train split...")
    raw_train = load_train()
    print(f"Loading {args.dataset} test split...")
    raw_test  = load_test()

    train_dataset = DatasetClass(
        raw_train, tokenizer, max_length=args.max_length, is_test=False)
    test_dataset  = DatasetClass(
        raw_test,  tokenizer, max_length=args.max_length, is_test=True)

    print(f"\nTrain samples : {len(train_dataset):,}")
    print(f"Test  samples : {len(test_dataset):,}")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── Sanity check ──────────────────────────────────────────────────────────
    test_samples  = [test_dataset[i] for i in range(3)]
    train_samples = [train_dataset[i] for i in range(10)]
    train_batch   = collator(train_samples)

    print("\n=== SANITY CHECK ===")
    labels     = train_batch['labels'][0]
    input_ids  = train_batch['input_ids'][0]
    non_masked = (labels != -100).sum().item()
    decoded    = tokenizer.decode(input_ids[labels != -100], skip_special_tokens=False)
    print(f"  Response tokens : {non_masked} / {labels.shape[0]}")
    print(f"  Response text   : {repr(decoded[:80])}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = ["num_trainable", "train_time_min", "accuracy", "failed"] + gpu_fields
    CSV_PATH   = os.path.join(output_root, f"{cfg['csv_name']}.csv")
    JSON_PATH  = os.path.join(output_root, f"{cfg['csv_name']}.json")

    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_results = {}

    for num_trainable in num_trainable_list:
        is_baseline = (num_trainable is None)
        run_label   = "baseline_full" if is_baseline else f"trainable_{num_trainable}"

        print(f"\n{'='*60}")
        print(f"  RUN: {run_label}")
        print(f"{'='*60}\n")

        torch.cuda.empty_cache()

        # Fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config = GenerationConfig(
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
        )

        reset_gpu_memory_stats()

        # Apply Token Theory
        if is_baseline:
            full_baseline(model)
        else:
            full(model,
                 num_trainable  = num_trainable,
                 target_modules = 'all',
                 skip_frozen    = True)

        count_parameters(model)

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

        merge(model)
        os.makedirs(run_dir, exist_ok=True)

        evaluator = EvaluatorClass(
            model          = model,
            tokenizer      = tokenizer,
            device         = "cuda",
            batch_size     = 16,
            max_new_tokens = 16,
        )
        results = evaluator.evaluate(test_dataset, split_name=run_label)

        row = {
            "num_trainable":  "baseline" if is_baseline else num_trainable,
            "train_time_min": round(train_time / 60, 2),
            "accuracy":       round(results["accuracy"], 4),
            "failed":         results.get("failed", 0),
            **gpu_stats,
        }
        all_results[run_label] = row

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        print(f"  ✓ Row saved → {CSV_PATH}")

        trainer.optimizer    = None
        trainer.lr_scheduler = None
        del model, trainer, evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n✓ Done: {run_label}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    print("\n" + "=" * 80)
    print(f"  SUMMARY  ({args.dataset.upper()} / LLaMA-3-8B / Token Theory Full)")
    print("=" * 80)

    gpu_header = "  ".join(f"gpu{i}_alloc_gb" for i in range(num_gpus))
    print(f"  {'run':>20}  {'accuracy':>10}  {'time_min':>10}  {gpu_header}")
    print("  " + "-" * 78)

    for label, row in all_results.items():
        gpu_vals = "        ".join(
            str(row.get(f"gpu{i}_peak_alloc_gb", "N/A"))
            for i in range(num_gpus)
        )
        print(f"  {label:>20}  {row['accuracy']:>10.4f}  "
              f"{row['train_time_min']:>10.2f}  {gpu_vals}")

    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved:\n  CSV  → {CSV_PATH}\n  JSON → {JSON_PATH}")


if __name__ == "__main__":
    main()