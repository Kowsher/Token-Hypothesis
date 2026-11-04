"""
LoRA VQA Training Sweep — Qwen3-VL-8B-Instruct
Token Theory LoRA fine-tuning across all datasets and all num_trainable values.
LoRA applied to language model attention projections only (q/k/v/o_proj).

Imports from:
    dataset.py   — Qwen3VLSafeTagDataset, Qwen3VLVQARadDataset, etc.
    collator.py  — Qwen3VLCollator
    evaluator.py — Qwen3VLSafeTagEvaluator, etc.

Usage:
    python vqa_lora_qwen3vl8b.py --dataset safetag
    python vqa_lora_qwen3vl8b.py --dataset vqarad
    python vqa_lora_qwen3vl8b.py --dataset stvqa
    python vqa_lora_qwen3vl8b.py --dataset m3cot
    python vqa_lora_qwen3vl8b.py --dataset mmk12
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

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from causal import lora, lora_baseline, merge
from dataset import (
    Qwen3VLSafeTagDataset,
    Qwen3VLVQARadDataset,
    Qwen3VLSTVQADataset,
    Qwen3VLM3cotDataset,
    Qwen3VLMMK12Dataset,
)
from collator import Qwen3VLCollator
from evaluator import (
    Qwen3VLSafeTagEvaluator,
    Qwen3VLVQARadEvaluator,
    Qwen3VLSTVQAEvaluator,
    Qwen3VLM3cotEvaluator,
    Qwen3VLMMK12Evaluator,
)


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "safetag": {
        "hf_name":        "XuankunRong/SafeTag-VL-3K",
        "filter_fn":      lambda x: x['images'] is not None and len(x['images']) > 0,
        "train_split":    "train",
        "test_split":     "test",
        "DatasetClass":   Qwen3VLSafeTagDataset,
        "EvalClass":      Qwen3VLSafeTagEvaluator,
        "max_new_tokens": 16,
        "max_length":     4024,
    },
    "vqarad": {
        "hf_name":        "flaviagiammarino/vqa-rad",
        "filter_fn":      lambda x: x['image'] is not None,
        "train_split":    "train",
        "test_split":     "test",
        "DatasetClass":   Qwen3VLVQARadDataset,
        "EvalClass":      Qwen3VLVQARadEvaluator,
        "max_new_tokens": 64,
        "max_length":     4048,
    },
    "stvqa": {
        "hf_name":        "hunarbatra/STVQA-7K",
        "filter_fn":      lambda x: x['images'] is not None,
        "train_split":    "train",
        "test_split":     "val",
        "DatasetClass":   Qwen3VLSTVQADataset,
        "EvalClass":      Qwen3VLSTVQAEvaluator,
        "max_new_tokens": 512,
        "max_length":     4048,
    },
    "m3cot": {
        "hf_name":        "LightChen2333/M3CoT",
        "filter_fn":      lambda x: x['image'] is not None,
        "train_split":    "train",
        "test_split":     "test",
        "DatasetClass":   Qwen3VLM3cotDataset,
        "EvalClass":      Qwen3VLM3cotEvaluator,
        "max_new_tokens": 512,
        "max_length":     4048,
    },
    "mmk12": {
        "hf_name":        "chamber111/VPPO_MMK12_validation",
        "filter_fn":      lambda x: x['images'] is not None and len(x['images']) > 0,
        "train_split":    "train",
        "test_split":     None,   # split manually 80/20
        "DatasetClass":   Qwen3VLMMK12Dataset,
        "EvalClass":      Qwen3VLMMK12Evaluator,
        "max_new_tokens": 512,
        "max_length":     8096,
    },
}

NUM_TRAINABLE_LIST = [None, 10, 25, 50, 100, 200, 300, 500, 700, 1000, 1500]

# LoRA config — attention projections in language model only
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_NAME_PREFIX    = 'model.language_model'


# =============================================================================
#  HELPERS
# =============================================================================

def freeze_conv3d(model):
    frozen = 0
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            for p in module.parameters():
                p.requires_grad = False
            frozen += 1
    print(f"  Froze {frozen} Conv3d layers (vision patch embedding)")
    return frozen


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total:,}  Trainable: {trainable:,}  Frozen: {total-trainable:,}")


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


def build_csv_fields(dataset_name, gpu_fields):
    base = ["num_trainable", "train_time_min", "accuracy", "failed"]
    if dataset_name == "vqarad":
        base += ["closed_acc", "open_acc"]
    return base + gpu_fields


def build_row(dataset_name, num_trainable, is_baseline, train_time, results, gpu_stats):
    row = {
        "num_trainable":  "baseline" if is_baseline else num_trainable,
        "train_time_min": round(train_time / 60, 2),
        "accuracy":       round(results.get("accuracy", 0.0), 4),
        "failed":         results.get("failed", 0),
    }
    if dataset_name == "vqarad":
        row["closed_acc"] = round(results.get("closed_acc", 0.0), 4)
        row["open_acc"]   = round(results.get("open_acc",   0.0), 4)
    row.update(gpu_stats)
    return row


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Qwen3-VL-8B VQA sweep — Token Theory LoRA fine-tuning")
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name",    type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--num_trainable", type=int, nargs="+", default=None,
                        help="Override default sweep list")
    parser.add_argument("--batch_size",    type=int,   default=2)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--lora_r",        type=int,   default=4)
    parser.add_argument("--lora_alpha",    type=float, default=5.0)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--logging_steps", type=int,   default=20)
    parser.add_argument("--output_root",   type=str,   default=None)
    args = parser.parse_args()

    cfg         = DATASET_CONFIGS[args.dataset]
    model_name  = args.model_name
    output_root = args.output_root or f"{args.dataset}_lora_qwen3vl"
    os.makedirs(output_root, exist_ok=True)

    num_trainable_list = [None] + (args.num_trainable or NUM_TRAINABLE_LIST[1:])

    lora_kwargs = dict(
        target_modules = LORA_TARGET_MODULES,
        name_prefix    = LORA_NAME_PREFIX,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
    )

    print(f"\n{'='*60}")
    print(f"  Qwen3-VL-8B LoRA Fine-tuning — Token Theory")
    print(f"  Dataset      : {args.dataset}  ({cfg['hf_name']})")
    print(f"  Model        : {model_name}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Grad accum   : {args.grad_accum}")
    print(f"  LR           : {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  LoRA r       : {args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Target mods  : {LORA_TARGET_MODULES}")
    print(f"  Name prefix  : {LORA_NAME_PREFIX}")
    print(f"  N values     : {num_trainable_list}")
    print(f"  Output       : {output_root}")
    print(f"{'='*60}\n")

    # ── Processor + Collator ──────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    collator = Qwen3VLCollator(processor=processor, label_mask_mode='response_only')

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {args.dataset}...")
    raw = load_dataset(cfg["hf_name"])
    raw = raw.filter(cfg["filter_fn"])
    print(raw)

    if cfg["test_split"] is None:
        split     = raw["train"].train_test_split(test_size=0.2, seed=42)
        raw_train = split["train"]
        raw_test  = split["test"]
        print(f"  mmk12: split train→{len(raw_train)}, test→{len(raw_test)} (seed=42)")
    else:
        raw_train = raw[cfg["train_split"]]
        raw_test  = raw[cfg["test_split"]]

    max_length = cfg["max_length"]
    print('max_length:', max_length)

    train_dataset = cfg["DatasetClass"](
        raw_train, processor, max_length=max_length, is_test=False)
    test_dataset  = cfg["DatasetClass"](
        raw_test,  processor, max_length=max_length, is_test=True)

    print(f"\nTrain : {len(train_dataset):,}")
    print(f"Test  : {len(test_dataset):,}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    samples     = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
    train_batch = collator(samples)
    print(f"\n=== SANITY CHECK ===")
    for k, v in train_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
    labels    = train_batch['labels'][0]
    input_ids = train_batch['input_ids'][0]
    n_train   = (labels != -100).sum().item()
    decoded   = processor.tokenizer.decode(
        input_ids[labels != -100], skip_special_tokens=False)
    print(f"  trainable tokens : {n_train} / {labels.shape[0]}")
    print(f"  trainable text   : {repr(decoded)}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = build_csv_fields(args.dataset, gpu_fields)
    CSV_PATH   = os.path.join(output_root, f"{args.dataset}_lora_qwen3vl8b.csv")
    JSON_PATH  = os.path.join(output_root, f"{args.dataset}_lora_qwen3vl8b.json")

    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_results = {}

    for num_trainable in num_trainable_list:
        is_baseline = (num_trainable is None)
        run_label   = "baseline_lora" if is_baseline else f"trainable_{num_trainable}"

        print(f"\n{'='*60}")
        print(f"  RUN: {run_label}")
        print(f"{'='*60}\n")

        torch.cuda.empty_cache()

        # Fresh model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        # Freeze Conv3d layers (vision encoder) before applying LoRA
        freeze_conv3d(model)

        # Apply LoRA — language model attention projections only
        if is_baseline:
            lora_baseline(model, **lora_kwargs)
        else:
            lora(model, num_trainable=num_trainable, **lora_kwargs)

        count_parameters(model)

        # Train
        run_dir = os.path.join(output_root, run_label)
        training_args = TrainingArguments(
            output_dir                  = run_dir,
            learning_rate               = args.lr,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs            = args.epochs,
            weight_decay                = 0.0,
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

        reset_gpu_memory_stats()
        train_start = time.time()
        trainer.train()
        train_time  = time.time() - train_start

        print(f"\n  Training time: {train_time:.1f}s  ({train_time/60:.2f} min)")
        print("  GPU memory (training peak):")
        gpu_stats = get_gpu_memory_stats()
        reset_gpu_memory_stats()

        # Merge LoRA weights back
        merge(model)

        # Evaluate
        evaluator = cfg["EvalClass"](
            model          = model,
            processor      = processor,
            collator       = collator,
            device         = "cuda",
            batch_size     = 16,
            max_new_tokens = cfg["max_new_tokens"],
        )
        results = evaluator.evaluate(test_dataset, split_name=run_label)

        # Save row
        row = build_row(args.dataset, num_trainable, is_baseline,
                        train_time, results, gpu_stats)
        all_results[run_label] = row

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)
        print(f"  ✓ Row saved → {CSV_PATH}")

        # Cleanup
        trainer.optimizer    = None
        trainer.lr_scheduler = None
        del model, trainer, evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n✓ Done: {run_label}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    print("\n" + "=" * 80)
    print(f"  SUMMARY  ({args.dataset.upper()} / Qwen3-VL-8B / Token Theory LoRA)")
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