"""
Common VQA DoRA Fine-tuning Script with Token Theory

Usage:
    python vqa_dora.py --dataset vqarad
    python vqa_dora.py --dataset stvqa --batch_size 2
    python vqa_dora.py --dataset m3cot --batch_size 1 --grad_accum 8
    python vqa_dora.py --dataset safetag --num_trainable 10 25 50 100 300 500 700 1000 1500

Supported datasets:
    vqarad   — flaviagiammarino/vqa-rad
    stvqa    — hunarbatra/STVQA-7K
    m3cot    — LightChen2333/M3CoT
    safetag  — XuankunRong/SafeTag-VL-3K
    mmk12    — chamber111/VPPO_MMK12_validation
"""

from huggingface_hub import login


# Log in using your Hugging Face token
login("")



import os
import json
import time
import csv
import gc
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_CACHE"] = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["HF_HOME"]            = "/ocean/projects/cis250089p/nprottasha/hf"

import torch
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from causal import dora, dora_baseline, merge
from collator import LLaVACollator


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "vqarad": {
        "hf_name":         "flaviagiammarino/vqa-rad",
        "filter_fn":       lambda x: x['image'] is not None,
        "train_split":     "train",
        "test_split":      "test",
        "dataset_class":   "VQARadDataset",
        "evaluator_class": "VQARadEvaluator",
        "max_new_tokens":  64,
        "default_batch":   3,
    },
    "stvqa": {
        "hf_name":         "hunarbatra/STVQA-7K",
        "filter_fn":       lambda x: x['images'] is not None,
        "train_split":     "train",
        "test_split":      "val",
        "dataset_class":   "STVQADataset",
        "evaluator_class": "STVQAEvaluator",
        "max_new_tokens":  512,
        "default_batch":   3,
    },
    "m3cot": {
        "hf_name":         "LightChen2333/M3CoT",
        "filter_fn":       lambda x: x['image'] is not None,
        "train_split":     "train",
        "test_split":      "test",
        "dataset_class":   "M3cotDataset",
        "evaluator_class": "M3cotEvaluator",
        "max_new_tokens":  512,
        "default_batch":   3,
    },
    "safetag": {
        "hf_name":         "XuankunRong/SafeTag-VL-3K",
        "filter_fn":       lambda x: x['images'] is not None and len(x['images']) > 0,
        "train_split":     "train",
        "test_split":      "test",
        "dataset_class":   "SafeTagDataset",
        "evaluator_class": "SafeTagEvaluator",
        "max_new_tokens":  16,
        "default_batch":   3,
    },
    "mmk12": {
        "hf_name":         "chamber111/VPPO_MMK12_validation",
        "filter_fn":       lambda x: x['images'] is not None and len(x['images']) > 0,
        "train_split":     "train",
        "test_split":      "test",
        "dataset_class":   "MMK12Dataset",
        "evaluator_class": "MMK12Evaluator",
        "max_new_tokens":  512,
        "default_batch":   3,
    },
}


# =============================================================================
#  DYNAMIC IMPORTS
# =============================================================================

def get_dataset_and_evaluator(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]

    if dataset_name == "vqarad":
        from dataset import VQARadDataset as DatasetClass
        from evaluator import VQARadEvaluator as EvaluatorClass
    elif dataset_name == "stvqa":
        from dataset import STVQADataset as DatasetClass
        from evaluator import STVQAEvaluator as EvaluatorClass
    elif dataset_name == "m3cot":
        from dataset import M3cotDataset as DatasetClass
        from evaluator import M3cotEvaluator as EvaluatorClass
    elif dataset_name == "safetag":
        from dataset import SafeTagDataset as DatasetClass
        from evaluator import SafeTagEvaluator as EvaluatorClass
    elif dataset_name == "mmk12":
        from dataset import MMK12Dataset as DatasetClass
        from evaluator import MMK12Evaluator as EvaluatorClass
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: {list(DATASET_CONFIGS.keys())}")

    return DatasetClass, EvaluatorClass


# =============================================================================
#  HELPERS
# =============================================================================

def freeze_conv2d(model):
    frozen = 0
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for p in module.parameters():
                p.requires_grad = False
            frozen += 1
    print(f"  Froze {frozen} Conv2d layers (patch_embedding)")
    return frozen


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
    parser = argparse.ArgumentParser(description="VQA DoRA Fine-tuning with Token Theory")

    parser.add_argument("--dataset",       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name",    type=str,
                        default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--num_trainable", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size",    type=int, default=None)
    parser.add_argument("--grad_accum",    type=int, default=4)
    parser.add_argument("--epochs",        type=int, default=1)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--dora_r",        type=int,   default=4)
    parser.add_argument("--dora_alpha",    type=float, default=5.0)
    parser.add_argument("--dora_dropout",  type=float, default=0.0)
    parser.add_argument("--max_length",    type=int, default=2048)
    parser.add_argument("--warmup_steps",  type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--output_root",   type=str, default=None)

    args = parser.parse_args()

    cfg         = DATASET_CONFIGS[args.dataset]
    model_name  = args.model_name
    batch_size  = args.batch_size or cfg["default_batch"]
    output_root = args.output_root or f"{args.dataset}_dora"
    os.makedirs(output_root, exist_ok=True)

    file_prefix = f"{args.dataset}_dora_llavaonevision7b"

    num_trainable_list = [None] + (
        args.num_trainable or [10, 25, 50, 100, 300, 500, 700, 1000, 1500])

    # DoRA config — attention projections in language model only
    DORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    DORA_NAME_PREFIX    = 'model.language_model'

    dora_kwargs = dict(
        target_modules = DORA_TARGET_MODULES,
        name_prefix    = DORA_NAME_PREFIX,
        r              = args.dora_r,
        lora_alpha     = args.dora_alpha,
        lora_dropout   = args.dora_dropout,
    )

    print(f"\n{'='*60}")
    print(f"  VQA DoRA Fine-tuning — Token Theory")
    print(f"  Dataset      : {args.dataset}  ({cfg['hf_name']})")
    print(f"  Model        : {model_name}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Grad accum   : {args.grad_accum}")
    print(f"  LR           : {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  DoRA r       : {args.dora_r}, alpha={args.dora_alpha}, dropout={args.dora_dropout}")
    print(f"  Target mods  : {DORA_TARGET_MODULES}")
    print(f"  Name prefix  : {DORA_NAME_PREFIX}")
    print(f"  N values     : {num_trainable_list}")
    print(f"  Output       : {output_root}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    DatasetClass, EvaluatorClass = get_dataset_and_evaluator(args.dataset)

    processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    processor.image_processor.image_grid_pinpoints = [[384, 384]]
    processor.image_processor.size = {"height": 384, "width": 384}
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"Loading {args.dataset}...")
    raw_dataset = load_dataset(cfg["hf_name"])
    raw_dataset = raw_dataset.filter(cfg["filter_fn"])
    print(raw_dataset)

    if args.dataset == "mmk12":
        split     = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
        raw_train = split["train"]
        raw_test  = split["test"]
        print(f"  mmk12: split train→{len(raw_train)}, test→{len(raw_test)} (seed=42)")
    else:
        raw_train = raw_dataset[cfg["train_split"]]
        raw_test  = raw_dataset[cfg["test_split"]]

    train_dataset = DatasetClass(
        dataset    = raw_train,
        processor  = processor,
        max_length = args.max_length,
        is_test    = False,
    )
    test_dataset = DatasetClass(
        dataset    = raw_test,
        processor  = processor,
        max_length = args.max_length,
        is_test    = True,
    )

    collator = LLaVACollator(processor=processor, label_mask_mode='response_only')

    print(f"\nTrain : {len(train_dataset):,}")
    print(f"Test  : {len(test_dataset):,}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = build_csv_fields(args.dataset, gpu_fields)
    CSV_PATH   = os.path.join(output_root, f"{file_prefix}.csv")
    JSON_PATH  = os.path.join(output_root, f"{file_prefix}.json")

    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Sweep ─────────────────────────────────────────────────────────────────
    all_results = {}

    for num_trainable in num_trainable_list:
        is_baseline = (num_trainable is None)
        run_label   = "baseline_dora" if is_baseline else f"trainable_{num_trainable}"

        print(f"\n{'='*60}")
        print(f"  RUN: {run_label}")
        print(f"{'='*60}\n")

        torch.cuda.empty_cache()

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.pad_token_id         = processor.tokenizer.pad_token_id
        model.config.image_grid_pinpoints = [[384, 384]]

        # Freeze Conv2d (patch_embedding) before applying DoRA
        freeze_conv2d(model)

        # Apply DoRA — language model attention projections only
        if is_baseline:
            dora_baseline(model, **dora_kwargs)
        else:
            dora(model, num_trainable=num_trainable, **dora_kwargs)

        count_parameters(model)

        run_dir = os.path.join(output_root, run_label)
        training_args = TrainingArguments(
            output_dir                  = run_dir,
            learning_rate               = args.lr,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs            = args.epochs,
            weight_decay                = 0.00,
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

        # Merge DoRA weights + restore arch
        merge(model)

        evaluator = EvaluatorClass(
            model          = model,
            processor      = processor,
            collator       = collator,
            device         = "cuda",
            batch_size     = 16,
            max_new_tokens = cfg["max_new_tokens"],
        )
        results = evaluator.evaluate(test_dataset, split_name=run_label)

        row = build_row(args.dataset, num_trainable, is_baseline,
                        train_time, results, gpu_stats)
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
    print(f"  SUMMARY  ({args.dataset.upper()} / LLaVA-OneVision-7B / Token Theory DoRA)")
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