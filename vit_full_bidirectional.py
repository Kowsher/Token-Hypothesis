"""
ViT Image Classification — Full Fine-tuning with Bidirectional Token Theory

Datasets:
    oxford_pets   — timm/oxford-iiit-pet              (37 classes)
    chest_xray    — hf-vision/chest-xray-pneumonia    ( 2 classes)
    cifar10       — uoft-cs/cifar10                   (10 classes)
    sun397        — tanganke/sun397                   (397 classes)
    eurosat       — tanganke/eurosat                  (10 classes)
    asl_mnist     — rrrajjj/American-Sign-Language-MNIST (275 classes)
    tomato_leaves — Vlone571/tomato-leaves-dataset    ( 8 classes)

Usage:
    python vit_full_bidirectional.py --dataset oxford_pets
    python vit_full_bidirectional.py --dataset chest_xray
    python vit_full_bidirectional.py --dataset cifar10
    python vit_full_bidirectional.py --dataset sun397
    python vit_full_bidirectional.py --dataset eurosat
    python vit_full_bidirectional.py --dataset asl_mnist
    python vit_full_bidirectional.py --dataset tomato_leaves
"""

import os
import json
import time
import csv
import gc
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HF_DATASETS_CACHE"]    = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["TRANSFORMERS_CACHE"]   = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["HF_HOME"]              = "/ocean/projects/cis250089p/nprottasha/hf"

import torch
from transformers import (
    ViTForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from bidirectional import full, full_baseline, merge
from dataset import (
    OxfordPetsDataset,
    ChestXRayDataset,
    CIFAR10Dataset,
    SUN397Dataset,
    EuroSATDataset,
    ASLMNISTDataset,
    TomatoLeavesDataset,
)
from evaluator import ImageCLSEvaluator

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAME         = "google/vit-large-patch16-224"
NUM_TRAINABLE_LIST = [None, 10, 25, 50, 100, 150, 180]
TARGET_MODULES     = ['query', 'key', 'value', 'dense']
NAME_PREFIX        = 'vit.encoder'


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "oxford_pets": {
        "hf_name":     "timm/oxford-iiit-pet",
        "num_classes": 37,
        "train_split": "train",
        "test_split":  "test",
    },
    "chest_xray": {
        "hf_name":     "hf-vision/chest-xray-pneumonia",
        "num_classes": 2,
        "train_split": "train",
        "test_split":  "test",
    },
    "cifar10": {
        "hf_name":     "uoft-cs/cifar10",
        "num_classes": 10,
        "train_split": "train",
        "test_split":  "test",
    },
    "sun397": {
        "hf_name":     "tanganke/sun397",
        "num_classes": 397,
        "train_split": "train",
        "test_split":  "test",
    },
    "eurosat": {
        "hf_name":     "tanganke/eurosat",
        "num_classes": 10,
        "train_split": "train",
        "test_split":  "test",
    },
    "asl_mnist": {
        "hf_name":     "rrrajjj/American-Sign-Language-MNIST",
        "num_classes": None,   # inferred
        "train_split": "train",
        "test_split":  "test",
    },
    "tomato_leaves": {
        "hf_name":     "Vlone571/tomato-leaves-dataset",
        "num_classes": 8,
        "train_split": "train",
        "test_split":  "test",   # falls back to validation if missing
    },
}


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_datasets(args, processor):
    cfg = DATASET_CONFIGS[args.dataset]
    ds  = load_dataset(cfg["hf_name"], cache_dir=os.environ["HF_DATASETS_CACHE"])
    print(ds)

    raw_train = ds[cfg["train_split"]]

    # Determine test split — fall back to validation or 80/20
    if cfg["test_split"] in ds:
        raw_test = ds[cfg["test_split"]]
    elif "validation" in ds:
        raw_test = ds["validation"]
        print(f"  ⚠ No test split — using validation as test")
    else:
        split     = raw_train.train_test_split(test_size=0.2, seed=42)
        raw_train = split["train"]
        raw_test  = split["test"]
        print(f"  ⚠ No test/val split — split train 80/20  "
              f"(train={len(raw_train)}, test={len(raw_test)})")

    dataset_map = {
        "oxford_pets":   OxfordPetsDataset,
        "chest_xray":    ChestXRayDataset,
        "cifar10":       CIFAR10Dataset,
        "sun397":        SUN397Dataset,
        "eurosat":       EuroSATDataset,
        "asl_mnist":     ASLMNISTDataset,
        "tomato_leaves": TomatoLeavesDataset,
    }
    DatasetClass  = dataset_map[args.dataset]
    train_dataset = DatasetClass(raw_train, processor)
    test_dataset  = DatasetClass(raw_test,  processor)
    return train_dataset, test_dataset


def infer_num_classes(train_dataset):
    labels = set()
    for i in range(min(len(train_dataset), 500)):
        labels.add(train_dataset[i]["labels"].item())
    return max(labels) + 1


# =============================================================================
#  HELPERS
# =============================================================================

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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ViT Image Classification — Bidirectional Token Theory Full")
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_name",    type=str,   default=MODEL_NAME)
    parser.add_argument("--num_trainable", type=int,   nargs="+", default=None)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--grad_accum",    type=int,   default=2)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--logging_steps", type=int,   default=20)
    parser.add_argument("--output_root",   type=str,   default=None)
    args = parser.parse_args()

    cfg         = DATASET_CONFIGS[args.dataset]
    model_name  = args.model_name
    output_root = args.output_root or f"{args.dataset}_vit_full"
    os.makedirs(output_root, exist_ok=True)

    num_trainable_list = [None] + (args.num_trainable or NUM_TRAINABLE_LIST[1:])

    print(f"\n{'='*60}")
    print(f"  ViT Image Classification — Bidirectional Token Theory Full")
    print(f"  Dataset    : {args.dataset}  ({cfg['hf_name']})")
    print(f"  Model      : {model_name}")
    print(f"  Batch size : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR         : {args.lr}  Epochs: {args.epochs}")
    print(f"  N values   : {num_trainable_list}")
    print(f"  Output     : {output_root}")
    print(f"{'='*60}\n")

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoImageProcessor.from_pretrained(model_name)

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_dataset, test_dataset = load_datasets(args, processor)

    print(f"\n  Train : {len(train_dataset):,}")
    print(f"  Test  : {len(test_dataset):,}")

    num_classes = cfg["num_classes"] or infer_num_classes(train_dataset)
    print(f"  Classes : {num_classes}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = ["num_trainable", "train_time_min",
                  "accuracy", "correct", "total", "failed"] + gpu_fields

    CSV_PATH  = os.path.join(output_root, f"{args.dataset}_vit_full.csv")
    JSON_PATH = os.path.join(output_root, f"{args.dataset}_vit_full.json")

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

        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels              = num_classes,
            ignore_mismatched_sizes = True,
            torch_dtype             = torch.bfloat16,
            use_safetensors         = True,
        ).to("cuda")

        if is_baseline:
            full_baseline(model)
        else:
            full(model,
                 num_trainable  = num_trainable,
                 target_modules = TARGET_MODULES,
                 name_prefix    = NAME_PREFIX,
                 skip_frozen    = True)

        count_parameters(model)
        reset_gpu_memory_stats()

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
            dataloader_num_workers      = 4,
        )

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = train_dataset,
        )

        train_start = time.time()
        trainer.train()
        train_time  = time.time() - train_start

        print(f"\n  Training time: {train_time:.1f}s  ({train_time/60:.2f} min)")
        gpu_stats = get_gpu_memory_stats()
        reset_gpu_memory_stats()

        merge(model)

        evaluator = ImageCLSEvaluator(
            model      = model,
            device     = "cuda",
            batch_size = 64,
        )
        results = evaluator.evaluate(test_dataset, split_name=run_label)

        row = {
            "num_trainable":  "baseline" if is_baseline else num_trainable,
            "train_time_min": round(train_time / 60, 2),
            "accuracy":       results.get("accuracy", 0.0),
            "correct":        results.get("correct",  0),
            "total":          results.get("total",    0),
            "failed":         results.get("failed",   0),
            **gpu_stats,
        }
        all_results[run_label] = {
            **row,
            "per_class_acc": results.get("per_class_acc", {}),
        }

        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(
                {k: v for k, v in row.items() if k in csv_fields})
        print(f"  ✓ Row saved → {CSV_PATH}")

        trainer.optimizer    = None
        trainer.lr_scheduler = None
        del model, trainer, evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n✓ Done: {run_label}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  SUMMARY  ({args.dataset.upper()} / ViT-Large / Token Theory Full)")
    print("=" * 70)
    print(f"  {'run':>20}  {'accuracy':>10}  {'correct':>8}  {'total':>8}  {'time_min':>10}")
    print("  " + "-" * 65)

    for label, row in all_results.items():
        print(f"  {label:>20}  {row['accuracy']:>10.4f}  "
              f"{row['correct']:>8}  {row['total']:>8}  "
              f"{row['train_time_min']:>10.2f}")

    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved:\n  CSV  → {CSV_PATH}\n  JSON → {JSON_PATH}")


if __name__ == "__main__":
    main()