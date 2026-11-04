"""
Video QA LoRA Fine-tuning Sweep — Qwen3-VL-8B-Instruct
Token Theory LoRA fine-tuning across all minimal_video_pairs subsets.

Usage:
    python vqa_lora_qwen3vl8b_video.py --dataset human_object_interactions
    python vqa_lora_qwen3vl8b_video.py --dataset intuitive_physics
    python vqa_lora_qwen3vl8b_video.py --dataset robot_object_interactions
    python vqa_lora_qwen3vl8b_video.py --dataset temporal_reasoning
    python vqa_lora_qwen3vl8b_video.py --dataset temporal_reasoning --num_trainable 10 25 50
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
from collator import Qwen3VLCollator
from dataset import Qwen3VLVideoDataset, SYSTEM_PROMPTS
from evaluator import VideoPairsEvaluator

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
HF_DATASET = "facebook/minimal_video_pairs"
VIDEO_BASE = (
    "/ocean/projects/cis250089p/nprottasha/tokens"
    "/minimal_video_pairs/videos"
)
NUM_TRAINABLE_LIST = [None, 10, 25, 50, 100, 300, 500, 700, 1000, 1500]

# LoRA config — attention projections in language model only
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_NAME_PREFIX    = 'model.language_model'


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "human_object_interactions": {
        "hf_subset":  "human_object_interactions",
        "max_length": 2048,
    },
    "intuitive_physics": {
        "hf_subset":  "intuitive_physics",
        "max_length": 2048,
    },
    "robot_object_interactions": {
        "hf_subset":  "robot_object_interactions",
        "max_length": 2048,
    },
    "temporal_reasoning": {
        "hf_subset":  "temporal_reasoning",
        "max_length": 2048,
    },
}


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
    print(f"  Froze {frozen} Conv3d layers")
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


# =============================================================================
#  MAIN
# =============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B Video QA — Token Theory LoRA Fine-tuning")
    parser.add_argument("--dataset",       type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Which minimal_video_pairs subset to train on")
    parser.add_argument("--model_name",    type=str,   default=MODEL_NAME)
    parser.add_argument("--num_trainable", type=int,   nargs="+", default=None)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--grad_accum",    type=int,   default=4)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--lora_r",        type=int,   default=4)
    parser.add_argument("--lora_alpha",    type=float, default=5.0)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--fps",           type=float, default=1.0)
    parser.add_argument("--max_frames",    type=int,   default=8)
    parser.add_argument("--test_size",     type=float, default=0.2)
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
    print(f"  Qwen3-VL-8B Video QA — Token Theory LoRA")
    print(f"  Dataset    : {HF_DATASET} / {cfg['hf_subset']}")
    print(f"  Video base : {VIDEO_BASE}")
    print(f"  Model      : {model_name}")
    print(f"  Batch size : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR         : {args.lr}  Epochs: {args.epochs}")
    print(f"  LoRA r     : {args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Target mods: {LORA_TARGET_MODULES}")
    print(f"  Name prefix: {LORA_NAME_PREFIX}")
    print(f"  FPS        : {args.fps}  Max frames: {args.max_frames}")
    print(f"  Test split : {args.test_size*100:.0f}%")
    print(f"  N values   : {num_trainable_list}")
    print(f"  Output     : {output_root}")
    print(f"{'='*60}\n")

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    #processor.image_processor.min_pixels = 256 * 32 * 32
    #processor.image_processor.max_pixels = 512 * 32 * 32

    collator = Qwen3VLCollator(processor=processor, label_mask_mode='response_only')

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {HF_DATASET} / {cfg['hf_subset']} from HuggingFace...")
    raw = load_dataset(
        HF_DATASET,
        cfg["hf_subset"],
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    print(raw)

    raw_all = raw["mini"]

    # Filter samples whose videos are not present locally
    def video_exists(item):
        return os.path.exists(os.path.join(VIDEO_BASE, item['video_path']))

    before  = len(raw_all)
    raw_all = raw_all.filter(video_exists)
    after   = len(raw_all)
    print(f"  Videos found: {after}/{before}  ({before-after} missing skipped)")

    split_data = raw_all.train_test_split(test_size=args.test_size, seed=42)
    raw_train  = split_data["train"]
    raw_test   = split_data["test"]
    print(f"  Train: {len(raw_train):,}  Test: {len(raw_test):,}")



    dataset_kwargs = dict(
        processor     = processor,
        max_length    = cfg["max_length"],
        video_base    = VIDEO_BASE,
        fps           = args.fps,
        max_frames    = args.max_frames,
        system_prompt = SYSTEM_PROMPTS[args.dataset],
    )
    train_dataset = Qwen3VLVideoDataset(raw_train, is_test=False, **dataset_kwargs)
    test_dataset  = Qwen3VLVideoDataset(raw_test,  is_test=True,  **dataset_kwargs)

    print(f"\nTrain : {len(train_dataset):,}")
    print(f"Test  : {len(test_dataset):,}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\n=== SANITY CHECK ===")
    samples     = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
    train_batch = collator(samples)
    for k, v in train_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
    labels    = train_batch['labels'][1]
    input_ids = train_batch['input_ids'][1]
    n_train   = (labels != -100).sum().item()
    decoded   = processor.tokenizer.decode(
        input_ids[labels != -100], skip_special_tokens=False)
    print(f"  trainable tokens : {n_train} / {labels.shape[0]}")
    print('input text: ', processor.tokenizer.decode(input_ids, skip_special_tokens=False))
    print(f"  trainable text   : {repr(decoded)}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = [
        "num_trainable", "train_time_min",
        "single_accuracy", "pair_accuracy",
        "correct", "total", "failed",
        "pair_correct", "pair_total",
    ] + gpu_fields

    CSV_PATH  = os.path.join(output_root, f"{args.dataset}_lora_qwen3vl8b.csv")
    JSON_PATH = os.path.join(output_root, f"{args.dataset}_lora_qwen3vl8b.json")

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

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        # Freeze Conv3d layers (video encoder) before applying LoRA
        freeze_conv3d(model)

        # Apply LoRA — language model attention projections only
        if is_baseline:
            lora_baseline(model, **lora_kwargs)
        else:
            lora(model, num_trainable=num_trainable, **lora_kwargs)

        count_parameters(model)

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
            #max_steps                   = 40,
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

        merge(model)

        evaluator = VideoPairsEvaluator(
            model          = model,
            processor      = processor,
            collator       = collator,
            device         = "cuda",
            batch_size     = 16,
            max_new_tokens = 8,
        )
        results = evaluator.evaluate(test_dataset, split_name=run_label)

        row = {
            "num_trainable":   "baseline" if is_baseline else num_trainable,
            "train_time_min":  round(train_time / 60, 2),
            "single_accuracy": results.get("single_accuracy", 0.0),
            "pair_accuracy":   results.get("pair_accuracy",   0.0),
            "correct":         results.get("correct",         0),
            "total":           results.get("total",           0),
            "failed":          results.get("failed",          0),
            "pair_correct":    results.get("pair_correct",    0),
            "pair_total":      results.get("pair_total",      0),
            **gpu_stats,
        }
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
    print("\n" + "=" * 70)
    print(f"  SUMMARY  ({args.dataset.upper()} / Qwen3-VL-8B / Token Theory LoRA)")
    print("=" * 70)

    gpu_header = "  ".join(f"gpu{i}_alloc_gb" for i in range(num_gpus))
    print(f"  {'run':>20}  {'single%':>8}  {'pair%':>8}  {'time_min':>10}  {gpu_header}")
    print("  " + "-" * 70)

    for label, row in all_results.items():
        gpu_vals = "        ".join(
            str(row.get(f"gpu{i}_peak_alloc_gb", "N/A"))
            for i in range(num_gpus)
        )
        print(f"  {label:>20}  {row['single_accuracy']:>8.2f}  "
              f"{row['pair_accuracy']:>8.2f}  {row['train_time_min']:>10.2f}  {gpu_vals}")

    with open(JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved:\n  CSV  → {CSV_PATH}\n  JSON → {JSON_PATH}")


if __name__ == "__main__":
    main()