"""
Full VQA Training Sweep — Qwen3-VL-8B-Instruct — Video-LevelGauge
Token Theory full fine-tuning on Cola-any/Video-LevelGauge.

Usage:
    python vqa_full_qwen3vl8b_levelgauge.py
    python vqa_full_qwen3vl8b_levelgauge.py --num_trainable 10 25 50
    python vqa_full_qwen3vl8b_levelgauge.py --batch_size 2 --grad_accum 8
"""

import os
import json
import time
import csv
import gc
import argparse

# ── Must be set before any vision/video imports ───────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]      = "0, 1"
os.environ["HF_DATASETS_CACHE"]         = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["TRANSFORMERS_CACHE"]        = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["HF_HOME"]                   = "/ocean/projects/cis250089p/nprottasha/hf"
os.environ["OPENCV_LOG_LEVEL"]          = "ERROR"
os.environ["DECORD_LOG_LEVEL"]          = "ERROR"
os.environ["FFMPEG_LOG_LEVEL"]          = "quiet"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
os.environ["DECORD_EOF_RETRY_MAX"]      = "20480"

import logging
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("decord").setLevel(logging.ERROR)

import numpy as np
import torch
import decord
import qwen_vl_utils.vision_process as _qvp

# ── Monkey-patch video backends — forces decord for ALL reads ─────────────────
def _read_video_decord_safe(ele):
    video_path = ele["video"]
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[7:]

    fps        = ele.get("fps", 1.0)
    max_frames = ele.get("max_frames", 8)

    try:
        vr           = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps    = vr.get_avg_fps()

        n_frames = min(max_frames, max(1, int(total_frames / video_fps * fps)))
        indices  = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        indices  = np.clip(indices, 0, max(0, total_frames - 2)).tolist()
        frames   = vr.get_batch(indices).asnumpy()               # (T, H, W, C)
        frames   = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

    except Exception as e:
        print(f"  ⚠ Runtime video fallback: {os.path.basename(video_path)} — {e}")
        frames    = torch.zeros(max_frames, 3, 224, 224, dtype=torch.uint8)
        video_fps = fps

    sample_fps = fps
    video_meta = {"video_fps": video_fps, "total_frames": frames.shape[0]}
    return frames, video_meta, sample_fps

_qvp.VIDEO_READER_BACKENDS["torchvision"] = _read_video_decord_safe
_qvp.VIDEO_READER_BACKENDS["decord"]      = _read_video_decord_safe
print("  ✓ Video reader patched → decord (swscaler bypass)")
# ─────────────────────────────────────────────────────────────────────────────

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from causal import full, full_baseline, merge
from collator import Qwen3VLCollator
from dataset import VideoLevelGaugeDataset, load_levelgauge
from evaluator import VideoLevelGaugeEvaluator

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAME         = "Qwen/Qwen3-VL-8B-Instruct"
HF_DATASET         = "Cola-any/Video-LevelGauge"
VIDEO_BASE         = "/ocean/projects/cis250089p/nprottasha/hf/Video-LevelGauge/LevelGauge/videos"
NUM_TRAINABLE_LIST = [2000, 3500, 3000]


# =============================================================================
#  HELPERS
# =============================================================================

def filter_bad_videos(raw_dataset, video_base, fps=1.0, max_frames=8):
    """
    Aggressively probe each video with decord to remove bad samples before
    training. Replicates the same frame indices used during training so any
    frame that would be sampled is tested here first.
    """
    print(f"  Scanning {len(raw_dataset)} videos...")
    good_indices = []
    bad_count    = 0

    for i in range(len(raw_dataset)):
        # video_name already contains the extension
        video_path = os.path.join(video_base, raw_dataset[i]["video_name"])

        if not os.path.exists(video_path):
            print(f"  ✗ Missing : {raw_dataset[i]['video_name']}")
            bad_count += 1
            continue

        try:
            vr           = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            video_fps    = vr.get_avg_fps()

            n_frames = min(max_frames, max(1, int(total_frames / video_fps * fps)))
            indices  = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            indices  = np.clip(indices, 0, max(0, total_frames - 2)).tolist()

            # Also probe near the end where EOF usually hits
            extra = [max(0, total_frames - 3), max(0, total_frames - 2)]
            probe = list(set(indices + extra))

            _ = vr.get_batch(probe)
            good_indices.append(i)

        except Exception as e:
            print(f"  ✗ Bad video: {raw_dataset[i]['video_name']} — {e}")
            bad_count += 1

    print(f"  ✓ {len(good_indices)} good  |  ✗ {bad_count} removed")
    return raw_dataset.select(good_indices)


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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B Video-LevelGauge — Token Theory full fine-tuning")
    parser.add_argument("--model_name",     type=str,   default=MODEL_NAME)
    parser.add_argument("--num_trainable",  type=int,   nargs="+", default=None,
                        help="Override default sweep list")
    parser.add_argument("--batch_size",     type=int,   default=1)
    parser.add_argument("--grad_accum",     type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--fps",            type=float, default=1.0)
    parser.add_argument("--max_frames",     type=int,   default=8)
    parser.add_argument("--max_length",     type=int,   default=4096)
    parser.add_argument("--test_size",      type=float, default=0.2)
    parser.add_argument("--warmup_steps",   type=int,   default=100)
    parser.add_argument("--logging_steps",  type=int,   default=20)
    parser.add_argument("--max_new_tokens", type=int,   default=64)
    parser.add_argument("--output_root",    type=str,   default=None)
    args = parser.parse_args()

    model_name  = args.model_name
    output_root = args.output_root or "levelgauge_full_qwen3vl"
    os.makedirs(output_root, exist_ok=True)

    num_trainable_list = [None] + (args.num_trainable or NUM_TRAINABLE_LIST[1:])

    print(f"\n{'='*60}")
    print(f"  Qwen3-VL-8B Video-LevelGauge — Token Theory Full")
    print(f"  Dataset    : {HF_DATASET}")
    print(f"  Video base : {VIDEO_BASE}")
    print(f"  Model      : {model_name}")
    print(f"  Batch size : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR         : {args.lr}  Epochs: {args.epochs}")
    print(f"  FPS        : {args.fps}  Max frames: {args.max_frames}")
    print(f"  Max length : {args.max_length}")
    print(f"  Test split : {args.test_size*100:.0f}%")
    print(f"  N values   : {num_trainable_list}")
    print(f"  Output     : {output_root}")
    print(f"{'='*60}\n")

    # ── Processor ─────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    collator = Qwen3VLCollator(processor=processor, label_mask_mode='response_only')

    # ── Load & split dataset ──────────────────────────────────────────────────
    print(f"Loading {HF_DATASET} from HuggingFace...")
    ds = load_dataset(
        HF_DATASET,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    print(ds)

    raw_train, raw_test = load_levelgauge(ds, test_size=args.test_size, seed=42)

    # ── Filter bad/corrupted videos before training ───────────────────────────
    print("\nFiltering bad videos from train split...")
    raw_train = filter_bad_videos(
        raw_train, VIDEO_BASE, fps=args.fps, max_frames=args.max_frames)
    print("\nFiltering bad videos from test split...")
    raw_test  = filter_bad_videos(
        raw_test,  VIDEO_BASE, fps=args.fps, max_frames=args.max_frames)

    print(f"\n  Train (clean): {len(raw_train):,}")
    print(f"  Test  (clean): {len(raw_test):,}")

    dataset_kwargs = dict(
        processor  = processor,
        video_base = VIDEO_BASE,
        max_length = args.max_length,
        fps        = args.fps,
        max_frames = args.max_frames,
    )
    train_dataset = VideoLevelGaugeDataset(raw_train, is_test=False, **dataset_kwargs)
    test_dataset  = VideoLevelGaugeDataset(raw_test,  is_test=True,  **dataset_kwargs)

    print(f"\n  Train dataset : {len(train_dataset):,}")
    print(f"  Test  dataset : {len(test_dataset):,}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\n=== SANITY CHECK ===")
    samples   = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
    meta_keys = ["answer", "gt_answer", "question_id", "question_type", "video_category"]
    clean     = [{k: v for k, v in s.items() if k not in meta_keys} for s in samples]
    train_batch = collator(clean)
    for k, v in train_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
    labels    = train_batch['labels'][0]
    input_ids = train_batch['input_ids'][0]
    n_train   = (labels != -100).sum().item()
    decoded   = processor.tokenizer.decode(
        input_ids[labels != -100], skip_special_tokens=False)
    print(f"  trainable tokens : {n_train} / {labels.shape[0]}")
    print(f"  trainable text   : {repr(decoded[:120])}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]

    csv_fields = [
        "num_trainable", "train_time_min",
        "accuracy", "correct", "total", "failed",
    ] + gpu_fields

    CSV_PATH  = os.path.join(output_root, "levelgauge_full_qwen3vl8b.csv")
    JSON_PATH = os.path.join(output_root, "levelgauge_full_qwen3vl8b.json")

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

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        freeze_conv3d(model)

        if is_baseline:
            full_baseline(model)
        else:
            full(model, num_trainable=num_trainable,
                 target_modules='all', skip_frozen=True)

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
            dataloader_num_workers      = 0,
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

        evaluator = VideoLevelGaugeEvaluator(
            model          = model,
            processor      = processor,
            collator       = collator,
            device         = "cuda",
            batch_size     = 4,
            max_new_tokens = args.max_new_tokens,
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
            "question_type_accuracy":   results.get("question_type_accuracy",   {}),
            "video_category_accuracy":  results.get("video_category_accuracy",  {}),
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
    print(f"  SUMMARY  (Video-LevelGauge / Qwen3-VL-8B / Token Theory Full)")
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