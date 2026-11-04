"""
Commonsense Reasoning — Token Theory full FT — LLaMA-3-8B
Unified script across all commonsense reasoning datasets.

Uses llama_tt.py / llama_token.py with LlamaPrefixForCausalLM.

Disk-conscious: NO model checkpoints saved. Only CSV/JSON results.

FIX vs previous version:
  Use an explicit balanced device_map for LlamaPrefixForCausalLM. The
  default device_map="auto" was misallocating layers as N grew (entire
  model collapsing onto GPU 0 at N>=100), causing OOM at N=500. Standard
  LlamaForCausalLM (used for the baseline and for eval) keeps "auto"
  because its accelerate heuristics work fine.

Usage:
    python commonsense_random.py --dataset arc_challenge
    python commonsense_random.py --dataset boolq --num_trainable 10 50 100
    python commonsense_random.py --dataset all
    python commonsense_random.py --dataset all --selection last
    python commonsense_random.py --dataset arc_challenge \
        --selection attention --attention_map_path /.../arc_combined_scores.pt

Auth (do this BEFORE running):
    huggingface-cli login
"""

import os
import json
import time
import csv
import gc
import argparse

os.environ["HF_DATASETS_CACHE"]  = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["HF_HOME"]            = "/ocean/projects/cis250258p/mkowsher/hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


# =============================================================================
#  DATASET REGISTRY
# =============================================================================

DATASET_CONFIGS = {
    "arc_challenge": {
        "load_train":    "load_arc_train",
        "load_test":     "load_arc_test",
        "dataset_class": "ARCDataset",
        "eval_class":    "ARCEvaluator",
        "output_root":   "random_arc_challange",
        "csv_name":      "random_arc_challange",
    },
    "arc_easy": {
        "load_train":    "load_arceasy_train",
        "load_test":     "load_arceasy_test",
        "dataset_class": "ARCEasyDataset",
        "eval_class":    "ARCEasyEvaluator",
        "output_root":   "random_arceasy",
        "csv_name":      "random_arceasy",
    },
    "boolq": {
        "load_train":    "load_boolq_train",
        "load_test":     "load_boolq_test",
        "dataset_class": "BoolQDataset",
        "eval_class":    "BoolQEvaluator",
        "output_root":   "random_boolq",
        "csv_name":      "random_boolq",
    },
    "hellaswag": {
        "load_train":    "load_hellaswag_train",
        "load_test":     "load_hellaswag_test",
        "dataset_class": "HellaSwagDataset",
        "eval_class":    "HellaSwagEvaluator",
        "output_root":   "random_hellaswag",
        "csv_name":      "random_hellaswag",
    },
    "openbookqa": {
        "load_train":    "load_openbookqa_train",
        "load_test":     "load_openbookqa_test",
        "dataset_class": "OpenBookQADataset",
        "eval_class":    "OpenBookQAEvaluator",
        "output_root":   "random_openbookqa",
        "csv_name":      "random_openbookqa",
    },
    "winogrande": {
        "load_train":    "load_winogrande_train",
        "load_test":     "load_winogrande_test",
        "dataset_class": "WinoGrandeDataset",
        "eval_class":    "WinoGrandeEvaluator",
        "output_root":   "random_winogrande",
        "csv_name":      "random_winogrande",
    },
}


def get_dataset_and_evaluator(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    import dataset as ds_module
    import evaluator as ev_module
    return (
        getattr(ds_module, cfg["load_train"]),
        getattr(ds_module, cfg["load_test"]),
        getattr(ds_module, cfg["dataset_class"]),
        getattr(ev_module, cfg["eval_class"]),
    )


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


def build_balanced_device_map(num_layers: int, num_gpus: int) -> dict:
    """
    Manual balanced device map for LlamaPrefixForCausalLM.

    Layout (e.g. 32 layers, 2 GPUs):
      GPU 0: embed_tokens, layers 0..15
      GPU 1: layers 16..31, norm, lm_head

    This avoids accelerate's auto-balancer mis-estimating layer sizes for
    custom subclasses, which was causing the model to collapse onto GPU 0
    at large N and OOM at N=500.
    """
    if num_gpus < 1:
        raise ValueError(f"need at least 1 GPU, got {num_gpus}")

    device_map = {"model.embed_tokens": 0}

    # Even split of layers
    layers_per_gpu = [num_layers // num_gpus] * num_gpus
    for i in range(num_layers % num_gpus):
        layers_per_gpu[i] += 1     # spread remainder across early GPUs

    layer_idx = 0
    for gpu in range(num_gpus):
        for _ in range(layers_per_gpu[gpu]):
            device_map[f"model.layers.{layer_idx}"] = gpu
            layer_idx += 1

    # Final norm and lm_head go on the LAST GPU (where the last layer lives,
    # so the residual stream doesn't have to cross devices for the final ops)
    last_gpu = num_gpus - 1
    device_map["model.norm"] = last_gpu
    device_map["lm_head"]    = last_gpu

    return device_map


def transfer_prefix_to_standard(prefix_model, model_name, dtype=torch.bfloat16):
    trained_state = {k: v.detach().cpu() for k, v in prefix_model.state_dict().items()}
    del prefix_model
    gc.collect()
    torch.cuda.empty_cache()

    standard = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
    )
    missing, unexpected = standard.load_state_dict(trained_state, strict=False)
    if missing or unexpected:
        print(f"  ! state_dict transfer — missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  ✓ state_dict transferred cleanly ({len(trained_state)} tensors)")
    del trained_state
    gc.collect()
    return standard


# =============================================================================
#  ONE DATASET END-TO-END
# =============================================================================

def run_one_dataset(args, dataset_name):
    cfg          = DATASET_CONFIGS[dataset_name]
    model_name   = args.model_name
    output_root  = args.output_root or cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)

    DEFAULT_NS = [None, 10, 25, 50, 100, 300, 500, 700, 900]
    num_trainable_list = (args.num_trainable or DEFAULT_NS[1:])

    print(f"\n{'='*60}")
    print(f"  Commonsense Reasoning — Token Theory ({args.selection})")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Model        : {model_name}")
    print(f"  Batch size   : {args.batch_size}  Grad accum: {args.grad_accum}")
    print(f"  LR           : {args.lr}  Epochs: {args.epochs}")
    print(f"  Selection    : {args.selection}")
    if args.selection == "attention":
        print(f"  Score file   : {args.attention_map_path}")
    print(f"  N values     : {num_trainable_list}")
    print(f"  Output       : {output_root}")
    print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ──────────────────────────────────────────────────────────────────
    load_train, load_test, DatasetClass, EvaluatorClass = \
        get_dataset_and_evaluator(dataset_name)

    print(f"Loading {dataset_name} train split...")
    raw_train = load_train()
    print(f"Loading {dataset_name} test split...")
    raw_test  = load_test()

    train_dataset = DatasetClass(raw_train, tokenizer,
                                 max_length=args.max_length, is_test=False)
    test_dataset  = DatasetClass(raw_test,  tokenizer,
                                 max_length=args.max_length, is_test=True)
    print(f"\nTrain samples : {len(train_dataset):,}")
    print(f"Test  samples : {len(test_dataset):,}\n")

    # ── Build balanced device map (computed once per dataset) ────────────────
    base_cfg_for_layers = AutoConfig.from_pretrained(model_name)
    num_layers = base_cfg_for_layers.num_hidden_layers
    num_gpus   = max(1, torch.cuda.device_count())
    balanced_dmap = build_balanced_device_map(num_layers, num_gpus)
    print(f"Balanced device map: {num_layers} layers across {num_gpus} GPU(s)")
    counts = {g: sum(1 for v in balanced_dmap.values() if v == g) for g in range(num_gpus)}
    print(f"  modules per GPU: {counts}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    gpu_fields = []
    for i in range(torch.cuda.device_count()):
        gpu_fields += [f"gpu{i}_peak_alloc_gb", f"gpu{i}_peak_reserved_gb"]
    csv_fields = ["num_trainable", "train_time_min", "accuracy", "failed"] + gpu_fields
    CSV_PATH   = os.path.join(output_root, f"{cfg['csv_name']}.csv")
    JSON_PATH  = os.path.join(output_root, f"{cfg['csv_name']}.json")
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    all_results = {}

    # ── Sweep ─────────────────────────────────────────────────────────────────
    for num_trainable in num_trainable_list:
        is_baseline = (num_trainable is None)
        run_label   = "baseline_full" if is_baseline else f"trainable_{num_trainable}"

        print(f"\n{'='*60}")
        print(f"  RUN: {dataset_name} / {run_label}")
        print(f"{'='*60}\n")

        torch.cuda.empty_cache()
        reset_gpu_memory_stats()

        # ── Build model + collator ───────────────────────────────────────────
        if is_baseline:
            print("  Mode: baseline — standard LlamaForCausalLM (device_map=auto)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
                label_pad_token_id=-100,
            )
        else:
            print(f"  Mode: TokenTune {args.selection} — LlamaPrefixForCausalLM, "
                  f"prefix_length={num_trainable}")
            print(f"  Using BALANCED device map (not auto)")

            base_cfg = AutoConfig.from_pretrained(model_name)
            prefix_cfg = LlamaConfigPrefix(**base_cfg.to_dict())
            prefix_cfg.prefix_length = num_trainable

            model = LlamaPrefixForCausalLM.from_pretrained(
                model_name,
                config=prefix_cfg,
                torch_dtype=torch.bfloat16,
                device_map=balanced_dmap,            # ← FIX: explicit even split
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
            eval_model = transfer_prefix_to_standard(model, model_name)
            eval_model.config.pad_token_id = tokenizer.pad_token_id
            eval_model.generation_config = GenerationConfig(
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # ── Evaluate ─────────────────────────────────────────────────────────
        evaluator = EvaluatorClass(
            model          = eval_model,
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

        # ── Cleanup ──────────────────────────────────────────────────────────
        del eval_model, evaluator
        gc.collect()
        torch.cuda.empty_cache()

        try:
            import shutil
            if os.path.isdir(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

        print(f"\n✓ Done: {run_label}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    print("\n" + "=" * 80)
    print(f"  SUMMARY  ({dataset_name.upper()} / LLaMA-3-8B / {args.selection})")
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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Commonsense Reasoning — Token Theory sweep")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()) + ["all"])
    parser.add_argument("--model_name",    type=str,
                        default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--num_trainable", type=int, nargs="+", default=None)
    parser.add_argument("--selection", type=str, default="random",
                        choices=["last", "random", "attention"])
    parser.add_argument("--attention_map_path", type=str, default=None)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--grad_accum",    type=int,   default=2)
    parser.add_argument("--epochs",        type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--max_length",    type=int,   default=1000)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--logging_steps", type=int,   default=50)
    parser.add_argument("--output_root",   type=str,   default='random_3b')
    args = parser.parse_args()

    if args.selection == "attention" and not args.attention_map_path:
        parser.error("--selection attention requires --attention_map_path")

    if args.dataset == "all":
        for name in DATASET_CONFIGS.keys():
            try:
                run_one_dataset(args, name)
            except Exception as e:
                print(f"\n!!! Dataset {name} failed: {type(e).__name__}: {e}")
                print("    Continuing to next dataset.\n")
                gc.collect()
                torch.cuda.empty_cache()
    else:
        run_one_dataset(args, args.dataset)


if __name__ == "__main__":
    main()