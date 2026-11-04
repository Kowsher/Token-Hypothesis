import re
import math
import torch
from tqdm import tqdm
import re as _re
from collections import defaultdict
 

class M3cotEvaluator:
    """Evaluator for VQA datasets using LLaVA-OneVision."""

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.collator = collator

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract the letter after 'Final Answer:'"""
        match = re.search(r"Final Answer:\s*([A-F])", text)
        return match.group(1).strip() if match else None

    def _run_batch(self, samples):
        """Run generation for a batch of samples"""

        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]

        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        """Evaluate the dataset"""

        self.model.eval()

        correct = 0
        total = 0
        failed = 0

        dataset_size = len(test_dataset)
        num_batches = math.ceil(dataset_size / self.batch_size)

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):

            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)

                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt)

                total += 1

        accuracy = correct / max(total - failed, 1)

        results = {
            "split": split_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "failed": failed,
        }

        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")

        return results




class STVQAEvaluator:
    """Evaluator for ST-VQA (M3CoT) datasets using LLaVA-OneVision."""

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract answer letter from model output.

        Handles all formats the model might produce:
            <answer>(A) in</answer>   → "A"
            <answer>A</answer>        → "A"
            <answer>(A)</answer>      → "A"
            Answer: (A)               → "A"  (fallback)
        """
        # primary: <answer> tag
        match = re.search(r"<answer>\s*\(?([A-F])\)?", text)
        if match:
            return match.group(1).strip()

        # fallback: bare letter near end of text
        match = re.search(r"\b([A-F])\b(?!.*\b[A-F]\b)", text)
        if match:
            return match.group(1).strip()

        return None

    def _run_batch(self, samples):
        """Run generation for a batch of samples."""
        ground_truths = [s.pop("answer") for s in samples]
        batch         = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]

        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        """Evaluate the dataset."""
        self.model.eval()

        correct      = 0
        total        = 0
        failed       = 0
        dataset_size = len(test_dataset)
        num_batches  = math.ceil(dataset_size / self.batch_size)

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):
            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)

                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted.upper() == gt.upper())

                total += 1

        accuracy = correct / max(total - failed, 1)

        results = {
            "split":    split_name,
            "accuracy": accuracy,
            "correct":  correct,
            "total":    total,
            "failed":   failed,
        }

        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")

        return results
    


class MMK12Evaluator:
    """Evaluator for MMK12 dataset using LLaVA-OneVision."""

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 32,   # answers are short — just a letter
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract answer letter from model output.

        Handles:
            <answer>(A)</answer>   → "A"
            <answer>A</answer>     → "A"
            (A)                    → "A"  fallback
            A                      → "A"  fallback
        """
        # primary: <answer> tag
        match = re.search(r"<answer>\s*\(?([A-D])\)?", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # fallback: (X) anywhere
        match = re.search(r"\(([A-D])\)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # fallback: bare letter
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()

        return None

    def _run_batch(self, samples):
        """Run generation for a batch of samples."""
        ground_truths = [s.pop("answer") for s in samples]
        batch         = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        """Evaluate the dataset."""
        self.model.eval()

        correct      = 0
        total        = 0
        failed       = 0
        dataset_size = len(test_dataset)
        num_batches  = math.ceil(dataset_size / self.batch_size)

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):
            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)

                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt.upper())

                total += 1

        accuracy = correct / max(total - failed, 1)

        results = {
            "split":    split_name,
            "accuracy": accuracy,
            "correct":  correct,
            "total":    total,
            "failed":   failed,
        }

        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")

        return results
    
    
    

class SafeTagEvaluator:
    """Evaluator for SafeTag-VL (binary safe/unsafe classification)."""

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract safe/unsafe from model output.

            <answer>unsafe</answer>  → "unsafe"
            <answer>safe</answer>    → "safe"
            unsafe                   → "unsafe"  fallback
            safe                     → "safe"    fallback
        """
        match = re.search(r"<answer>\s*(safe|unsafe)\s*</answer>", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # check unsafe first — longer match, less ambiguous
        if re.search(r"\bunsafe\b", text, re.IGNORECASE):
            return "unsafe"
        if re.search(r"\bsafe\b", text, re.IGNORECASE):
            return "safe"

        return None

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch         = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        self.model.eval()

        correct      = 0
        total        = 0
        failed       = 0
        dataset_size = len(test_dataset)
        num_batches  = math.ceil(dataset_size / self.batch_size)

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):
            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)

                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt.lower())

                total += 1

        accuracy = correct / max(total - failed, 1)

        results = {
            "split":    split_name,
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
        }

        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")

        return results
    
    

class VisualSphinxEvaluator:
    """Evaluator for VisualSphinx-Seeds dataset using LLaVA-OneVision."""

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 16,   # answer is just a letter
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract answer letter from model output.

            <answer>(C)</answer>  → "C"
            <answer>C</answer>    → "C"
            (C)                   → "C"  fallback
            C                     → "C"  fallback
        """
        # primary: <answer> tag
        match = re.search(r"<answer>\s*\(?([A-D])\)?", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # fallback: (X) anywhere
        match = re.search(r"\(([A-D])\)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # fallback: bare letter
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()

        return None

    def _run_batch(self, samples):
        """Run generation for a batch of samples."""
        ground_truths = [s.pop("answer") for s in samples]
        batch         = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        """Evaluate the dataset."""
        self.model.eval()

        correct      = 0
        total        = 0
        failed       = 0
        dataset_size = len(test_dataset)
        num_batches  = math.ceil(dataset_size / self.batch_size)

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):
            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)

                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt.upper())

                total += 1

        accuracy = correct / max(total - failed, 1)

        results = {
            "split":    split_name,
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
        }

        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")

        return results
    


class VQARadEvaluator:
    """
    Evaluator for VQA-RAD dataset using LLaVA-OneVision.
    Uses exact match (case-insensitive, stripped).
    Also separately reports closed (yes/no) vs open-ended question accuracy.
    """

    def __init__(
        self,
        model,
        processor,
        collator,
        device: str = "cuda",
        batch_size: int = 4,
        max_new_tokens: int = 64,
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def normalize(text: str) -> str:
        """Lowercase, strip whitespace, punctuation, and any answer tags."""
        # strip <answer> and </answer> tags anywhere in the text
        text = re.sub(r"</?\s*answer\s*>", "", text, flags=re.IGNORECASE)
        return text.strip().lower().strip('.,!?;:')

    @staticmethod
    def extract_answer(text: str) -> str:
        """
        Extract answer from model output.

        Handles all formats:
            <answer>yes</answer>  → "yes"   (full tag)
            yes</answer>          → "yes"   (seeded: model continues from <answer> seed)
            yes                   → "yes"   (no tags at all)
        """
        if not text or not text.strip():
            return None

        # normalize strips all tags — just return first line
        cleaned = VQARadEvaluator.normalize(text)
        first_line = cleaned.split('\n')[0].strip()
        return first_line if first_line else None

    @staticmethod
    def is_closed_question(answer: str) -> bool:
        return answer.strip().lower() in ("yes", "no")

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch         = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)

        image_sizes = batch.get("image_sizes")
        if image_sizes is not None:
            image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[
                    self.im_end_token_id,
                    self.processor.tokenizer.eos_token_id,
                ],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name: str = "Test") -> dict:
        self.model.eval()

        total          = 0
        failed         = 0
        correct        = 0
        closed_correct = 0
        closed_total   = 0
        open_correct   = 0
        open_total     = 0

        dataset_size      = len(test_dataset)
        num_batches       = math.ceil(dataset_size / self.batch_size)
        first_batch_logged = False

        for start in tqdm(
            range(0, dataset_size, self.batch_size),
            total=num_batches,
            desc=f"Evaluating {split_name}",
            leave=True,
        ):
            samples = [
                test_dataset[i]
                for i in range(start, min(start + self.batch_size, dataset_size))
            ]

            decoded, ground_truths = self._run_batch(samples)

            # debug: log first 3 predictions
            if not first_batch_logged:
                print("\n=== DEBUG: First batch predictions ===")
                for pred_text, gt in zip(decoded[:3], ground_truths[:3]):
                    extracted = self.extract_answer(pred_text)
                    print(f"  GT       : {repr(gt)}")
                    print(f"  RAW PRED : {repr(pred_text[:120])}")
                    print(f"  EXTRACTED: {repr(extracted)}")
                    print()
                first_batch_logged = True

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                gt_norm   = self.normalize(gt)

                if predicted is None:
                    failed += 1
                    total  += 1
                    continue

                is_correct = int(predicted == gt_norm)
                correct   += is_correct
                total     += 1

                if self.is_closed_question(gt_norm):
                    closed_correct += is_correct
                    closed_total   += 1
                else:
                    open_correct += is_correct
                    open_total   += 1

        evaluated  = total - failed
        accuracy   = correct        / max(evaluated,    1)
        closed_acc = closed_correct / max(closed_total, 1)
        open_acc   = open_correct   / max(open_total,   1)

        results = {
            "split":        split_name,
            "accuracy":     round(accuracy,   4),
            "closed_acc":   round(closed_acc, 4),
            "open_acc":     round(open_acc,   4),
            "correct":      correct,
            "total":        total,
            "failed":       failed,
            "closed_total": closed_total,
            "open_total":   open_total,
        }

        print(f"\n=== [{split_name}] Results ===")
        print(f"  Overall accuracy : {accuracy:.4f}  ({correct}/{evaluated})")
        print(f"  Closed (yes/no)  : {closed_acc:.4f}  ({closed_correct}/{closed_total})")
        print(f"  Open-ended       : {open_acc:.4f}  ({open_correct}/{open_total})")
        print(f"  Failed           : {failed}/{total}")

        return results
    
    
    
    
    
    
"""
Commonsense Reasoning Evaluator
Evaluates accuracy on 8 test sets individually after a single training run.

Test sets:
    - ARC-Challenge
    - ARC-Easy
    - BoolQ
    - HellaSwag
    - OpenBookQA
    - PIQA
    - Social_i_QA
    - WinoGrande
"""
import warnings
warnings.filterwarnings("ignore")
import re
import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import CommonsenseDataset, load_json_from_url, load_json_from_path


# ─── Test set URLs ────────────────────────────────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters"
    "/refs/heads/main/dataset"
)

TEST_URLS = {
    "WinoGrande":    f"{BASE_URL}/winogrande/test.json",
    "ARC-Challenge": f"{BASE_URL}/ARC-Challenge/test.json",
    "ARC-Easy":      f"{BASE_URL}/ARC-Easy/test.json",
    "BoolQ":         f"{BASE_URL}/boolq/test.json",
    "HellaSwag":     f"{BASE_URL}/hellaswag/test.json",
    "OpenBookQA":    f"{BASE_URL}/openbookqa/test.json",
    "PIQA":          f"{BASE_URL}/piqa/test.json",
    "Social_i_QA":   f"{BASE_URL}/social_i_qa/test.json",

}


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and strip whitespace/punctuation."""
    return text.strip().lower().rstrip(".")


def extract_answer(generated: str) -> str:
    """
    Extract the answer token from generated text.

    Handles formats:
        "the correct answer is answer3"
        "the correct answer is true"
        "the correct answer is solution1"
        "the correct answer is option2"
        "the correct answer is ending4"
    """
    text = normalize(generated)

    # Try to find "the correct answer is <answer>"
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")

    # Fallback: last token
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for text-only generation batches ───────────────────────────────

def _test_collate(batch: List[Dict]) -> Dict:
    """Collate test samples — stack tensors, keep answers and prompts as lists."""
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"]  for b in batch]
    prompts        = [b["prompt"]  for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Single-dataset evaluator ─────────────────────────────────────────────────

@dataclass
class EvalResult:
    dataset:   str
    accuracy:  float
    correct:   int
    total:     int
    failed:    int


class CommonsenseEvaluator:
    """
    Evaluates a single commonsense test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(
        self,
        dataset: CommonsenseDataset,
        dataset_name: str,
    ) -> EvalResult:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate,
        )

        correct = 0
        failed  = 0
        total   = 0

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    # Decode only newly generated tokens
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        result   = EvalResult(
            dataset=dataset_name,
            accuracy=accuracy,
            correct=correct,
            total=total,
            failed=failed,
        )

        print(f"  [{dataset_name}]  accuracy={accuracy:.4f}  "
              f"correct={correct}/{total}  failed={failed}")
        return result


# ─── Multi-dataset evaluator ──────────────────────────────────────────────────

class CommonsenseMultiEvaluator:
    """
    Evaluates all 8 commonsense test sets after a single training run.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate
        max_length:     Max tokenisation length for test examples
        local_paths:    Optional dict of {dataset_name: local_json_path}
                        to avoid downloading. Missing keys fall back to URL.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
        max_length: int = 512,
        local_paths: Optional[Dict[str, str]] = None,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length     = max_length
        self.local_paths    = local_paths or {}

        self._evaluator = CommonsenseEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

    def _load_test(self, name: str) -> CommonsenseDataset:
        if name in self.local_paths:
            print(f"  Loading {name} from {self.local_paths[name]}")
            data = load_json_from_path(self.local_paths[name])
        else:
            url  = TEST_URLS[name]
            print(f"  Downloading {name} from {url}")
            data = load_json_from_url(url)
        return CommonsenseDataset(
            data, self.tokenizer,
            max_length=self.max_length,
            is_test=True,
        )

    def evaluate_all(
        self,
        datasets: Optional[List[str]] = None,
    ) -> Dict[str, EvalResult]:
        """
        Evaluate on all (or a subset of) test sets.

        Args:
            datasets: List of dataset names to evaluate.
                      Defaults to all 8 if None.

        Returns:
            Dict of {dataset_name: EvalResult}
        """
        names   = datasets or list(TEST_URLS.keys())
        results = {}

        print(f"\n{'─'*60}")
        print(f"  Evaluating {len(names)} commonsense test sets")
        print(f"{'─'*60}")

        for name in names:
            test_ds = self._load_test(name)
            result  = self._evaluator.evaluate(test_ds, name)
            results[name] = result

        # Summary
        print(f"\n{'='*60}")
        print(f"  COMMONSENSE EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Dataset':<16}  {'Accuracy':>10}  {'Correct':>8}  {'Total':>6}")
        print(f"  {'-'*52}")
        accs = []
        for name, r in results.items():
            print(f"  {name:<16}  {r.accuracy:>10.4f}  {r.correct:>8}  {r.total:>6}")
            accs.append(r.accuracy)
        avg = sum(accs) / len(accs) if accs else 0.0
        print(f"  {'-'*52}")
        print(f"  {'Average':<16}  {avg:>10.4f}")
        print(f"{'='*60}\n")

        return results

    def results_to_dict(self, results: Dict[str, EvalResult]) -> dict:
        """Convert EvalResult objects to plain dict for JSON serialisation."""
        out = {}
        for name, r in results.items():
            out[name] = {
                "accuracy": round(r.accuracy, 4),
                "correct":  r.correct,
                "total":    r.total,
                "failed":   r.failed,
            }
        accs = [r.accuracy for r in results.values()]
        out["average_accuracy"] = round(sum(accs) / len(accs), 4) if accs else 0.0
        return out
    
    
    
"""
ARC-Challenge Evaluator

Evaluates accuracy on the ARC-Challenge test set after training.
Expected answer format: "answerX" (answer1, answer2, answer3, answer4)
"""

import re
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ARCDataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is answer3" → "answer3"
    """
    text = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    # fallback: last token
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class ARCEvaluator:
    """
    Evaluator for ARC-Challenge test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: ARCDataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    # Decode only newly generated tokens
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [ARC-Challenge | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
ARC-Easy Evaluator

Evaluates accuracy on the ARC-Easy test set after training.
Expected answer format: "answerX" (answer1, answer2, answer3, answer4)
"""


from dataset import ARCEasyDataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is answer3" → "answer3"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_arceasy(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class ARCEasyEvaluator:
    """
    Evaluator for ARC-Easy test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: ARCEasyDataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_arceasy,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [ARC-Easy | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
BoolQ Evaluator

Evaluates accuracy on the BoolQ test set after training.
Expected answer format: "true" or "false"
"""


from dataset import BoolQDataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is true" → "true"
             "the correct answer is false" → "false"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_boolq(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class BoolQEvaluator:
    """
    Evaluator for BoolQ test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: BoolQDataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_boolq,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [BoolQ | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
"""
HellaSwag Evaluator

Evaluates accuracy on the HellaSwag test set after training.
Expected answer format: "endingX" (ending1, ending2, ending3, ending4)
"""


from dataset import HellaSwagDataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is ending3" → "ending3"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_hellaswag(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class HellaSwagEvaluator:
    """
    Evaluator for HellaSwag test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: HellaSwagDataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_hellaswag,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [HellaSwag | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
OpenBookQA Evaluator

Evaluates accuracy on the OpenBookQA test set after training.
Expected answer format: "answerX" (answer1, answer2, answer3, answer4)
"""



from dataset import OpenBookQADataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is answer2" → "answer2"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_openbookqa(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class OpenBookQAEvaluator:
    """
    Evaluator for OpenBookQA test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: OpenBookQADataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_openbookqa,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [OpenBookQA | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
Social_i_QA Evaluator

Evaluates accuracy on the Social_i_QA test set after training.
Expected answer format: "answerX" (answer1, answer2, answer3)
"""


from dataset import SocialIQADataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is answer2" → "answer2"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_socialiqa(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class SocialIQAEvaluator:
    """
    Evaluator for Social_i_QA test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: SocialIQADataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_socialiqa,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [Social_i_QA | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
Social_i_QA Evaluator

Evaluates accuracy on the Social_i_QA test set after training.
Expected answer format: "answerX" (answer1, answer2, answer3)
"""



from dataset import SocialIQADataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is answer2" → "answer2"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_socialiqa(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class SocialIQAEvaluator:
    """
    Evaluator for Social_i_QA test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: SocialIQADataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_socialiqa,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [Social_i_QA | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
"""
WinoGrande Evaluator

Evaluates accuracy on the WinoGrande test set after training.
Expected answer format: "option1" or "option2"
"""


from dataset import WinoGrandeDataset


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def extract_answer(generated: str) -> str:
    """
    Extract answer token from generated text.
    Handles: "the correct answer is option1" → "option1"
    """
    text  = normalize(generated)
    match = re.search(r"the correct answer is\s+(\S+)", text)
    if match:
        return match.group(1).rstrip(".,;")
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_winogrande(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Evaluator ────────────────────────────────────────────────────────────────

class WinoGrandeEvaluator:
    """
    Evaluator for WinoGrande test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (default 16 — answers are short)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_new_tokens: int = 16,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def evaluate(self, dataset: WinoGrandeDataset, split_name: str = "test") -> Dict:
        """
        Run generation on all test samples and compute accuracy.

        Returns:
            dict with accuracy, correct, total, failed, split
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_winogrande,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer(pred_text)
                    if normalize(pred) == normalize(gold):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0

        print(f"\n  [WinoGrande | {split_name}]  "
              f"accuracy={accuracy:.4f}  correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "split":    split_name,
        }
        
        
        
        
"""
Math Reasoning Multi-Evaluator

Evaluates accuracy on 7 math test sets after a single training run.
- Numeric answers : correct if |pred - gold| <= tolerance (default 0.1)
- Letter answers  : exact match after normalization (MathQA: A/B/C/D/E)

Test sets: AddSub, MultiArith, SVAMP, SingleEq, GSM8K, MathQA, MAWPS
"""


from dataset import MathDataset, load_math_test, TEST_URLS_MATH


# ─── Answer extraction ────────────────────────────────────────────────────────

def normalize_str(text: str) -> str:
    return text.strip().lower().rstrip(".,;")


def try_parse_float(text: str) -> Optional[float]:
    """Try to parse text as float. Returns None if not possible."""
    try:
        # strip common suffixes like "tickets", "apples", etc.
        text = text.strip().rstrip(".,;")
        return float(text)
    except ValueError:
        return None


def extract_answer_math(generated: str) -> str:
    """
    Extract the final answer from generated math text.

    Priority:
    1. "The answer is X"  — most common pattern in training data
    2. Last number in the text — fallback for numeric answers
    3. Last token — fallback for letter answers (A/B/C/D/E)
    """
    text = generated.strip()

    # Pattern: "The answer is X" or "the answer is X"
    match = re.search(r"[Tt]he answer is\s*[:\-]?\s*([A-Ea-e\d\.\-]+)", text)
    if match:
        return match.group(1).rstrip(".,;")

    # Pattern: last number in text (handles "= 16 tickets" etc.)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    # Fallback: last token
    tokens = text.split()
    return tokens[-1].rstrip(".,;") if tokens else ""


def is_correct(pred: str, gold: str, tolerance: float = 0.1) -> bool:
    """
    Compare predicted answer to gold answer.
    - If both parse as float: correct if |pred - gold| <= tolerance
    - Otherwise: exact string match after normalization
    """
    pred_float = try_parse_float(pred)
    gold_float = try_parse_float(gold)

    if pred_float is not None and gold_float is not None:
        return abs(pred_float - gold_float) <= tolerance

    # Letter answer or non-numeric
    return normalize_str(pred) == normalize_str(gold)


# ─── Collator for test batches ────────────────────────────────────────────────

def _test_collate_math(batch: List[Dict]) -> Dict:
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    answers        = [b["answer"] for b in batch]
    prompts        = [b["prompt"] for b in batch]
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "answer":         answers,
        "prompt":         prompts,
    }


# ─── Single-dataset evaluator ─────────────────────────────────────────────────

class MathEvaluator:
    """
    Evaluator for a single math test set.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate (512 — math needs CoT)
        tolerance:      Numeric answer tolerance (default 0.1)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 512,
        tolerance: float = 0.1,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.tolerance      = tolerance

    def evaluate(self, dataset: MathDataset, dataset_name: str) -> Dict:
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_test_collate_math,
        )

        correct = 0
        failed  = 0
        total   = 0

        pbar = tqdm(loader, desc=f"Evaluating [{dataset_name}]", unit="batch")

        with torch.no_grad():
            for batch in pbar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gold_answers   = batch["answer"]

                try:
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_tokens = generated_ids[:, input_ids.shape[1]:]
                    decoded    = self.tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    failed += len(gold_answers)
                    total  += len(gold_answers)
                    pbar.set_postfix(accuracy=0.0, correct=correct, total=total)
                    continue

                for pred_text, gold in zip(decoded, gold_answers):
                    pred = extract_answer_math(pred_text)
                    if is_correct(pred, gold, self.tolerance):
                        correct += 1
                    total += 1

                acc_so_far = correct / total if total > 0 else 0.0
                pbar.set_postfix(accuracy=f"{acc_so_far:.4f}", correct=correct, total=total)

        accuracy = correct / total if total > 0 else 0.0
        print(f"\n  [{dataset_name}]  accuracy={accuracy:.4f}  "
              f"correct={correct}/{total}  failed={failed}")

        return {
            "accuracy": round(accuracy, 4),
            "correct":  correct,
            "total":    total,
            "failed":   failed,
            "dataset":  dataset_name,
        }


# ─── Multi-dataset evaluator ──────────────────────────────────────────────────

class MathMultiEvaluator:
    """
    Evaluates all 7 math test sets after a single training run.

    Args:
        model:          HuggingFace causal LM
        tokenizer:      HuggingFace tokenizer
        device:         'cuda' or 'cpu'
        batch_size:     Generation batch size
        max_new_tokens: Max tokens to generate
        max_length:     Max tokenisation length for test examples
        tolerance:      Numeric answer tolerance (default 0.1)
        local_paths:    Optional dict {dataset_name: local_json_path}
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 512,
        max_length: int = 1024,
        tolerance: float = 0.1,
        local_paths: Optional[Dict[str, str]] = None,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length     = max_length
        self.tolerance      = tolerance
        self.local_paths    = local_paths or {}

        self._evaluator = MathEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            tolerance=tolerance,
        )

    def _load_test(self, name: str) -> MathDataset:
        local_path = self.local_paths.get(name, None)
        data = load_math_test(name, local_path=local_path)
        return MathDataset(
            data, self.tokenizer,
            max_length=self.max_length,
            is_test=True,
        )

    def evaluate_all(
        self,
        datasets: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate on all (or a subset of) math test sets.

        Args:
            datasets: List of dataset names. Defaults to all 7 if None.

        Returns:
            Dict of {dataset_name: result_dict}
        """
        names   = datasets or list(TEST_URLS_MATH.keys())
        results = {}

        print(f"\n{'─'*60}")
        print(f"  Evaluating {len(names)} math test sets  (tolerance={self.tolerance})")
        print(f"{'─'*60}")

        for name in names:
            test_ds      = self._load_test(name)
            result       = self._evaluator.evaluate(test_ds, name)
            results[name] = result

        # Summary
        print(f"\n{'='*60}")
        print(f"  MATH EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Dataset':<12}  {'Accuracy':>10}  {'Correct':>8}  {'Total':>6}")
        print(f"  {'-'*44}")
        accs = []
        for name, r in results.items():
            print(f"  {name:<12}  {r['accuracy']:>10.4f}  "
                  f"{r['correct']:>8}  {r['total']:>6}")
            accs.append(r['accuracy'])
        avg = sum(accs) / len(accs) if accs else 0.0
        print(f"  {'-'*44}")
        print(f"  {'Average':<12}  {avg:>10.4f}")
        print(f"{'='*60}\n")

        return results

    def results_to_dict(self, results: Dict[str, Dict]) -> dict:
        """Convert results to plain dict for JSON serialisation."""
        out = {}
        for name, r in results.items():
            out[name] = {
                "accuracy": r["accuracy"],
                "correct":  r["correct"],
                "total":    r["total"],
                "failed":   r["failed"],
            }
        accs = [r["accuracy"] for r in results.values()]
        out["average_accuracy"] = round(sum(accs) / len(accs), 4) if accs else 0.0
        return out
    
    
"""
Qwen3-VL Evaluator Classes for all 5 VQA datasets.

Key differences from LLaVA evaluators:
    - No image_sizes field (Qwen3-VL uses image_grid_thw instead)
    - pixel_values + image_grid_thw must be tensors — ensured here before generate()
    - Same extract_answer logic as LLaVA evaluators

Classes:
    Qwen3VLM3cotEvaluator
    Qwen3VLSTVQAEvaluator
    Qwen3VLMMK12Evaluator
    Qwen3VLSafeTagEvaluator
    Qwen3VLVQARadEvaluator
"""

import re
import math
import torch
from tqdm import tqdm


def _to_tensor(val, device):
    """
    Ensure pixel_values / image_grid_thw are tensors on the correct device.
    Fixes the 'list has no attribute .type()' error during generate().
    """
    if isinstance(val, torch.Tensor):
        return val.to(device)
    if isinstance(val, list):
        return torch.cat([
            v if v.dim() > 1 else v.unsqueeze(0) for v in val
        ], dim=0).to(device)
    return val


# =============================================================================
#  M3COT
# =============================================================================

class Qwen3VLM3cotEvaluator:
    """Evaluator for M3CoT dataset using Qwen3-VL."""

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=512):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        match = re.search(r"Final Answer:\s*([A-F])", text)
        return match.group(1).strip() if match else None

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = _to_tensor(batch["pixel_values"], self.device)
        image_grid_thw = _to_tensor(batch["image_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[self.im_end_token_id,
                               self.processor.tokenizer.eos_token_id],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)
        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()
        correct = total = failed = 0
        dataset_size = len(test_dataset)

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt)
                total += 1

        accuracy = correct / max(total - failed, 1)
        results = {"split": split_name, "accuracy": accuracy,
                   "correct": correct, "total": total, "failed": failed}
        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")
        return results


# =============================================================================
#  STVQA
# =============================================================================

class Qwen3VLSTVQAEvaluator:
    """Evaluator for ST-VQA dataset using Qwen3-VL."""

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=512):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        match = re.search(r"<answer>\s*\(?([A-F])\)?", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"\b([A-F])\b(?!.*\b[A-F]\b)", text)
        if match:
            return match.group(1).strip()
        return None

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = _to_tensor(batch["pixel_values"], self.device)
        image_grid_thw = _to_tensor(batch["image_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[self.im_end_token_id,
                               self.processor.tokenizer.eos_token_id],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)
        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()
        correct = total = failed = 0
        dataset_size = len(test_dataset)

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted.upper() == gt.upper())
                total += 1

        accuracy = correct / max(total - failed, 1)
        results = {"split": split_name, "accuracy": accuracy,
                   "correct": correct, "total": total, "failed": failed}
        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")
        return results


# =============================================================================
#  MMK12
# =============================================================================

class Qwen3VLMMK12Evaluator:
    """Evaluator for MMK12 dataset using Qwen3-VL."""

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=32):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        match = re.search(r"<answer>\s*\(?([A-D])\)?", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\(([A-D])\)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()
        return None

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = _to_tensor(batch["pixel_values"], self.device)
        image_grid_thw = _to_tensor(batch["image_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[self.im_end_token_id,
                               self.processor.tokenizer.eos_token_id],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)
        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()
        correct = total = failed = 0
        dataset_size = len(test_dataset)

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt.upper())
                total += 1

        accuracy = correct / max(total - failed, 1)
        results = {"split": split_name, "accuracy": accuracy,
                   "correct": correct, "total": total, "failed": failed}
        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")
        return results


# =============================================================================
#  SAFETAG
# =============================================================================

class Qwen3VLSafeTagEvaluator:
    """Evaluator for SafeTag-VL binary safe/unsafe classification using Qwen3-VL."""

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=16):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def extract_answer(text: str) -> str:
        match = re.search(r"<answer>\s*(safe|unsafe)\s*</answer>", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        if re.search(r"\bunsafe\b", text, re.IGNORECASE):
            return "unsafe"
        if re.search(r"\bsafe\b", text, re.IGNORECASE):
            return "safe"
        return None

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = _to_tensor(batch["pixel_values"], self.device)
        image_grid_thw = _to_tensor(batch["image_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[self.im_end_token_id,
                               self.processor.tokenizer.eos_token_id],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)
        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()
        correct = total = failed = 0
        dataset_size = len(test_dataset)

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            decoded, ground_truths = self._run_batch(samples)

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                if predicted is None:
                    failed += 1
                else:
                    correct += int(predicted == gt.lower())
                total += 1

        accuracy = correct / max(total - failed, 1)
        results = {"split": split_name, "accuracy": round(accuracy, 4),
                   "correct": correct, "total": total, "failed": failed}
        print(f"\n=== [{split_name}] Accuracy: {correct}/{total} = {accuracy:.4f} ===")
        print(f"Failed extractions: {failed}/{total}")
        return results


# =============================================================================
#  VQARAD
# =============================================================================

class Qwen3VLVQARadEvaluator:
    """
    Evaluator for VQA-RAD dataset using Qwen3-VL.
    Reports overall + closed (yes/no) + open-ended accuracy.
    """

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=64):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @staticmethod
    def normalize(text: str) -> str:
        text = re.sub(r"</?\s*answer\s*>", "", text, flags=re.IGNORECASE)
        return text.strip().lower().strip('.,!?;:')

    @staticmethod
    def extract_answer(text: str) -> str:
        if not text or not text.strip():
            return None
        cleaned = Qwen3VLVQARadEvaluator.normalize(text)
        first_line = cleaned.split('\n')[0].strip()
        return first_line if first_line else None

    @staticmethod
    def is_closed_question(answer: str) -> bool:
        return answer.strip().lower() in ("yes", "no")

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer") for s in samples]
        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = _to_tensor(batch["pixel_values"], self.device)
        image_grid_thw = _to_tensor(batch["image_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=[self.im_end_token_id,
                               self.processor.tokenizer.eos_token_id],
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True)
        return decoded, ground_truths

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()
        total = failed = correct = 0
        closed_correct = closed_total = 0
        open_correct   = open_total   = 0
        dataset_size = len(test_dataset)
        first_batch_logged = False

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            decoded, ground_truths = self._run_batch(samples)

            if not first_batch_logged:
                print("\n=== DEBUG: First batch predictions ===")
                for pred_text, gt in zip(decoded[:3], ground_truths[:3]):
                    extracted = self.extract_answer(pred_text)
                    print(f"  GT       : {repr(gt)}")
                    print(f"  RAW PRED : {repr(pred_text[:120])}")
                    print(f"  EXTRACTED: {repr(extracted)}")
                    print()
                first_batch_logged = True

            for pred_text, gt in zip(decoded, ground_truths):
                predicted = self.extract_answer(pred_text)
                gt_norm   = self.normalize(gt)

                if predicted is None:
                    failed += 1
                    total  += 1
                    continue

                is_correct = int(predicted == gt_norm)
                correct   += is_correct
                total     += 1

                if self.is_closed_question(gt_norm):
                    closed_correct += is_correct
                    closed_total   += 1
                else:
                    open_correct += is_correct
                    open_total   += 1

        evaluated  = total - failed
        accuracy   = correct        / max(evaluated,    1)
        closed_acc = closed_correct / max(closed_total, 1)
        open_acc   = open_correct   / max(open_total,   1)

        results = {
            "split": split_name, "accuracy": round(accuracy, 4),
            "closed_acc": round(closed_acc, 4), "open_acc": round(open_acc, 4),
            "correct": correct, "total": total, "failed": failed,
            "closed_total": closed_total, "open_total": open_total,
        }
        print(f"\n=== [{split_name}] Results ===")
        print(f"  Overall accuracy : {accuracy:.4f}  ({correct}/{evaluated})")
        print(f"  Closed (yes/no)  : {closed_acc:.4f}  ({closed_correct}/{closed_total})")
        print(f"  Open-ended       : {open_acc:.4f}  ({open_correct}/{open_total})")
        print(f"  Failed           : {failed}/{total}")
        return results
    
    


# =============================================================================
#  EVALUATOR
# =============================================================================

# =============================================================================
#  EVALUATOR — matches official compute_metrics from util.py
# =============================================================================

def _to_tensor(val, device):
    if isinstance(val, torch.Tensor):
        return val.to(device)
    if isinstance(val, list):
        return torch.cat([
            v if v.dim() > 1 else v.unsqueeze(0) for v in val
        ], dim=0).to(device)
    return val


def _extract_pred(text: str) -> Optional[str]:
    """
    Extract predicted letter A or B from model output.
    Seeded with <answer> so output is like: A</answer> or B</answer>
    Returns "A", "B", or None.
    """
    text = text.strip()

    # Seeded output — first non-space char is usually the answer
    if text and text[0].upper() in ('A', 'B'):
        return text[0].upper()

    # Fallback: find A or B anywhere
    match = re.search(r'\b([AB])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def _compute_metrics(records):
    """
    Official compute_metrics from util.py.
    records: list of dicts with 'video_id', 'rating'
    Returns (single_accuracy * 100, pair_accuracy * 100)
    """
    single_correct = sum(r['rating'] for r in records)

    result_by_vid = {}
    for r in records:
        # base video_id = strip last _N suffix (e.g. _1, _2)
        base = "_".join(r['video_id'].split("_")[:-1])
        result_by_vid.setdefault(base, []).append(r)

    pair_correct = 0
    pair_total   = 0
    for base, recs in result_by_vid.items():
        if len(recs) == 2:
            pair_total   += 1
            pair_correct += int(all(r['rating'] == 1 for r in recs))

    n = max(len(records), 1)
    single_acc = single_correct / n * 100
    pair_acc   = pair_correct / max(pair_total, 1) * 100

    return single_acc, pair_acc, pair_total, pair_correct


class VideoPairsEvaluator:
    """
    Evaluator for minimal_video_pairs.
    Reports single_accuracy and pair_accuracy matching official util.py metrics.
    """

    def __init__(self, model, processor, collator, device="cuda",
                 batch_size=4, max_new_tokens=8):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _run_batch(self, samples):
        ground_truths = [s.pop("answer")   for s in samples]
        video_ids     = [s.pop("video_id") for s in samples]
        for s in samples:
            s.pop("answer_text", None)
            s.pop("candidates",  None)

        batch = self.collator(samples)

        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        gen_kwargs = dict(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = self.max_new_tokens,
            do_sample      = False,
            eos_token_id   = [self.im_end_id,
                               self.processor.tokenizer.eos_token_id],
            pad_token_id   = self.processor.tokenizer.eos_token_id,
        )

        if "pixel_values" in batch:
            gen_kwargs["pixel_values"]   = _to_tensor(batch["pixel_values"], self.device)
        if "image_grid_thw" in batch:
            gen_kwargs["image_grid_thw"] = _to_tensor(batch["image_grid_thw"], self.device)
        if "pixel_values_videos" in batch:
            gen_kwargs["pixel_values_videos"] = _to_tensor(
                batch["pixel_values_videos"], self.device)
        if "video_grid_thw" in batch:
            gen_kwargs["video_grid_thw"] = _to_tensor(batch["video_grid_thw"], self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**gen_kwargs)

        trimmed = generated_ids[:, input_ids.shape[1]:]
        decoded = self.processor.tokenizer.batch_decode(
            trimmed, skip_special_tokens=True)

        # Decode input text for debug (skip_special_tokens=False to see question)
        input_texts = self.processor.tokenizer.batch_decode(
            input_ids.cpu(), skip_special_tokens=True)

        return decoded, ground_truths, video_ids, input_texts

    def evaluate(self, test_dataset, split_name="Test") -> dict:
        self.model.eval()

        all_records  = []
        failed       = 0
        total        = 0
        dataset_size = len(test_dataset)
        first_logged = False

        for start in tqdm(range(0, dataset_size, self.batch_size),
                          total=math.ceil(dataset_size / self.batch_size),
                          desc=f"Evaluating {split_name}"):
            samples = [test_dataset[i]
                       for i in range(start, min(start + self.batch_size, dataset_size))]
            try:
                decoded, ground_truths, video_ids, input_texts = self._run_batch(samples)
            except Exception as e:
                print(f"  ⚠ Batch failed: {e}")
                failed += len(samples)
                total  += len(samples)
                continue

            if not first_logged:
                print("\n=== DEBUG: First predictions ===")
                for pred, gt, vid, inp in zip(
                        decoded[:3], ground_truths[:3],
                        video_ids[:3], input_texts[:3]):
                    pred_letter = _extract_pred(pred)
                    match = "OK" if pred_letter == gt else "WRONG"
                    # Show last 300 chars of input (question + options)
                    print(f"  video_id : {vid}")
                    print(f"  INPUT    : ...{inp[-300:].strip()}")
                    print(f"  GT       : {gt}")
                    print(f"  RAW PRED : {repr(pred[:80])}")
                    print(f"  EXTRACTED: {pred_letter} [{match}]")
                    print()
                first_logged = True

            for pred_text, gt, video_id in zip(decoded, ground_truths, video_ids):
                pred_letter = _extract_pred(pred_text)
                rating      = 0

                if pred_letter is None:
                    failed += 1
                else:
                    rating = 1 if pred_letter == gt else 0

                total += 1
                all_records.append({
                    "video_id":   video_id,
                    "pred_idx":   pred_letter,
                    "answer_idx": gt,
                    "rating":     rating,
                })

        single_acc, pair_acc, pair_total, pair_correct = _compute_metrics(all_records)
        correct = sum(r['rating'] for r in all_records)

        results = {
            "split":            split_name,
            "single_accuracy":  round(single_acc, 2),
            "pair_accuracy":    round(pair_acc,   2),
            "accuracy":         round(single_acc / 100, 4),   # compat with sweep CSV
            "correct":          correct,
            "total":            total,
            "failed":           failed,
            "pair_total":       pair_total,
            "pair_correct":     pair_correct,
        }

        print(f"\n=== [{split_name}] Results ===")
        print(f"  Single accuracy : {single_acc:.2f}%  ({correct}/{total})")
        print(f"  Pair   accuracy : {pair_acc:.2f}%  ({pair_correct}/{pair_total})")
        print(f"  Failed          : {failed}/{total}")

        return results
    
    
    

class VideoMathQAEvaluator:
    """
    Evaluator for MBZUAI/VideoMathQA (multi_binary split).

    Metrics:
        - overall accuracy      : % of questions answered correctly (A or B)
        - per-category accuracy : breakdown by 'category' field
        - failed                : samples where no valid A/B could be extracted

    Usage:
        evaluator = VideoMathQAEvaluator(
            model=model, processor=processor, collator=collator,
            device="cuda", batch_size=4, max_new_tokens=512,
        )
        results = evaluator.evaluate(test_dataset, split_name="baseline_full")
    """

    def __init__(self, model, processor, collator,
                 device="cuda", batch_size=4, max_new_tokens=512):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    def _extract_answer(self, text):
        """
        Extract A or B from generated text. Tries in order:
          1. 'Answer: A' / 'Answer: B'
          2. Standalone A or B at end of string
          3. First A or B anywhere
        """
        text = text.strip()
        m = _re.search(r'Answer\s*:\s*([AB])', text, _re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = _re.search(r'\b([AB])\b\s*$', text, _re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = _re.search(r'\b([AB])\b', text, _re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    def evaluate(self, dataset, split_name="eval"):
        self.model.eval()

        all_preds   = []
        correct     = 0
        failed      = 0
        total       = 0
        cat_correct = defaultdict(int)
        cat_total   = defaultdict(int)

        meta_keys = ["answer", "question_id", "mbinary_id", "category"]

        print(f"\n{'='*60}")
        print(f"  Evaluating: {split_name}  ({len(dataset)} samples)")
        print(f"  Batch size: {self.batch_size}  Max new tokens: {self.max_new_tokens}")
        print(f"{'='*60}")

        for batch_start in range(0, len(dataset), self.batch_size):
            batch_end     = min(batch_start + self.batch_size, len(dataset))
            batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

            # Pop metadata before collating
            answers      = [s.pop("answer")      for s in batch_samples]
            question_ids = [s.pop("question_id") for s in batch_samples]
            mbinary_ids  = [s.pop("mbinary_id")  for s in batch_samples]
            categories   = [s.pop("category")    for s in batch_samples]

            batch          = self.collator(batch_samples)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            generate_kwargs = dict(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                temperature    = None,
                top_p          = None,
                pad_token_id   = self.processor.tokenizer.pad_token_id,  
                eos_token_id   = self.processor.tokenizer.eos_token_id,  
            )
            for key in ["pixel_values", "pixel_values_videos",
                        "image_grid_thw", "video_grid_thw"]:
                if key in batch:
                    generate_kwargs[key] = batch[key].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**generate_kwargs)

            input_len = input_ids.shape[1]
            new_ids   = output_ids[:, input_len:]
            decoded   = self.processor.tokenizer.batch_decode(
                new_ids, skip_special_tokens=True)

            for i, (gen_text, gt_answer, category) in enumerate(
                    zip(decoded, answers, categories)):

                pred       = self._extract_answer(gen_text)
                is_correct = (pred == gt_answer.upper()) if pred else False

                if pred is None:
                    failed += 1
                if is_correct:
                    correct += 1

                cat_total[category]   += 1
                cat_correct[category] += int(is_correct)
                total                 += 1

                all_preds.append({
                    "mbinary_id":  mbinary_ids[i],
                    "question_id": question_ids[i],
                    "category":    category,
                    "gt_answer":   gt_answer,
                    "pred_answer": pred,
                    "correct":     is_correct,
                    "generated":   gen_text,
                })

            if (batch_start // self.batch_size + 1) % 10 == 0:
                print(f"  [{batch_end}/{len(dataset)}]  "
                      f"acc={correct/total:.3f}  failed={failed}")

        category_accuracy = {
            cat: round(cat_correct[cat] / cat_total[cat], 4)
            for cat in sorted(cat_total.keys())
        }
        overall_accuracy = round(correct / total, 4) if total > 0 else 0.0

        print(f"\n  {'─'*50}")
        print(f"  Overall accuracy : {overall_accuracy:.4f}  "
              f"({correct}/{total})  failed={failed}")
        print(f"\n  Per-category accuracy:")
        for cat, acc in category_accuracy.items():
            print(f"    {cat:<35} {acc:.4f}  ({cat_correct[cat]}/{cat_total[cat]})")
        print(f"  {'─'*50}\n")

        return {
            "accuracy":          overall_accuracy,
            "correct":           correct,
            "total":             total,
            "failed":            failed,
            "category_accuracy": category_accuracy,
            "category_correct":  dict(cat_correct),
            "category_total":    dict(cat_total),
            "predictions":       all_preds,
        }
        
        
        
        
        



class VideoLevelGaugeEvaluator:
    """
    Evaluator for Cola-any/Video-LevelGauge.

    Metrics:
        - overall accuracy          : % of questions answered correctly (A-D)
        - per question_type accuracy : e.g. Spatial Relationship, Action, etc.
        - per video_category accuracy: e.g. Media-Tutorial, etc.
        - failed                    : samples where no valid A/B/C/D extracted

    The model is expected to output reasoning followed by '<answer>X'.
    Answer extraction tries multiple fallback patterns robustly.

    Usage:
        evaluator = VideoLevelGaugeEvaluator(
            model          = model,
            processor      = processor,
            collator       = collator,
            device         = "cuda",
            batch_size     = 4,
            max_new_tokens = 64,
        )
        results = evaluator.evaluate(test_dataset, split_name="baseline_lora")
    """

    def __init__(
        self,
        model,
        processor,
        collator,
        device         = "cuda",
        batch_size     = 4,
        max_new_tokens = 64,
    ):
        self.model          = model
        self.processor      = processor
        self.collator       = collator
        self.device         = device
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens

    # ── Answer extraction ──────────────────────────────────────────────────────

    def _extract_answer(self, text):
        """
        Extract A/B/C/D from generated text. Tries in order:
          1. '<answer>X'  — primary expected format
          2. 'Answer: X'  — alternative phrasing
          3. Standalone X at end of string
          4. First occurrence of A/B/C/D anywhere
        Returns None if nothing found.
        """
        text = text.strip()

        # 1. <answer>X  (primary)
        m = re.search(r'<answer>\s*([ABCD])', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 2. Answer: X
        m = re.search(r'Answer\s*:\s*([ABCD])', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 3. Standalone A/B/C/D at end of string
        m = re.search(r'\b([ABCD])\b\s*$', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 4. First occurrence of A/B/C/D anywhere
        m = re.search(r'\b([ABCD])\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return None

    # ── Evaluation loop ────────────────────────────────────────────────────────

    def evaluate(self, dataset, split_name="eval"):
        """
        Run evaluation over the full dataset.

        Args:
            dataset    : VideoLevelGaugeDataset with is_test=True
            split_name : Label for logging

        Returns:
            dict with keys:
                accuracy, correct, total, failed,
                question_type_accuracy  (dict),
                video_category_accuracy (dict),
                predictions             (list of per-sample dicts)
        """
        self.model.eval()

        all_preds    = []
        correct      = 0
        failed       = 0
        total        = 0

        qtype_correct  = defaultdict(int)
        qtype_total    = defaultdict(int)
        vcat_correct   = defaultdict(int)
        vcat_total     = defaultdict(int)

        meta_keys = ["answer", "gt_answer", "question_id",
                     "question_type", "video_category"]

        print(f"\n{'='*60}")
        print(f"  Evaluating: {split_name}  ({len(dataset)} samples)")
        print(f"  Batch size: {self.batch_size}  "
              f"Max new tokens: {self.max_new_tokens}")
        print(f"{'='*60}")

        for batch_start in range(0, len(dataset), self.batch_size):
            batch_end     = min(batch_start + self.batch_size, len(dataset))
            batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

            # Pop metadata before collating
            answers        = [s.pop("answer")         for s in batch_samples]
            gt_answers     = [s.pop("gt_answer")      for s in batch_samples]
            question_ids   = [s.pop("question_id")    for s in batch_samples]
            question_types = [s.pop("question_type")  for s in batch_samples]
            video_cats     = [s.pop("video_category") for s in batch_samples]

            batch          = self.collator(batch_samples)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            generate_kwargs = dict(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = self.max_new_tokens,
                do_sample      = False,
                temperature    = None,
                top_p          = None,
                pad_token_id   = self.processor.tokenizer.pad_token_id,
                eos_token_id   = self.processor.tokenizer.eos_token_id,
            )
            for key in ["pixel_values", "pixel_values_videos",
                        "image_grid_thw", "video_grid_thw"]:
                if key in batch:
                    generate_kwargs[key] = batch[key].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**generate_kwargs)

            input_len = input_ids.shape[1]
            new_ids   = output_ids[:, input_len:]
            decoded   = self.processor.tokenizer.batch_decode(
                new_ids, skip_special_tokens=True)

            for i, (gen_text, gt_letter, qtype, vcat) in enumerate(
                    zip(decoded, answers, question_types, video_cats)):

                pred       = self._extract_answer(gen_text)
                is_correct = (pred == gt_letter.upper()) if pred else False

                if pred is None:
                    failed += 1
                if is_correct:
                    correct += 1

                qtype_total[qtype]   += 1
                qtype_correct[qtype] += int(is_correct)
                vcat_total[vcat]     += 1
                vcat_correct[vcat]   += int(is_correct)
                total                += 1

                all_preds.append({
                    "question_id":    question_ids[i],
                    "question_type":  qtype,
                    "video_category": vcat,
                    "gt_answer":      gt_answers[i],
                    "gt_letter":      gt_letter,
                    "pred_letter":    pred,
                    "correct":        is_correct,
                    "generated":      gen_text,
                })

            if (batch_start // self.batch_size + 1) % 10 == 0:
                print(f"  [{batch_end}/{len(dataset)}]  "
                      f"acc={correct/total:.3f}  failed={failed}")

        # ── Per-type and per-category accuracy ────────────────────────────────
        qtype_accuracy = {
            qt: round(qtype_correct[qt] / qtype_total[qt], 4)
            for qt in sorted(qtype_total.keys())
        }
        vcat_accuracy = {
            vc: round(vcat_correct[vc] / vcat_total[vc], 4)
            for vc in sorted(vcat_total.keys())
        }
        overall_accuracy = round(correct / total, 4) if total > 0 else 0.0

        # ── Print summary ──────────────────────────────────────────────────────
        print(f"\n  {'─'*50}")
        print(f"  Overall accuracy : {overall_accuracy:.4f}  "
              f"({correct}/{total})  failed={failed}")

        print(f"\n  Per question_type accuracy:")
        for qt, acc in qtype_accuracy.items():
            c = qtype_correct[qt]
            t = qtype_total[qt]
            print(f"    {qt:<35} {acc:.4f}  ({c}/{t})")

        print(f"\n  Per video_category accuracy:")
        for vc, acc in vcat_accuracy.items():
            c = vcat_correct[vc]
            t = vcat_total[vc]
            print(f"    {vc:<35} {acc:.4f}  ({c}/{t})")

        print(f"  {'─'*50}\n")

        return {
            "accuracy":               overall_accuracy,
            "correct":                correct,
            "total":                  total,
            "failed":                 failed,
            "question_type_accuracy": qtype_accuracy,
            "video_category_accuracy": vcat_accuracy,
            "qtype_correct":          dict(qtype_correct),
            "qtype_total":            dict(qtype_total),
            "vcat_correct":           dict(vcat_correct),
            "vcat_total":             dict(vcat_total),
            "predictions":            all_preds,
        }
        
        
"""
Image Classification Evaluator for ViT (google/vit-base-patch16-224)

Metrics:
    - overall accuracy    : top-1 accuracy
    - per-class accuracy  : breakdown by class index
    - failed              : samples that raised errors during inference
"""

import torch
from collections import defaultdict


class ImageCLSEvaluator:
    """
    Evaluator for image classification with ViT.

    Works with all 6 image classification datasets:
        ImagenetteDataset, BeansDataset, OxfordPetsDataset,
        ChestXRayDataset, RVLCDIPDataset, AIWD6Dataset

    Usage:
        evaluator = ImageCLSEvaluator(
            model      = model,
            device     = "cuda",
            batch_size = 32,
        )
        results = evaluator.evaluate(test_dataset, split_name="baseline_full")
    """

    def __init__(
        self,
        model,
        device     = "cuda",
        batch_size = 32,
    ):
        self.model      = model
        self.device     = device
        self.batch_size = batch_size

    def evaluate(self, dataset, split_name="eval", id2label=None):
        """
        Run evaluation over the full dataset.

        Args:
            dataset    : Any ImageCLS dataset (returns pixel_values + labels)
            split_name : Label for logging
            id2label   : Optional dict {int: str} for class name display

        Returns:
            dict with keys:
                accuracy         : float  overall top-1
                correct          : int
                total            : int
                failed           : int
                per_class_acc    : dict {class_id: accuracy}
                per_class_correct: dict {class_id: correct count}
                per_class_total  : dict {class_id: total count}
        """
        self.model.eval()

        correct        = 0
        failed         = 0
        total          = 0
        class_correct  = defaultdict(int)
        class_total    = defaultdict(int)

        print(f"\n{'='*60}")
        print(f"  Evaluating: {split_name}  ({len(dataset)} samples)")
        print(f"  Batch size: {self.batch_size}")
        print(f"{'='*60}")

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = True,
        )

        for batch_idx, batch in enumerate(loader):
            try:
                pixel_values = batch["pixel_values"].to(self.device)
                labels       = batch["labels"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    logits  = outputs.logits                     # [B, num_classes]
                    preds   = logits.argmax(dim=-1)              # [B]

                for pred, label in zip(preds, labels):
                    pred  = pred.item()
                    label = label.item()
                    class_total[label]   += 1
                    class_correct[label] += int(pred == label)
                    if pred == label:
                        correct += 1
                    total += 1

            except Exception as e:
                failed += len(batch["labels"])
                print(f"  ⚠ Batch {batch_idx} failed: {e}")
                continue

            if (batch_idx + 1) % 20 == 0:
                print(f"  [{total}/{len(dataset)}]  "
                      f"acc={correct/total:.3f}  failed={failed}")

        # ── Per-class accuracy ─────────────────────────────────────────────────
        per_class_acc = {}
        for cls_id in sorted(class_total.keys()):
            acc = class_correct[cls_id] / class_total[cls_id] \
                  if class_total[cls_id] > 0 else 0.0
            per_class_acc[cls_id] = round(acc, 4)

        overall_accuracy = round(correct / total, 4) if total > 0 else 0.0

        # ── Print summary ──────────────────────────────────────────────────────
        print(f"\n  {'─'*50}")
        print(f"  Overall accuracy : {overall_accuracy:.4f}  "
              f"({correct}/{total})  failed={failed}")
        print(f"\n  Per-class accuracy:")
        for cls_id, acc in per_class_acc.items():
            c    = class_correct[cls_id]
            t    = class_total[cls_id]
            name = id2label[cls_id] if id2label and cls_id in id2label \
                   else f"class_{cls_id}"
            print(f"    {name:<30} {acc:.4f}  ({c}/{t})")
        print(f"  {'─'*50}\n")

        return {
            "accuracy":          overall_accuracy,
            "correct":           correct,
            "total":             total,
            "failed":            failed,
            "per_class_acc":     per_class_acc,
            "per_class_correct": dict(class_correct),
            "per_class_total":   dict(class_total),
        }