"""
LLaVA Collator for pre-processed datasets.

Use when your Dataset.__getitem__ already calls processor() and returns tensors.
This collator ONLY:
    1. Pads / stacks tensors across samples in a batch
    2. Builds labels (input_ids with -100 on padding + image + prompt tokens)

Do NOT use LlavaOnevisionProcessor here — samples are already tokenized.
"""

import torch
from typing import Any, Dict, List, Optional


class LLaVACollator:

    def __init__(
        self,
        processor,
        label_mask_mode: str = 'response_only',
        pad_token_id:    Optional[int] = None,
    ):
        """
        Args:
            processor:        LLaVA processor — used only to read tokenizer metadata
                              (pad_token_id, image_token_id). NOT called on samples.
            label_mask_mode:  'response_only' — train on assistant reply only  (recommended)
                              'text_only'     — train on all text, mask image tokens
                              'all'           — mask only padding
            pad_token_id:     Override pad token id (default: from processor.tokenizer)
        """
        if label_mask_mode not in ('response_only', 'text_only', 'all'):
            raise ValueError("label_mask_mode must be 'response_only', 'text_only', or 'all'")

        self.processor       = processor
        self.label_mask_mode = label_mask_mode

        tok = processor.tokenizer
        self.pad_token_id = pad_token_id or tok.pad_token_id or tok.eos_token_id

        # Image placeholder token id
        self._image_token_id = tok.convert_tokens_to_ids("<image>")
        if self._image_token_id == tok.unk_token_id:
            self._image_token_id = None

    # ------------------------------------------------------------------

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        """
        Find the first token index AFTER the last 'assistant' marker.
        Everything before → -100 in labels.
        """
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        for i in range(len(tokens) - 1, -1, -1):
            if 'assistant' in (tokens[i] or '').lower():
                return i + 1
        return 0  # fallback: train on full sequence

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()

        # Mask padding
        labels[attention_mask == 0] = -100

        # Mask image placeholder tokens
        if self._image_token_id is not None:
            labels[input_ids == self._image_token_id] = -100

        if self.label_mask_mode == 'response_only':
            resp_start = self._find_response_start(input_ids)
            if resp_start > 0:
                labels[:resp_start] = -100

        return labels

    # ------------------------------------------------------------------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── Separate tensor keys from non-tensor metadata ─────────────────────
        meta_keys   = {k for f in features for k in f if not isinstance(f[k], torch.Tensor)}
        tensor_keys = {k for f in features for k in f if isinstance(f[k], torch.Tensor)}
        tensor_keys.discard('labels')  # we rebuild labels below

        batch = {}

        # ── Stack / pad 1D and 2D tensors  ────────────────────────────────────
        for key in tensor_keys:
            vals = [f[key] for f in features if key in f]

            if not vals:
                continue

            # 1D tensors (input_ids, attention_mask) — pad to same length
            if vals[0].dim() == 1:
                max_len = max(v.shape[0] for v in vals)
                pad_val = self.pad_token_id if key == 'input_ids' else 0
                padded  = []
                for v in vals:
                    pad_size = max_len - v.shape[0]
                    if pad_size > 0:
                        # left pad (LLaVA uses left padding)
                        v = torch.cat([torch.full((pad_size,), pad_val, dtype=v.dtype), v])
                    padded.append(v)
                batch[key] = torch.stack(padded)

            # Higher-dim tensors (pixel_values, image_sizes) — stack if same shape
            else:
                try:
                    batch[key] = torch.stack(vals)
                except RuntimeError:
                    # Variable shapes (e.g. different numbers of image patches)
                    # — keep as list, model handles it internally
                    batch[key] = vals

        # ── Build labels ───────────────────────────────────────────────────────
        if 'input_ids' in batch and 'attention_mask' in batch:
            B      = batch['input_ids'].shape[0]
            labels = torch.full_like(batch['input_ids'], -100)
            for i in range(B):
                labels[i] = self._build_labels(
                    batch['input_ids'][i],
                    batch['attention_mask'][i],
                )
            batch['labels'] = labels

        # ── Ensure integer tensors are int64 ───────────────────────────────────
        for key in ('input_ids', 'attention_mask', 'token_type_ids'):
            if key in batch and isinstance(batch[key], torch.Tensor):
                if batch[key].is_floating_point():
                    batch[key] = batch[key].long()

        return batch

    def __repr__(self):
        return (
            f"LLaVACollator("
            f"label_mask_mode='{self.label_mask_mode}')"
        )
        
        
        
        
import torch
from typing import Any, Dict, List, Optional


class InstructionMaskingCollator:

    def __init__(
        self,
        tokenizer,
        response_template: str = "### Response:",
        pad_token_id: Optional[int] = None,
    ):
        self.tokenizer         = tokenizer
        self.response_template = response_template
        self.pad_token_id      = pad_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id

        # Pre-tokenize the response template for fast matching
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    # ------------------------------------------------------------------

    def _find_response_start(self, input_ids: List[int]) -> int:
        """Return the index of the first token AFTER '### Response:'"""
        n = len(self.response_token_ids)
        for i in range(len(input_ids) - n, -1, -1):
            if input_ids[i : i + n] == self.response_token_ids:
                return i + n  # start training AFTER the template
        return 0  # fallback: train on everything

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask instruction tokens (everything before ### Response:)
        resp_start = self._find_response_start(input_ids.tolist())
        if resp_start > 0:
            labels[:resp_start] = -100

        return labels

    # ------------------------------------------------------------------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list      = [f["input_ids"] for f in features]
        attention_mask_list = [f.get("attention_mask") for f in features]

        # Pad to longest sequence in batch (left padding)
        max_len = max(ids.shape[0] for ids in input_ids_list)

        padded_input_ids      = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_size = max_len - ids.shape[0]
            if pad_size > 0:
                ids  = torch.cat([torch.full((pad_size,), self.pad_token_id, dtype=ids.dtype), ids])
                mask = torch.cat([torch.zeros(pad_size, dtype=mask.dtype), mask]) if mask is not None else None
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask if mask is not None else torch.ones_like(ids))

        batch = {
            "input_ids":      torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
        }

        # Build labels with instruction masking
        B      = batch["input_ids"].shape[0]
        labels = torch.full_like(batch["input_ids"], -100)
        for i in range(B):
            labels[i] = self._build_labels(batch["input_ids"][i], batch["attention_mask"][i])
        batch["labels"] = labels

        return batch
    
    
    
"""
Qwen3-VL Collator for pre-processed datasets.

Key differences from LLaVACollator:
    - Image token  : <|vision_start|>...<|vision_end|>  (not <image>)
    - Assistant marker : <|im_start|>assistant  (Qwen chat template)
    - pixel_values : torch.cat along dim=0 (flat patches, not stacked)
    - image_grid_thw : torch.cat along dim=0 (spatial grid info)
    - Non-tensor keys (answer, answers, question_type) : stripped from batch
"""

import torch
from typing import Any, Dict, List, Optional


# Non-tensor metadata keys — strip from batch before passing to model
_META_KEYS = {'answer', 'answers', 'answer_full', 'question_type', 'split'}


class Qwen3VLCollator:

    def __init__(
        self,
        processor,
        label_mask_mode: str = 'response_only',
        pad_token_id:    Optional[int] = None,
    ):
        if label_mask_mode not in ('response_only', 'all'):
            raise ValueError("label_mask_mode must be 'response_only' or 'all'")

        self.processor       = processor
        self.label_mask_mode = label_mask_mode

        tok = processor.tokenizer
        self.pad_token_id = pad_token_id or tok.pad_token_id or tok.eos_token_id

        self._vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id   = tok.convert_tokens_to_ids("<|vision_end|>")
        if self._vision_start_id == tok.unk_token_id:
            self._vision_start_id = None
        if self._vision_end_id == tok.unk_token_id:
            self._vision_end_id = None

    # ------------------------------------------------------------------

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i] or ''
            if 'assistant' in tok.lower():
                return i + 1
        return 0

    def _mask_vision_tokens(self, labels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self._vision_start_id is None or self._vision_end_id is None:
            return labels
        in_vision = False
        for i in range(len(input_ids)):
            token_id = input_ids[i].item()
            if token_id == self._vision_start_id:
                in_vision = True
            if in_vision:
                labels[i] = -100
            if token_id == self._vision_end_id:
                in_vision = False
        return labels

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels = self._mask_vision_tokens(labels, input_ids)
        if self.label_mask_mode == 'response_only':
            resp_start = self._find_response_start(input_ids)
            if resp_start > 0:
                labels[:resp_start] = -100
        return labels

    # ------------------------------------------------------------------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Collect all keys, drop metadata and labels (labels rebuilt below)
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        all_keys -= _META_KEYS
        all_keys.discard('labels')

        batch = {}

        for key in all_keys:
            vals = [f[key] for f in features if key in f]
            if not vals:
                continue

            # 1D tensors — left-pad to max length
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() == 1:
                max_len = max(v.shape[0] for v in vals)
                pad_val = self.pad_token_id if key == 'input_ids' else 0
                padded  = []
                for v in vals:
                    pad_size = max_len - v.shape[0]
                    if pad_size > 0:
                        v = torch.cat([
                            torch.full((pad_size,), pad_val, dtype=v.dtype), v])
                    padded.append(v)
                batch[key] = torch.stack(padded)

            # pixel_values — cat along patch dim (each image → variable patches)
            elif key == 'pixel_values':
                try:
                    batch[key] = torch.cat([
                        v if v.dim() > 1 else v.unsqueeze(0) for v in vals
                    ], dim=0)
                except RuntimeError:
                    batch[key] = torch.stack(vals)

            # image_grid_thw — cat along batch dim
            elif key == 'image_grid_thw':
                try:
                    batch[key] = torch.cat([
                        v if v.dim() > 1 else v.unsqueeze(0) for v in vals
                    ], dim=0)
                except RuntimeError:
                    batch[key] = vals

            # Other tensors — stack if possible
            elif isinstance(vals[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(vals)
                except RuntimeError:
                    batch[key] = vals

            # Non-tensor — keep as list (shouldn't reach here after _META_KEYS strip)
            else:
                batch[key] = vals

        # Build labels
        if 'input_ids' in batch and 'attention_mask' in batch:
            B      = batch['input_ids'].shape[0]
            labels = torch.full_like(batch['input_ids'], -100)
            for i in range(B):
                labels[i] = self._build_labels(
                    batch['input_ids'][i],
                    batch['attention_mask'][i],
                )
            batch['labels'] = labels

        # Ensure integer dtypes
        for key in ('input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'):
            if key in batch and isinstance(batch[key], torch.Tensor):
                if batch[key].is_floating_point():
                    batch[key] = batch[key].long()

        return batch

    def __repr__(self):
        return f"Qwen3VLCollator(label_mask_mode='{self.label_mask_mode}')"
    


import os
import ast
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from qwen_vl_utils import process_vision_info


# =============================================================================
#  UPDATED COLLATOR — handles both image and video tensor keys
# =============================================================================

# All non-tensor metadata keys — stripped before model sees the batch
_META_KEYS = {
    'answer', 'answers', 'answer_full', 'answer_text',
    'question_type', 'candidates', 'video_id', 'split',
}


class Qwen3VLCollator:
    """
    Collator for Qwen3-VL — handles both image and video inputs.

    Image keys : pixel_values      [N_patches, C*H*W]
                 image_grid_thw    [N_images, 3]
    Video keys : pixel_values_videos [N_frames, C*H*W]
                 video_grid_thw      [N_videos, 3]
    Both use torch.cat along dim=0.
    """

    def __init__(self, processor, label_mask_mode='response_only',
                 pad_token_id=None):
        if label_mask_mode not in ('response_only', 'all'):
            raise ValueError("label_mask_mode must be 'response_only' or 'all'")

        self.processor       = processor
        self.label_mask_mode = label_mask_mode

        tok = processor.tokenizer
        self.pad_token_id = pad_token_id or tok.pad_token_id or tok.eos_token_id

        self._vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id   = tok.convert_tokens_to_ids("<|vision_end|>")
        if self._vision_start_id == tok.unk_token_id:
            self._vision_start_id = None
        if self._vision_end_id == tok.unk_token_id:
            self._vision_end_id = None

    def _find_response_start(self, input_ids: torch.Tensor) -> int:
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i] or ''
            if 'assistant' in tok.lower():
                return i + 1
        return 0

    def _mask_vision_tokens(self, labels: torch.Tensor,
                             input_ids: torch.Tensor) -> torch.Tensor:
        if self._vision_start_id is None or self._vision_end_id is None:
            return labels
        in_vision = False
        for i in range(len(input_ids)):
            token_id = input_ids[i].item()
            if token_id == self._vision_start_id:
                in_vision = True
            if in_vision:
                labels[i] = -100
            if token_id == self._vision_end_id:
                in_vision = False
        return labels

    def _build_labels(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels = self._mask_vision_tokens(labels, input_ids)
        if self.label_mask_mode == 'response_only':
            resp_start = self._find_response_start(input_ids)
            if resp_start > 0:
                labels[:resp_start] = -100
        return labels

    def _cat_tensor_list(self, vals):
        """cat variable-length tensors along dim=0 (for pixel patch tensors)."""
        try:
            return torch.cat([
                v if v.dim() > 1 else v.unsqueeze(0) for v in vals
            ], dim=0)
        except RuntimeError:
            return torch.stack(vals)

    def __call__(self, features):
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        all_keys -= _META_KEYS
        all_keys.discard('labels')

        batch = {}

        for key in all_keys:
            vals = [f[key] for f in features if key in f]
            if not vals:
                continue

            # 1D tensors — left-pad
            if isinstance(vals[0], torch.Tensor) and vals[0].dim() == 1:
                max_len = max(v.shape[0] for v in vals)
                pad_val = self.pad_token_id if key == 'input_ids' else 0
                padded  = []
                for v in vals:
                    pad_size = max_len - v.shape[0]
                    if pad_size > 0:
                        v = torch.cat([
                            torch.full((pad_size,), pad_val, dtype=v.dtype), v])
                    padded.append(v)
                batch[key] = torch.stack(padded)

            # Image patch tensors — cat along patch dim
            elif key in ('pixel_values', 'pixel_values_videos'):
                batch[key] = self._cat_tensor_list(vals)

            # Grid info tensors — cat along batch dim
            elif key in ('image_grid_thw', 'video_grid_thw'):
                batch[key] = self._cat_tensor_list(vals)

            # Other tensors
            elif isinstance(vals[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(vals)
                except RuntimeError:
                    batch[key] = vals

            else:
                batch[key] = vals

        # Build labels
        if 'input_ids' in batch and 'attention_mask' in batch:
            B      = batch['input_ids'].shape[0]
            labels = torch.full_like(batch['input_ids'], -100)
            for i in range(B):
                labels[i] = self._build_labels(
                    batch['input_ids'][i], batch['attention_mask'][i])
            batch['labels'] = labels

        # Ensure integer dtypes
        for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
            if key in batch and isinstance(batch[key], torch.Tensor):
                if batch[key].is_floating_point():
                    batch[key] = batch[key].long()

        return batch

    def __repr__(self):
        return f"Qwen3VLCollator(label_mask_mode='{self.label_mask_mode}')"



