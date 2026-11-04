# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 The Hugging Face Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# ============================================================================
# llama_token.py — TokenTune dual-path Llama, last-N selection.
# Compatible with transformers >= 4.46 (modernized RoPE API).
#
# Multi-GPU notes:
#   - _no_split_modules tells HF to keep each LlamaPrefixDecoderLayer atomic
#     on one GPU, instead of splitting attn / mlp across devices.
#   - The decoder layer's forward defensively moves all prefix tensors to
#     the layer's own device (accelerate's hooks don't always handle the
#     extra hidden_states_prefix / attention_mask_prefix kwargs).
# ============================================================================

import math
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging, ModelOutput, PaddingStrategy


logger = logging.get_logger(__name__)


# ============================================================================
#  SAMPLE IDENTIFIER (for attention-score lookup)
# ============================================================================

def generate_identifier(input_ids) -> str:
    """SHA256 hash of input_ids, used as a key into precomputed attention maps.

    Accepts torch.Tensor (any shape — flattened) or list. Returns first 16
    hex chars of the hash, matching the TokenSeek format.
    """
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() > 1:
            input_ids = input_ids.flatten()
        input_list = input_ids.tolist()
    else:
        input_list = list(input_ids)
    input_str = ",".join(map(str, input_list))
    return hashlib.sha256(input_str.encode()).hexdigest()[:16]


# ============================================================================
#  DATA COLLATOR — last-N / random-K / attention-based selection
# ============================================================================

@dataclass
class DataCollatorWithPaddingForPrefix:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    prefix_length: Union[int, float] = 16
    # ── Selection mode ──────────────────────────────────────────────────────
    # "last"      = last-N tokens are trainable (causal.py style, recommended
    #               for instruction-tuning datasets where labels live at the end)
    # "random"    = K random positions + position 0 + last position
    #               (original llama_token.py / TokenTune paper behavior)
    # "attention" = top-K by pre-computed attention scores + pos 0 + last
    #               (TokenSeek-style content-aware selection — requires
    #               attention_map_path to point at a torch.load-able file)
    selection: str = "last"
    # Required when selection == "attention", ignored otherwise.
    # Expected file format (torch.save'd dict):
    #   { hash_str -> { 'len': int,
    #                   'scores': 1D tensor of importance scores,
    #                   '<frac>': pre-baked indices for fractional prefix_length } }
    # where hash_str = generate_identifier(input_ids).
    attention_map_path: Optional[str] = None

    def __post_init__(self):
        self.attention_map = None
        if self.selection == "attention":
            if self.attention_map_path is None:
                raise ValueError(
                    "selection='attention' requires attention_map_path "
                    "to point at a torch.load-able file with precomputed scores."
                )
            self.attention_map = torch.load(self.attention_map_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_batch = []
        position_ids_all = []
        position_ids_prefix_all = []

        for i in range(len(features)):
            input_ids = []
            attention_mask = []
            position_ids = []
            labels_ids = [-100]
            input_ids_prefix = []
            attention_mask_prefix = []
            position_ids_prefix = []
            labels_ids_prefix = [-100]

            seq_len = len(features[i]["input_ids"])

            if self.selection == "last":
                # ── LAST-N: trainable = positions [seq_len - N, seq_len) ─────
                if int(self.prefix_length) == self.prefix_length:
                    N = min(int(self.prefix_length), seq_len)
                else:
                    N = min(int(self.prefix_length * seq_len), seq_len)
                idx_with_grad = list(range(seq_len - N, seq_len))

            elif self.selection == "random":
                # ── RANDOM-K: K random positions + pos 0 + last position ─────
                # (original TokenTune paper / llama_token.py behavior)
                if int(self.prefix_length) == self.prefix_length:
                    idx_with_grad = np.random.choice(
                        range(1, seq_len - 1),
                        min(int(self.prefix_length), seq_len - 2),
                        replace=False,
                    ).tolist()
                else:
                    idx_with_grad = np.random.choice(
                        range(1, seq_len - 2),
                        min(
                            int(self.prefix_length * (seq_len - 2)),
                            seq_len - 2,
                        ),
                        replace=False,
                    ).tolist()
                idx_with_grad = [0] + idx_with_grad
                idx_with_grad.append(seq_len - 1)

            elif self.selection == "attention":
                # ── ATTENTION-BASED: top-K by precomputed importance scores ─
                # (TokenSeek-style content-aware selection)
                key = generate_identifier(features[i]["input_ids"])
                if key not in self.attention_map:
                    raise KeyError(
                        f"No precomputed scores for input hash {key}. "
                        "Did you run the scoring pipeline on this dataset?"
                    )
                entry = self.attention_map[key]
                assert seq_len == entry["len"], (
                    f"length mismatch for {key}: input has {seq_len} tokens "
                    f"but attention map has {entry['len']}"
                )

                if int(self.prefix_length) == self.prefix_length:
                    # Top-K by score, masking out positions 0 and last
                    scores = entry["scores"]
                    if not isinstance(scores, torch.Tensor):
                        scores = torch.tensor(scores)
                    mask = torch.full_like(scores, float("-inf"), dtype=torch.float)
                    mask[1:-1] = scores[1:-1].float()
                    K = min(int(self.prefix_length), seq_len - 2)
                    _, indices = torch.topk(mask, k=K)
                    idx_with_grad = indices.tolist()
                else:
                    # Pre-baked indices for this fractional prefix_length
                    idx_with_grad = entry[str(self.prefix_length)]

                # Always include position 0 and last position
                idx_with_grad = [0] + idx_with_grad
                idx_with_grad.append(seq_len - 1)

            else:
                raise ValueError(
                    f"Unknown selection mode {self.selection!r}. "
                    f"Expected one of: 'last', 'random', 'attention'."
                )

            for idx, id in enumerate(features[i]["input_ids"]):
                if idx in idx_with_grad:
                    input_ids_prefix.append(id)
                    attention_mask_prefix.append(features[i]["attention_mask"][idx])
                    position_ids_prefix.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids_prefix.append(features[i]["labels"][idx + 1])
                else:
                    input_ids.append(id)
                    attention_mask.append(features[i]["attention_mask"][idx])
                    position_ids.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids.append(features[i]["labels"][idx + 1])

            assert len(labels_ids_prefix) == len(input_ids_prefix), (
                len(labels_ids_prefix), len(input_ids_prefix),
            )
            if len(labels_ids) > len(input_ids):
                labels_ids = labels_ids[1:]
            if len(labels_ids_prefix) > len(input_ids_prefix):
                labels_ids_prefix = labels_ids_prefix[1:]
            position_ids_all.append(position_ids)
            position_ids_prefix_all.append(position_ids_prefix)
            sample_batch.append({
                "input_ids_prefix": input_ids_prefix,
                "attention_mask_prefix": attention_mask_prefix,
                "labels_ids_prefix": labels_ids_prefix,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels_ids,
            })

        batch = {k: [dic[k] for dic in sample_batch] for k in sample_batch[0]}

        for k, v in batch.items():
            if k.endswith("prefix"):
                if int(self.prefix_length) == self.prefix_length:
                    batch[k] = [vv[: int(self.prefix_length)] for vv in v]
                    max_len = max(max([len(vv) for vv in v]), int(self.prefix_length))
                    batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
                else:
                    batch[k] = [vv[: int(self.prefix_length * max([len(vv) for vv in v]))] for vv in v]
                    max_len = max(max([len(vv) for vv in v]),
                                  int(self.prefix_length * max([len(vv) for vv in v])))
                    batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])
            elif k == "label":
                batch[k] = torch.tensor(v)
            else:
                batch[k] = [vv[: self.max_length] for vv in v]
                max_len = max(max([len(vv) for vv in v]), 2)
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in v])

        for i in range(len(position_ids_all)):
            max_len = batch["input_ids"].shape[1]
            position_ids_all[i] = position_ids_all[i][:max_len]
            if len(position_ids_all[i]) < max_len:
                position_ids_all[i] = position_ids_all[i] + [0] * (max_len - len(position_ids_all[i]))
            max_len = batch["input_ids_prefix"].shape[1]
            position_ids_prefix_all[i] = position_ids_prefix_all[i][:max_len]
            if len(position_ids_prefix_all[i]) < max_len:
                position_ids_prefix_all[i] = position_ids_prefix_all[i] + [0] * (max_len - len(position_ids_prefix_all[i]))

        batch["position_ids_prefix"] = torch.tensor(position_ids_prefix_all, dtype=batch["input_ids"].dtype)
        batch["position_ids"] = torch.tensor(position_ids_all, dtype=batch["input_ids"].dtype)
        batch["labels"] = batch["labels_ids_prefix"]
        del batch["labels_ids_prefix"]
        del batch["labels_ids"]
        return batch


# ============================================================================
#  CONFIG
# ============================================================================

class LlamaConfigPrefix(LlamaConfig):
    def __init__(
        self,
        prefix_length: int = 8,
        prefix_hidden_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prefix_length = prefix_length
        self.prefix_hidden_size = prefix_hidden_size


# ============================================================================
#  OUTPUT TYPES
# ============================================================================

@dataclass
class CausalLMOutputWithPastAndPrefix(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPastAndPrefix(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    last_hidden_state_prefix: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_prefix: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# ============================================================================
#  MASK BUILDERS
# ============================================================================

def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _make_causal_mask_prefix(input_ids_shape, position_ids, position_ids_prefix, dtype, device, past_key_values_length=0):
    bsz, seq_len, pre_len = input_ids_shape
    mask = torch.full((bsz, seq_len + pre_len, seq_len + pre_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    mask_prefix = torch.gather(mask, 1, position_ids_prefix.unsqueeze(-1).repeat(1, 1, seq_len + pre_len))
    mask = torch.gather(mask, 1, position_ids.unsqueeze(-1).repeat(1, 1, seq_len + pre_len))
    mask_prefix = torch.gather(mask_prefix, 2, torch.cat((position_ids, position_ids_prefix), axis=1).unsqueeze(-2).repeat(1, pre_len, 1))
    mask = torch.gather(mask, 2, torch.cat((position_ids, position_ids_prefix), axis=1).unsqueeze(-2).repeat(1, seq_len, 1))
    return mask.unsqueeze(1), mask_prefix.unsqueeze(1)


def _expand_mask_prefix(mask, dtype, seq_len=None, pre_len=None):
    bsz, src_len = mask.size()
    tgt_len = pre_len + seq_len if seq_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# ============================================================================
#  TOP-LEVEL CAUSAL LM
# ============================================================================

class LlamaPrefixForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    # ── Critical for device_map="auto" ────────────────────────────────────────
    # Tells HF to treat each LlamaPrefixDecoderLayer as an atomic unit when
    # sharding across GPUs. Without this, HF inherits ["LlamaDecoderLayer"]
    # from the parent — which doesn't match our class name — and may split
    # attention / MLP across devices.
    _no_split_modules = ["LlamaPrefixDecoderLayer"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False  # we use eager for the dual-path attention

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.model = LlamaPrefixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prefix_length: int = (
            config.prefix_length if hasattr(config, "prefix_length") else 0
        )
        self.post_init()

    def get_input_embeddings(self):    return self.model.embed_tokens
    def set_input_embeddings(self, v): self.model.embed_tokens = v
    def get_output_embeddings(self):   return self.lm_head
    def set_output_embeddings(self, v):self.lm_head = v
    def set_decoder(self, d):          self.model = d
    def get_decoder(self):             return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_ids_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastAndPrefix]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            input_ids_prefix=input_ids_prefix, attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        prefix_output = outputs[1]
        # Move to lm_head's device for the final projection
        prefix_output = prefix_output.to(self.lm_head.weight.device)
        logits = self.lm_head(prefix_output).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndPrefix(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_states_prefix=outputs.hidden_states_prefix,
            attentions=outputs.attentions,
        )


# ============================================================================
#  LLAMA PREFIX MODEL
# ============================================================================

class LlamaPrefixModel(LlamaPreTrainedModel):
    _no_split_modules = ["LlamaPrefixDecoderLayer"]

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaPrefixDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.prefix_length: int = (
            config.prefix_length if hasattr(config, "prefix_length") else 0
        )
        self.post_init()

    def get_input_embeddings(self):    return self.embed_tokens
    def set_input_embeddings(self, v): self.embed_tokens = v

    def _prepare_decoder_attention_mask(
        self, attention_mask, attention_mask_prefix, input_shape,
        position_ids, position_ids_prefix, inputs_embeds, past_key_values_length,
    ):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask, combined_attention_mask_prefix = _make_causal_mask_prefix(
                input_shape, position_ids, position_ids_prefix,
                inputs_embeds.dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask_prefix(
                torch.cat((attention_mask, attention_mask_prefix), axis=1),
                inputs_embeds.dtype,
                seq_len=input_shape[-2], pre_len=input_shape[-1],
            ).to(inputs_embeds.device)
            expanded_attn_mask, expanded_attn_mask_prefix = (
                expanded_attn_mask[:, :, : input_shape[-2], :],
                expanded_attn_mask[:, :, input_shape[-2] :, :],
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
            combined_attention_mask_prefix = (
                expanded_attn_mask_prefix if combined_attention_mask_prefix is None
                else expanded_attn_mask_prefix + combined_attention_mask_prefix
            )
        return combined_attention_mask, combined_attention_mask_prefix

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        input_ids_prefix=None, attention_mask_prefix=None, position_ids_prefix=None,
        past_key_values=None, inputs_embeds=None, prefix_embeds=None,
        use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if position_ids_prefix is None:
            if int(self.prefix_length) == self.prefix_length:
                position_ids_prefix = torch.arange(
                    past_key_values_length,
                    int(self.prefix_length) + past_key_values_length,
                    dtype=torch.long, device=device,
                )
            else:
                position_ids_prefix = torch.arange(
                    past_key_values_length,
                    input_ids_prefix.shape[1] + past_key_values_length,
                    dtype=torch.long, device=device,
                )
            position_ids_prefix = position_ids_prefix.unsqueeze(0).view(-1, seq_length)

        # Embeddings — embed_tokens lives on its assigned device
        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        if prefix_embeds is None:
            prefix_embeds = self.embed_tokens(input_ids_prefix.to(self.embed_tokens.weight.device))

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool, device=inputs_embeds.device,
            )
        if attention_mask_prefix is None:
            if int(self.prefix_length) == self.prefix_length:
                attention_mask_prefix = torch.ones(
                    (batch_size, int(self.prefix_length)), device=inputs_embeds.device,
                )
            else:
                attention_mask_prefix = torch.ones(
                    (batch_size, prefix_embeds.shape[0]), device=inputs_embeds.device,
                )

        attention_mask, attention_mask_prefix = self._prepare_decoder_attention_mask(
            attention_mask, attention_mask_prefix,
            (batch_size, seq_length, prefix_embeds.shape[1]),
            position_ids, position_ids_prefix,
            inputs_embeds, past_key_values_length,
        )

        hidden_states = inputs_embeds
        hidden_states_prefix = prefix_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing.")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_prefix = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                all_hidden_states_prefix = all_hidden_states_prefix + (hidden_states_prefix,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                hidden_states_prefix=hidden_states_prefix,
                attention_mask_prefix=attention_mask_prefix,
                position_ids_prefix=position_ids_prefix,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            hidden_states_prefix = layer_outputs[1]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final norm — move to norm's device
        norm_device = self.norm.weight.device
        hidden_states_prefix = self.norm(hidden_states_prefix.to(norm_device))
        with torch.no_grad():
            hidden_states = self.norm(hidden_states.to(norm_device))

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_hidden_states_prefix = all_hidden_states_prefix + (hidden_states_prefix,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states,
                            hidden_states_prefix, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPastAndPrefix(
            last_hidden_state=hidden_states,
            last_hidden_state_prefix=hidden_states_prefix,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            hidden_states_prefix=all_hidden_states_prefix,
            attentions=all_self_attns,
        )


# ============================================================================
#  DECODER LAYER  — defensive device alignment for prefix tensors
# ============================================================================

class LlamaPrefixDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaPrefixAttention(config=config)
        self.mlp = LlamaPrefixMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hidden_states_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        # ── DEFENSIVE DEVICE ALIGNMENT ───────────────────────────────────────
        # Ensure every tensor input is on this layer's device. Accelerate's
        # AlignDevicesHook normally does this for standard layers, but our
        # extra prefix kwargs aren't always picked up reliably across hooks.
        layer_device = next(self.parameters()).device

        hidden_states = hidden_states.to(layer_device, non_blocking=True)
        if hidden_states_prefix is not None:
            hidden_states_prefix = hidden_states_prefix.to(layer_device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(layer_device, non_blocking=True)
        if attention_mask_prefix is not None:
            attention_mask_prefix = attention_mask_prefix.to(layer_device, non_blocking=True)
        if position_ids is not None:
            position_ids = position_ids.to(layer_device, non_blocking=True)
        if position_ids_prefix is not None:
            position_ids_prefix = position_ids_prefix.to(layer_device, non_blocking=True)

        residual = hidden_states
        residual_prefix = hidden_states_prefix

        with torch.no_grad():
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states_prefix = self.input_layernorm(hidden_states_prefix)

        (hidden_states, hidden_states_prefix, self_attn_weights, present_key_value) = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            hidden_states_prefix=hidden_states_prefix, attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states_prefix = residual_prefix + hidden_states_prefix

        residual = hidden_states
        residual_prefix = hidden_states_prefix

        with torch.no_grad():
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states_prefix = self.post_attention_layernorm(hidden_states_prefix)
        hidden_states_prefix = self.mlp(hidden_states_prefix)
        hidden_states_prefix = residual_prefix + hidden_states_prefix

        outputs = (hidden_states, hidden_states_prefix)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache:         outputs += (present_key_value,)
        return outputs


# ============================================================================
#  ATTENTION (modernized RoPE)
# ============================================================================

class LlamaPrefixAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hidden_states_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()
        _, prefix_len, _ = hidden_states_prefix.size()

        with torch.no_grad():
            query_states = self.q_proj(hidden_states)
            key_states   = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_prefix = self.q_proj(hidden_states_prefix)
        key_states_prefix   = self.k_proj(hidden_states_prefix)
        value_states_prefix = self.v_proj(hidden_states_prefix)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states_prefix = query_states_prefix.view(bsz, prefix_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_prefix   = key_states_prefix.view(bsz, prefix_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_prefix = value_states_prefix.view(bsz, prefix_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        with torch.no_grad():
            cos_main, sin_main = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_main, sin_main)

        cos_pref, sin_pref = self.rotary_emb(value_states_prefix, position_ids_prefix)
        query_states_prefix, key_states_prefix = apply_rotary_pos_emb(
            query_states_prefix, key_states_prefix, cos_pref, sin_pref
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states_prefix   = repeat_kv(key_states_prefix, self.num_key_value_groups)
        value_states_prefix = repeat_kv(value_states_prefix, self.num_key_value_groups)

        with torch.no_grad():
            attn_weights = torch.matmul(
                query_states,
                torch.cat((key_states, key_states_prefix), axis=2).transpose(2, 3),
            ) / math.sqrt(self.head_dim)

        attn_weights_prefix = torch.matmul(
            query_states_prefix,
            torch.cat((key_states, key_states_prefix), axis=2).transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if attention_mask_prefix is not None:
            attn_weights_prefix = attn_weights_prefix + attention_mask_prefix

        with torch.no_grad():
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(
                attn_weights, torch.cat((value_states, value_states_prefix), axis=2),
            )

        attn_weights_prefix = nn.functional.softmax(
            attn_weights_prefix, dim=-1, dtype=torch.float32,
        ).to(query_states_prefix.dtype)

        attn_output_prefix = torch.matmul(
            attn_weights_prefix, torch.cat((value_states, value_states_prefix), axis=2),
        )

        with torch.no_grad():
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output_prefix = attn_output_prefix.transpose(1, 2).contiguous().reshape(bsz, prefix_len, self.hidden_size)

        with torch.no_grad():
            attn_output = self.o_proj(attn_output)
        attn_output_prefix = self.o_proj(attn_output_prefix)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_output_prefix, attn_weights, past_key_value


# ============================================================================
#  MLP
# ============================================================================

class LlamaPrefixMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))