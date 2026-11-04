# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 The Hugging Face Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# TokenTune-style prefix architecture for Llama, updated for modern transformers (>=4.46).
#
# Key API changes vs the original file:
#   * Rotary embedding now lives at the model level (LlamaModel.rotary_emb), not per-attention.
#   * cos/sin are computed once per forward and passed down as `position_embeddings`.
#   * `apply_rotary_pos_emb(q, k, cos, sin)` no longer takes position_ids.
#   * Removed deprecated LlamaLinearScalingRotaryEmbedding / LlamaDynamicNTKScalingRotaryEmbedding.
#
# Dual-path semantics (TokenTune) preserved:
#   * "prefix" path = K randomly selected tokens, carries gradients
#   * "main" path   = remaining tokens, wrapped in torch.no_grad() to save activation memory
#   * Both paths share parameters; loss is computed only on prefix tokens.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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


# =============================================================================
#                              DATA COLLATOR
# =============================================================================

@dataclass
class DataCollatorWithPaddingForPrefix:
    """
    Splits each sequence into K "prefix" tokens (selected for gradient flow)
    and the remaining "main" tokens. Pads each group separately.

    Output batch keys:
      input_ids, attention_mask, position_ids,
      input_ids_prefix, attention_mask_prefix, position_ids_prefix,
      labels   (= shifted labels for prefix tokens only)
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    prefix_length: Union[int, float] = 16

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

            if int(self.prefix_length) == self.prefix_length:
                idx_with_grad = np.random.choice(
                    range(1, seq_len - 1),
                    min(int(self.prefix_length), seq_len - 2),
                    replace=False,
                ).tolist()
            else:
                idx_with_grad = np.random.choice(
                    range(1, seq_len - 2),
                    min(int(self.prefix_length * (seq_len - 2)), seq_len - 2),
                    replace=False,
                ).tolist()

            idx_with_grad = [0] + idx_with_grad
            idx_with_grad.append(seq_len - 1)

            for idx, tok_id in enumerate(features[i]["input_ids"]):
                if idx in idx_with_grad:
                    input_ids_prefix.append(tok_id)
                    attention_mask_prefix.append(features[i]["attention_mask"][idx])
                    position_ids_prefix.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids_prefix.append(features[i]["labels"][idx + 1])
                else:
                    input_ids.append(tok_id)
                    attention_mask.append(features[i]["attention_mask"][idx])
                    position_ids.append(idx)
                    if idx + 1 < len(features[i]["labels"]):
                        labels_ids.append(features[i]["labels"][idx + 1])

            assert len(labels_ids_prefix) == len(input_ids_prefix), (
                len(labels_ids_prefix), len(input_ids_prefix)
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

        batch = {k: [d[k] for d in sample_batch] for k in sample_batch[0]}

        for k, v in batch.items():
            if k.endswith("prefix"):
                if int(self.prefix_length) == self.prefix_length:
                    batch[k] = [vv[: int(self.prefix_length)] for vv in v]
                    max_len = max(max(len(vv) for vv in v), int(self.prefix_length))
                else:
                    max_seq = max(len(vv) for vv in v)
                    batch[k] = [vv[: int(self.prefix_length * max_seq)] for vv in v]
                    max_len = max(max_seq, int(self.prefix_length * max_seq))
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in batch[k]])
            elif k == "label":
                batch[k] = torch.tensor(v)
            else:
                batch[k] = [vv[: self.max_length] for vv in v]
                max_len = max(max(len(vv) for vv in batch[k]), 2)
                batch[k] = torch.tensor([vv + [0] * (max_len - len(vv)) for vv in batch[k]])

        # Pad position id lists to match batch tensor widths
        for i in range(len(position_ids_all)):
            max_len = batch["input_ids"].shape[1]
            position_ids_all[i] = position_ids_all[i][:max_len]
            if len(position_ids_all[i]) < max_len:
                position_ids_all[i] += [0] * (max_len - len(position_ids_all[i]))

            max_len = batch["input_ids_prefix"].shape[1]
            position_ids_prefix_all[i] = position_ids_prefix_all[i][:max_len]
            if len(position_ids_prefix_all[i]) < max_len:
                position_ids_prefix_all[i] += [0] * (max_len - len(position_ids_prefix_all[i]))

        batch["position_ids_prefix"] = torch.tensor(position_ids_prefix_all, dtype=batch["input_ids"].dtype)
        batch["position_ids"] = torch.tensor(position_ids_all, dtype=batch["input_ids"].dtype)
        batch["labels"] = batch["labels_ids_prefix"]
        del batch["labels_ids_prefix"]
        del batch["labels_ids"]
        return batch


# =============================================================================
#                                 CONFIG
# =============================================================================

class LlamaConfigPrefix(LlamaConfig):
    """LlamaConfig + prefix-specific fields."""

    def __init__(
        self,
        prefix_length: Union[int, float] = 8,
        prefix_hidden_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.prefix_length = prefix_length
        self.prefix_hidden_size = prefix_hidden_size


# =============================================================================
#                            OUTPUT DATACLASSES
# =============================================================================

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


# =============================================================================
#                          MASK UTILITIES (PREFIX-AWARE)
# =============================================================================

def _make_causal_mask_prefix(
    input_ids_shape: torch.Size,
    position_ids: torch.Tensor,
    position_ids_prefix: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """Return two causal masks (one for the main path, one for the prefix path)."""
    bsz, seq_len, pre_len = input_ids_shape
    mask = torch.full(
        (bsz, seq_len + pre_len, seq_len + pre_len),
        torch.finfo(dtype).min,
        device=device,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    mask_prefix = torch.gather(
        mask, 1, position_ids_prefix.unsqueeze(-1).repeat(1, 1, seq_len + pre_len)
    )
    mask = torch.gather(
        mask, 1, position_ids.unsqueeze(-1).repeat(1, 1, seq_len + pre_len)
    )
    mask_prefix = torch.gather(
        mask_prefix, 2,
        torch.cat((position_ids, position_ids_prefix), dim=1)
        .unsqueeze(-2).repeat(1, pre_len, 1),
    )
    mask = torch.gather(
        mask, 2,
        torch.cat((position_ids, position_ids_prefix), dim=1)
        .unsqueeze(-2).repeat(1, seq_len, 1),
    )
    return mask.unsqueeze(1), mask_prefix.unsqueeze(1)


def _expand_mask_prefix(
    mask: torch.Tensor,
    dtype: torch.dtype,
    seq_len: Optional[int] = None,
    pre_len: Optional[int] = None,
):
    """Expand `[bsz, src_len]` to `[bsz, 1, tgt_len, src_len]`."""
    bsz, src_len = mask.size()
    tgt_len = pre_len + seq_len if seq_len is not None else src_len
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


# =============================================================================
#                                ATTENTION
# =============================================================================

class LlamaPrefixAttention(nn.Module):
    """Multi-headed attention with TokenTune dual-path forward."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.attention_bias)
        # Note: no per-layer rotary_emb; it lives at the model level now.

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.FloatTensor] = None,
        position_ids_prefix: Optional[torch.LongTensor] = None,
        position_embeddings_prefix: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _, prefix_len, _ = hidden_states_prefix.size()

        # ── Q/K/V projections ────────────────────────────────────────────────
        with torch.no_grad():
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states_prefix = self.q_proj(hidden_states_prefix)
        key_states_prefix = self.k_proj(hidden_states_prefix)
        value_states_prefix = self.v_proj(hidden_states_prefix)

        # ── Reshape to (bsz, n_heads, seq, head_dim) ─────────────────────────
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states_prefix = query_states_prefix.view(bsz, prefix_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_prefix = key_states_prefix.view(bsz, prefix_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_prefix = value_states_prefix.view(bsz, prefix_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ── Apply RoPE (modern API: cos/sin already gathered for these positions) ──
        cos, sin = position_embeddings
        cos_p, sin_p = position_embeddings_prefix
        with torch.no_grad():
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states_prefix, key_states_prefix = apply_rotary_pos_emb(
            query_states_prefix, key_states_prefix, cos_p, sin_p
        )

        # ── Repeat KV heads for GQA (no-op for Llama-2 which has no GQA) ─────
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_states_prefix = repeat_kv(key_states_prefix, self.num_key_value_groups)
        value_states_prefix = repeat_kv(value_states_prefix, self.num_key_value_groups)

        kv_seq_len = key_states.shape[-2]
        kv_prefix_len = key_states_prefix.shape[-2]

        # Both paths attend to the concatenation of all keys/values
        all_keys = torch.cat((key_states, key_states_prefix), dim=2)
        all_values = torch.cat((value_states, value_states_prefix), dim=2)

        # ── Main path attention (no grad) ────────────────────────────────────
        with torch.no_grad():
            attn_weights = torch.matmul(query_states, all_keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len + kv_prefix_len):
                raise ValueError(
                    f"attn_weights shape mismatch: got {attn_weights.size()}, "
                    f"expected {(bsz, self.num_heads, q_len, kv_seq_len + kv_prefix_len)}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len + kv_prefix_len):
                    raise ValueError(
                        f"attention_mask shape mismatch: got {attention_mask.size()}, "
                        f"expected {(bsz, 1, q_len, kv_seq_len + kv_prefix_len)}"
                    )
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, all_values)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

        # ── Prefix path attention (with grad) ────────────────────────────────
        attn_weights_prefix = torch.matmul(
            query_states_prefix, all_keys.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask_prefix is not None:
            if attention_mask_prefix.size() != (bsz, 1, kv_prefix_len, kv_seq_len + kv_prefix_len):
                raise ValueError(
                    f"attention_mask_prefix shape mismatch: got {attention_mask_prefix.size()}, "
                    f"expected {(bsz, 1, kv_prefix_len, kv_seq_len + kv_prefix_len)}"
                )
            attn_weights_prefix = attn_weights_prefix + attention_mask_prefix

        attn_weights_prefix = nn.functional.softmax(
            attn_weights_prefix, dim=-1, dtype=torch.float32
        ).to(query_states_prefix.dtype)
        attn_output_prefix = torch.matmul(attn_weights_prefix, all_values)
        attn_output_prefix = attn_output_prefix.transpose(1, 2).contiguous().reshape(bsz, prefix_len, self.hidden_size)
        attn_output_prefix = self.o_proj(attn_output_prefix)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_output_prefix, attn_weights, None  # past_key_value not maintained


# =============================================================================
#                                  MLP
# =============================================================================

class LlamaPrefixMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
#                              DECODER LAYER
# =============================================================================

class LlamaPrefixDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaPrefixAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaPrefixMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states_prefix: Optional[torch.Tensor] = None,
        attention_mask_prefix: Optional[torch.Tensor] = None,
        position_ids_prefix: Optional[torch.Tensor] = None,
        position_embeddings_prefix: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        residual_prefix = hidden_states_prefix

        with torch.no_grad():
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states_prefix = self.input_layernorm(hidden_states_prefix)

        # Self attention
        hidden_states, hidden_states_prefix, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            hidden_states_prefix=hidden_states_prefix,
            attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix,
            position_embeddings_prefix=position_embeddings_prefix,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states_prefix = residual_prefix + hidden_states_prefix

        # MLP
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
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# =============================================================================
#                                  MODEL
# =============================================================================

class LlamaPrefixModel(LlamaPreTrainedModel):
    """Transformer decoder built from LlamaPrefixDecoderLayer."""

    _no_split_modules = ["LlamaPrefixDecoderLayer"]
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_sdpa = False
    _supports_flash_attn_2 = False

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaPrefixDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # NEW: model-level rotary embedding (shared across all layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.prefix_length = config.prefix_length if hasattr(config, "prefix_length") else 0

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        attention_mask_prefix,
        input_shape,
        position_ids,
        position_ids_prefix,
        inputs_embeds,
        past_key_values_length,
    ):
        # input_shape: (batch_size, seq_len, prefix_len)
        combined_attention_mask = None
        combined_attention_mask_prefix = None

        if input_shape[-2] > 1:
            combined_attention_mask, combined_attention_mask_prefix = _make_causal_mask_prefix(
                input_shape,
                position_ids,
                position_ids_prefix,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded = _expand_mask_prefix(
                torch.cat((attention_mask, attention_mask_prefix), dim=1),
                inputs_embeds.dtype,
                seq_len=input_shape[-2],
                pre_len=input_shape[-1],
            ).to(inputs_embeds.device)
            expanded_main = expanded[:, :, : input_shape[-2], :]
            expanded_prefix = expanded[:, :, input_shape[-2]:, :]
            combined_attention_mask = (
                expanded_main if combined_attention_mask is None
                else expanded_main + combined_attention_mask
            )
            combined_attention_mask_prefix = (
                expanded_prefix if combined_attention_mask_prefix is None
                else expanded_prefix + combined_attention_mask_prefix
            )

        return combined_attention_mask, combined_attention_mask_prefix

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
        prefix_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndPrefix]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("specify either input_ids or inputs_embeds, not both")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("must specify either input_ids or inputs_embeds")

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

        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)

        if prefix_embeds is None:
            if input_ids_prefix is None:
                raise ValueError(
                    "input_ids_prefix is required at training time. "
                    "Use DataCollatorWithPaddingForPrefix to construct batches."
                )
            prefix_embeds = self.embed_tokens(input_ids_prefix)

        prefix_seq_len = prefix_embeds.shape[1]

        if position_ids_prefix is None:
            position_ids_prefix = torch.arange(
                past_key_values_length, prefix_seq_len + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            position_ids_prefix = position_ids_prefix.long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.long, device=device,
            )
        if attention_mask_prefix is None:
            attention_mask_prefix = torch.ones(
                (batch_size, prefix_seq_len),
                dtype=torch.long, device=device,
            )

        attention_mask, attention_mask_prefix = self._prepare_decoder_attention_mask(
            attention_mask,
            attention_mask_prefix,
            (batch_size, seq_length, prefix_seq_len),
            position_ids,
            position_ids_prefix,
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        hidden_states_prefix = prefix_embeds

        # NEW: compute rotary cos/sin once at the model level for both paths
        cos_main, sin_main = self.rotary_emb(hidden_states, position_ids)
        cos_pref, sin_pref = self.rotary_emb(hidden_states_prefix, position_ids_prefix)
        position_embeddings = (cos_main, sin_main)
        position_embeddings_prefix = (cos_pref, sin_pref)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_hidden_states_prefix = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                all_hidden_states_prefix += (hidden_states_prefix,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                hidden_states_prefix=hidden_states_prefix,
                attention_mask_prefix=attention_mask_prefix,
                position_ids_prefix=position_ids_prefix,
                position_embeddings_prefix=position_embeddings_prefix,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            hidden_states_prefix = layer_outputs[1]

            # layer_outputs = (hidden, hidden_prefix, [attn], [past_kv])
            cur_idx = 2
            if output_attentions:
                all_self_attns += (layer_outputs[cur_idx],)
                cur_idx += 1
            if use_cache:
                next_decoder_cache += (layer_outputs[cur_idx],)

        hidden_states_prefix = self.norm(hidden_states_prefix)
        with torch.no_grad():
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            all_hidden_states_prefix += (hidden_states_prefix,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states,
                hidden_states_prefix, all_self_attns,
            ] if v is not None)

        return BaseModelOutputWithPastAndPrefix(
            last_hidden_state=hidden_states,
            last_hidden_state_prefix=hidden_states_prefix,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            hidden_states_prefix=all_hidden_states_prefix,
            attentions=all_self_attns,
        )


# =============================================================================
#                              CAUSAL LM HEAD
# =============================================================================

class LlamaPrefixForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["LlamaPrefixDecoderLayer"]
    _supports_cache_class = False
    _supports_static_cache = False
    _supports_sdpa = False
    _supports_flash_attn_2 = False

    def __init__(self, config: LlamaConfigPrefix):
        super().__init__(config)
        self.model = LlamaPrefixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prefix_length = config.prefix_length if hasattr(config, "prefix_length") else 0
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_ids_prefix=input_ids_prefix,
            attention_mask_prefix=attention_mask_prefix,
            position_ids_prefix=position_ids_prefix,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # We only compute logits/loss for the prefix tokens (the only ones with grad).
        prefix_output = outputs.last_hidden_state_prefix if return_dict else outputs[1]
        logits = self.lm_head(prefix_output).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndPrefix(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            hidden_states_prefix=outputs.hidden_states_prefix,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered = ()
        for layer_past in past_key_values:
            reordered += (
                tuple(p.index_select(0, beam_idx.to(p.device)) for p in layer_past),
            )
        return reordered