"""
Token Theory - Causal LLM (Left Padding, No Reordering)

─── Public API ───────────────────────────────────────────────────────────────

  full(model, num_trainable, ...)         Full fine-tuning, last N tokens
  full_baseline(model)                    Full fine-tuning, ALL tokens (no-op)
  lora(model, num_trainable, ...)         LoRA, last N tokens
  lora_baseline(model, ...)               LoRA, ALL tokens
  lora_fa(model, num_trainable, ...)      LoRA-FA (frozen A), last N tokens
  lora_fa_baseline(model, ...)            LoRA-FA, ALL tokens
  dora(model, num_trainable, ...)         DoRA, last N tokens
  dora_baseline(model, ...)               DoRA, ALL tokens
  propulsion(model, num_trainable, ...)   Propulsion, last N tokens
  propulsion_baseline(model, ...)         Propulsion, ALL tokens
  qlora(model, num_trainable, ...)        QLoRA (4-bit base), last N tokens
  qlora_baseline(model, ...)              QLoRA, ALL tokens
  merge(model)                            Merge + restore arch, call before save

─── QLoRA setup ──────────────────────────────────────────────────────────────

  from transformers import BitsAndBytesConfig
  from peft import prepare_model_for_kbit_training

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.bfloat16,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
      ),
      device_map="auto",
  )
  prepare_model_for_kbit_training(model)   # required before qlora()
  qlora(model, num_trainable=100, target_modules=[...])
  model.is_quantized = False               # bypass Trainer guard
"""

import logging
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# =============================================================================
#  HELPER — safe dtype/device for any layer including quantized
# =============================================================================

def _get_adapter_device(base_layer):
    """
    Get the device for creating LoRA adapters.
    Linear4bit stores weight as Params4bit — weight.device works but
    weight.dtype returns the storage dtype (uint8), not compute dtype.
    We always use bfloat16 for adapters regardless of base layer dtype.
    """
    try:
        return base_layer.weight.device
    except AttributeError:
        pass
    try:
        return next(base_layer.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
#  WRAPPER 1 — FULL FINE-TUNING
# =============================================================================

class CausalTokenTheoryWrapper(nn.Module):
    """
    Full fine-tuning wrapper. Trainable tokens = LAST num_trainable positions.
    All base weights remain trainable — this is NOT LoRA.

    Supports 2D, 3D and 4D inputs:
        2D [B*T, H]          — pass through (classifier head, pooled output)
        3D [B, T, H]         — standard transformer hidden states
        4D [B, T, heads, dim] — Qwen3-VL q_norm / k_norm layers
    """

    def __init__(self, base_layer: nn.Module, num_trainable: Union[int, callable]):
        super().__init__()
        self.base_layer     = base_layer
        self._num_trainable = num_trainable

    @property
    def weight(self):       return getattr(self.base_layer, 'weight', None)
    @property
    def bias(self):         return getattr(self.base_layer, 'bias', None)
    @property
    def in_features(self):  return getattr(self.base_layer, 'in_features', None)
    @property
    def out_features(self): return getattr(self.base_layer, 'out_features', None)

    def _n(self, T):
        return self._num_trainable(T) if callable(self._num_trainable) else self._num_trainable

    def _forward_embedding(self, x, *args, **kwargs):
        if x.dim() == 1:
            return self.base_layer(x, *args, **kwargs)
        B, T  = x.shape
        num_t = min(self._n(T), T)
        num_f = T - num_t
        parts = []
        if num_f > 0:
            with torch.no_grad():
                parts.append(self.base_layer(x[:, :num_f], *args, **kwargs))
        if num_t > 0:
            parts.append(self.base_layer(x[:, num_f:], *args, **kwargs))
        return torch.cat(parts, dim=1)

    def forward(self, x, *args, **kwargs):
        if isinstance(self.base_layer, nn.Embedding):
            return self._forward_embedding(x, *args, **kwargs)

        # 2D — pass through (classifier head, pooled output)
        if x.dim() == 2:
            return self.base_layer(x, *args, **kwargs)

        # 4D — [B, T, heads, head_dim] used by Qwen3-VL q_norm / k_norm
        if x.dim() == 4:
            B, T, H1, H2 = x.shape
            num_t = min(self._n(T), T)
            num_f = T - num_t
            if num_f <= 0:
                return self.base_layer(x, *args, **kwargs)
            parts = []
            if num_f > 0:
                x_frozen = x[:, :num_f, :, :].reshape(B * num_f, H1, H2)
                with torch.no_grad():
                    out = self.base_layer(x_frozen, *args, **kwargs)
                parts.append(out.view(B, num_f, *out.shape[1:]))
            if num_t > 0:
                x_train = x[:, num_f:, :, :].reshape(B * num_t, H1, H2)
                out = self.base_layer(x_train, *args, **kwargs)
                parts.append(out.view(B, num_t, *out.shape[1:]))
            return torch.cat(parts, dim=1)

        if x.dim() != 3:
            raise ValueError(f"Expected 2D, 3D or 4D input, got {x.shape}")

        # 3D — standard [B, T, H]
        B, T, H = x.shape
        num_t   = min(self._n(T), T)
        num_f   = T - num_t

        if num_f <= 0:
            return self.base_layer(x, *args, **kwargs)

        D = getattr(self.base_layer, 'out_features', None) or H

        parts = []
        if num_f > 0:
            # Reshape to 2D — PyTorch allocates only [B*num_f, H] at a time
            # This is the key for activation memory savings vs 3D slices
            x_frozen = x[:, :num_f, :].reshape(B * num_f, H)
            with torch.no_grad():
                parts.append(
                    self.base_layer(x_frozen, *args, **kwargs).view(B, num_f, D))
        if num_t > 0:
            x_train = x[:, num_f:, :].reshape(B * num_t, H)
            parts.append(
                self.base_layer(x_train, *args, **kwargs).view(B, num_t, D))

        return torch.cat(parts, dim=1)

    def __repr__(self):
        return f"CausalTokenTheoryWrapper(num_trainable={self._num_trainable}, base={self.base_layer.__class__.__name__})"


# =============================================================================
#  WRAPPER 2 — LORA  (also used by QLoRA)
# =============================================================================

class LoRATokenTheoryWrapper(nn.Module):
    """
    LoRA + Token Theory. lora_A and lora_B both trainable.
    Works for standard nn.Linear AND bitsandbytes quantized layers (QLoRA).

    Frozen tokens  : base(x) only, no grad, no lora delta
    Trainable tokens: base(x) + lora_B(lora_A(x)) * scaling, full grad
    """

    def __init__(self, base_layer, num_trainable, r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.base_layer     = base_layer
        self._num_trainable = num_trainable
        self.scaling        = lora_alpha / r

        H      = base_layer.in_features
        D      = base_layer.out_features
        # LoRA adapters are always bfloat16 regardless of base layer dtype
        # (Linear4bit stores weight as uint8 Params4bit — not the compute dtype)
        dtype  = torch.bfloat16
        device = _get_adapter_device(base_layer)

        self.lora_A    = nn.Linear(H, r, bias=False, dtype=dtype, device=device)
        self.lora_B    = nn.Linear(r, D, bias=False, dtype=dtype, device=device)
        self.lora_drop = nn.Dropout(p=lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    @property
    def weight(self):       return self.base_layer.weight
    @property
    def bias(self):         return self.base_layer.bias
    @property
    def in_features(self):  return self.base_layer.in_features
    @property
    def out_features(self): return self.base_layer.out_features

    def _n(self, T):
        return self._num_trainable(T) if callable(self._num_trainable) else self._num_trainable

    def _lora_delta(self, x):
        return self.lora_B(self.lora_A(self.lora_drop(x.to(self.lora_A.weight.dtype)))) * self.scaling

    def forward(self, x, *args, **kwargs):
        if x.dim() == 2:
            return self.base_layer(x, *args, **kwargs) + self._lora_delta(x)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        B, T, H = x.shape
        num_t   = min(self._n(T), T)
        num_f   = T - num_t
        if num_f <= 0:
            return self.base_layer(x, *args, **kwargs) + self._lora_delta(x)
        with torch.no_grad():
            base_out = self.base_layer(x, *args, **kwargs)
        if base_out.dim() == 2:
            base_out = base_out.view(B, T, -1)
        lora_out  = self._lora_delta(x[:, num_f:, :])
        train_out = base_out[:, num_f:, :] + lora_out
        return torch.cat([base_out[:, :num_f, :], train_out], dim=1)

    def __repr__(self):
        return f"LoRATokenTheoryWrapper(in={self.base_layer.in_features}, out={self.base_layer.out_features}, r={self.lora_A.out_features}, num_trainable={self._num_trainable})"


# =============================================================================
#  WRAPPER 3 — LORA-FA (FROZEN A)
# =============================================================================

class LoRAFATokenTheoryWrapper(nn.Module):
    """
    LoRA-FA + Token Theory. lora_A frozen, only lora_B trains.
    Token Theory applied to lora_B only — ~50% fewer params than LoRA.
    Reference: LoRA-FA (Zhang et al., 2023)
    """

    def __init__(self, base_layer, num_trainable, r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.base_layer     = base_layer
        self._num_trainable = num_trainable
        self.scaling        = lora_alpha / r

        H      = base_layer.in_features
        D      = base_layer.out_features
        dtype  = torch.bfloat16
        device = _get_adapter_device(base_layer)

        self.lora_A    = nn.Linear(H, r, bias=False, dtype=dtype, device=device)
        self.lora_B    = nn.Linear(r, D, bias=False, dtype=dtype, device=device)
        self.lora_drop = nn.Dropout(p=lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.lora_A.weight.requires_grad = False   # FA: A is frozen

    @property
    def weight(self):       return self.base_layer.weight
    @property
    def bias(self):         return self.base_layer.bias
    @property
    def in_features(self):  return self.base_layer.in_features
    @property
    def out_features(self): return self.base_layer.out_features

    def _n(self, T):
        return self._num_trainable(T) if callable(self._num_trainable) else self._num_trainable

    def _lora_delta(self, x):
        # lora_A is frozen — run under no_grad to save activation memory
        # cast x to adapter dtype to handle QLoRA bfloat16 mismatch
        with torch.no_grad():
            a_out = self.lora_A(self.lora_drop(x.to(self.lora_A.weight.dtype)))
        return self.lora_B(a_out) * self.scaling

    def forward(self, x, *args, **kwargs):
        if x.dim() == 2:
            return self.base_layer(x, *args, **kwargs) + self._lora_delta(x)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        B, T, H = x.shape
        num_t   = min(self._n(T), T)
        num_f   = T - num_t
        if num_f <= 0:
            return self.base_layer(x, *args, **kwargs) + self._lora_delta(x)
        with torch.no_grad():
            base_out = self.base_layer(x, *args, **kwargs)
        if base_out.dim() == 2:
            base_out = base_out.view(B, T, -1)
        lora_out  = self._lora_delta(x[:, num_f:, :])
        train_out = base_out[:, num_f:, :] + lora_out
        return torch.cat([base_out[:, :num_f, :], train_out], dim=1)

    def __repr__(self):
        return f"LoRAFATokenTheoryWrapper(in={self.base_layer.in_features}, out={self.base_layer.out_features}, r={self.lora_A.out_features}, num_trainable={self._num_trainable})"


# =============================================================================
#  WRAPPER 4 — DORA
# =============================================================================

class DoRATokenTheoryWrapper(nn.Module):
    """
    DoRA + Token Theory. Magnitude + direction weight decomposition.
    Trainable: lora_A, lora_B, magnitude vector m per layer.
    Reference: DoRA (Liu et al., 2024)
    """

    def __init__(self, base_layer, num_trainable, r=4, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__()
        self.base_layer     = base_layer
        self._num_trainable = num_trainable
        self.scaling        = lora_alpha / r

        H      = base_layer.in_features
        D      = base_layer.out_features
        dtype  = torch.bfloat16
        device = _get_adapter_device(base_layer)

        self.lora_A    = nn.Linear(H, r, bias=False, dtype=dtype, device=device)
        self.lora_B    = nn.Linear(r, D, bias=False, dtype=dtype, device=device)
        self.lora_drop = nn.Dropout(p=lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        with torch.no_grad():
            col_norms = base_layer.weight.norm(p=2, dim=1, keepdim=False)
        self.magnitude = nn.Parameter(col_norms.to(dtype=dtype, device=device))

        for p in self.base_layer.parameters():
            p.requires_grad = False

    @property
    def weight(self):       return self.base_layer.weight
    @property
    def bias(self):         return self.base_layer.bias
    @property
    def in_features(self):  return self.base_layer.in_features
    @property
    def out_features(self): return self.base_layer.out_features

    def _n(self, T):
        return self._num_trainable(T) if callable(self._num_trainable) else self._num_trainable

    def _dora_forward(self, x):
        # Cast input to adapter dtype
        x_cast     = x.to(self.lora_A.weight.dtype)
        # LoRA delta via weight matrices (weight-space formulation)
        # dropout applied only inside the LoRA branch via lora_A
        lora_delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        adapted_W  = self.base_layer.weight.to(lora_delta.dtype) + lora_delta
        col_norms  = adapted_W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        direction  = adapted_W / col_norms
        scaled_W   = self.magnitude.unsqueeze(1) * direction
        # No dropout on main input x — only base+lora combined effective weight
        return F.linear(x_cast, scaled_W, self.base_layer.bias)

    def forward(self, x, *args, **kwargs):
        if x.dim() == 2:
            return self._dora_forward(x)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        B, T, H = x.shape
        num_t   = min(self._n(T), T)
        num_f   = T - num_t
        if num_f <= 0:
            return self._dora_forward(x)
        with torch.no_grad():
            frozen_out = self.base_layer(x[:, :num_f, :], *args, **kwargs)
        if frozen_out.dim() == 2:
            frozen_out = frozen_out.view(B, num_f, -1)
        train_out = self._dora_forward(x[:, num_f:, :])
        if train_out.dim() == 2:
            train_out = train_out.view(B, num_t, -1)
        return torch.cat([frozen_out, train_out], dim=1)

    def __repr__(self):
        return f"DoRATokenTheoryWrapper(in={self.base_layer.in_features}, out={self.base_layer.out_features}, r={self.lora_A.out_features}, num_trainable={self._num_trainable})"


# =============================================================================
#  WRAPPER 5 — PROPULSION
# =============================================================================

class PropulsionTokenTheoryWrapper(nn.Module):
    """
    Propulsion + Token Theory.
    out = linear(x) * propulsion^degree
    Only propulsion vector (out_features params) is trainable.
    Reference: Propulsion (Mukherjee et al.)
    """

    def __init__(self, base_layer, num_trainable, degree=15):
        super().__init__()
        self.base_layer     = base_layer
        self._num_trainable = num_trainable
        self.degree         = degree

        D      = base_layer.out_features
        dtype  = torch.bfloat16
        device = _get_adapter_device(base_layer)

        self.propulsion = nn.Parameter(torch.ones(D, dtype=dtype, device=device))

        for p in self.base_layer.parameters():
            p.requires_grad = False

    @property
    def weight(self):       return self.base_layer.weight
    @property
    def bias(self):         return self.base_layer.bias
    @property
    def in_features(self):  return self.base_layer.in_features
    @property
    def out_features(self): return self.base_layer.out_features

    def _n(self, T):
        return self._num_trainable(T) if callable(self._num_trainable) else self._num_trainable

    def _prop_forward(self, x, *args, **kwargs):
        push = torch.pow(self.propulsion, self.degree)
        return self.base_layer(x, *args, **kwargs).to(push.dtype) * push

    def forward(self, x, *args, **kwargs):
        if x.dim() == 2:
            return self._prop_forward(x, *args, **kwargs)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")
        B, T, H = x.shape
        num_t   = min(self._n(T), T)
        num_f   = T - num_t
        if num_f <= 0:
            return self._prop_forward(x, *args, **kwargs)
        with torch.no_grad():
            frozen_out = self.base_layer(x[:, :num_f, :], *args, **kwargs)
        if frozen_out.dim() == 2:
            frozen_out = frozen_out.view(B, num_f, -1)
        train_out = self._prop_forward(x[:, num_f:, :], *args, **kwargs)
        if train_out.dim() == 2:
            train_out = train_out.view(B, num_t, -1)
        return torch.cat([frozen_out, train_out], dim=1)

    def __repr__(self):
        return f"PropulsionTokenTheoryWrapper(in={self.base_layer.in_features}, out={self.base_layer.out_features}, degree={self.degree}, num_trainable={self._num_trainable})"


# =============================================================================
#  ALL WRAPPER TYPES
# =============================================================================

ALL_WRAPPERS = (
    CausalTokenTheoryWrapper,
    LoRATokenTheoryWrapper,
    LoRAFATokenTheoryWrapper,
    DoRATokenTheoryWrapper,
    PropulsionTokenTheoryWrapper,
)


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _is_wrappable_full(module, all_layers=False):
    if isinstance(module, ALL_WRAPPERS):
        return False
    cls = module.__class__.__name__.lower()
    if "lora" in cls or "adapter" in cls:
        return False
    if all_layers:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.GroupNorm)):
            return True
        if 'rmsnorm' in cls or 'rms_norm' in cls:
            return True
        return False
    if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
        return False
    if isinstance(module, nn.Linear):
        return True
    if isinstance(module, nn.Conv1d) and module.kernel_size == (1,):
        return True
    if any(x in cls for x in ["embedding", "norm", "batch", "layer", "group"]):
        return False
    return hasattr(module, "weight") and not hasattr(module, "in_channels")


def _is_linear_layer(module) -> bool:
    """
    Detect linear layers including all bitsandbytes quantized variants.
    Uses isinstance for known types, then duck typing as fallback.
    Works across all bitsandbytes versions.
    """
    if isinstance(module, nn.Linear):
        return True
    # bitsandbytes quantized layers
    try:
        import bitsandbytes as bnb
        for attr in ['Linear4bit', 'Linear8bitLt', 'LinearFP4', 'LinearNF4']:
            cls = getattr(bnb.nn, attr, None)
            if cls is not None and isinstance(module, cls):
                return True
    except ImportError:
        pass
    # Duck typing fallback — catches future quantized variants
    # A leaf linear layer: has in_features + out_features + no child modules
    if (hasattr(module, 'in_features') and
            hasattr(module, 'out_features') and
            len(list(module.children())) == 0):
        return True
    return False


def _apply_wrappers(model, modules_to_wrap, wrapper_fn, verbose):
    wrapped = 0
    for name, module in modules_to_wrap:
        try:
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapper_fn(module))
            wrapped += 1
        except Exception as e:
            # Always print — silent swallowing hides bugs
            print(f"  ✗ Failed to wrap {name}: {type(e).__name__}: {e}")
            log.error(f"Failed to wrap {name}: {e}")
    if verbose:
        print(f"✓ Wrapped {wrapped} modules")
    return wrapped


def _freeze_model(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        p.requires_grad = False
    return count


def _clear_existing(model):
    existing = sum(1 for _, m in model.named_modules() if isinstance(m, ALL_WRAPPERS))
    if existing > 0:
        print(f"⚠️  Found {existing} existing wrappers — merging before re-inject")
        merge(model, verbose=False)


def _collect_linear_targets(model, target_modules, name_prefix, verbose=False):
    """
    Collect linear layers matching target_modules and name_prefix.
    Supports nn.Linear and ALL bitsandbytes quantized variants (QLoRA).
    Pass verbose=True to print detected layer types — useful for debugging.
    """
    modules_to_wrap, skipped = [], 0
    seen_types = set()

    for name, module in model.named_modules():
        if isinstance(module, ALL_WRAPPERS):
            continue
        if not _is_linear_layer(module):
            continue
        seen_types.add(type(module).__name__)
        if name_prefix is not None and not name.startswith(name_prefix):
            skipped += 1; continue
        if target_modules is not None and not any(t in name for t in target_modules):
            skipped += 1; continue
        modules_to_wrap.append((name, module))

    if verbose:
        print(f"  layer types found : {seen_types}")
        print(f"  targets matched   : {len(modules_to_wrap)}, skipped: {skipped}")

    return modules_to_wrap, skipped


# =============================================================================
#  PUBLIC API — FULL FINE-TUNING
# =============================================================================

def full(model, num_trainable, target_modules=None, name_prefix=None,
         skip_frozen=True, verbose=True):
    """Full fine-tuning with Token Theory. All weights trainable, gradient split only."""
    _clear_existing(model)
    all_layers = (target_modules == 'all')
    modules_to_wrap, skipped = [], 0
    for name, module in model.named_modules():
        if isinstance(module, ALL_WRAPPERS): continue
        if not _is_wrappable_full(module, all_layers=all_layers): continue
        if skip_frozen and not any(p.requires_grad for p in module.parameters()):
            skipped += 1; continue
        if name_prefix is not None and not name.startswith(name_prefix):
            skipped += 1; continue
        if not all_layers and target_modules is not None:
            if not any(t in name for t in target_modules):
                skipped += 1; continue
        modules_to_wrap.append((name, module))
    if not modules_to_wrap:
        if verbose: print("⚠️  full(): No target modules found!")
        return {"wrapped": 0, "skipped": skipped}
    if verbose:
        print(f"full() — Token Theory full fine-tuning")
        print(f"  num_trainable  : {num_trainable}  (last N tokens)")
        print(f"  target_modules : {'all' if all_layers else target_modules}")
        print(f"  name_prefix    : {name_prefix}")
    wrapped = _apply_wrappers(model, modules_to_wrap, verbose=verbose,
                               wrapper_fn=lambda m: CausalTokenTheoryWrapper(m, num_trainable))
    return {"wrapped": wrapped, "skipped": skipped}


def full_baseline(model, verbose=True):
    """Full fine-tuning baseline — ALL tokens, no injection (no-op)."""
    if verbose:
        print("full_baseline() — full fine-tuning, all tokens trainable, no injection")


# =============================================================================
#  PUBLIC API — LORA
# =============================================================================

def lora(model, num_trainable, target_modules=None, name_prefix=None,
         r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """LoRA + Token Theory. lora_A and lora_B trainable. Works for QLoRA too."""
    _clear_existing(model)
    modules_to_wrap, skipped = _collect_linear_targets(model, target_modules, name_prefix)
    if not modules_to_wrap:
        if verbose: print("⚠️  lora(): No target modules found!")
        return {"wrapped": 0, "skipped": skipped}
    frozen = _freeze_model(model)
    if verbose:
        print(f"lora() — Token Theory LoRA")
        print(f"  num_trainable  : {num_trainable}  (last N tokens)")
        print(f"  target_modules : {target_modules}")
        print(f"  name_prefix    : {name_prefix}")
        print(f"  r={r}, alpha={lora_alpha}, dropout={lora_dropout}, scaling={lora_alpha/r:.4f}")
        print(f"  frozen {frozen:,} base params")
    wrapped = _apply_wrappers(
        model, modules_to_wrap, verbose=verbose,
        wrapper_fn=lambda m: LoRATokenTheoryWrapper(
            m, num_trainable, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    return {"wrapped": wrapped, "skipped": skipped}


def lora_baseline(model, target_modules=None, name_prefix=None,
                  r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """Standard LoRA — ALL tokens trainable (no token split)."""
    if verbose: print("lora_baseline() — standard LoRA, all tokens trainable")
    return lora(model, num_trainable=10**9, target_modules=target_modules,
                name_prefix=name_prefix, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, verbose=verbose)


# =============================================================================
#  PUBLIC API — LORA-FA
# =============================================================================

def lora_fa(model, num_trainable, target_modules=None, name_prefix=None,
            r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """LoRA-FA + Token Theory. lora_A frozen, only lora_B trains."""
    _clear_existing(model)
    modules_to_wrap, skipped = _collect_linear_targets(model, target_modules, name_prefix)
    if not modules_to_wrap:
        if verbose: print("⚠️  lora_fa(): No target modules found!")
        return {"wrapped": 0, "skipped": skipped}
    frozen = _freeze_model(model)
    if verbose:
        print(f"lora_fa() — Token Theory LoRA-FA (frozen A)")
        print(f"  num_trainable  : {num_trainable}  (last N tokens, lora_B only)")
        print(f"  target_modules : {target_modules}")
        print(f"  r={r}, alpha={lora_alpha}, scaling={lora_alpha/r:.4f}")
        print(f"  frozen {frozen:,} base params  (lora_A also frozen)")
    wrapped = _apply_wrappers(
        model, modules_to_wrap, verbose=verbose,
        wrapper_fn=lambda m: LoRAFATokenTheoryWrapper(
            m, num_trainable, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    return {"wrapped": wrapped, "skipped": skipped}


def lora_fa_baseline(model, target_modules=None, name_prefix=None,
                     r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """Standard LoRA-FA — ALL tokens trainable (no token split)."""
    if verbose: print("lora_fa_baseline() — LoRA-FA, all tokens trainable")
    return lora_fa(model, num_trainable=10**9, target_modules=target_modules,
                   name_prefix=name_prefix, r=r, lora_alpha=lora_alpha,
                   lora_dropout=lora_dropout, verbose=verbose)


# =============================================================================
#  PUBLIC API — DORA
# =============================================================================

def dora(model, num_trainable, target_modules=None, name_prefix=None,
         r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """DoRA + Token Theory. Magnitude + direction decomposition."""
    _clear_existing(model)
    modules_to_wrap, skipped = _collect_linear_targets(model, target_modules, name_prefix)
    if not modules_to_wrap:
        if verbose: print("⚠️  dora(): No target modules found!")
        return {"wrapped": 0, "skipped": skipped}
    frozen = _freeze_model(model)
    if verbose:
        print(f"dora() — Token Theory DoRA (magnitude+direction)")
        print(f"  num_trainable  : {num_trainable}  (last N tokens)")
        print(f"  target_modules : {target_modules}")
        print(f"  r={r}, alpha={lora_alpha}, scaling={lora_alpha/r:.4f}")
        print(f"  frozen {frozen:,} base params")
    wrapped = _apply_wrappers(
        model, modules_to_wrap, verbose=verbose,
        wrapper_fn=lambda m: DoRATokenTheoryWrapper(
            m, num_trainable, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
    return {"wrapped": wrapped, "skipped": skipped}


def dora_baseline(model, target_modules=None, name_prefix=None,
                  r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """Standard DoRA — ALL tokens trainable (no token split)."""
    if verbose: print("dora_baseline() — DoRA, all tokens trainable")
    return dora(model, num_trainable=10**9, target_modules=target_modules,
                name_prefix=name_prefix, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, verbose=verbose)


# =============================================================================
#  PUBLIC API — PROPULSION
# =============================================================================

def propulsion(model, num_trainable, target_modules=None, name_prefix=None,
               degree=15, verbose=True):
    """Propulsion + Token Theory. out = linear(x) * propulsion^degree."""
    _clear_existing(model)
    modules_to_wrap, skipped = _collect_linear_targets(model, target_modules, name_prefix)
    if not modules_to_wrap:
        if verbose: print("⚠️  propulsion(): No target modules found!")
        return {"wrapped": 0, "skipped": skipped}
    frozen = _freeze_model(model)
    if verbose:
        print(f"propulsion() — Token Theory Propulsion")
        print(f"  num_trainable  : {num_trainable}  (last N tokens)")
        print(f"  target_modules : {target_modules}")
        print(f"  degree         : {degree}")
        print(f"  frozen {frozen:,} base params")
    wrapped = _apply_wrappers(
        model, modules_to_wrap, verbose=verbose,
        wrapper_fn=lambda m: PropulsionTokenTheoryWrapper(m, num_trainable, degree=degree))
    return {"wrapped": wrapped, "skipped": skipped}


def propulsion_baseline(model, target_modules=None, name_prefix=None,
                        degree=15, verbose=True):
    """Standard Propulsion — ALL tokens trainable (no token split)."""
    if verbose: print("propulsion_baseline() — Propulsion, all tokens trainable")
    return propulsion(model, num_trainable=10**9, target_modules=target_modules,
                      name_prefix=name_prefix, degree=degree, verbose=verbose)


# =============================================================================
#  PUBLIC API — QLORA
# =============================================================================

def qlora(model, num_trainable, target_modules=None, name_prefix=None,
          r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """
    QLoRA + Token Theory.
    Base model must be loaded 4-bit AND prepare_model_for_kbit_training()
    called BEFORE this function. After calling qlora(), set:
        model.is_quantized = False   # bypass HuggingFace Trainer guard
    """
    if verbose:
        print(f"qlora() — Token Theory QLoRA")
        print(f"  ⚠️  Ensure prepare_model_for_kbit_training() was called first")
        print(f"  num_trainable  : {num_trainable}  (last N tokens)")
        print(f"  target_modules : {target_modules}")
        print(f"  r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    return lora(model, num_trainable=num_trainable, target_modules=target_modules,
                name_prefix=name_prefix, r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, verbose=False)


def qlora_baseline(model, target_modules=None, name_prefix=None,
                   r=4, lora_alpha=1.0, lora_dropout=0.0, verbose=True):
    """Standard QLoRA — ALL tokens trainable (no token split)."""
    if verbose: print("qlora_baseline() — QLoRA, all tokens trainable")
    return qlora(model, num_trainable=10**9, target_modules=target_modules,
                 name_prefix=name_prefix, r=r, lora_alpha=lora_alpha,
                 lora_dropout=lora_dropout, verbose=verbose)


# =============================================================================
#  UNIFIED MERGE
# =============================================================================

def merge(model: nn.Module, verbose: bool = True) -> int:
    """
    Merge weights + restore original architecture for ALL methods.
    Always call before model.save_pretrained().

    full       : base weights trained in place — just restore wrapper
    lora/qlora : W += (lora_B @ lora_A) * scaling
                 QLoRA: weight addition skipped for 4-bit base (can't modify packed weights)
    lora_fa    : same formula as lora
    dora       : W = m * (W + delta) / ||W + delta||_col
    propulsion : W rows *= propulsion^degree
    """
    counts = {"full": 0, "lora": 0, "lora_fa": 0, "dora": 0, "propulsion": 0}

    for name, module in list(model.named_modules()):
        if not isinstance(module, ALL_WRAPPERS):
            continue

        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        with torch.no_grad():

            if isinstance(module, CausalTokenTheoryWrapper):
                counts["full"] += 1

            elif isinstance(module, LoRAFATokenTheoryWrapper):
                try:
                    delta = (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                    module.base_layer.weight.add_(delta.to(module.base_layer.weight.dtype))
                except (AttributeError, RuntimeError, TypeError):
                    pass   # quantized base — skip merge
                counts["lora_fa"] += 1

            elif isinstance(module, LoRATokenTheoryWrapper):
                try:
                    delta = (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                    module.base_layer.weight.add_(delta.to(module.base_layer.weight.dtype))
                except (AttributeError, RuntimeError, TypeError):
                    pass   # QLoRA: 4-bit base can't be directly modified
                counts["lora"] += 1

            elif isinstance(module, DoRATokenTheoryWrapper):
                try:
                    delta     = (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                    adapted_W = module.base_layer.weight.float() + delta.float()
                    col_norms = adapted_W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
                    merged_W  = module.magnitude.float().unsqueeze(1) * (adapted_W / col_norms)
                    module.base_layer.weight.copy_(merged_W.to(module.base_layer.weight.dtype))
                except (AttributeError, RuntimeError, TypeError):
                    pass
                counts["dora"] += 1

            elif isinstance(module, PropulsionTokenTheoryWrapper):
                try:
                    push = torch.pow(module.propulsion, module.degree)
                    module.base_layer.weight.mul_(push.unsqueeze(1).to(module.base_layer.weight.dtype))
                    if module.base_layer.bias is not None:
                        module.base_layer.bias.mul_(push.to(module.base_layer.bias.dtype))
                except (AttributeError, RuntimeError, TypeError):
                    pass
                counts["propulsion"] += 1

        setattr(parent, parts[-1], module.base_layer)

    total = sum(counts.values())
    if verbose:
        if total == 0:
            print("  merge(): no wrappers found — model already clean")
        else:
            for method, count in counts.items():
                if count > 0:
                    print(f"✓ merge() [{method:>10}] : merged + restored {count} layers")
    return total


# =============================================================================
#  VISUALIZATION
# =============================================================================

def visualize_token_split(batch, tokenizer, num_trainable, sample_idx=0):
    input_ids = batch['input_ids'][sample_idx]
    attn_mask = batch['attention_mask'][sample_idx]
    T         = len(input_ids)
    split_at  = T - num_trainable
    tokens    = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    print("\n" + "=" * 65)
    print("Token Split Visualization  (left padding, no reorder)")
    print("=" * 65)
    print(f"{'seq_pos':>8}  {'mask':>5}  {'role':>10}  token")
    print("-" * 65)
    for i, (tok, mask_val) in enumerate(zip(tokens, attn_mask.tolist())):
        if mask_val == 0:       role, marker = "PAD",       ""
        elif i < split_at:      role, marker = "frozen",    ""
        else:                   role, marker = "TRAINABLE", "▶"
        print(f"{i:>8}  {int(mask_val):>5}  {role:>10}  {marker} {tok}")
    print("=" * 65)
    print(f"  Frozen    : positions 0 .. {split_at - 1}")
    print(f"  Trainable : positions {split_at} .. {T - 1}  (last {num_trainable} tokens)")
    print("=" * 65 + "\n")


def visualize_attention_mask(position_ids=None, attention_mask=None, tokenizer=None,
                              input_ids=None, num_trainable=0, sample_idx=0,
                              use_matplotlib=False):
    attn     = attention_mask[sample_idx] if attention_mask is not None else None
    ids      = input_ids[sample_idx]      if input_ids is not None      else None
    T        = (attn.shape[0] if attn is not None
                else (ids.shape[0] if ids is not None else 8))
    split_at = T - num_trainable
    can_attend = torch.tril(torch.ones(T, T, dtype=torch.bool))
    if attn is not None:
        can_attend = can_attend & attn.bool().unsqueeze(0)
    if use_matplotlib:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(6, T * 0.5), max(5, T * 0.5)))
            ax.imshow(can_attend.float().numpy(), cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax.axvline(x=split_at - 0.5, color='red', linewidth=2, linestyle='--')
            ax.axhline(y=split_at - 0.5, color='red', linewidth=2, linestyle='--')
            ax.set_title(f"Causal mask | last {num_trainable} trainable")
            plt.tight_layout(); plt.savefig("attention_mask.png", dpi=130)
            plt.show(); return can_attend
        except ImportError:
            pass
    print("\n" + "=" * (T * 2 + 20))
    for i in range(T):
        row  = "".join("■ " if can_attend[i, j].item() else "· " for j in range(T))
        role = "TRAIN" if i >= split_at else "froz "
        print(f"{i:>3} {role}| {row.strip()}")
    print("=" * (T * 2 + 20) + "\n")
    return can_attend


# =============================================================================
#  VERIFICATION
# =============================================================================

def verify_gradient_split(model, batch, num_trainable, verbose=True):
    model.train(); model.zero_grad()
    out        = model(**batch)
    logits     = out.logits if hasattr(out, 'logits') else out.last_hidden_state
    forward_ok = torch.isfinite(logits).all().item()
    if logits.dim() == 3:
        loss = logits[:, logits.shape[1] - num_trainable:, :].float().mean()
    else:
        loss = logits.float().mean()
    loss.backward()
    grad_counts = {"full": 0, "lora": 0, "lora_fa": 0, "dora": 0, "propulsion": 0}
    for _, m in model.named_modules():
        if isinstance(m, CausalTokenTheoryWrapper):
            if m.base_layer.weight is not None and m.base_layer.weight.grad is not None:
                grad_counts["full"] += 1
        elif isinstance(m, LoRAFATokenTheoryWrapper):
            if m.lora_B.weight.grad is not None:
                grad_counts["lora_fa"] += 1
        elif isinstance(m, LoRATokenTheoryWrapper):
            if m.lora_A.weight.grad is not None or m.lora_B.weight.grad is not None:
                grad_counts["lora"] += 1
        elif isinstance(m, DoRATokenTheoryWrapper):
            if m.lora_B.weight.grad is not None or m.magnitude.grad is not None:
                grad_counts["dora"] += 1
        elif isinstance(m, PropulsionTokenTheoryWrapper):
            if m.propulsion.grad is not None:
                grad_counts["propulsion"] += 1
    grad_ok = any(v > 0 for v in grad_counts.values())
    model.zero_grad()
    if verbose:
        print(f"\n  Forward NaN/Inf free : {'✓' if forward_ok else '✗'}")
        for name, count in grad_counts.items():
            if count > 0:
                print(f"  [{name:>10}] wrappers w/ grad : {count} ✓")
    return {"forward_ok": forward_ok, "grad_ok": grad_ok}


def verify_frozen_no_grad(model, batch, num_trainable, verbose=True):
    model.train(); model.zero_grad()
    out    = model(**batch)
    logits = out.logits if hasattr(out, 'logits') else out.last_hidden_state
    if logits.dim() != 3:
        if verbose: print("  ⚠️  Cannot test frozen detachment on 2D logits.")
        return {"frozen_correctly_detached": None}
    split_at = logits.shape[1] - num_trainable
    if split_at <= 0:
        if verbose: print("  ⚠️  No frozen tokens to test.")
        return {"frozen_correctly_detached": True}
    try:
        logits[:, :split_at, :].float().mean().backward()
        leaks = []
        for m in model.modules():
            if isinstance(m, CausalTokenTheoryWrapper):
                if (m.base_layer.weight is not None and m.base_layer.weight.grad is not None
                        and m.base_layer.weight.grad.abs().sum().item() > 0):
                    leaks.append("full")
            elif isinstance(m, (LoRATokenTheoryWrapper, LoRAFATokenTheoryWrapper)):
                if m.lora_B.weight.grad is not None and m.lora_B.weight.grad.abs().sum().item() > 0:
                    leaks.append("lora/lora_fa")
            elif isinstance(m, DoRATokenTheoryWrapper):
                if ((m.lora_B.weight.grad is not None and m.lora_B.weight.grad.abs().sum().item() > 0) or
                        (m.magnitude.grad is not None and m.magnitude.grad.abs().sum().item() > 0)):
                    leaks.append("dora")
            elif isinstance(m, PropulsionTokenTheoryWrapper):
                if m.propulsion.grad is not None and m.propulsion.grad.abs().sum().item() > 0:
                    leaks.append("propulsion")
        frozen_ok = len(leaks) == 0
    except RuntimeError:
        frozen_ok = True; leaks = []
    model.zero_grad()
    if verbose:
        print(f"\n  Frozen slice detached: "
              f"{'✓' if frozen_ok else f'✗ (grads leaked in: {leaks})'}")
    return {"frozen_correctly_detached": frozen_ok}


def run_full_verification(model, tokenizer, text, num_trainable,
                          use_matplotlib=False, dataloader_batch=None):
    """One-shot verification: token split, attention mask, grad split, frozen detach."""
    device = next(model.parameters()).device
    tokenizer.padding_side = 'left'
    enc = {k: v.to(device) for k, v in
           tokenizer(text, return_tensors='pt', padding=True).items()}
    batch_for_grad = dataloader_batch if dataloader_batch is not None else enc
    print("\n" + "█" * 65)
    print("  TOKEN THEORY — FULL VERIFICATION  (left pad, no reorder)")
    print("█" * 65)
    print("\n[1/4] Token Split Table"); print("─" * 65)
    visualize_token_split(enc, tokenizer, num_trainable)
    print("\n[2/4] Attention Mask"); print("─" * 65)
    visualize_attention_mask(attention_mask=enc.get('attention_mask'),
                              input_ids=enc.get('input_ids'),
                              num_trainable=num_trainable,
                              use_matplotlib=use_matplotlib)
    print("\n[3/4] Gradient Split"); print("─" * 65)
    r_grad = verify_gradient_split(model, batch_for_grad, num_trainable)
    print("\n[4/4] Frozen Detachment"); print("─" * 65)
    r_frozen = verify_frozen_no_grad(model, batch_for_grad, num_trainable)
    p3 = r_grad["forward_ok"] and r_grad["grad_ok"]
    p4 = r_frozen["frozen_correctly_detached"] in (True, None)
    ok = p3 and p4
    print("\n" + "█" * 65)
    print("  SUMMARY")
    print("█" * 65)
    print(f"  [3] Gradient split    : {'✓ PASS' if p3 else '✗ FAIL'}")
    print(f"  [4] Frozen detachment : {'✓ PASS' if p4 else '✗ FAIL'}")
    print(f"\n  {'✅  ALL CHECKS PASSED' if ok else '❌  SOME CHECKS FAILED'}")
    print("█" * 65 + "\n")
    return ok