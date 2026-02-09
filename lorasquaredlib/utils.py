from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from loralib.utils import INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION

from .layers import LinearLoRASquared


def lorasquared_state_dict(
    model: nn.Module, *, include_router: bool = True
) -> dict[str, torch.Tensor]:
    """
    Collect only LoRA^2 (shared/expert) parameters and optionally router weights.
    """
    state = {}
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("lora_shared", "lora_expert")):
            state[name] = param.detach().cpu()
        elif include_router and "router" in name:
            state[name] = param.detach().cpu()
    return state


def save_lorasquared(
    model: nn.Module,
    path: str,
    *,
    include_router: bool = True,
) -> None:
    """
    Save LoRA^2 adapters (and optionally router) to ``path``.
    """
    torch.save({"state_dict": lorasquared_state_dict(model, include_router=include_router)}, path)


def load_lorasquared(
    model: nn.Module,
    path: str,
    *,
    include_router: bool = True,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Load LoRA^2 adapters (and optionally router) from ``path`` into ``model``.
    Returns (missing_keys, unexpected_keys) from ``load_state_dict`` for transparency.
    """
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("state_dict", checkpoint)
    filtered = {}
    for k, v in state.items():
        if any(tag in k for tag in ("lora_shared", "lora_expert")):
            filtered[k] = v
        elif include_router and "router" in k:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return missing, unexpected


def _cosine_squared_shared_expert(
    A_shared: torch.Tensor,
    B_shared: torch.Tensor,
    A_expert: torch.Tensor,
    B_expert: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute cosine-squared similarity between shared and expert low-rank updates
    using the identity:
        <B_s A_s, B_e A_e>_F = trace(A_s A_e^T B_e^T B_s)
        ||B A||_F^2          = trace(A A^T B^T B)
    Shapes: A_* = (r, d_in), B_* = (d_out, r)
    """
    # r_s x r_e
    cross_A = A_shared @ A_expert.t()
    # r_e x r_s
    cross_B = B_expert.t() @ B_shared
    inner = (cross_A * cross_B.t()).sum()

    gram_A_shared = A_shared @ A_shared.t()      # r_s x r_s
    gram_B_shared = B_shared.t() @ B_shared      # r_s x r_s
    norm_shared_sq = (gram_A_shared * gram_B_shared.t()).sum()

    gram_A_expert = A_expert @ A_expert.t()      # r_e x r_e
    gram_B_expert = B_expert.t() @ B_expert      # r_e x r_e
    norm_expert_sq = (gram_A_expert * gram_B_expert.t()).sum()

    denom = (norm_shared_sq.clamp_min(0).sqrt() *
             norm_expert_sq.clamp_min(0).sqrt() + eps)
    return (inner / denom) ** 2


def shared_expert_orthogonality_loss(
    layers: Optional[Iterable[nn.Module]],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Sum of cosine-squared similarities between the shared LoRA branch and every
    expert branch across all LoRA^2-augmented linear projections.

    Args:
        layers: Iterable returned by ``apply_lorasquared`` (each is a wrapped
                attention module). If None/empty, returns a scalar zero tensor.
        eps:    Numerical stability term added to the denominator.
    """
    if not layers:
        return torch.tensor(0.0, device="cpu")

    total: torch.Tensor | None = None

    def maybe_accumulate(linear: nn.Module):
        nonlocal total
        if not isinstance(linear, LinearLoRASquared):
            return
        if linear.r_shared == 0 or linear.r_expert == 0 or linear.n_experts == 0:
            return

        A_shared = linear.lora_shared_A
        B_shared = linear.lora_shared_B

        for A_expert, B_expert in zip(linear.lora_expert_A, linear.lora_expert_B):
            csq = _cosine_squared_shared_expert(
                A_shared, B_shared, A_expert, B_expert, eps
            )
            total = csq if total is None else total + csq

    for attn in layers:
        for proj in getattr(attn, "q_proj", None), getattr(attn, "k_proj", None), getattr(attn, "v_proj", None), getattr(attn, "proj", None):
            if proj is not None:
                maybe_accumulate(proj)

    if total is None:
        return torch.tensor(0.0, device=layers[0].q_proj.weight.device)
    return total


class PlainMultiheadAttentionLoRASquared(nn.Module):
    """
    Multi-head attention layer enhanced with shared + expert LoRA adapters.

    The interface mirrors ``nn.MultiheadAttention`` while adding an optional
    ``expert_index`` argument to the forward call. When omitted, only the shared
    adapters contribute; passing an expert id (or a collection of ids) activates
    their additional low-rank updates.
    """

    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: Sequence[str] = ("q", "k", "v", "o"),
        *,
        r_shared: int,
        r_expert: int,
        n_experts: int,
        alpha_shared: float = 1.0,
        alpha_expert: float = 1.0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.dropout = 0.0  # this module does not retrain the original dropout
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        self.active_expert = None

        # Reconstruct projections to preserve the exact weights from the original module.
        self.q_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None
        )

        with torch.no_grad():
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = (
                existing_mha.in_proj_bias.data
                if existing_mha.in_proj_bias is not None
                else None
            )

            self.q_proj.weight.data.copy_(existing_weight[: self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[: self.embed_dim])

            self.k_proj.weight.data.copy_(
                existing_weight[self.embed_dim : 2 * self.embed_dim, :]
            )
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(
                    existing_bias[self.embed_dim : 2 * self.embed_dim]
                )

            self.v_proj.weight.data.copy_(existing_weight[2 * self.embed_dim :, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2 * self.embed_dim :])

            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        def maybe_wrap(layer: nn.Linear) -> nn.Module:
            return LinearLoRASquared(
                layer,
                r_shared=r_shared,
                r_expert=r_expert,
                n_experts=n_experts,
                alpha_shared=alpha_shared,
                alpha_expert=alpha_expert,
                dropout_rate=dropout_rate,
                fan_in_fan_out=False,
            )

        self.q_proj = maybe_wrap(self.q_proj) if "q" in enable_lora else self.q_proj
        self.k_proj = maybe_wrap(self.k_proj) if "k" in enable_lora else self.k_proj
        self.v_proj = maybe_wrap(self.v_proj) if "v" in enable_lora else self.v_proj
        self.proj = maybe_wrap(self.proj) if "o" in enable_lora else self.proj

    @staticmethod
    def _project(
        layer: nn.Module, tensor: torch.Tensor, expert_index
    ) -> torch.Tensor:
        if isinstance(layer, LinearLoRASquared):
            return layer(tensor, expert_index=expert_index)
        return layer(tensor)

    def set_active_expert(self, expert_index) -> None:
        if isinstance(expert_index, torch.Tensor):
            self.active_expert = expert_index.detach().cpu()
        else:
            self.active_expert = expert_index

    def set_average_expert_mode(self, enabled: bool) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.proj):
            if isinstance(module, LinearLoRASquared):
                module.set_average_expert_mode(enabled)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        expert_index=None,
    ):
        expert_index = (
            expert_index if expert_index is not None else self.active_expert
        )
        return self.forward_module(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            expert_index=expert_index,
        )

    def forward_module(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
        expert_index=None,
    ):
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = self._project(self.q_proj, query, expert_index)
        k = self._project(self.k_proj, key, expert_index)
        v = self._project(self.v_proj, value, expert_index)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, "
                        f"but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, "
                        f"but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.0

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3)
            .contiguous()
            .view(bsz * tgt_len, embed_dim)
        )
        attn_output = self._project(self.proj, attn_output, expert_index)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def train(self, mode: bool = True) -> "PlainMultiheadAttentionLoRASquared":
        super().train(mode)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.proj):
            module.train(mode)
        return self


def apply_lorasquared(
    clip_model: nn.Module,
    *,
    backbone: str,
    encoder: str = "both",
    position: str = "all",
    params: Sequence[str] = ("q", "k", "v"),
    r_shared: int,
    r_expert: int,
    n_experts: int,
    alpha_shared: float = 1.0,
    alpha_expert: float = 1.0,
    dropout_rate: float = 0.0,
    enable_router: bool = False,
    router_temperature: float = 1.0,
    router_mode: str = "weighted",
    verbose: bool = True,
) -> List[PlainMultiheadAttentionLoRASquared]:
    """
    Inject LoRA-squared adapters into the CLIP transformer blocks.

    Returns:
        A list of the wrapped attention modules for convenience (e.g. toggling experts).
    """
    if encoder not in ("text", "vision", "both"):
        raise ValueError(f"Unknown encoder '{encoder}'.")
    if r_shared <= 0 and r_expert <= 0:
        raise ValueError("At least one of r_shared or r_expert must be > 0.")
    if r_expert > 0 and n_experts <= 0:
        raise ValueError("n_experts must be > 0 when r_expert is positive.")

    wrapped_layers: List[PlainMultiheadAttentionLoRASquared] = []

    if encoder in ("text", "both"):
        indices = INDEX_POSITIONS_TEXT[position]
        text_encoder = clip_model.transformer
        for idx, block in enumerate(text_encoder.resblocks):
            if verbose:
                print(f"[LoRA^2][text] residual block {idx}")
            if idx in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_mha = PlainMultiheadAttentionLoRASquared(
                            submodule,
                            enable_lora=params,
                            r_shared=r_shared,
                            r_expert=r_expert,
                            n_experts=n_experts,
                            alpha_shared=alpha_shared,
                            alpha_expert=alpha_expert,
                            dropout_rate=dropout_rate,
                            enable_router=enable_router,
                            router_temperature=router_temperature,
                            router_mode=router_mode,
                        )
                        setattr(block, name, new_mha)
                        wrapped_layers.append(new_mha)

    if encoder in ("vision", "both"):
        if backbone not in INDEX_POSITIONS_VISION:
            raise KeyError(
                f"Backbone '{backbone}' not recognised in INDEX_POSITIONS_VISION."
            )
        indices = INDEX_POSITIONS_VISION[backbone][position]
        vision_encoder = clip_model.visual.transformer
        for idx, block in enumerate(vision_encoder.resblocks):
            if verbose:
                print(f"[LoRA^2][vision] residual block {idx}")
            if idx in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_mha = PlainMultiheadAttentionLoRASquared(
                            submodule,
                            enable_lora=params,
                            r_shared=r_shared,
                            r_expert=r_expert,
                            n_experts=n_experts,
                            alpha_shared=alpha_shared,
                            alpha_expert=alpha_expert,
                            dropout_rate=dropout_rate,
                            enable_router=enable_router,
                            router_temperature=router_temperature,
                            router_mode=router_mode,
                        )
                        setattr(block, name, new_mha)
                        wrapped_layers.append(new_mha)

    return wrapped_layers


def mark_only_lorasquared_as_trainable(
    model: nn.Module, *, include_bias: bool = False
) -> None:
    """
    Disable gradients for all parameters except the shared/expert LoRA branches.
    """
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("lora_shared", "lora_expert")):
            param.requires_grad = True
        elif include_bias and "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lorasquared_parameters(
    model: nn.Module, *, include_bias: bool = False
) -> List[torch.nn.Parameter]:
    """
    Collect shared/expert LoRA parameters for optimizer construction.
    """
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("lora_shared", "lora_expert")):
            params.append(param)
        elif include_bias and "bias" in name:
            params.append(param)
    return params


def resolve_expert_indices(
    expert_value: Union[str, int, Iterable[int], torch.Tensor, None],
    n_experts: int,
) -> Union[None, int, List[int]]:
    """
    Normalize expert selection inputs to a consistent representation.

    Returns:
        None if no expert should be activated, an ``int`` if a single expert
        is requested, or a list of ints for multiple experts.
    """
    if expert_value is None:
        return None

    if isinstance(expert_value, torch.Tensor):
        if expert_value.numel() == 0:
            return None
        if expert_value.numel() == 1:
            expert_value = int(expert_value.item())
        else:
            raise ValueError(
                "Tensor-based expert selection must contain a single index."
            )

    if isinstance(expert_value, int):
        _validate_indices([expert_value], n_experts)
        return expert_value

    if isinstance(expert_value, (list, tuple, set)):
        indices = [int(idx) for idx in expert_value]
        _validate_indices(indices, n_experts)
        if not indices:
            return None
        return indices if len(indices) > 1 else indices[0]

    if isinstance(expert_value, str):
        value = expert_value.strip()
        if value == "":
            return None
        lowered = value.lower()
        if lowered in ("none", "null"):
            return None
        if lowered == "all":
            indices = list(range(n_experts))
            _validate_indices(indices, n_experts)
            if not indices:
                return None
            return indices if len(indices) > 1 else indices[0]
        parts = [part.strip() for part in value.split(",") if part.strip() != ""]
        if len(parts) > 1:
            indices = [int(part) for part in parts]
            _validate_indices(indices, n_experts)
            return indices
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Unable to parse expert selection from '{expert_value}'."
            ) from exc
        _validate_indices([index], n_experts)
        return index

    index = int(expert_value)
    _validate_indices([index], n_experts)
    return index


def _validate_indices(indices: List[int], n_experts: int) -> None:
    for idx in indices:
        if idx < 0:
            raise ValueError(f"Expert index {idx} must be non-negative.")
        if n_experts > 0 and idx >= n_experts:
            raise ValueError(
                f"Expert index {idx} is out of range for {n_experts} experts."
            )


def set_active_expert_for_layers(
    layers: Optional[Iterable[nn.Module]], expert_index
) -> None:
    if not layers:
        return
    for layer in layers:
        if hasattr(layer, "set_active_expert"):
            layer.set_active_expert(expert_index)


def set_average_expert_mode_for_layers(
    layers: Optional[Iterable[nn.Module]], enabled: bool
) -> None:
    if not layers:
        return
    for layer in layers:
        if hasattr(layer, "set_average_expert_mode"):
            layer.set_average_expert_mode(enabled)
