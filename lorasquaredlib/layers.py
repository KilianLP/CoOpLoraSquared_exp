import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


ExpertSelector = Optional[Union[int, Sequence[int], Iterable[int], torch.Tensor]]


class LinearLoRASquared(nn.Linear):
    """
    Linear layer augmented with one shared LoRA branch and a pool of expert LoRA branches.

    The shared branch is always active, while expert branches contribute only when an
    index (or collection of indices) is passed to the forward method.

    Args:
        existing_linear: The pretrained linear layer to augment.
        r_shared: Rank of the shared LoRA adapter. Set to 0 to disable.
        r_expert: Rank of each expert adapter. Set to 0 to disable experts.
        n_experts: Number of expert adapters to instantiate.
        alpha_shared: Scaling factor for shared LoRA (alpha / r convention).
        alpha_expert: Scaling factor for expert LoRA branches.
        dropout_rate: Dropout applied to the input before the low-rank projections.
        fan_in_fan_out: Flag mirroring the LoRA convention for weight orientation.
        freeze_base: If True, keeps the original weight/bias frozen.
    """

    def __init__(
        self,
        existing_linear: nn.Linear,
        r_shared: int,
        r_expert: int,
        n_experts: int,
        alpha_shared: float = 1.0,
        alpha_expert: float = 1.0,
        dropout_rate: float = 0.0,
        fan_in_fan_out: bool = False,
        freeze_base: bool = True,
        enable_router: bool = False,
        router_temperature: float = 1.0,
        router_mode: str = "weighted",
    ) -> None:
        super().__init__(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            bias=existing_linear.bias is not None,
        )

        self.load_state_dict(existing_linear.state_dict())

        self.r_shared = r_shared
        self.r_expert = r_expert
        self.n_experts = n_experts
        self.alpha_shared = alpha_shared
        self.alpha_expert = alpha_expert
        self.fan_in_fan_out = fan_in_fan_out
        self.average_expert_mode = False
        self.router_mode = router_mode
        self.router_temperature = router_temperature
        self.router_enabled = False

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.t()

        self.scaling_shared = alpha_shared / r_shared if r_shared > 0 else 0.0
        self.scaling_expert = alpha_expert / r_expert if r_expert > 0 else 0.0
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        if enable_router and self.n_experts > 0:
            self.router = nn.Linear(self.in_features, self.n_experts, bias=True)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        else:
            self.router = None

        # Shared LoRA branch
        if self.r_shared > 0:
            self.lora_shared_A = nn.Parameter(
                torch.zeros(self.r_shared, self.in_features)
            )
            self.lora_shared_B = nn.Parameter(
                torch.zeros(self.out_features, self.r_shared)
            )
            self._reset_shared_parameters()
        else:
            self.register_parameter("lora_shared_A", None)
            self.register_parameter("lora_shared_B", None)

        # Expert LoRA branches
        if self.r_expert > 0 and self.n_experts > 0:
            self.lora_expert_A = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(self.r_expert, self.in_features))
                    for _ in range(self.n_experts)
                ]
            )
            self.lora_expert_B = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(self.out_features, self.r_expert))
                    for _ in range(self.n_experts)
                ]
            )
            self._reset_expert_parameters()
        else:
            self.lora_expert_A = nn.ParameterList()
            self.lora_expert_B = nn.ParameterList()

    def _reset_shared_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_shared_B)

    def _reset_expert_parameters(self) -> None:
        for param in self.lora_expert_A:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in self.lora_expert_B:
            nn.init.zeros_(param)

    def _drop_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None and self.training:
            return self.dropout(x)
        return x

    def _normalize_indices(
        self,
        expert_index: ExpertSelector,
        batch_shape: Tuple[int, ...],
        total_items: int,
        device: torch.device,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        if expert_index is None:
            return [], None

        per_sample: Optional[torch.Tensor] = None
        if isinstance(expert_index, torch.Tensor):
            flattened = expert_index.reshape(-1)
            if flattened.numel() == 1:
                indices = [int(flattened.item())]
            else:
                per_sample = self._expand_routing(
                    flattened, batch_shape, total_items, device
                )
                self._validate_indices(per_sample.tolist())
                unique_indices = per_sample.unique(sorted=False).tolist()
                return unique_indices, per_sample
        elif isinstance(expert_index, int):
            indices = [expert_index]
        elif isinstance(expert_index, (list, tuple, set)):
            indices = list(expert_index)
        else:
            indices = list(expert_index)

        self._validate_indices(indices)
        return indices, per_sample

    def _expand_routing(
        self,
        flattened: torch.Tensor,
        batch_shape: Tuple[int, ...],
        total_items: int,
        device: torch.device,
    ) -> torch.Tensor:
        flattened = flattened.detach().to(device=device, dtype=torch.long).reshape(-1)
        if flattened.numel() == total_items:
            return flattened

        if not batch_shape:
            raise ValueError("Cannot broadcast expert selection for an empty batch.")

        ndim = len(batch_shape)
        for axis, axis_size in enumerate(batch_shape):
            if flattened.numel() != axis_size:
                continue
            after = int(math.prod(batch_shape[axis + 1 :])) if axis + 1 < ndim else 1
            before = int(math.prod(batch_shape[:axis])) if axis > 0 else 1
            expanded = flattened.repeat_interleave(after)
            if before > 1:
                expanded = expanded.repeat(before)
            if expanded.numel() == total_items:
                return expanded

        raise ValueError(
            "Per-sample expert selection expects either one index per element "
            f"({total_items}) or a size matching one of the batch axes {batch_shape}; "
            f"got {flattened.numel()}."
        )

    def _validate_indices(self, indices: Sequence[int]) -> None:
        for idx in indices:
            if not 0 <= idx < self.n_experts:
                raise IndexError(
                    f"Expert index {idx} is out of range for {self.n_experts} experts."
                )

    def _expert_projection(
        self, dropped: torch.Tensor, idx: int
    ) -> torch.Tensor:
        proj = dropped @ self.lora_expert_A[idx].t()
        proj = proj @ self.lora_expert_B[idx].t()
        return proj * self.scaling_expert

    def set_average_expert_mode(self, enabled: bool) -> None:
        self.average_expert_mode = enabled

    def _apply_shared(self, dropped: torch.Tensor) -> torch.Tensor:
        if self.r_shared == 0:
            return dropped.new_zeros((dropped.shape[0], self.out_features))
        update = dropped @ self.lora_shared_A.t()
        update = update @ self.lora_shared_B.t()
        return update * self.scaling_shared

    def _apply_experts(
        self, dropped: torch.Tensor, indices: Sequence[int]
    ) -> torch.Tensor:
        if self.r_expert == 0 or len(indices) == 0:
            return dropped.new_zeros((dropped.shape[0], self.out_features))
        update = None
        for idx in indices:
            proj = self._expert_projection(dropped, idx)
            update = proj if update is None else update + proj
        if update is None:
            return dropped.new_zeros((dropped.shape[0], self.out_features))
        if self.average_expert_mode and len(indices) > 0:
            update = update / len(indices)
        return update

    def _apply_router_mixture(
        self, dropped: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.r_expert == 0 or self.n_experts == 0:
            return dropped.new_zeros((dropped.shape[0], self.out_features))
        update = dropped.new_zeros((dropped.shape[0], self.out_features))
        for idx in range(self.n_experts):
            coeff = weights[:, idx].unsqueeze(-1)
            if torch.all(coeff == 0):
                continue
            proj = self._expert_projection(dropped, idx)
            update = update + coeff * proj
        return update

    def set_router_state(
        self,
        *,
        enabled: Optional[bool] = None,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if enabled is not None and self.router is not None:
            self.router_enabled = enabled
        if mode is not None:
            self.router_mode = mode
        if temperature is not None:
            self.router_temperature = temperature

    def _apply_per_sample_experts(
        self, dropped: torch.Tensor, routing: torch.Tensor, indices: Sequence[int]
    ) -> torch.Tensor:
        if self.r_expert == 0 or routing.numel() == 0 or len(indices) == 0:
            return dropped.new_zeros((dropped.shape[0], self.out_features))
        update = dropped.new_zeros((dropped.shape[0], self.out_features))
        for idx in indices:
            mask = routing == idx
            if not mask.any():
                continue
            proj = self._expert_projection(dropped[mask], idx)
            update[mask] += proj
        return update

    def forward(
        self, x: torch.Tensor, expert_index: ExpertSelector = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor whose last dimension equals ``in_features``.
            expert_index:
                - ``None``: only the shared LoRA branch is used.
                - ``int`` or iterable of ints: activate the same experts for every
                  element in the batch (their contributions are summed).
                - 1D ``torch.Tensor`` whose length matches either the flattened batch
                  size or any single batch axis: selects one expert per element or
                  per sample axis, respectively.
        """
        result = nn.functional.linear(x, self.weight, self.bias)
        batch_shape = tuple(x.shape[:-1])
        flat_input = x.reshape(-1, x.shape[-1])
        dropped = self._drop_input(flat_input)
        view_shape = batch_shape + (self.out_features,) if batch_shape else (self.out_features,)

        if self.r_shared > 0:
            shared = self._apply_shared(dropped).view(*view_shape)
            result = result + shared

        use_router = self.router is not None and self.router_enabled

        if use_router:
            temp = max(float(self.router_temperature), 1e-5)
            if self.router_mode == "weighted":
                weights = torch.softmax(self.router(flat_input) / temp, dim=-1)
            elif self.router_mode == "gumbel":
                weights = F.gumbel_softmax(
                    self.router(flat_input), tau=temp, hard=False, dim=-1
                )
            elif self.router_mode == "ste":
                weights = F.gumbel_softmax(
                    self.router(flat_input), tau=temp, hard=True, dim=-1
                )
            elif self.router_mode == "ste_softmax":
                logits = self.router(flat_input) / temp
                soft = torch.softmax(logits, dim=-1)
                hard = torch.zeros_like(soft)
                hard.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
                weights = hard + (soft - soft.detach()) 
            else:
                raise ValueError(f"Unknown router_mode '{self.router_mode}'.")

            expert_update = self._apply_router_mixture(dropped, weights)
            result = result + expert_update.view(*view_shape)
        else:
            total_items = flat_input.shape[0]
            indices, routing = self._normalize_indices(
                expert_index,
                batch_shape=batch_shape,
                total_items=total_items,
                device=x.device,
            )
            if routing is not None:
                per_sample_update = self._apply_per_sample_experts(dropped, routing, indices)
                result = result + per_sample_update.view(*view_shape)
            elif indices:
                expert_update = self._apply_experts(dropped, indices)
                result = result + expert_update.view(*view_shape)

        return result

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, r_shared={self.r_shared}, r_expert={self.r_expert}, "
            f"n_experts={self.n_experts}, alpha_shared={self.alpha_shared}, "
            f"alpha_expert={self.alpha_expert}"
        )
