"""Hooks for extracting and ablating per-head attention outputs."""

import torch
import torch.nn as nn


class AttentionHeadExtractionStore:
    """Captures per-head attention outputs (input to the attention output projection) per layer.

    The concatenated head outputs have shape [B, seq_len, num_heads * head_dim]
    just before the output projection (GPT-2: attn.c_proj).
    We reshape to [B, seq_len, num_heads, head_dim] to expose individual head outputs.

    Registers a forward pre-hook on each layer's attention output-projection module.
    """

    def __init__(
        self,
        named_c_proj_layers: dict[int, nn.Module],
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Initialise the store and register hooks.

        Args:
            named_c_proj_layers: dict mapping layer index to the layer's attention output-projection
                module (GPT-2: attn.c_proj).
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head (hidden_size // num_heads).
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_c_proj_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, c_proj: nn.Module) -> None:
        def _pre_hook_fn(module: nn.Module, args: tuple, layer_idx: int = layer_idx) -> tuple:
            concat_heads = args[0]  # [batch, seq_len, num_heads * head_dim]
            batch, seq_len, _ = concat_heads.shape
            per_head = concat_heads.reshape(batch, seq_len, self.num_heads, self.head_dim)
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(per_head.detach().clone().cpu())
            return args

        handle = c_proj.register_forward_pre_hook(_pre_hook_fn)
        self.handles[layer_idx] = handle

    def get_layer_heads(self, layer_idx: int) -> list[torch.Tensor]:
        """Retrieve stored per-head activation tensors for a layer.

        Each tensor has shape [B, seq_len, num_heads, head_dim].
        """
        act = self.activations.get(layer_idx)
        if act is None:
            raise ValueError(f"No head activations stored for layer: {layer_idx}")
        return act

    def clear(self) -> None:
        """Clear all stored activations."""
        self.activations.clear()

    def remove_all_hooks(self) -> None:
        """Remove all registered forward pre-hooks."""
        for h in self.handles.values():
            h.remove()
        self.handles.clear()

    def __del__(self) -> None:
        """Remove hooks on garbage collection."""
        self.remove_all_hooks()


class AttentionHeadAblationHook:
    """Ablates specific attention heads by zeroing their outputs before the output projection.

    Registers a forward pre-hook on a single layer's attention output-projection
    module (GPT-2: attn.c_proj) that sets the specified head outputs (all head_dim
    dimensions) to 0.0.
    """

    def __init__(
        self,
        c_proj: nn.Module,
        head_indices: list[int],
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Register the ablation hook.

        Args:
            c_proj: The attention output-projection module (GPT-2: attn.c_proj) to hook into.
            head_indices: Indices of heads to zero out.
            num_heads: Total number of attention heads.
            head_dim: Dimension of each head.
        """
        self.head_indices = head_indices
        self.num_heads = num_heads
        self.head_dim = head_dim

        def _pre_hook_fn(module: nn.Module, args: tuple) -> tuple:
            concat_heads = args[0].clone()  # [batch, seq_len, num_heads * head_dim]
            batch, seq_len, hidden = concat_heads.shape
            per_head = concat_heads.reshape(batch, seq_len, num_heads, head_dim)
            per_head[:, :, head_indices, :] = 0.0
            new_concat = per_head.reshape(batch, seq_len, hidden)
            return (new_concat,) + args[1:]

        self.handle = c_proj.register_forward_pre_hook(_pre_hook_fn)

    def remove(self) -> None:
        """Remove the hook."""
        self.handle.remove()

    def clear(self) -> None:
        """No-op; present for interface consistency."""
        pass
