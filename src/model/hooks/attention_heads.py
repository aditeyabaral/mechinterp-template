"""Hooks for extracting and ablating individual attention heads.

Background (what an attention head is):
    A transformer block's self-attention sub-layer is split into `num_heads` independent
    "heads", each working in its own `head_dim`-sized subspace (num_heads * head_dim =
    hidden_size). Each head decides which earlier positions to read from and writes a
    `head_dim` vector. Just before the attention output projection (GPT-2: attn.c_proj),
    all heads' outputs are concatenated into one [.., num_heads * head_dim] vector.
    Heads are a key unit of mechinterp analysis (e.g. "induction heads").

    To inspect heads individually we grab the input to c_proj and reshape that flat
    concatenation back into a [.., num_heads, head_dim] tensor, separating the heads.

This is generic infrastructure -- you normally will NOT need to edit this file.
"""

import torch
import torch.nn as nn


class AttentionHeadExtractionStore:
    """Captures each attention head's output (the input to attn.c_proj) per layer.

    Stores layer_idx -> list[tensor], one tensor per forward pass, reshaped to
    [batch, seq_len, num_heads, head_dim] so heads are separated along their own axis.
    """

    def __init__(
        self,
        named_c_proj_layers: dict[int, nn.Module],
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Register a capture pre-hook on each layer's attention output-projection module.

        Args:
            named_c_proj_layers: dict mapping layer index to the layer's attention output-projection
                module (GPT-2: attn.c_proj). Pass {} to register nothing (a no-op store).
            num_heads: Number of attention heads (needed to un-concatenate the heads).
            head_dim: Dimension of each attention head (hidden_size // num_heads).
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_c_proj_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, c_proj: nn.Module) -> None:
        # Pre-hook: args[0] is the concatenated head outputs about to enter c_proj.
        def _pre_hook_fn(module: nn.Module, args: tuple, layer_idx: int = layer_idx) -> tuple:
            concat_heads = args[0]  # [batch, seq_len, num_heads * head_dim]
            batch, seq_len, _ = concat_heads.shape
            # Split the flat last dim back into (num_heads, head_dim) so we can index a head.
            per_head = concat_heads.reshape(batch, seq_len, self.num_heads, self.head_dim)
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(per_head.detach().clone().cpu())
            return args  # unchanged -> observe only

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
    """Ablates ("knocks out") whole attention heads by zeroing their outputs before c_proj.

    The intervention counterpart to the extraction store above: it deletes the chosen heads'
    contributions so you can measure how much the model's output depends on them. All
    head_dim dimensions of each selected head are set to zero. Register this BEFORE an
    extraction store on the same module if you want the store to see the ablated values.
    """

    def __init__(
        self,
        c_proj: nn.Module,
        head_indices: list[int],
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Register the ablation hook on one layer's attention output-projection.

        Args:
            c_proj: The attention output-projection module (GPT-2: attn.c_proj) to hook into.
            head_indices: Indices of the heads to zero out.
            num_heads: Total number of attention heads.
            head_dim: Dimension of each head.
        """
        self.head_indices = head_indices
        self.num_heads = num_heads
        self.head_dim = head_dim

        def _pre_hook_fn(module: nn.Module, args: tuple) -> tuple:
            # Same reshape trick as the store, but here we edit and feed the result back in.
            concat_heads = args[0].clone()  # [batch, seq_len, num_heads * head_dim]
            batch, seq_len, hidden = concat_heads.shape
            per_head = concat_heads.reshape(batch, seq_len, num_heads, head_dim)
            per_head[:, :, head_indices, :] = 0.0  # zero every dim of the selected heads
            # Flatten the heads back together so c_proj receives its expected shape.
            new_concat = per_head.reshape(batch, seq_len, hidden)
            return (new_concat,) + args[1:]

        self.handle = c_proj.register_forward_pre_hook(_pre_hook_fn)

    def remove(self) -> None:
        """Remove the hook."""
        self.handle.remove()

    def clear(self) -> None:
        """No-op; present for interface consistency."""
        pass
