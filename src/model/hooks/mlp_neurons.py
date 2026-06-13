"""Hooks for extracting and ablating MLP intermediate neuron activations."""

import torch
import torch.nn as nn


class MlpNeuronExtractionStore:
    """Captures the intermediate MLP activations (input to down_proj) per layer.

    In LlamaMLP: down_proj(act_fn(gate_proj(x)) * up_proj(x))
    The input to down_proj is the intermediate neuron activation tensor,
    shape [B, seq_len, intermediate_size]. These are the "neurons" we analyse.

    Registers a forward pre-hook on each layer's mlp.down_proj.
    """

    def __init__(self, named_down_proj_layers: dict[int, nn.Module]) -> None:
        """Initialise the store and register hooks.

        Args:
            named_down_proj_layers: dict mapping layer index to the layer's mlp.down_proj module.
        """
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_down_proj_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, down_proj: nn.Module) -> None:
        def _pre_hook_fn(module: nn.Module, args: tuple, layer_idx: int = layer_idx) -> tuple:
            intermediate = args[0]  # [B, seq_len, intermediate_size]
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(intermediate.detach().clone().cpu())
            return args

        handle = down_proj.register_forward_pre_hook(_pre_hook_fn)
        self.handles[layer_idx] = handle

    def get_layer_neurons(self, layer_idx: int) -> list[torch.Tensor]:
        """Retrieve stored intermediate activation tensors for a layer.

        Each tensor has shape [B, seq_len, intermediate_size].
        """
        act = self.activations.get(layer_idx)
        if act is None:
            raise ValueError(f"No neuron activations stored for layer: {layer_idx}")
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


class MlpNeuronAblationHook:
    """Ablates specific MLP neurons by zeroing their intermediate activations.

    Registers a forward pre-hook on a single layer's mlp.down_proj that sets
    the specified neuron indices to 0.0 before the down projection.
    """

    def __init__(self, down_proj: nn.Module, neuron_indices: list[int]) -> None:
        """Register the ablation hook.

        Args:
            down_proj: The mlp.down_proj module to hook into.
            neuron_indices: Indices of neurons to zero out in the intermediate activation.
        """
        self.neuron_indices = neuron_indices

        def _pre_hook_fn(module: nn.Module, args: tuple) -> tuple:
            intermediate = args[0].clone()  # [B, seq_len, intermediate_size]
            intermediate[:, :, neuron_indices] = 0.0
            return (intermediate,) + args[1:]

        self.handle = down_proj.register_forward_pre_hook(_pre_hook_fn)

    def remove(self) -> None:
        """Remove the hook."""
        self.handle.remove()

    def clear(self) -> None:
        """No-op; present for interface consistency."""
        pass
