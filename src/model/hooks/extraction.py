"""Module for extracting and storing activations from multiple transformer layers."""

import torch
import torch.nn as nn


class MultiLayerActivationStore:
    """Collects and stores activations from specified transformer layers.

    Activations are stored as layer_idx -> list[torch.Tensor] where each tensor is [B, seq_len, D]
    """

    def __init__(self, named_layers: dict[int, nn.Module]) -> None:
        """Initialize the MultiLayerActivationStore.

        Args:
            named_layers (dict[int, nn.Module]): A dictionary mapping layer indices to their corresponding modules
        """
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, layer_module: nn.Module) -> None:
        """Internal method to hook into a single layer.

        Args:
            layer_idx (int): Index of the layer.
            layer_module (nn.Module): Module object of the layer.
        """

        def _hook_fn(
            module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor, layer_idx: int = layer_idx
        ) -> None:
            """Hook function to store activations.

            Args:
                module (nn.Module): The layer module.
                input_tensor (torch.Tensor): The input tensor to the layer.
                output_tensor (torch.Tensor): The output tensor from the layer.
                layer_idx (int): Index of the layer.
            """
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]  # Ignore auxiliary outputs
            if layer_idx not in self.activations:
                # Initialize list for this layer
                self.activations[layer_idx] = list()
            self.activations[layer_idx].append(output_tensor.detach().clone().cpu())

        handle = layer_module.register_forward_hook(_hook_fn)
        self.handles[layer_idx] = handle

    def get_layer_activations(self, layer_idx: int) -> list[torch.Tensor]:
        """Retrieve the full activation tensors [B, T, D] for a layer."""
        act = self.activations.get(layer_idx)
        if act is None:
            raise ValueError(f"No activations stored for layer: {layer_idx}")
        return act

    def clear(self) -> None:
        """Clear all stored activations."""
        self.activations.clear()

    def remove_all_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self.handles.values():
            h.remove()
        self.handles.clear()

    def __del__(self) -> None:
        """Destructor to ensure all hooks are removed."""
        self.remove_all_hooks()
