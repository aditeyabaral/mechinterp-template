"""Capture the residual-stream activations at the output of one or more transformer blocks.

Background (the mechinterp idea):
    A decoder-only transformer processes tokens through a stack of identical "blocks"
    (layers). Each block reads from and writes back to the "residual stream" -- the
    running hidden-state vector of shape [batch, seq_len, hidden_size] that flows from
    the embedding, through every block, to the final unembedding. The output of block i
    *is* the residual stream after layer i. Capturing it lets us inspect what the model
    "knows" at each depth.

How we capture it (PyTorch forward hooks):
    A forward hook is a function you attach to an nn.Module; PyTorch calls it every time
    that module runs a forward pass, handing you the module's inputs and output. We attach
    one to each transformer block and simply copy its output tensor into a list. Hooks are
    non-invasive: they observe the computation without changing it (this class never returns
    a modified tensor). Always remove hooks when done (remove_all_hooks) so they don't fire
    on later, unrelated forward passes.

This is generic infrastructure -- you normally will NOT need to edit this file.
"""

import torch
import torch.nn as nn


class MultiLayerActivationStore:
    """Collects residual-stream activations from a set of transformer blocks via forward hooks.

    Activations are stored as layer_idx -> list[torch.Tensor]. The list has one entry per
    forward pass the module saw. During generation the model is called once for the full
    prompt and then once per generated token, so the list looks like:
        index 0  -> the prompt pass,   tensor shape [batch, prompt_len, hidden_size]
        index 1+ -> each new token,    tensor shape [batch, 1, hidden_size]
    """

    def __init__(self, named_layers: dict[int, nn.Module]) -> None:
        """Register a capture hook on every layer in `named_layers`.

        Args:
            named_layers (dict[int, nn.Module]): Maps a layer index to the block module to hook.
        """
        # activations[layer_idx] accumulates the tensors seen at that layer (one per forward pass).
        self.activations: dict[int, list[torch.Tensor]] = {}
        # handles[layer_idx] is the object returned when registering a hook; call .remove() to detach it.
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, layer_module: nn.Module) -> None:
        """Attach a forward hook to a single block so its output is recorded.

        Args:
            layer_idx (int): Index of the layer (used as the dict key).
            layer_module (nn.Module): The block module to hook.
        """

        def _hook_fn(
            module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor, layer_idx: int = layer_idx
        ) -> None:
            """Runs after `layer_module`'s forward pass; appends a copy of its output.

            PyTorch passes the module, its input, and its output. `layer_idx` is bound as a
            default argument so each layer's hook records into the correct dict entry (a
            common Python gotcha: without this, every closure would share the last layer_idx).
            """
            # A transformer block returns a tuple (hidden_states, optional extras); we want
            # the hidden states, which is the residual stream.
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]  # Ignore auxiliary outputs (e.g. attention weights, cache)
            if layer_idx not in self.activations:
                self.activations[layer_idx] = list()
            # .detach(): drop autograd tracking (we only inspect, never backprop).
            # .clone():  copy so a later in-place op can't corrupt our saved value.
            # .cpu():    move off the GPU so we don't pin VRAM across many prompts.
            self.activations[layer_idx].append(output_tensor.detach().clone().cpu())

        # register_forward_hook returns a handle; keep it so we can remove the hook later.
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
