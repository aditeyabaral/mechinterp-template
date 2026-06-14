"""Hooks for extracting and ablating MLP "neuron" activations.

Background (what an MLP "neuron" is):
    Each transformer block has an MLP (feed-forward) sub-layer. In GPT-2 it computes
    c_proj(act_fn(c_fc(x))): the input x [hidden_size] is projected UP to a wider
    intermediate vector [intermediate_size] (~4x hidden), a non-linearity is applied,
    then projected back DOWN to hidden_size. Each entry of that wide intermediate vector
    is one "neuron". Neurons are a popular unit of mechinterp analysis because individual
    ones sometimes respond to human-interpretable features.

    We capture the neurons by reading the INPUT to the down-projection (c_proj), which is
    exactly the post-activation intermediate vector. We use a forward PRE-hook (fires just
    before the module runs and sees its inputs), rather than a regular forward hook (fires
    after and sees its output).

This is generic infrastructure -- you normally will NOT need to edit this file.
"""

import torch
import torch.nn as nn


class MlpNeuronExtractionStore:
    """Captures the MLP neuron activations (the input to mlp.c_proj) for each hooked layer.

    Stores layer_idx -> list[tensor], one tensor per forward pass, each of shape
    [batch, seq_len, intermediate_size]. (During generation: index 0 is the prompt pass,
    indices 1+ are single generated tokens.)
    """

    def __init__(self, named_c_proj_layers: dict[int, nn.Module]) -> None:
        """Register a capture pre-hook on each layer's MLP down-projection module.

        Args:
            named_c_proj_layers: dict mapping layer index to the layer's MLP neuron-projection module
                (GPT-2: mlp.c_proj). Pass {} to register nothing (a no-op store).
        """
        self.activations: dict[int, list[torch.Tensor]] = {}
        self.handles: dict[int, torch.utils.hooks.RemovableHandle] = {}

        for idx, layer in named_c_proj_layers.items():
            self._register_layer(idx, layer)

    def _register_layer(self, layer_idx: int, c_proj: nn.Module) -> None:
        # A forward PRE-hook receives the positional args the module is about to be called
        # with. For c_proj, args[0] is the neuron activation tensor we want to record.
        def _pre_hook_fn(module: nn.Module, args: tuple, layer_idx: int = layer_idx) -> tuple:
            intermediate = args[0]  # [batch, seq_len, intermediate_size] -- the MLP neurons
            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            # detach/clone/cpu: see extraction.py for why. Returning args unchanged means
            # this hook only observes -- it does not alter what c_proj receives.
            self.activations[layer_idx].append(intermediate.detach().clone().cpu())
            return args

        handle = c_proj.register_forward_pre_hook(_pre_hook_fn)
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
    """Ablates ("knocks out") specific MLP neurons by forcing their activations to zero.

    This is an intervention, not an observation: "ablation" means we delete a component's
    contribution and watch how the model's output changes. If zeroing neuron N noticeably
    changes the answer, that's causal evidence neuron N matters for the behaviour.

    Unlike the extraction store, this pre-hook MODIFIES the tensor: it zeroes the chosen
    neuron indices and returns the edited tensor, so the down-projection sees the ablated
    activations. Register it BEFORE any extraction store on the same module so the store
    records the already-zeroed values (PyTorch fires pre-hooks in registration order).
    """

    def __init__(self, c_proj: nn.Module, neuron_indices: list[int]) -> None:
        """Register the ablation hook on one layer's MLP down-projection.

        Args:
            c_proj: The MLP neuron-projection module (GPT-2: mlp.c_proj) to hook into.
            neuron_indices: Indices (into intermediate_size) of the neurons to zero out.
        """
        self.neuron_indices = neuron_indices

        def _pre_hook_fn(module: nn.Module, args: tuple) -> tuple:
            # .clone() first: the incoming tensor may be needed elsewhere / be a view, so we
            # edit a copy rather than mutating the model's own activation in place.
            intermediate = args[0].clone()  # [batch, seq_len, intermediate_size]
            intermediate[:, :, neuron_indices] = 0.0  # zero the chosen neurons at every position
            # Return the modified args tuple -> c_proj now runs on the ablated activations.
            return (intermediate,) + args[1:]

        self.handle = c_proj.register_forward_pre_hook(_pre_hook_fn)

    def remove(self) -> None:
        """Remove the hook."""
        self.handle.remove()

    def clear(self) -> None:
        """No-op; present for interface consistency."""
        pass
