"""Activation Patching Hook for Neural Network Layers."""

import torch
import torch.nn as nn


class ActivationPatchingHook:
    """This class registers a forward hook on a given layer to patch activations during the forward pass."""

    def __init__(
        self,
        layer: nn.Module,
        patch: torch.Tensor = None,
        extraction_indices: list[int] = None,
        patching_indices: list[int] = None,
        alpha: float = 1.0,
    ) -> None:
        """This class registers a forward hook on a given layer to patch activations during the forward pass.

        Args:
            layer (nn.Module): The layer to register the hook on.
            patch (torch.Tensor): The activations to patch into the layer.
            extraction_indices (list[int]): Indices of tokens to extract activations from.
            patching_indices (list[int]): Indices of tokens to patch into.
            alpha (float): The patching coefficient. Defaults to 1.0.
        """
        assert len(extraction_indices) == len(patching_indices), (
            "Extraction and patching indices must have the same length."
        )
        assert 0.0 <= alpha <= 1.0, "Alpha must be between 0 and 1."

        self.patch = patch
        self.extraction_indices = extraction_indices
        self.patching_indices = patching_indices
        self.alpha = alpha
        self.handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> None | torch.Tensor:
        """The hook function that patches the activations of the layer.

        Args:
            module (nn.Module): The layer module.
            input_tensor (torch.Tensor): The input tensor to the layer.
            output_tensor (torch.Tensor): The output tensor from the layer.

        Returns:
            tuple: The patched output tensor.
        """
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]  # [beam size, sequence length, dimension]

        # If there are no activations to patch or the output tensor has only one token
        # simply return the output unmodified.
        if self.patch is None or output_tensor.shape[1] == 1:
            return None

        # The patching process replaces the activations of the specified tokens with the patched activations
        # and keeps the rest of the activations unchanged.
        # alpha is used to blend the activations if needed.
        # 1.0 means full patching, otherwise it combines as alpha * activations + (1-alpha) * original_activations
        final_tensor = output_tensor.clone()
        extraction_indices = torch.tensor(self.extraction_indices, device=output_tensor.device)
        patching_indices = torch.tensor(self.patching_indices, device=output_tensor.device)
        if self.alpha == 1.0:
            # Full patching
            final_tensor[:, patching_indices, :] = self.patch[:, extraction_indices, :]
        else:
            # Blended patching
            final_tensor[:, patching_indices, :] = (
                self.alpha * self.patch[:, extraction_indices, :]
                + (1 - self.alpha) * output_tensor[:, patching_indices, :]
            )

        return final_tensor

    def clear(self) -> None:
        """Clear the activations and tokens to patch."""
        self.patch = None
        self.extraction_indices = None
        self.patching_indices = None

    def remove(self) -> None:
        """Remove the hook from the layer."""
        self.handle.remove()
