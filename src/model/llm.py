"""Module defining a generic Large Language Model (LLM) class with methods for generation and intervention."""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hooks import MultiLayerActivationStore


class LargeLanguageModel:
    """A generic Large-Language Model (LLM) class to serve as the base class for different models."""

    PROMPT_TEMPLATE_SMALL_LM = "{}\n{}="

    def __init__(
        self,
        model_path: str,
        bnb_config: BitsAndBytesConfig | None = None,
        device: torch.device = None,
    ) -> None:
        """Initialize the LargeLanguageModel class."""
        self.model_path = model_path
        self.bnb_config = bnb_config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            attn_implementation="eager",
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def get_layers(self, layers: list[int] | None) -> dict[int, nn.Module]:
        """Get the layers to extract and patch activations into.

        Args:
            layers (Optional[list[int]]): list of layer indices to extract and patch. If None, all layers are returned.

        Returns:
            dict[int, nn.Module]: A dictionary of layer idx: layer module pairs for the specified layers.
        """
        all_layers = list(self.model.model.layers)
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i] for i in layers}

    def get_mlp_down_proj_layers(self, layers: list[int] | None = None) -> dict[int, nn.Module]:
        """Get the mlp.down_proj modules for specified layers.

        The input to down_proj is the MLP intermediate activation (the "neurons").

        Args:
            layers: List of layer indices. If None, all layers are returned.

        Returns:
            dict mapping layer index to the layer's mlp.down_proj module.
        """
        all_layers = list(self.model.model.layers)
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i].mlp.down_proj for i in layers}

    def get_embedding_layer(self) -> nn.Module:
        """Get the token embedding layer (embed_tokens).

        Returns the output of embed_tokens, which is the representation after
        token embedding and before any transformer block processes it.
        For LLaMA-style models, positional information (RoPE) is applied inside
        each attention layer, so this is purely the token embedding.

        Returns:
            The embed_tokens module.
        """
        return self.model.model.embed_tokens

    def get_attn_o_proj_layers(self, layers: list[int] | None = None) -> dict[int, nn.Module]:
        """Get the self_attn.o_proj modules for specified layers.

        The input to o_proj is the concatenated per-head attention outputs.

        Args:
            layers: List of layer indices. If None, all layers are returned.

        Returns:
            dict mapping layer index to the layer's self_attn.o_proj module.
        """
        all_layers = list(self.model.model.layers)
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i].self_attn.o_proj for i in layers}

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads in the model."""
        return self.model.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        if hasattr(self.model.config, "head_dim"):
            return self.model.config.head_dim
        return self.model.config.hidden_size // self.model.config.num_attention_heads

    def generate(
        self,
        prompt: str,
        attention_layers: list[int] | None = None,
        max_new_tokens: int = 50,
    ) -> tuple[str, str, torch.Tensor, dict, torch.Tensor]:
        """Generate text from the model using greedy decoding.

        Args:
            prompt (str): The input prompt.
            attention_layers (Optional[list[int]]): List of layer indices to extract attention from.
                If None, no attentions are extracted.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            generated_text (str): Decoded text.
            completion (str): The generated completion text.
            logits (Tensor): [seq_len_generated, vocab_size] logits per token.
            attentions (dict): A dictionary mapping layer indices to lists of attention tensors.
            output_ids (Tensor): [prompt + seq_len_generated] token IDs.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # [1, seq_len]
        generated_ids = input_ids.clone()

        logits_generated = []
        attentions_generated = dict()
        output_attns = attention_layers is not None and len(attention_layers) > 0
        # Greedy decoding loop
        with torch.no_grad():
            past_key_values = None
            next_token_id = None
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = self.model(
                        input_ids=generated_ids, past_key_values=None, use_cache=True, output_attentions=output_attns
                    )
                else:
                    outputs = self.model(
                        input_ids=next_token_id.unsqueeze(0),  # Only the new token [1, 1]
                        past_key_values=past_key_values,  # Reuse cached keys/values
                        use_cache=True,
                        output_attentions=output_attns,
                    )

                past_key_values = outputs.past_key_values
                logits = outputs.logits  # [1, seq_len, vocab_size]

                # Extract attention weights
                # outputs.attentions is a tuple of (num_layers,) tensors
                # Each tensor shape: [batch_size, num_heads, seq_len_current, seq_len_total]
                if outputs.attentions is not None and attention_layers is not None:
                    for layer_idx, layer_attention in enumerate(outputs.attentions):
                        if layer_idx not in attention_layers:
                            continue
                        if layer_idx not in attentions_generated:
                            attentions_generated[layer_idx] = []

                        # Extract attention for the last token (the one just generated)
                        # layer_attention shape: [1, num_heads, seq_len_current, seq_len_total]
                        # We want: [num_heads, seq_len_total]
                        token_attention = layer_attention[0, :, -1, :].detach().cpu()
                        attentions_generated[layer_idx].append(token_attention)

                # Take last token's logits
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                logits_generated.append(next_token_logits.squeeze(0).detach().cpu())
                # Greedy selection
                next_token_id = next_token_logits.argmax(dim=-1)  # [1]
                # Append token
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
                # Stop if EOS token generated
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        # Stack logits
        logits_generated = torch.stack(logits_generated)  # [seq_len_generated, vocab_size]
        output_ids = generated_ids.squeeze(0)  # [prompt_length + seq_len_generated]
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        completion = self.tokenizer.decode(
            output_ids[input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        return generated_text, completion, logits_generated, attentions_generated, output_ids

    def extract_activations(
        self,
        prompt: str,
        layers: dict[int, nn.Module],
        max_new_tokens: int = 50,
    ) -> tuple[dict[int, list[Tensor]], str, str, Tensor, dict, Tensor]:
        """Run a forward pass on the given prompt and store activations from specified layers.

        Args:
            prompt (str): The prompt to run the model on.
            layers (dict[int, nn.Module]): The layers to extract activations from.
            max_new_tokens (int): Tokens to generate.

        Returns:
            MultiLayerActivationStore: A store containing activations for the given prompt
            str: The generated response from the model.
            str: The entire completion from the generated response.
            Tensor: The logits of the generated tokens.
            Tensor: The output ids of the generated tokens.
        """
        store = MultiLayerActivationStore(layers)
        generated_text, completion, logits, attentions, output_ids = self.generate(
            prompt, list(layers.keys()), max_new_tokens
        )
        activations = dict(store.activations)
        store.remove_all_hooks()
        store.clear()
        return (
            activations,
            generated_text,
            completion,
            logits,
            attentions,
            output_ids,
        )
