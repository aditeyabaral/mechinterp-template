"""Wrapper around a GPT-2 LM exposing generation and activation extraction/intervention hooks.

This wrapper navigates a GPT-2 module tree (the architecture trained by
src/train/train_model.py) via four architecture-specific accessors:

    | what                    | GPT-2 module           |
    | ----------------------- | ---------------------- |
    | decoder blocks          | model.transformer.h    |
    | MLP neuron projection   | block.mlp.c_proj       |
    | attention output proj   | block.attn.c_proj      |
    | token embeddings        | model.transformer.wte  |

The "MLP neuron projection" and "attention output projection" are the modules
whose *input* is, respectively, the MLP intermediate activations (the "neurons")
and the concatenated per-head attention outputs. The hooks attach to those inputs.
The rest of the class and all of the hooks in model/hooks are architecture-agnostic.
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hooks import MultiLayerActivationStore


class LargeLanguageModel:
    """A generic wrapper exposing a decoder-only LM's layers for extraction and intervention."""

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
            quantization_config=bnb_config,  # optional 4-bit quantization to fit bigger models in memory
            attn_implementation="eager",  # plain attention so output_attentions works (to read attention weights)
            dtype="auto",  # use the checkpoint's native precision
            device_map="auto",  # let accelerate place the model on GPU(s)/CPU automatically
        )
        self.model.eval()  # inference mode: disable dropout, etc. (we are not training here)

    # --------------------------------------------------------------------- #
    # Architecture-specific accessors (GPT-2).
    #
    # These four methods are the ONLY place that knows the model's module layout.
    # Everything else in this file and in model/hooks is architecture-agnostic.
    # To target a different decoder-only architecture, update just these:
    #   GPT-2 transformer.h / mlp.c_proj / attn.c_proj / transformer.wte
    #   (Llama would be model.layers / mlp.down_proj / self_attn.o_proj / embed_tokens).
    # --------------------------------------------------------------------- #

    def _decoder_blocks(self) -> list[nn.Module]:
        """Return the list of decoder block (layer) modules in order (GPT-2: transformer.h)."""
        return list(self.model.transformer.h)

    def get_layers(self, layers: list[int] | None) -> dict[int, nn.Module]:
        """Get the decoder block modules to extract and patch activations into.

        Args:
            layers (Optional[list[int]]): list of layer indices to extract and patch. If None, all layers are returned.

        Returns:
            dict[int, nn.Module]: A dictionary of layer idx: block module pairs for the specified layers.
        """
        all_layers = self._decoder_blocks()
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i] for i in layers}

    def get_mlp_neuron_layers(self, layers: list[int] | None = None) -> dict[int, nn.Module]:
        """Get the MLP neuron-projection modules for specified layers (GPT-2: mlp.c_proj).

        The input to this module is the MLP intermediate activation (the "neurons").

        Args:
            layers: List of layer indices. If None, all layers are returned.

        Returns:
            dict mapping layer index to the layer's MLP neuron-projection module.
        """
        all_layers = self._decoder_blocks()
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i].mlp.c_proj for i in layers}

    def get_embedding_layer(self) -> nn.Module:
        """Get the token embedding module (GPT-2: transformer.wte).

        Its output is the representation after token embedding and before any
        transformer block processes it. In GPT-2 positional embeddings are added
        separately (transformer.wpe), so this is purely the token embedding.

        Returns:
            The token embedding module.
        """
        return self.model.transformer.wte

    def extract_token_embeddings(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Look up the token embeddings for a prompt straight from the embedding layer.

        This is a pure embedding lookup (transformer.wte): the pre-transformer
        representation of each token, before any block (or the positional embedding)
        processes it. No generation or forward pass through the transformer blocks
        is performed.

        Args:
            prompt (str): The input text to embed.

        Returns:
            token_ids (Tensor): LongTensor [seq_len] of input token IDs.
            embeddings (Tensor): FloatTensor [seq_len, hidden_size] of token embeddings.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)  # [1, seq_len]
        with torch.no_grad():
            embeddings = self.get_embedding_layer()(input_ids)  # [1, seq_len, hidden_size]
        return input_ids.squeeze(0).detach().cpu(), embeddings.squeeze(0).detach().cpu()

    def get_attn_output_layers(self, layers: list[int] | None = None) -> dict[int, nn.Module]:
        """Get the attention output-projection modules for specified layers (GPT-2: attn.c_proj).

        The input to this module is the concatenated per-head attention outputs.

        Args:
            layers: List of layer indices. If None, all layers are returned.

        Returns:
            dict mapping layer index to the layer's attention output-projection module.
        """
        all_layers = self._decoder_blocks()
        if layers is None:
            layers = list(range(len(all_layers)))
        return {i: all_layers[i].attn.c_proj for i in layers}

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

        Why we hand-roll the loop instead of calling model.generate(): writing the decoding
        loop ourselves means our hooks fire on each forward pass in a predictable way, and we
        get the per-step logits/attentions we want for analysis.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # [1, seq_len]
        generated_ids = input_ids.clone()  # grows by one token each step

        logits_generated = []
        attentions_generated = dict()
        output_attns = attention_layers is not None and len(attention_layers) > 0
        # Greedy decoding: at each step take the single most-likely next token (argmax), append
        # it, and feed it back in. no_grad() since we never backprop during analysis.
        with torch.no_grad():
            past_key_values = None
            next_token_id = None
            for step in range(max_new_tokens):
                if step == 0:
                    # Step 0: run the FULL prompt through the model in one forward pass.
                    # This is the pass our extraction hooks read prompt-position activations from.
                    outputs = self.model(
                        input_ids=generated_ids, past_key_values=None, use_cache=True, output_attentions=output_attns
                    )
                else:
                    # Later steps: feed ONLY the one new token. The KV cache (past_key_values)
                    # holds the keys/values for all previous positions, so we don't recompute
                    # the whole sequence each time -- a standard generation speed-up.
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

                # The model outputs a logit (score) for every vocab token at every position;
                # the prediction for the NEXT token lives at the last position.
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                logits_generated.append(next_token_logits.squeeze(0).detach().cpu())
                # Greedy selection: pick the single highest-scoring token (deterministic, no sampling).
                next_token_id = next_token_logits.argmax(dim=-1)  # [1]
                # Append it to the running sequence; it becomes next step's input.
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
                # Stop early if the model emits the end-of-sequence token.
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
    ) -> dict[str, object]:
        """Run a forward pass on the given prompt and store activations from specified layers.

        Args:
            prompt (str): The prompt to run the model on.
            layers (dict[int, nn.Module]): The layers to extract activations from.
            max_new_tokens (int): Tokens to generate.

        Returns:
            dict with keys:
                "activations":    dict[int, list[Tensor]] per-layer activations from generation.
                "generated_text": str full decoded text (prompt + completion).
                "completion":     str generated completion only.
                "logits":         Tensor [seq_len_generated, vocab_size] per-token logits.
                "attentions":     dict[int, list[Tensor]] attention weights per layer.
                "output_ids":     Tensor [prompt_len + seq_len_generated] token IDs.
        """
        store = MultiLayerActivationStore(layers)
        generated_text, completion, logits, attentions, output_ids = self.generate(
            prompt, list(layers.keys()), max_new_tokens
        )
        activations = dict(store.activations)
        store.remove_all_hooks()
        store.clear()
        return {
            "activations": activations,
            "generated_text": generated_text,
            "completion": completion,
            "logits": logits,
            "attentions": attentions,
            "output_ids": output_ids,
        }

    def extract(
        self,
        prompt: str,
        layers: dict[int, nn.Module],
        max_new_tokens: int = 50,
    ) -> dict[str, object]:
        """Run residual-stream activation extraction and token-embedding extraction together.

        Convenience wrapper that combines extract_activations (per-layer activations
        captured during generation) with extract_token_embeddings (the static token
        embeddings for the prompt), returning everything in a single dict.

        Args:
            prompt (str): The prompt to run the model on.
            layers (dict[int, nn.Module]): The decoder blocks to extract activations from.
            max_new_tokens (int): Tokens to generate.

        Returns:
            dict with keys:
                "activations":    dict[int, list[Tensor]] per-layer activations from generation.
                "embeddings":     Tensor [prompt_len, hidden_size] prompt token embeddings.
                "token_ids":      Tensor [prompt_len] prompt token IDs (for the embeddings).
                "generated_text": str full decoded text (prompt + completion).
                "completion":     str generated completion only.
                "logits":         Tensor [seq_len_generated, vocab_size] per-token logits.
                "attentions":     dict[int, list[Tensor]] attention weights per layer.
                "output_ids":     Tensor [prompt_len + seq_len_generated] token IDs.
        """
        result = self.extract_activations(prompt, layers, max_new_tokens=max_new_tokens)
        token_ids, embeddings = self.extract_token_embeddings(prompt)
        return {
            "activations": result["activations"],
            "embeddings": embeddings,
            "token_ids": token_ids,
            "generated_text": result["generated_text"],
            "completion": result["completion"],
            "logits": result["logits"],
            "attentions": result["attentions"],
            "output_ids": result["output_ids"],
        }
