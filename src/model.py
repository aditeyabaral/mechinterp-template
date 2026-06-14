"""Load a decoder-only LM into a TransformerLens `TransformerBridge` for analysis.

WHY TransformerLens (TL):
    Mechinterp needs two things a plain HuggingFace model does not give you cleanly:
      1. CAPTURE -- read the internal activations (residual stream, MLP neurons, attention
         head outputs, ...) at every layer and every token position, in one forward pass.
      2. INTERVENTION -- edit those activations mid-forward-pass (e.g. zero out a neuron) to
         test whether a component CAUSALLY matters.
    TransformerLens exposes both through a single, uniform "hook" interface, so you don't have
    to hand-write PyTorch forward hooks per architecture. This template uses it everywhere.

WHAT a `TransformerBridge` is (the "v3" entry point):
    `TransformerBridge` wraps ANY HuggingFace model -- including a small GPT-2 you trained from
    scratch in src/train/ -- and re-exposes it with TransformerLens's standard hook names. That
    means the SAME analysis code runs on real `gpt2` and on your custom toy model: you only ever
    change the model path. The bridge reads the model's real config, so cfg.n_layers / n_heads /
    d_head automatically reflect whatever you trained.

The hook names you'll use downstream (see src/inference.py) are uniform across architectures:
    blocks.{i}.hook_resid_post   residual stream after block i   [batch, pos, d_model]
    blocks.{i}.mlp.hook_post     MLP intermediate "neurons"      [batch, pos, d_mlp]
    blocks.{i}.attn.hook_z       per-head attention output       [batch, pos, n_heads, d_head]
    hook_embed                   token embedding (pre-block 0)   [batch, pos, d_model]

A note on config names: TransformerLens renames the HuggingFace GPT-2 config fields, so the size
you set during training maps to a `model.cfg` field here. The equivalences you'll meet:
    HF GPT2Config (src/train) ->  model.cfg (here)
    n_layer                   ->  n_layers      (number of blocks)
    n_embd                    ->  d_model       (residual stream width)
    n_inner / intermediate    ->  d_mlp         (MLP hidden width = number of neurons)
    n_head                    ->  n_heads       (attention heads per block)
    n_embd / n_head           ->  d_head        (size of each head)

This file is generic infrastructure -- you normally will NOT need to edit it.
"""

import torch
from transformer_lens.model_bridge import TransformerBridge


def load_model(model_path: str, device: torch.device | str | None = None) -> TransformerBridge:
    """Boot a HuggingFace model (local path or Hub repo) into a TransformerLens bridge.

    Works for both workflows this template supports:
      - a real pretrained model, e.g. load_model("gpt2"), and
      - a toy model you trained from scratch (src/train/), e.g. load_model("you/your-toy-gpt2").

    Note: load custom toy models FROM A PATH OR HUB REPO (what `save_pretrained` / `push_to_hub`
    write), not from an in-memory model object -- the bridge reads tokenizer/config metadata that
    only exists once the model has been saved.

    Args:
        model_path: HuggingFace Hub repo id or local directory of the model to load.
        device: Where to place the model ("cuda", "cpu", or a torch.device). Defaults to CUDA
            if available, else CPU.

    Returns:
        A TransformerBridge in eval mode. Use it with:
          - model.run_with_cache(tokens)            -> (logits, cache) for activation capture
          - model.run_with_hooks(tokens, fwd_hooks) -> logits, applying interventions
          - model.hooks(fwd_hooks=...)              -> context manager (e.g. to ablate during generate)
          - model.generate(...)                     -> autoregressive decoding
        Its model.cfg exposes n_layers, n_heads, d_head, d_model, etc.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # boot_transformers loads the HF model + tokenizer and wraps them in the TL hook interface.
    # We pass the device through; everything else (config, sizes) is inferred from the checkpoint.
    model = TransformerBridge.boot_transformers(model_path, device=str(device))
    model.eval()  # inference mode: disable dropout etc. (we never train during analysis)
    return model
