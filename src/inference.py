"""Inference loop: generate on each prompt and capture / ablate internal activations via TransformerLens.

This file works on any decoder-only model out of the box (it talks to the model only through the
TransformerLens `TransformerBridge` from src/model.py). Two functions are the task-specific
override points you may want to customise (both have sensible defaults and `TODO` markers):
  - find_answer_span(generated_text):        which span of the generation is "the answer".
  - find_positions_of_interest(model, prompt): which prompt token positions to capture for geometry.
Everything else (the capture, ablation hooks, and the run loop) is task-agnostic.

HOW CAPTURE AND ABLATION WORK HERE (the TransformerLens idea):
    Instead of hand-writing PyTorch forward hooks per architecture, we name the activation we want
    with a TransformerLens "hook name" and let the library do the rest. Two calls cover everything:
      - model.run_with_cache(tokens) -> (logits, cache): runs the model once and hands back a
        `cache` you can index by hook name to read EVERY layer/position activation. We use:
            blocks.{l}.hook_resid_post   residual stream after block l   [batch, pos, d_model]
            blocks.{l}.mlp.hook_post     MLP intermediate "neurons"      [batch, pos, d_mlp]
            blocks.{l}.attn.hook_z       per-head attention output       [batch, pos, n_heads, d_head]
            hook_embed                   token embedding (pre-block 0)   [batch, pos, d_model]
      - model.run_with_hooks(tokens, fwd_hooks=[(name, fn), ...]): runs the model while calling
        each fn on the named activation, letting fn EDIT it (we set entries to 0 to ablate).

    Because the cache holds all positions from a single pass over the full output sequence, we can
    read the answer token's activations directly -- there is no "last token has no forward pass"
    edge case to work around.
"""

import re
from typing import Any

import torch
from tqdm.auto import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

from utils.dataset import PromptDataset


def find_answer_span(generated_text: str) -> tuple[str, int, int] | None:
    """Locate the model's answer inside the text it generated.

    WHY this exists: the model just produces a stream of tokens; we need to know which part of
    that stream is "the answer" so we can read activations at the right place. This function
    returns the character span of the answer; inference.py then maps it to tokens and captures
    activations at the answer's LAST token.

    The default grabs the first whitespace-delimited chunk of the generation (a good target for
    short, single-"word" answers like a number). Override it if your answers look different.

    Args:
        generated_text: The decoded text the model generated after the prompt.

    Returns:
        (answer_text, start_char, end_char): the answer substring and its character offsets
        within generated_text, or None if no answer could be found (that prompt is then skipped).
    """
    # TODO (optional): change how the answer is located for your task. Return the (text, start,
    # end) of the answer substring, or None to skip the prompt. The default below works for many
    # short answers; only change it if your answer format needs something more specific.
    #
    # Example (an arithmetic task whose answer is an integer, possibly negative, e.g. "12" or "-46"):
    #     match = re.search(r"^\s*(-?\d+)", generated_text)
    match = re.search(r"^\s*(\S+)", generated_text)
    if not match:
        return None
    return match.group(1), match.start(1), match.end(1)


def find_positions_of_interest(model: TransformerBridge, prompt: str) -> dict[str, int | None]:
    """Return extra PROMPT token positions to capture activations at (for --capture-geometry).

    WHY this exists: by default we capture activations only at the answer token. But often you
    also want activations at specific INPUT positions -- e.g. for "7+5=" you might study the
    representations sitting on the "7", the "5", and the "+". This function names those positions
    so inference.py records residual/MLP/head/embedding activations there too, stored under
    result["geometry"][name]. The default returns nothing, which is fine if you only care about
    the answer.

    Args:
        model: The TransformerLens bridge. Useful helpers: `model.to_str_tokens(prompt,
            prepend_bos=False)` returns the prompt split into per-token strings (so the i-th
            entry is the string of token i), which makes it easy to find a token's index.
        prompt: The full prompt string.

    Returns:
        dict mapping a name you choose to a token index within the prompt (or None to skip it).
        IMPORTANT: index tokens WITHOUT a BOS prefix, to match how inference.py tokenizes prompts
        (prepend_bos=False), so the indices line up with the captured activations.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): return the prompt token positions you want to study, e.g.
    #     {"operator": 5, "first_operand": 3}. Leave it returning {} to capture only the answer.
    # ----------------------------------------------------------------------------------- #
    #
    # Example (a single-digit addition task with prompts like "7+5="), capturing the '+' operator
    # and the '=' sign by scanning the per-token strings of the prompt:
    #
    #     str_tokens = model.to_str_tokens(prompt, prepend_bos=False)  # ["7", "+", "5", "=", ...]
    #     positions: dict[str, int | None] = {}
    #     for idx, tok in enumerate(str_tokens):
    #         if tok.strip() == "+":
    #             positions["operator"] = idx
    #         elif tok.strip() == "=":
    #             positions["eq_sign"] = idx
    #     return positions
    #
    # ----------------------------------------------------------------------------------- #
    return {}


def _map_char_to_token(model: TransformerBridge, token_ids: torch.Tensor, target_char_position: int) -> int | None:
    """Map a character position (in the decoded text) to the token index that covers it.

    We decode the tokens one more at a time; the first prefix whose decoded text is long enough to
    reach `target_char_position` is the token that contains that character.

    Args:
        model: The TransformerLens bridge (for decoding via model.to_string).
        token_ids: 1-D tensor of token ids that were decoded.
        target_char_position: A character index into the decoded text.

    Returns:
        The token index containing that character, or None if it can't be mapped.
    """
    for token_idx in range(len(token_ids)):
        text_so_far = model.to_string(token_ids[: token_idx + 1])
        if len(text_so_far) > target_char_position:
            return token_idx
    return None


def _locate_answer(
    model: TransformerBridge,
    prompt_length: int,
    output_ids: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[int | None, torch.Tensor | None, str | None, torch.Tensor | None]:
    """Find the answer span in the generated text and report which token to read activations from.

    Steps:
      1. Decode the generated tokens (everything after the prompt) to text.
      2. Ask `find_answer_span` which substring is "the answer" (you override that per task).
      3. Map that substring's first and last characters back to token indices.

    CONVENTION: we key everything off the answer span's LAST token (e.g. the '2' of "42", the
    '3' of "-63"). The caller then reads all activations at that one ABSOLUTE position, so every
    captured tensor describes the same token. We also return the full span's token ids / string so
    you still know the whole answer, and the logits the last answer token was predicted from.

    Args:
        model: The TransformerLens bridge (used for decoding).
        prompt_length: Number of prompt tokens (generation starts right after this index).
        output_ids: Full token sequence [prompt tokens + generated tokens], shape [seq_len].
        logits: Logits over the full output sequence, shape [1, seq_len, vocab_size].

    Returns:
        (answer_position, answer_token_ids, answer_token, answer_logits), or all None if no answer
        span could be located. answer_position is the ABSOLUTE index (into output_ids) of the
        answer span's last token.
    """
    generated_ids = output_ids[prompt_length:]
    if len(generated_ids) == 0:
        return None, None, None, None
    generated_text = model.to_string(generated_ids)

    span = find_answer_span(generated_text)
    if span is None:
        return None, None, None, None
    answer_text, answer_start_char, answer_end_char = span

    # Translate character positions in the generated text into token indices (relative to the
    # generated chunk).
    answer_start_token = _map_char_to_token(model, generated_ids, answer_start_char)
    answer_end_token = _map_char_to_token(model, generated_ids, answer_end_char - 1)
    if answer_start_token is None or answer_end_token is None:
        return None, None, None, None

    # The answer span may be several tokens (e.g. "42" -> ['4', '2']); keep the whole span for
    # reference but key the position off its LAST token (answer_end_token).
    answer_token_ids = generated_ids[answer_start_token : answer_end_token + 1]
    answer_token = model.to_string(answer_token_ids)
    if answer_token.strip() != answer_text:
        return None, None, None, None

    # Absolute position of the answer span's last token within the full output sequence.
    answer_position = prompt_length + answer_end_token

    # Logits the last answer token was predicted from: in an autoregressive model the prediction
    # for position p is computed at position p-1, so we read logits one step earlier.
    pred_position = answer_position - 1
    if 0 <= pred_position < logits.shape[1]:
        answer_logits = logits[0, pred_position].detach().cpu()
    else:
        answer_logits = None

    return answer_position, answer_token_ids, answer_token, answer_logits


def build_ablation_hooks(
    mlp_ablation: dict[int, list[int]] | None,
    head_ablation: dict[int, list[int]] | None,
) -> list[tuple[str, Any]]:
    """Build the TransformerLens forward-hook list that zeroes out the requested neurons/heads.

    Each entry is a (hook_name, fn) pair. TransformerLens calls fn(activation, hook) during the
    forward pass and uses fn's return value in place of the original activation, so setting some
    entries to 0 ABLATES (knocks out) those components -- the core causal test of mechinterp.

    Args:
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} of MLP neurons to zero, applied
            to blocks.{layer}.mlp.hook_post (shape [batch, pos, d_mlp]).
        head_ablation: Optional dict {layer_idx: [head indices]} of attention heads to zero,
            applied to blocks.{layer}.attn.hook_z (shape [batch, pos, n_heads, d_head]).

    Returns:
        A list of (hook_name, hook_fn) pairs suitable for model.run_with_hooks / model.hooks.
        Empty if no ablation was requested.
    """
    hooks: list[tuple[str, Any]] = []

    if mlp_ablation:
        for layer_idx, neuron_indices in mlp_ablation.items():
            # idxs bound as a default arg so each layer's closure keeps its own indices
            # (the classic Python late-binding gotcha).
            def mlp_hook(act: torch.Tensor, hook: HookPoint, idxs: list[int] = list(neuron_indices)) -> torch.Tensor:
                act[..., idxs] = 0.0  # zero these neurons at every position
                return act

            hooks.append((f"blocks.{layer_idx}.mlp.hook_post", mlp_hook))

    if head_ablation:
        for layer_idx, head_indices in head_ablation.items():

            def head_hook(act: torch.Tensor, hook: HookPoint, idxs: list[int] = list(head_indices)) -> torch.Tensor:
                act[:, :, idxs, :] = 0.0  # zero these heads' outputs at every position
                return act

            hooks.append((f"blocks.{layer_idx}.attn.hook_z", head_hook))

    return hooks


def _capture_geometry(
    cache: dict[str, torch.Tensor],
    layer_indices: list[int],
    positions: dict[str, int | None],
) -> dict[str, dict[int, dict[str, torch.Tensor | None]]]:
    """Slice residual / MLP / head / embedding activations at named prompt positions out of the cache.

    Args:
        cache: The activation cache from model.run_with_cache over the full output sequence.
        layer_indices: Layer indices to extract from.
        positions: Mapping {name: token_index} of prompt positions (from find_positions_of_interest);
            any index may be None.

    Returns:
        dict mapping each position name to layer_idx ->
        {"residual": Tensor | None, "mlp": Tensor | None, "heads": Tensor | None}.
        layer_idx=-1 holds the pre-transformer token embedding (residual only, mlp/heads=None).
    """
    geometry: dict[str, dict] = {name: {} for name in positions}
    embed = cache["hook_embed"][0]  # [seq_len, d_model]

    for name, pos in positions.items():
        # layer_idx = -1 holds the token embedding (the pre-block representation).
        in_embed = pos is not None and 0 <= pos < embed.shape[0]
        geometry[name][-1] = {
            "residual": embed[pos].detach().cpu() if in_embed else None,
            "mlp": None,
            "heads": None,
        }
        for layer_idx in layer_indices:
            resid = cache[f"blocks.{layer_idx}.hook_resid_post"][0]  # [seq_len, d_model]
            mlp = cache[f"blocks.{layer_idx}.mlp.hook_post"][0]  # [seq_len, d_mlp]
            heads = cache[f"blocks.{layer_idx}.attn.hook_z"][0]  # [seq_len, n_heads, d_head]
            in_range = pos is not None and 0 <= pos < resid.shape[0]
            geometry[name][layer_idx] = {
                "residual": resid[pos].detach().cpu() if in_range else None,
                "mlp": mlp[pos].detach().cpu() if in_range else None,
                "heads": heads[pos].detach().cpu() if in_range else None,  # [n_heads, d_head]
            }

    return geometry


def _run_single_prompt(
    model: TransformerBridge,
    prompt: str,
    prompt_length: int,
    layer_indices: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> dict[str, Any]:
    """Run a single prompt through the model, capturing answer-token and geometry activations.

    Flow:
      1. Generate a completion (with any ablation hooks active, so the generation reflects the
         knocked-out components).
      2. Re-run the FULL output sequence once -- with the same hooks -- to get logits, and (when
         capturing geometry) a cache of every layer/position activation.
      3. Locate the answer span's last token and read all activations at that single position.

    Args:
        model: The TransformerLens bridge.
        prompt: Full prompt string.
        prompt_length: Number of tokens in the prompt (prepend_bos=False).
        layer_indices: Layer indices to capture from.
        max_new_tokens: Maximum tokens to generate.
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} to zero out.
        head_ablation: Optional dict {layer_idx: [head indices]} to zero out.
        capture_geometry: If True, read per-position activations (MLP neurons, heads, residual,
            embedding) and store output_ids. Set False during ablation sweeps to store less data.

    Returns:
        Dict with keys: text, completion, output_ids (None when capture_geometry=False), and an
        answer sub-dict (position, token_id, token, mlp_neurons, attn_heads, residual, embedding,
        logits) plus a geometry sub-dict (empty when capture_geometry=False).
    """
    ablation_hooks = build_ablation_hooks(mlp_ablation, head_ablation)
    prompt_tokens = model.to_tokens(prompt, prepend_bos=False)  # [1, prompt_length]

    # 1. Generate with ablation applied. Greedy decoding (do_sample=False) so runs are
    #    deterministic. We pass already-tokenized input (built with prepend_bos=False above) so
    #    token indices stay aligned with prompt_length.
    with model.hooks(fwd_hooks=ablation_hooks):
        out_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="tokens",
            verbose=False,
        )
    output_ids = out_tokens[0].detach().cpu()  # [prompt_length + n_generated]
    generated_text = model.to_string(output_ids)
    completion = model.to_string(output_ids[prompt_length:]).strip()

    # 2. Re-run the whole output sequence once (same ablation) to read logits, plus a full
    #    activation cache when we need geometry. One pass covers every position, including the
    #    answer's last token.
    if capture_geometry:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits, cache = model.run_with_cache(out_tokens)
    else:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits = model.run_with_hooks(out_tokens, return_type="logits")
        cache = None

    # 3. Locate the answer span's last token and the absolute position to read activations at.
    answer_position, answer_token_ids, answer_token, answer_logits = _locate_answer(
        model, prompt_length, output_ids, logits
    )

    # 4. Read all activations at the answer token's position (same position across every tensor,
    #    so they all describe the same token).
    if capture_geometry and cache is not None and answer_position is not None:
        pos = answer_position
        mlp_neurons = {i: cache[f"blocks.{i}.mlp.hook_post"][0, pos].detach().cpu() for i in layer_indices}
        attn_heads = {i: cache[f"blocks.{i}.attn.hook_z"][0, pos].detach().cpu() for i in layer_indices}
        answer_residual = {i: cache[f"blocks.{i}.hook_resid_post"][0, pos].detach().cpu() for i in layer_indices}
        answer_embedding = cache["hook_embed"][0, pos].detach().cpu()

        positions = find_positions_of_interest(model, prompt)
        geometry = _capture_geometry(cache, layer_indices, positions)
    else:
        mlp_neurons = {i: None for i in layer_indices}
        attn_heads = {i: None for i in layer_indices}
        answer_residual = {i: None for i in layer_indices}
        answer_embedding = None
        geometry = {}

    return {
        "text": generated_text,
        "completion": completion,
        "output_ids": output_ids if capture_geometry else None,
        "answer": {
            "position": answer_position,
            "token_id": answer_token_ids,
            "token": answer_token,
            "mlp_neurons": mlp_neurons,
            "attn_heads": attn_heads,
            "residual": answer_residual,
            "embedding": answer_embedding,
            "logits": answer_logits if answer_logits is not None else None,
        },
        "geometry": geometry,
    }


def run(
    model: TransformerBridge,
    dataset: PromptDataset,
    layers: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Run inference over a dataset, capturing MLP-neuron, per-head, residual, and geometry activations.

    Args:
        model: The TransformerLens bridge.
        dataset: Dataset whose `.prompts` is a list of prompt strings to run.
        layers: Layer indices to capture from.
        max_new_tokens: Max tokens to generate per prompt.
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} to zero on every prompt.
        head_ablation: Optional dict {layer_idx: [head indices]} to zero on every prompt.
        capture_geometry: If True, capture activations at the answer token and at the
            find_positions_of_interest positions. Set False during ablation sweeps to avoid
            storing per-feature geometry data.

    Returns:
        List of result dicts, one per prompt (see _run_single_prompt for the per-prompt schema).
    """
    results: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(tqdm(dataset.prompts, desc="Running neuron inference")):
        # prompt_length (in tokens, no BOS) marks the boundary between prompt and generated tokens.
        prompt_length = model.to_tokens(prompt, prepend_bos=False).shape[1]

        run_result = _run_single_prompt(
            model,
            prompt,
            prompt_length,
            layers,
            max_new_tokens,
            mlp_ablation=mlp_ablation,
            head_ablation=head_ablation,
            capture_geometry=capture_geometry,
        )

        results.append(
            {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "prompt_length": prompt_length,
                "result": run_result,
            }
        )

    return results
