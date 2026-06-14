"""Inference loop capturing MLP intermediate neuron activations and per-head attention outputs.

This file works on any decoder-only model out of the box. Two functions are the task-specific
override points you may want to customise (both have sensible defaults and `TODO` markers):
  - find_answer_span(generated_text):       which span of the generation is "the answer".
  - find_positions_of_interest(tok, prompt): which prompt token positions to capture for geometry.
Everything else (the capture stores, ablation hooks, and the run loop) is task-agnostic.
"""

import re
from typing import Any

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from model import LargeLanguageModel
from model.hooks import (
    AttentionHeadAblationHook,
    AttentionHeadExtractionStore,
    MlpNeuronAblationHook,
    MlpNeuronExtractionStore,
    MultiLayerActivationStore,
)
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
    # Example (operator-overloading arithmetic study): match a signed integer, handling both the
    # normal format (e.g. "-46") and the reverse-digit training format (e.g. "64-"):
    #     match = re.search(r"^\s*([+-]?\d+[+-]?)", generated_text)
    match = re.search(r"^\s*(\S+)", generated_text)
    if not match:
        return None
    return match.group(1), match.start(1), match.end(1)


def find_positions_of_interest(tokenizer: PreTrainedTokenizerFast, prompt: str) -> dict[str, int | None]:
    """Return extra PROMPT token positions to capture activations at (for --capture-geometry).

    WHY this exists: by default we capture activations only at the answer token. But often you
    also want activations at specific INPUT positions -- e.g. for "7+5=" you might study the
    representations sitting on the "7", the "5", and the "+". This function names those positions
    so inference.py records residual/MLP/head/embedding activations there too, stored under
    result["geometry"][name]. The default returns nothing, which is fine if you only care about
    the answer.

    Args:
        tokenizer: The model's tokenizer. `tokenizer(prompt, return_offsets_mapping=True)` gives
            you, for each token, the (start_char, end_char) it covers -- handy for finding the
            token index of a particular character in the prompt.
        prompt: The full prompt string.

    Returns:
        dict mapping a name you choose to a token index within the prompt (or None to skip it).
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): return the prompt token positions you want to study, e.g.
    #     {"operator": 5, "first_operand": 3}. Leave it returning {} to capture only the answer.
    # ----------------------------------------------------------------------------------- #
    #
    # Example (operator-overloading arithmetic study), capturing the two operands, the operator,
    # and the '=' of the final 'a<op>b=' question by mapping their character positions to tokens:
    #
    #     enc = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    #     offsets = enc["offset_mapping"][0].tolist()   # [(start_char, end_char), ...] per token
    #
    #     def char_to_token(char_pos: int) -> int | None:
    #         for tok_idx, (start, end) in enumerate(offsets):
    #             if start <= char_pos < end:
    #                 return tok_idx
    #         return None
    #
    #     # find the char positions of the operands / operator / '=' in `prompt`,
    #     # map each via char_to_token(...), and return them under names you choose:
    #     return {"operand_a": ia, "operand_b": ib, "operator": iop, "eq_sign": ieq}
    #
    # ----------------------------------------------------------------------------------- #
    return {}


def _map_char_to_token_position(
    tokenizer: PreTrainedTokenizerFast, token_ids: torch.Tensor, target_char_position: int
) -> int | None:
    """Map a character position to token position using incremental decoding.

    Fallback method when offset_mapping is not available.

    Args:
        tokenizer: The tokenizer used to decode tokens.
        token_ids: The token IDs that were decoded.
        target_char_position: The character position in the decoded text.

    Returns:
        The token index that contains or starts at the target character position,
        or None if mapping fails.
    """
    for token_idx in range(len(token_ids)):
        text_so_far = tokenizer.decode(token_ids[: token_idx + 1], skip_special_tokens=True)
        if len(text_so_far) > target_char_position:
            return token_idx
    return None


def _locate_answer(
    llm: LargeLanguageModel,
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
    '3' of "-63"). The caller then reads all activations at that one token, so every captured
    tensor describes the same position. We also return the full span's token ids / string so you
    still know the whole answer, and the logits the last answer token was sampled from.

    Args:
        llm: The model wrapper (used for its tokenizer).
        prompt_length: Number of prompt tokens (generation starts right after this).
        output_ids: Full token sequence [prompt tokens + generated tokens].
        logits: Per-generated-token logits, shape [n_generated, vocab_size].

    Returns:
        (answer_position_in_output, answer_token_ids, answer_token, answer_logits), or all None
        if no answer span could be located. answer_position_in_output is the ABSOLUTE index (into
        output_ids) of the answer span's last token.
    """
    generated_ids = output_ids[prompt_length:]
    if len(generated_ids) == 0:
        return None, None, None, None
    generated_text = llm.tokenizer.decode(generated_ids, skip_special_tokens=True)

    span = find_answer_span(generated_text)
    if span is None:
        return None, None, None, None
    answer_text, answer_start_char, answer_end_char = span

    # Translate character positions in the generated text into token indices.
    answer_start_token = _map_char_to_token_position(llm.tokenizer, generated_ids, answer_start_char)
    answer_end_token = _map_char_to_token_position(llm.tokenizer, generated_ids, answer_end_char - 1)
    if answer_start_token is None or answer_end_token is None:
        return None, None, None, None
    if not (0 <= answer_start_token < logits.shape[0] and 0 <= answer_end_token < logits.shape[0]):
        return None, None, None, None

    # The answer span may be several tokens (e.g. "42" -> ['4', '2']); keep the whole span for
    # reference but key the position off its LAST token (answer_end_token).
    answer_token_ids = generated_ids[answer_start_token : answer_end_token + 1]
    answer_token = llm.tokenizer.decode(answer_token_ids, skip_special_tokens=True)
    if answer_token.strip() != answer_text:
        return None, None, None, None

    # Logits the last answer token was sampled from (in an autoregressive model these are computed
    # one step earlier, at the position just before the token).
    answer_logits = logits[answer_end_token].detach().clone().cpu()
    answer_position_in_output = prompt_length + answer_end_token
    return answer_position_in_output, answer_token_ids, answer_token, answer_logits


def _extract_all_activations(  # noqa: C901
    activations: dict[int, list[torch.Tensor]],
    mlp_store: MlpNeuronExtractionStore,
    head_store: AttentionHeadExtractionStore,
    layer_indices: list[int],
    positions: dict[str, int | None],
    embedding_tensor: torch.Tensor | None = None,
) -> dict[str, dict[int, dict[str, torch.Tensor | None]]]:
    """Extract residual stream, MLP neuron, and attention head activations at prompt token positions.

    Uses the step-0 full-prompt pass for all three activation types:
      - Residual stream: activations[layer][0], shape [1, prompt_len, hidden_size]
      - MLP neurons: mlp_store.activations[layer][0], shape [1, prompt_len, intermediate_size]
      - Attention heads: head_store.activations[layer][0], shape [1, prompt_len, num_heads, head_dim]

    If embedding_tensor is provided, also populates layer_idx=-1 with the token embedding
    representation (before any transformer block), with mlp=None and heads=None.

    All sliced at the specified token positions to produce per-position, per-layer dicts.

    Args:
        activations: Residual stream activations from llm.extract_activations().
        mlp_store: MLP intermediate activation store (hooked for the forward pass).
        head_store: Attention head activation store (hooked for the forward pass).
        layer_indices: Layer indices to extract from.
        positions: Mapping {name: token_index} of prompt positions to capture (from
            find_positions_of_interest); any index may be None.
        embedding_tensor: Optional [prompt_len, hidden_size] tensor from the token embedding
            layer (embed_tokens output at step 0). If provided, stored at layer_idx=-1.

    Returns:
        dict mapping each position name to layer_idx ->
        {"residual": Tensor | None, "mlp": Tensor | None, "heads": Tensor | None}.
        layer_idx=-1 holds the pre-transformer token embedding (residual only, mlp/heads=None).
    """
    geometry: dict[str, dict] = {name: {} for name in positions}

    # Populate layer_idx=-1 with the token embedding (pre-transformer representation).
    if embedding_tensor is not None:
        for name, pos in positions.items():
            if pos is None or pos < 0 or pos >= embedding_tensor.shape[0]:
                geometry[name][-1] = {"residual": None, "mlp": None, "heads": None}
            else:
                geometry[name][-1] = {
                    "residual": embedding_tensor[pos].detach().clone().cpu(),
                    "mlp": None,
                    "heads": None,
                }

    for layer_idx in layer_indices:
        # Step-0 residual stream tensor: [1, prompt_len, hidden_size] → [prompt_len, hidden_size]
        resid_steps = activations.get(layer_idx, [])
        step0_resid = None
        if resid_steps:
            t = resid_steps[0]
            step0_resid = t[0] if t.dim() == 3 else (t.unsqueeze(0) if t.dim() == 1 else t)

        # Step-0 MLP neuron tensor: [1, prompt_len, intermediate_size] → [prompt_len, intermediate_size]
        mlp_steps = mlp_store.activations.get(layer_idx, [])
        step0_mlp = None
        if mlp_steps:
            t = mlp_steps[0]
            step0_mlp = t[0] if t.dim() == 3 else t

        # Step-0 attention head tensor: [1, prompt_len, num_heads, head_dim] → [prompt_len, num_heads, head_dim]
        head_steps = head_store.activations.get(layer_idx, [])
        step0_heads = None
        if head_steps:
            t = head_steps[0]
            step0_heads = t[0] if t.dim() == 4 else t

        for name, pos in positions.items():
            if pos is None:
                geometry[name][layer_idx] = {"residual": None, "mlp": None, "heads": None}
                continue

            def _slice(tensor: torch.Tensor | None, pos: int = pos) -> torch.Tensor | None:
                if tensor is None or pos < 0 or pos >= tensor.shape[0]:
                    return None
                return tensor[pos].detach().clone().cpu()

            geometry[name][layer_idx] = {
                "residual": _slice(step0_resid),
                "mlp": _slice(step0_mlp),
                "heads": _slice(step0_heads),  # [num_heads, head_dim]
            }

    return geometry


def _extract_neuron_tensors_at_position(
    mlp_store: MlpNeuronExtractionStore,
    head_store: AttentionHeadExtractionStore,
    generation_step: int,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor | None], dict[int, torch.Tensor | None]]:
    """Read each store's tensor at one generation step and slice out its last position.

    The stores hold one tensor per forward pass (index 0 = the prompt pass, indices 1.. = each
    generated token). We read the tensor at `generation_step` and take its last position, giving
    the MLP neurons / attention heads at the token fed in on that step.

    Args:
        mlp_store: Store of MLP intermediate activations per layer per generation step.
        head_store: Store of per-head attention outputs per layer per generation step.
        generation_step: Which forward pass to read (see convention in _run_single_prompt).
        layer_indices: Layer indices to extract from.

    Returns:
        mlp_neurons: dict mapping layer_idx to Tensor[intermediate_size], or None if unavailable.
        attn_heads: dict mapping layer_idx to Tensor[num_heads, head_dim], or None if unavailable.
    """
    mlp_neurons: dict[int, torch.Tensor | None] = {}
    attn_heads: dict[int, torch.Tensor | None] = {}

    for layer_idx in layer_indices:
        # MLP intermediate neurons at this step's last position.
        mlp_acts = mlp_store.activations.get(layer_idx, [])
        if generation_step < len(mlp_acts):
            act = mlp_acts[generation_step]  # [B, seq_len, intermediate_size]
            if act.dim() == 3:
                act = act[0, -1, :]  # [intermediate_size]
            elif act.dim() == 2:
                act = act[-1, :]
            mlp_neurons[layer_idx] = act
        else:
            mlp_neurons[layer_idx] = None

        # Per-head attention outputs at this step's last position.
        head_acts = head_store.activations.get(layer_idx, [])
        if generation_step < len(head_acts):
            act = head_acts[generation_step]  # [B, seq_len, num_heads, head_dim]
            if act.dim() == 4:
                act = act[0, -1, :, :]  # [num_heads, head_dim]
            elif act.dim() == 3:
                act = act[-1, :, :]
            attn_heads[layer_idx] = act
        else:
            attn_heads[layer_idx] = None

    return mlp_neurons, attn_heads


def _register_ablation_hooks(
    llm: LargeLanguageModel,
    neuron_layers: dict[int, nn.Module],
    attn_layers: dict[int, nn.Module],
    mlp_ablation: dict[int, list[int]] | None,
    head_ablation: dict[int, list[int]] | None,
) -> list:
    """Register MLP and attention-head ablation hooks and return the hook objects.

    Ablation hooks must be registered before extraction stores so that PyTorch
    fires them first; the extraction store then captures the already-zeroed
    activations, which is what actually flowed through the model.

    Args:
        llm: The large language model (used for num_attention_heads / head_dim).
        neuron_layers: Dict mapping layer_idx to the MLP neuron-projection modules (GPT-2: mlp.c_proj).
        attn_layers: Dict mapping layer_idx to the attention output-projection modules (GPT-2: attn.c_proj).
        mlp_ablation: Optional dict mapping layer_idx to MLP neuron indices to zero.
        head_ablation: Optional dict mapping layer_idx to attention head indices to zero.

    Returns:
        List of registered hook objects; call .remove() on each to clean up.
    """
    hooks: list = []
    if mlp_ablation:
        for layer_idx, neuron_indices in mlp_ablation.items():
            if layer_idx in neuron_layers:
                hooks.append(MlpNeuronAblationHook(neuron_layers[layer_idx], neuron_indices))
    if head_ablation:
        for layer_idx, head_indices in head_ablation.items():
            if layer_idx in attn_layers:
                hooks.append(
                    AttentionHeadAblationHook(
                        attn_layers[layer_idx], head_indices, llm.num_attention_heads, llm.head_dim
                    )
                )
    return hooks


def _run_single_prompt(  # noqa: C901
    llm: LargeLanguageModel,
    prompt: str,
    prompt_length: int,
    layers: dict[int, nn.Module],
    neuron_layers: dict[int, nn.Module],
    attn_layers: dict[int, nn.Module],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> dict[str, Any]:
    """Run a single prompt through the model, capturing neuron-level and geometry activations.

    Args:
        llm: The large language model.
        prompt: Full prompt string.
        prompt_length: Number of tokens in the prompt.
        layers: Transformer layer modules for residual-stream extraction.
        neuron_layers: Dict mapping layer_idx to MLP neuron-projection modules.
        attn_layers: Dict mapping layer_idx to attention output-projection modules.
        max_new_tokens: Maximum tokens to generate.
        mlp_ablation: Optional dict mapping layer_idx to MLP neuron indices to zero out.
        head_ablation: Optional dict mapping layer_idx to attention head indices to zero out.
        capture_geometry: If True, extract step-0 activations at the positions returned by
            find_positions_of_interest and store output_ids. Set to False during ablation
            sweeps to reduce stored data.

    Returns:
        Dict with keys: text, completion, output_ids (None when capture_geometry=False), and
        an answer sub-dict (position, token_id, token, mlp_neurons, attn_heads, residual, logits)
        plus a geometry sub-dict (empty when capture_geometry=False).
    """
    layer_indices = list(layers.keys())

    # ORDER MATTERS. Register ablation hooks FIRST, extraction stores SECOND. PyTorch fires
    # pre-hooks in registration order, so the ablation zeroes the tensor before the store
    # records it -- meaning the store captures the activations that *actually* flowed through
    # the model (post-ablation), not the originals.
    _ablation_hooks = _register_ablation_hooks(llm, neuron_layers, attn_layers, mlp_ablation, head_ablation)

    # Attach the capture stores. When capture_geometry is False (e.g. during a big ablation
    # sweep) we attach empty no-op stores so we only pay for the residual stream + logits and
    # skip the heavier per-neuron / per-head / embedding capture.
    if capture_geometry:
        mlp_store = MlpNeuronExtractionStore(neuron_layers)
        head_store = AttentionHeadExtractionStore(attn_layers, llm.num_attention_heads, llm.head_dim)
        emb_store = MultiLayerActivationStore({-1: llm.get_embedding_layer()})  # key -1 = token embedding
    else:
        mlp_store = MlpNeuronExtractionStore({})
        head_store = AttentionHeadExtractionStore({}, llm.num_attention_heads, llm.head_dim)
        emb_store = None

    # Generate. This single call runs the whole decoding loop; every hook above fires on each
    # forward pass and fills its store. `activations` is the residual stream (per layer, per step).
    result = llm.extract_activations(prompt, layers, max_new_tokens=max_new_tokens)
    activations = result["activations"]
    generated_text = result["generated_text"]
    completion = result["completion"]
    logits = result["logits"]
    output_ids = result["output_ids"]

    # Ablation hooks have done their job for this prompt; detach them so they don't fire again.
    for hook in _ablation_hooks:
        hook.remove()

    # Find the answer span in the generation and the absolute position of its LAST token.
    answer_position_in_output, answer_token_ids, answer_token, answer_logits = _locate_answer(
        llm, prompt_length, output_ids, logits
    )

    if capture_geometry and answer_position_in_output is not None:
        # ------------------------------------------------------------------------------------- #
        # WHICH TOKEN DO WE READ, AND FROM WHICH STORED TENSOR?
        # ------------------------------------------------------------------------------------- #
        # We read activations at the answer span's LAST token (e.g. '2' of "42", '3' of "-63").
        # A token's activations are produced on the forward pass where that token is the INPUT.
        # Generation runs as: step 0 feeds the whole prompt, then steps 1, 2, ... each feed one
        # newly generated token. So the answer's last token (generated index `answer_pos_in_gen`)
        # is the input on generation step `answer_pos_in_gen + 1` -- the "+1" skips the step-0
        # prompt pass. We use this same step index for the residual stream, MLP neurons, attention
        # heads, and the token embedding, so all four describe exactly the same token.
        #
        # Edge case: if the answer's last token is also the very last token generated (nothing
        # came after it), the model never fed it back in, so there is no such forward pass and
        # these captures will be None. Allow a couple of extra tokens (--max-new-tokens) to avoid
        # this. (The logits are unaffected: they are computed one step earlier, before the token.)
        # ------------------------------------------------------------------------------------- #
        answer_pos_in_gen = answer_position_in_output - prompt_length
        answer_token_step = answer_pos_in_gen + 1

        mlp_neurons, attn_heads = _extract_neuron_tensors_at_position(
            mlp_store, head_store, answer_token_step, layer_indices
        )

        answer_residual: dict[int, torch.Tensor | None] = {}
        for layer_idx in layer_indices:
            acts = activations.get(layer_idx, [])  # one entry per forward pass
            if answer_token_step < len(acts):
                act = acts[answer_token_step]
                # Single-token pass -> shape [1, 1, hidden]; take the last (only) position.
                if act.dim() == 3:
                    act = act[0, -1, :]
                elif act.dim() == 2:
                    act = act[-1, :]
                answer_residual[layer_idx] = act.detach().clone().cpu()
            else:
                answer_residual[layer_idx] = None

        # Token embedding (layer_idx = -1) at the SAME step, so it refers to the same token.
        answer_embedding: torch.Tensor | None = None
        if emb_store is not None:
            emb_acts = emb_store.activations.get(-1, [])
            if answer_token_step < len(emb_acts):
                t = emb_acts[answer_token_step]
                if t.dim() == 3:
                    answer_embedding = t[0, -1, :].detach().clone().cpu()
                elif t.dim() == 2:
                    answer_embedding = t[-1, :].detach().clone().cpu()

        positions = find_positions_of_interest(llm.tokenizer, prompt)

        # Extract prompt-phase embedding tensor: step-0 output of embed_tokens [prompt_len, hidden].
        step0_embedding: torch.Tensor | None = None
        if emb_store is not None:
            emb_acts = emb_store.activations.get(-1, [])
            if emb_acts:
                t = emb_acts[0]
                step0_embedding = t[0] if t.dim() == 3 else t  # [prompt_len, hidden]

        geometry = _extract_all_activations(
            activations,
            mlp_store,
            head_store,
            layer_indices,
            positions,
            embedding_tensor=step0_embedding,
        )
    else:
        mlp_neurons = {i: None for i in layer_indices}
        attn_heads = {i: None for i in layer_indices}
        answer_residual = {i: None for i in layer_indices}
        answer_embedding = None
        geometry = {}

    mlp_store.remove_all_hooks()
    mlp_store.clear()
    head_store.remove_all_hooks()
    head_store.clear()
    if emb_store is not None:
        emb_store.remove_all_hooks()
        emb_store.clear()

    return {
        "text": generated_text,
        "completion": completion,
        "output_ids": output_ids if capture_geometry else None,
        "answer": {
            "position": answer_position_in_output,
            "token_id": answer_token_ids,
            "token": answer_token,
            "mlp_neurons": mlp_neurons,
            "attn_heads": attn_heads,
            "residual": answer_residual,
            "embedding": answer_embedding,
            "logits": answer_logits.cpu() if answer_logits is not None else None,
        },
        "geometry": geometry,
    }


def run(
    llm: LargeLanguageModel,
    dataset: PromptDataset,
    layers: dict[int, nn.Module],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Run inference capturing MLP intermediate, per-head, residual, and geometry activations.

    Args:
        llm: The large language model.
        dataset: Dataset whose `.prompts` is a list of prompt strings to run.
        layers: Transformer layer modules (used for residual stream extraction via extract_activations).
        max_new_tokens: Max tokens to generate per prompt.
        mlp_ablation: Optional dict mapping layer_idx to list of MLP neuron indices to zero out
            on every prompt.
        head_ablation: Optional dict mapping layer_idx to list of attention head indices to zero out
            on every prompt.
        capture_geometry: If True, capture step-0 activations at the answer token and at the
            find_positions_of_interest positions. Set to False during ablation sweeps to avoid
            storing per-feature geometry data.

    Returns:
        List of result dicts, one per prompt, each with mlp_neurons and attn_heads at the
        answer token plus geometry activations at the positions of interest.
    """
    # Resolve the concrete modules to hook once, up front (same for every prompt).
    neuron_layers = llm.get_mlp_neuron_layers(list(layers.keys()))
    attn_layers = llm.get_attn_output_layers(list(layers.keys()))

    # Process prompts one at a time, collecting one result dict per prompt.
    results: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(tqdm(dataset.prompts, desc="Running neuron inference")):
        # prompt_length (in tokens) marks the boundary between prompt and generated tokens.
        prompt_length = len(llm.tokenizer(prompt, return_tensors="pt")["input_ids"][0])

        run_result = _run_single_prompt(
            llm,
            prompt,
            prompt_length,
            layers,
            neuron_layers,
            attn_layers,
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
