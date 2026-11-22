"""Inference loop: generate on each prompt and capture / ablate internal activations via TransformerLens.

This file works on any decoder-only model out of the box (it talks to the model only through the
TransformerLens `TransformerBridge` from src/model.py). Two functions are the task-specific
override points you may want to customise (both have sensible defaults and `TODO` markers):
  - find_answer_span(generated_text):        which span of the generation is "the answer".
  - find_positions_of_interest(model, prompt, metadata): which prompt token positions to capture.
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
            hook_pos_embed               positional embedding (pre-block 0; not in rotary models) [batch, pos, d_model]
            blocks.0.hook_resid_pre      token_embed + pos_embed (block 0's input) [batch, pos, d_model]
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

from train.scoring import is_correct
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
        generated_text: The decoded text the model generated after the prompt, up to (not including)
            the first special token it generated, such as the <eos> that ends its answer.

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


def find_positions_of_interest(
    model: TransformerBridge, prompt: str, metadata: dict[str, Any]
) -> dict[str, int | None]:
    """Return extra PROMPT token positions to capture activations at (for --capture-activations).

    WHY this exists: by default we capture activations only at the answer's tokens. But often you
    also want activations at specific INPUT positions -- e.g. for "7+5=" you might study the
    representations sitting on the "7", the "5", and the "+". This function names those positions
    so inference.py records residual/MLP/head/embedding activations there too, stored under
    result["activations"][name]. The default returns nothing, which is fine if you only care about
    the answer.

    Args:
        model: The TransformerLens bridge. Useful helpers: `model.to_str_tokens(prompt)` returns
            the prompt split into per-token strings (so the i-th entry is the string of token i),
            which makes it easy to find a token's index.
        prompt: The full prompt string.
        metadata: The prompt's metadata (see PromptDataset.generate_prompts). Positions often need
            the ground truth, e.g. to tell which number in the prompt is the first operand.

    Returns:
        dict mapping a name you choose to a token index within the prompt (or None to skip it).
        Don't use names starting with "answer_": those are the answer's own positions.
        IMPORTANT: index the tokens exactly as `model.to_tokens(prompt)` produces them, which is how
        inference.py tokenizes prompts. That includes a BOS token at index 0 if, and only if, your
        tokenizer adds one (load_model in src/model.py sets this up to match training), and
        `model.to_str_tokens(prompt)` follows the same rule, so its indices always line up.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): return the prompt token positions you want to study, e.g.
    #     {"operator": 5, "first_operand": 3}. Leave it returning {} to capture only the answer.
    # ----------------------------------------------------------------------------------- #
    #
    # Example (a single-digit addition task with prompts like "7+5="), capturing the '+' operator
    # and the '=' sign by scanning the per-token strings of the prompt:
    #
    #     str_tokens = model.to_str_tokens(prompt)  # ["7", "+", "5", "=", ...]
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


def _cut_at_special_token(model: TransformerBridge, generated_ids: torch.Tensor) -> torch.Tensor:
    """Return the generated tokens before the first special token (<eos>, <pad>, ...).

    The model ends its answer by writing <eos>, and TransformerLens decodes special tokens as text
    ("<eos>", "<|endoftext|>", ...), so anything from the first special token on is not the answer.
    """
    special_ids = set(model.tokenizer.all_special_ids)
    for idx, token_id in enumerate(generated_ids.tolist()):
        if token_id in special_ids:
            return generated_ids[:idx]
    return generated_ids


def _locate_answer(
    model: TransformerBridge,
    prompt_length: int,
    output_ids: torch.Tensor,
    logits: torch.Tensor | None,
) -> tuple[int | None, torch.Tensor | None, str | None, torch.Tensor | None]:
    """Find the answer span in the generated text and report which token to read activations from.

    Steps:
      1. Decode the generated tokens (everything after the prompt, up to the first special token
         such as <eos>) to text.
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
        logits: Logits over the full output sequence, shape [1, seq_len, vocab_size], or None if
            they weren't computed (then answer_logits is None).

    Returns:
        (answer_position, answer_token_ids, answer_token, answer_logits), or all None if no answer
        span could be located. answer_position is the ABSOLUTE index (into output_ids) of the
        answer span's last token.
    """
    # The answer ends at the first special token (<eos>, <pad>, ...): decoding keeps special tokens, so
    # without this cut a correct "12<eos>" would be read as the answer "12<eos>".
    generated_ids = _cut_at_special_token(model, output_ids[prompt_length:])
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
    if logits is not None and 0 <= pred_position < logits.shape[1]:
        answer_logits = logits[0, pred_position].detach().cpu().clone()  # a copy, not a view (see _read_activations_at)
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


def _read_activations_at(
    cache: dict[str, torch.Tensor],
    layer_indices: list[int],
    pos: int | None,
) -> dict[str, Any]:
    """Read MLP-neuron, per-head, residual, and embedding activations at ONE token position.

    Shared by `_activations_at_positions` (one call per named position) and `_run_single_prompt`
    (one call for the answer position), so the cache-indexing logic lives in exactly one place.

    Args:
        cache: The activation cache from model.run_with_cache over the full output sequence.
        layer_indices: Layer indices to read per-layer activations from.
        pos: The absolute token position to read, or None (or out of range) to get all-None
            placeholders -- e.g. when the answer couldn't be located, or a prompt position
            couldn't be parsed.

    Returns:
        {"mlp_neurons": {layer_idx: Tensor | None}, "attn_heads": {layer_idx: Tensor | None},
         "residual": {layer_idx: Tensor | None}, "token_embedding": Tensor | None,
         "pos_embedding": Tensor | None, "resid_pre": Tensor | None}. "resid_pre" is block 0's
         actual input (blocks.0.hook_resid_pre, i.e. token_embedding + pos_embedding), read
         directly from the cache rather than summed so it stays correct for any architecture.
         "pos_embedding" is None for models without a learned positional embedding (e.g. Llama,
         whose rotary embedding acts inside attention).
    """
    embed = cache["hook_embed"][0]  # [seq_len, d_model]
    in_range = pos is not None and 0 <= pos < embed.shape[0]
    has_pos_embed = "hook_pos_embed" in cache

    # .clone(): on the CPU, .cpu() returns a VIEW of the cache tensor, and torch.save would then write
    # the whole tensor behind it (every position) instead of this one position.
    return {
        "mlp_neurons": {
            i: cache[f"blocks.{i}.mlp.hook_post"][0, pos].detach().cpu().clone() if in_range else None
            for i in layer_indices
        },
        "attn_heads": {
            i: cache[f"blocks.{i}.attn.hook_z"][0, pos].detach().cpu().clone() if in_range else None
            for i in layer_indices
        },
        "residual": {
            i: cache[f"blocks.{i}.hook_resid_post"][0, pos].detach().cpu().clone() if in_range else None
            for i in layer_indices
        },
        "token_embedding": embed[pos].detach().cpu().clone() if in_range else None,
        "pos_embedding": cache["hook_pos_embed"][0, pos].detach().cpu().clone() if in_range and has_pos_embed else None,
        "resid_pre": cache["blocks.0.hook_resid_pre"][0, pos].detach().cpu().clone() if in_range else None,
    }


def _activations_at_positions(
    cache: dict[str, torch.Tensor],
    layer_indices: list[int],
    positions: dict[str, int | None],
) -> dict[str, dict[int, dict[str, torch.Tensor | None]]]:
    """Slice residual / MLP / head / embedding activations at named positions out of the cache.

    Args:
        cache: The activation cache from model.run_with_cache over the full output sequence.
        layer_indices: Layer indices to extract from.
        positions: Mapping {name: token_index}: find_positions_of_interest's prompt positions and the
            answer's tokens ("answer_0", ...); any index may be None.

    Returns:
        dict mapping each position name to layer_idx ->
        {"residual": Tensor | None, "mlp": Tensor | None, "heads": Tensor | None}.
        layer_idx=-1 holds the pre-block-0 representation: "resid_pre" is block 0's actual input
        (blocks.0.hook_resid_pre = token_embedding + pos_embedding), "mlp"/"heads"=None (nothing
        runs before block 0), plus the decomposed "token_embedding" and "pos_embedding" so you can
        separate content from position.
    """
    activations: dict[str, dict] = {}
    for name, pos in positions.items():
        act = _read_activations_at(cache, layer_indices, pos)
        activations[name] = {
            # layer_idx = -1 holds the pre-block-0 representation.
            -1: {
                "resid_pre": act["resid_pre"],
                "mlp": None,
                "heads": None,
                "token_embedding": act["token_embedding"],
                "pos_embedding": act["pos_embedding"],
            },
            **{
                layer_idx: {
                    "residual": act["residual"][layer_idx],
                    "mlp": act["mlp_neurons"][layer_idx],
                    "heads": act["attn_heads"][layer_idx],
                }
                for layer_idx in layer_indices
            },
        }

    return activations


def _run_single_prompt(
    model: TransformerBridge,
    prompt: str,
    metadata: dict[str, Any],
    prompt_length: int,
    layer_indices: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_activations: bool = True,
    capture_logits: bool = False,
) -> dict[str, Any]:
    """Run a single prompt through the model, capturing activations at the answer and named positions.

    Flow:
      1. Generate a completion (with any ablation hooks active, so the generation reflects the
         knocked-out components).
      2. Re-run the FULL output sequence once -- with the same hooks -- to get logits, and (when
         capturing activations) a cache of every layer/position activation. Skipped when nothing is
         captured: the answer itself comes from the generated tokens alone.
      3. Locate the answer span and read activations at each of its tokens (and at the prompt
         positions of find_positions_of_interest).

    Args:
        model: The TransformerLens bridge.
        prompt: Full prompt string.
        metadata: The prompt's metadata (see PromptDataset), passed to find_positions_of_interest.
        prompt_length: Number of tokens in the prompt (as model.to_tokens counts them).
        layer_indices: Layer indices to capture from.
        max_new_tokens: Maximum tokens to generate.
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} to zero out.
        head_ablation: Optional dict {layer_idx: [head indices]} to zero out.
        capture_activations: If True, read per-position activations (MLP neurons, heads, residual,
            token/positional embedding) and store output_ids. False stores only the generated answer.
        capture_logits: If True, store the logits that predict each answer token and the <eos>
            after it. They are often most of a file's size, and they can be rebuilt from the last
            layer's residual stream: model.unembed(model.ln_final(resid)).

    Returns:
        Dict with keys: text, completion, output_ids (None when capture_activations=False), correct
        (whether the generation is exactly the answer's tokens then <eos>, see src/train/scoring.py;
        None without a ground-truth "answer" in the metadata), and an answer sub-dict (position,
        token_id, token, mlp_neurons, attn_heads, residual, token_embedding, pos_embedding,
        resid_pre, logits) plus an activations sub-dict (empty when capture_activations=False). positions (empty
        unless capture_activations or capture_logits) maps each captured position's name to its
        absolute token index: the find_positions_of_interest names, plus "answer_0", "answer_1", ...
        for the answer's tokens in the order they were written. logits (empty unless capture_logits)
        maps "answer_0", ..., and "eos" to the logits that predicted that token, i.e. those computed
        one position earlier.
    """
    ablation_hooks = build_ablation_hooks(mlp_ablation, head_ablation)
    prompt_tokens = model.to_tokens(prompt)  # [1, prompt_length], tokenized as in training

    # 1. Generate with ablation applied. Greedy decoding (do_sample=False) so runs are
    #    deterministic. We pass already-tokenized input (built above) so token indices stay
    #    aligned with prompt_length.
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
    #    activation cache when we capture activations. One pass covers every position, including the
    #    answer's last token. Nothing to capture -> no second pass.
    logits, cache = None, None
    if capture_activations:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits, cache = model.run_with_cache(out_tokens)
    elif capture_logits:
        with model.hooks(fwd_hooks=ablation_hooks):
            logits = model.run_with_hooks(out_tokens, return_type="logits")

    # 3. Locate the answer span's last token and the absolute position to read activations at.
    answer_position, answer_token_ids, answer_token, answer_logits = _locate_answer(
        model, prompt_length, output_ids, logits
    )

    # Whether the generation is right: exactly the answer's tokens, then <eos> (src/train/scoring.py).
    # None when the prompt carries no ground-truth "answer" in its metadata.
    correct = None
    if metadata.get("answer") is not None:
        answer_ids = model.tokenizer(metadata["answer"], add_special_tokens=False)["input_ids"]
        correct = is_correct(output_ids[prompt_length:].tolist(), answer_ids, model.tokenizer.eos_token_id)

    # The absolute position of every answer token, named by the order they were written: answer_0 is
    # the first token of the answer, answer_1 the second, and so on (the last is answer_position).
    answer_positions: dict[str, int] = {}
    if answer_position is not None:
        n_answer = len(answer_token_ids)
        for i in range(n_answer):
            answer_positions[f"answer_{i}"] = answer_position - n_answer + 1 + i
    positions: dict[str, int | None] = {}
    if capture_activations or capture_logits:
        positions = {**find_positions_of_interest(model, prompt, metadata), **answer_positions}

    # The logits that predicted each answer token, and the <eos> after the answer: in an
    # autoregressive model the prediction for position p is computed at position p - 1.
    position_logits: dict[str, torch.Tensor] = {}
    if capture_logits and answer_position is not None:
        predicted = {name: pos - 1 for name, pos in answer_positions.items()}
        predicted["eos"] = answer_position
        position_logits = {name: logits[0, pos].detach().cpu().clone() for name, pos in predicted.items()}

    # 4. Read all activations at the answer token's position (same position across every tensor, so
    #    they all describe the same token), plus the activations at the prompt positions and at every
    #    answer token. These are independent: the PROMPT positions are known from the prompt alone,
    #    so they're captured even when the answer couldn't be located (e.g. garbage generation,
    #    which leaves no answer tokens); only the answer-position activations need answer_position
    #    (a None pos yields all-None placeholders).
    if capture_activations and cache is not None:
        answer_act = _read_activations_at(cache, layer_indices, answer_position)
        mlp_neurons = answer_act["mlp_neurons"]
        attn_heads = answer_act["attn_heads"]
        answer_residual = answer_act["residual"]
        answer_token_embedding = answer_act["token_embedding"]
        answer_pos_embedding = answer_act["pos_embedding"]
        answer_resid_pre = answer_act["resid_pre"]

        activations = _activations_at_positions(cache, layer_indices, positions)
    else:
        mlp_neurons = {i: None for i in layer_indices}
        attn_heads = {i: None for i in layer_indices}
        answer_residual = {i: None for i in layer_indices}
        answer_token_embedding = None
        answer_pos_embedding = None
        answer_resid_pre = None
        activations = {}

    return {
        "text": generated_text,
        "completion": completion,
        "output_ids": output_ids if capture_activations else None,
        "correct": correct,
        "answer": {
            "position": answer_position,
            "token_id": answer_token_ids,
            "token": answer_token,
            "mlp_neurons": mlp_neurons,
            "attn_heads": attn_heads,
            "residual": answer_residual,
            "token_embedding": answer_token_embedding,
            "pos_embedding": answer_pos_embedding,
            "resid_pre": answer_resid_pre,
            "logits": answer_logits if capture_logits else None,
        },
        "activations": activations,
        "positions": positions,
        "logits": position_logits,
    }


def run(
    model: TransformerBridge,
    dataset: PromptDataset,
    layers: list[int],
    max_new_tokens: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
    capture_activations: bool = True,
    capture_logits: bool = False,
) -> list[dict[str, Any]]:
    """Run inference over a dataset, capturing MLP-neuron, per-head, residual, and embedding activations.

    Args:
        model: The TransformerLens bridge.
        dataset: Dataset whose `.prompts` is a list of {"prompt": str, "metadata": dict} entries
            to run (see src/utils/dataset.py).
        layers: Layer indices to capture from.
        max_new_tokens: Max tokens to generate per prompt.
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} to zero on every prompt.
        head_ablation: Optional dict {layer_idx: [head indices]} to zero on every prompt.
        capture_activations: If True, capture activations at every answer token and at the
            find_positions_of_interest positions. False stores only the generated answers (much smaller).
        capture_logits: If True, also store the logits that predict each answer token and the
            <eos> after it (see _run_single_prompt).

    Returns:
        List of result dicts, one per prompt (in dataset order), each with keys "prompt",
        "prompt_length", "metadata" (that prompt's ground truth -- see src/utils/dataset.py), and
        "result" (see _run_single_prompt for that sub-dict's schema).
    """
    results: list[dict[str, Any]] = []
    for entry in tqdm(dataset.prompts, desc="Running neuron inference"):
        prompt = entry["prompt"]
        metadata = entry.get("metadata", {})
        # prompt_length (in tokens, as model.to_tokens counts them) marks the boundary between prompt
        # and generated tokens.
        prompt_length = model.to_tokens(prompt).shape[1]

        run_result = _run_single_prompt(
            model,
            prompt,
            metadata,
            prompt_length,
            layers,
            max_new_tokens,
            mlp_ablation=mlp_ablation,
            head_ablation=head_ablation,
            capture_activations=capture_activations,
            capture_logits=capture_logits,
        )

        results.append(
            {
                "prompt": prompt,
                "prompt_length": prompt_length,
                "metadata": metadata,
                "result": run_result,
            }
        )

    return results


# ---------------------------------------------------------------------------------------------- #
# Teacher-forced scoring, used by main.py's intervention sweep.
#
# Generating an answer token by token takes one forward pass PER TOKEN, per prompt -- slow when an
# ablation sweep repeats it for hundreds of components. Teacher forcing scores a prompt in ONE pass
# instead: we feed the prompt followed by its CORRECT answer and <eos>, and read the model's top
# prediction at each answer step. The position before the answer predicts the first answer token,
# each answer token predicts the next one, and the last predicts <eos>.
#
# Why this gives the same verdict as generation: greedy generation writes the right answer exactly
# when, at every step, the top token is the right next token, given the right answer so far -- which
# is what teacher forcing checks. So "every step right" == "greedy generation writes the answer and
# then <eos>" (src/train/scoring.py's rule), as long as --max-new-tokens leaves room for them.
# main.py checks this on the baseline and reports any prompt where the two disagree. As a bonus,
# teacher forcing tells you WHICH answer step an ablation breaks.
#
# Steps are columns of a matrix: "step0", "step1", ... (the answer's tokens, in the order they are
# written), then "eos". NOT_APPLICABLE marks steps a prompt doesn't have (a 2-token answer has no
# step2).
# ---------------------------------------------------------------------------------------------- #
NOT_APPLICABLE = 255


def _step_names(n_steps: int) -> list[str]:
    """The step names for answers of up to n_steps tokens: step0, ..., step{n_steps-1}, then eos."""
    return [f"step{i}" for i in range(n_steps)] + ["eos"]


def build_teacher_forced_batches(
    model: TransformerBridge, dataset: PromptDataset, batch_size: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Tokenize every prompt with its target answer and <eos>, grouped into batches of one sequence length.

    Prompts are grouped by (prompt length, answer length), so no batch needs padding: batched results
    equal unbatched ones, and every sequence in a batch has its answer steps at the same positions.
    Built once, then reused for every ablation.

    Args:
        model: The TransformerLens bridge (its tokenizer and device are used).
        dataset: The prompts; each entry's metadata must carry "answer", the ground-truth answer
            text (the same field generation is scored against).
        batch_size: Maximum sequences per forward pass.

    Returns:
        (batches, names): names are the step columns (see _step_names). Each batch is
        {"indices": prompt indices [B], "tokens": [B, prompt_length + answer_length + 1] on the
        model's device, "prompt_length": int, "columns": the column of each answer step, in order}.
    """
    eos_id = model.tokenizer.eos_token_id
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for i, entry in enumerate(dataset.prompts):
        answer = entry.get("metadata", {}).get("answer")
        if answer is None:
            raise ValueError("Teacher-forced scoring needs a ground-truth 'answer' in every prompt's metadata")
        prompt_tokens = model.to_tokens(entry["prompt"])[0].cpu()  # tokenized as in training
        # The answer as training tokenized it: on its own, without special tokens (src/train/train_model.py).
        answer_ids = model.tokenizer(answer, add_special_tokens=False)["input_ids"]
        tokens = torch.cat([prompt_tokens, torch.tensor([*answer_ids, eos_id])])
        groups.setdefault((len(prompt_tokens), len(answer_ids)), []).append((i, tokens))

    longest = max((n_answer for _, n_answer in groups), default=0)
    names = _step_names(longest)
    batches: list[dict[str, Any]] = []
    for (prompt_length, n_answer), members in sorted(groups.items()):
        columns = list(range(n_answer)) + [names.index("eos")]
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            batches.append(
                {
                    "indices": torch.tensor([i for i, _ in chunk]),
                    "tokens": torch.stack([tokens for _, tokens in chunk]).to(model.cfg.device),
                    "prompt_length": prompt_length,
                    "columns": columns,
                }
            )
    return batches, names


@torch.inference_mode()
def score_teacher_forced(
    model: TransformerBridge,
    batches: list[dict[str, Any]],
    n_prompts: int,
    n_columns: int,
    mlp_ablation: dict[int, list[int]] | None = None,
    head_ablation: dict[int, list[int]] | None = None,
) -> torch.Tensor:
    """Whether the model's top token is right at every answer step, with optional ablation, in one pass per batch.

    Args:
        model: The TransformerLens bridge.
        batches: From build_teacher_forced_batches.
        n_prompts: Number of prompts in the dataset the batches were built from.
        n_columns: Number of step columns (len of the names build_teacher_forced_batches returned).
        mlp_ablation: Optional dict {layer_idx: [neuron indices]} to zero out (as in build_ablation_hooks).
        head_ablation: Optional dict {layer_idx: [head indices]} to zero out.

    Returns:
        uint8 tensor [n_prompts, n_columns]: 1 if that step's top token is right, 0 if not,
        NOT_APPLICABLE if the prompt has no such step.
    """
    correct = torch.full((n_prompts, n_columns), NOT_APPLICABLE, dtype=torch.uint8)
    with model.hooks(fwd_hooks=build_ablation_hooks(mlp_ablation, head_ablation)):
        for batch in batches:
            tokens, start, columns = batch["tokens"], batch["prompt_length"], batch["columns"]
            n_steps = len(columns)
            logits = model(tokens, return_type="logits")  # [B, seq_len, vocab]
            # The prediction at position p is for token p + 1, so positions start-1 .. start+n_steps-2
            # predict the answer tokens and <eos> at positions start .. start+n_steps-1.
            predicted = logits[:, start - 1 : start - 1 + n_steps].argmax(dim=-1)
            right = (predicted == tokens[:, start : start + n_steps]).to(torch.uint8).cpu()
            correct[batch["indices"][:, None], torch.tensor(columns)[None, :]] = right
    return correct


def whole_answer_correct(correct: torch.Tensor) -> torch.Tensor:
    """[n_prompts] bool: every answer step the prompt has is right, <eos> included (score_teacher_forced's output).

    The same rule as src/train/scoring.py's is_correct: the answer's exact tokens, then <eos>.
    """
    return ((correct == 1) | (correct == NOT_APPLICABLE)).all(dim=1)


def accuracy(right: torch.Tensor) -> float:
    """The share of True values, from exact counts divided in double precision.

    Exact counts make equal results give EXACTLY equal accuracies, so rankings of components by
    their drop have no spurious ties or orderings (a float32 mean can break ties arbitrarily).
    """
    return right.sum().item() / right.numel() if right.numel() else float("nan")


def step_accuracy(correct: torch.Tensor, names: list[str]) -> dict[str, float]:
    """Accuracy at each answer step, over the prompts that have that step (score_teacher_forced's output)."""
    result = {}
    for column, name in enumerate(names):
        values = correct[:, column]
        present = values != NOT_APPLICABLE
        if present.any():
            result[name] = accuracy(values[present] == 1)
    return result
