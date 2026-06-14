"""Sparse linear analysis: find which neurons / attention heads are "important" from saved activations.

This is an OPTIONAL analysis step. It turns the activations captured by main.py into a
spec file (analysis.json) that `python src/main.py --intervention analysis.json` then uses
to ablate the flagged neurons/heads and measure their causal effect.

Pipeline:
  1. Load saved inference results (.pt files from main.py) -> one row per prompt, each carrying
     the MLP-neuron and attention-head activations captured at the answer token.
  2. Group rows into CONDITIONS (assign_condition). A "condition" is any subset of rows you want
     to analyse separately so you can compare them -- e.g. A vs B vs C. (Default: one condition.)
  3. For each condition and layer, build a feature matrix X of shape [N_rows, num_mlp + num_heads]:
     the MLP neuron activations concatenated with one L2 norm per attention head.
  4. Fit an L1-regularised Lasso to predict a scalar target (build_target). L1 drives most
     coefficients to exactly zero, so the features with non-zero coefficients are the small set
     the model actually relies on -- the "important" neurons/heads for that condition.
  5. Save analysis.json: per layer, per condition, the list of important feature indices.

Two task-specific placeholders (search this file for TODO):
  - assign_condition(row, metadata): which condition a result belongs to (default: "all").
  - build_target(row, metadata):     the scalar the Lasso predicts (default: the answer token id).

Feature index convention (shared with main.py's intervention mode): indices 0..num_mlp-1 are MLP
neurons; indices num_mlp..num_mlp+num_heads-1 are attention heads.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def assign_condition(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Return the name of the condition this result row belongs to, or None to skip it.

    A condition is just a label; rows sharing a label are analysed together. Returning
    several distinct labels across your rows lets you compare conditions (A vs B vs ...).

    Args:
        row: One per-prompt result dict (from the saved "result" list).
        metadata: The file's metadata dict.

    Returns:
        A condition name, or None to exclude this row from the analysis.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): split your rows into the conditions you want to COMPARE. Return a label
    # (any string) per row; rows with the same label are analysed together and the Lasso runs once
    # per label. Return None to drop a row. The default ("all") puts every row in one group, which
    # is fine if you just want a single set of important features.
    #
    # WHY: comparing conditions answers "are DIFFERENT neurons/heads important in situation A vs B?"
    # -- e.g. prompts the model got right vs wrong, or two kinds of question. You define the split
    # from whatever each row contains:
    #     row["prompt"]                    -> the prompt string you ran
    #     row["result"]["answer"]["token"] -> the answer string the model generated
    #     row["result"]["completion"]      -> the full generated text
    # ----------------------------------------------------------------------------------- #
    #
    # Example: compare prompts whose generated answer is a single digit vs anything else:
    #     ans = (row["result"]["answer"]["token"] or "").strip()
    #     return "single_digit" if ans.isdigit() and len(ans) == 1 else "other"
    #
    return "all"


def build_target(row: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    """Return the scalar value the Lasso should predict from the activations, or None to skip.

    The Lasso looks for neurons whose activations linearly predict this target, so choose a
    target that captures the behaviour you care about. The default (the generated answer token
    id) probes "which neurons carry the model's output".

    Args:
        row: One per-prompt result dict.
        metadata: The file's metadata dict.

    Returns:
        A float target, or None to exclude this row.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): return the scalar the Lasso should try to predict from the activations.
    # The Lasso then keeps only the few neurons/heads whose activations linearly predict it -- those
    # are the "important" ones for that quantity. So pick a target that captures the behaviour you
    # care about. Return None to drop a row.
    #
    # The default below uses the id of the answer's last token (a generic "which neurons carry the
    # output token" probe). Override it with something meaningful -- often a number you can compute
    # from the prompt or the answer. Available fields: row["prompt"],
    # row["result"]["answer"]["token"], row["result"]["answer"]["token_id"].
    # ----------------------------------------------------------------------------------- #
    #
    # Example (arithmetic): regress against the TRUE numeric answer, computed from the prompt's last
    # line "...\n7+5=" (this works even when the model is wrong, since it doesn't use the output):
    #     question = row["prompt"].rsplit("\n", 1)[-1].rstrip("=")   # "7+5"
    #     a, b = question.split("+")
    #     return float(int(a) + int(b))
    #
    answer = row["result"]["answer"]
    token_id = answer.get("token_id")
    if token_id is None:
        return None
    # token_id may be a tensor of several ids (if the answer span is >1 token, e.g. "42" -> ['4','2']).
    # We use the LAST one, because inference.py captures activations at the answer span's LAST token,
    # so the target and the captured activations describe the same token.
    ids = torch.as_tensor(token_id).flatten()
    if ids.numel() == 0:
        return None
    return float(ids[-1].item())


def _load_rows(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load every .pt result file in a directory into one list of rows, plus the metadata.

    Args:
        results_dir: Directory containing .pt files saved by main.py (normal mode).

    Returns:
        (rows, metadata): all per-prompt result rows concatenated, and the metadata of the
        first file (layer indices, num_attention_heads, ... are assumed consistent across files).
    """
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    pt_files = sorted(results_dir.glob("*.pt"))
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        # main.py normal mode saves {"result": [...], "metadata": {...}}.
        if "result" not in data:
            continue
        if not metadata:
            metadata = data.get("metadata", {})
        rows.extend(data["result"])
    return rows, metadata


def _feature_vector(answer: dict[str, Any], layer_idx: int) -> np.ndarray | None:
    """Build one feature row for a layer: [MLP neurons | per-head L2 norm], or None if unavailable.

    Each attention head outputs a [head_dim] vector; we summarise it by its L2 norm (one number
    per head, "how much this head fired") so heads and neurons live in the same feature matrix.
    """
    mlp = answer.get("mlp_neurons", {}).get(layer_idx)
    heads = answer.get("attn_heads", {}).get(layer_idx)
    if mlp is None or heads is None:
        return None
    mlp = torch.as_tensor(mlp).float()  # [num_mlp]
    heads = torch.as_tensor(heads).float()  # [num_heads, head_dim]
    head_norms = heads.norm(dim=-1) if heads.dim() == 2 else heads  # [num_heads]
    return np.concatenate([mlp.numpy(), head_norms.numpy()])


def build_condition_matrices(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    layer_indices: list[int],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Group rows by condition and build a per-layer (X, y) feature matrix for each.

    Returns condition_name -> layer_idx -> {"X": [N, num_mlp + num_heads], "y": [N]}.
    """
    # buckets[condition][layer] = {"X": [...], "y": [...]} (lists, converted to arrays at the end)
    buckets: dict[str, dict[int, dict[str, list]]] = {}

    for row in rows:
        condition = assign_condition(row, metadata)
        if condition is None:
            continue
        target = build_target(row, metadata)
        if target is None:
            continue
        answer = row["result"]["answer"]
        for layer_idx in layer_indices:
            feat = _feature_vector(answer, layer_idx)
            if feat is None:
                continue
            layer_bucket = buckets.setdefault(condition, {}).setdefault(layer_idx, {"X": [], "y": []})
            layer_bucket["X"].append(feat)
            layer_bucket["y"].append(target)

    # Convert the accumulated lists into numpy arrays.
    matrices: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for condition, by_layer in buckets.items():
        matrices[condition] = {}
        for layer_idx, d in by_layer.items():
            matrices[condition][layer_idx] = {
                "X": np.array(d["X"], dtype=np.float32),
                "y": np.array(d["y"], dtype=np.float32),
            }
    return matrices


def run_lasso(x: np.ndarray, y: np.ndarray, min_samples: int = 10) -> list[int] | None:
    """Fit a cross-validated Lasso and return the indices of the important (non-zero) features.

    Both X and y are standardised first so coefficients are comparable and the L1 penalty
    treats every feature on the same scale.

    Args:
        x: Feature matrix [N, n_features].
        y: Target vector [N].
        min_samples: Skip (return None) if there are fewer than this many rows.

    Returns:
        Sorted list of feature indices with a non-zero coefficient, or None if skipped.
    """
    if x.shape[0] < min_samples:
        return None
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = (y - y.mean()) / (y.std() + 1e-8)
    lasso = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=-1).fit(x_scaled, y_scaled)
    return [int(i) for i in np.where(np.abs(lasso.coef_) > 0)[0]]


def main() -> None:
    """Load results, run the Lasso per condition per layer, and write analysis.json."""
    parser = argparse.ArgumentParser(description="Find important neurons/heads via sparse (Lasso) regression.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory with .pt result files from main.py.")
    parser.add_argument("--output", "-o", type=str, default="analysis.json", help="Where to write the analysis JSON.")
    args = parser.parse_args()

    rows, metadata = _load_rows(Path(args.dir))
    if not rows:
        raise SystemExit(f"No usable .pt result files found in {args.dir}")

    # layer indices and head count come from the metadata main.py saved.
    layer_indices = metadata.get("layer_indices", [])
    num_heads = metadata.get("num_attention_heads", 0)
    print(f"Loaded {len(rows)} rows | layers={layer_indices} | num_heads={num_heads}")

    matrices = build_condition_matrices(rows, metadata, layer_indices)
    print(f"Conditions: {sorted(matrices)}")

    # Figure out num_mlp from a feature vector (its length minus the head columns).
    num_mlp = 0
    for by_layer in matrices.values():
        for d in by_layer.values():
            if d["X"].shape[0] > 0:
                num_mlp = d["X"].shape[1] - num_heads
                break
        if num_mlp:
            break

    # analysis.json: per layer, per condition, the important feature indices (consumed by main.py).
    analysis: dict[str, Any] = {"num_mlp_neurons": num_mlp, "num_heads": num_heads, "layers": {}}
    for layer_idx in layer_indices:
        conditions_out: dict[str, Any] = {}
        for condition, by_layer in matrices.items():
            d = by_layer.get(layer_idx)
            if d is None:
                continue
            important = run_lasso(d["X"], d["y"])
            if important is None:
                continue
            conditions_out[condition] = {"important": important}
            print(f"  layer {layer_idx} | condition '{condition}': {len(important)} important features")
        analysis["layers"][str(layer_idx)] = {"conditions": conditions_out}

    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {args.output}")
    print(f"Run interventions with:  python src/main.py -m <model> --intervention {args.output}")


if __name__ == "__main__":
    main()
