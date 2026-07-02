"""Sparse linear analysis: find which neurons / attention heads are "important" from saved activations.

This is an OPTIONAL analysis step. It turns the activations captured by main.py into a
spec file (analysis.json) that `python src/main.py --intervention analysis.json` then uses
to ablate the flagged neurons/heads and measure their causal effect.

Pipeline (one .pt file at a time -- see main() -- so only one file's raw activations are ever
resident in memory; each file is freed before the next is loaded, since activation files are large.
Only the compact feature matrices, not the raw rows, accumulate across files):
  1. Load saved inference results (.pt files from main.py) -> one row per prompt, each carrying
     the MLP-neuron and attention-head activations captured at the answer token.
  2. Optionally drop rows you don't want to learn from (keep_row) -- e.g. keep only the prompts the
     model answered correctly.
  3. Group the kept rows into CONDITIONS (assign_condition). A "condition" is any subset of rows you want
     to analyse separately so you can compare them -- e.g. A vs B vs C. (Default: one condition.)
  4. For each condition and layer, build a feature matrix X of shape [N_rows, num_mlp + num_heads]:
     the MLP neuron activations concatenated with one L2 norm per attention head.
  5. Fit an L1-regularised Lasso to predict a scalar target (build_target). L1 drives most
     coefficients to exactly zero, so the features with non-zero coefficients are the small set
     the model actually relies on -- the "important" neurons/heads for that condition.
  6. Save analysis.json: per layer, per condition, the important feature indices and their Lasso
     coefficients (weights), plus a top-level "conditions" block recording each condition's row count.

Three task-specific placeholders (search this file for TODO):
  - keep_row(row, metadata):         whether to include a result at all (default: keep every row).
  - assign_condition(row, metadata): which condition a result belongs to (default: "all").
  - build_target(row, metadata):     the scalar the Lasso predicts (default: the answer token id).

Feature index convention (shared with main.py's intervention mode): indices 0..num_mlp-1 are MLP
neurons; indices num_mlp..num_mlp+num_heads-1 are attention heads.

Output format (analysis.json), consumed by `main.py --intervention`:
    {
      "num_mlp_neurons": int, "num_heads": int,
      "conditions": {                          # one entry per condition (from assign_condition)
        "<condition>": {"n_rows": int}         # how many rows backed this condition's fits
      },
      "layers": {
        "<layer_idx>": {                       # only layers/conditions with enough rows appear
          "<condition>": {"features": [int, ...], "weight": [float, ...]}
        }
      }
    }
Each "features" entry is a feature index (see the convention above); the matching "weight" is its
signed Lasso coefficient (effect size + direction of correlation with the target). At each layer,
main.py ablates the UNION of every condition's features. If you change this format, update the
reader in main.py's intervention block to match.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def keep_row(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Return True to include this result row in the analysis, False to drop it.

    Runs before assign_condition/build_target, so a dropped row never reaches the Lasso. The
    default keeps every row (identical to having no filter); override it to exclude rows whose
    activations would mislead the regression.

    Args:
        row: One per-prompt result dict (from the saved "baseline" list).
        metadata: The file's metadata dict.

    Returns:
        True to keep the row, False to drop it.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): drop rows you don't want the Lasso to learn from. The most common filter:
    # keep only the prompts the model answered CORRECTLY, since a wrong answer's activations at the
    # answer token encode whatever (wrong) computation produced it, not the target behaviour --
    # pairing your target with those activations would mislabel the regression data.
    #
    # main.py already defines this exact correctness check (its is_correct) -- import and call it so
    # the two stay in sync, rather than duplicating the comparison here:
    #     from main import is_correct
    #     return is_correct(row)
    # (is_correct assumes your prompts carry a ground-truth "answer" in their metadata -- see
    # PromptDataset.generate_prompts in src/utils/dataset.py.)
    # ----------------------------------------------------------------------------------- #
    #
    return True


def assign_condition(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    """Return the name of the condition this result row belongs to, or None to skip it.

    A condition is just a label; rows sharing a label are analysed together. Returning
    several distinct labels across your rows lets you compare conditions (A vs B vs ...).

    Args:
        row: One per-prompt result dict (from the saved "baseline" list).
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


def _load_file(pt_file: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Load one .pt result file, or None if it is not a main.py normal-mode result file.

    Loaded one file at a time (see main()) so only one file's raw activations are resident at
    once; the caller extracts the compact feature vectors it needs and frees this file's rows
    before loading the next.

    Args:
        pt_file: A .pt file saved by main.py (normal mode).

    Returns:
        (rows, metadata), or None if the file has no "baseline" key (main.py saves the unablated
        run under "baseline" in both normal and intervention mode).
    """
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    if "baseline" not in data:
        return None
    return data["baseline"], data.get("metadata", {})


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


def accumulate_features(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    layer_indices: list[int],
    buckets: dict[str, dict[int, dict[str, list]]],
) -> None:
    """Extract one file's kept rows into shared per-condition/per-layer feature buckets.

    Mutates `buckets` in place -- buckets[condition][layer] = {"X": [...], "y": [...]} (lists) --
    so it can be called once per file to accumulate across files. Rows dropped by keep_row, or by
    assign_condition/build_target returning None, are skipped. Only the compact feature vectors are
    retained, so the caller can free this file's `rows` afterwards even though `buckets` lives on.
    """
    for row in rows:
        if not keep_row(row, metadata):
            continue
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


def finalize_matrices(
    buckets: dict[str, dict[int, dict[str, list]]],
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Convert accumulated feature lists into numpy arrays, once every file has been processed.

    Returns condition_name -> layer_idx -> {"X": [N, num_mlp + num_heads], "y": [N]}.
    """
    matrices: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for condition, by_layer in buckets.items():
        matrices[condition] = {}
        for layer_idx, d in by_layer.items():
            matrices[condition][layer_idx] = {
                "X": np.array(d["X"], dtype=np.float32),
                "y": np.array(d["y"], dtype=np.float32),
            }
    return matrices


def run_lasso(x: np.ndarray, y: np.ndarray, min_samples: int = 10) -> dict[str, list] | None:
    """Fit a cross-validated Lasso and return the important (non-zero-coefficient) features.

    Both X and y are standardised first so coefficients are comparable and the L1 penalty
    treats every feature on the same scale.

    Args:
        x: Feature matrix [N, n_features].
        y: Target vector [N].
        min_samples: Skip (return None) if there are fewer than this many rows.

    Returns:
        {"features": [indices], "weight": [coefficients]} (same order, index ascending) for the
        features with a non-zero coefficient, or None if skipped. The signed coefficient is the
        feature's standardised effect size, so it carries the direction of correlation with the
        target too, not just whether the feature matters.
    """
    if x.shape[0] < min_samples:
        return None
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = (y - y.mean()) / (y.std() + 1e-8)
    lasso = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=-1).fit(x_scaled, y_scaled)
    important = np.where(np.abs(lasso.coef_) > 0)[0]
    return {"features": [int(i) for i in important], "weight": [float(lasso.coef_[i]) for i in important]}


if __name__ == "__main__":
    # Load results, run the Lasso per condition per layer, and write analysis.json.
    parser = argparse.ArgumentParser(description="Find important neurons/heads via sparse (Lasso) regression.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory with .pt result files from main.py.")
    parser.add_argument("--output", "-o", type=str, default="analysis.json", help="Where to write the analysis JSON.")
    args = parser.parse_args()

    pt_files = sorted(Path(args.dir).glob("*.pt"))
    if not pt_files:
        raise SystemExit(f"No .pt result files found in {args.dir}")

    # Process one .pt file at a time: extract its compact feature vectors into shared buckets, then
    # free the file's raw rows before loading the next -- so only one file's activations are ever
    # resident, while the small feature matrices accumulate across files.
    # buckets[condition][layer] = {"X": [...], "y": [...]}; layer/head counts come from the metadata
    # main.py saved (assumed consistent across files, since they share the same model/--layers config).
    buckets: dict[str, dict[int, dict[str, list]]] = {}
    layer_indices: list[int] = []
    num_heads = 0
    for pt_file in pt_files:
        loaded = _load_file(pt_file)
        if loaded is None:
            continue
        rows, metadata = loaded
        if not layer_indices:
            layer_indices = metadata.get("layer_indices", [])
            num_heads = metadata.get("num_attention_heads", 0)
        accumulate_features(rows, metadata, layer_indices, buckets)
        print(f"{pt_file.name}: {len(rows)} rows")
        del rows, loaded  # free this file's raw activations before loading the next

    matrices = finalize_matrices(buckets)
    if not matrices:
        raise SystemExit(
            f"No usable rows found in {args.dir} -- the .pt files may lack a 'baseline' key, or every "
            "row was dropped by keep_row / assign_condition / build_target (all returning None/False)."
        )
    print(f"Conditions: {sorted(matrices)} | layers={layer_indices} | num_heads={num_heads}")

    # Figure out num_mlp from a feature vector (its length minus the head columns).
    num_mlp = 0
    for by_layer in matrices.values():
        for d in by_layer.values():
            if d["X"].shape[0] > 0:
                num_mlp = d["X"].shape[1] - num_heads
                break
        if num_mlp:
            break

    # analysis.json (consumed by main.py): per layer, per condition, the important features and
    # their Lasso coefficients. A top-level "conditions" block records how many rows backed each
    # condition, so a short feature list can be told apart from "too little data to fit reliably".
    analysis: dict[str, Any] = {
        "num_mlp_neurons": num_mlp,
        "num_heads": num_heads,
        "conditions": {
            condition: {"n_rows": max((d["X"].shape[0] for d in by_layer.values()), default=0)}
            for condition, by_layer in matrices.items()
        },
        "layers": {},
    }
    for layer_idx in layer_indices:
        layer_out: dict[str, Any] = {}
        for condition, by_layer in matrices.items():
            d = by_layer.get(layer_idx)
            if d is None:
                continue
            flagged = run_lasso(d["X"], d["y"])
            if flagged is None:
                continue
            layer_out[condition] = flagged  # {"features": [...], "weight": [...]}
            print(f"  layer {layer_idx} | condition '{condition}': {len(flagged['features'])} important features")
        analysis["layers"][str(layer_idx)] = layer_out

    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {args.output}")
    print(f"Run interventions with:  python src/main.py -m <model> --intervention {args.output}")
