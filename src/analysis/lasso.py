"""Sparse linear analysis: find which neurons / attention heads are "important" from saved activations.

This is an OPTIONAL analysis step. It turns the activations captured by main.py into a
spec file (lasso_shortlist.json) that `python src/main.py --intervention lasso_shortlist.json` then uses
to ablate the flagged neurons/heads and measure their causal effect.

Pipeline (one .pt file at a time -- see main() -- so only one file's raw activations are ever
resident in memory; each file is freed before the next is loaded, since activation files are large.
Only the compact feature matrices, not the raw rows, accumulate across files):
  1. Load saved inference results (.pt files from main.py --capture-activations) -> one row per prompt,
     each carrying the MLP-neuron and attention-head activations captured at every named POSITION
     (every answer token, plus find_positions_of_interest's prompt positions).
  2. Optionally drop rows you don't want to learn from (keep_row) -- by default, keep only the
     prompts the model answered correctly.
  3. Group the kept rows into CONDITIONS (assign_condition). A "condition" is any subset of rows you want
     to analyse separately so you can compare them -- e.g. A vs B vs C. (Default: one condition.)
  4. For each condition, position and layer, build a feature matrix X of shape
     [N_rows, num_mlp + num_heads]: the MLP neuron activations concatenated with one L2 norm per
     attention head.
  5. Fit an L1-regularised ("lasso") logistic regression -- a PROBE -- to predict a class target
     (build_target) from X, on 80% of the rows, and score it by accuracy on the other 20%. L1 drives
     most coefficients to exactly zero, so the features with non-zero coefficients are the set the
     probe relies on -- the "important" neurons/heads for that condition and position. A probe that
     can't beat chance on the held-out rows has found nothing, and selects no features.
  6. Save lasso_shortlist.json: per layer, per condition and position, the important feature indices, their
     weights and the held-out accuracy, plus a top-level "conditions" block recording row counts.

A caveat worth knowing: a probe finds where information can be READ, which is not the same as what the
model USES. That is why the next step ablates the flagged components (main.py --intervention). For a
small model you can skip the probe entirely: `main.py --intervention all` ablates every component.

Three task-specific placeholders (search this file for TODO):
  - keep_row(row, metadata):                  whether to include a result at all (default: correct answers).
  - assign_condition(row, metadata):          which condition a result belongs to (default: "all").
  - build_target(row, metadata, position):    the class the probe at a position predicts (default: the
                                              next answer token the model writes).

Feature index convention (shared with main.py's intervention mode): indices 0..num_mlp-1 are MLP
neurons; indices num_mlp..num_mlp+num_heads-1 are attention heads.

Output format (lasso_shortlist.json), consumed by `main.py --intervention`:
    {
      "num_mlp_neurons": int, "num_heads": int,
      "conditions": {                          # one entry per condition|position (from assign_condition)
        "<condition>|<position>": {"n_rows": int}   # how many rows backed this condition's fits
      },
      "layers": {
        "<layer_idx>": {                       # only layers/conditions with enough rows appear
          "<condition>|<position>": {"features": [int, ...], "weight": [float, ...], "score": float,
                                     "chance": float, "p_value": float, "C": float, "n_train": int,
                                     "n_test": int}
        }
      }
    }
Each "features" entry is a feature index (see the convention above); the matching "weight" is its
largest absolute coefficient over the target's classes (its standardised effect size). L1 logistic
probes are NOT very sparse where the target is easy to read (they can keep most features, with tiny
weights), so rank features by weight rather than trusting the selection alone. "score" is the held-out
accuracy, "chance" the accuracy of always guessing the most common class, and "p_value" a one-sided
binomial test of score > chance; a fit with p >= 0.01 selects no features. At each layer, main.py
ablates the UNION of every condition's features. If you change this format, update the
reader in main.py's intervention block to match.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def keep_row(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Return True to include this result row in the analysis, False to drop it.

    Runs before assign_condition/build_target, so a dropped row never reaches the probe. The
    default keeps only the prompts the model answered correctly (if they carry a ground-truth
    "answer"; otherwise every row); override it to exclude other rows whose activations would
    mislead the probe.

    Args:
        row: One per-prompt result dict (from the saved "baseline" list).
        metadata: The file's metadata dict.

    Returns:
        True to keep the row, False to drop it.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): drop rows you don't want the probe to learn from. The default keeps only
    # the prompts the model answered CORRECTLY, since a wrong answer's activations encode whatever
    # (wrong) computation produced it, not the target behaviour -- pairing your target with those
    # activations would mislabel the probe's data.
    #
    # row["result"]["correct"] is set by inference.py with src/train/scoring.py's rule (the answer's
    # exact tokens, then <eos>) -- the same rule training and main.py score with. Prompts without a
    # ground-truth "answer" in their metadata can't be checked (correct is None), so they are kept.
    # ----------------------------------------------------------------------------------- #
    #
    return row["result"]["correct"] is not False


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
    # per label (and per position). Return None to drop a row. The default ("all") puts every row in one group, which
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


def build_target(row: dict[str, Any], metadata: dict[str, Any], position: str) -> int | None:
    """Return the class the probe at `position` should predict from the activations there, or None to skip.

    A probe looks for neurons/heads whose activations at a position tell the target apart, so choose
    a target that captures the behaviour you care about. The target is a CLASS (any int label), not a
    number on a line, so a feature counts even if the model doesn't encode the target linearly.

    The default is the next answer token the model writes: at answer token i (position
    "answer_<i>") it is answer token i + 1, and at a prompt position it is the FIRST answer token (the
    one the model will eventually compute). The answer's last token predicts <eos>, the same for
    every row, so it is skipped. Classes are token ids.

    Args:
        row: One per-prompt result dict.
        metadata: The file's metadata dict.
        position: A captured position name: one of find_positions_of_interest's, or "answer_<i>".

    Returns:
        An int class, or None to exclude this row at this position.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): return the class the probe at `position` should predict. Return None to
    # drop a row at that position. Available fields: row["prompt"], row["metadata"] (your ground
    # truth, e.g. the operands), row["result"]["answer"]["token"] / ["token_id"].
    # ----------------------------------------------------------------------------------- #
    #
    # Example (single-digit addition, prompts like "...\n7+5="): does the representation at the
    # "+" (find_positions_of_interest's "operator") already know whether the sum carries?
    #     if position != "operator":
    #         return None
    #     return int(int(row["metadata"]["answer"]) >= 10)
    #
    token_ids = row["result"]["answer"]["token_id"]
    if token_ids is None:
        return None
    ids = torch.as_tensor(token_ids).flatten().tolist()
    if position.startswith("answer_"):
        step = int(position[len("answer_") :]) + 1  # the next token written
    else:
        step = 0  # a prompt position: the first answer token
    return ids[step] if step < len(ids) else None


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


def _feature_vector(acts: dict[str, Any]) -> np.ndarray | None:
    """Build one feature row for a layer: [MLP neurons | per-head L2 norm], or None if unavailable.

    `acts` is one position's activations at one layer, as main.py saves them in
    row["result"]["geometry"][position][layer]. Each attention head outputs a [head_dim] vector; we
    summarise it by its L2 norm (one number per head, "how much this head fired") so heads and
    neurons live in the same feature matrix.
    """
    mlp = acts.get("mlp")
    heads = acts.get("heads")
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

    Mutates `buckets` in place -- buckets["<condition>|<position>"][layer] = {"X": [...], "y": [...]}
    (lists) -- so it can be called once per file to accumulate across files. Rows dropped by keep_row, or by
    assign_condition/build_target returning None, are skipped. Only the compact feature vectors are
    retained, so the caller can free this file's `rows` afterwards even though `buckets` lives on.
    """
    for row in rows:
        if not keep_row(row, metadata):
            continue
        condition = assign_condition(row, metadata)
        if condition is None:
            continue
        for position, by_layer in row["result"]["geometry"].items():
            target = build_target(row, metadata, position)
            if target is None:
                continue
            for layer_idx in layer_indices:
                feat = _feature_vector(by_layer.get(layer_idx, {}))
                if feat is None:
                    continue
                key = f"{condition}|{position}"
                layer_bucket = buckets.setdefault(key, {}).setdefault(layer_idx, {"X": [], "y": []})
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
                "y": np.array(d["y"], dtype=np.int64),
            }
    return matrices


def run_lasso(x: np.ndarray, y: np.ndarray, min_samples: int = 50, n_jobs: int = -1) -> dict[str, Any] | None:
    """Fit a cross-validated L1 logistic regression (a sparse probe) and return the important features.

    The rows are split 80/20 (fixed seed). The regularisation strength C is chosen by 3-fold
    cross-validation on the 80%, with the ONE-STANDARD-ERROR rule (the strongest regularisation whose
    accuracy is within one standard error of the best -- the sparsest fit that is about as good),
    then refitted on the 80% and scored by accuracy on the held-out 20%. X is standardised (on the
    80%) so coefficients are comparable and the L1 penalty treats every feature on the same scale.

    Args:
        x: Feature matrix [N, n_features].
        y: Target classes [N] (ints).
        min_samples: Skip (return None) if there are fewer than this many rows, or fewer than two classes
            (in all the rows, or in the 80% the probe is trained on).
        n_jobs: Processes for the cross-validation folds (1 when the caller runs several fits in parallel).

    Returns:
        {"features": [indices], "weight": [importances], "score", "chance", "p_value", "C", "n_train",
        "n_test"}: the features with a non-zero coefficient for any class (index ascending; none if the
        held-out accuracy isn't above chance) and, as importance, their largest absolute coefficient
        over the classes; or None if skipped. "score" is the held-out accuracy, "chance" that of always
        guessing the most common class of the training rows, "p_value" the one-sided binomial test of
        score > chance, and "C" the chosen inverse regularisation strength.
    """
    if x.shape[0] < min_samples or len(np.unique(y)) < 2:
        return None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if len(np.unique(y_train)) < 2:  # every row of the other classes landed in the 20%: nothing to fit
        return None
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    cv = LogisticRegressionCV(
        Cs=np.logspace(-3, 1, 6),
        l1_ratios=[1.0],  # pure L1
        solver="saga",
        cv=3,
        scoring="accuracy",
        max_iter=500,
        random_state=42,
        n_jobs=n_jobs,
        refit=False,
        use_legacy_attributes=False,
    ).fit(x_train, y_train)
    # One-standard-error rule. Where the target can't be read at all, every C scores about the same, so
    # this picks the sparsest fit instead of keeping every feature.
    scores = cv.scores_ if not isinstance(cv.scores_, dict) else next(iter(cv.scores_.values()))
    scores = np.asarray(scores).reshape(np.asarray(scores).shape[0], -1)  # [folds, Cs]
    mean, se = scores.mean(axis=0), scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    best = int(np.argmax(mean))
    c_value = float(cv.Cs_[np.flatnonzero(mean >= mean[best] - se[best])[0]])  # Cs_ is ascending
    probe = LogisticRegression(C=c_value, l1_ratio=1.0, solver="saga", max_iter=500, random_state=42)
    probe.fit(x_train, y_train)
    importance = np.abs(probe.coef_).max(axis=0)
    important = np.where(importance > 0)[0]
    classes, counts = np.unique(y_train, return_counts=True)
    most_common = classes[np.argmax(counts)]
    score = float(probe.score(x_test, y_test))
    chance = float(np.mean(y_test == most_common))
    # A probe that doesn't beat always guessing the most common class on the held-out rows (one-sided
    # binomial test, p < 0.01) has found nothing to read, so it selects no features; its score is
    # still reported.
    p_value = float(binomtest(round(score * len(y_test)), len(y_test), chance, alternative="greater").pvalue)
    if p_value >= 0.01:
        important = important[:0]
    return {
        "features": [int(i) for i in important],
        "weight": [float(importance[i]) for i in important],
        "score": score,
        "chance": chance,
        "p_value": p_value,
        "C": c_value,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


if __name__ == "__main__":
    # Load results, fit a probe per condition, position and layer, and write lasso_shortlist.json.
    parser = argparse.ArgumentParser(description="Find important neurons/heads with sparse (L1) probes.")
    parser.add_argument("--dir", "-d", type=str, required=True, help="Directory with .pt result files from main.py.")
    parser.add_argument(
        "--output", "-o", type=str, default="lasso_shortlist.json", help="Where to write the shortlist JSON."
    )
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
            f"No usable rows found in {args.dir} -- the .pt files may lack a 'baseline' key or activations "
            "(run main.py with --capture-activations), or every row was dropped by keep_row / assign_condition / "
            "build_target (all returning None/False; by default keep_row keeps only correct answers)."
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

    # lasso_shortlist.json (consumed by main.py): per layer, per condition, the important features and
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
    # The fits are independent, so they run in parallel, one process per fit (each fit single-process).
    jobs = [
        (layer_idx, condition, by_layer[layer_idx])
        for layer_idx in layer_indices
        for condition, by_layer in matrices.items()
        if layer_idx in by_layer
    ]
    start = time.time()
    fits = Parallel(n_jobs=-1)(delayed(run_lasso)(d["X"], d["y"], n_jobs=1) for _, _, d in jobs)
    print(f"{len(jobs)} fits in {time.time() - start:.0f}s")
    for layer_idx in layer_indices:
        layer_out: dict[str, Any] = {}
        for (job_layer, condition, _), flagged in zip(jobs, fits):
            if job_layer != layer_idx or flagged is None:
                continue
            layer_out[condition] = flagged  # {"features": [...], "weight": [...], "score": ..., ...}
            print(
                f"  layer {layer_idx} | '{condition}': {len(flagged['features'])} important features, "
                f"held-out accuracy {flagged['score']:.3f} (chance {flagged['chance']:.3f}, "
                f"{flagged['n_train']}+{flagged['n_test']} rows)"
            )
        analysis["layers"][str(layer_idx)] = layer_out

    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {args.output}")
    print(f"Run interventions with:  python src/main.py -m <model> --intervention {args.output}")
