"""Plot the causal effect of zeroing each ablated neuron/head, from an intervention .pt file.

OPTIONAL, and a STARTING POINT rather than a finished tool: it ships one ready-to-run figure plus
three reusable helpers you build your own figures on top of. Run it as:

    python src/plot_ablations.py --file <intervention .pt> --num-mlp-neurons <d_mlp>

  heatmap_ablations.png: layers (rows) x within-layer component index (cols), cell = accuracy_drop
    (baseline_accuracy - ablated_accuracy). Columns 0..num_mlp-1 are MLP neurons; the rest are
    attention heads. Components that weren't ablated are left blank (grey).

WHAT YOU NEED FIRST (read this before running): each ablation entry must carry an `accuracy_drop`,
which intervention mode records via `is_correct` in src/main.py. That default scores a row correct
when the model's answer (row["result"]["answer"]["token"]) equals an "answer" stored in that
prompt's metadata, so the drops are only meaningful if your prompts carry a ground-truth "answer"
(see PromptDataset.generate_prompts in src/utils/dataset.py) AND `is_correct` matches your task.
Without that, every drop is 0 and the heatmap is blank. See the `load_ablations` docstring for the
exact fields expected.

WHAT YOU CAN CHANGE / HOW TO EXTEND:
  - The three task-agnostic helpers are the reusable core; they know nothing about your task:
      load_ablations(pt_path)  -- read an intervention file's per-component summary (drops the heavy
                                  per-prompt rows so only the small {layer, feature, drop, ...} list
                                  stays in memory).
      neuron_label(...)        -- a readable component tag, e.g. "L0MLP1" / "L2A2".
      _save_heatmap(matrix, …) -- render ANY [rows x cols] matrix as a 0-centered diverging heatmap.
  - `make_heatmap` is the one figure built on those helpers. Copy it as a template for your own
    matrix-shaped views (e.g. average the drop across several files, or restrict to attention heads).
  - A common next figure is a SCATTER comparing two runs/conditions (which components matter in
    setting A vs setting B). A worked, commented `make_scatter` example sits just above `__main__`
    below -- uncomment it, then call it from `__main__` with two intervention files.
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: render to a file without a display server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402


def load_ablations(pt_path: Path) -> list[dict[str, Any]] | None:
    """Load one intervention file's per-component ablation summary, dropping the heavy result rows.

    Args:
        pt_path: A .pt file written by `src/main.py --intervention` (has an "ablations" key).

    Returns:
        A list of {layer_idx, type, local_idx, feature_idx, accuracy_drop} dicts -- one per ablated
        component -- or None if the file is absent, empty (a still-running / failed run), or not an
        intervention file. The large per-prompt `result` rows are discarded here, so only the small
        summary stays in memory (each .pt can be ~1GB).

    Raises:
        KeyError: if the ablations don't record `accuracy_drop` (see this module's docstring -- you
            need to add accuracy scoring to intervention mode first).
    """
    if not pt_path.exists() or pt_path.stat().st_size == 0:
        print(f"  [skip] {pt_path.name}: missing or empty (incomplete run)")
        return None
    loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "ablations" not in loaded:
        print(f"  [skip] {pt_path.name}: no 'ablations' key (not an intervention file)")
        return None
    if loaded["ablations"] and "accuracy_drop" not in loaded["ablations"][0]:
        raise KeyError(
            f"{pt_path.name}: ablations have no 'accuracy_drop'. Intervention mode must score "
            "accuracy and store accuracy_drop per ablation -- see the module docstring."
        )
    summary = [
        {
            "layer_idx": a["layer_idx"],
            "type": a["type"],
            "local_idx": a["local_idx"],
            "feature_idx": a["feature_idx"],
            "accuracy_drop": a["accuracy_drop"],
        }
        for a in loaded["ablations"]
    ]
    del loaded  # free the heavy result rows we don't need before anything else loads
    return summary


def neuron_label(layer_idx: int, feat_type: str, local_idx: int) -> str:
    """Human-readable component tag, e.g. 'L0MLP1' (MLP neuron) or 'L2A2' (attention head)."""
    kind = "MLP" if feat_type == "mlp" else "A"
    return f"L{layer_idx}{kind}{local_idx}"


def make_heatmap(
    ablations: list[dict[str, Any]],
    layer_indices: list[int],
    num_mlp: int,
    num_heads: int,
    out: Path,
) -> None:
    """Layer x within-layer-component heatmap of accuracy_drop from one intervention file.

    Each column is a component within a layer: indices 0..num_mlp-1 are MLP neurons, the rest are
    attention heads (this is the same feature-index convention main.py / lasso.py use). A component
    that wasn't ablated is left as NaN, drawn grey, so you only see the ones the run actually tested.

    Args:
        ablations: The summary list from `load_ablations`.
        layer_indices: The layers captured in the run (the heatmap's rows), from the file metadata.
        num_mlp: Number of MLP neurons per layer (d_mlp) -- where the head columns begin.
        num_heads: Number of attention heads per layer.
        out: Path to write the .png to.
    """
    num_per_layer = num_mlp + num_heads
    rows = sorted(set(layer_indices))
    row_of = {layer: i for i, layer in enumerate(rows)}

    matrix = np.full((len(rows), num_per_layer), np.nan, dtype=float)
    for a in ablations:
        if a["layer_idx"] in row_of and 0 <= a["feature_idx"] < num_per_layer:
            matrix[row_of[a["layer_idx"]], a["feature_idx"]] = a["accuracy_drop"]

    _save_heatmap(
        matrix,
        row_labels=[f"L{layer}" for layer in rows],
        xlabel=f"component index  [MLP 0..{num_mlp - 1} | heads {num_mlp}..{num_per_layer - 1}]",
        ylabel="layer",
        title="Ablation accuracy drop per component",
        out=out,
        col_tick_step=max(num_per_layer // 16, 1),
        mlp_boundary=num_mlp,
    )


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
    col_tick_step: int,
    mlp_boundary: int | None = None,
    col_tick_labels: list[str] | None = None,
) -> None:
    """Render a (rows x many-cols) value matrix with a 0-centered diverging colormap via imshow.

    NaN cells are masked and drawn grey (use them for "not measured"). The colormap is symmetric
    around 0 so positive and negative values are comparable at a glance.

    Args:
        matrix: The [n_rows, n_cols] values to plot (may contain NaN).
        row_labels: One label per row.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Figure title.
        out: Path to write the .png to.
        col_tick_step: Label every Nth column index (ignored if col_tick_labels is given).
        mlp_boundary: If set, draw a vertical divider before this column (the MLP|head split).
        col_tick_labels: Explicit per-column labels (for narrow plots); overrides col_tick_step.
    """
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size and np.any(finite) else 1.0
    vmax = vmax or 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")

    width = min(max(matrix.shape[1] / 80.0, 8.0), 40.0)
    fig, ax = plt.subplots(figsize=(width, 1.2 + 0.5 * matrix.shape[0]))
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="accuracy drop")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if col_tick_labels is not None:
        ax.set_xticks(range(len(col_tick_labels)))
        ax.set_xticklabels(col_tick_labels, fontsize=7, rotation=90)
    else:
        xticks = list(range(0, matrix.shape[1], col_tick_step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=6, rotation=90)

    if mlp_boundary is not None:
        ax.axvline(mlp_boundary - 0.5, color="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------------------- #
# EXAMPLE -- extend with your own figure. A scatter comparing two intervention runs (e.g.
# two conditions, or two models): each point is one component, x = its accuracy_drop in run
# A, y = in run B. A component flagged in only one run is 0-filled on the other axis, so it
# lands on an axis; the y=x line marks components equally important to both. Uncomment, then
# call it from `__main__` below with two .pt files.
#
#     def make_scatter(file_a: Path, file_b: Path, out: Path) -> None:
#         """Scatter of per-component accuracy_drop in run A (x) vs run B (y)."""
#         # Key each component by (layer, feature) so the same component lines up across runs.
#         a = {(r["layer_idx"], r["feature_idx"]): r for r in (load_ablations(file_a) or [])}
#         b = {(r["layer_idx"], r["feature_idx"]): r for r in (load_ablations(file_b) or [])}
#         fig, ax = plt.subplots(figsize=(8, 8))
#         for key in sorted(set(a) | set(b)):
#             info = a.get(key) or b.get(key)                   # label info from whichever run has it
#             x = a[key]["accuracy_drop"] if key in a else 0.0  # 0-fill: not flagged in run A
#             y = b[key]["accuracy_drop"] if key in b else 0.0  # 0-fill: not flagged in run B
#             ax.scatter(x, y, s=28, alpha=0.7)
#             ax.annotate(neuron_label(info["layer_idx"], info["type"], info["local_idx"]), (x, y), fontsize=5)
#         ax.axline((0, 0), slope=1, linestyle="--", color="grey")  # y = x: equally important to both
#         ax.set_xlabel("accuracy drop (run A)")
#         ax.set_ylabel("accuracy drop (run B)")
#         fig.savefig(out, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         print(f"  saved {out}")
# ----------------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot intervention/ablation effects (accuracy-drop heatmap).")
    parser.add_argument("--file", "-f", required=True, help="An intervention .pt file from src/main.py.")
    parser.add_argument(
        "--num-mlp-neurons",
        type=int,
        required=True,
        help="MLP neurons per layer (d_mlp) -- where head columns begin. It's lasso.json's "
        "num_mlp_neurons, or your model's intermediate size.",
    )
    parser.add_argument("--output", "-o", default="heatmap_ablations.png", help="Where to write the figure.")
    args = parser.parse_args()

    pt_path = Path(args.file)
    ablations = load_ablations(pt_path)
    if not ablations:
        raise SystemExit(f"No usable ablations in {pt_path}")

    # Layer indices and head count come from the metadata main.py saved alongside the ablations.
    metadata = torch.load(pt_path, map_location="cpu", weights_only=False).get("metadata", {})
    layer_indices = metadata.get("layer_indices", sorted({a["layer_idx"] for a in ablations}))
    num_heads = metadata.get("num_attention_heads", 0)

    sns.set_theme(style="whitegrid")
    make_heatmap(ablations, layer_indices, args.num_mlp_neurons, num_heads, Path(args.output))
    print(f"Done. Figure at {args.output}")
