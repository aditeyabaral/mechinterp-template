"""Plot the causal effect of zeroing each ablated neuron/head, from intervention .pt files.

Point it at a DIRECTORY of intervention runs (exactly like src/analysis/lasso.py takes --dir)
and it reads everything it needs from the files:

    python src/analysis/plot_ablations.py --dir <dir of intervention .pt files> [--output plots]

Each .pt in the directory is one intervention run -- one `main.py --intervention` invocation.
We call each run a SETTING. Every figure's cell/point value is accuracy_drop (baseline_accuracy -
ablated_accuracy). Un-measured cells are grey. From the runs it draws:

  HEATMAPS -- split by component type (MLP vs heads), with the layer axis in the shared per-layer
  colours (the same shape/colour scheme as the scatter):
    1. heatmap_<setting>_mlp.png    -- one per run: layers (rows) x MLP-neuron index (cols).
    2. heatmap_<setting>_heads.png  -- one per run: layers (rows) x attention-head index (cols).
    3. heatmap_settings_mlp.png     -- settings (rows) x flattened (layer, MLP neuron) (cols), width
                                       n_layers * num_mlp; read a column down to compare one neuron
                                       across runs. Layer blocks are divided and labelled L0, L1, ...
    4. heatmap_settings_heads.png   -- settings (rows) x flattened (layer, attention head) (cols),
                                       width n_layers * num_heads. (3 & 4 need >= 2 runs.)

  SCATTER MATRIX, one per PAIR of runs -> scatter_<A>_vs_<B>.png: one point per component,
  x = its accuracy_drop in run A, y = in run B (needs at least two runs). Each point's SHAPE is its
  type (circle = MLP neuron, star = attention head) and its COLOUR is its layer.

     WHY THIS IS INTERESTING: a component's accuracy_drop is how far the task accuracy falls when
     that one neuron/head is switched off (ablated) -- a big drop means the model leaned on it, so
     it's an "important" component for the task. Plotting run A's drops against run B's asks whether
     the SAME components carry the task in both settings. Points near the y=x line are equally
     load-bearing in both (a shared mechanism); points far off it matter in one setting but not the
     other (the model solves the two settings with different circuitry).

Nothing here needs editing to run -- the layer count, head count and MLP/head split (num_mlp) all
come from each file's saved metadata, and every .pt in --dir is used as a setting (labelled by its
filename, which is how it appears on the scatter axes and in the saved figure names).

WHAT YOU NEED FIRST (read this before running): each ablation entry must carry an `accuracy_drop`,
which intervention mode records via `is_correct` in src/main.py. That default scores a row correct
when the model's answer (row["result"]["answer"]["token"]) equals an "answer" stored in that
prompt's metadata, so the drops are only meaningful if your prompts carry a ground-truth "answer"
(see PromptDataset.generate_prompts in src/utils/dataset.py) AND `is_correct` matches your task.
Without that, every drop is 0 and the figures are blank. See the `load_ablations` docstring for the
exact fields expected.

WHAT YOU CAN CHANGE / HOW TO EXTEND:
  - The task-agnostic loaders live in src/utils/ablations.py (shared with src/analysis/plot_circuits.py); they
    know nothing about your task:
      load_ablations(pt_path)  -- read an intervention file's per-component summary + metadata (drops
                                  the heavy per-prompt rows so only the small summary stays in memory).
      infer_num_mlp(...)       -- where the head columns begin (the MLP|head split), read from the file.
      discover_settings/load_settings -- list + load every .pt in a --dir once.
      neuron_label(...)        -- a readable component tag, e.g. "L0MLP1" / "L2A2".
  - The reusable figure helpers:
      categorical_color_map(layers) -- the shared per-layer colours used across every figure (lives in
                                  src/utils/plotting.py, shared with src/analysis/plot_circuits.py).
      _save_heatmap(matrix, …) -- render ANY [rows x cols] matrix as a 0-centered diverging heatmap
                                  (supports per-layer coloured ticks and layer-block dividers).
  - `make_layer_component_heatmap` / `make_setting_component_heatmap` are the heatmaps, built on those
    helpers via the `_drop_matrix` / `_flat_drop_matrix` shapers. Copy one as a template for your own
    matrix-shaped views (e.g. average the drop across several runs, or restrict to a layer range).
  - `make_scatter` / `make_scatter_matrix` are the second figure: a per-component scatter of run A
    vs run B (shape = type, colour = layer), drawn for every pair of runs. Copy `make_scatter` as a
    template for other two-run comparisons (e.g. plot only heads, or size points by effect).
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: render to a file without a display server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402  (proxy handles for the scatter legends)

# This script lives in src/analysis/; put src/ on the import path so utils.* resolves when run
# as `python src/analysis/plot_ablations.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ablations import common_geometry, discover_settings, infer_num_mlp, load_settings, neuron_label  # noqa: E402
from utils.plotting import categorical_color_map, save_figure  # noqa: E402

# Heatmaps are split by component TYPE (the "shape" distinction from the scatter) and use the SAME
# per-layer colours (via categorical_color_map) on the layer axis, so the two figure families read alike.
TYPE_NOUN = {"mlp": "MLP neuron", "head": "attention head"}


def _drop_matrix(ablations: list[dict[str, Any]], layers: list[int], count: int, feat_type: str) -> np.ndarray:
    """[layers x within-layer index] matrix of accuracy_drop for ONE component type (NaN elsewhere).

    `count` is num_mlp (for feat_type "mlp") or num_heads (for "head"); columns are the within-type
    local index (a["local_idx"]). Un-ablated cells stay NaN so _save_heatmap draws them grey.
    """
    row_of = {layer: i for i, layer in enumerate(layers)}
    matrix = np.full((len(layers), count), np.nan, dtype=float)
    for a in ablations:
        if a["type"] == feat_type and a["layer_idx"] in row_of and 0 <= a["local_idx"] < count:
            matrix[row_of[a["layer_idx"]], a["local_idx"]] = a["accuracy_drop"]
    return matrix


def _flat_drop_matrix(
    ablations_by_setting: dict[str, list[dict[str, Any]]], layers: list[int], count: int, feat_type: str
) -> tuple[np.ndarray, list[str]]:
    """[settings x (layer, within-layer index)] matrix for ONE type; the (layer, idx) axis is flattened.

    Column for (layer, local_idx) is layer_position * count + local_idx, so each layer occupies a
    contiguous block of `count` columns and the width is len(layers) * count. Returns (matrix, the
    setting labels in row order).
    """
    layer_pos = {layer: i for i, layer in enumerate(layers)}
    settings = list(ablations_by_setting)
    matrix = np.full((len(settings), len(layers) * count), np.nan, dtype=float)
    for si, setting in enumerate(settings):
        for a in ablations_by_setting[setting]:
            if a["type"] == feat_type and a["layer_idx"] in layer_pos and 0 <= a["local_idx"] < count:
                matrix[si, layer_pos[a["layer_idx"]] * count + a["local_idx"]] = a["accuracy_drop"]
    return matrix, settings


def make_layer_component_heatmap(
    label: str, ablations: list[dict[str, Any]], layers: list[int], count: int, feat_type: str, out: Path
) -> None:
    """Layer (rows) x within-layer <type> index (cols) heatmap for ONE run -- plots 1 (mlp) and 2 (head).

    Rows are layers, coloured with the shared per-layer palette; cells are the accuracy_drop of that
    component in this run (grey where it wasn't ablated).

    Args:
        label: The setting's name (goes in the title).
        ablations: That setting's summary list from `load_ablations`.
        layers: The run's layers (rows), in order.
        count: num_mlp (feat_type "mlp") or num_heads (feat_type "head") -- the number of columns.
        feat_type: "mlp" or "head".
        out: Path to write the .png to.
    """
    matrix = _drop_matrix(ablations, layers, count, feat_type)
    layer_color = categorical_color_map(layers)
    noun = TYPE_NOUN[feat_type]
    _save_heatmap(
        matrix,
        row_labels=[f"L{layer}" for layer in layers],
        row_colors=[layer_color[layer] for layer in layers],
        xlabel=f"{noun} index (0..{count - 1})",
        ylabel="layer",
        title=f"{label}: {noun} ablation accuracy drop",
        out=out,
        col_tick_step=max(count // 16, 1),
    )


def make_setting_component_heatmap(
    ablations_by_setting: dict[str, list[dict[str, Any]]], layers: list[int], count: int, feat_type: str, out: Path
) -> None:
    """Setting (rows) x flattened (layer, <type> index) (cols) heatmap -- plots 3 (mlp) and 4 (head).

    One row per setting, so you can read a single component's accuracy_drop DOWN a column to compare
    it across runs. The x-axis flattens (layer, index) into len(layers) * count columns; each layer's
    block is separated by a divider and labelled with a layer-coloured "L{layer}" tick at its centre.

    Args:
        ablations_by_setting: {setting label: summary list}, one entry per run (rows, in this order).
        layers: The layers to lay out along x (usually the union across settings), in order.
        count: num_mlp (feat_type "mlp") or num_heads (feat_type "head") -- columns per layer block.
        feat_type: "mlp" or "head".
        out: Path to write the .png to.
    """
    matrix, settings = _flat_drop_matrix(ablations_by_setting, layers, count, feat_type)
    layer_color = categorical_color_map(layers)
    noun = TYPE_NOUN[feat_type]
    _save_heatmap(
        matrix,
        row_labels=settings,
        xlabel=f"(layer, {noun} index) flattened -- each layer block spans index 0..{count - 1}",
        ylabel="setting",
        title=f"{noun} ablation accuracy drop per setting",
        out=out,
        col_ticks=[i * count + count // 2 for i in range(len(layers))],  # one tick per layer block, centred
        col_tick_labels=[f"L{layer}" for layer in layers],
        col_tick_colors=[layer_color[layer] for layer in layers],
        col_dividers=[i * count for i in range(1, len(layers))],  # split the layer blocks
    )


def _save_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
    col_tick_step: int | None = None,
    col_ticks: list[int] | None = None,
    col_tick_labels: list[str] | None = None,
    col_tick_colors: list[Any] | None = None,
    col_dividers: list[int] | None = None,
    row_colors: list[Any] | None = None,
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
        col_tick_step: Label every Nth column index (used when col_ticks is not given).
        col_ticks: Explicit x positions to tick (pair with col_tick_labels); overrides col_tick_step.
        col_tick_labels: Labels for col_ticks (defaults to the positions themselves).
        col_tick_colors: One colour per x tick label (e.g. a per-layer colour).
        col_dividers: X positions to draw a vertical divider before (e.g. layer-block boundaries).
        row_colors: One colour per row tick label (e.g. a per-layer colour).
    """
    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size and np.any(finite) else 1.0
    vmax = vmax or 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")

    width = min(max(matrix.shape[1] / 80.0, 8.0), 40.0)
    fig, ax = plt.subplots(figsize=(width, 1.2 + 0.5 * matrix.shape[0]))
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.grid(False)  # the seaborn theme's gridlines would otherwise show over the heatmap cells
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="accuracy drop")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if row_colors is not None:
        for tick, color in zip(ax.get_yticklabels(), row_colors):
            tick.set_color(color)

    if col_ticks is not None:
        ax.set_xticks(col_ticks)
        ax.set_xticklabels(col_tick_labels if col_tick_labels is not None else col_ticks, fontsize=7, rotation=90)
    else:
        step = col_tick_step or max(matrix.shape[1] // 16, 1)
        xticks = list(range(0, matrix.shape[1], step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=6, rotation=90)
    if col_tick_colors is not None:
        for tick, color in zip(ax.get_xticklabels(), col_tick_colors):
            tick.set_color(color)

    for divider in col_dividers or []:
        ax.axvline(divider - 0.5, color="black", linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, out)


# Point STYLE for the scatter: shape encodes the component type, colour (via categorical_color_map)
# encodes its layer.
TYPE_MARKER = {"mlp": "o", "head": "*"}  # MLP neuron = circle, attention head = star


def make_scatter(
    label_a: str,
    ablations_a: list[dict[str, Any]],
    label_b: str,
    ablations_b: list[dict[str, Any]],
    out: Path,
) -> None:
    """Scatter of per-component accuracy_drop in setting A (x) vs setting B (y).

    accuracy_drop is baseline accuracy minus the accuracy after ablating (switching off) that one
    component -- both accuracies are the fraction of prompts answered correctly, so the drop is the
    AVERAGE change in performance across the run's prompts. A larger value means the model relied on
    that component more. Each point is one component (neuron or head), keyed by (layer_idx, type,
    local_idx) so the SAME component lines up across the two runs -- letting you see whether a
    component that mattered in setting A also mattered in setting B. (We key on type + local_idx, not
    the flattened feature_idx, so runs with different num_mlp -- e.g. different models -- still align.)

    Each point is styled to show WHICH component it is: its SHAPE is the component type (circle = MLP
    neuron, star = attention head) and its COLOUR is the layer (e.g. a layer-0 head is a star in the
    layer-0 colour, a layer-1 neuron a circle in the layer-1 colour). Two legends decode the shapes
    and the layer colours.

    A component ablated in only one setting is 0-filled on the other axis -- read as "that run didn't
    test it, so assume ~no effect there" -- so it lands on an axis. The dashed y=x line marks
    components equally important to both settings; points far off it matter more in one than the
    other (evidence the two settings use partly different internal mechanisms).

    Args:
        label_a: Name of setting A (x-axis).
        ablations_a: Setting A's summary list from `load_ablations`.
        label_b: Name of setting B (y-axis).
        ablations_b: Setting B's summary list from `load_ablations`.
        out: Path to write the .png to.
    """
    a = {(r["layer_idx"], r["type"], r["local_idx"]): r for r in ablations_a}
    b = {(r["layer_idx"], r["type"], r["local_idx"]): r for r in ablations_b}
    keys = sorted(set(a) | set(b))
    layers = sorted({key[0] for key in keys})
    layer_color = categorical_color_map(layers)

    fig, ax = plt.subplots(figsize=(8, 8))
    for key in keys:
        info = a.get(key) or b.get(key)  # label info from whichever run has this component
        x = a[key]["accuracy_drop"] if key in a else 0.0  # 0-fill: not tested in setting A
        y = b[key]["accuracy_drop"] if key in b else 0.0  # 0-fill: not tested in setting B
        ax.scatter(
            x,
            y,
            marker=TYPE_MARKER.get(info["type"], "o"),  # shape = component type
            color=layer_color[info["layer_idx"]],  # colour = layer
            s=45,
            alpha=0.8,
            edgecolors="none",
        )
        ax.annotate(neuron_label(info["layer_idx"], info["type"], info["local_idx"]), (x, y), fontsize=4, alpha=0.6)

    ax.axline((0, 0), slope=1, linestyle="--", color="grey")  # y = x: equally important to both
    ax.set_xlabel(f"avg accuracy drop ({label_a})")
    ax.set_ylabel(f"avg accuracy drop ({label_b})")
    ax.set_title(f"Per-component ablation effect: {label_a} vs {label_b}")

    # Two legends: one decoding the SHAPES (component type), one the COLOURS (layer). Both use grey/
    # neutral proxy markers for the type legend so it reads as "shape only", and per-layer colours for
    # the layer legend. add_artist keeps the first legend when the second is drawn.
    type_handles = [
        Line2D([], [], marker=TYPE_MARKER["mlp"], linestyle="none", color="grey", label="MLP neuron"),
        Line2D([], [], marker=TYPE_MARKER["head"], linestyle="none", color="grey", label="attention head"),
    ]
    layer_handles = [
        Line2D([], [], marker="s", linestyle="none", color=layer_color[layer], label=f"layer {layer}")
        for layer in layers
    ]
    ax.add_artist(ax.legend(handles=type_handles, title="component", loc="upper left", fontsize=8))
    ax.legend(handles=layer_handles, title="layer", loc="lower right", fontsize=8, ncol=max(1, len(layers) // 12))

    save_figure(fig, out)


def make_scatter_matrix(ablations_by_setting: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    """Draw one `make_scatter` per PAIR of settings (all combinations) into out_dir.

    Args:
        ablations_by_setting: {label: summary list from load_ablations}, one entry per setting.
        out_dir: Directory to write scatter_<A>_vs_<B>.png files into (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for label_a, label_b in itertools.combinations(ablations_by_setting, 2):  # every pair, in order
        out = out_dir / f"scatter_{label_a}_vs_{label_b}.png"
        make_scatter(label_a, ablations_by_setting[label_a], label_b, ablations_by_setting[label_b], out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot intervention/ablation effects (heatmaps + scatter matrix).")
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Directory of intervention .pt files from src/main.py (like src/analysis/lasso.py's --dir). Each "
        "file is one setting; layer/head/MLP counts are read from each file, so nothing else is needed.",
    )
    parser.add_argument("--output", "-o", default="plots", help="Directory to write the figures into.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    # Each .pt in --dir is one setting, labelled by its filename.
    settings = discover_settings(Path(args.dir))
    if not settings:
        raise SystemExit(f"No .pt files found in {args.dir}.")

    loaded = load_settings(settings)  # {label: (summary, metadata)}; each ~1GB file read once
    if not loaded:
        raise SystemExit(
            f"No usable intervention runs among {sorted(settings)} -- the .pt files may be incomplete "
            "or not intervention outputs (need an 'ablations' key). See the [skip] notes above."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    abl_by_setting = {label: ablations for label, (ablations, _) in loaded.items()}

    # (1 & 2) Per setting: one layer x MLP-neuron heatmap and one layer x attention-head heatmap.
    # Everything each needs (layers, head count, MLP/head split) comes from that file's own metadata,
    # with infer_num_mlp as a fallback for files saved before main.py recorded num_mlp_neurons.
    for label, (ablations, metadata) in loaded.items():
        layers = metadata.get("layer_indices") or sorted({a["layer_idx"] for a in ablations})
        num_heads = metadata.get("num_attention_heads", 0)
        num_mlp = infer_num_mlp(ablations, metadata)
        if num_mlp:
            make_layer_component_heatmap(label, ablations, layers, num_mlp, "mlp", out_dir / f"heatmap_{label}_mlp.png")
        else:
            print(
                f"  [skip {label} MLP heatmap]: can't tell where heads start (no num_mlp_neurons and "
                "no head was ablated)."
            )
        if num_heads:
            make_layer_component_heatmap(
                label, ablations, layers, num_heads, "head", out_dir / f"heatmap_{label}_heads.png"
            )

    # (3 & 4) Across settings (needs >= 2): setting x flattened (layer, component) heatmaps, one for
    # MLP neurons and one for heads. Use the union of layers and the largest MLP/head counts so runs
    # with different sizes still line up (absent cells stay grey), matching the scatter's 0-fill idea.
    if len(loaded) >= 2:
        all_layers, max_mlp, max_heads = common_geometry(loaded)
        if max_mlp:
            make_setting_component_heatmap(
                abl_by_setting, all_layers, max_mlp, "mlp", out_dir / "heatmap_settings_mlp.png"
            )
        if max_heads:
            make_setting_component_heatmap(
                abl_by_setting, all_layers, max_heads, "head", out_dir / "heatmap_settings_heads.png"
            )

        # Scatter for every pair of settings. Reuses the summaries already loaded.
        make_scatter_matrix(abl_by_setting, out_dir)
    else:
        print(
            f"  only one usable setting ({next(iter(loaded))}) -- add more intervention runs to --dir "
            "for the setting-vs-setting heatmaps and scatters."
        )

    print(f"Done. Figures in {out_dir}/")
