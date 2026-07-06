"""Draw the surviving ablation circuit as a node-and-edge diagram, from intervention .pt files.

Point it at a DIRECTORY of intervention runs (exactly like src/lasso.py and src/plot_ablations.py
take --dir) and it reads everything it needs from the files:

    python src/plot_circuits.py --dir <dir of intervention .pt files> [--output plots] [--percentile 99]

Each .pt in the directory is one intervention run -- one `main.py --intervention` invocation. As in
src/plot_ablations.py we call each run a SETTING (each is one .pt, labelled by its filename). This
script turns every setting into its own CIRCUIT DIAGRAM image, plus one combined image.

WHAT A CIRCUIT DIAGRAM IS: a graph with one COLUMN of nodes per layer. Every column holds ALL the
components of that layer -- the MLP neurons first (feature indices 0..num_mlp-1), then the attention
heads (num_mlp..num_mlp+num_heads-1). A node's position is FIXED by (layer, feature index), so the
same neuron sits at the same spot in every image and you can compare settings at a glance.

WHAT "SURVIVING" MEANS: for each layer we take the accuracy_drop of every component that run ablated
and keep only the ones in the top-p PERCENTILE (default 99; try 95 with --percentile if the surviving
set is too sparse to see). accuracy_drop is baseline_accuracy - ablated_accuracy, so a survivor is a
component the model leaned on heavily in that setting. EVERY neuron/head is drawn (all at the same
size, so the whole population is visible); survivors are told apart by COLOUR (blue = MLP neuron,
orange = attention head) against the grey non-survivors, which makes the SPARSITY of the circuit
visible -- a few load-bearing components picked out against the full population.

WHAT THE EDGES MEAN: within a setting we connect every surviving component of one layer to every
surviving component of the next (a complete bipartite link between adjacent survivor sets). The
reasoning is deliberately simple: our ablation experiment showed BOTH endpoints are causally needed
for the correct answer in that setting, so we treat them as co-participating in one circuit. (This is
a coarse "they both matter" edge, not a measured connection -- see HOW TO EXTEND to make it stronger.)

THE FIGURES: S + 1 separate images --
  * one image PER setting (circuit_<setting>.png), its edges drawn in black;
  * one COMBINED image (circuit_combined.png) overlaying every setting's edges at once, each setting
    in its OWN colour, so you can see where the settings route through the same components and where
    they diverge.

SHARED NODES: components that survive in EVERY setting are the shared-circuit candidates -- the parts
that look causally necessary regardless of the setting. They are ringed with a bold red border in all
panels (including the combined one), so they stand out wherever they appear.

TERMINAL OUTPUT: alongside the images, it also prints each setting's survivors layer by layer with
their accuracy drops (highest first) -- the same information the diagram encodes, in a form you can
read, copy, or grep without opening the figures.

Nothing here needs editing to run: the layer count, head count and MLP/head split (num_mlp) all come
from each file's saved metadata (via the shared loaders in src/utils/ablations.py), and every .pt in
--dir is used, labelled by its filename.

WHAT YOU NEED FIRST: like src/plot_ablations.py, every ablation entry must carry an `accuracy_drop`
(intervention mode records it via `is_correct` in src/main.py). If your prompts have no ground-truth
"answer" the drops are all 0, the percentile filter keeps nothing, and the circuit is empty. See the
src/plot_ablations.py module docstring for the exact fields and how to add accuracy scoring.

WHAT YOU CAN CHANGE / HOW TO EXTEND:
  - `compute_survivors` is the whole selection rule (top-p percentile of accuracy_drop per layer).
    Swap it for a fixed threshold, a top-k, or "drop above X" to change what counts as a survivor.
  - The edge rule lives in `_edges`: right now it is "connect all survivors of adjacent layers". To
    make it a real connection you would replace it with an activation/attribution measurement between
    the two components (this file intentionally does NOT compute any correlation -- everything comes
    straight from the ablation .pt files).
  - `draw_setting_panel` / `draw_combined_panel` are the two panel kinds; copy one to add another view
    (e.g. only heads, or only the shared sub-circuit).
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: render to a file without a display server
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402  (proxy handles for the legends)
from matplotlib.patches import PathPatch  # noqa: E402  (curved edges)
from matplotlib.path import Path as MplPath  # noqa: E402  (curved edges)

# The task-agnostic loaders live in utils/ablations.py, shared with src/plot_ablations.py; the figure
# helpers live in utils/plotting.py (also shared).
from utils.ablations import common_geometry, discover_settings, load_settings, neuron_label  # noqa: E402
from utils.plotting import categorical_color_map, save_figure  # noqa: E402

# Node styling: EVERY neuron/head is drawn at the same size so the whole population is visible;
# survivors are told apart by COLOUR (type) against grey non-survivors, not by size.
MLP_COLOR = "steelblue"
HEAD_COLOR = "darkorange"
BACKGROUND_COLOR = "lightgrey"
SHARED_RING_COLOR = "red"  # bold border on components that survive in EVERY setting
NODE_SIZE = 18  # one size for all nodes -- survivors stand out by colour + edges, not by being bigger


def compute_survivors(ablations: list[dict[str, Any]], layers: list[int], percentile: float) -> dict[int, set[int]]:
    """Per layer, the set of component feature indices in the top-p percentile of accuracy_drop.

    The percentile is taken over the components that run actually ablated in that layer (the only
    ones with a measured accuracy_drop), matching the "grey = not ablated" convention elsewhere.

    Args:
        ablations: One run's summary list from `load_ablations` (each has layer_idx, feature_idx,
            accuracy_drop).
        layers: The layers to build survivor sets for, in column order.
        percentile: Keep components with accuracy_drop >= this percentile of the layer's drops
            (e.g. 99 keeps the top 1%). Try 95 if the surviving set is too sparse.

    Returns:
        {layer_idx: set of surviving feature indices}. Layers with no ablated components map to an
        empty set (so the diagram simply has no survivors there).
    """
    drops: dict[int, dict[int, float]] = defaultdict(dict)  # drops[layer_idx][feature_idx] = accuracy_drop
    for a in ablations:
        drops[a["layer_idx"]][a["feature_idx"]] = a["accuracy_drop"]

    survivors: dict[int, set[int]] = {}
    for layer in layers:
        neuron_drops = drops.get(layer, {})
        if not neuron_drops:
            survivors[layer] = set()
            continue
        threshold = float(np.percentile(np.array(list(neuron_drops.values())), percentile))
        survivors[layer] = {feat for feat, drop in neuron_drops.items() if drop >= threshold}
    return survivors


def print_survivors(label: str, ablations: list[dict[str, Any]], survivors: dict[int, set[int]], num_mlp: int) -> None:
    """Print one setting's surviving components and their accuracy drops (deltas), layer by layer.

    The same information the diagram encodes, but as text: for each layer it lists the survivors
    (the top-percentile components from `compute_survivors`) sorted by accuracy_drop descending, each
    tagged L{layer}{MLP|A}{local} (via `neuron_label`) and annotated with its drop. Layers with no
    survivors are skipped.

    Args:
        label: The setting label (its .pt filename stem).
        ablations: That setting's summary list from `load_ablations` (carries the accuracy_drop).
        survivors: That setting's survivor sets from `compute_survivors`.
        num_mlp: The MLP|head split -- feat < num_mlp is an MLP neuron, the rest are attention heads.
    """
    drops: dict[int, dict[int, float]] = defaultdict(dict)  # drops[layer_idx][feature_idx] = accuracy_drop
    for a in ablations:
        drops[a["layer_idx"]][a["feature_idx"]] = a["accuracy_drop"]

    print(f"  [{label}] surviving components per layer (tag=accuracy_drop, highest first):")
    printed_any = False
    for layer in sorted(survivors):
        feats = survivors[layer]
        if not feats:
            continue
        printed_any = True
        ranked = sorted(feats, key=lambda f: drops[layer].get(f, 0.0), reverse=True)
        tags = [
            f"{neuron_label(layer, 'mlp' if f < num_mlp else 'head', f if f < num_mlp else f - num_mlp)}"
            f"={drops[layer].get(f, 0.0):+.3f}"
            for f in ranked
        ]
        print(f"    L{layer}: " + ", ".join(tags))
    if not printed_any:
        print("    (none -- the percentile filter kept nothing; try a lower --percentile)")


def compute_shared(survivors_per_setting: dict[str, dict[int, set[int]]], layers: list[int]) -> dict[int, set[int]]:
    """Per layer, the components that survive in EVERY setting -- the shared-circuit candidates.

    Args:
        survivors_per_setting: {setting label: survivor sets from `compute_survivors`}.
        layers: The layers to intersect over.

    Returns:
        {layer_idx: set of feature indices present in all settings' survivor sets for that layer}.
    """
    shared: dict[int, set[int]] = {}
    for layer in layers:
        per_setting = [survivors[layer] for survivors in survivors_per_setting.values()]
        shared[layer] = set.intersection(*per_setting) if per_setting else set()
    return shared


def _build_graph(layers: list[int], n_total: int) -> tuple[nx.Graph, dict[str, tuple[int, int]]]:
    """The full node set (all components in every layer) and their fixed positions.

    Node id is "L{layer}_{feat}"; position is (column of the layer, -feat) so layers march left to
    right and feature index runs top to bottom -- identical in every panel for easy comparison.
    """
    graph = nx.Graph()
    positions: dict[str, tuple[int, int]] = {}
    for col, layer in enumerate(layers):
        for feat in range(n_total):
            node = f"L{layer}_{feat}"
            graph.add_node(node)
            positions[node] = (col, -feat)
    return graph, positions


def _node_styles(
    survivors: dict[int, set[int]], shared: dict[int, set[int]], layers: list[int], num_mlp: int, n_total: int
) -> tuple[list[Any], list[int], list[Any], list[float]]:
    """(colours, sizes, border colours, border widths) for every node, in `_build_graph` order.

    Every node is the SAME size; survivors are coloured by type (MLP vs head) while non-survivors are
    grey. A component in the `shared` set (survives in all settings) gets a bold red border.
    """
    node_colors: list[Any] = []
    node_sizes: list[int] = []
    border_colors: list[Any] = []
    border_widths: list[float] = []
    for layer in layers:
        for feat in range(n_total):
            node_colors.append(
                (MLP_COLOR if feat < num_mlp else HEAD_COLOR) if feat in survivors[layer] else BACKGROUND_COLOR
            )
            node_sizes.append(NODE_SIZE)  # same size for every neuron/head
            if feat in shared.get(layer, set()):
                border_colors.append(SHARED_RING_COLOR)
                border_widths.append(1.8)
            else:
                border_colors.append("none")
                border_widths.append(0.0)
    return node_colors, node_sizes, border_colors, border_widths


def _edges(survivors: dict[int, set[int]], layers: list[int]) -> list[tuple[str, str]]:
    """Edges connecting every surviving component of one layer to every survivor of the next."""
    return [
        (f"L{prev}_{a}", f"L{curr}_{b}")
        for prev, curr in zip(layers, layers[1:])
        for a in survivors[prev]
        for b in survivors[curr]
    ]


def _draw_arced_edges(
    ax: plt.Axes,
    positions: dict[str, tuple[int, int]],
    edgelist: list[tuple[str, str]],
    color: str | tuple[float, ...],
    bow: float,
) -> None:
    """Draw each edge as a quadratic Bezier that bows sideways (in x) by a constant `bow`.

    We bow in the HORIZONTAL (x) direction on purpose. The y-axis packs hundreds of neurons while the
    x-axis holds only a few layers, so the plot is extremely anisotropic; a normal `arc3` arc bows
    perpendicular to the edge, which under that anisotropy makes curvature depend wildly on the edge's
    slope (near-horizontal edges curve, near-vertical ones look straight). Offsetting the Bezier
    control point horizontally instead gives every edge the same visible curve regardless of slope.
    A per-setting-signed `bow` also makes settings that share the SAME neuron pair bow to different
    sides in the combined figure, so their colours stay separable instead of overlapping into one line.
    """
    for u, v in edgelist:
        (x1, y1), (x2, y2) = positions[u], positions[v]
        control = ((x1 + x2) / 2 + bow, (y1 + y2) / 2)  # midpoint pushed sideways in x by `bow`
        path = MplPath([(x1, y1), control, (x2, y2)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, alpha=0.7, linewidth=1.0))


def _finish_axes(ax: plt.Axes, layers: list[int], title: str) -> None:
    """Label the layer columns, drop the y-axis clutter and title the panel."""
    ax.set_title(title)
    ax.set_frame_on(False)
    # Pin the x-range to exactly the layer columns (plus a little padding). Without this, the arced
    # edges' control points would drag the autoscale outward and squash the columns into a thin central
    # strip; fixing xlim keeps the layers evenly spread across the (landscape) width.
    ax.set_aspect("auto")
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{layer}" for layer in layers])
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)


def draw_setting_panel(
    survivors: dict[int, set[int]],
    shared: dict[int, set[int]],
    layers: list[int],
    num_mlp: int,
    num_heads: int,
    ax: plt.Axes,
    title: str,
) -> None:
    """One setting's circuit: survivor nodes + black edges between adjacent survivors, shared ringed red.

    Args:
        survivors: This setting's survivor sets from `compute_survivors`.
        shared: The across-setting shared set (for the red highlight); pass {} to disable.
        layers: The layers (columns), in order.
        num_mlp: Number of MLP neurons (the MLP|head split; heads occupy feat >= num_mlp).
        num_heads: Number of attention heads.
        ax: The axis to draw onto.
        title: Panel title (the setting label).
    """
    n_total = num_mlp + num_heads
    graph, positions = _build_graph(layers, n_total)
    colors, sizes, border_colors, border_widths = _node_styles(survivors, shared, layers, num_mlp, n_total)
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_color=colors, node_size=sizes, edgecolors=border_colors, linewidths=border_widths
    )
    _draw_arced_edges(ax, positions, _edges(survivors, layers), color="black", bow=0.4)
    _finish_axes(ax, layers, title)


def draw_combined_panel(
    survivors_per_setting: dict[str, dict[int, set[int]]],
    shared: dict[int, set[int]],
    setting_colors: dict[str, Any],
    layers: list[int],
    num_mlp: int,
    num_heads: int,
    ax: plt.Axes,
) -> None:
    """All settings overlaid: nodes = union of survivors, edges drawn once per setting in its colour.

    Every setting's edges get a DIFFERENT colour so you can see where the settings share components
    and where they route apart. Shared nodes (survive in all settings) keep their bold red border.

    To stop settings that share the SAME neuron pair from hiding each other (identical lines would
    overlap and only the last colour would show), each setting's edges bow to a different side (a
    distinct signed `bow`), so overlapping edges fan out into separate, individually visible arcs.
    """
    n_total = num_mlp + num_heads
    graph, positions = _build_graph(layers, n_total)
    union = {layer: set().union(*(s[layer] for s in survivors_per_setting.values())) for layer in layers}
    colors, sizes, border_colors, border_widths = _node_styles(union, shared, layers, num_mlp, n_total)
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_color=colors, node_size=sizes, edgecolors=border_colors, linewidths=border_widths
    )
    n_settings = len(survivors_per_setting)
    for i, (setting, survivors) in enumerate(survivors_per_setting.items()):
        bow = (i - (n_settings - 1) / 2) * 0.35  # symmetric fan: each setting bows by a distinct amount/side
        _draw_arced_edges(ax, positions, _edges(survivors, layers), color=setting_colors[setting], bow=bow)
    _finish_axes(ax, layers, "All settings combined")
    # The setting-colour legend is added (stacked with the node legend) by save_combined_figure.


def _node_legend_handles() -> list[Line2D]:
    """Proxy handles decoding the node styles: the two component types and the shared-across-settings ring."""
    return [
        Line2D([], [], marker="o", linestyle="none", color=MLP_COLOR, label="MLP neuron"),
        Line2D([], [], marker="o", linestyle="none", color=HEAD_COLOR, label="attention head"),
        Line2D([], [], marker="o", linestyle="none", color=BACKGROUND_COLOR, label="not surviving"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=SHARED_RING_COLOR,
            markeredgewidth=1.8,
            label="shared across all settings",
        ),
    ]


def save_setting_figure(
    label: str,
    survivors: dict[int, set[int]],
    shared: dict[int, set[int]],
    layers: list[int],
    num_mlp: int,
    num_heads: int,
    out: Path,
) -> None:
    """One image for a single setting's circuit (black edges, shared nodes ringed), saved at dpi 300.

    Args:
        label: The setting's label (its title).
        survivors: This setting's survivor sets from `compute_survivors`.
        shared: The across-setting shared set (red ring); {} to disable.
        layers: The layers (columns), in order.
        num_mlp: Number of MLP neurons per layer.
        num_heads: Number of attention heads per layer.
        out: Path to write the .png to.
    """
    # Landscape: layers run left-to-right, so the figure widens with the number of layers.
    fig, ax = plt.subplots(figsize=(max(4.0 * len(layers), 10.0), 8.0))
    draw_setting_panel(survivors, shared, layers, num_mlp, num_heads, ax, label)
    ax.legend(handles=_node_legend_handles(), loc="upper right", ncol=1, fontsize=7, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out)


def save_combined_figure(
    survivors_per_setting: dict[str, dict[int, set[int]]],
    shared: dict[int, set[int]],
    setting_colors: dict[str, Any],
    layers: list[int],
    num_mlp: int,
    num_heads: int,
    out: Path,
) -> None:
    """One image overlaying every setting's circuit (per-setting edge colours), saved at dpi 300.

    Args:
        survivors_per_setting: {setting label: survivor sets from `compute_survivors`}.
        shared: The across-setting shared set (red ring); {} to disable.
        setting_colors: {setting label: edge colour} from `categorical_color_map`.
        layers: The layers (columns), in order.
        num_mlp: Number of MLP neurons per layer.
        num_heads: Number of attention heads per layer.
        out: Path to write the .png to.
    """
    # Landscape: layers run left-to-right, so the figure widens with the number of layers.
    fig, ax = plt.subplots(figsize=(max(4.0 * len(layers), 10.0), 8.0))
    draw_combined_panel(survivors_per_setting, shared, setting_colors, layers, num_mlp, num_heads, ax)
    # Two legends stacked in the top-right corner: node styles on top, setting edge colours below.
    node_legend = ax.legend(
        handles=_node_legend_handles(), loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7, framealpha=0.9
    )
    ax.add_artist(node_legend)  # keep it when the second legend is drawn
    setting_handles = [
        Line2D([], [], color=setting_colors[setting], label=setting) for setting in survivors_per_setting
    ]
    ax.legend(
        handles=setting_handles,
        title="setting (edge colour)",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.78),
        ncol=1,
        fontsize=7,
        framealpha=0.9,
    )
    fig.tight_layout()
    save_figure(fig, out)


def make_circuit_figures(
    survivors_per_setting: dict[str, dict[int, set[int]]],
    layers: list[int],
    num_mlp: int,
    num_heads: int,
    out_dir: Path,
) -> None:
    """Write the S + 1 images: one circuit_<setting>.png per setting plus one circuit_combined.png.

    The combined image is only written when there are >= 2 settings (with one setting it would just
    duplicate that setting's image).

    Args:
        survivors_per_setting: {setting label: survivor sets from `compute_survivors`}.
        layers: The layers (columns), in order.
        num_mlp: Number of MLP neurons per layer.
        num_heads: Number of attention heads per layer.
        out_dir: Directory to write the .png files into.
    """
    # "Shared" only means something when there are >= 2 settings to intersect; with one setting every
    # survivor would trivially be "shared", so pass {} to ring nothing in that case.
    shared = compute_shared(survivors_per_setting, layers) if len(survivors_per_setting) >= 2 else {}
    setting_colors = categorical_color_map(list(survivors_per_setting))
    for label, survivors in survivors_per_setting.items():
        save_setting_figure(label, survivors, shared, layers, num_mlp, num_heads, out_dir / f"circuit_{label}.png")
    if len(survivors_per_setting) >= 2:
        save_combined_figure(
            survivors_per_setting, shared, setting_colors, layers, num_mlp, num_heads, out_dir / "circuit_combined.png"
        )
    else:
        print("  only one setting -- skipping circuit_combined.png (it would duplicate that setting's image).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the surviving ablation circuit per setting (one image per setting + a combined image)."
    )
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        help="Directory of intervention .pt files from src/main.py (like src/plot_ablations.py's --dir). "
        "Each file is one setting; layer/head/MLP counts are read from each file.",
    )
    parser.add_argument("--output", "-o", default="plots", help="Directory to write the figures into.")
    parser.add_argument(
        "--percentile",
        "-p",
        type=float,
        default=99.0,
        help="Keep components in this top percentile of accuracy_drop per layer (default 99; try 95 if sparse).",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    settings = discover_settings(Path(args.dir))  # {label: .pt path}, one per setting, labelled by filename
    if not settings:
        raise SystemExit(f"No .pt files found in {args.dir}.")

    loaded = load_settings(settings)  # {label: (summary, metadata)}; skips non-intervention / empty files
    if not loaded:
        raise SystemExit(
            f"No usable intervention runs among {sorted(settings)} -- the .pt files may be incomplete or not "
            "intervention outputs (need an 'ablations' key). See the [skip] notes above."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common node geometry so the same neuron sits at the same spot in every panel (union of the runs'
    # layers + the largest MLP/head counts); see common_geometry for the shared-model assumption.
    layers, num_mlp, num_heads = common_geometry(loaded)
    if not num_mlp:
        raise SystemExit(
            "Can't tell where attention heads start (no num_mlp_neurons in metadata and no head was ablated) -- "
            "the MLP|head split is needed to lay out the columns."
        )

    survivors_per_setting = {
        label: compute_survivors(ablations, layers, args.percentile) for label, (ablations, _) in loaded.items()
    }
    # Print the surviving components (top percentile) and their accuracy drops per layer, so the same
    # information the diagram encodes is also readable in the terminal.
    for label, (ablations, _) in loaded.items():
        print_survivors(label, ablations, survivors_per_setting[label], num_mlp)

    make_circuit_figures(survivors_per_setting, layers, num_mlp, num_heads, out_dir)

    print(f"Done. Figures in {out_dir}/")
