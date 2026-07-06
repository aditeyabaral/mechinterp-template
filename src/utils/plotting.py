"""Small figure helpers shared by the plotting scripts (src/plot_ablations.py, src/plot_circuits.py).

Kept here so the two scripts colour their categories the same way and save figures identically,
rather than each carrying its own copy. Nothing here is task-specific -- it's pure matplotlib glue.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def categorical_color_map(keys: list) -> dict[Any, Any]:
    """Give each key its own distinct colour: a categorical palette for a few, a sampled one if many.

    Used to colour layers (in plot_ablations.py) and settings (in plot_circuits.py) consistently.
    `tab10`/`tab20` give well-separated categorical colours for up to 20 keys; beyond that there is
    no categorical palette big enough, so we sample the continuous `viridis` map evenly instead.

    Args:
        keys: The items to colour (e.g. layer indices or setting labels), in the order you want them
            assigned colours.

    Returns:
        {key: colour} covering every key.
    """
    if len(keys) <= 10:
        cmap = plt.get_cmap("tab10")  # 10 distinct, well-separated colours
        return {key: cmap(i) for i, key in enumerate(keys)}
    if len(keys) <= 20:
        cmap = plt.get_cmap("tab20")
        return {key: cmap(i) for i, key in enumerate(keys)}
    cmap = plt.get_cmap("viridis")  # too many for a categorical palette -> sample a continuous one
    return {key: cmap(i / (len(keys) - 1)) for i, key in enumerate(keys)}


def save_figure(fig: plt.Figure, out: Path, dpi: int = 300) -> None:
    """Write a figure to `out` (tight bounding box), close it to free memory, and print where it went."""
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
