"""Load per-component ablation summaries from intervention .pt files.

These helpers know nothing about any particular task or figure -- they just read what
`src/main.py --intervention` saved. They are imported by BOTH src/analysis/plot_ablations.py (heatmaps +
scatter) and src/analysis/plot_circuits.py (circuit diagrams) so the file format is parsed in exactly one
place. Each intervention .pt is treated as one SETTING, labelled by its filename (stem), mirroring
how src/analysis/lasso.py takes a --dir of runs.
"""

from pathlib import Path
from typing import Any

import torch


def _metric_value(entry: dict[str, Any], metric: str) -> float:
    """One ablation entry's drop under `metric`, the score the plots show.

    "accuracy_drop" is the whole-answer drop (the default); "mean_step_accuracy_drop" the drop
    averaged over the answer steps; "step:<name>" the drop at one step, where <name> is one of the
    run's metadata["step_names"] (e.g. "step:step0" for the answer's first token, "step:eos" for
    stopping), NaN if no prompt has that step.
    """
    if metric == "accuracy_drop":
        return float(entry[metric])
    if metric == "mean_step_accuracy_drop":
        drops = entry["step_accuracy_drop"]
        return sum(drops.values()) / len(drops)
    if metric.startswith("step:"):
        return float(entry["step_accuracy_drop"].get(metric[len("step:") :], float("nan")))
    raise ValueError(f"Unknown metric {metric!r}: use accuracy_drop, mean_step_accuracy_drop or step:<name>")


def load_ablations(pt_path: Path, metric: str = "accuracy_drop") -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Load one intervention file's per-component ablation summary + metadata, dropping heavy rows.

    Args:
        pt_path: A .pt file written by `src/main.py --intervention` (has an "ablations" key).
        metric: Which drop the summary's "accuracy_drop" holds (see _metric_value); the default is the
            whole-answer accuracy drop.

    Returns:
        (summary, metadata), or None if the file is absent, empty (a still-running / failed run), or
        not an intervention file. `summary` is a list of {layer_idx, type, local_idx, feature_idx,
        accuracy_drop} dicts (one per ablated component); `metadata` is the run's saved metadata dict
        (layer_indices, num_attention_heads, num_mlp_neurons, ...). The large per-prompt `result`
        rows are discarded here, so only the small summary + metadata stay in memory (each .pt ~1GB).

    Raises:
        KeyError: if the ablations don't record `accuracy_drop` (a file not written by main.py --intervention).
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
            f"{pt_path.name}: ablations have no 'accuracy_drop' -- was it written by src/main.py --intervention?"
        )
    summary = [
        {
            "layer_idx": a["layer_idx"],
            "type": a["type"],
            "local_idx": a["local_idx"],
            "feature_idx": a["feature_idx"],
            "accuracy_drop": _metric_value(a, metric),
        }
        for a in loaded["ablations"]
    ]
    metadata = loaded.get("metadata", {})
    del loaded  # free the heavy result rows we don't need before anything else loads
    return summary, metadata


def infer_num_mlp(ablations: list[dict[str, Any]], metadata: dict[str, Any]) -> int | None:
    """Where the head columns begin (d_mlp), read from the file so you never pass it on the CLI.

    Prefers metadata["num_mlp_neurons"] (main.py saves it). Falls back for older files that lack it:
    a head's feature_idx is num_mlp + its local_idx (the shared main.py/lasso.py index convention),
    so any head ablation gives num_mlp = feature_idx - local_idx. Returns None only if neither is
    available (no saved value AND no head was ablated) -- then the MLP|head split can't be placed.
    """
    if metadata.get("num_mlp_neurons") is not None:
        return int(metadata["num_mlp_neurons"])
    for a in ablations:
        if a["type"] == "head":
            return a["feature_idx"] - a["local_idx"]
    return None


def neuron_label(layer_idx: int, feat_type: str, local_idx: int) -> str:
    """Human-readable component tag, e.g. 'L0MLP1' (MLP neuron) or 'L2A2' (attention head)."""
    kind = "MLP" if feat_type == "mlp" else "A"
    return f"L{layer_idx}{kind}{local_idx}"


def discover_settings(directory: Path) -> dict[str, Path]:
    """The settings to plot: every .pt in `directory`, labelled by filename (stem).

    Mirrors how src/analysis/lasso.py takes a --dir of runs. Non-intervention .pt files are filtered out later
    by load_ablations (it returns None for them), so this can safely list every .pt it finds.
    """
    return {path.stem: path for path in sorted(directory.glob("*.pt"))}


def common_layout(loaded: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]) -> tuple[list[int], int, int]:
    """The shared node layout across every loaded setting: (layers, num_mlp, num_heads).

    Both plotting scripts lay every setting out on the SAME axes so a given neuron/head sits in the
    same place in every figure. That needs one common layout, so we take the UNION of the runs'
    layers (a run missing a layer just leaves that column empty) and the LARGEST num_mlp / num_heads
    seen (assumes the runs share a model, the normal case; mixing different-width models would
    misplace the head columns).

    Args:
        loaded: {setting label: (summary, metadata)} from load_settings.

    Returns:
        (layers sorted ascending, num_mlp, num_heads). num_mlp is 0 if it can't be inferred from any
        file (no saved num_mlp_neurons and no head ablated) -- the caller decides whether that's fatal.
    """
    layers = sorted(
        {layer for _, meta in loaded.values() for layer in (meta.get("layer_indices") or [])}
        | {a["layer_idx"] for ablations, _ in loaded.values() for a in ablations}
    )
    num_mlp = max((infer_num_mlp(ablations, meta) or 0) for ablations, meta in loaded.values())
    num_heads = max(meta.get("num_attention_heads", 0) for _, meta in loaded.values())
    return layers, num_mlp, num_heads


def load_settings(
    settings: dict[str, Path], metric: str = "accuracy_drop"
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Load each setting's .pt once, keeping only the ones with usable ablations.

    Args:
        settings: {label: intervention .pt path} (from discover_settings).
        metric: Which drop to load (see _metric_value).

    Returns:
        {label: (summary, metadata)} for the files that loaded -- missing/empty/non-intervention
        files are skipped (load_ablations prints why). Loading once here means the ~1GB files are
        each read a single time, then every figure reuses the summaries.
    """
    loaded = {}
    for label, path in settings.items():
        result = load_ablations(Path(path), metric)
        if result and result[0]:  # has ablations
            loaded[label] = result
    return loaded
