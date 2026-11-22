"""Split-half check of an ablation sweep: were there enough prompts?

    python src/analysis/split_half.py <intervention .pt from main.py --intervention>

Each component's accuracy drop is measured again on the even-indexed prompts and on the odd-indexed
ones, and the two rankings of components are compared with a Spearman correlation. Near 1 means the
ranking doesn't depend on which prompts were sampled, so --num-prompts was large enough; a low value
means you need more prompts.

Trust the "meaningful" number: it only counts components whose drop over all prompts is above
--meaningful-drop. Hundreds of near-zero drops rank in random order and would hide whether the
components that DO matter are ranked reliably.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

# This script lives in src/analysis/; put src/ on the import path so inference.py resolves when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference  # noqa: E402


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman correlation, or None where it is undefined (fewer than 3 values, or a constant list)."""
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def split_half(
    baseline_correct: torch.Tensor, ablations: list[dict[str, Any]], names: list[str], meaningful_drop: float
) -> dict[str, dict[str, float | int | None]]:
    """Spearman correlation of the components' drops on the even- vs odd-indexed prompts.

    For the whole-answer drop and for the drop averaged over the answer steps, over all components and
    over the meaningful ones (drop above meaningful_drop on all prompts).
    """
    halves = [slice(0, None, 2), slice(1, None, 2)]
    drops: dict[str, list[list[float]]] = {"accuracy_drop": [], "mean_step_accuracy_drop": []}
    for half in halves:
        base_whole = inference.accuracy(inference.whole_answer_correct(baseline_correct[half]))
        base_steps = inference.step_accuracy(baseline_correct[half], names)
        whole, steps = [], []
        for entry in ablations:
            correct = entry["correct"][half]
            whole.append(base_whole - inference.accuracy(inference.whole_answer_correct(correct)))
            acc = inference.step_accuracy(correct, names)
            steps.append(float(np.mean([base_steps[step] - acc[step] for step in acc])))
        drops["accuracy_drop"].append(whole)
        drops["mean_step_accuracy_drop"].append(steps)

    # Which components are meaningful is decided on ALL prompts, not on either half.
    full_drop = {
        "accuracy_drop": [entry["accuracy_drop"] for entry in ablations],
        "mean_step_accuracy_drop": [float(np.mean(list(entry["step_accuracy_drop"].values()))) for entry in ablations],
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, (even, odd) in drops.items():
        meaningful = [i for i, drop in enumerate(full_drop[metric]) if drop > meaningful_drop]
        result[metric] = {
            "all": _spearman(even, odd),
            "meaningful": _spearman([even[i] for i in meaningful], [odd[i] for i in meaningful]),
            "n_meaningful": len(meaningful),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-half check: are a sweep's component rankings stable?")
    parser.add_argument("path", help="An intervention .pt file from main.py --intervention.")
    parser.add_argument(
        "--meaningful-drop",
        type=float,
        default=0.005,
        help="Only components whose drop is above this count as meaningful (default 0.005).",
    )
    args = parser.parse_args()

    data = torch.load(args.path, map_location="cpu", weights_only=False)
    if "ablations" not in data:
        raise SystemExit(f"{args.path} is not an intervention run (no 'ablations'); run main.py --intervention first")
    result = split_half(
        data["baseline_correct"], data["ablations"], data["metadata"]["step_names"], args.meaningful_drop
    )
    for metric, stats in result.items():
        print(
            f"{metric}: Spearman over all components {stats['all']}, over the {stats['n_meaningful']} "
            f"meaningful ones {stats['meaningful']}"
        )
