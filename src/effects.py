"""Summarise an intervention run: how much did each ablated component actually matter?

This is the FINAL step of the workflow and the payoff of the whole exercise. It reads the `.pt`
file written by `python src/main.py --intervention ...` (which holds a baseline run plus one run
per component with that component switched off) and reports, for each ablated neuron / attention
head, an EFFECT SIZE -- how much the model's answers changed when that component was removed. The
components with the largest effect are the ones the model causally relies on for the behaviour.

Two effect measures are reported:

  - answer_change_rate: the fraction of prompts whose answer changed versus the baseline. This
    needs NO ground truth, so it works for any task out of the box -- a bigger number means the
    component mattered more to whatever the model was doing.

  - accuracy_drop: baseline accuracy minus the ablated accuracy. This requires knowing the RIGHT
    answer for each prompt, which is task-specific, so you supply it by implementing `is_correct`
    below. Until you do, this column shows "n/a" and only answer_change_rate is used.

Run it after an intervention pass:

    python src/effects.py --file <the --intervention output>.pt           # ranked table
    python src/effects.py --file <...>.pt --plot effects.png              # + a bar chart

The result is a table (and, with --plot, a bar chart) ranked by effect size -- read it top-down to
see which components the model most depends on, then go investigate what those components actually
compute (that part is the open-ended research; this template gets you the ranked, causal shortlist).
Both MLP neurons and attention heads appear in the ranking (the "type" column / bar colour).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def is_correct(row: dict[str, Any], metadata: dict[str, Any]) -> bool | None:
    """Return whether the model's generated answer is correct for this prompt, or None if unknown.

    Implement this to unlock the `accuracy_drop` column (how much ablating a component hurts the
    model's ACTUAL task performance). Returning None for a row leaves it out of the accuracy
    figures; the default returns None for everything, so accuracy is simply not reported until you
    fill this in. (The answer_change_rate effect measure works regardless.)

    Args:
        row: One per-prompt result dict (an entry of a "result" list).
        metadata: The intervention file's metadata dict.

    Returns:
        True/False if you can judge this prompt, or None to skip it.
    """
    # ----------------------------------------------------------------------------------- #
    # TODO (optional): decide whether the generated answer is right for your task. Available
    # fields: row["prompt"], row["result"]["answer"]["token"], row["result"]["completion"].
    # ----------------------------------------------------------------------------------- #
    #
    # Example (the single-digit addition demo, whose prompts end with "...a+b="): compute the
    # true sum from the prompt and compare it to the answer the model generated:
    #
    #     question = row["prompt"].rsplit("\n", 1)[-1].rstrip("=")     # e.g. "7+5"
    #     a, b = question.split("+")
    #     generated = (row["result"]["answer"]["token"] or "").strip()
    #     return generated == str(int(a) + int(b))
    #
    return None


def _answer_token(row: dict[str, Any]) -> str | None:
    """Return the (stripped) answer string the model produced for a row, or None if there isn't one."""
    answer = row.get("result", {}).get("answer", {})
    token = answer.get("token")
    return token.strip() if isinstance(token, str) else None


def _answers_by_prompt(rows: list[dict[str, Any]]) -> dict[int, str | None]:
    """Index a list of result rows by their prompt_idx, mapping each to its answer string."""
    return {row.get("prompt_idx", i): _answer_token(row) for i, row in enumerate(rows)}


def _answer_change_rate(baseline: dict[int, str | None], ablated_rows: list[dict[str, Any]]) -> float:
    """Fraction of prompts whose answer differs from baseline once this component is ablated.

    Baseline and ablated runs use the same prompts in the same order, so we align them by
    prompt_idx and count how many answers flipped.
    """
    ablated = _answers_by_prompt(ablated_rows)
    shared = [k for k in ablated if k in baseline]
    if not shared:
        return 0.0
    changed = sum(ablated[k] != baseline[k] for k in shared)
    return changed / len(shared)


def _accuracy(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> float | None:
    """Mean correctness over the rows where is_correct gives a verdict, or None if it never does."""
    verdicts = [is_correct(row, metadata) for row in rows]
    defined = [bool(v) for v in verdicts if v is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def summarise(data: dict[str, Any]) -> dict[str, Any]:
    """Compute the baseline accuracy and a per-component effect report from a loaded intervention file.

    Args:
        data: The dict loaded from a `main.py --intervention` .pt file (has "baseline" and
            "ablations" keys).

    Returns:
        {"baseline_accuracy": float | None, "components": [per-component effect dicts]}, with the
        components sorted by answer_change_rate (largest effect first).
    """
    metadata = data.get("metadata", {})
    baseline_rows = data["baseline"]
    baseline_answers = _answers_by_prompt(baseline_rows)
    baseline_accuracy = _accuracy(baseline_rows, metadata)

    components: list[dict[str, Any]] = []
    for ablation in data.get("ablations", []):
        rows = ablation.get("result", [])
        ablated_accuracy = _accuracy(rows, metadata)
        accuracy_drop = (
            None if (baseline_accuracy is None or ablated_accuracy is None) else baseline_accuracy - ablated_accuracy
        )
        components.append(
            {
                "layer_idx": ablation.get("layer_idx"),
                "type": ablation.get("type"),
                "local_idx": ablation.get("local_idx"),
                "feature_idx": ablation.get("feature_idx"),
                "answer_change_rate": _answer_change_rate(baseline_answers, rows),
                "ablated_accuracy": ablated_accuracy,
                "accuracy_drop": accuracy_drop,
                "num_prompts": len(rows),
            }
        )

    components.sort(key=lambda c: c["answer_change_rate"], reverse=True)
    return {"baseline_accuracy": baseline_accuracy, "components": components}


def save_plot(summary: dict[str, Any], path: str, top: int) -> None:
    """Save a horizontal bar chart of the top components (by answer-change rate) to `path`.

    MLP neurons and attention heads are coloured differently, so you can see at a glance which kind
    of component dominates. Uses a headless matplotlib backend, so it works over SSH / with no
    display. (matplotlib is imported lazily here so a plain `--file` run needs nothing extra.)

    Args:
        summary: The dict returned by `summarise`.
        path: Where to write the image, e.g. "effects.png".
        top: How many of the highest-effect components to show.
    """
    import matplotlib

    matplotlib.use("Agg")  # render straight to a file; no interactive window needed
    import matplotlib.pyplot as plt

    components = summary["components"][:top]
    if not components:
        print("Nothing to plot (no ablations found).")
        return

    labels = [f"L{c['layer_idx']} {c['type']}#{c['local_idx']}" for c in components]
    values = [c["answer_change_rate"] for c in components]
    colors = ["tab:orange" if c["type"] == "head" else "tab:blue" for c in components]

    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.4 * len(components))))
    ax.barh(range(len(components)), values, color=colors)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # largest effect at the top
    ax.set_xlabel("answer change rate (fraction of prompts whose answer flipped when ablated)")
    ax.set_title("Component ablation effect")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
    ]
    ax.legend(legend_handles, ["MLP neuron", "attention head"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Load an intervention .pt file, rank components by effect size, and print (optionally save) the report."""
    parser = argparse.ArgumentParser(description="Rank ablated components by how much they changed the answers.")
    parser.add_argument("--file", "-f", type=str, required=True, help="The .pt file from `main.py --intervention`.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional path to save the report as JSON.")
    parser.add_argument("--plot", "-p", type=str, default=None, help="Optional path to save a bar chart (PNG).")
    parser.add_argument("--top", "-t", type=int, default=20, help="How many top components to print/plot (default 20).")
    args = parser.parse_args()

    data = torch.load(args.file, map_location="cpu", weights_only=False)
    if "baseline" not in data or "ablations" not in data:
        raise SystemExit(
            f"{args.file} is not an intervention file (expected 'baseline' and 'ablations' keys). "
            "Produce one with: python src/main.py -m <model> --intervention analysis.json"
        )

    summary = summarise(data)
    baseline_accuracy = summary["baseline_accuracy"]
    components = summary["components"]

    acc_str = "n/a (implement is_correct)" if baseline_accuracy is None else f"{baseline_accuracy:.3f}"
    print(f"Baseline accuracy: {acc_str}")
    print(f"Ablated {len(components)} components, ranked by how much switching each one off changed the answers:\n")

    header = f"{'rank':>4}  {'component':<16}{'answer_change':>14}{'acc_drop':>10}"
    print(header)
    print("-" * len(header))
    for rank, c in enumerate(components[: args.top], start=1):
        component = f"L{c['layer_idx']} {c['type']}#{c['local_idx']}"
        drop = "n/a" if c["accuracy_drop"] is None else f"{c['accuracy_drop']:+.3f}"
        print(f"{rank:>4}  {component:<16}{c['answer_change_rate']:>14.3f}{drop:>10}")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nSaved full report to {args.output}")

    if args.plot:
        save_plot(summary, args.plot, args.top)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
