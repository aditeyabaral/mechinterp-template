"""Strip large tensor fields from saved inference result .pt files to reduce storage size.

A capture run saves big activation tensors (residual stream, MLP neurons, attention heads,
embeddings, logits) in each .pt file. Once you've finished the heavy analysis you often want a
small copy that keeps only the lightweight bookkeeping (prompts, answers, metadata) -- e.g. to
share results or skim them. This script writes such a stripped copy of every file in a directory.

`strip_result` already targets the result schema this template's main.py produces (both normal
mode and --intervention mode). If you add your own heavy fields to the saved results, add their
keys to the lists in `_strip_run` below.
"""

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

# The heavy fields to drop. `_RUN_KEYS` are top-level keys of a single per-prompt run dict;
# `_ANSWER_KEYS` are the big tensors inside that run's "answer" sub-dict.
_RUN_KEYS = ("output_ids", "geometry")
_ANSWER_KEYS = ("mlp_neurons", "attn_heads", "residual", "embedding", "logits")


def _strip_run(run: dict) -> None:
    """Drop the heavy tensors from one per-prompt run dict (the inner "result"), in place."""
    for key in _RUN_KEYS:
        run.pop(key, None)
    answer = run.get("answer")
    if isinstance(answer, dict):
        for key in _ANSWER_KEYS:
            answer.pop(key, None)


def _strip_rows(rows: list[dict]) -> None:
    """Strip every row produced by inference.run.

    Each row looks like {"prompt_idx", "prompt", "prompt_length", "result": {<run dict>}}.
    """
    for row in rows:
        run = row.get("result")
        if isinstance(run, dict):
            _strip_run(run)


def strip_result(data: dict) -> dict:
    """Remove the heavy tensor fields from a loaded .pt result file (in place) and return it.

    Handles both layouts main.py saves:
      - normal mode:        {"result": [rows], "metadata": {...}}
      - intervention mode:  {"baseline": [rows], "ablations": [{..., "result": [rows]}], "metadata": {...}}

    Args:
        data: The dict loaded from a saved .pt file.

    Returns:
        The same dict with large tensor fields removed.
    """
    if "baseline" in data:  # intervention-mode file
        _strip_rows(data["baseline"])
        for ablation in data.get("ablations", []):
            _strip_rows(ablation.get("result", []))
    elif "result" in data:  # normal-mode file
        _strip_rows(data["result"])
    return data


def main() -> None:
    """Strip every .pt file in a directory and write the lightweight copies to a 'stripped' subdir."""
    parser = argparse.ArgumentParser(description="Strip large tensor fields from inference result .pt files.")
    parser.add_argument("--dir", required=True, help="Directory containing inference .pt files.")
    args = parser.parse_args()

    src_dir = Path(args.dir)
    stripped_dir = src_dir / "stripped"
    stripped_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(src_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {src_dir}")

    for pt_file in tqdm(pt_files, desc="Stripping"):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            data = strip_result(data)
            torch.save(data, stripped_dir / pt_file.name)
            del data
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
