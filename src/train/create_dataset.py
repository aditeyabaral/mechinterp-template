"""Generate a synthetic training dataset, check it, and save it (locally, and to the HF Hub with --hub-name).

This is a task-agnostic skeleton: it parses CLI arguments, checks the dataset, saves it to
--output-dir and, with --hub-name, also pushes it to the Hub. You implement `build_dataset()`
to produce the actual data for your task, and optionally `item_key()` for the leakage check.

`build_dataset` must return a `datasets.DatasetDict` with "train" and "validation" splits.
Each row must contain at least a "prompt" and an "answer" column; add any extra metadata
columns your training/eval needs. What the splits are for:
  - train:      what the model learns from.
  - validation: scored during training to pick the best checkpoint (src/train/train_model.py),
                and a good source of held-out prompts for the analysis (src/utils/dataset.py).

A leakage-safe recipe to follow inside `build_dataset` (strongly recommended):
  1. Enumerate every atomic "item" your questions are built from (e.g. each (a, b)
     operand pair) and shuffle/split them into disjoint train/val pools BEFORE
     sampling any example. This is what prevents leakage between the splits.
  2. For each row, sample the question item from the split's own pool, but draw any
     few-shot examples from the TRAIN pool only -- so a val question is never seen
     during training in any form, while val prompts still demonstrate the pattern.
`check_dataset` then verifies that no item (see `item_key`) appears in two splits, before
anything is saved or pushed.

Big datasets: build each split with `Dataset.from_generator(..., cache_dir=cache_dir)`, as the
example does. The rows are then written to disk as they are generated, instead of being held in
memory, under a temporary directory next to --output-dir (not /tmp, which is often small).

A self-contained worked example (single-digit addition, 4-shot) is given in the
commented `build_dataset` body below; delete it and write your own.
"""

import argparse
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict

SPLITS = ("train", "validation")


def item_key(row: dict) -> object:
    """Return the "item" a row's question is built from, for the leakage check in check_dataset.

    Two rows with the same key ask the same question, so check_dataset fails if a key appears in
    two splits. The default is the whole prompt (no prompt may appear in two splits). That misses
    leakage when prompts differ only in their few-shot examples, so return the question's item
    instead if you can, e.g. the (a, b) operands.

    Args:
        row: One dataset row as a dict of its columns.
    """
    # ----------------------------------------------------------------------- #
    # TODO (optional): return the item this row's question is built from.
    # ----------------------------------------------------------------------- #
    #
    # Example (the single-digit addition task below, whose rows carry "a" and "b" columns):
    #
    #     return (row["a"], row["b"])
    #
    # ----------------------------------------------------------------------- #
    return row["prompt"]


def _split_keys(dataset: Dataset, split: str) -> set:
    """Check one split's rows and return the item_key of each, reading the split in batches."""
    if len(dataset) == 0:
        raise ValueError(f"The {split!r} split is empty")
    missing = {"prompt", "answer"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"The {split!r} split has no {sorted(missing)} column")
    keys = set()
    for batch in dataset.iter(batch_size=100_000):
        for j, (prompt, answer) in enumerate(zip(batch["prompt"], batch["answer"])):
            if not (isinstance(prompt, str) and isinstance(answer, str) and prompt and answer):
                raise ValueError(f"{split}: prompt and answer must be non-empty strings, got {prompt!r}, {answer!r}")
            keys.add(item_key({column: values[j] for column, values in batch.items()}))
    return keys


def check_dataset(dataset: DatasetDict) -> None:
    """Raise if the dataset breaks a rule: the splits, the columns, or leakage between the splits.

    Run on the dataset before it is saved or pushed, so a broken dataset never reaches training.
    Reads the splits in batches, so it works on datasets that don't fit in memory.
    """
    if set(dataset) != set(SPLITS):
        raise ValueError(f"build_dataset must return the splits {SPLITS}, got {sorted(dataset)}")
    keys = {split: _split_keys(dataset[split], split) for split in SPLITS}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1 :]:
            shared = keys[first] & keys[second]
            if shared:
                raise ValueError(
                    f"{len(shared)} items appear in both {first!r} and {second!r} (e.g. {next(iter(shared))!r}): "
                    "the splits leak. Split your items into disjoint pools before sampling rows."
                )


def build_dataset(train_size: int, val_size: int, seed: int, holdout: float, cache_dir: str) -> DatasetDict:
    """Build the train/validation DatasetDict for your task.

    Args:
        train_size: Number of training rows to generate.
        val_size: Number of validation rows to generate.
        seed: Random seed for reproducibility.
        holdout: Fraction of unique items to reserve for validation questions.
        cache_dir: A temporary directory on disk to generate big splits into (see the module
            docstring); pass it to Dataset.from_generator.

    Returns:
        A DatasetDict with "train" and "validation" splits. Each row must contain
        at least "prompt" and "answer" columns.
    """
    # ----------------------------------------------------------------------- #
    # TODO: implement dataset generation for your task.
    # ----------------------------------------------------------------------- #
    #
    # Example (single-digit addition, 4-shot, leakage-safe -- matches the commented examples
    # in src/train/train_tokenizer.py and src/utils/dataset.py):
    #
    #     from functools import partial
    #
    #     import numpy as np
    #     from datasets import Dataset
    #
    #     few_shot = 4
    #
    #     # 1. Enumerate every (a, b) item, then split into disjoint train/val pools so a
    #     #    validation question never appears as a training question.
    #     items = [(a, b) for a in range(10) for b in range(10)]
    #     rng = np.random.default_rng(seed)
    #     rng.shuffle(items)
    #     n_holdout = max(1, int(len(items) * holdout))
    #     pools = {"val": items[:n_holdout], "train": items[n_holdout:]}
    #
    #     # 2. Build one split: the question comes from `q_pool`, few-shot ALWAYS from train. Each
    #     #    shot looks like "7+5=12" and the prompt ends at "=", exactly the shape produced by
    #     #    src/utils/dataset.py's example -- the same prompts you'll later run inference on. The
    #     #    a/b/answer columns mirror the ground-truth dict that example stores under "metadata".
    #     #    Rows are YIELDED one at a time, so Dataset.from_generator can write them to disk as
    #     #    they come (fine for millions of rows).
    #     def generate_split(total: int, q_pool: list, split_name: str, split_seed: int):
    #         srng = np.random.default_rng(split_seed)
    #         fs_pool = pools["train"]
    #         for i in range(total):
    #             shots = []                                     # the solved examples shown first
    #             for j in srng.integers(len(fs_pool), size=few_shot):
    #                 fa, fb = fs_pool[j]
    #                 shots.append(f"{fa}+{fb}={fa + fb}")       # e.g. "7+5=12"
    #             a, b = q_pool[srng.integers(len(q_pool))]
    #             prompt = "\n".join(shots) + f"\n{a}+{b}="      # ends at "=", answer to come
    #             yield {"_id": f"{split_name}-{i}", "a": a, "b": b, "answer": str(a + b), "prompt": prompt}
    #
    #     # partial, not gen_kwargs: from_generator would treat a list in gen_kwargs as shards.
    #     def split(total: int, pool: str, name: str, split_seed: int) -> Dataset:
    #         rows = partial(generate_split, total, pools[pool], name, split_seed)
    #         return Dataset.from_generator(rows, cache_dir=cache_dir)
    #
    #     return DatasetDict({
    #         "train": split(train_size, "train", "train", seed),
    #         "validation": split(val_size, "val", "validation", seed + 1),
    #     })
    #
    # ----------------------------------------------------------------------- #
    raise NotImplementedError("Implement build_dataset() for your task -- see the example in comments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset and save it locally (and to the Hub)")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub dataset repo to also push to (e.g. your-username/your-dataset-name). The dataset is "
        "always saved locally to --output-dir; without --hub-name nothing is pushed (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/dataset",
        help="Local directory to save the dataset to.",
    )
    parser.add_argument("--train-size", type=int, default=1_000_000, help="Number of training rows")
    parser.add_argument("--val-size", type=int, default=100_000, help="Number of validation rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (val uses seed + 1)")
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.1,
        help="Fraction of unique items held out for val questions (default: 0.1). "
        "Independent of --val-size, which controls the number of val rows.",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the Hub")
    args = parser.parse_args()

    print(f"Generating dataset | train={args.train_size:,} val={args.val_size:,}\n")

    # Generated rows go to a temporary directory next to the output, on disk (deleted at the end).
    output_dir = Path(args.output_dir).resolve()
    made_parent = not output_dir.parent.exists()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as cache_dir:
            dataset = build_dataset(
                train_size=args.train_size,
                val_size=args.val_size,
                seed=args.seed,
                holdout=args.holdout,
                cache_dir=cache_dir,
            )
            check_dataset(dataset)
            print(dataset)
            print("\nAll dataset checks passed.")

            dataset.save_to_disk(str(output_dir))
            print(f"\nSaved dataset to {output_dir}")
            if args.hub_name:
                dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
                print(f"Pushed dataset to hub: {args.hub_name}")
            else:
                print("(pass --hub-name to also push it to the Hub)")
    except NotImplementedError:
        if made_parent:  # build_dataset isn't written yet: leave no empty directory behind
            output_dir.parent.rmdir()
        raise
