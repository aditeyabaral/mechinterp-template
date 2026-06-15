"""Generate a synthetic training dataset and save it (locally by default, or push to the HF Hub).

This is a task-agnostic skeleton: it parses CLI arguments and saves a DatasetDict (locally by
default, or pushes it to the Hub with --hub-name). You implement `build_dataset()` to produce
the actual data for your task.

`build_dataset` must return a `datasets.DatasetDict` with "train" and "validation"
splits. Each row must contain at least a "prompt" and an "answer" column; add any
extra metadata columns your training/eval needs.

A leakage-safe recipe to follow inside `build_dataset` (strongly recommended):
  1. Enumerate every atomic "item" your questions are built from (e.g. each (a, b)
     operand pair) and shuffle/split them into disjoint train/val pools BEFORE
     sampling any example. This is what prevents train/val leakage.
  2. For each row, sample the question item from the split's own pool, but draw any
     few-shot examples from the TRAIN pool only -- so a val question is never seen
     during training in any form, while val prompts still demonstrate the pattern.

A self-contained worked example (single-digit addition, 4-shot) is given in the
commented `build_dataset` body below; delete it and write your own.
"""

import argparse

from datasets import DatasetDict


def build_dataset(train_size: int, val_size: int, seed: int, val_holdout: float) -> DatasetDict:
    """Build the train/validation DatasetDict for your task.

    Args:
        train_size: Number of training rows to generate.
        val_size: Number of validation rows to generate.
        seed: Random seed for reproducibility.
        val_holdout: Fraction of unique items to reserve for validation questions.

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
    #     split_idx = int(len(items) * (1 - val_holdout))
    #     pools = {"train": items[:split_idx], "val": items[split_idx:]}
    #
    #     def render(a: int, b: int) -> str:
    #         return f"{a}+{b}={a + b}"  # one solved example, e.g. "7+5=12"
    #
    #     # 2. Build one split: the question comes from `q_pool`, few-shot always from train.
    #     def generate_split(total: int, q_pool: list, split_name: str, split_seed: int) -> dict:
    #         srng = np.random.default_rng(split_seed)
    #         fs_pool = pools["train"]
    #         rows = []
    #         for i in range(total):
    #             a, b = q_pool[srng.integers(len(q_pool))]
    #             fs = [render(*fs_pool[j]) for j in srng.integers(len(fs_pool), size=few_shot)]
    #             prompt = "\n".join(fs) + f"\n{a}+{b}="  # ends at "=", the model produces the sum
    #             rows.append(
    #                 {"_id": f"{split_name}-{i}", "question": f"{a}+{b}", "answer": str(a + b), "prompt": prompt}
    #             )
    #         return {k: [r[k] for r in rows] for k in rows[0]}
    #
    #     train_data = generate_split(train_size, pools["train"], "train", seed)
    #     val_data = generate_split(val_size, pools["val"], "validation", seed + 1)
    #     return DatasetDict({
    #         "train": Dataset.from_dict(train_data),
    #         "validation": Dataset.from_dict(val_data),
    #     })
    #
    # ----------------------------------------------------------------------- #
    raise NotImplementedError("Implement build_dataset() for your task -- see the example in comments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset and save it locally or to the Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub dataset repo to push to (e.g. your-username/your-dataset-name). "
        "If omitted, the dataset is saved locally to --output-dir instead (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/dataset",
        help="Local directory to save the dataset to when --hub-name is not given.",
    )
    parser.add_argument("--train-size", type=int, default=1_000_000, help="Number of training rows")
    parser.add_argument("--val-size", type=int, default=100_000, help="Number of validation rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (val uses seed + 1)")
    parser.add_argument(
        "--val-holdout",
        type=float,
        default=0.1,
        help="Fraction of unique items held out for val questions (default: 0.1). "
        "Independent of --val-size, which controls the number of val rows.",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the Hub")
    args = parser.parse_args()

    print(f"Generating dataset | train={args.train_size:,} val={args.val_size:,}\n")

    dataset = build_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        val_holdout=args.val_holdout,
    )

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}  (pass --hub-name to push to the Hub instead)")
