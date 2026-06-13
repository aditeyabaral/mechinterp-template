"""Generate the arithmetic overloading HuggingFace dataset for training small language models.

Produces a single 8-shot HF dataset with 6 configs:
  - "100": baseline, 100% standard arithmetic across all 6 operators
  - "{split_pct}": 1 operator combo (all 6 ops) x 5 split percentages (99, 95, 90, 75, 50)

Each config has train and validation splits with independently specified sizes.
The standard portion always covers all 6 operators equally to avoid cross-config confounds.

Train/val leakage is prevented by partitioning the (a, b) operand pairs per (base_op, target_op)
pair before sampling:
  - Train examples: question and all 8 few-shot examples draw from the train pool.
  - Val examples:   question draws from the val pool; few-shot examples draw from the train pool.

This guarantees:
  - A val question (op, a, b) is never seen during training in any form (question or few-shot).
  - Val few-shot examples use familiar train-pool pairs, so the model can recognise the pattern;
    the test is whether it generalises that pattern to the unseen val question.
"""

import argparse
import os
import random
import sys
from itertools import permutations

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import PromptDataset

# All supported operators, derived from PromptDataset's symbol map
ALL_OPERATORS = list(PromptDataset.OPERATOR_SYMBOL_TO_NAME_MAP.keys())

# Standard split percentages (percentage of examples that are standard arithmetic)
SPLIT_PERCENTAGES = [99, 95, 90, 75, 50]

FEW_SHOT = 8


def build_pair_pools(
    operators: list[str],
    standard_pct: int,
    max_digits: int,
    seed: int,
    pair_holdout: float,
) -> dict[tuple[str, str], dict[str, list[tuple[int, int]]]]:
    """Partition valid (a, b) pairs into train/val pools per (base_op, target_op) pair.

    Each (base_op, target_op) pair gets its own independent shuffle and split, so:
      - No (base_op, a, b) triple ever appears in both train and val questions or few-shot.
      - Different operator pairs are partitioned independently: (3, 5) may be
        in train for +→+ and in val for *→* with no leakage between them.

    Args:
        operators: Operator combo whose overloaded pairs to include.
        standard_pct: Percentage of examples that are standard arithmetic.
        max_digits: Maximum number of digits for each operand.
        seed: Base random seed; each operator pair uses seed + its enumeration index.
        pair_holdout: Fraction of unique (a, b) pairs reserved for validation questions.
            Independent of --val-ratio, which controls the number of val examples.

    Returns:
        Dict mapping (base_op, target_op) -> {"train": [...], "val": [...]}.
    """
    max_op = 10**max_digits - 1

    # Collect all (base_op, target_op) pairs this config uses
    all_pairs: list[tuple[str, str]] = [(op, op) for op in ALL_OPERATORS]
    if standard_pct < 100:
        all_pairs.extend(permutations(operators, 2))

    pools: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = {}
    for i, (base_op, target_op) in enumerate(all_pairs):
        needs_nonzero_b = {"/", "%"}
        min_op_b = 1 if (base_op in needs_nonzero_b or target_op in needs_nonzero_b) else 0

        candidates = [(a, b) for a in range(max_op + 1) for b in range(min_op_b, max_op + 1)]

        # Independent deterministic shuffle per operator pair
        rng = random.Random(seed + i)
        rng.shuffle(candidates)

        split_idx = int(len(candidates) * (1 - pair_holdout))
        pools[(base_op, target_op)] = {
            "train": candidates[:split_idx],
            "val": candidates[split_idx:],
        }

    return pools


def _reverse_digits(answer: str) -> str:
    """Reverse the digit order of an answer string, moving any minus sign to the end.

    For positive numbers, digits are simply reversed (LSD first).
    For negative numbers, the sign is placed LAST because column subtraction
    determines the sign only after processing the most significant digit — the
    final unresolved borrow reveals negativity at the end of computation, so
    the sign token should also be generated last.

    Examples:
        "495" -> "594"
        "-46" -> "64-"
        "7"   -> "7"
    """
    return answer[::-1]


def generate_example(
    example_id: int,
    base_op: str,
    target_op: str,
    max_digits: int = 1,
    question_ab: tuple[int, int] | None = None,
    fs_pool: list[tuple[int, int]] | None = None,
    reverse: bool = False,
    pad_zero: bool = False,
) -> dict:
    """Generate a single dataset row for a given base->target operation pair.

    For standard examples, base_op == target_op and few-shot answers follow
    normal arithmetic. For overloaded examples, few-shot answers follow
    the target_op semantics applied to the base_op expression.

    Args:
        example_id: Unique integer ID for this example.
        base_op: The operator symbol used in the expression (e.g. "+").
        target_op: The operator whose semantics determine the answer. Equal to
            base_op for standard examples, different for overloaded ones.
        max_digits: Maximum number of digits for each operand.
        question_ab: If provided, use this (a, b) pair as the question operands
            instead of sampling randomly.
        fs_pool: If provided, sample all few-shot (a, b) pairs from this pool
            instead of the full operand range. Should always be the train pool
            to prevent val-pool pairs from leaking into training contexts.
        reverse: If True, reverse the digit order of all answers (few-shot and
            question), following the technique from arxiv.org/abs/2307.03381.
        pad_zero (bool): If True, pad operands in the printed expression with leading
            zeros up to `max_digits`. Answers are computed from integer values and
            are NOT padded. Defaults to False.
    """
    is_standard = base_op == target_op
    overloading = not is_standard
    overloading_operator = None if is_standard else target_op

    fs_examples = []
    for _ in range(FEW_SHOT):
        if fs_pool is not None:
            fs_a, fs_b = fs_pool[random.randint(0, len(fs_pool) - 1)]
        else:
            fs_a, fs_b = None, None
        expression, answer_orig, answer_ovld = PromptDataset.generate_equation(
            overloading=overloading,
            operator=base_op,
            overloading_operator=overloading_operator,
            max_digits=max_digits,
            a=fs_a,
            b=fs_b,
            pad_zero=pad_zero,
        )
        answer = str(answer_orig if is_standard else answer_ovld)
        if reverse:
            answer = _reverse_digits(answer)
        fs_examples.append(f"{expression}={answer}")

    qa, qb = question_ab if question_ab is not None else (None, None)
    expression, answer_orig, answer_ovld = PromptDataset.generate_equation(
        overloading=overloading,
        operator=base_op,
        overloading_operator=overloading_operator,
        max_digits=max_digits,
        a=qa,
        b=qb,
        pad_zero=pad_zero,
    )
    answer = str(answer_orig if is_standard else answer_ovld)
    if reverse:
        answer = _reverse_digits(answer)
    prompt = "\n".join(fs_examples) + f"\n{expression}="

    return {
        "_id": str(example_id),
        "base_operation": base_op,
        "target_operation": target_op,
        "fs_examples": fs_examples,
        "question": expression,
        "answer": answer,
        "prompt": prompt,
    }


def generate_split(
    total: int,
    standard_pct: int,
    operators: list[str],
    seed: int,
    split_name: str = "split",
    max_digits: int = 1,
    pools: dict | None = None,
    is_val: bool = False,
    reverse: bool = False,
    pad_zero: bool = False,
) -> dict:
    """Generate all examples for one split (train or val).

    Question (a, b) pairs are sampled from the split-specific pool (train or val).
    Few-shot (a, b) pairs are always sampled from the train pool, so that:
      - Val question pairs are never seen during training in any form.
      - Few-shot examples use familiar pairs that demonstrate the pattern clearly.

    Args:
        total: Total number of examples to generate.
        standard_pct: Percentage of examples that are standard arithmetic (0-100).
        operators: Operator combo whose overloaded pairs to include (ignored if standard_pct==100).
        seed: Random seed for reproducibility.
        split_name: Label shown in the progress bar (e.g. "train", "val").
        max_digits: Maximum number of digits for each operand.
        pools: Pre-partitioned (a, b) pair pools from build_pair_pools. If None, operands
            are sampled freely with no leakage guarantee.
        is_val: Whether this is the validation split (selects the "val" question sub-pool).
        reverse: If True, reverse the digit order of all answers in generated examples.
        pad_zero (bool): If True, pad operands in the printed expression with leading
            zeros up to `max_digits`. Answers are computed from integer values and
            are NOT padded. Defaults to False.

    Returns:
        Dict of lists ready for Dataset.from_dict().
    """
    random.seed(seed)
    pool_rng = np.random.default_rng(seed)
    split_key = "val" if is_val else "train"

    examples: list[dict] = []
    eid = 0

    def emit_examples(base_op: str, target_op: str, count: int, pbar: tqdm) -> None:
        """Append `count` examples for one (base_op, target_op) pair."""
        nonlocal eid
        if pools is not None:
            q_pool = pools[(base_op, target_op)][split_key]
            fs_pool = pools[(base_op, target_op)]["train"]
            q_indices = pool_rng.integers(len(q_pool), size=count)
            abs_ = [q_pool[idx] for idx in q_indices]
        else:
            fs_pool = None
            abs_ = [None] * count
        for ab in abs_:
            examples.append(
                generate_example(eid, base_op, target_op, max_digits, ab, fs_pool, reverse=reverse, pad_zero=pad_zero)
            )
            examples[-1]["_id"] = f"{split_name}-{eid}"
            eid += 1
            pbar.update(1)

    def distribute(subtotal: int, op_pairs: list[tuple[str, str]], pbar: tqdm) -> None:
        """Distribute `subtotal` examples evenly across `op_pairs`."""
        n = len(op_pairs)
        per = subtotal // n
        remainder = subtotal % n
        for i, (base_op, target_op) in enumerate(op_pairs):
            emit_examples(base_op, target_op, per + (1 if i < remainder else 0), pbar)

    with tqdm(total=total, desc=f"  {split_name}", leave=False) as pbar:
        if standard_pct == 100:
            distribute(total, [(op, op) for op in ALL_OPERATORS], pbar)
        else:
            standard_total = round(total * standard_pct / 100)
            distribute(standard_total, [(op, op) for op in ALL_OPERATORS], pbar)
            distribute(total - standard_total, list(permutations(operators, 2)), pbar)

    random.shuffle(examples)
    return {k: [ex[k] for ex in examples] for k in examples[0]}


def build_config(
    operators: list[str],
    standard_pct: int,
    train_size: int,
    val_size: int,
    seed: int,
    max_digits: int = 1,
    pair_holdout: float = 0.1,
    reverse: bool = False,
    pad_zero: bool = False,
) -> DatasetDict:
    """Build a DatasetDict with train and validation splits for one config.

    Args:
        operators: Operator combo whose overloaded pairs to include.
        standard_pct: Percentage of examples that are standard arithmetic (0-100).
        train_size: Number of examples in the training split.
        val_size: Number of examples in the validation split.
        seed: Random seed for the training split (val uses seed+1).
        max_digits: Maximum number of digits for each operand.
        pair_holdout: Fraction of unique (a, b) pairs reserved for validation questions.
            Controls what the model is tested on, independent of val_size.
        reverse: If True, reverse the digit order of all answers in generated examples.
        pad_zero (bool): If True, pad operands in the printed expression with leading
            zeros up to `max_digits`. Answers are computed from integer values and
            are NOT padded. Defaults to False.

    Returns:
        DatasetDict with "train" and "validation" keys.
    """
    pools = build_pair_pools(operators, standard_pct, max_digits, seed, pair_holdout)
    train_data = generate_split(
        train_size,
        standard_pct,
        operators,
        seed,
        split_name="train",
        max_digits=max_digits,
        pools=pools,
        is_val=False,
        reverse=reverse,
        pad_zero=pad_zero,
    )
    val_data = generate_split(
        val_size,
        standard_pct,
        operators,
        seed + 1,
        split_name="val",
        max_digits=max_digits,
        pools=pools,
        is_val=True,
        reverse=reverse,
        pad_zero=pad_zero,
    )
    return DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "validation": Dataset.from_dict(val_data),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arithmetic overloading HF dataset")
    parser.add_argument("--train-size", type=int, default=1_000_000, help="Number of training examples per config")
    parser.add_argument("--val-size", type=int, default=100_000, help="Number of validation examples per config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (val uses seed+1)")
    parser.add_argument("--n", type=int, default=1, help="Maximum number of digits per operand")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HuggingFace Hub repo (e.g. username/dataset-name); defaults to synthetic-dataset-{--n}d",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the hub")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the digit order of all answers (few-shot and question), following arxiv.org/abs/2307.03381.",
    )
    parser.add_argument(
        "--pad-zero",
        action="store_true",
        help="Pad all operands with leading zeros up to `--n` digits in prompts (answers remain unpadded).",
    )
    parser.add_argument(
        "--pair-holdout",
        type=float,
        default=0.1,
        help="Fraction of unique (a,b) operand pairs held out for val questions (default: 0.1). "
        "Independent of --val-size, which controls the number of val examples.",
    )
    args = parser.parse_args()

    def _size_str(n: int) -> str:
        if n % 1_000_000 == 0:
            return f"{n // 1_000_000}M"
        if n % 1_000 == 0:
            return f"{n // 1_000}K"
        return str(n)

    if args.hub_name is None:
        suffix_parts = []
        if args.reverse:
            suffix_parts.append("reverse")
        if args.pad_zero:
            suffix_parts.append("padzero")
        suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
        args.hub_name = (
            f"arithmetic-circuit-overloading/synthetic-dataset-v2-{args.n}d"
            f"-{_size_str(args.train_size)}"
            f"-{_size_str(args.val_size)}"
            f"-{args.pair_holdout}"
            f"{suffix}"
        )

    val_size = args.val_size

    # Build list of all 6 configs: 1 baseline + 5 splits
    configs: list[tuple[str, list[str], int]] = [("100", ALL_OPERATORS, 100)]
    for pct in SPLIT_PERCENTAGES:
        configs.append((str(pct), ALL_OPERATORS, pct))

    print(f"Generating {len(configs)} configs | train={args.train_size:,} val={val_size:,} per config")
    print(f"Total examples: {len(configs) * (args.train_size + val_size):,}\n")

    for config_name, operators, standard_pct in tqdm(configs, desc="Configs"):
        dataset = build_config(
            operators=operators,
            standard_pct=standard_pct,
            train_size=args.train_size,
            val_size=val_size,
            seed=args.seed,
            max_digits=args.n,
            pair_holdout=args.pair_holdout,
            reverse=args.reverse,
            pad_zero=args.pad_zero,
        )

        dataset.push_to_hub(args.hub_name, config_name=config_name, max_shard_size=args.shard_size)
        tqdm.write(f"  Pushed {config_name} to hub")
