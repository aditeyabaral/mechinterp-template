"""Utility functions for directory operations."""

import argparse

import utils


def _sanitize_model_path(s: str) -> str:
    """Sanitizes the model path by replacing '/' with '--'."""
    return str(s).replace("/", "--")


def _sanitize_operator(op: str) -> str:
    """Sanitizes the operator by mapping it to its name."""
    return utils.dataset.PromptDataset.OPERATOR_SYMBOL_TO_NAME_MAP.get(op, op)


def generate_output_path(args: argparse.Namespace) -> str:
    """Generates and returns the output path.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The generated output path.
    """
    if args.output is not None:
        return args.output

    intervention = getattr(args, "intervention", None) is not None
    base_components = [
        f"[m={_sanitize_model_path(args.model_path)}]",
        f"[p={args.num_prompts}]",
        f"[fs={args.few_shot_examples}]",
        f"[rv={getattr(args, 'reverse', False)}]",
        f"[pz={getattr(args, 'pad_zero', False)}]",
        f"[op={_sanitize_operator(args.operator)}]",
    ]
    if args.overloading_operator is not None:
        base_components.append(f"[oop={_sanitize_operator(args.overloading_operator)}]")
    base_components.append(f"[int={intervention}]")

    return "_".join(base_components) + ".pt"
