"""Parser utilities for adding inference arguments."""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-line arguments for inference to the parser.

    Args:
        parser: The argument parser to add arguments to.
    """
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="The path to the model on Huggingface or the local path to the model.",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Enable 4-bit quantization using BitsAndBytes.",
    )
    parser.add_argument(
        "--layers",
        "-l",
        nargs="+",
        type=int,
        required=False,
        help="List of layer indices to analyze. Defaults to all decoder layers.",
    )
    parser.add_argument(
        "--num-prompts",
        "-p",
        default=1000,
        type=int,
        help="The number of examples to evaluate on. Defaults to 1000.",
    )
    parser.add_argument(
        "--few-shot-examples",
        "-fs",
        type=int,
        required=True,
        help="The number of few-shot examples to use.",
    )
    parser.add_argument(
        "--operator",
        "-o",
        type=str,
        required=True,
        help="The operator to use in the equation.",
    )
    parser.add_argument(
        "--overloading-operator",
        "-oo",
        type=str,
        required=False,
        help="The overloading operator to use in the equation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "-mnt",
        type=int,
        required=False,
        default=200,
        help="The maximum number of new tokens to generate. Defaults to 200.",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        help="Directory where the output .pt file will be saved. Filename is auto-generated from run parameters.",
    )
    parser.add_argument(
        "--max-digits",
        "-md",
        type=int,
        required=False,
        default=1,
        help="Maximum number of digits per operand when generating prompts. Defaults to 1.",
    )
    parser.add_argument(
        "--reverse",
        "-rv",
        action="store_true",
        help="Use reverse-digit answer format (LSD-first), matching models trained with --reverse.",
    )
    parser.add_argument(
        "--pad-zero",
        action="store_true",
        help="Pad operands with leading zeros up to `--max-digits` when generating prompts; answers remain unpadded.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        required=False,
        default=None,
        help="Path to analysis.json. When provided, runs individual neuron ablations after the baseline.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        required=False,
        default=42,
        help="The seed to use for reproducibility. Defaults to 42.",
    )
    parser.add_argument(
        "--capture-geometry",
        action="store_true",
        help="Capture MLP neuron, attention head, and geometry activations. Omit for intervention/ablation runs to reduce file size.",
    )
