"""Command-line arguments for the inference / activation-extraction entry point (src/main.py).

The generic run arguments (model, layers, output, ...) are provided as-is. Add any
task-specific arguments your prompts or analysis need in the marked TODO section;
the operator-overloading arithmetic study's arguments are kept below as a commented
example.
"""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add inference command-line arguments to the parser.

    Args:
        parser: The argument parser to add arguments to.
    """
    # --- Generic run arguments (useful for most analysis runs) ---
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="HuggingFace repo or local path to the model.",
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
        help="Number of prompts to evaluate on. Defaults to 1000.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "-mnt",
        type=int,
        required=False,
        default=200,
        help="Maximum number of new tokens to generate. Defaults to 200.",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        help="Directory where the output file will be saved.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )

    # --- Core mechinterp toggles (capture + intervention) ---
    # The flags are generic; their specifics are defined by your inference code:
    #   - what activations / token positions --capture-geometry records, and
    #   - the file format the --intervention spec is read from.
    parser.add_argument(
        "--capture-geometry",
        action="store_true",
        help="Capture detailed per-position activations (MLP neurons, attention heads, residual stream) "
        "in addition to the model output. Omit for lightweight runs such as large ablation sweeps.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Path to an intervention spec (e.g. a JSON listing neurons/heads to ablate or patch). "
        "When provided, an intervention pass is run after the baseline.",
    )

    # ----------------------------------------------------------------------- #
    # TODO: add task-specific arguments for your study here.
    # ----------------------------------------------------------------------- #
    #
    # Example (operator-overloading arithmetic study):
    #
    #     parser.add_argument("--operator", "-o", type=str, required=True,
    #                         help="The operator to use in the equation.")
    #     parser.add_argument("--overloading-operator", "-oo", type=str, required=False,
    #                         help="The overloading operator to use in the equation.")
    #     parser.add_argument("--few-shot-examples", "-fs", type=int, required=True,
    #                         help="Number of few-shot examples per prompt.")
    #     parser.add_argument("--max-digits", "-md", type=int, default=1,
    #                         help="Maximum number of digits per operand.")
    #     parser.add_argument("--reverse", "-rv", action="store_true",
    #                         help="Use reverse-digit answer format (LSD-first).")
    #     parser.add_argument("--pad-zero", action="store_true",
    #                         help="Pad operands with leading zeros up to --max-digits.")
    #
    # ----------------------------------------------------------------------- #
