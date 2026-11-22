"""Command-line arguments for the inference / activation-extraction entry point (src/main.py).

The generic run arguments (model, layers, output, ...) are provided as-is. Add any
task-specific arguments your prompts or analysis need in the marked TODO section;
a small worked example is kept below as a commented illustration.
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
        "--layers",
        "-l",
        nargs="+",
        type=int,
        required=False,
        help="Layer indices to capture from (and to ablate with --intervention all). Defaults to all decoder layers.",
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
        default=None,
        help="Maximum number of new tokens to generate. Defaults to the longest ground-truth answer (in "
        "tokens) plus one for <eos>, or 16 if the prompts carry no 'answer'.",
    )
    parser.add_argument(
        "--output",
        "-out",
        type=str,
        help="Path of the output .pt file. Defaults to a name built from the run's parameters (src/utils/dir.py).",
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
    #   - what activations / token positions --capture-activations records, and
    #   - the file format the --intervention spec is read from.
    parser.add_argument(
        "--capture-activations",
        action="store_true",
        help="Capture detailed per-position activations (MLP neurons, attention heads, residual stream) "
        "at every answer token and at find_positions_of_interest's prompt positions, in addition to the "
        "model output. Not needed for an ablation sweep (--intervention), which only scores answers.",
    )
    parser.add_argument(
        "--capture-logits",
        action="store_true",
        help="Also store the logits that predict each answer token and the <eos> after it. They are often "
        "most of a file's size, and can be rebuilt from the last layer's residual stream.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Path to an intervention spec (e.g. the output json of src/analysis/lasso.py) listing "
        "neurons/heads to ablate, or 'all' to ablate every MLP neuron and attention head of --layers "
        "one at a time. When provided, an ablation sweep is run after the baseline.",
    )
    parser.add_argument(
        "--ablation-type",
        choices=["zero", "mean"],
        default="zero",
        help="How the sweep switches a component off: 'zero' (default) sets it to 0; 'mean' sets it to its average "
        "over every prompt-and-answer token.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Sequences per forward pass in the teacher-forced ablation sweep. Defaults to 256; lower it "
        "if you run out of GPU memory.",
    )

    # ----------------------------------------------------------------------- #
    # TODO: add task-specific arguments for your study here.
    # ----------------------------------------------------------------------- #
    #
    # Example (a single-digit addition task, e.g. prompts like "7+5="):
    #
    #     parser.add_argument("--few-shot-examples", "-fs", type=int, default=4,
    #                         help="Number of solved examples to show before the question.")
    #     parser.add_argument("--max-digits", "-md", type=int, default=1,
    #                         help="Maximum number of digits per operand.")
    #
    # You would then forward these from src/main.py into PromptDataset.generate_prompts.
    #
    # ----------------------------------------------------------------------- #
