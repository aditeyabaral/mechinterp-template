"""Build the output path/filename for saved inference results.

Implement `generate_output_path` to encode a run's parameters into a filename so
that different runs don't overwrite each other. The operator-overloading arithmetic
study's version is kept below as a commented example.
"""

import argparse


def generate_output_path(args: argparse.Namespace) -> str:
    """Generate the output path for a run's results.

    If `args.output` is set, it is honored directly. Otherwise, build a filename
    from the run parameters in `args`.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The output path for the run's results.
    """
    if args.output is not None:
        return args.output

    # ----------------------------------------------------------------------- #
    # TODO: build a filename that captures this run's parameters.
    # ----------------------------------------------------------------------- #
    #
    # Example (operator-overloading arithmetic study):
    #
    #     def _sanitize_model_path(s: str) -> str:
    #         return str(s).replace("/", "--")
    #
    #     components = [
    #         f"[m={_sanitize_model_path(args.model_path)}]",
    #         f"[p={args.num_prompts}]",
    #         f"[fs={args.few_shot_examples}]",
    #         f"[op={args.operator}]",
    #     ]
    #     if args.overloading_operator is not None:
    #         components.append(f"[oop={args.overloading_operator}]")
    #     return "_".join(components) + ".pt"
    #
    # ----------------------------------------------------------------------- #
    raise NotImplementedError("Implement generate_output_path() for your task -- see the example in comments.")
