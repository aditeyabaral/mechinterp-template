"""Build the output path/filename for saved inference results.

`generate_output_path` has a sensible default (model name + #prompts), so the pipeline
runs as-is. Customize it to encode more of your run's parameters into the filename so
different runs don't overwrite each other. The arithmetic study's version is in comments.
"""

import argparse


def generate_output_path(args: argparse.Namespace) -> str:
    """Generate the output path for a run's results.

    If `args.output` is set, it is honored directly. Otherwise, a filename is built
    from the run parameters in `args`.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The output path for the run's results.
    """
    if args.output is not None:
        return args.output

    # ----------------------------------------------------------------------- #
    # TODO: customize the filename to capture this run's parameters so that
    # different runs don't overwrite each other. The default below uses just the
    # model name and prompt count.
    # ----------------------------------------------------------------------- #
    #
    # Example (operator-overloading arithmetic study), adding the operators used:
    #
    #     components = [
    #         f"[m={model_name}]",
    #         f"[p={args.num_prompts}]",
    #         f"[op={args.operator}]",
    #     ]
    #     if args.overloading_operator is not None:
    #         components.append(f"[oop={args.overloading_operator}]")
    #     return "_".join(components) + ".pt"
    #
    # ----------------------------------------------------------------------- #
    model_name = str(args.model_path).replace("/", "--")
    intervention = getattr(args, "intervention", None) is not None
    return f"[m={model_name}]_[p={args.num_prompts}]_[int={intervention}].pt"
