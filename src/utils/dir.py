"""Build the output path/filename for saved inference results.

`generate_output_path` has a sensible default that encodes every generic run parameter
(model, prompts, layers, capture and intervention settings, seed, ...), so runs that differ in
any of them never overwrite each other. Add your own task-specific arguments to it (see the
commented example) so those are covered too.
"""

import argparse
from pathlib import Path


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

    model_name = str(args.model_path).replace("/", "--")
    intervention = args.intervention
    if intervention is not None and intervention != "all":
        intervention = Path(intervention).stem  # e.g. "analysis" for analysis.json
    components = [
        f"[m={model_name}]",
        f"[p={args.num_prompts}]",
        f"[l={'-'.join(map(str, args.layers)) if args.layers else 'all'}]",
        f"[mnt={args.max_new_tokens}]",
        f"[acts={args.capture_activations}]",
        f"[logits={args.capture_logits}]",
        f"[int={intervention}]",
        f"[seed={args.seed}]",
    ]

    # ----------------------------------------------------------------------- #
    # TODO (optional): add your task-specific arguments to the filename, so runs
    # that differ in them don't overwrite each other.
    # ----------------------------------------------------------------------- #
    #
    # Example (a task with an extra --max-digits argument):
    #
    #     components.append(f"[digits={args.max_digits}]")
    #
    # ----------------------------------------------------------------------- #
    return "_".join(components) + ".pt"
