"""Strip large tensor fields from saved inference result .pt files to reduce storage size.

After an analysis run you often keep big activation/logit tensors in the saved .pt
file. This script writes a lightweight copy of each file with those heavy fields
removed (keeping metadata, answers, and other small fields).

Implement `strip_result` for your saved result schema -- it should drop the heavy
keys your inference writes. The operator-overloading arithmetic study's version is
kept below as a commented example. The directory-walking `main()` is generic.
"""

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm


def strip_result(data: dict) -> dict:
    """Remove heavy tensor fields from one loaded result dict (in place) and return it.

    Args:
        data: A result dict loaded from a saved .pt file.

    Returns:
        The same dict with large tensor fields removed.
    """
    # ----------------------------------------------------------------------- #
    # TODO: drop the heavy keys your inference saves for your result schema.
    # ----------------------------------------------------------------------- #
    #
    # Example (operator-overloading arithmetic study), where each prompt has
    # "original"/"overloaded" runs carrying activation/geometry tensors:
    #
    #     def _strip_run(run: dict) -> None:
    #         run.pop("output_ids", None)
    #         run.pop("geometry", None)
    #         answer = run.get("answer")
    #         if isinstance(answer, dict):
    #             for key in ("mlp_neurons", "attn_heads", "residual", "embedding", "logits"):
    #                 answer.pop(key, None)
    #
    #     def _strip_prompt_list(prompt_list: list[dict]) -> None:
    #         for prompt in prompt_list:
    #             for run_key in ("original", "overloaded"):
    #                 run = prompt.get(run_key)
    #                 if isinstance(run, dict):
    #                     _strip_run(run)
    #
    #     if "baseline" in data:                       # intervention-mode file
    #         _strip_prompt_list(data["baseline"])
    #         for abl in data.get("ablations", []):
    #             _strip_prompt_list(abl["result"])
    #     elif "result" in data:                       # normal-mode file
    #         _strip_prompt_list(data["result"])
    #     return data
    #
    # ----------------------------------------------------------------------- #
    raise NotImplementedError("Implement strip_result() for your task -- see the example in comments.")


def main() -> None:
    """Strip every .pt file in a directory and write the lightweight copies to a 'stripped' subdir."""
    parser = argparse.ArgumentParser(description="Strip large tensor fields from inference result .pt files.")
    parser.add_argument("--dir", required=True, help="Directory containing inference .pt files.")
    args = parser.parse_args()

    src_dir = Path(args.dir)
    stripped_dir = src_dir / "stripped"
    stripped_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(src_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {src_dir}")

    for pt_file in tqdm(pt_files, desc="Stripping"):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            data = strip_result(data)
            torch.save(data, stripped_dir / pt_file.name)
            del data
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
