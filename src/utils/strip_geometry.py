"""Strip large geometry fields from inference result files to reduce storage size."""

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm


def _strip_run(run: dict) -> None:
    run.pop("output_ids", None)
    run.pop("geometry", None)
    answer = run.get("answer")
    if isinstance(answer, dict):
        for key in ("mlp_neurons", "attn_heads", "residual", "embedding", "logits"):
            answer.pop(key, None)


def _strip_prompt_list(prompt_list: list[dict]) -> None:
    for prompt in prompt_list:
        for run_key in ("original", "overloaded"):
            run = prompt.get(run_key)
            if isinstance(run, dict):
                _strip_run(run)


def strip_geometry(data: dict) -> dict:
    """Strip geometry fields from a loaded save_data dict (in-place).

    Handles both normal mode ({"result": [...]}) and intervention mode
    ({"baseline": [...], "ablations": [...]}).
    """
    if "baseline" in data:
        _strip_prompt_list(data["baseline"])
        for abl in data.get("ablations", []):
            _strip_prompt_list(abl["result"])
    elif "result" in data:
        _strip_prompt_list(data["result"])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip geometry fields from inference result .pt files.")
    parser.add_argument("--dir", required=True, help="Directory containing inference .pt files.")
    args = parser.parse_args()

    src_dir = Path(args.dir)
    stripped_dir = src_dir / "stripped"
    stripped_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(f for f in src_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {src_dir}")

    for pt_file in tqdm(pt_files, desc="Stripping geometry"):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            data = strip_geometry(data)
            torch.save(data, stripped_dir / pt_file.name)
            del data
        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
