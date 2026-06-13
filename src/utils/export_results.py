"""Convert .pt inference result files to JSON, removing activations and logits."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def remove_tensors_recursive(obj: Any) -> Any:  # noqa: ANN401
    """Recursively remove all torch.Tensor objects from a data structure.

    Args:
        obj: Any object (dict, list, tensor, etc.)

    Returns:
        Object with all tensors removed (replaced with None or removed from dicts)
    """
    if isinstance(obj, torch.Tensor):
        return None
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = remove_tensors_recursive(value)
            # Remove keys that had tensors (now None), but keep keys that were originally None
            if cleaned_value is not None or value is None:
                cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list | tuple):  # noqa: UP038
        # Keep all items, even if they became None (they were tensors)
        return [remove_tensors_recursive(item) for item in obj]
    # For other types (str, int, float, bool, None), return as-is
    return obj


def remove_tensors_from_result(result: dict) -> dict:
    """Remove tensor data (activations, logits, attentions, output_ids) from a result dict.

    Args:
        result: A single result dictionary from inference

    Returns:
        A cleaned result dictionary with only JSON-serializable data
    """
    return remove_tensors_recursive(result)


def convert_pt_to_json(pt_path: Path, output_path: Path) -> None:
    """Convert a .pt file to JSON, removing tensor data.

    Args:
        pt_path: Path to the .pt file
        output_path: Output path for the JSON file
    """
    if not pt_path.exists():
        raise FileNotFoundError(f"File not found: {pt_path}")

    # Load the .pt file
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, map_location="cpu")

    # Handle different data structures
    if isinstance(data, dict):
        if "result" in data:
            results_list = data["result"]
            metadata = data.get("metadata", {})
        elif "results" in data:
            results_list = data["results"]
            metadata = data.get("metadata", {})
        else:
            # Assume the whole dict is a single result
            results_list = [data]
            metadata = {}
    elif isinstance(data, list):
        results_list = data
        metadata = {}
    else:
        raise ValueError(f"Unexpected data structure in {pt_path}")

    # Clean all results
    cleaned_results = [remove_tensors_from_result(result) for result in results_list]

    # Create output dict
    output_data = {
        "result": cleaned_results,
        "metadata": metadata,
    }

    # Save as JSON
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Converted {len(cleaned_results)} results, removed tensors")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert .pt inference result files to JSON, removing activations and logits"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing .pt files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory (default: input_dir/json)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input must be a directory: {input_path}")

    # Find all .pt files in the input directory
    pt_files = list(input_path.glob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {input_path}")
        return

    print(f"Found {len(pt_files)} .pt files")

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: input_dir/json
        output_dir = input_path / "json"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each .pt file
    for pt_file in pt_files:
        output_path = output_dir / pt_file.with_suffix(".json").name

        try:
            convert_pt_to_json(pt_file, output_path)
        except Exception as e:
            print(f"Error converting {pt_file.name}: {e}")
            continue


if __name__ == "__main__":
    main()
