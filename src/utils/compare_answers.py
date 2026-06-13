"""Check for mismatches between ground truth answers and model answers in JSON files."""

import argparse
import json
import sys
from pathlib import Path

# Reuse comparison helpers from analysis utilities
_UTIL_PATH = Path(__file__).resolve().parents[1] / "analysis" / "clean"
if str(_UTIL_PATH) not in sys.path:
    sys.path.append(str(_UTIL_PATH))

import util  # type: ignore  # noqa: E402


def check_mismatches(json_path: Path) -> None:
    """Check for answer mismatches in a JSON file and print prompt indices."""
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    print(f"Checking {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    results_list = _normalize_results(data)
    (
        mismatches_original,
        mismatches_overloaded,
        both_correct,
        only_orig_correct,
        only_over_correct,
        both_wrong,
    ) = _evaluate_results(results_list)

    # Print results
    print(f"\nTotal results checked: {len(results_list)}")
    print(
        f"Both correct: {both_correct}, "
        f"Only original correct: {only_orig_correct}, "
        f"Only overloaded correct: {only_over_correct}, "
        f"Both wrong: {both_wrong}"
    )
    print(f"\nOriginal mismatches: {len(mismatches_original)}")
    if mismatches_original:
        print("Prompt indices with original answer mismatches:")
        for prompt_idx, expected, actual in mismatches_original:
            print(f"  Prompt {prompt_idx}: expected {expected}, got {actual}")

    print(f"\nOverloaded mismatches: {len(mismatches_overloaded)}")
    if mismatches_overloaded:
        print("Prompt indices with overloaded answer mismatches:")
        for prompt_idx, expected, actual in mismatches_overloaded:
            print(f"  Prompt {prompt_idx}: expected {expected}, got {actual}")

    if not mismatches_original and not mismatches_overloaded:
        print("\nAll answers match!")


def _normalize_results(data: object) -> list[dict]:
    """Normalize input data to a list of result dicts."""
    if isinstance(data, dict):
        if "result" in data:
            return data["result"]  # type: ignore[return-value]
        if "results" in data:
            return data["results"]  # type: ignore[return-value]
        return [data]  # type: ignore[list-item]
    if isinstance(data, list):
        return data  # type: ignore[return-value]
    raise ValueError("Unexpected data structure")


def _evaluate_results(
    results_list: list[dict],
) -> tuple[
    list[tuple[str, str | int | float | None, str | int | float | None]],
    list[tuple[str, str | int | float | None, str | int | float | None]],
    int,
    int,
    int,
    int,
]:
    """Compute correctness buckets and collect mismatches."""
    mismatches_original: list[tuple[str, str | int | float | None, str | int | float | None]] = []
    mismatches_overloaded: list[tuple[str, str | int | float | None, str | int | float | None]] = []

    both_correct = only_orig_correct = only_over_correct = both_wrong = 0

    for result in results_list:
        prompt_idx = result.get("prompt_idx", "unknown")
        orig_token = result.get("original", {}).get("answer", {}).get("token")
        over_token = result.get("overloaded", {}).get("answer", {}).get("token")
        ans_orig = result.get("answer_original")
        ans_over = result.get("answer_overloaded")

        orig_correct = util.compare_values(orig_token, ans_orig)
        over_correct = util.compare_values(over_token, ans_over)

        if orig_correct and over_correct:
            both_correct += 1
        elif orig_correct and not over_correct:
            only_orig_correct += 1
        elif not orig_correct and over_correct:
            only_over_correct += 1
        else:
            both_wrong += 1

        if not orig_correct:
            mismatches_original.append((prompt_idx, ans_orig, orig_token))

        if not over_correct:
            mismatches_overloaded.append((prompt_idx, ans_over, over_token))

    return (
        mismatches_original,
        mismatches_overloaded,
        both_correct,
        only_orig_correct,
        only_over_correct,
        both_wrong,
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for mismatches between ground truth and model answers in JSON files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file or directory containing JSON files",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if input_path.is_file():
        if not input_path.suffix == ".json":
            raise ValueError(f"Input file must be a .json file: {input_path}")
        check_mismatches(input_path)
    elif input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            print(f"No .json files found in {input_path}")
            return

        print(f"Found {len(json_files)} .json files\n")
        for json_file in json_files:
            try:
                check_mismatches(json_file)
                print("\n" + "=" * 80 + "\n")
            except Exception as e:
                print(f"Error checking {json_file.name}: {e}\n")
                continue
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
