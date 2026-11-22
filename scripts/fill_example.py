"""Fill in the template with its own commented example task (single-digit addition), to see the pipeline run.

    python scripts/fill_example.py [repo_dir]

It uncomments the worked examples in src/train/train_tokenizer.py, src/train/create_dataset.py,
src/train/train_model.py and src/utils/dataset.py, in place. Use it on a copy you want to try out (the
Colab quickstart and the CI smoke test do); for your own task, write those parts yourself instead.
"""

import sys
from pathlib import Path


def _uncomment_example(path: Path, marker: str, indent: str) -> None:
    """Uncomment the example block that starts at `marker` and ends at the next "# ----" line."""
    lines = path.read_text().split("\n")
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        raise SystemExit(f"{path}: example marker {marker!r} not found")
    prefix = indent + "#     "
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("# ----"):
            break
        if lines[i].startswith(prefix):
            lines[i] = indent + lines[i][len(prefix) :]
        elif lines[i].strip() == "#":
            lines[i] = ""
    path.write_text("\n".join(lines))


def _replace(path: Path, old: str, new: str) -> None:
    """Replace `old` with `new` in a file, exactly once."""
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected {old!r} exactly once")
    path.write_text(text.replace(old, new))


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ".") / "src"

    tokenizer = src / "train/train_tokenizer.py"
    _replace(tokenizer, "VOCAB_CHARS: list[str] = []", 'VOCAB_CHARS: list[str] = sorted("0123456789+=\\n")')
    samples = '["2+3=5\\n", "7+5=12\\n", "9+8=17\\n3+3="]'
    _replace(tokenizer, "SAMPLE_TEXTS: list[str] = []", f"SAMPLE_TEXTS: list[str] = {samples}")

    dataset = src / "train/create_dataset.py"
    _uncomment_example(dataset, "# Example (the single-digit addition task below", "    ")  # item_key
    _uncomment_example(dataset, "# Example (single-digit addition, 4-shot, leakage-safe", "    ")  # build_dataset

    model = src / "train/train_model.py"
    _replace(model, 'TOKENIZER_NAME: str = ""', 'TOKENIZER_NAME: str = "./artifacts/tokenizer"')
    _replace(model, 'DATASET_NAME: str = ""', 'DATASET_NAME: str = "./artifacts/dataset"')

    _uncomment_example(src / "utils/dataset.py", "# Example (single-digit addition, 4-shot)", "        ")
    print("Filled in the single-digit addition example.")
