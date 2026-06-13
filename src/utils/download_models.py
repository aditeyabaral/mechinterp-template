"""Script to download models from Hugging Face Hub into the local cache."""

import argparse

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub into the local cache.")
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="List of model IDs to download from Hugging Face Hub.",
    )
    args = parser.parse_args()

    for model_id in args.models:
        print(f"Downloading {model_id} into HF cache...")
        snapshot_download(repo_id=model_id, force_download=True)
        print(f"Finished downloading {model_id}.")
