"""Simple HuggingFace Hub Management Tool."""

import argparse
import sys

from huggingface_hub import HfApi, list_datasets, list_models
from tqdm.auto import tqdm


class HFHubManager:
    """Manager for HuggingFace Hub operations."""

    def __init__(self) -> None:
        """Initialize the HF Hub manager and authenticate."""
        self.api = HfApi()
        self.organization = "arithmetic-circuit-overloading"
        self.username = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with HuggingFace Hub."""
        try:
            whoami_info = self.api.whoami()
            self.username = whoami_info["name"]
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please login with: huggingface-cli login")
            sys.exit(1)

    def get_models(self) -> list:
        """Get organization models."""
        return list(list_models(author=self.organization))

    def get_datasets(self) -> list:
        """Get organization datasets."""
        return list(list_datasets(author=self.organization))

    def _delete_repos(self, items: list, repo_type: str) -> None:
        """Delete repositories with progress tracking.

        Args:
            items: List of items to delete
            repo_type: Type of repository ('model' or 'dataset')
        """
        for item in tqdm(items):
            try:
                self.api.delete_repo(repo_id=item.id, repo_type=repo_type)
                print(f"Deleted: {item.id}")
            except Exception as e:
                print(f"Failed: {item.id}: {e}")

    def list_models(self) -> None:
        """List all models."""
        models = self.get_models()
        if not models:
            print("No models found.")
            return

        for model in models:
            print(model.id)

    def list_datasets(self) -> None:
        """List all datasets."""
        datasets = self.get_datasets()
        if not datasets:
            print("No datasets found.")
            return

        for dataset in datasets:
            print(f"{dataset.id}")

    def delete_all_models(self) -> None:
        """Delete all models except tokenizer."""
        models = self.get_models()
        # Filter out tokenizer
        to_delete = [m for m in models if not m.id.endswith("/tokenizer")]

        if not to_delete:
            print("No models to delete (only tokenizer found).")
            return

        print(f"Deleting {len(to_delete)} models...")
        self._delete_repos(to_delete, "model")

    def delete_all_datasets(self) -> None:
        """Delete all datasets."""
        datasets = self.get_datasets()
        if not datasets:
            print("No datasets to delete.")
            return

        print(f"Deleting {len(datasets)} datasets...")
        self._delete_repos(datasets, "dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage HF Hub models/datasets for the organization")
    parser.add_argument(
        "--list",
        "-l",
        choices=["model", "dataset"],
        nargs="+",
        help="List all models and/or datasets. Example: --list model dataset",
    )
    parser.add_argument(
        "--delete",
        "-d",
        choices=["model", "dataset"],
        nargs="+",
        help="Delete all models and/or datasets. Example: --delete model dataset",
    )

    args = parser.parse_args()

    if not args.list and not args.delete:
        parser.print_help()
        sys.exit(0)

    manager = HFHubManager()

    # perform requested listings
    if args.list:
        if "model" in args.list:
            manager.list_models()
        if "dataset" in args.list:
            manager.list_datasets()

    # perform requested deletions
    if args.delete:
        if "model" in args.delete:
            manager.delete_all_models()
        if "dataset" in args.delete:
            manager.delete_all_datasets()
