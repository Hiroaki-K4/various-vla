#!/usr/bin/env python3
"""
Download LIBERO object and goal datasets.
spatial is assumed to already exist.
"""

import subprocess
import sys
from pathlib import Path


def download_dataset(dataset_name: str) -> bool:
    """Download a specific LIBERO dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "libero/benchmark_scripts/download_libero_datasets.py",
        "--datasets",
        dataset_name,
        "--use-huggingface",
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {dataset_name}: {e}")
        return False


def main():
    datasets_dir = Path("libero/libero/datasets")
    datasets_to_download = ["libero_object", "libero_goal"]

    print("LIBERO Multi-Task Setup")
    print(f"Target directory: {datasets_dir}")

    # Download missing datasets
    for dataset in datasets_to_download:
        dataset_path = datasets_dir / dataset
        if dataset_path.exists():
            print(f"✓ {dataset} already exists")
        else:
            print(f"⏳ {dataset} not found, downloading...")
            if not download_dataset(dataset):
                print(f"✗ Failed to download {dataset}")
                return 1

    # Check all datasets are present
    print(f"\n{'='*60}")
    print("Final dataset status:")
    print(f"{'='*60}")

    all_datasets = ["libero_spatial", "libero_object", "libero_goal"]
    for dataset in all_datasets:
        dataset_path = datasets_dir / dataset
        if dataset_path.exists():
            num_files = len(list(dataset_path.glob("*.hdf5")))
            print(f"✓ {dataset}: {num_files} files")
        else:
            print(f"✗ {dataset}: NOT FOUND")

    print(f"\n✓ Download complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
