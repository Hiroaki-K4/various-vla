#!/usr/bin/env python3
"""
Download LIBERO datasets (object, goal) and convert to 256 resolution.
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


def resize_dataset(dataset_name: str, target_res: int = 256) -> bool:
    """Resize dataset using rerender_dataset.py"""
    datasets_dir = Path("libero/libero/datasets")
    src_dir = datasets_dir / dataset_name
    dst_dir = datasets_dir / f"{dataset_name}_{target_res}"

    print(f"\n{'='*60}")
    print(f"Converting {dataset_name} to {target_res}x{target_res}...")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "rerender_dataset.py",
        "--src-dir",
        str(src_dir),
        "--dst-dir",
        str(dst_dir),
        "--res",
        str(target_res),
        "--skip-existing",
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to resize {dataset_name}: {e}")
        return False


def main():
    datasets_dir = Path("libero/libero/datasets")
    datasets_to_process = ["libero_object", "libero_goal"]
    target_res = 256

    print("LIBERO Multi-Task Setup with Resize")
    print(f"Target resolution: {target_res}x{target_res}")
    print(f"Target directory: {datasets_dir}")

    # Download missing datasets
    for dataset in datasets_to_process:
        dataset_path = datasets_dir / dataset
        if dataset_path.exists():
            print(f"✓ {dataset} already exists")
        else:
            print(f"⏳ {dataset} not found, downloading...")
            if not download_dataset(dataset):
                print(f"✗ Failed to download {dataset}")
                return 1

    # Resize datasets
    for dataset in datasets_to_process:
        dataset_256_path = datasets_dir / f"{dataset}_{target_res}"
        if dataset_256_path.exists():
            num_files = len(list(dataset_256_path.glob("*.hdf5")))
            print(f"✓ {dataset}_{target_res} already exists ({num_files} files)")
        else:
            if not resize_dataset(dataset, target_res):
                print(f"✗ Failed to resize {dataset}")
                return 1

    # Check all datasets are present
    print(f"\n{'='*60}")
    print("Final dataset status:")
    print(f"{'='*60}")

    all_datasets = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
    ]
    for dataset in all_datasets:
        # Check original
        orig_path = datasets_dir / dataset
        if orig_path.exists():
            num_files = len(list(orig_path.glob("*.hdf5")))
            print(f"✓ {dataset}: {num_files} files")

        # Check 256 version
        res256_path = datasets_dir / f"{dataset}_{target_res}"
        if res256_path.exists():
            num_files = len(list(res256_path.glob("*.hdf5")))
            print(f"✓ {dataset}_{target_res}: {num_files} files")

    print(f"\n✓ Setup complete!")
    print(f"\nTo train with multi-task datasets at {target_res}x{target_res}:")
    print(f"  Update train_libero.py to use:")
    print(f"    dataset_dir = [")
    print(f'      "libero/libero/datasets/libero_spatial_{target_res}",')
    print(f'      "libero/libero/datasets/libero_object_{target_res}",')
    print(f'      "libero/libero/datasets/libero_goal_{target_res}",')
    print(f"    ]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
