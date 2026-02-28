#!/usr/bin/env python3
"""
Download biomedical and benchmark datasets for Phase 1.2 reproduction.

Usage:
    python scripts/get_biomed_dataset.py --dataset davis2017-val
    python scripts/get_biomed_dataset.py --dataset ctc-hela
    python scripts/get_biomed_dataset.py --list

Datasets:
    davis2017-val   - DAVIS 2017 validation set (480p, ~600MB)
    ctc-hela        - Cell Tracking Challenge DIC-C2DH-HeLa (~50MB)
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path


DATASETS = {
    "davis2017-val": {
        # Primary: GitHub release mirror; Fallback: manual Google Drive download
        "url": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
        "fallback_url": "https://drive.google.com/uc?id=1kiVKvKk3mNWXXXXXXXXXXXXXXXXXXXX",  # placeholder
        "manual_url": "https://davischallenge.org/davis2017/code.html",
        "description": "DAVIS 2017 train/val set, 480p resolution",
        "size": "~600MB",
        "extract_to": "davis2017",
    },
    "ctc-hela": {
        "url": "http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",
        "description": "Cell Tracking Challenge - HeLa cells (DIC microscopy)",
        "size": "~50MB",
        "extract_to": "ctc-hela",
    },
    "ctc-hela-gt": {
        "url": "http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",
        "description": "CTC HeLa ground truth (included in main download)",
        "size": "included",
        "extract_to": "ctc-hela",
    },
}


def download_file(url: str, dest_path: str, desc: str = "Downloading") -> None:
    """Download a file with progress indicator."""
    print(f"{desc}: {url}")

    def progress_hook(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size))
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, progress_hook)
    print()


def extract_archive(archive_path: str, dest_dir: str) -> None:
    """Extract zip or tar archive."""
    print(f"Extracting to: {dest_dir}")

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def download_davis2017_val(output_dir: str) -> str:
    """Download DAVIS 2017 validation set."""
    dataset_info = DATASETS["davis2017-val"]
    extract_dir = os.path.join(output_dir, dataset_info["extract_to"])

    if os.path.exists(extract_dir):
        print(f"DAVIS 2017 already exists at: {extract_dir}")
        return extract_dir

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "davis2017.zip")

    download_file(dataset_info["url"], zip_path, "Downloading DAVIS 2017")
    extract_archive(zip_path, extract_dir)

    # Clean up zip
    os.remove(zip_path)

    print(f"DAVIS 2017 ready at: {extract_dir}")
    return extract_dir


def download_ctc_hela(output_dir: str) -> str:
    """Download Cell Tracking Challenge HeLa dataset."""
    dataset_info = DATASETS["ctc-hela"]
    extract_dir = os.path.join(output_dir, dataset_info["extract_to"])

    if os.path.exists(extract_dir):
        print(f"CTC HeLa already exists at: {extract_dir}")
        return extract_dir

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "ctc-hela.zip")

    download_file(dataset_info["url"], zip_path, "Downloading CTC HeLa")
    extract_archive(zip_path, extract_dir)

    # Clean up zip
    os.remove(zip_path)

    print(f"CTC HeLa ready at: {extract_dir}")
    return extract_dir


def list_datasets() -> None:
    """Print available datasets."""
    print("\nAvailable datasets:\n")
    for name, info in DATASETS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    Size: {info['size']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download biomedical datasets for Phase 1.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/biomed",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.print_help()
        print("\nError: --dataset is required (or use --list to see options)")
        sys.exit(1)

    # Resolve output directory
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / args.output_dir

    if args.dataset == "davis2017-val":
        download_davis2017_val(str(output_dir))
    elif args.dataset in ("ctc-hela", "ctc-hela-gt"):
        download_ctc_hela(str(output_dir))

    print("\nDataset ready! See docs/phase1_paper_selection.md for next steps.")


if __name__ == "__main__":
    main()

