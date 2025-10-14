import os
import shutil
from pathlib import Path

def split_and_move_files(input_dir, output_dir, train_n=600, val_n=200, test_n=200):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = sorted(input_dir.glob("*.root"))

    if len(files) < train_n + val_n + test_n:
        raise ValueError(f"Not enough files ({len(files)}) for split: "
                         f"{train_n + val_n + test_n} required")

    # Define target subfolders
    train_dir = output_dir / "train/raw"
    val_dir = output_dir / "validation/raw"
    test_dir = output_dir / "test/raw"

    # Make sure target directories exist
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split files
    train_files = files[:train_n]
    val_files = files[train_n:train_n + val_n]
    test_files = files[train_n + val_n:train_n + val_n + test_n]

    # Move files to respective directories
    for f in train_files:
        shutil.move(str(f), train_dir / f.name)
    for f in val_files:
        shutil.move(str(f), val_dir / f.name)
    for f in test_files:
        shutil.move(str(f), test_dir / f.name)

    print(f" Moved {len(train_files)} train, {len(val_files)} validation, and {len(test_files)} test files.")

if __name__ == "__main__":
    input_path = "/vols/cms/mm1221/Independent/Files_merge/photons_2_merged/"
    output_path = "/vols/cms/mm1221/Independent/Data/photons_2/"
    split_and_move_files(input_path, output_path)
