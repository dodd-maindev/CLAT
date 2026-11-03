"""Automated downloader for MIL-VT pretrained weights from author's Google Drive."""

import glob
import os
import shutil
import subprocess
import sys


def download_milvt_weights() -> None:
    """Downloads and places MIL-VT pretrained checkpoint into data/weights/."""
    weights_dir = os.path.abspath("data/weights")
    os.makedirs(weights_dir, exist_ok=True)
    target_path = os.path.join(weights_dir, "MIL_VIT_pretrain.pth.tar")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 10 * 1024 * 1024:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"Pretrained weights already present at: {target_path} ({size_mb:.2f} MB)")
        return

    print("Downloading MIL-VT weights from author's Google Drive folder...")
    folder_url = "https://drive.google.com/drive/folders/1YgdhA7BK6Unrs2lOflOd9rPTrwm17gdf"

    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown

    gdown.download_folder(folder_url, output=weights_dir, quiet=False, use_cookies=False)

    candidates = (
        glob.glob(os.path.join(weights_dir, "**", "*.pth.tar"), recursive=True)
        + glob.glob(os.path.join(weights_dir, "**", "*.tar"), recursive=True)
        + glob.glob(os.path.join(weights_dir, "**", "*.pth"), recursive=True)
    )

    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {weights_dir} after download.")

    source_file = candidates[0]
    for f in candidates:
        name_lower = os.path.basename(f).lower()
        if "fundus" in name_lower or "vt_small" in name_lower or "mil_vit" in name_lower:
            source_file = f
            break

    if os.path.abspath(source_file) != os.path.abspath(target_path):
        print(f"Moving {source_file} -> {target_path}")
        shutil.copyfile(source_file, target_path)

    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"Successfully prepared MIL-VT weights: {target_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_milvt_weights()
