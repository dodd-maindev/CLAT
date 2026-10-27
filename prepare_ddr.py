"""Preprocessing utility for DDR retinal fundus dataset."""

import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Set, Tuple
import cv2
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


class DdrDatasetPreprocessor:
    """Preprocesses DDR dataset by cropping black borders and resizing to square."""

    def __init__(
        self,
        source_directory: str = "~/workspace_clat/DDR_extracted",
        target_directory: str = "data/DDR/fundus_384",
        annotation_path: str = "data/annotation_DDR_disease.csv",
        image_size: int = 384,
        max_workers: int = 16,
    ) -> None:
        """Initializes the dataset preprocessor parameters."""
        self.source_dir = os.path.expanduser(source_directory)
        self.target_dir = target_directory
        self.annotation_path = annotation_path
        self.image_size = image_size
        self.max_workers = max_workers

    def _auto_extract_zip_parts(self) -> None:
        """Extracts multi-part zip archives if extracted directory is empty."""
        zip_candidates = sorted(glob(os.path.expanduser("~/workspace_clat/DDR-Dataset/*.zip")))
        if not zip_candidates:
            return
        os.makedirs(self.source_dir, exist_ok=True)
        print(f"Found {len(zip_candidates)} archives. Extracting to {self.source_dir}...")
        for archive_path in tqdm(zip_candidates, desc="Extracting archives"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(self.source_dir)

    def _pad_square(self, image: Image.Image) -> Image.Image:
        """Pads an image with zeros to make width and height equal."""
        width, height = image.size
        if width == height:
            return image
        if width > height:
            pad = (width - height) // 2
            return ImageOps.expand(image, border=(0, pad, 0, pad), fill=0)
        pad = (height - width) // 2
        return ImageOps.expand(image, border=(pad, 0, pad, 0), fill=0)

    def process_single_image(self, source_and_target: Tuple[str, str]) -> None:
        """Processes a single fundus image with Otsu cropping and padding."""
        source_path, target_path = source_and_target
        if os.path.exists(target_path):
            return
        gray_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            return
        _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        column_indices = [c for c in range(thresholded.shape[1]) if thresholded[:, c].sum() > 0]
        row_indices = [r for r in range(thresholded.shape[0]) if thresholded[r, :].sum() > 0]
        pil_image = Image.open(source_path).convert("RGB")
        if column_indices and row_indices:
            pil_image = pil_image.crop(
                (column_indices[0], row_indices[0], column_indices[-1] + 1, row_indices[-1] + 1)
            )
        squared = self._pad_square(pil_image)
        squared.resize((self.image_size, self.image_size), Image.BILINEAR).save(target_path, "JPEG")

    def run(self) -> None:
        """Scans extracted DDR files and executes concurrent preprocessing."""
        os.makedirs(self.target_dir, exist_ok=True)
        all_files = glob(f"{self.source_dir}/**/*.*", recursive=True)
        if not all_files:
            self._auto_extract_zip_parts()
            all_files = glob(f"{self.source_dir}/**/*.*", recursive=True)
        disease_dataframe = pd.read_csv(self.annotation_path)
        target_ids: Set[str] = set(disease_dataframe["ID"].astype(str).tolist())
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        image_mapping = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in all_files
            if os.path.splitext(p)[1] in valid_extensions
        }
        task_pairs = [
            (image_mapping[image_id], os.path.join(self.target_dir, f"{image_id}.jpg"))
            for image_id in target_ids
            if image_id in image_mapping
        ]
        print(f"Matched {len(task_pairs)} / {len(target_ids)} DDR images. Starting conversion...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(self.process_single_image, task_pairs), total=len(task_pairs)))
        print(f"Completed! Total images in {self.target_dir}: {len(os.listdir(self.target_dir))}")


if __name__ == "__main__":
    DdrDatasetPreprocessor().run()
