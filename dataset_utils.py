import os
import json
import shutil
import time

import torch
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
import requests
import zipfile
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

logger = logging.getLogger(__name__)

def download_coco128_dataset(
    download_dir: str,
    force_download: bool = False
) -> Tuple[Path, Path]:
    """
    Download COCO 128 dataset.

    Args:
        download_dir: Directory to download dataset
        force_download: Whether to force download even if files exist

    Returns:
        Tuple of (images_dir, labels_dir)
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)

    dataset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = download_dir / "coco128.zip"

    if force_download or not zip_path.exists():
        logger.info("Downloading COCO 128 dataset...")
        try:
            response = requests.get(dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc=zip_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        except Exception as e:
            # Clean up partial download if it fails
            if zip_path.exists():
                zip_path.unlink()
            logger.error(f"Failed to download dataset: {e}")
            raise

    # Extract dataset if needed
    extract_dir = download_dir / "coco128"
    if not extract_dir.exists() or force_download:
        logger.info("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
        except Exception as e:
            # Clean up partial extraction if it fails
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir)
            logger.error(f"Failed to extract dataset: {e}")
            raise

    # Verify and return paths
    images_dir = extract_dir / "images" / "train2017"
    labels_dir = extract_dir / "labels" / "train2017"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Dataset structure not found. Expected:\n"
            f"Images: {images_dir}\n"
            f"Labels: {labels_dir}"
        )

    return images_dir, labels_dir
class DataManager:
    """Manages data organization, splits, and weight storage for the CLIP project."""

    def __init__(self, base_dir: str = "project_data"):
        """
        Initialize data manager with the following structure:
        project_data/
        ├── data/
        │   ├── raw/                 # Original COCO128 data
        │   ├── processed/           # Processed and split datasets
        │   │   ├── train/
        │   │   ├── val/
        │   │   └── test/
        ├── weights/                 # Model weights
        │   ├── clip/               # Original CLIP weights
        │   └── optimized/          # Optimized embeddings
        └── results/                # Training results and logs
        """
        self.base_dir = Path(base_dir)

        # Create directory structure
        self.dirs = {
            'raw': self.base_dir / 'data' / 'raw',
            'processed': self.base_dir / 'data' / 'processed',
            'weights': self.base_dir / 'weights',
            'clip_weights': self.base_dir / 'weights' / 'clip',
            'optimized_weights': self.base_dir / 'weights' / 'optimized',
            'results': self.base_dir / 'results'

        }

        self._create_directories()

    def _create_directories(self):
        """Create the directory structure."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, force_download: bool = False) -> Path:
        """Download dataset to raw directory."""

        images_dir, labels_dir = download_coco128_dataset(
            self.dirs['raw'],
            force_download=force_download
        )
        return self.dirs['raw']

    def create_splits(
            self,
            train_size: float = 0.7,
            val_size: float = 0.15,
            test_size: float = 0.15,
            random_seed: int = 42
    ) -> Dict[str, Tuple[List[Path], List[Path]]]:
        """
        Create train/val/test splits and organize the data.

        Returns:
            Dictionary containing paths for images and labels for each split
        """
        if not abs(train_size + val_size + test_size - 1.0) < 1e-5:
            raise ValueError("Split proportions must sum to 1")

        # Get all image and label files
        images_dir = Path("project_data/data/raw/coco128/images/train2017")
        labels_dir = Path("project_data/data/raw/coco128/labels/train2017")
        # Create sets for efficient lookup
        image_files = list(images_dir.glob("*.[jp][pn][g]"))
        all_label_files = list(labels_dir.glob("*.txt"))
        image_stems = {img.stem for img in image_files}
        label_stems = {label.stem for label in all_label_files}

        # Find common stems (files that have both image and label)
        common_stems = image_stems.intersection(label_stems)

        # Filter files to keep only those with matching pairs
        valid_images = [img for img in image_files if img.stem in common_stems]
        valid_labels = [labels_dir / f"{img.stem}.txt" for img in valid_images]

        print(f"Total images: {len(image_files)}")
        print(f"Total labels: {len(all_label_files)}")
        print(f"Matching pairs: {len(valid_images)}")

        # First split: train and temporary
        train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
            valid_images, valid_labels,
            train_size=train_size,
            random_state=random_seed
        )

        # Second split: validation and test from temporary
        relative_val_size = val_size / (val_size + test_size)
        val_imgs, test_imgs, val_labels, test_labels = train_test_split(
            temp_imgs, temp_labels,
            train_size=relative_val_size,
            random_state=random_seed
        )

        # Organize splits into processed directory
        splits = {
            'train': (train_imgs, train_labels),
            'val': (val_imgs, val_labels),
            'test': (test_imgs, test_labels)
        }

        # Copy files to their respective directories
        for split_name, (split_imgs, split_labels) in splits.items():
            split_dir = self.dirs['processed'] / split_name
            split_dir.mkdir(exist_ok=True)

            # Create images and labels subdirectories
            imgs_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            imgs_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            # Copy files
            for img, label in zip(split_imgs, split_labels):
                if img.exists() and label.exists():  # Additional check before copying
                    shutil.copy2(img, imgs_dir / img.name)
                    shutil.copy2(label, labels_dir / label.name)
        # Save split information
        split_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'random_seed': random_seed,
            'num_samples': {
                'train': len(train_imgs),
                'val': len(val_imgs),
                'test': len(test_imgs)
            }
        }

        with open(self.dirs['processed'] / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=4)

        return splits

    def save_weights(self, weights: Dict, name: str, optimize_type: str = 'clip') -> Path:
        """
        Save model weights with versioning.

        Args:
            weights: Dictionary containing weight tensors
            name: Name for the weights file
            optimize_type: 'clip' for original CLIP weights or 'optimized' for optimized embeddings
        """
        weights_dir = self.dirs['clip_weights'] if optimize_type == 'clip' else self.dirs['optimized_weights']

        # Create version number
        existing_versions = [int(p.stem.split('_v')[-1]) for p in weights_dir.glob(f"{name}_v*")]
        version = max(existing_versions, default=0) + 1

        # Save weights
        weights_path = weights_dir / f"{name}_v{version}.pt"
        torch.save(weights, weights_path)

        # Save metadata
        metadata = {
            'version': version,
            'name': name,
            'type': optimize_type,
            'timestamp': time.time(),
        }

        with open(weights_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved weights to {weights_path}")
        return weights_path

    def load_latest_weights(self, name: str, optimize_type: str = 'clip') -> Tuple[Dict, Dict]:
        """
        Load the latest version of weights.

        Returns:
            Tuple of (weights, metadata)
        """
        weights_dir = self.dirs['clip_weights'] if optimize_type == 'clip' else self.dirs['optimized_weights']

        # Find latest version
        weight_files = list(weights_dir.glob(f"{name}_v*.pt"))
        if not weight_files:
            raise FileNotFoundError(f"No weights found for {name}")

        latest_weights = max(weight_files, key=lambda p: int(p.stem.split('_v')[-1]))

        # Load weights and metadata
        weights = torch.load(latest_weights)
        with open(latest_weights.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)

        return weights, metadata