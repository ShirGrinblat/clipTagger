import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import logging
from typing import Tuple, List, Optional, Callable  # Fixed: callable -> Callable
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
from PIL import ImageFilter
logger = logging.getLogger(__name__)


class RandomGaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class ColorJitterWithGray:
    def __init__(self, p_gray=0.2):
        self.p_gray = p_gray
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.to_gray = T.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p_gray:
            return self.to_gray(img)
        return self.color_jitter(img)


def get_clip_transform(preprocess, train=True):
    if not train:
        return preprocess

    normalize = None
    for t in preprocess.transforms:
        if isinstance(t, T.Normalize):
            normalize = t
            break

    train_transforms = T.Compose([
        T.RandomResizedCrop(
            224,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        T.ToTensor(),
        normalize
    ])

    return train_transforms
class COCO128Dataset(Dataset):
    """Custom Dataset for COCO128 with location classification using txt labels."""

    def __init__(
            self,
            images_dir: str,
            labels_dir: str,
            transform: Optional[Callable] = None,  # Fixed: callable -> Callable
            cache_images: bool = True,
            train: bool = True
    ):
        """
        Initialize COCO128 dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label txt files
            transform: Image preprocessing transform
            cache_images: Whether to cache images in memory
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.base_transform = transform
        self.transform = get_clip_transform(transform, train) if transform else None
        self.cache_images = cache_images

        # Validate paths
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Load COCO classes
        self.indoor_classes = self._load_indoor_classes()

        # Load dataset
        self.image_files, self.labels = self._load_dataset()

        # Cache for images
        self.image_cache = {} if cache_images else None

        logger.info(
            f"Loaded dataset with {len(self.image_files)} images "
            f"({sum(self.labels)} indoor, {len(self.labels) - sum(self.labels)} outdoor)"
        )

    @staticmethod
    def _load_indoor_classes() -> set:
        """Load indoor class IDs."""
        return {
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
            73, 74, 75, 76, 77, 78, 79
        }

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load dataset by matching image files with their label files."""
        image_files = []
        labels = []

        for img_file in self.images_dir.glob("*.[jp][pn][g]"):
            label_file = self.labels_dir / f"{img_file.stem}.txt"

            if label_file.exists():
                is_indoor = self._process_label_file(label_file)
                image_files.append(img_file)
                labels.append(1 if is_indoor else 0)

        return image_files, labels

    def _process_label_file(self, label_path: Path) -> bool:
        """Process a label file to determine if the image is indoor."""
        try:
            labels = np.loadtxt(label_path)
            if labels.size == 0:
                return False

            # Reshape if only one label
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)

            # Check if any class is in indoor_classes
            return any(int(label[0]) in self.indoor_classes for label in labels)

        except Exception as e:
            logger.warning(f"Error processing label file {label_path}: {e}")
            return False

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_files[idx]

        # Try to get image from cache
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[image_path] = image
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                # Return a black image as fallback
                image = Image.new('RGB', (224, 224), 'black')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]