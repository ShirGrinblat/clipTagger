import logging
from pathlib import Path
from typing import Tuple
import torch
import argparse
from embedding_optimizer import EmbeddingOptimizer
from coco_dataset import COCO128Dataset
from dataset_utils import DataManager
from cross_validator import CLIPCrossValidator
from CLIP import clip


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run CLIP-based cross-validation experiment.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory for the project.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory to store results.")
    parser.add_argument('--force_download', action='store_true', help="Force dataset re-download.")
    parser.add_argument('--model_name', type=str, default='RN50', help="CLIP model name (default: RN50).")
    parser.add_argument('--n_splits', type=int, default=3, help="Number of cross-validation splits (default: 3).")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs (default: 20).")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (default: 4).")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument('--patience', type=int, default=7, help="Early stopping patience (default: 7).")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay for optimizer (default: 0.001).")
    parser.add_argument('--min_delta', type=float, default=0.001, help="Minimum delta for early stopping (default: 0.001).")
    parser.add_argument('--no_cache', action='store_true', help="Disable image caching in dataset.")
    return parser.parse_args()


def setup_experiment(args) -> Tuple[COCO128Dataset, EmbeddingOptimizer]:
    """
    Set up the dataset and embedding optimizer for the experiment.
    """
    # Initialize data manager and handle dataset download
    data_manager = DataManager(base_dir=args.base_dir)
    if args.force_download:
        data_manager.download_dataset()

    # Get paths for the dataset
    images_dir = Path(f"{args.base_dir}/data/raw/coco128/images/train2017")
    labels_dir = Path(f"{args.base_dir}/data/raw/coco128/labels/train2017")

    # Initialize CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model_name, device=device)

    # Create the dataset
    dataset = COCO128Dataset(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        transform=preprocess,
        cache_images=not args.no_cache
    )

    # Log dataset statistics
    total_samples = len(dataset)
    indoor_samples = sum(1 for _, label in dataset if label == 1)
    outdoor_samples = total_samples - indoor_samples
    logger.info(f"Dataset statistics:")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Indoor samples: {indoor_samples}")
    logger.info(f"Outdoor samples: {outdoor_samples}")

    embedding_optimizer = EmbeddingOptimizer(
        model_name=args.model_name,
        device=device,
        results_dir=args.results_dir
    )

    return dataset, embedding_optimizer


def main():
    """
    Main function to run the cross-validation experiment.
    """
    args = parse_arguments()
    logger.info("Setting up experiment...")
    dataset, embedding_optimizer = setup_experiment(args)

    # Initialize cross-validator
    cross_validator = CLIPCrossValidator(
        embedding_optimizer=embedding_optimizer,
        dataset=dataset,
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_seed,
        results_dir=args.results_dir
    )

    # Run cross-validation
    logger.info("Starting cross-validation...")
    results = cross_validator.run_cross_validation(
        tags=['indoors', 'outdoors'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        weight_decay=args.weight_decay,
        min_delta=args.min_delta
    )

    # Log results
    logger.info("\nCross-validation Results:")
    logger.info(f"Mean accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    logger.info(f"Best fold: {results['best_fold']} with accuracy: {results['best_accuracy']:.3f}")

    # Additional fold-specific results
    logger.info("\nPer-fold Results:")
    for fold_result in results['fold_results']:
        logger.info(
            f"Fold {fold_result['fold']}: "
            f"Validation Accuracy = {fold_result['val_accuracy']:.3f}, "
            f"Train Size = {fold_result['train_size']}, "
            f"Validation Size = {fold_result['val_size']}"
        )

    # Print class distribution in best fold
    best_fold_result = results['fold_results'][results['best_fold'] - 1]
    logger.info(f"\nBest fold ({results['best_fold']}) details:")
    logger.info(f"Training samples: {best_fold_result['train_size']}")
    logger.info(f"Validation samples: {best_fold_result['val_size']}")


if __name__ == "__main__":
    main()