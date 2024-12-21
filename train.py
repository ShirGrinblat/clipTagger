import argparse
import logging
from pathlib import Path
from embedding_optimizer import EmbeddingOptimizer
from coco_dataset import COCO128Dataset
from dataset_utils import DataManager
from CLIP import clip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the CLIP embedding optimization pipeline."""
    parser = argparse.ArgumentParser(description="Run CLIP embedding optimization pipeline.")

    # Dataset and paths
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for dataset and outputs.")
    parser.add_argument("--force_download", action="store_true", help="Force dataset download if already exists.")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of data for training set.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of data for validation set.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion of data for testing set.")

    # Model and optimizer parameters
    parser.add_argument("--model_name", type=str, default="RN50", help="Name of the CLIP model to use (e.g., RN50).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for optimization.")
    parser.add_argument("--learning_rate", type=float, default=0.004, help="Learning rate for optimization.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for optimization.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="L2 regularization weight decay.")
    parser.add_argument("--min_delta", type=float, default=0.01, help="Minimum delta for early stopping.")
    parser.add_argument("--tags", nargs="+", default=["indoors", "outdoors"], help="Tags to optimize embeddings for.")

    args = parser.parse_args()

    try:
        # Initialize data manager
        data_manager = DataManager(base_dir=args.base_dir)

        # Download dataset if needed and create splits
        logger.info("Setting up dataset...")
        data_manager.download_dataset(force_download=args.force_download)
        splits = data_manager.create_splits(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size
        )

        # Initialize CLIP model and get preprocessing function
        logger.info("Initializing CLIP model...")
        _, preprocess = clip.load(args.model_name)

        # Create datasets for each split
        datasets = {}
        for split_name, (split_imgs, split_labels) in splits.items():
            is_train = split_name == 'train'
            datasets[split_name] = COCO128Dataset(
                images_dir=data_manager.dirs['processed'] / split_name / 'images',
                labels_dir=data_manager.dirs['processed'] / split_name / 'labels',
                transform=preprocess,
                cache_images=True,
                train=is_train  # Only apply full augmentations to training set
            )
            logger.info(f"{split_name} set size: {len(datasets[split_name])}")

        # Initialize optimizer
        optimizer = EmbeddingOptimizer(
            model_name=args.model_name,
            results_dir=data_manager.dirs['results']
        )

        # Run optimization
        logger.info("Starting embedding optimization...")
        optimized_embeddings, history = optimizer.optimize_embeddings(
            dataset=datasets['train'],
            validation_dataset=datasets['val'],
            tags=args.tags,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            weight_decay=args.weight_decay,
            min_delta=args.min_delta
        )

        # Save optimized embeddings
        data_manager.save_weights(
            {'embeddings': optimized_embeddings},
            name='indoor_outdoor_embeddings',
            optimize_type='optimized'
        )

        # Test final performance
        test_accuracy = optimizer.evaluate(datasets['test'], optimized_embeddings)
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()