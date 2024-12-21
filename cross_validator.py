import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import logging
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class CLIPCrossValidator:
    def __init__(
            self,
            embedding_optimizer,
            dataset,
            n_splits=3,
            shuffle=True,
            random_state=42,
            results_dir="cross_validation_results"
    ):
        self.embedding_optimizer = embedding_optimizer
        self.dataset = dataset
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_cross_validation(
            self,
            tags=['indoors', 'outdoors'],
            epochs=15,
            batch_size=8,
            learning_rate=0.001,
            patience=5,
            weight_decay=0.001,
            min_delta=0.001
    ) -> Dict:
        logger.info(f"\nStarting {self.n_splits}-fold cross-validation")
        logger.info("=" * 50)

        # Get indices for all samples
        indices = np.arange(len(self.dataset))

        all_fold_results = []
        best_accuracy = 0
        best_embeddings = None
        best_fold = 0

        # Progress bar for folds
        fold_pbar = tqdm(enumerate(self.kf.split(indices)), total=self.n_splits,
                         desc="Cross-validation folds")

        for fold, (train_idx, val_idx) in fold_pbar:
            logger.info(f"\nFold {fold + 1}/{self.n_splits}")
            logger.info("-" * 30)

            # Create train and validation datasets
            train_dataset = Subset(self.dataset, train_idx)
            val_dataset = Subset(self.dataset, val_idx)

            # Log split sizes
            train_indoor = sum(1 for i in train_idx if self.dataset[i][1] == 1)
            train_outdoor = len(train_idx) - train_indoor
            val_indoor = sum(1 for i in val_idx if self.dataset[i][1] == 1)
            val_outdoor = len(val_idx) - val_indoor

            logger.info(f"Training set: {len(train_idx)} samples ({train_indoor} indoor, {train_outdoor} outdoor)")
            logger.info(f"Validation set: {len(val_idx)} samples ({val_indoor} indoor, {val_outdoor} outdoor)")

            # Optimize embeddings for this fold
            optimized_embeddings, history = self.embedding_optimizer.optimize_embeddings(
                dataset=train_dataset,
                validation_dataset=val_dataset,
                tags=tags,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                weight_decay=weight_decay,
                min_delta=min_delta
            )

            # Evaluate on validation set
            val_accuracy = self.embedding_optimizer.evaluate(val_dataset, optimized_embeddings)

            # Store results for this fold
            fold_result = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_indoor': train_indoor,
                'train_outdoor': train_outdoor,
                'val_indoor': val_indoor,
                'val_outdoor': val_outdoor,
                'val_accuracy': val_accuracy,
                'history': history
            }

            all_fold_results.append(fold_result)

            # Update best results if necessary
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_embeddings = optimized_embeddings
                best_fold = fold + 1

            # Log fold results
            logger.info(f"Fold {fold + 1} completed:")
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

            # Update progress bar description
            fold_pbar.set_description(f"Fold {fold + 1}/{self.n_splits} - Acc: {val_accuracy:.4f}")

        # Calculate aggregate metrics
        validation_accuracies = [result['val_accuracy'] for result in all_fold_results]
        mean_accuracy = np.mean(validation_accuracies)
        std_accuracy = np.std(validation_accuracies)

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.results_dir / timestamp
        save_dir.mkdir(exist_ok=True)

        results = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'best_fold': best_fold,
            'best_accuracy': best_accuracy,
            'fold_results': all_fold_results,
            'parameters': {
                'n_splits': self.n_splits,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'patience': patience,
                'weight_decay': weight_decay,
                'min_delta': min_delta
            }
        }

        # Save best embeddings
        torch.save(best_embeddings, save_dir / f"best_embeddings_fold_{best_fold}.pt")

        # Save results
        with open(save_dir / "cross_validation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Plot cross-validation results
        self._plot_cross_validation_results(all_fold_results, save_dir)

        logger.info("\nCross-validation completed!")
        logger.info("=" * 50)
        logger.info(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"Best fold: {best_fold} (accuracy: {best_accuracy:.4f})")
        logger.info(f"Results saved to {save_dir}")

        return results

    def _plot_cross_validation_results(self, fold_results: List[Dict], save_dir: Path):
        """Generate plots for cross-validation results."""
        plt.figure(figsize=(12, 6))

        # Plot validation accuracies
        accuracies = [result['val_accuracy'] for result in fold_results]
        folds = range(1, len(accuracies) + 1)

        plt.plot(folds, accuracies, 'bo-', label='Validation Accuracy')
        plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                    label=f'Mean Accuracy: {np.mean(accuracies):.4f}')

        plt.title('Cross-validation Results')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.savefig(save_dir / 'cross_validation_plot.png')
        plt.close()