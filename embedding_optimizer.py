import os
import torch
import logging
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from CLIP import clip
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingOptimizer:
    def __init__(
            self,
            model_name: str = "RN50",
            device: Optional[str] = None,
            results_dir: str = "optimization_results"
    ):
        """
        Initialize CLIP model and optimization setup.

        Args:
            model_name: CLIP model variant to use
            device: Computation device (cuda/cpu)
            results_dir: Directory to save optimization results
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        logger.info(f"Initializing CLIP model {model_name} on {self.device}")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            # Freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        self.history = {
            'loss': [],
            'accuracy': [],
            'embedding_distances': []
        }

    def optimize_embeddings(
            self,
            dataset: Dataset,
            validation_dataset: Dataset,
            tags: List[str] = ['indoors', 'outdoors'],
            epochs: int = 15,
            learning_rate: float = 0.001,  # Reduced from 0.01 to 0.001
            batch_size: int = 8,
            patience: int = 5,
            weight_decay : float = 0.001,
            min_delta: float = 0.001  # Minimum change to qualify as an improvement
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize text embeddings with early stopping.
        """
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size)

        # Tokenize original tags
        tag_tokens = clip.tokenize(tags).to(self.device)
        original_embeddings = self.model.encode_text(tag_tokens).detach()

        # Create optimizable embeddings
        optimizable_embeddings = original_embeddings.clone().to(self.device)
        optimizable_embeddings.requires_grad = True

        # Create optimizer with learning rate scheduler
        embedding_optimizer = torch.optim.Adam([optimizable_embeddings], lr=learning_rate,weight_decay=weight_decay )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            embedding_optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_embeddings = None
        patience_counter = 0
        best_epoch = 0

        for epoch in tqdm(range(epochs), desc="Optimizing Embeddings"):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizable_embeddings, embedding_optimizer
            )

            # Validation phase
            val_loss, val_acc = self._validate(val_loader, optimizable_embeddings)

            # Update learning rate
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_embeddings = optimizable_embeddings.clone()
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            # Calculate embedding distances
            embedding_dist = torch.norm(
                optimizable_embeddings[0] - optimizable_embeddings[1]
            ).item()

            # Update history
            self.history['loss'].append((train_loss, val_loss))
            self.history['accuracy'].append((train_acc, val_acc))
            self.history['embedding_distances'].append(embedding_dist)

            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                    f"LR={embedding_optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}")
                break

        # Save results
        self._save_results(tags, best_embeddings)
        self._plot_training_history()

        return best_embeddings, self.history

    def _train_epoch(
            self,
            dataloader: DataLoader,
            embeddings: torch.Tensor,
            optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Run one epoch of training."""
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Encode images
            with torch.no_grad():
                image_features = self.model.encode_image(images)

            # Compute logits and loss
            logits = image_features @ embeddings.T
            loss = self._compute_embedding_loss(logits, labels)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(dataloader), correct / total

    def _validate(
            self,
            dataloader: DataLoader,
            embeddings: torch.Tensor
    ) -> Tuple[float, float]:
        """Run validation."""
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                image_features = self.model.encode_image(images)
                logits = image_features @ embeddings.T
                loss = self._compute_embedding_loss(logits, labels)

                total_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(dataloader), correct / total

    def _compute_embedding_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            margin: float = 0.1
    ) -> torch.Tensor:
        """Compute custom loss to maximize embedding separation."""
        ce_loss = torch.nn.CrossEntropyLoss()(logits, labels)
        softmax_logits = torch.softmax(logits, dim=1)
        confidence_loss = -torch.mean(torch.max(softmax_logits, dim=1)[0])
        l2_reg = torch.mean(torch.norm(logits, dim=1))
        return ce_loss + margin * confidence_loss + 0.01 * l2_reg

    def _save_results(self, tags: List[str], embeddings: torch.Tensor) -> None:
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.results_dir / timestamp
        save_dir.mkdir(exist_ok=True)

        torch.save(embeddings, save_dir / "optimized_embeddings.pt")

        config = {
            "tags": tags,
            "device": self.device,
            "history": self.history
        }

        with open(save_dir / "results.json", "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Results saved to {save_dir}")

    def evaluate(
            self,
            dataset: Dataset,
            embeddings: torch.Tensor
    ) -> float:
        """
        Evaluate model performance on a dataset using optimized embeddings.

        Args:
            dataset: Dataset to evaluate on
            embeddings: Optimized text embeddings

        Returns:
            float: Accuracy on the dataset
        """
        dataloader = DataLoader(dataset, batch_size=32)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Encode images and compute logits
                image_features = self.model.encode_image(images)
                logits = image_features @ embeddings.T

                # Calculate accuracy
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        logger.info(f"Evaluation Accuracy: {accuracy:.4f}")
        return accuracy
    def _plot_training_history(self) -> None:
        """Plot and save training history."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        train_loss, val_loss = zip(*self.history['loss'])
        ax1.plot(train_loss, label='Train Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        train_acc, val_acc = zip(*self.history['accuracy'])
        ax2.plot(train_acc, label='Train Accuracy')
        ax2.plot(val_acc, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        ax3.plot(self.history['embedding_distances'])
        ax3.set_title('Embedding Distances')

        plt.tight_layout()
        plt.savefig(self.results_dir / "training_history.png")
        plt.close()