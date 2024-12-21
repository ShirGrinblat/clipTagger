import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from CLIP import clip
from torchvision import transforms
from PIL import Image
import os

# Configure logging and set style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('ggplot')  # or another Matplotlib built-in style


class EnhancedModelComparison:
    def __init__(self, model_name: str, optimized_embeddings_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)

        # Load optimized embeddings
        self.optimized_embeddings = torch.load(optimized_embeddings_path, map_location=self.device, weights_only=False)

        # Initialize text prompts and embeddings for indoor/outdoor
        self.text_prompts = ["This is an outdoor image.", "This is an indoor image."]
        text_tokens = clip.tokenize(self.text_prompts).to(self.device)
        with torch.no_grad():
            self.original_embeddings = self.model.encode_text(text_tokens)
            self.original_embeddings = self.original_embeddings / self.original_embeddings.norm(dim=-1, keepdim=True)


class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, transform=None):
        self.images_dir = images_dir
        self.image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if
                            img.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class UnlabeledModelComparison(EnhancedModelComparison):
    def evaluate_on_unlabeled(self, dataset: MyCustomDataset, batch_size: int = 32) -> dict:
        """Evaluate models on unlabeled data."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        predictions = {'original': [], 'optimized': []}
        probabilities = {'original': [], 'optimized': []}
        sample_ids = []

        for batch_idx, images in enumerate(tqdm(dataloader, desc="Evaluating on unlabeled data")):
            images = images.to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                for model_type, embeddings in [
                    ('original', self.original_embeddings),
                    ('optimized', self.optimized_embeddings)
                ]:
                    similarity = image_features @ embeddings.T
                    logits = similarity * self.model.logit_scale.exp()
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    predictions[model_type].extend(preds.cpu().numpy())
                    probabilities[model_type].extend(probs.cpu().numpy())

            sample_ids.extend(range(
                batch_idx * batch_size,
                min((batch_idx + 1) * batch_size, len(dataset))
            ))

        return {
            'predictions': predictions,
            'probabilities': {k: np.array(v) for k, v in probabilities.items()},
            'sample_ids': np.array(sample_ids)
        }

    def analyze_unlabeled_results(self, results: dict, save_dir: str = 'unlabeled_analysis'):
        """Analyze and visualize results for unlabeled data."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        predictions = results['predictions']
        probabilities = results['probabilities']

        # Agreement Analysis
        agreement = (np.array(predictions['original']) == np.array(predictions['optimized'])).mean()
        logger.info(f"Agreement between models: {agreement:.2%}")

        # Confidence Analysis
        self._plot_confidence_comparison(probabilities, save_dir)
        self._plot_agreement_heatmap(predictions, probabilities, save_dir)

    def _plot_confidence_comparison(self, probabilities: dict, save_dir: Path):
        """Plot confidence comparison between models."""
        plt.figure(figsize=(10, 8))

        for model_type, probs in probabilities.items():
            sns.histplot(probs.max(axis=1), bins=30, label=model_type.capitalize(), alpha=0.5)

        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_agreement_heatmap(self, predictions: dict, probabilities: dict, save_dir: Path):
        """Plot agreement heatmap between models."""
        orig_preds = predictions['original']
        opt_preds = predictions['optimized']

        agreement_matrix = np.zeros((2, 2))  # 2x2 for binary classification
        for o, p in zip(orig_preds, opt_preds):
            agreement_matrix[o, p] += 1

        sns.heatmap(agreement_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=['Original', 'Optimized'], yticklabels=['Original', 'Optimized'])
        plt.title('Agreement Heatmap')
        plt.xlabel('Optimized Predictions')
        plt.ylabel('Original Predictions')
        plt.savefig(save_dir / 'agreement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare models on unlabeled data')

    parser.add_argument('--model_name', type=str, default='RN50',
                        help='CLIP model name')
    parser.add_argument('--optimized_embeddings', type=str, required=True,
                        help='Path to optimized embeddings file')
    parser.add_argument('--unlabeled_images_dir', type=str, required=True,
                        help='Path to unlabeled images directory')
    parser.add_argument('--output_dir', type=str, default='unlabeled_results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')

    args = parser.parse_args()

    unlabeled_comparison = UnlabeledModelComparison(
        model_name=args.model_name,
        optimized_embeddings_path=args.optimized_embeddings
    )

    unlabeled_dataset = MyCustomDataset(
        images_dir=args.unlabeled_images_dir,
        transform=unlabeled_comparison.preprocess
    )

    logger.info(f"Loaded unlabeled dataset with {len(unlabeled_dataset)} images")

    results = unlabeled_comparison.evaluate_on_unlabeled(unlabeled_dataset, batch_size=args.batch_size)

    unlabeled_comparison.analyze_unlabeled_results(results, save_dir=args.output_dir)
    logger.info(f"Analysis and visualizations saved in {args.output_dir}")


if __name__ == '__main__':
    main()
