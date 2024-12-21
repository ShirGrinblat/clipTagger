import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from CLIP import clip
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image), image_path


class Tagger:
    def __init__(self, tag_dict, prompt_templates=None, model_name="RN50", model_path=None, batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.model_name = model_name

        # Load the CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.clip_preprocess = clip.load(self.model_name, device=self.device)

        # Load a pretrained model if specified
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Using default CLIP weights instead")
        else:
            print("Using default CLIP weights")

        # Define preprocessing
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711))
        ])

        self.tag_dict = tag_dict
        self.prompt_templates = prompt_templates or {}
        self.default_template = "This is a {answer} image."
        self._generate_text_embeddings()

    def _generate_text_embeddings(self):
        """Generate and cache text embeddings for all prompts."""
        self.text_prompts = []
        self.category_ranges = {}
        start_idx = 0

        print("Generating text embeddings for categories:")
        for category, tags in self.tag_dict.items():
            template = self.prompt_templates.get(category, self.default_template)
            print(f"Category: {category}")
            print(f"Template: {template}")

            for tag in tags:
                prompt = template.format(category=category, answer=tag)
                print(f"  - {prompt}")
                self.text_prompts.append(prompt)

            end_idx = start_idx + len(tags)
            self.category_ranges[category] = (start_idx, end_idx)
            start_idx = end_idx

        text_tokens = clip.tokenize(self.text_prompts).to(self.device)
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(text_tokens).float()
            # Normalize embeddings
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)

    def _predict_batch(self, image_batch, threshold=0.2):
        """Make predictions for a batch of images with detailed probabilities."""
        with torch.no_grad():
            # Encode and normalize image features
            image_features = self.model.encode_image(image_batch.to(self.device)).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores
            similarity = image_features @ self.text_embeddings.T

            # Scale logits
            logits = similarity * self.model.logit_scale.exp()

        predictions = []
        for img_logits in logits:
            img_predictions = {}
            for category, (start_idx, end_idx) in self.category_ranges.items():
                category_logits = img_logits[start_idx:end_idx]

                # Calculate probabilities using softmax
                category_probs = torch.nn.functional.softmax(category_logits, dim=-1)
                probs = category_probs.cpu().numpy()

                # Create dictionary of all probabilities
                tag_probs = {
                    tag: float(prob)
                    for tag, prob in zip(self.tag_dict[category], probs)
                }

                # Get the highest probability tag
                max_tag, max_prob = max(tag_probs.items(), key=lambda x: x[1])

                # Only include prediction if it meets threshold
                if max_prob > threshold:
                    img_predictions[category] = {
                        "tag": max_tag,
                        "confidence": max_prob,
                        "all_probabilities": tag_probs
                    }
                else:
                    img_predictions[category] = None

            predictions.append(img_predictions)

        return predictions

    def tag_images(self, folder_path, threshold=0.2):
        """Tag all images in the specified folder."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[-1].lower() in valid_extensions
        ]

        if not image_paths:
            print("No valid images found in the folder.")
            return []

        print(f"Found {len(image_paths)} images to process")
        dataset = ImageDataset(image_paths, self.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        all_predictions = []
        for image_batch, paths in tqdm(dataloader, desc="Processing images"):
            batch_predictions = self._predict_batch(image_batch, threshold=threshold)
            for path, preds in zip(paths, batch_predictions):
                preds["image_path"] = path
                all_predictions.append(preds)

        return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Enhanced Image Tagging with CLIP")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to the folder containing images")
    parser.add_argument("--model_name", type=str, default="RN50",
                        help="Name of the CLIP model to use (e.g., 'RN50')")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pretrained model file (optional)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing images")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Confidence threshold for predictions (default: 0.2)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to the JSON config file with prompts and tags")
    parser.add_argument("--output_file", type=str, default="predictions.json",
                        help="Path to save the predictions (default: predictions.json)")

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        prompt_templates = config.get("prompt_templates", {})
        tag_dict = config.get("tag_dict", {})

    # Initialize tagger
    tagger = Tagger(
        tag_dict,
        prompt_templates=prompt_templates,
        model_name=args.model_name,
        model_path=args.model_path,
        batch_size=args.batch_size
    )

    # Process images
    predictions = tagger.tag_images(args.image_folder, threshold=args.threshold)

    # Save predictions
    print(f"\nSaving predictions to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    # Print summary
    print("\nPrediction Summary:")
    for pred in predictions:
        print(f"\nImage: {pred['image_path']}")
        for category, result in pred.items():
            if category != 'image_path' and result is not None:
                print(f"  {category}:")
                print(f"    Selected: {result['tag']} (confidence: {result['confidence']:.3f})")
                print("    All probabilities:")
                for tag, prob in result['all_probabilities'].items():
                    print(f"      - {tag}: {prob:.3f}")


if __name__ == "__main__":
    main()