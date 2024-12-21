# CLIP Tagging System

## Overview

This project implements a tagging system using PyTorch and CLIP for efficient image classification based on a dictionary of tags. It includes features to optimize tag embeddings for better separation of logits, leveraging object-oriented programming and deep learning optimizations.

## Features

- Zero-shot classification using CLIP
- Tag embedding optimization for enhanced classification accuracy
- Demonstrates results on a subset of the COCO dataset

## Requirements

- Python 3.8+
- PyTorch
- OpenAI CLIP

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```
2. **Clone the repository**:
   ```bash
   git clone https://github.com/ShirGrinblat/clipTagger.git
   cd clipTagger
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Clone and install CLIP**:
   ```bash
   git clone https://github.com/openai/CLIP.git
   cd CLIP
   pip install -r requirements.txt
   cd ..
   ```

## Usage

### Part 1: Tagging System

1. Customize tags in `prompts_tags.json`, add images to the `images` folder, and run:
   ```bash
   python tagger.py --image_folder images --model_name RN50 --batch_size 16 --threshold 0.3 --config_file prompts_tags.json
   ```

### Part 2: Optimization

1. **Run optimization (no cross-validation)**:
   ```bash
   python run_optimize_embeddings.py --base_dir project_data --model_name RN50 --epochs 100 --learning_rate 0.004 --batch_size 32 --patience 3 --weight_decay 0.01 --min_delta 0.01 --tags indoors outdoors
   ```
2. **Run optimization with cross-validation**:
   ```bash
   python run_optimize_embeddings_cv.py --base_dir ./project_data --results_dir ./project_data/results --force_download --model_name RN50 --n_splits 3 --random_seed 42 --epochs 25 --batch_size 8 --learning_rate 0.001 --patience 5 --weight_decay 0.0005 --min_delta 0.0005 --no_cache
   ```
3. **Use optimized weights**:
   ```bash
   python tagger.py --image_folder images --model_name RN50 --batch_size 16 --threshold 0.3 --config_file prompts_tags.json --model_path project_data/weights/optimized/indoor_outdoor_embeddings_v.pt
   ```
4. **compare models**:
   ```bash
   python compare_models.py --model_name RN50 --optimized_embeddings project_data/results/20241221_201352/best_embeddings_fold_2.pt --unlabeled_images_dir images --output_dir output_results --batch_size 32
   ```

## Contact

For inquiries or feedback:

- **Name**: Shir Grinblat  
- **Email**: [shirgrinblat@gmail.com](mailto:shirgrinblat@gmail.com)  
- **LinkedIn**: [Profile](https://www.linkedin.com/in/shir-grinblat/)  
- **GitHub**: [Repository](https://github.com/ShirGrinblat)  
