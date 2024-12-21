# CLIP tagging systems

## Description

This project implements a simplified tagging system using PyTorch and CLIP. The goal is to classify images based on a dictionary of tags efficiently and optimize the tag embeddings for better separation of logits. This project assesses object-oriented programming and deep learning framework optimization skills.

## Requirements

- Python 3.8+
- PyTorch
- CLIP

## Features

- Zero-shot classification using CLIP
- Optimized text embeddings for improved classification logits
- Demonstrates results on a subset of COCO dataset

Follow these steps to set up the project locally:
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/ShirGrinblat/clipTagger.git
   ```
3. Navigate to the project directory:
   ```bash
   cd clipTagger
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Clone the repository CLIP:
   ```bash
   git clone https://github.com/openai/CLIP.git cd CLIP
   ```
6. Install dependencies CLIP:
   ```bash
   pip install -r requirements.txt
   cd ..
   ```

## Usage

### Part 1: Tagging System

1. you can change the tags in  [prompts_tags.json](prompts_tags.json) and add images in images folder and run :
     ```bash
    python tagger.py --image_folder images --model_name RN50 --batch_size 16 --threshold 0.3 --config_file prompts_tags.json
   ```

### Part 2: Optimization

1. Run the optimization script to improve text embeddings:
   ```bash
   python run_optimize_embeddings.py --base_dir project_data --model_name RN50 --epochs 100 --learning_rate 0.004 --batch_size 32 --patience 3 --weight_decay 0.01 --min_delta 0.01 --tags indoors outdoors 
   ```
2. Compare the logits before and after optimization:
   ```bash
   python tagger.py --image_folder images --model_name RN50 --batch_size 16 --threshold 0.3 --config_file prompts_tags.json --model_path project_data/weights/optimized/indoor_outdoor_embeddings_v.pt

   ```

## Contact

For any questions or suggestions, contact:

- **Name**: Shir Grinblat 
- **Email**: shirgrinblat@gmail.com
- **LinkedIn**: [link](https://www.linkedin.com/in/shir-grinblat/)  
- **GitHub**: [link](https://github.com/ShirGrinblat)
