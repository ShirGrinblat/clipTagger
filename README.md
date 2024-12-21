# CLIP tagging systems

## Description

This project implements a simplified tagging system using PyTorch and CLIP. The goal is to classify images based on a dictionary of tags efficiently and optimize the tag embeddings for better separation of logits. This project assesses object-oriented programming and deep learning framework optimization skills.

## Features

- Zero-shot classification using CLIP
- Optimized text embeddings for improved classification logits
- Demonstrates results on a subset of COCO dataset


Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/ShirGrinblat/clipTagger.git
   ```
2. Navigate to the project directory:
   ```bash
   cd clipTagger
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Clone the repository CLIP:
   ```bash
   git clone https://github.com/openai/CLIP.git cd CLIP
   ```
5. Install dependencies CLIP:
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
   python optimize_embeddings.py
   ```
2. Compare the logits before and after optimization:
   ```python
   optimizer = EmbeddingOptimizer()
   optimized_tags = optimizer.optimize(tag_dict)
   print(optimized_tags)
   ```

## Requirements

- Python 3.8+
- PyTorch
- CLIP
- COCO dataset (subset for demonstration)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, contact:

- **Name**: Your Name  
- **Email**: your.email@example.com  
- **LinkedIn**: [Your Profile](https://linkedin.com/in/your-profile)  
- **GitHub**: [Your GitHub](https://github.com/username)
