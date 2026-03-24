# Data Directory Structure

This directory contains all data files for the RAG Dog Breed Analyzer.

## Directory Layout

```
data/
├── images/
│   ├── raw/              # Your original dog breed images
│   │   ├── golden_retriever/
│   │   ├── labrador/
│   │   └── ...           # Organize by breed name
│   └── processed/        # Resized/normalized images (auto-generated)
│
├── documents/            # Dog breed text descriptions
│   ├── breed_descriptions.json
│   ├── breed_characteristics.csv
│   └── *.txt or *.pdf files
│
├── eval_dataset/         # Evaluation Q&A pairs for RAGAS
│   ├── questions.json    # Test questions
│   └── ground_truth.json # Expected answers
│
└── vector_db/            # ChromaDB storage (auto-generated)
    └── chroma/
```

## How to Add Your Dog Breed Data

### 1. Add Dog Images
Place your dog images in `images/raw/` organized by breed:

```bash
data/images/raw/
├── golden_retriever/
│   ├── img1.jpg
│   ├── img2.jpg
├── labrador/
│   ├── img1.jpg
└── german_shepherd/
    └── img1.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

### 2. Add Breed Descriptions
Add text descriptions in `documents/`:

- **JSON format** (recommended):
```json
{
  "breeds": [
    {
      "name": "Golden Retriever",
      "size": "Large",
      "temperament": "Friendly, Intelligent, Devoted",
      "energy_level": "High",
      "grooming_needs": "Moderate to High",
      "description": "Full breed description here..."
    }
  ]
}
```

- **Text files**: One `.txt` file per breed
- **PDFs**: Breed guides or official documentation

### 3. Evaluation Dataset (Optional - will be auto-generated)
The system will create evaluation Q&A pairs, but you can add your own in:
`eval_dataset/questions.json`

## Data Sources

Suggested datasets to download:
- Stanford Dogs Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/
- Kaggle Dog Breed: https://www.kaggle.com/c/dog-breed-identification
- AKC Breed Info: Manual scraping with attribution

## Notes

- The `vector_db/` folder will be auto-generated when you first run the app
- The `processed/` folder will contain auto-resized images for the app
- Minimum recommended: 5-10 breeds with 5-10 images each
