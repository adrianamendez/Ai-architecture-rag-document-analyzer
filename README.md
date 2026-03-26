# 🐕 RAG Dog Breed Analyzer

A multimodal Retrieval-Augmented Generation (RAG) system for analyzing dog breed information with local LLMs and retrieval-quality evaluation.

## Demo Mode Note

- `streamlit_app.py` is a notebook-aligned demo page with **fixed/hardcoded outputs** for reproducibility.
- Live local inference requires Ollama running on the user machine:
  - `ollama serve`
  - `ollama pull llama3.2:latest`
  - `ollama pull llava`
- For configurable interactive execution, use `src/app.py`.

## Task Overview and Methodology

This project demonstrates:
- **Multimodal RAG**: Text + Image understanding using LLaVA/Llama3 (Ollama)
- **RAG vs direct LLM** comparisons in controlled demo scenarios
- **Two-stage retrieval** with cross-encoder reranking
- **Chunking strategy comparison**: fixed-size, semantic, combined
- **Quantitative local evaluation metrics** (no RAGAS dependency in demo flow)
- **Visualization of retrieval quality and embedding behavior**

References:
- Retrieval-Augmented Generation (RAG) course materials
- Two-stage retrieval + reranking references
- Ollama documentation: https://ollama.com/

Goals:
- Implement a document analysis app on a local GenAI framework
- Extract structured insights from dog breed text/image data
- Visualize quality and retrieval behavior with charts
- Validate quality with deterministic retrieval/answer metrics
- Keep demo flows reproducible with clear side-by-side comparisons

## Multimodal Retrieval Design Note

- **Implemented in the main app (`src/app.py` via `RAGEngine`):**
  - Text documents are indexed with sentence-transformer embeddings.
  - Breed images are indexed in a separate Chroma collection using CLIP embeddings.
  - Query-time retrieval runs in both spaces and fuses text+image scores by breed.
  - Fused context is then sent to generation (Ollama/LLaVA).
- **Notebook-aligned demo app (`streamlit_app.py`):**
  - Uses fixed outputs for reproducibility and presentation.
- **Tradeoff:** multimodal indexing improves image-driven retrieval quality, but requires more preprocessing time, storage, and runtime complexity than text-only retrieval.

## Datasets Used

- **Dog Breeds General Dataset (tabular, no images):**  
  https://www.kaggle.com/datasets/marshuu/dog-breeds
- **Expanded Ranking CSV Dataset:**  
  https://www.kaggle.com/datasets/jainaru/dog-breeds-ranking-best-to-worst
- **Dog Breeds Image Dataset:**  
  https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset
- **Second Demonstration Dataset (Stanford Dogs):**  
  https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

## Project Walkthrough

1. Setup and configuration
2. Document processing and chunking
3. Demo 1: Family Dogs query (No RAG vs limited/base RAG vs expanded RAG)
4. Demo 2: Nami multimodal retrieval (vision + retrieval)
5. Simple evaluation metrics (`Hit@k`, `Precision@k`, `Top similarity`, `Answer coverage`)
6. Embedding-space and radar visualizations

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from https://ollama.ai

   # Pull required models
   ollama pull llama3.2:latest
   ollama pull llava
   ollama pull nomic-embed-text

   # Start Ollama server
   ollama serve
   ```

### Installation

```bash
# 1. Clone/navigate to project directory
cd rag-document-analyzer

# 2. Create virtual environment (if not exists)
python3 -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify data is organized
ls data/images/raw  # Should show breed folders
ls data/documents   # Should show dog_breeds.csv
```

### Running the Application

```bash
# Option A: Main interactive app
streamlit run src/app.py

# Option B: Demo app aligned with notebook (fixed demo outputs)
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### Core Functionality
- **64 Dog Breeds** with text characteristics and images
- **Vector Database** (ChromaDB) with efficient retrieval
- **Cross-Encoder Reranking** for improved ranking quality
- **Ollama Integration** for local inference (no API key)
- **Notebook-aligned fixed 3-way comparisons** in `streamlit_app.py`
- **Full multimodal retrieval** (text + image vector fusion) in `src/app.py`

### Main App Evaluation (`src/app.py`)
- **RAGAS evaluation pipeline** with:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

### Demo Metrics (`streamlit_app.py`)
- **Hit@k**: Expected breed appears in retrieved top-k documents
- **Precision@k**: Fraction of relevant docs in top-k
- **Top similarity**: Best retrieval similarity score
- **Answer coverage**: Final answer references retrieved evidence

### Visualizations
- **Quality Radar Chart** for deterministic metrics
- **Embedding Space Map** for breed similarity structure
- **Comparison tables** for retrieval/answer behavior

## 📁 Project Structure

```
rag-document-analyzer/
├── streamlit_app.py               # Notebook-aligned demo app (fixed 3-way comparison)
│
├── src/
│   ├── app.py                     # Main Streamlit web application
│   ├── config.py                   # Configuration and settings
│   ├── document_processor.py       # Multimodal document ingestion
│   ├── rag_engine.py              # RAG with ChromaDB + reranking
│   ├── evaluator.py               # Evaluation utilities
│   ├── visualizer.py              # Plotly visualizations
│   └── eval_dataset_generator.py  # Generate Q&A pairs
│
├── data/
│   ├── images/raw/                # Dog breed images (64 breeds)
│   ├── documents/                 # Breed text data (CSV)
│   ├── breed_mapping.json         # Links images + text
│   ├── vector_db/                 # ChromaDB storage (auto-generated)
│   └── eval_dataset/              # Evaluation Q&A pairs (auto-generated)
│
├── scripts/
│   └── organize_data.py           # Data organization script
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎮 Usage Guide

### 1. Main App (`src/app.py`)

In the Streamlit sidebar:
1. Select **Model Type** (text_only, multimodal, etc.)
2. Choose **Chunking Strategy** (fixed_size, semantic, combined)
3. Enable/disable **Reranking**
4. Click **"Initialize RAG Engine"**

### 2. Query Dog Breeds

In the **Query** tab:
- Type your question or click sample questions
- View the generated answer
- Inspect retrieved contexts and metadata
- See response time and images used

**Sample Questions:**
- "What are the characteristics of Golden Retrievers?"
- "Which dog breeds are hypoallergenic?"
- "Compare Beagle and Chihuahua"
- "What breeds are good with children?"

### 3. Run Evaluation (Main App)

In the **Evaluation** tab:
1. **Generate Evaluation Dataset** (30-50 Q&A pairs)
2. **Run Evaluation** on subset (10-20 samples recommended)
3. View evaluation metrics in radar chart
4. Compare different RAG strategies

### 4. Visualize Embeddings

In the **Visualizations** tab:
- Generate embedding space maps (UMAP or t-SNE)
- See how breeds cluster by similarity
- Adjust number of breeds to visualize

### 5. Browse Dataset

In the **Dataset Info** tab:
- View all 64 breeds
- Explore breed characteristics
- See dataset statistics

### 6. Notebook-Aligned Demo (`streamlit_app.py`)

Use this app when you want the exact demo flow aligned with `rag_demo_notebook.ipynb`:
- Demo 1: Family query with **No RAG / limited RAG / expanded RAG**
- Demo 2: Nami images with **vision-only / limited RAG / expanded RAG**
- Uses fixed/hardcoded notebook outputs for reproducibility
- Shows baseline deterministic metrics as presented in the notebook demo

## 🧪 Testing Individual Components

```bash
# Test configuration
python src/config.py

# Test document processor
python src/document_processor.py

# Test RAG engine
python src/rag_engine.py

# Generate evaluation dataset
python src/eval_dataset_generator.py

# Test evaluator
python src/evaluator.py

# Test visualizations
python src/visualizer.py
```

## 📊 Dataset Information

- **Source**: Kaggle Dog Breeds Image Dataset + Custom CSV
- **Total Breeds**: 64 (with both images and text)
- **Total Images**: ~1,000 images
- **Characteristics**: Country, fur color, height, temperament, health issues

## 🔧 Configuration

Edit `src/config.py` to customize:
- Embedding models
- Chunking parameters
- Retrieval settings
- Reranking configuration
- Visualization options
- Ollama endpoints

## 📈 Example Results

The project tracks retrieval quality with deterministic metrics:

| Scenario | Hit@k | Precision@k | Top similarity | Answer coverage |
|----------|------:|------------:|---------------:|----------------:|
| Demo 1 - Family query | 0.0 | 0.0 | 0.487 | 1.0 |
| Demo 2 - Nami query | 1.0 | 0.2 | 0.720 | 1.0 |

*(Example scores - run your own evaluation for actual results)*

## 🛠️ Advanced Usage

### Custom Evaluation Dataset

Edit `src/eval_dataset_generator.py` to:
- Add custom question templates
- Modify Q&A generation logic
- Include domain-specific queries

### Add More Breeds

1. Add breed images to `data/images/raw/breed_name/`
2. Update `data/documents/dog_breeds.csv`
3. Run `python scripts/organize_data.py`
4. Reinitialize RAG engine in the app

### Change LLM Models

In `src/config.py`, add new models to `MODELS` dict:
```python
"custom_model": {
    "name": "mistral",
    "description": "Custom model",
    "supports_vision": False,
    "temperature": 0.7,
}
```

## 📝 Requirements

All task requirements addressed:
- ✅ **GenAI Framework**: Ollama (local, free, no API keys)
- ✅ **Document Analysis**: 64 dog breeds with text + images
- ✅ **Data Extraction**: Breed characteristics, temperament, health info
- ✅ **Visualization**: Quality radar charts + embedding space maps
- ✅ **Evaluation Metrics**: Deterministic retrieval metrics (Hit@k, Precision@k, Top similarity, coverage)
- ✅ **Evaluation Dataset**: 50+ Q&A pairs with ground truth

### Ninja Challenges:
- ⚠️ **Corpus Update**: Can add breeds without full rebuild (incremental ingestion)
- ❌ **Access Control**: Not implemented (could add user-based filtering)
- ✅ **Multi-modal RAG**: Images + text with LLaVA vision model
- ✅ **RAG Evaluation**: Deterministic retrieval-quality metrics + comparative demos

## 🐛 Troubleshooting

### Ollama not available
```bash
# Start Ollama
ollama serve

# Verify models are pulled
ollama list

# Pull if missing
ollama pull llama3
ollama pull llava
```

### ChromaDB errors
```bash
# Delete and rebuild vector DB
rm -rf data/vector_db/*
# Then reinitialize in the app
```

### Live demo buttons fail
- Ensure Ollama is running: `ollama serve`
- Ensure required models exist: `ollama list`
- Pull missing models:
  - `ollama pull llama3.2:latest`
  - `ollama pull llava`
- If startup imports fail, reinstall dependencies: `pip install -r requirements.txt --upgrade`

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📚 Resources

- **Ollama**: https://ollama.ai/
- **ChromaDB**: https://www.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset

## 🤝 Contributing

This is a training project. Improvements welcome:
- Add more breeds
- Improve evaluation metrics
- Add new visualizations
- Optimize retrieval strategies

## 📄 License

Educational use only. Dog breed data from Kaggle (see attribution).

## 👥 Author

Created for AI/RAG training course.

---

**Questions?** Check the inline documentation in each module!
