# 🐕 RAG Dog Breed Analyzer

A multimodal Retrieval-Augmented Generation (RAG) system for analyzing dog breed information with comprehensive evaluation using RAGAS metrics.

## 📋 Overview

This application demonstrates advanced RAG techniques including:
- **Multimodal RAG**: Text + Image understanding using LLaVA/Llama3 (Ollama)
- **Multiple Chunking Strategies**: Fixed-size, semantic, and combined chunking
- **Reranking**: Cross-encoder reranking for improved retrieval precision
- **RAGAS Evaluation**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Interactive Visualizations**: Radar charts and embedding space maps

## 🎯 Features

### ✅ Core Functionality
- **64 Dog Breeds** with text characteristics and images
- **Vector Database** (ChromaDB) with efficient retrieval
- **Cross-Encoder Reranking** for improved accuracy
- **Ollama Integration** for local LLM inference (no API keys required!)

### 📊 Evaluation Metrics (RAGAS)
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Were the right docs retrieved?
- **Context Recall**: Were all relevant docs found?

### 🎨 Visualizations
- **RAGAS Radar Chart**: Compare different RAG strategies
- **Embedding Space Map**: Visualize breed similarities (UMAP/t-SNE)
- **Metrics Comparison**: Bar charts and tables

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from https://ollama.ai

   # Pull required models
   ollama pull llama3
   ollama pull llava

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
# Start Streamlit app
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
rag-document-analyzer/
├── src/
│   ├── app.py                      # Streamlit web application
│   ├── config.py                   # Configuration and settings
│   ├── document_processor.py       # Multimodal document ingestion
│   ├── rag_engine.py              # RAG with ChromaDB + reranking
│   ├── evaluator.py               # RAGAS evaluation metrics
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

### 1. Initialize RAG Engine

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

### 3. Run Evaluation

In the **Evaluation** tab:
1. **Generate Evaluation Dataset** (30-50 Q&A pairs)
2. **Run Evaluation** on subset (10-20 samples recommended)
3. View RAGAS metrics in radar chart
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

## 📈 Evaluation Results

The system compares different RAG strategies:

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|-------------|------------------|-------------------|----------------|
| Text-only | 0.85 | 0.78 | 0.72 | 0.68 |
| Multimodal | 0.88 | 0.82 | 0.79 | 0.75 |
| Multimodal + Rerank | 0.92 | 0.89 | 0.86 | 0.81 |

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
- ✅ **Visualization**: RAGAS radar charts + embedding space maps
- ✅ **Evaluation Metrics**: Faithfulness, relevancy, precision, recall
- ✅ **Evaluation Dataset**: 50+ Q&A pairs with ground truth

### Ninja Challenges:
- ⚠️ **Corpus Update**: Can add breeds without full rebuild (incremental ingestion)
- ❌ **Access Control**: Not implemented (could add user-based filtering)
- ✅ **Multi-modal RAG**: Images + text with LLaVA vision model
- ✅ **RAG Evaluation**: Full RAGAS metrics suite

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

### RAGAS evaluation fails
- Requires OpenAI API key for some metrics (can set dummy key)
- Or use simplified metrics in `src/evaluator.py`

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📚 Resources

- **RAGAS Documentation**: https://docs.ragas.io/
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