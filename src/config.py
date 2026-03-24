"""
Configuration module for RAG Dog Breed Analyzer.
Contains all settings for Ollama models, paths, and parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "raw"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
EVAL_DIR = DATA_DIR / "eval_dataset"

# Ensure directories exist
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Dataset files
BREED_MAPPING_FILE = DATA_DIR / "breed_mapping.json"
DOG_BREEDS_CSV = DOCUMENTS_DIR / "dog_breeds.csv"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Model configurations for different RAG strategies
MODELS = {
    "text_only": {
        "name": "llama3",
        "description": "Text-only RAG using Llama 3",
        "supports_vision": False,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "multimodal": {
        "name": "llava",
        "description": "Multimodal RAG using LLaVA",
        "supports_vision": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "multimodal_bakllava": {
        "name": "bakllava",
        "description": "Multimodal RAG using BakLLaVA",
        "supports_vision": True,
        "temperature": 0.7,
        "top_p": 0.9,
    },
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "multimodal_model": "clip-ViT-B-32",  # For image+text embeddings
    "dimension": 384,  # MiniLM dimension
    "batch_size": 32,
}

# Chunking strategies
CHUNKING_STRATEGIES = {
    "fixed_size": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "description": "Fixed size chunks with overlap"
    },
    "semantic": {
        "by_attribute": True,
        "description": "Chunk by breed attributes (one chunk per characteristic)"
    },
    "combined": {
        "description": "Combine text + image metadata into single chunk"
    }
}

# Retrieval configuration
RETRIEVAL_CONFIG = {
    "top_k": 10,  # Number of initial documents to retrieve
    "rerank_top_n": 3,  # Number of documents after reranking
    "similarity_threshold": 0.5,  # Minimum similarity score
    "use_reranking": True,  # Enable cross-encoder reranking
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# ChromaDB configuration
CHROMA_CONFIG = {
    "collection_name": "dog_breeds",
    "distance_metric": "cosine",  # cosine, l2, or ip
    "persist_directory": str(VECTOR_DB_DIR),
}

# RAGAS evaluation metrics
RAGAS_METRICS = [
    "faithfulness",           # Is answer grounded in context?
    "answer_relevancy",       # Does answer address the question?
    "context_precision",      # Were right docs retrieved?
    "context_recall",         # Were all relevant docs found?
]

# Visualization configuration
VIZ_CONFIG = {
    "radar_chart": {
        "metrics": ["size", "energy_level", "grooming_needs", "trainability", "health_score"],
        "color_scheme": "plotly",
    },
    "embedding_map": {
        "method": "umap",  # umap or tsne
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
    }
}

# Prompt templates
PROMPTS = {
    "text_only_rag": """You are a dog breed expert. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""",

    "multimodal_rag": """You are a dog breed expert with access to both text information and images.
Use the provided context (text and image descriptions) to answer the question accurately.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Images described: {image_descriptions}

Question: {question}

Answer:""",

    "image_description": """Describe this dog breed image in detail. Include:
- Physical characteristics (size, color, coat type)
- Distinctive features
- Overall appearance

Image: {image_path}

Description:""",

    "evaluation_qa_generation": """Based on this dog breed information, generate a question and answer pair.
Make the question specific and the answer factual based on the provided information.

Breed Information:
{breed_info}

Generate:
Question: [Your question here]
Answer: [Detailed answer based on the information]
""",
}

# Streamlit UI configuration
UI_CONFIG = {
    "title": "🐕 RAG Dog Breed Analyzer",
    "subtitle": "Multimodal Document Analysis with Evaluation",
    "theme": "light",
    "sidebar_width": 300,
}

# Evaluation dataset size
EVAL_DATASET_SIZE = 50  # Number of Q&A pairs to generate

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}


def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get configuration for a specific model type."""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODELS.keys())}")
    return MODELS[model_type]


def get_chunking_strategy(strategy_name: str) -> Dict[str, Any]:
    """Get configuration for a specific chunking strategy."""
    if strategy_name not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")
    return CHUNKING_STRATEGIES[strategy_name]


# Check if Ollama is available
def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    # Test configuration
    print("=" * 60)
    print("RAG Dog Breed Analyzer - Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Vector DB: {VECTOR_DB_DIR}")
    print(f"\nAvailable Models: {list(MODELS.keys())}")
    print(f"Chunking Strategies: {list(CHUNKING_STRATEGIES.keys())}")
    print(f"RAGAS Metrics: {RAGAS_METRICS}")
    print(f"\nOllama Available: {check_ollama_available()}")
    print("=" * 60)
