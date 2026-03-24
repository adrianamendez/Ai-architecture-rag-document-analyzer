"""
RAG Engine with vector search, reranking, and Ollama LLM integration.
Supports both text-only and multimodal RAG strategies.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
import requests
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_CONFIG,
    RETRIEVAL_CONFIG,
    OLLAMA_BASE_URL,
    MODELS,
    PROMPTS,
)
from document_processor import Document, DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine with retrieval, reranking, and generation."""

    def __init__(
        self,
        model_type: str = "text_only",
        use_reranking: bool = True,
        chunking_strategy: str = "combined",
    ):
        """
        Initialize RAG engine.

        Args:
            model_type: Type of model to use ('text_only', 'multimodal', etc.)
            use_reranking: Whether to use cross-encoder reranking
            chunking_strategy: Document chunking strategy
        """
        self.model_type = model_type
        self.model_config = MODELS[model_type]
        self.use_reranking = use_reranking
        self.chunking_strategy = chunking_strategy

        # Initialize document processor
        self.doc_processor = DocumentProcessor(chunking_strategy=chunking_strategy)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {CHROMA_CONFIG['persist_directory']}")
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_CONFIG['persist_directory']
        )

        # Collection name based on strategy
        collection_name = f"{CHROMA_CONFIG['collection_name']}_{chunking_strategy}"

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": CHROMA_CONFIG['distance_metric']}
            )
            logger.info(f"Created new collection: {collection_name}")

        # Initialize reranker if enabled
        self.reranker = None
        if use_reranking:
            logger.info(f"Loading reranker: {RETRIEVAL_CONFIG['reranker_model']}")
            self.reranker = CrossEncoder(RETRIEVAL_CONFIG['reranker_model'])

    def ingest_documents(self, force_rebuild: bool = False) -> int:
        """
        Ingest all dog breed documents into vector database.

        Args:
            force_rebuild: If True, delete existing collection and rebuild

        Returns:
            Number of documents ingested
        """
        # Check if collection already has documents
        if self.collection.count() > 0 and not force_rebuild:
            logger.info(f"Collection already has {self.collection.count()} documents")
            return self.collection.count()

        if force_rebuild:
            logger.info("Force rebuild: Deleting existing collection")
            collection_name = self.collection.name
            self.chroma_client.delete_collection(name=collection_name)
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": CHROMA_CONFIG['distance_metric']}
            )

        logger.info("Processing all breed documents...")
        documents = self.doc_processor.process_all_breeds()

        logger.info("Generating embeddings...")
        documents = self.doc_processor.generate_embeddings(documents)

        logger.info(f"Ingesting {len(documents)} documents into ChromaDB...")

        # Prepare data for ChromaDB
        ids = [doc.doc_id for doc in documents]
        embeddings = [doc.embedding.tolist() for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents_text[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"✓ Ingested {len(documents)} documents successfully")
        return len(documents)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        if top_k is None:
            top_k = RETRIEVAL_CONFIG['top_k']

        # Generate query embedding
        query_embedding = self.doc_processor.embedding_model.encode(
            query,
            convert_to_numpy=True,
        )

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
            }
            documents.append(doc)

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using cross-encoder.

        Args:
            query: Original search query
            documents: List of retrieved documents
            top_n: Number of top documents to keep after reranking

        Returns:
            Reranked documents
        """
        if not self.reranker:
            logger.warning("Reranker not initialized, returning original documents")
            return documents

        if top_n is None:
            top_n = RETRIEVAL_CONFIG['rerank_top_n']

        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        logger.info(f"Reranked {len(documents)} documents, returning top {top_n}")
        return reranked[:top_n]

    def generate_answer(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        include_images: bool = None,
    ) -> Dict[str, Any]:
        """
        Generate answer using Ollama LLM.

        Args:
            query: User question
            context_documents: Retrieved and reranked documents
            include_images: Whether to include images (for multimodal models)

        Returns:
            Dictionary with answer and metadata
        """
        if include_images is None:
            include_images = self.model_config['supports_vision']

        # Build context string
        context = "\n\n".join([
            f"Document {i + 1} (Breed: {doc['metadata'].get('breed', 'Unknown')}):\n{doc['content']}"
            for i, doc in enumerate(context_documents)
        ])

        # Handle images for multimodal models
        image_descriptions = ""
        images_data = []

        if include_images and self.model_config['supports_vision']:
            # Get images from context documents
            for doc in context_documents[:1]:  # Use images from top document
                breed = doc['metadata'].get('breed')
                if breed:
                    images = self.doc_processor.get_breed_images(breed, max_images=2)
                    images_data.extend(images)

            if images_data:
                image_descriptions = f"Provided {len(images_data)} images of the breed."

        # Choose prompt template
        if include_images and images_data:
            prompt_template = PROMPTS['multimodal_rag']
            prompt = prompt_template.format(
                context=context,
                image_descriptions=image_descriptions,
                question=query,
            )
        else:
            prompt_template = PROMPTS['text_only_rag']
            prompt = prompt_template.format(
                context=context,
                question=query,
            )

        # Call Ollama API
        logger.info(f"Generating answer with {self.model_config['name']}...")

        # Prepare request payload
        payload = {
            "model": self.model_config['name'],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config['temperature'],
                "top_p": self.model_config['top_p'],
            }
        }

        # Add images for multimodal models
        if images_data:
            payload["images"] = [img['base64'] for img in images_data[:2]]  # Max 2 images

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            answer = result.get('response', 'No response generated')

            return {
                'answer': answer,
                'context_documents': context_documents,
                'model': self.model_config['name'],
                'images_used': len(images_data),
                'prompt': prompt,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'context_documents': context_documents,
                'model': self.model_config['name'],
                'images_used': 0,
                'error': str(e),
            }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query: retrieve, rerank, and generate.

        Args:
            question: User question
            top_k: Number of documents to retrieve initially
            use_reranking: Whether to use reranking (overrides instance setting)

        Returns:
            Complete RAG response with answer and metadata
        """
        logger.info(f"Processing query: {question}")

        # Retrieve documents
        documents = self.retrieve(query=question, top_k=top_k)

        # Rerank if enabled
        if use_reranking is None:
            use_reranking = self.use_reranking

        if use_reranking and self.reranker:
            documents = self.rerank(query=question, documents=documents)

        # Generate answer
        result = self.generate_answer(query=question, context_documents=documents)

        # Add retrieval metadata
        result['retrieval_count'] = len(documents)
        result['reranking_used'] = use_reranking
        result['chunking_strategy'] = self.chunking_strategy

        return result

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection."""
        count = self.collection.count()

        # Get sample documents to analyze
        sample = self.collection.get(limit=min(100, count))

        breeds = set()
        if sample['metadatas']:
            breeds = set(meta.get('breed', 'Unknown') for meta in sample['metadatas'])

        return {
            'total_documents': count,
            'unique_breeds': len(breeds),
            'collection_name': self.collection.name,
            'chunking_strategy': self.chunking_strategy,
            'distance_metric': CHROMA_CONFIG['distance_metric'],
        }


def main():
    """Test RAG engine."""
    print("=" * 60)
    print("RAG Engine Test")
    print("=" * 60)

    # Test with text-only model
    print("\n1. Testing Text-Only RAG (Llama3)")
    print("-" * 60)

    engine = RAGEngine(
        model_type="text_only",
        use_reranking=True,
        chunking_strategy="combined",
    )

    # Ingest documents (will skip if already exists)
    doc_count = engine.ingest_documents()
    print(f"Documents in collection: {doc_count}")

    # Get stats
    stats = engine.get_collection_stats()
    print(f"\nCollection Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test query
    print("\n2. Testing Query")
    print("-" * 60)

    test_queries = [
        "What are good dog breeds for families with children?",
        "Which breeds are hypoallergenic?",
        "Tell me about Golden Retrievers",
    ]

    for query in test_queries[:1]:  # Test with first query only
        print(f"\nQuery: {query}")
        result = engine.query(query)

        print(f"\nAnswer: {result['answer'][:300]}...")
        print(f"\nMetadata:")
        print(f"  Model: {result['model']}")
        print(f"  Retrieved docs: {result['retrieval_count']}")
        print(f"  Reranking used: {result['reranking_used']}")
        print(f"  Context breeds: {[doc['metadata']['breed'] for doc in result['context_documents']]}")

    print("\n" + "=" * 60)
    print("✓ RAG Engine test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
