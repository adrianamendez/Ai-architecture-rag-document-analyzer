"""
RAG Engine with vector search, reranking, and Ollama LLM integration.
Supports both text-only and multimodal RAG strategies.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import chromadb
import requests
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_CONFIG,
    RETRIEVAL_CONFIG,
    OLLAMA_BASE_URL,
    MODELS,
    PROMPTS,
)
from document_processor import DocumentProcessor

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
        image_collection_name = f"{collection_name}_images"

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

        # Separate image embedding collection for multimodal retrieval.
        # It is created for every engine to keep ingestion/query logic consistent.
        try:
            self.image_collection = self.chroma_client.get_collection(
                name=image_collection_name
            )
            logger.info(f"Loaded existing image collection: {image_collection_name}")
        except:
            self.image_collection = self.chroma_client.create_collection(
                name=image_collection_name,
                metadata={"hnsw:space": CHROMA_CONFIG['distance_metric']}
            )
            logger.info(f"Created image collection: {image_collection_name}")

        # Initialize reranker if enabled
        self.reranker = None
        if use_reranking:
            logger.info(f"Loading reranker: {RETRIEVAL_CONFIG['reranker_model']}")
            self.reranker = CrossEncoder(RETRIEVAL_CONFIG['reranker_model'])

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure metadata values are compatible with Chroma types.
        Nested dicts/objects are converted to JSON strings.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = value
            else:
                # Convert nested dicts/objects to JSON-safe strings.
                try:
                    sanitized[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    sanitized[key] = str(value)
        return sanitized

    def ingest_documents(self, force_rebuild: bool = False) -> int:
        """
        Ingest all dog breed documents into vector database.

        Args:
            force_rebuild: If True, delete existing collection and rebuild

        Returns:
            Number of documents ingested
        """
        if force_rebuild:
            logger.info("Force rebuild: Deleting existing collections")
            text_collection_name = self.collection.name
            image_collection_name = self.image_collection.name
            self.chroma_client.delete_collection(name=text_collection_name)
            self.chroma_client.delete_collection(name=image_collection_name)
            self.collection = self.chroma_client.create_collection(
                name=text_collection_name,
                metadata={"hnsw:space": CHROMA_CONFIG['distance_metric']}
            )
            self.image_collection = self.chroma_client.create_collection(
                name=image_collection_name,
                metadata={"hnsw:space": CHROMA_CONFIG['distance_metric']}
            )

        text_count = self.collection.count()
        image_count = self.image_collection.count()

        # Both collections already available.
        if text_count > 0 and image_count > 0 and not force_rebuild:
            logger.info(
                f"Collections already populated (text={text_count}, image={image_count})"
            )
            return text_count

        # Text already available; only image vectors missing.
        if text_count > 0 and image_count == 0 and not force_rebuild:
            logger.info("Text collection already populated; ingesting missing image vectors only")
            self._ingest_image_vectors()
            return text_count

        logger.info("Processing all breed documents...")
        documents = self.doc_processor.process_all_breeds()

        logger.info("Generating embeddings...")
        documents = self.doc_processor.generate_embeddings(documents)

        logger.info(f"Ingesting {len(documents)} documents into ChromaDB...")

        # Prepare data for ChromaDB
        ids = [doc.doc_id for doc in documents]
        embeddings = [doc.embedding.tolist() for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents]

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

        # Ingest image vectors for multimodal retrieval.
        if self.image_collection.count() == 0:
            self._ingest_image_vectors()

        return len(documents)

    def _ingest_image_vectors(self) -> int:
        """Ingest CLIP image embeddings into dedicated image collection."""
        logger.info("Processing image entries for multimodal indexing...")
        image_entries = self.doc_processor.process_all_images_for_indexing(
            max_images_per_breed=5
        )
        if not image_entries:
            logger.warning("No image entries available to ingest")
            return 0

        ids = [entry["id"] for entry in image_entries]
        embeddings = [entry["embedding"].tolist() for entry in image_entries]
        documents_text = [entry["document"] for entry in image_entries]
        metadatas = [self._sanitize_metadata(entry["metadata"]) for entry in image_entries]

        batch_size = 100
        for i in range(0, len(image_entries), batch_size):
            batch_end = min(i + batch_size, len(image_entries))
            self.image_collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents_text[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"✓ Ingested {len(image_entries)} image vectors successfully")
        return len(image_entries)

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

    def retrieve_images(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant image entries in CLIP space using text query.
        """
        if top_k is None:
            top_k = RETRIEVAL_CONFIG["image_top_k"]

        if self.image_collection.count() == 0:
            logger.info("Image collection is empty; skipping image retrieval")
            return []

        query_embedding = self.doc_processor.encode_query_in_image_space(query)
        results = self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        images = []
        for i in range(len(results["ids"][0])):
            images.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                }
            )
        logger.info(f"Retrieved {len(images)} image entries")
        return images

    @staticmethod
    def _normalize_scores(values: List[float]) -> List[float]:
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [1.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def fuse_multimodal_results(
        self,
        text_docs: List[Dict[str, Any]],
        image_docs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fuse text and image retrieval by breed score, then return best text docs.
        """
        if not text_docs:
            return []

        text_weight = RETRIEVAL_CONFIG.get("text_weight", 0.7)
        image_weight = RETRIEVAL_CONFIG.get("image_weight", 0.3)

        text_scores = self._normalize_scores([float(d.get("similarity", 0.0)) for d in text_docs])
        image_scores = self._normalize_scores([float(d.get("similarity", 0.0)) for d in image_docs])

        breed_text_score: Dict[str, float] = {}
        best_text_doc_by_breed: Dict[str, Dict[str, Any]] = {}
        for doc, score in zip(text_docs, text_scores):
            breed = doc.get("metadata", {}).get("breed", "Unknown")
            if score > breed_text_score.get(breed, -1.0):
                breed_text_score[breed] = score
                best_text_doc_by_breed[breed] = doc

        breed_image_score: Dict[str, float] = {}
        for doc, score in zip(image_docs, image_scores):
            breed = doc.get("metadata", {}).get("breed", "Unknown")
            if score > breed_image_score.get(breed, -1.0):
                breed_image_score[breed] = score

        fused = []
        for breed, text_score in breed_text_score.items():
            image_score = breed_image_score.get(breed, 0.0)
            fusion_score = text_weight * text_score + image_weight * image_score
            doc = dict(best_text_doc_by_breed[breed])
            doc["text_norm_score"] = float(text_score)
            doc["image_norm_score"] = float(image_score)
            doc["fusion_score"] = float(fusion_score)
            fused.append(doc)

        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        logger.info(
            f"Fused multimodal results: {len(fused)} breeds "
            f"(text_weight={text_weight}, image_weight={image_weight})"
        )
        return fused[:top_k]

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

        # Try Ollama-native route first, then common compatible routes.
        endpoints = [
            f"{OLLAMA_BASE_URL}/api/generate",
            f"{OLLAMA_BASE_URL}/api/chat",
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
        ]

        last_error = None
        result = None
        answer = None
        used_endpoint = None

        for endpoint in endpoints:
            try:
                if endpoint.endswith("/api/generate"):
                    req_payload = payload
                elif endpoint.endswith("/api/chat"):
                    req_payload = {
                        "model": self.model_config["name"],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": payload["options"],
                    }
                    if images_data:
                        req_payload["images"] = [img["base64"] for img in images_data[:2]]
                else:
                    req_payload = {
                        "model": self.model_config["name"],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "temperature": self.model_config["temperature"],
                        "top_p": self.model_config["top_p"],
                    }

                response = requests.post(endpoint, json=req_payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                used_endpoint = endpoint

                if endpoint.endswith("/api/generate"):
                    answer = result.get("response", "No response generated")
                elif endpoint.endswith("/api/chat"):
                    answer = result.get("message", {}).get("content", "No response generated")
                else:
                    answer = (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "No response generated")
                    )
                break
            except Exception as e:
                last_error = e
                continue

        try:
            if answer is None:
                raise RuntimeError(str(last_error) if last_error else "Unknown generation error")

            return {
                'answer': answer,
                'context_documents': context_documents,
                'model': self.model_config['name'],
                'images_used': len(images_data),
                'prompt': prompt,
                'endpoint_used': used_endpoint,
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
        if top_k is None:
            top_k = RETRIEVAL_CONFIG["top_k"]

        # Retrieve documents (text-only or fused multimodal)
        if self.model_config.get("supports_vision", False):
            text_docs = self.retrieve(query=question, top_k=max(top_k, RETRIEVAL_CONFIG["top_k"]))
            image_docs = self.retrieve_images(query=question, top_k=RETRIEVAL_CONFIG["image_top_k"])
            documents = self.fuse_multimodal_results(
                text_docs=text_docs,
                image_docs=image_docs,
                top_k=top_k,
            )
        else:
            documents = self.retrieve(query=question, top_k=top_k)
            image_docs = []

        # Rerank if enabled
        if use_reranking is None:
            use_reranking = self.use_reranking

        if use_reranking and self.reranker:
            documents = self.rerank(query=question, documents=documents)

        # Generate answer
        result = self.generate_answer(query=question, context_documents=documents)

        # Add retrieval metadata
        result['retrieval_count'] = len(documents)
        result['image_retrieval_count'] = len(image_docs)
        result['reranking_used'] = use_reranking
        result['chunking_strategy'] = self.chunking_strategy
        result['multimodal_fusion_used'] = bool(self.model_config.get("supports_vision", False))

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
            'total_image_vectors': self.image_collection.count(),
            'unique_breeds': len(breeds),
            'collection_name': self.collection.name,
            'image_collection_name': self.image_collection.name,
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
