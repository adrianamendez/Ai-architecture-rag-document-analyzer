"""
Document Processor for multimodal dog breed data.
Handles ingestion of images and text, applies chunking strategies, and generates embeddings.
"""

import json
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    BREED_MAPPING_FILE,
    IMAGES_DIR,
    EMBEDDING_CONFIG,
    CHUNKING_STRATEGIES,
)

logger = logging.getLogger(__name__)


class Document:
    """Represents a single document chunk with metadata."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id or f"{metadata.get('breed', 'unknown')}_{metadata.get('chunk_id', 0)}"
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
        }


class DocumentProcessor:
    """Process dog breed documents (text + images) with various chunking strategies."""

    def __init__(self, chunking_strategy: str = "combined"):
        """
        Initialize document processor.

        Args:
            chunking_strategy: One of 'fixed_size', 'semantic', or 'combined'
        """
        self.chunking_strategy = chunking_strategy
        self.strategy_config = CHUNKING_STRATEGIES.get(chunking_strategy, {})

        # Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_CONFIG['text_model']}")
        self.embedding_model = SentenceTransformer(EMBEDDING_CONFIG['text_model'])

        # Load breed data
        self.breed_data = self._load_breed_data()
        logger.info(f"Loaded {len(self.breed_data)} breeds")

    @staticmethod
    def _slug(value: str) -> str:
        """Create a safe, stable slug for IDs."""
        slug = re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')
        return slug or "unknown"

    def _make_doc_id(self, breed_name: str, breed_info: Dict[str, Any], suffix: str) -> str:
        """
        Build a stable unique document ID.
        Includes image folder to avoid collisions when breed names are duplicated.
        """
        breed_slug = self._slug(breed_name)
        folder_slug = self._slug(breed_info.get('image_folder', 'unknown_folder'))
        source_idx = breed_info.get('source_index', 'na')
        return f"{breed_slug}__{folder_slug}__src{source_idx}__{suffix}"

    def _load_breed_data(self) -> Dict[str, Any]:
        """Load breed mapping data."""
        with open(BREED_MAPPING_FILE, 'r') as f:
            data = json.load(f)

        # Preserve duplicate breed names by creating unique keys for subsequent rows.
        # First row keeps the plain name for backward compatibility with .get("Breed Name").
        breed_map: Dict[str, Any] = {}
        seen: Dict[str, int] = {}
        for idx, breed in enumerate(data['breeds']):
            name = breed['name']
            seen[name] = seen.get(name, 0) + 1
            unique_key = name if seen[name] == 1 else f"{name}__{seen[name]}"

            enriched = dict(breed)
            enriched['canonical_name'] = name
            enriched['source_index'] = idx
            enriched['unique_key'] = unique_key
            breed_map[unique_key] = enriched

        return breed_map

    def process_all_breeds(self) -> List[Document]:
        """
        Process all dog breeds and return document chunks.

        Returns:
            List of Document objects
        """
        all_documents = []

        for breed_name, breed_info in self.breed_data.items():
            logger.info(f"Processing breed: {breed_name}")
            breed_docs = self.process_breed(breed_name, breed_info)
            all_documents.extend(breed_docs)

        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents

    def process_breed(self, breed_name: str, breed_info: Dict[str, Any]) -> List[Document]:
        """
        Process a single breed using the configured chunking strategy.

        Args:
            breed_name: Name of the breed
            breed_info: Breed information from mapping file

        Returns:
            List of Document objects for this breed
        """
        canonical_name = breed_info.get('canonical_name', breed_name)

        if self.chunking_strategy == "fixed_size":
            return self._chunk_fixed_size(canonical_name, breed_info)
        elif self.chunking_strategy == "semantic":
            return self._chunk_semantic(canonical_name, breed_info)
        else:  # combined
            return self._chunk_combined(canonical_name, breed_info)

    def _chunk_fixed_size(
        self, breed_name: str, breed_info: Dict[str, Any]
    ) -> List[Document]:
        """
        Create fixed-size chunks from breed text.

        This strategy treats all text as a continuous stream and chunks by character count.
        """
        documents = []
        characteristics = breed_info['characteristics']

        # Combine all text
        full_text = f"Breed: {breed_name}\n"
        for key, value in characteristics.items():
            full_text += f"{key}: {value}\n"

        # Approx token-based chunking (whitespace tokenization).
        chunk_size = self.strategy_config.get('chunk_size', 512)
        chunk_overlap = self.strategy_config.get('chunk_overlap', 50)
        step = max(1, chunk_size - chunk_overlap)
        tokens = full_text.split()

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            if chunk_tokens:
                chunk = " ".join(chunk_tokens)
                chunk_id = i // step
                doc = Document(
                    content=chunk,
                    metadata={
                        'breed': breed_name,
                        'chunk_id': chunk_id,
                        'chunk_strategy': 'fixed_size',
                        'image_folder': breed_info['image_folder'],
                        'token_start': i,
                        'token_end': i + len(chunk_tokens),
                    },
                    doc_id=self._make_doc_id(breed_name, breed_info, f"fixed_{chunk_id}")
                )
                documents.append(doc)

        return documents

    def _chunk_semantic(
        self, breed_name: str, breed_info: Dict[str, Any]
    ) -> List[Document]:
        """
        Create semantic chunks by breed attributes.

        Each characteristic becomes its own chunk for precise retrieval.
        """
        documents = []
        characteristics = breed_info['characteristics']

        # Create sentence-level semantic units, then window by sentence count.
        semantic_sentences = []
        for attr, value in characteristics.items():
            if str(value).strip():
                semantic_sentences.append(f"{attr}: {value}.")

        if not semantic_sentences:
            return documents

        max_sentences = self.strategy_config.get('max_sentences_per_chunk', 3)
        sentence_overlap = self.strategy_config.get('sentence_overlap', 1)
        step = max(1, max_sentences - sentence_overlap)

        for i in range(0, len(semantic_sentences), step):
            chunk_sentences = semantic_sentences[i:i + max_sentences]
            if not chunk_sentences:
                continue

            chunk_id = i // step
            chunk_text = f"Breed: {breed_name}\n" + "\n".join(chunk_sentences)
            doc = Document(
                content=chunk_text,
                metadata={
                    'breed': breed_name,
                    'chunk_id': chunk_id,
                    'chunk_strategy': 'semantic',
                    'image_folder': breed_info['image_folder'],
                    'sentence_start': i,
                    'sentence_end': i + len(chunk_sentences),
                },
                doc_id=self._make_doc_id(breed_name, breed_info, f"semantic_{chunk_id}")
            )
            documents.append(doc)

        return documents

    def _chunk_combined(
        self, breed_name: str, breed_info: Dict[str, Any]
    ) -> List[Document]:
        """
        Create combined multimodal chunks (text + image metadata).

        This strategy creates a comprehensive chunk with text and references to images.
        """
        characteristics = breed_info['characteristics']

        # Build comprehensive text chunk
        chunk_text = f"Breed: {breed_name}\n\n"

        # Add all characteristics
        for key, value in characteristics.items():
            chunk_text += f"{key}: {value}\n"

        # Get image paths
        image_folder = IMAGES_DIR / breed_info['image_folder']
        image_paths = []
        if image_folder.exists():
            image_paths = list(image_folder.glob('*.jpg'))[:5]  # Take first 5 images

        # Add image metadata
        if image_paths:
            chunk_text += f"\nAvailable Images: {len(image_paths)} images\n"
            chunk_text += f"Image Folder: {breed_info['image_folder']}\n"

        # Create single comprehensive document
        doc = Document(
            content=chunk_text,
            metadata={
                'breed': breed_name,
                'chunk_id': 0,
                'chunk_strategy': 'combined',
                'image_folder': breed_info['image_folder'],
                'image_count': len(image_paths),
                'image_paths': [str(p) for p in image_paths],
                'characteristics': characteristics,
            },
            doc_id=self._make_doc_id(breed_name, breed_info, "combined_0")
        )

        return [doc]

    def generate_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for all documents.

        Args:
            documents: List of Document objects

        Returns:
            Same documents with embeddings added
        """
        logger.info(f"Generating embeddings for {len(documents)} documents...")

        # Extract text content
        texts = [doc.content for doc in documents]

        # Generate embeddings in batches
        batch_size = EMBEDDING_CONFIG['batch_size']
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            all_embeddings.extend(embeddings)

        # Attach embeddings to documents
        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding

        logger.info("Embeddings generated successfully")
        return documents

    def get_image_base64(self, image_path: Path) -> str:
        """
        Convert image to base64 string for LLaVA.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def process_image_for_vision_model(self, image_path: Path) -> Dict[str, Any]:
        """
        Process image for vision model input.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image data and metadata
        """
        import io

        img = Image.open(image_path)

        # Convert to RGB if necessary (JPEG doesn't support RGBA or P mode)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Resize if too large (LLaVA works best with smaller images)
        max_size = 512
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert resized image to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            'path': str(image_path),
            'base64': image_b64,
            'size': img.size,
            'mode': img.mode,
        }

    def get_breed_images(self, breed_name: str, max_images: int = 3) -> List[Dict[str, Any]]:
        """
        Get processed images for a specific breed.

        Args:
            breed_name: Name of the breed
            max_images: Maximum number of images to return

        Returns:
            List of processed image dictionaries
        """
        breed_info = self.breed_data.get(breed_name)
        if not breed_info:
            return []

        image_folder = IMAGES_DIR / breed_info['image_folder']
        if not image_folder.exists():
            return []

        image_paths = list(image_folder.glob('*.jpg'))[:max_images]
        return [self.process_image_for_vision_model(p) for p in image_paths]


def main():
    """Test document processing."""
    print("=" * 60)
    print("Document Processor Test")
    print("=" * 60)

    # Test each chunking strategy
    for strategy in ["fixed_size", "semantic", "combined"]:
        print(f"\n\nTesting {strategy} chunking strategy:")
        print("-" * 60)

        processor = DocumentProcessor(chunking_strategy=strategy)

        # Process first 3 breeds
        sample_breeds = list(processor.breed_data.items())[:3]
        all_docs = []

        for breed_name, breed_info in sample_breeds:
            docs = processor.process_breed(breed_name, breed_info)
            all_docs.extend(docs)
            print(f"\n{breed_name}: {len(docs)} chunks created")
            for i, doc in enumerate(docs[:2]):  # Show first 2 chunks
                print(f"  Chunk {i + 1} preview: {doc.content[:150]}...")

        # Generate embeddings
        docs_with_embeddings = processor.generate_embeddings(all_docs)
        print(f"\nTotal documents: {len(docs_with_embeddings)}")
        print(f"Embedding shape: {docs_with_embeddings[0].embedding.shape}")

    print("\n" + "=" * 60)
    print("✓ Document processing test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


# Backward-compatible alias used by notebooks/imports created before refactor.
MultimodalDocumentProcessor = DocumentProcessor
