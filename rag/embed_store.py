"""
Vector store: turn chunks into vectors and support similarity search over them.

This starter ships with a TF-IDF backend (same technique from the Week 14 lab) so
the whole project runs immediately with zero API keys and no model downloads.

Upgrade path (for your final project — do this once the pipeline works end-to-end):
- [DONE] Swap TfidfVectorizer for real embeddings, e.g.:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(texts)
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

import os
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from .ingest import Chunk


class VectorStore:
    def __init__(self):
        """Vectorize and embed to ChromaDB."""
        self.embedder = SentenceTransformer(
            "model/SentenceTransformer/", device=os.getenv("DEVICE")
        )

        self.store = chromadb.Client()
        self.collection = None
        self.collection_id = None
        self.dataset_chunk_registry = {}

    def build(self, chunks: List[Chunk], dataset: str) -> None:
        """Embed the chunks as vectors and store in ChromaDB."""
        self.collection_id = dataset.replace(" ", "_").lower()
        self.collection = self.store.get_or_create_collection(
            name=self.collection_id, metadata={"hnsw:space": "cosine"}
        )
        self.dataset_chunk_registry[self.collection_id] = chunks

        if self.collection.count() > 0:
            return

        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "title": c.doc_title,
                "author": c.doc_author,
                "date": c.doc_date
            }
            for c in chunks
        ]

        vectors = self.embedder.encode([f"search_document: {c.text}" for c in chunks], device=os.getenv("DEVICE"))
        self.collection.add(
            # ids=[f"c{i}" for i in range(len(documents))],
            ids=ids,
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadatas
        )

    def switch_collection(self, dataset: str) -> None:
        """Switch between collections without reindexing."""
        self.collection_id = dataset.replace(" ", "_").lower()
        self.collection = self.store.get_collection(
            name=self.collection_id
        )

    def query(self, query_text: str, top_k: int = 3, threshold: float = 1.00) -> List[Tuple[Chunk, float]]:
        """Vectorize the query and retrieve top_k matching results."""
        if self.collection is None:
            return []

        chunks = self.dataset_chunk_registry.get(self.collection_id, [])
        chunk_ids = {c.chunk_id: c for c in chunks}

        query_vec = self.embedder.encode([f"search_query={query_text}"], device=os.getenv("DEVICE"))
        results = self.collection.query(
            query_embeddings=query_vec.tolist(), n_results=top_k
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        matched_ids = results["ids"][0]
        distances = results["distances"][0]

        output = []
        for doc_id, distance in zip(matched_ids, distances):
            # idx = int(doc_id[1:])
            # if idx >= len(chunks):
            #     continue
            # chunk = chunks[idx]
            chunk = chunk_ids.get(doc_id)
            if not chunk:
                continue

            similarity_score = 1.0 - distance

            if similarity_score >= threshold:
                output.append((chunk, float(similarity_score)))

        return output
