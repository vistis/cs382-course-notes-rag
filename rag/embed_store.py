import os
import math
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from .ingest import Chunk


class VectorStore:
    def __init__(self):
        """Setup embedder, re-ranker, and ChromaDB."""
        self.embedder = SentenceTransformer(
            "model/SentenceTransformer/", device=os.getenv("DEVICE")
        )
        self.reranker = CrossEncoder(
            "model/CrossEncoder/", device=os.getenv("DEVICE")
        )

        self.store = chromadb.Client()
        self.collection = None
        self.collection_id = None
        self.dataset_chunk_registry = {}

    def build(self, chunks: List[Chunk], dataset: str) -> None:
        """Embed the chunks as vectors and store in ChromaDB."""
        self.collection_id = dataset.replace(" ", "_").lower()
        self.collection = self.store.get_or_create_collection(
            name=self.collection_id
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

    def query(self, query_text: str, top_k: int = 3, threshold: float = 0.20) -> List[Tuple[Chunk, float]]:
        """Vectorize the query and retrieve top_k matching results."""
        if self.collection is None:
            return []

        chunks = self.dataset_chunk_registry.get(self.collection_id, [])
        chunk_ids = {c.chunk_id: c for c in chunks}

        k_candidates = top_k * 5

        query_vec = self.embedder.encode([f"search_query={query_text}"], device=os.getenv("DEVICE"))
        results = self.collection.query(
            query_embeddings=query_vec.tolist(), n_results=k_candidates
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        matched_ids = results["ids"][0]

        candidates = []
        for doc_id in matched_ids:
            chunk = chunk_ids.get(doc_id)
            if chunk:
                candidates.append(chunk)

        if not candidates:
            return []

        rerank_inputs = [[query_text, chunk.text] for chunk in candidates]
        rerank_scores = self.reranker.predict(rerank_inputs)

        chunk_scores = []
        for chunk, raw_score in zip(candidates, rerank_scores):
            similarity_score = 1 / (1 + math.exp(-raw_score))

            if similarity_score >= threshold:
                chunk_scores.append((chunk, similarity_score))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        return chunk_scores[:top_k]
