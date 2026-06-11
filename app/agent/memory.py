"""
ContentCraft AI — Memory Store
Uses ChromaDB + sentence-transformers to persist past topic→script pairs.
Lets the agent reference prior generations for variety and speed.
"""

import json
import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class MemoryStore:
    def __init__(self, persist_dir: str = "chroma_db"):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="contentcraft_topics",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def save(self, topic: str, script: list) -> None:
        """Persist a topic + its generated script."""
        embedding = self._embedder.encode(topic).tolist()
        self._collection.upsert(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[topic],
            metadatas=[{"script_json": json.dumps(script)}],
        )

    async def search(self, topic: str, top_k: int = 2) -> List[dict]:
        """Return top_k similar past topics with their scripts."""
        if self._collection.count() == 0:
            return []

        embedding = self._embedder.encode(topic).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Only return if reasonably similar (cosine distance < 0.5)
            if dist < 0.5:
                out.append({
                    "topic": doc,
                    "script": json.loads(meta["script_json"]),
                    "similarity": round(1 - dist, 3),
                })
        return out

    def clear(self) -> None:
        """Wipe all stored memories (useful for testing)."""
        self._client.delete_collection("contentcraft_topics")
        self._collection = self._client.get_or_create_collection(
            name="contentcraft_topics",
            metadata={"hnsw:space": "cosine"},
        )


# Singleton
memory_store = MemoryStore()
