"""ChromaDB reader for semantic search."""
import json
import asyncio
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings


class ChromaDBReader:
    """
    Handles reading/searching from Chroma DB with Jina embedding function.
    """

    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "diabetes_guidelines_v1",
        embedding_function=None,
    ):
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.client = None
        self.collection = None

    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            print(f"✓ ChromaDB client initialized: {self.chroma_db_path}")

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✓ Loaded collection: {self.collection_name}")
            print(f"  • Total chunks: {self.collection.count()}")
        except Exception:
            if self.embedding_function:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )
                print(f"✓ Loaded collection: {self.collection_name}")
                print(f"  • Total chunks: {self.collection.count()}")
            else:
                raise Exception(
                    f"Collection '{self.collection_name}' not found. "
                    "Make sure you've run 04_vector_store_v1.ipynb first."
                )

    def _unflatten_metadata(self, flat_metadata: Dict) -> Dict:
        """Unflatten metadata (parse JSON strings back to objects)."""
        unflattened = {}
        for key, value in flat_metadata.items():
            try:
                if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
                    unflattened[key] = json.loads(value)
                else:
                    unflattened[key] = value
            except Exception:
                unflattened[key] = value
        return unflattened

    def _search_sync(
        self, query: str, n_results: int = 5, where: Dict | None = None, min_similarity: float = 0.4
    ) -> List[Dict]:
        """
        Synchronous implementation of search.
        """
        if not self.collection:
            self.initialize()

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        seen_chunk_ids = set()

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            relevance_score = 1 - results["distances"][0][i]

            if relevance_score < min_similarity:
                continue

            if chunk_id in seen_chunk_ids:
                continue

            chunk_data = {
                "chunk_id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": self._unflatten_metadata(results["metadatas"][0][i]),
                "relevance_score": relevance_score,
                "distance": results["distances"][0][i],
            }
            formatted_results.append(chunk_data)
            seen_chunk_ids.add(chunk_id)

        return formatted_results

    async def search(
        self, query: str, n_results: int = 5, where: Dict | None = None, min_similarity: float = 0.4
    ) -> List[Dict]:
        """
        Search the collection with semantic search (async).
        """
        return await asyncio.to_thread(self._search_sync, query, n_results, where, min_similarity)
