"""
Vector Client for Economic Analysis semantic search
"""
from typing import List, Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

from config.settings import VECTOR_DB_PATH, VECTOR_COLLECTION, EMBEDDING_MODEL


class VectorClient:
    """ChromaDB client for semantic search."""

    def __init__(self, db_path: str = None, collection_name: str = None):
        self.db_path = str(db_path or VECTOR_DB_PATH)
        self.collection_name = collection_name or VECTOR_COLLECTION
        self._client = None
        self._collection = None
        self._model = None

    @property
    def client(self):
        if self._client is None:
            if not CHROMADB_AVAILABLE:
                raise RuntimeError("chromadb package not installed")
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def model(self):
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise RuntimeError("sentence-transformers package not installed")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.model.encode(text).tolist()

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Dict = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity

                if score >= min_score:
                    output.append({
                        'report_id': doc_id,
                        'score': score,
                        'document': results['documents'][0][i] if results['documents'] else None,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })

        return output

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'total_vectors': self.collection.count(),
            'collection_name': self.collection_name,
        }


if __name__ == "__main__":
    client = VectorClient()
    print("Stats:", client.get_stats())
