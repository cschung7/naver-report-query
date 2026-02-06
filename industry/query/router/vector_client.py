"""
Vector Search Client for Industry Analysis Reports

Uses ChromaDB for vector storage and sentence-transformers for Korean embeddings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Optional, Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from config.settings import VECTOR_DB_PATH, VECTOR_COLLECTION, EMBEDDING_MODEL


class VectorClient:
    """Vector search client for semantic queries on Industry Analysis reports."""

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        self.persist_directory = persist_directory or str(VECTOR_DB_PATH)
        self.collection_name = collection_name or VECTOR_COLLECTION
        self._client = None
        self._collection = None
        self._embedding_model = None

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            if not CHROMADB_AVAILABLE:
                raise RuntimeError("chromadb package not installed")
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        """Get or create the vector collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def embedding_model(self):
        """Lazy-load sentence-transformers model."""
        if self._embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...", flush=True)
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                print("Embedding model ready!", flush=True)
            except (ImportError, OSError) as e:
                print(f"Warning: Embedding model unavailable: {e}", flush=True)
                self._embedding_model = False  # Mark as failed, not None
        return self._embedding_model if self._embedding_model else None

    def warmup(self):
        """Force-load the embedding model for faster first query."""
        model = self.embedding_model
        return model is not None

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        model = self.embedding_model
        if model is None:
            return None
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for multiple texts."""
        model = self.embedding_model
        if model is None:
            return None
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict] = None,
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the vector collection.

        Args:
            documents: List of text documents to embed
            ids: Unique IDs for each document
            metadatas: Optional metadata for each document
            batch_size: Number of documents to process at once

        Returns:
            Number of documents added
        """
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size] if metadatas else None

            # Generate embeddings
            embeddings = self.embed_texts(batch_docs)
            if embeddings is None:
                raise RuntimeError("Embedding model unavailable - cannot add documents")

            # Add to collection
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                ids=batch_ids,
                metadatas=batch_meta
            )

            total_added += len(batch_docs)
            print(f"  Added {total_added}/{len(documents)} documents...", flush=True)

        return total_added

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Dict = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search for semantically similar documents.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            where: Optional metadata filter
            include: Fields to include in results (default: documents, metadatas, distances)

        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        # Generate query embedding
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )

        # Flatten results (query returns nested lists)
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results.get('documents') else [],
            'metadatas': results['metadatas'][0] if results.get('metadatas') else [],
            'distances': results['distances'][0] if results.get('distances') else [],
        }

    def search_with_scores(
        self,
        query: str,
        n_results: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search and return results with similarity scores.

        Args:
            query: Search query text
            n_results: Maximum number of results
            score_threshold: Minimum similarity score (0-1, cosine similarity)

        Returns:
            List of dicts with id, document, metadata, and similarity_score
        """
        results = self.search(query, n_results=n_results)

        scored_results = []
        for i, doc_id in enumerate(results['ids']):
            # Convert distance to similarity (cosine distance -> similarity)
            distance = results['distances'][i]
            similarity = 1 - distance  # Cosine distance to similarity

            if similarity >= score_threshold:
                scored_results.append({
                    'id': doc_id,
                    'document': results['documents'][i] if results['documents'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                    'similarity_score': round(similarity, 4)
                })

        return scored_results

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'persist_directory': self.persist_directory,
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self._collection = None

    def clear_collection(self):
        """Clear all documents from the collection."""
        # Get all IDs and delete them
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)


def create_report_embedding_text(report: Dict) -> str:
    """
    Create a text representation of a report for embedding.

    Combines title, summary, industry info, and key insights into a single searchable text.
    """
    parts = []

    # Title (most important)
    if report.get('title'):
        parts.append(f"제목: {report['title']}")

    # Summary
    if report.get('summary'):
        parts.append(f"요약: {report['summary']}")

    # Industry
    if report.get('industry'):
        parts.append(f"산업: {report['industry']}")

    # Cycle stage
    if report.get('cycle_stage'):
        parts.append(f"사이클: {report['cycle_stage']}")

    # Cycle drivers
    if report.get('cycle_drivers'):
        drivers = report['cycle_drivers']
        if isinstance(drivers, list):
            drivers = ', '.join(str(d) for d in drivers[:5])
        parts.append(f"사이클 동인: {drivers}")

    # Demand trend
    if report.get('demand_trend'):
        parts.append(f"수요 동향: {report['demand_trend']}")

    # Demand drivers
    if report.get('demand_drivers'):
        drivers = report['demand_drivers']
        if isinstance(drivers, list):
            drivers = ', '.join(str(d) for d in drivers[:5])
        parts.append(f"수요 동인: {drivers}")

    # Key themes
    if report.get('key_themes'):
        themes = report['key_themes']
        if isinstance(themes, list):
            themes = ', '.join(str(t) for t in themes[:5])
        parts.append(f"주요 테마: {themes}")

    # Investment timing
    if report.get('investment_timing'):
        parts.append(f"투자 시점: {report['investment_timing']}")

    # Geography
    if report.get('geography'):
        parts.append(f"지역: {report['geography']}")

    return '\n'.join(parts)


if __name__ == "__main__":
    # Test the vector client
    client = VectorClient()

    print("=" * 60)
    print("Vector Client Test")
    print("=" * 60)

    # Check stats
    stats = client.get_stats()
    print(f"\nCollection stats: {stats}")

    if stats['total_documents'] > 0:
        # Test search
        test_queries = [
            "반도체 산업 사이클 전망",
            "자동차 수요 증가",
            "철강 업황 회복",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            results = client.search_with_scores(query, n_results=3)
            for r in results:
                print(f"  [{r['similarity_score']:.3f}] {r['metadata'].get('title', 'N/A')[:50]}")
    else:
        print("\nNo documents in collection. Run ingestion first.")
