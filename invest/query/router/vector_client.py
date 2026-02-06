"""
Vector Search Client for Investment Strategy Reports

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
    """Vector search client for semantic queries on Investment Strategy reports."""

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
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model ready!", flush=True)
        return self._embedding_model

    def warmup(self):
        """Force-load the embedding model for faster first query."""
        _ = self.embedding_model
        return True

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
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

    Combines title, summary, themes, and key insights into a single searchable text.
    """
    parts = []

    # Title (most important)
    if report.get('title'):
        parts.append(f"제목: {report['title']}")

    # Summary
    if report.get('summary'):
        parts.append(f"요약: {report['summary']}")

    # Key thesis
    if report.get('key_thesis'):
        thesis = report['key_thesis']
        if isinstance(thesis, list):
            thesis = ' '.join(str(t) for t in thesis)
        parts.append(f"핵심 논지: {thesis}")

    # Market outlook
    if report.get('market_outlook'):
        parts.append(f"시장 전망: {report['market_outlook']}")

    # Do list (recommendations)
    if report.get('do_list'):
        do_items = report['do_list']
        if isinstance(do_items, list):
            do_items = ', '.join(str(d) for d in do_items[:5])
        parts.append(f"추천: {do_items}")

    # Avoid list
    if report.get('avoid_list'):
        avoid_items = report['avoid_list']
        if isinstance(avoid_items, list):
            avoid_items = ', '.join(str(a) for a in avoid_items[:5])
        parts.append(f"회피: {avoid_items}")

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
            "AI 반도체 투자 전략",
            "채권 비중 조절",
            "미국 금리 인상 영향",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            results = client.search_with_scores(query, n_results=3)
            for r in results:
                print(f"  [{r['similarity_score']:.3f}] {r['metadata'].get('title', 'N/A')[:50]}")
    else:
        print("\nNo documents in collection. Run ingestion first.")
