"""
Vector DB Client for Semantic Search

Uses ChromaDB + sentence-transformers for FREE local vector search.
Replaces Gemini File API for cost-effective semantic Q&A.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import hashlib

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

from config.settings import get_settings


class VectorClient:
    """
    Vector DB client for semantic search using ChromaDB.

    Cost: $0 (local embeddings + local storage)
    Speed: ~10-50ms per query
    """

    # Korean-optimized multilingual model
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, settings=None, model_name: Optional[str] = None):
        self.settings = settings or get_settings()

        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package not installed. "
                "Run: pip install chromadb"
            )

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Run: pip install sentence-transformers"
            )

        # Setup paths
        self.vector_db_path = self.settings.project_root / "vector_db"
        self.vector_db_path.mkdir(exist_ok=True)

        # Initialize embedding model
        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create/get collection
        self.collection = self.client.get_or_create_collection(
            name="naver_reports",
            metadata={"hnsw:space": "cosine"}
        )

        # Track indexed files
        self._load_index_state()

    def _get_state_path(self) -> Path:
        """Get path to index state file."""
        return self.vector_db_path / "index_state.json"

    def _load_index_state(self):
        """Load indexed files state."""
        state_path = self._get_state_path()
        if state_path.exists():
            with open(state_path, 'r') as f:
                self._indexed_files = json.load(f)
        else:
            self._indexed_files = {}

    def _save_index_state(self):
        """Save indexed files state."""
        with open(self._get_state_path(), 'w') as f:
            json.dump(self._indexed_files, f, indent=2)

    def _file_hash(self, path: Path) -> str:
        """Get file hash for change detection."""
        return hashlib.md5(f"{path}_{path.stat().st_mtime}".encode()).hexdigest()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]

    def index_md_file(
        self,
        md_path: str,
        report_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Index a markdown file into the vector DB.

        Args:
            md_path: Path to the MD file
            report_id: Unique report identifier
            metadata: Additional metadata (company, issuer, date, etc.)
            force: Force re-indexing even if already indexed

        Returns:
            Indexing result with chunk count
        """
        path = Path(md_path)
        if not path.exists():
            return {'error': f'File not found: {md_path}', 'chunks': 0}

        # Check if already indexed
        file_hash = self._file_hash(path)
        if not force and md_path in self._indexed_files:
            if self._indexed_files[md_path] == file_hash:
                return {'status': 'already_indexed', 'chunks': 0}

        # Read content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {'error': str(e), 'chunks': 0}

        if not content.strip():
            return {'status': 'empty_file', 'chunks': 0}

        # Chunk the content
        chunks = self._chunk_text(content)

        # Prepare metadata
        base_metadata = metadata or {}
        base_metadata['report_id'] = report_id
        base_metadata['file_path'] = str(path)
        base_metadata['indexed_at'] = datetime.now().isoformat()

        # Generate embeddings and add to collection
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{report_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            chunk_meta = base_metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['chunk_count'] = len(chunks)
            metadatas.append(chunk_meta)

        # Delete existing chunks for this report
        try:
            existing = self.collection.get(
                where={"report_id": report_id}
            )
            if existing['ids']:
                self.collection.delete(ids=existing['ids'])
        except Exception:
            pass

        # Add new chunks
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Update state
        self._indexed_files[md_path] = file_hash
        self._save_index_state()

        return {
            'status': 'indexed',
            'chunks': len(chunks),
            'report_id': report_id
        }

    def index_batch(
        self,
        md_files: List[Dict[str, Any]],
        force: bool = False,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Index multiple MD files.

        Args:
            md_files: List of dicts with 'path', 'report_id', 'metadata'
            force: Force re-indexing
            progress_callback: Optional callback for progress updates

        Returns:
            Batch indexing results
        """
        results = {
            'total': len(md_files),
            'indexed': 0,
            'skipped': 0,
            'errors': 0,
            'total_chunks': 0
        }

        for i, file_info in enumerate(md_files):
            result = self.index_md_file(
                md_path=file_info['path'],
                report_id=file_info['report_id'],
                metadata=file_info.get('metadata', {}),
                force=force
            )

            if result.get('status') == 'indexed':
                results['indexed'] += 1
                results['total_chunks'] += result.get('chunks', 0)
            elif result.get('status') == 'already_indexed':
                results['skipped'] += 1
            else:
                results['errors'] += 1

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(md_files))

        return results

    def query(
        self,
        question: str,
        report_ids: Optional[List[str]] = None,
        company: Optional[str] = None,
        n_results: int = 10,
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Query the vector DB with semantic search.

        Args:
            question: User question
            report_ids: Optional list of report IDs to filter
            company: Optional company filter
            n_results: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            Query results with answer synthesis
        """
        # Build filter
        where_filter = None
        if report_ids:
            where_filter = {"report_id": {"$in": report_ids}}
        elif company:
            where_filter = {"company": company}

        # Generate query embedding
        query_embedding = self.embedding_model.encode([question])[0].tolist()

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        if not results['documents'][0]:
            return {
                'answer': "관련 문서를 찾을 수 없습니다.",
                'chunks_found': 0,
                'sources': []
            }

        # Process results (convert distance to similarity)
        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - dist  # cosine distance to similarity
            if similarity >= min_score:
                chunks.append({
                    'text': doc,
                    'metadata': meta,
                    'similarity': similarity
                })

        if not chunks:
            return {
                'answer': "관련성 높은 문서를 찾을 수 없습니다.",
                'chunks_found': 0,
                'sources': []
            }

        # Synthesize answer from chunks
        answer = self._synthesize_answer(question, chunks)

        # Get unique sources
        sources = []
        seen_reports = set()
        for chunk in chunks:
            report_id = chunk['metadata'].get('report_id', '')
            if report_id and report_id not in seen_reports:
                seen_reports.add(report_id)
                sources.append({
                    'report_id': report_id,
                    'company': chunk['metadata'].get('company', ''),
                    'issuer': chunk['metadata'].get('issuer', ''),
                    'similarity': chunk['similarity']
                })

        return {
            'answer': answer,
            'chunks_found': len(chunks),
            'sources': sources[:5],
            'top_chunks': chunks[:3]
        }

    def _synthesize_answer(self, question: str, chunks: List[Dict]) -> str:
        """
        Synthesize answer from retrieved chunks.

        Note: This is a simple extraction-based synthesis.
        For better answers, integrate with LLM (GPT/Claude).
        """
        # Extract key information
        companies = set()
        issuers = set()
        dates = set()
        key_texts = []

        for chunk in chunks[:5]:
            meta = chunk['metadata']
            if meta.get('company'):
                companies.add(meta['company'])
            if meta.get('issuer'):
                issuers.add(meta['issuer'])
            if meta.get('issue_date'):
                dates.add(meta['issue_date'][:10])

            # Extract relevant sentences
            text = chunk['text']
            sentences = text.split('.')
            for sent in sentences[:3]:
                if len(sent.strip()) > 20:
                    key_texts.append(sent.strip())

        # Build answer
        answer_parts = []

        if companies:
            answer_parts.append(f"관련 기업: {', '.join(sorted(companies))}")

        if issuers:
            answer_parts.append(f"분석 기관: {', '.join(sorted(issuers))}")

        if dates:
            sorted_dates = sorted(dates, reverse=True)
            answer_parts.append(f"최근 리포트: {sorted_dates[0]}")

        answer_parts.append("\n[관련 내용]")
        for i, text in enumerate(key_texts[:5], 1):
            answer_parts.append(f"{i}. {text}.")

        return "\n".join(answer_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector DB statistics."""
        count = self.collection.count()

        return {
            'total_chunks': count,
            'indexed_files': len(self._indexed_files),
            'model': self.model_name,
            'db_path': str(self.vector_db_path)
        }

    def clear_collection(self) -> int:
        """Clear all documents from collection. USE WITH CAUTION."""
        count = self.collection.count()

        # Delete collection and recreate
        self.client.delete_collection("naver_reports")
        self.collection = self.client.get_or_create_collection(
            name="naver_reports",
            metadata={"hnsw:space": "cosine"}
        )

        self._indexed_files = {}
        self._save_index_state()

        return count


# CLI for testing and indexing
if __name__ == "__main__":
    import sys

    try:
        client = VectorClient()
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        stats = client.get_stats()
        print(f"\nVector DB Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'index':
        # Index sample file
        path = sys.argv[2] if len(sys.argv) > 2 else None
        if path:
            result = client.index_md_file(
                md_path=path,
                report_id=Path(path).stem,
                metadata={'source': 'cli'}
            )
            print(f"Indexed: {result}")
        else:
            print("Usage: python -m query.vector_client index <path>")

    elif len(sys.argv) > 1 and sys.argv[1] == 'query':
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "삼성전자 투자의견"
        result = client.query(question)
        print(f"\nQuestion: {question}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nChunks found: {result['chunks_found']}")
        print(f"Sources: {len(result['sources'])}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'clear':
        confirm = input("This will delete ALL vectors. Type 'yes' to confirm: ")
        if confirm == 'yes':
            deleted = client.clear_collection()
            print(f"Cleared {deleted} chunks")
        else:
            print("Aborted")

    else:
        print("Usage: python -m query.vector_client [stats|index <path>|query <question>|clear]")
