"""
Smart Query Router for Economic Analysis Reports
Combines PostgreSQL, Neo4j, and Vector search with intelligent caching.
"""
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .pg_client import PostgresClient
from .vector_client import VectorClient


@dataclass
class QueryResult:
    """Query result container."""
    query: str
    intent: str
    reports: List[Dict]
    answer: str = ""
    sources_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    cache_hit: str = "MISS"


class EconQuery:
    """Economic Analysis query router with multi-source search and caching."""

    # Intent patterns for economic analysis
    INTENT_PATTERNS = {
        'interest_rate': ['금리', '기준금리', 'rate', '인상', '인하', 'fed', '연준'],
        'exchange_rate': ['환율', '달러', '원화', '엔화', '유로', 'forex', 'fx'],
        'inflation': ['물가', '인플레이션', 'cpi', '소비자물가', '생산자물가'],
        'employment': ['고용', '실업', '일자리', '취업', 'employment', 'job'],
        'gdp_growth': ['gdp', '성장률', '경제성장', '경기', '성장'],
        'trade': ['무역', '수출', '수입', '무역수지', '경상수지'],
        'monetary_policy': ['통화정책', '양적완화', 'qe', '테이퍼링', '긴축'],
        'fiscal_policy': ['재정정책', '정부지출', '예산', '세금', '감세'],
        'global_economy': ['글로벌', '세계경제', '국제', 'global'],
        'us_economy': ['미국', '미 경제', 'us', '연준', 'fed'],
        'china_economy': ['중국', '중국경제', 'china', '인민은행'],
        'general_search': []
    }

    def __init__(
        self,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 100,
        semantic_cache_threshold: float = 0.85
    ):
        self.pg_client = PostgresClient()
        self.vector_client = VectorClient()

        # Cache settings
        self.cache_ttl = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self.semantic_threshold = semantic_cache_threshold

        # Caches
        self.exact_cache: Dict[str, tuple] = {}  # hash -> (result, timestamp, embedding)
        self.cache_embeddings: List[tuple] = []  # [(hash, embedding, timestamp)]

        # Metrics
        self.metrics = {
            'total_queries': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
        }

    def _detect_intent(self, query: str) -> str:
        """Detect query intent based on keywords."""
        query_lower = query.lower()

        for intent, keywords in self.INTENT_PATTERNS.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return 'general_search'

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _check_exact_cache(self, cache_key: str) -> Optional[QueryResult]:
        """Check exact cache match."""
        if cache_key in self.exact_cache:
            result, timestamp, _ = self.exact_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.exact_cache[cache_key]
        return None

    def _check_semantic_cache(self, query_embedding: np.ndarray) -> Optional[tuple]:
        """Check semantic cache for similar queries."""
        if not self.cache_embeddings:
            return None

        current_time = time.time()
        best_match = None
        best_score = 0

        valid_embeddings = []
        for cache_key, embedding, timestamp in self.cache_embeddings:
            if current_time - timestamp < self.cache_ttl:
                valid_embeddings.append((cache_key, embedding, timestamp))
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                if similarity > best_score and similarity >= self.semantic_threshold:
                    best_score = similarity
                    best_match = (cache_key, similarity)

        self.cache_embeddings = valid_embeddings
        return best_match

    def _add_to_cache(self, cache_key: str, result: QueryResult, embedding: np.ndarray):
        """Add result to cache."""
        timestamp = time.time()

        if len(self.exact_cache) >= self.max_cache_size:
            oldest_key = min(self.exact_cache, key=lambda k: self.exact_cache[k][1])
            del self.exact_cache[oldest_key]

        self.exact_cache[cache_key] = (result, timestamp, embedding)
        self.cache_embeddings.append((cache_key, embedding, timestamp))

    def query(
        self,
        question: str,
        max_reports: int = 10,
        use_cache: bool = True,
        date_from: str = None,
        date_to: str = None
    ) -> QueryResult:
        """Execute query with caching and multi-source search."""
        start_time = time.time()
        self.metrics['total_queries'] += 1

        cache_key = self._get_cache_key(question)

        # Check exact cache
        if use_cache:
            cached = self._check_exact_cache(cache_key)
            if cached:
                self.metrics['exact_hits'] += 1
                cached.cache_hit = "EXACT HIT"
                cached.execution_time_ms = round((time.time() - start_time) * 1000, 1)
                return cached

        # Generate embedding for semantic search and cache
        query_embedding = np.array(self.vector_client.embed_text(question))

        # Check semantic cache
        if use_cache:
            semantic_match = self._check_semantic_cache(query_embedding)
            if semantic_match:
                match_key, similarity = semantic_match
                if match_key in self.exact_cache:
                    self.metrics['semantic_hits'] += 1
                    result, _, _ = self.exact_cache[match_key]
                    result.cache_hit = f"SEMANTIC HIT ({int(similarity*100)}%)"
                    result.execution_time_ms = round((time.time() - start_time) * 1000, 1)
                    return result

        self.metrics['misses'] += 1

        # Detect intent
        intent = self._detect_intent(question)

        # Extract keywords
        keywords = [w for w in question.split() if len(w) >= 2]

        # Search PostgreSQL
        pg_results = self.pg_client.search_reports(
            query=question,
            keywords=keywords,
            limit=max_reports,
            date_from=date_from,
            date_to=date_to
        )

        # Search vectors
        vector_results = self.vector_client.search(
            query=question,
            n_results=max_reports,
            min_score=0.5
        )

        # Combine and deduplicate
        seen_ids = set()
        combined_reports = []

        for r in pg_results:
            if r['report_id'] not in seen_ids:
                seen_ids.add(r['report_id'])
                combined_reports.append(r)

        # Add vector results not in PostgreSQL results
        for vr in vector_results:
            if vr['report_id'] not in seen_ids:
                seen_ids.add(vr['report_id'])
                combined_reports.append({
                    'report_id': vr['report_id'],
                    'title': vr['metadata'].get('title', ''),
                    'issuer': vr['metadata'].get('issuer', ''),
                    'issue_date': vr['metadata'].get('issue_date', ''),
                    'summary': vr['document'][:500] if vr['document'] else '',
                    'vector_score': vr['score'],
                    'pdf_link': vr['metadata'].get('pdf_link', ''),
                })

        # Generate answer summary
        answer = self._generate_answer(question, intent, combined_reports[:max_reports])

        result = QueryResult(
            query=question,
            intent=intent,
            reports=combined_reports[:max_reports],
            answer=answer,
            sources_used=['postgresql', 'vector'],
            execution_time_ms=round((time.time() - start_time) * 1000, 1),
            cache_hit="MISS"
        )

        # Add to cache
        if use_cache:
            self._add_to_cache(cache_key, result, query_embedding)

        return result

    def _generate_answer(self, query: str, intent: str, reports: List[Dict]) -> str:
        """Generate a summary answer from reports."""
        if not reports:
            return "관련 리포트를 찾을 수 없습니다."

        issuers = list(set(r.get('issuer', 'N/A') for r in reports if r.get('issuer')))
        latest_date = max((r.get('issue_date', '') for r in reports), default='N/A')

        answer_parts = [
            f"분석 기관: {', '.join(issuers[:5])}",
            f"최근 리포트: {str(latest_date)[:10]}",
            f"리포트 {len(reports)}건 발견",
            "",
            "[관련 내용]"
        ]

        for i, r in enumerate(reports[:5], 1):
            title = r.get('title', 'N/A')
            summary = r.get('summary', '')[:200] if r.get('summary') else ''
            answer_parts.append(f"{i}. {r.get('issuer', 'N/A')}: {title}")
            if summary:
                answer_parts.append(f"   {summary}...")

        return "\n".join(answer_parts)

    def get_cache_metrics(self) -> Dict:
        """Get cache performance metrics."""
        total = self.metrics['total_queries']
        hits = self.metrics['exact_hits'] + self.metrics['semantic_hits']

        return {
            'metrics': self.metrics,
            'hit_rate_pct': round(hits / total * 100, 1) if total > 0 else 0,
            'cache_size': len(self.exact_cache),
        }

    def clear_cache(self):
        """Clear all caches."""
        self.exact_cache.clear()
        self.cache_embeddings.clear()


if __name__ == "__main__":
    eq = EconQuery()
    result = eq.query("금리 인상 전망")
    print(f"Intent: {result.intent}")
    print(f"Reports: {len(result.reports)}")
    print(f"Time: {result.execution_time_ms}ms")
    print(f"\nAnswer:\n{result.answer}")
