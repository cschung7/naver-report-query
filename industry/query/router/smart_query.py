"""
IndustryQuery - Intelligent Query Router for Industry Analysis

Routes queries to appropriate data sources based on intent detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge, Info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from .pg_client import PostgresClient
from .neo4j_client import Neo4jClient
from .vector_client import VectorClient

# Prometheus metrics
QUERY_COUNTER = Counter(
    'industryquery_queries_total',
    'Total number of queries processed',
    ['cache_type', 'intent']
)

QUERY_LATENCY = Histogram(
    'industryquery_query_duration_seconds',
    'Query processing time in seconds',
    ['cache_type'],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

CACHE_SIZE = Gauge(
    'industryquery_cache_entries',
    'Number of entries in cache',
    ['cache_type']
)

CACHE_HIT_RATE = Gauge(
    'industryquery_cache_hit_rate',
    'Cache hit rate percentage'
)

REPORTS_RETURNED = Histogram(
    'industryquery_reports_returned',
    'Number of reports returned per query',
    buckets=[0, 1, 5, 10, 20, 50, 100]
)

DB_STATUS = Gauge(
    'industryquery_database_status',
    'Database connection status (1=up, 0=down)',
    ['database']
)


class QueryIntent(Enum):
    """Query intent types for Industry Analysis."""
    INDUSTRY_OUTLOOK = "industry_outlook"
    CYCLE_ANALYSIS = "cycle_analysis"
    DEMAND_SUPPLY = "demand_supply"
    COMPETITIVE_LANDSCAPE = "competitive_landscape"
    REGULATORY_IMPACT = "regulatory_impact"
    TECHNOLOGY_TREND = "technology_trend"
    ISSUER_RESEARCH = "issuer_research"
    GENERAL_SEARCH = "general_search"


@dataclass
class QueryResult:
    """Result of an IndustryQuery."""
    query: str
    intent: str
    answer: str = ""
    reports: List[Dict] = field(default_factory=list)
    industries: List[Dict] = field(default_factory=list)
    cycles: List[Dict] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    # Graph-enriched fields
    related_industries: List[Dict] = field(default_factory=list)
    cycle_distribution: List[Dict] = field(default_factory=list)
    graph_insights: Dict = field(default_factory=dict)
    # Semantic search fields
    semantic_matches: List[Dict] = field(default_factory=list)


class QueryAnalyzer:
    """Analyzes queries to extract intent and entities."""

    # Intent patterns
    INTENT_PATTERNS = {
        QueryIntent.INDUSTRY_OUTLOOK: [
            r'산업\s*전망', r'업종\s*전망', r'업계\s*전망',
            r'outlook', r'view', r'방향',
            r'호황|불황|상승|하락|침체',
        ],
        QueryIntent.CYCLE_ANALYSIS: [
            r'사이클|주기|업황', r'cycle',
            r'턴어라운드|회복|정점|저점',
            r'upcycle|downcycle',
        ],
        QueryIntent.DEMAND_SUPPLY: [
            r'수요|공급|수급',
            r'demand|supply',
            r'과잉|부족|균형',
        ],
        QueryIntent.COMPETITIVE_LANDSCAPE: [
            r'경쟁|점유율|시장\s*구조',
            r'competition|market\s*share',
            r'1위|선두|후발',
        ],
        QueryIntent.REGULATORY_IMPACT: [
            r'규제|정책|법률|법안',
            r'regulation|policy',
            r'정부|지원|제재',
        ],
        QueryIntent.TECHNOLOGY_TREND: [
            r'기술|혁신|R&D',
            r'technology|innovation',
            r'신기술|차세대',
        ],
        QueryIntent.ISSUER_RESEARCH: [
            r'(키움|하나|미래에셋|신한|대신|한화|SK|삼성|IBK).*?(증권|투자)',
            r'애널리스트|analyst', r'리포트|보고서',
        ],
    }

    # Industry mappings
    INDUSTRY_MAP = {
        '반도체': '반도체', '메모리': '반도체', 'HBM': '반도체',
        '자동차': '자동차', 'EV': '자동차', '전기차': '자동차',
        '철강': '철강금속', '금속': '철강금속',
        '화학': '석유화학', '석화': '석유화학',
        '건설': '건설', '부동산': '건설',
        '은행': '은행', '금융': '금융',
        '제약': '제약', '바이오': '바이오',
        '통신': '통신', '5G': '통신',
        '게임': '게임', '엔터': '미디어',
        '유틸리티': '유틸리티', '전력': '유틸리티',
    }

    # Cycle stage mappings
    CYCLE_MAP = {
        '상승': 'UPCYCLE', '호황': 'UPCYCLE', '성장': 'UPCYCLE',
        '하락': 'DOWNCYCLE', '불황': 'DOWNCYCLE', '침체': 'DOWNCYCLE',
        '정점': 'PEAK', '피크': 'PEAK',
        '저점': 'TROUGH', '바닥': 'TROUGH',
        '회복': 'RECOVERY', '턴어라운드': 'RECOVERY',
    }

    # Issuer mappings
    ISSUER_MAP = {
        '키움': '키움증권',
        '하나': '하나증권',
        '미래에셋': '미래에셋증권',
        '신한': '신한투자증권',
        '대신': '대신증권',
        '한화': '한화투자증권',
        'SK': 'SK증권',
        'IBK': 'IBK투자증권',
        '유진': '유진투자증권',
        '유안타': '유안타증권',
    }

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract intent and entities."""
        result = {
            'intent': QueryIntent.GENERAL_SEARCH,
            'industry': None,
            'cycle_stage': None,
            'issuer': None,
            'keywords': [],
        }

        query_lower = query.lower()

        # Detect intent
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    result['intent'] = intent
                    break
            if result['intent'] != QueryIntent.GENERAL_SEARCH:
                break

        # Extract industry
        for keyword, industry in self.INDUSTRY_MAP.items():
            if keyword in query:
                result['industry'] = industry
                break

        # Extract cycle stage
        for keyword, cycle in self.CYCLE_MAP.items():
            if keyword in query:
                result['cycle_stage'] = cycle
                break

        # Extract issuer
        for keyword, issuer in self.ISSUER_MAP.items():
            if keyword in query:
                result['issuer'] = issuer
                break

        # Extract keywords for search
        keywords = re.findall(r'[\w가-힣]+', query)
        stopwords = {'의', '을', '를', '이', '가', '은', '는', '에', '로', '와', '과', '어떻게', '뭐', '무엇'}
        result['keywords'] = [k for k in keywords if k not in stopwords and len(k) > 1]

        return result


class IndustryQuery:
    """
    Intelligent query router for Industry Analysis reports.

    Routes queries based on detected intent and returns relevant results.
    Uses PostgreSQL (structured), Neo4j (graph), and ChromaDB (semantic).
    """

    def __init__(self, cache_ttl_seconds: int = 300, max_cache_size: int = 100,
                 semantic_cache_threshold: float = 0.85):
        self.analyzer = QueryAnalyzer()
        self._pg_client = None
        self._neo4j_client = None
        self._vector_client = None
        # Exact match cache: {cache_key: (result, timestamp)}
        self._cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._max_cache_size = max_cache_size
        # Semantic cache: [(embedding, result, timestamp, original_query)]
        self._semantic_cache: List[Tuple[List[float], QueryResult, datetime, str]] = []
        self._semantic_threshold = semantic_cache_threshold
        # Cache metrics
        self._metrics = {
            'total_queries': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'started_at': datetime.now()
        }
        self._recent_queries: List[Dict] = []
        self._max_recent = 50

    @property
    def pg_client(self) -> PostgresClient:
        if self._pg_client is None:
            self._pg_client = PostgresClient()
        return self._pg_client

    @property
    def neo4j_client(self) -> Neo4jClient:
        if self._neo4j_client is None:
            try:
                self._neo4j_client = Neo4jClient()
            except Exception:
                self._neo4j_client = None
        return self._neo4j_client

    @property
    def vector_client(self) -> VectorClient:
        if self._vector_client is None:
            try:
                self._vector_client = VectorClient()
            except Exception:
                self._vector_client = None
        return self._vector_client

    def _cache_key(self, question: str, max_reports: int) -> str:
        """Generate cache key from query parameters."""
        raw = f"{question.strip().lower()}:{max_reports}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[QueryResult]:
        """Get cached result if valid."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return result
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, result: QueryResult):
        """Cache a result with LRU eviction."""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (result, datetime.now())

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _get_semantic_cached(self, query_embedding: List[float]) -> Optional[Tuple[QueryResult, float, str]]:
        """Find semantically similar cached result."""
        now = datetime.now()
        best_match = None
        best_score = 0.0
        best_query = ""

        # Clean expired entries
        self._semantic_cache = [
            entry for entry in self._semantic_cache
            if now - entry[2] < self._cache_ttl
        ]

        for emb, result, timestamp, orig_query in self._semantic_cache:
            similarity = self._cosine_similarity(query_embedding, emb)
            if similarity >= self._semantic_threshold and similarity > best_score:
                best_score = similarity
                best_match = result
                best_query = orig_query

        if best_match:
            return (best_match, best_score, best_query)
        return None

    def _set_semantic_cache(self, embedding: List[float], result: QueryResult, query: str):
        """Add to semantic cache with LRU eviction."""
        if len(self._semantic_cache) >= self._max_cache_size:
            self._semantic_cache.sort(key=lambda x: x[2])
            self._semantic_cache.pop(0)
        self._semantic_cache.append((embedding, result, datetime.now(), query))

    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._semantic_cache.clear()

    def reset_metrics(self):
        """Reset cache metrics."""
        self._metrics = {
            'total_queries': 0,
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'started_at': datetime.now()
        }
        self._recent_queries.clear()

    def _log_query(self, query: str, cache_type: str, exec_time_ms: float, similarity: float = None):
        """Log a query for metrics and recent history."""
        self._metrics['total_queries'] += 1

        if cache_type == 'exact':
            self._metrics['exact_hits'] += 1
        elif cache_type == 'semantic':
            self._metrics['semantic_hits'] += 1
        else:
            self._metrics['misses'] += 1

        # Log to recent queries
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:50] + '...' if len(query) > 50 else query,
            'cache_type': cache_type,
            'exec_time_ms': round(exec_time_ms, 2)
        }
        if similarity:
            entry['similarity'] = round(similarity, 4)

        self._recent_queries.append(entry)
        if len(self._recent_queries) > self._max_recent:
            self._recent_queries.pop(0)

        # Log to console
        if cache_type == 'exact':
            logger.info(f"[CACHE] EXACT HIT | {exec_time_ms:.1f}ms | {query[:40]}")
        elif cache_type == 'semantic':
            logger.info(f"[CACHE] SEMANTIC HIT ({similarity:.0%}) | {exec_time_ms:.1f}ms | {query[:40]}")
        else:
            logger.info(f"[CACHE] MISS | {exec_time_ms:.1f}ms | {query[:40]}")

        # Record Prometheus metrics
        QUERY_LATENCY.labels(cache_type=cache_type).observe(exec_time_ms / 1000.0)
        self._update_prometheus_gauges()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        now = datetime.now()
        valid_exact = sum(1 for _, ts in self._cache.values() if now - ts < self._cache_ttl)
        valid_semantic = sum(1 for _, _, ts, _ in self._semantic_cache if now - ts < self._cache_ttl)
        return {
            'exact_entries': len(self._cache),
            'semantic_entries': len(self._semantic_cache),
            'valid_exact': valid_exact,
            'valid_semantic': valid_semantic,
            'max_size': self._max_cache_size,
            'ttl_seconds': self._cache_ttl.total_seconds(),
            'semantic_threshold': self._semantic_threshold
        }

    def get_cache_metrics(self) -> Dict:
        """Get cache hit/miss metrics."""
        total = self._metrics['total_queries']
        exact = self._metrics['exact_hits']
        semantic = self._metrics['semantic_hits']
        misses = self._metrics['misses']

        hit_rate = ((exact + semantic) / total * 100) if total > 0 else 0
        exact_rate = (exact / total * 100) if total > 0 else 0
        semantic_rate = (semantic / total * 100) if total > 0 else 0

        uptime = (datetime.now() - self._metrics['started_at']).total_seconds()

        return {
            'total_queries': total,
            'exact_hits': exact,
            'semantic_hits': semantic,
            'misses': misses,
            'hit_rate_pct': round(hit_rate, 1),
            'exact_hit_rate_pct': round(exact_rate, 1),
            'semantic_hit_rate_pct': round(semantic_rate, 1),
            'uptime_seconds': round(uptime, 0),
            'queries_per_minute': round(total / (uptime / 60), 2) if uptime > 0 else 0
        }

    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent query history."""
        return self._recent_queries[-limit:]

    def _update_prometheus_gauges(self):
        """Update Prometheus gauge metrics."""
        # Cache sizes
        CACHE_SIZE.labels(cache_type='exact').set(len(self._cache))
        CACHE_SIZE.labels(cache_type='semantic').set(len(self._semantic_cache))

        # Hit rate
        total = self._metrics['total_queries']
        if total > 0:
            hits = self._metrics['exact_hits'] + self._metrics['semantic_hits']
            CACHE_HIT_RATE.set((hits / total) * 100)

        # Database status
        DB_STATUS.labels(database='postgresql').set(1 if self._pg_client else 0)
        DB_STATUS.labels(database='neo4j').set(1 if self._neo4j_client else 0)
        DB_STATUS.labels(database='vector').set(1 if self._vector_client else 0)

    def query(
        self,
        question: str,
        max_reports: int = 20,
        verbose: bool = False,
        date_from: str = None,
        date_to: str = None
    ) -> QueryResult:
        """
        Execute intelligent query with multi-stage search strategy.

        Args:
            question: User's query
            max_reports: Maximum reports to return
            verbose: Print debug info

        Returns:
            QueryResult with relevant reports and answer
        """
        start_time = datetime.now()

        # Check exact cache first
        cache_key = self._cache_key(f"{question}|{date_from}|{date_to}", max_reports)
        cached_result = self._get_cached(cache_key)
        if cached_result is not None:
            cached_result.execution_time_ms = 0.1
            self._log_query(question, 'exact', 0.1)
            QUERY_COUNTER.labels(cache_type='exact', intent=cached_result.intent).inc()
            REPORTS_RETURNED.observe(len(cached_result.reports))
            if verbose:
                print("[Cache] EXACT HIT - returning cached result")
            return cached_result

        # Check semantic cache if vector client available (skip when date filters applied)
        query_embedding = None
        if self.vector_client:
            query_embedding = self.vector_client.embed_text(question)
        if query_embedding is not None and not date_from and not date_to:
            semantic_hit = self._get_semantic_cached(query_embedding)
            if semantic_hit:
                result, score, orig_query = semantic_hit
                result.execution_time_ms = 0.2
                self._log_query(question, 'semantic', 0.2, score)
                QUERY_COUNTER.labels(cache_type='semantic', intent=result.intent).inc()
                REPORTS_RETURNED.observe(len(result.reports))
                if verbose:
                    print(f"[Cache] SEMANTIC HIT ({score:.2%}) - similar to: {orig_query[:30]}...")
                return result

        # Analyze query
        analysis = self.analyzer.analyze(question)
        intent = analysis['intent']
        keywords = analysis['keywords']

        if verbose:
            print(f"[Intent] {intent.value}")
            print(f"[Analysis] {analysis}")

        result = QueryResult(
            query=question,
            intent=intent.value
        )

        reports = []

        # Multi-stage search strategy based on intent
        try:
            # Stage 1: Intent-specific search
            if intent == QueryIntent.CYCLE_ANALYSIS and analysis['cycle_stage']:
                reports = self.pg_client.search_by_cycle(analysis['cycle_stage'], limit=max_reports)
                if verbose:
                    print(f"[Cycle Search] Found {len(reports)} reports")

            elif intent == QueryIntent.INDUSTRY_OUTLOOK and analysis['industry']:
                reports = self.pg_client.search_by_industry([analysis['industry']], limit=max_reports)
                if verbose:
                    print(f"[Industry Search] Found {len(reports)} reports")

            # Stage 2: Broad text search if stage 1 didn't find enough
            if len(reports) < max_reports:
                search_params = {
                    'keywords': keywords,
                    'limit': max_reports - len(reports),
                    'broad_search': True
                }

                if analysis['industry']:
                    search_params['industry'] = analysis['industry']

                if analysis['issuer']:
                    search_params['issuer'] = analysis['issuer']

                if analysis['cycle_stage']:
                    search_params['cycle_stage'] = analysis['cycle_stage']

                if date_from:
                    search_params['date_from'] = date_from
                if date_to:
                    search_params['date_to'] = date_to

                additional = self.pg_client.search_reports(**search_params)

                # Merge results avoiding duplicates
                existing_ids = {r['report_id'] for r in reports}
                for r in additional:
                    if r['report_id'] not in existing_ids:
                        reports.append(r)
                        existing_ids.add(r['report_id'])

                if verbose:
                    print(f"[Broad Search] Added {len(additional)} reports, total: {len(reports)}")

            # Stage 3: Fallback
            if not reports and keywords:
                main_keyword = max(keywords, key=len) if keywords else None
                if main_keyword:
                    fallback_params = {
                        'query': main_keyword,
                        'limit': max_reports,
                        'broad_search': True
                    }
                    if date_from:
                        fallback_params['date_from'] = date_from
                    if date_to:
                        fallback_params['date_to'] = date_to
                    reports = self.pg_client.search_reports(**fallback_params)
                    if verbose:
                        print(f"[Fallback Search] Found {len(reports)} reports with '{main_keyword}'")

            # Stage 4: Last resort
            if not reports:
                reports = self.pg_client.get_recent_reports(days=90, limit=max_reports)
                result.sources_used.append('recent_fallback')
                if verbose:
                    print(f"[Recent Fallback] Showing {len(reports)} recent reports")

            result.reports = reports[:max_reports]
            result.sources_used.append('postgresql')

        except Exception as e:
            if verbose:
                print(f"[PostgreSQL] Error: {e}")
            import traceback
            traceback.print_exc()

        # Neo4j Graph Integration
        try:
            if self.neo4j_client:
                main_keyword = max(keywords, key=len) if keywords else None

                if main_keyword:
                    graph_search = self.neo4j_client.search_graph(main_keyword, limit=max_reports)

                    # Get related industries
                    if graph_search.get('related_industries'):
                        result.related_industries = graph_search['related_industries']

                    # Merge graph reports
                    graph_reports = []
                    for source in ['by_industry', 'by_cycle', 'by_issuer']:
                        graph_reports.extend(graph_search.get(source, []))

                    existing_ids = {r['report_id'] for r in result.reports}
                    for gr in graph_reports:
                        if gr.get('report_id') and gr['report_id'] not in existing_ids:
                            merged_report = {
                                'report_id': gr['report_id'],
                                'title': gr.get('title', ''),
                                'issuer': gr.get('issuer', ''),
                                'issue_date': gr.get('issue_date', ''),
                                'industry': gr.get('industry'),
                                'cycle_stage': gr.get('cycle_stage'),
                                'demand_trend': gr.get('demand_trend'),
                            }
                            result.reports.append(merged_report)
                            existing_ids.add(gr['report_id'])

                result.reports = result.reports[:max_reports]

                # Sort by issue_date DESC
                result.reports = self._sort_reports_by_date(result.reports)

                # Get cycle distribution
                result.cycle_distribution = self.neo4j_client.get_cycle_distribution()

                result.sources_used.append('neo4j')

        except Exception as e:
            if verbose:
                print(f"[Neo4j] Error: {e}")

        # Vector/Semantic Search Integration
        try:
            if self.vector_client:
                semantic_results = self.vector_client.search_with_scores(
                    query=question,
                    n_results=max_reports,
                    score_threshold=0.4
                )

                if verbose:
                    print(f"[Vector Search] Found {len(semantic_results)} semantic matches")

                result.semantic_matches = [
                    {
                        'report_id': r['id'],
                        'similarity_score': r['similarity_score'],
                        'title': r['metadata'].get('title', ''),
                        'issuer': r['metadata'].get('issuer', ''),
                        'issue_date': r['metadata'].get('issue_date', ''),
                    }
                    for r in semantic_results
                ]

                # Merge semantic results
                existing_ids = {r['report_id'] for r in result.reports}
                for r in semantic_results:
                    if r['id'] not in existing_ids and len(result.reports) < max_reports:
                        result.reports.append({
                            'report_id': r['id'],
                            'title': r['metadata'].get('title', ''),
                            'issuer': r['metadata'].get('issuer', ''),
                            'issue_date': r['metadata'].get('issue_date', ''),
                            'industry': r['metadata'].get('industry'),
                            'semantic_score': r['similarity_score'],
                        })
                        existing_ids.add(r['id'])

                result.reports = result.reports[:max_reports]
                result.sources_used.append('vector')

        except Exception as e:
            if verbose:
                print(f"[Vector Search] Error: {e}")

        # Final sort by date DESC (recent first)
        result.reports = self._sort_reports_by_date(result.reports)

        # Synthesize answer
        self._synthesize_answer(result, analysis)

        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log cache miss
        self._log_query(question, 'miss', result.execution_time_ms)
        QUERY_COUNTER.labels(cache_type='miss', intent=result.intent).inc()
        REPORTS_RETURNED.observe(len(result.reports))

        # Cache the result
        self._set_cache(cache_key, result)

        # Semantic cache (skip when date filters to avoid pollution)
        if not date_from and not date_to:
            if query_embedding is not None:
                self._set_semantic_cache(query_embedding, result, question)
            elif self.vector_client:
                emb = self.vector_client.embed_text(question)
                if emb is not None:
                    self._set_semantic_cache(emb, result, question)

        return result

    def _sort_reports_by_date(self, reports: List[Dict]) -> List[Dict]:
        """Sort reports by issue_date descending (most recent first)."""
        def get_date_key(r):
            date = r.get('issue_date')
            if date is None:
                return ''
            # Handle datetime objects
            if hasattr(date, 'strftime'):
                return date.strftime('%Y-%m-%d')
            # Handle strings
            return str(date)[:10]

        return sorted(reports, key=get_date_key, reverse=True)

    def _synthesize_answer(self, result: QueryResult, analysis: Dict):
        """Synthesize answer from results."""
        if not result.reports:
            result.answer = "관련 리포트를 찾지 못했습니다."
            return

        # Extract key information
        issuers = list(set(r['issuer'] for r in result.reports if r.get('issuer')))
        industries = list(set(r.get('industry') for r in result.reports if r.get('industry')))
        cycles = list(set(r.get('cycle_stage') for r in result.reports if r.get('cycle_stage')))

        latest = result.reports[0] if result.reports else None

        lines = []

        # Header
        lines.append(f"분석 기관: {', '.join(issuers[:5])}")
        if latest:
            lines.append(f"최근 리포트: {latest.get('issue_date', 'N/A')}")
        lines.append(f"리포트 {len(result.reports)}건 발견")

        # Industry summary
        if industries:
            lines.append(f"\n산업: {', '.join(industries[:5])}")

        # Cycle summary
        if cycles:
            cycle_kr = {
                'UPCYCLE': '상승 사이클',
                'DOWNCYCLE': '하락 사이클',
                'PEAK': '정점',
                'TROUGH': '저점',
                'RECOVERY': '회복',
            }
            cycle_str = ', '.join(cycle_kr.get(c, c) for c in cycles[:3])
            lines.append(f"사이클: {cycle_str}")

        # Key content
        lines.append("\n[관련 내용]")
        for i, report in enumerate(result.reports[:5], 1):
            title = report.get('title', '')
            issuer = report.get('issuer', 'Unknown')
            summary = report.get('summary', '')[:200] if report.get('summary') else ''
            lines.append(f"{i}. {issuer}: {title}")
            if summary:
                lines.append(f"   {summary}...")

            if report.get('industry'):
                lines.append(f"   [산업] {report['industry']}")
            if report.get('cycle_stage'):
                lines.append(f"   [사이클] {report['cycle_stage']}")
            if report.get('demand_trend'):
                lines.append(f"   [수요] {report['demand_trend']}")

        # Related industries
        if result.related_industries:
            lines.append("\n[관련 산업]")
            for ind in result.related_industries[:5]:
                lines.append(f"  - {ind.get('related_industry', '')}: {ind.get('co_occurrences', 0)}건")

        # Cycle distribution
        if result.cycle_distribution:
            lines.append("\n[사이클 분포]")
            for cycle in result.cycle_distribution[:5]:
                lines.append(f"  - {cycle.get('cycle_stage', '')}: {cycle.get('count', 0)}건")

        # Semantic matches
        if result.semantic_matches:
            lines.append("\n[의미 검색 결과]")
            for match in result.semantic_matches[:3]:
                score_pct = int(match['similarity_score'] * 100)
                lines.append(f"  - [{score_pct}%] {match.get('issuer', '')}: {match.get('title', '')[:40]}")

        result.answer = '\n'.join(lines)


if __name__ == "__main__":
    iq = IndustryQuery()

    test_queries = [
        "반도체 산업 전망",
        "자동차 사이클 회복",
        "철강 수급 전망",
        "하나증권 최근 리포트",
    ]

    print("=" * 60)
    print("IndustryQuery Test")
    print("=" * 60)

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = iq.query(q, max_reports=3, verbose=True)
        print(f"Intent: {result.intent}")
        print(f"Time: {result.execution_time_ms:.0f}ms")
        print(f"Reports: {len(result.reports)}")
        print("-" * 40)
