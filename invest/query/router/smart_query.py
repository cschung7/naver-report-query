"""
SmartQuery - Intelligent Query Router for Investment Strategy

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
    'smartquery_queries_total',
    'Total number of queries processed',
    ['cache_type', 'intent']
)

QUERY_LATENCY = Histogram(
    'smartquery_query_duration_seconds',
    'Query processing time in seconds',
    ['cache_type'],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

CACHE_SIZE = Gauge(
    'smartquery_cache_entries',
    'Number of entries in cache',
    ['cache_type']
)

CACHE_HIT_RATE = Gauge(
    'smartquery_cache_hit_rate',
    'Cache hit rate percentage'
)

REPORTS_RETURNED = Histogram(
    'smartquery_reports_returned',
    'Number of reports returned per query',
    buckets=[0, 1, 5, 10, 20, 50, 100]
)

DB_STATUS = Gauge(
    'smartquery_database_status',
    'Database connection status (1=up, 0=down)',
    ['database']
)


class QueryIntent(Enum):
    """Query intent types for Investment Strategy."""
    MARKET_OUTLOOK = "market_outlook"
    ASSET_ALLOCATION = "asset_allocation"
    SECTOR_STRATEGY = "sector_strategy"
    THEME_PLAY = "theme_play"
    MACRO_ANALYSIS = "macro_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    ISSUER_RESEARCH = "issuer_research"
    GENERAL_SEARCH = "general_search"


@dataclass
class QueryResult:
    """Result of a SmartQuery."""
    query: str
    intent: str
    answer: str = ""
    reports: List[Dict] = field(default_factory=list)
    themes: List[Dict] = field(default_factory=list)
    allocations: List[Dict] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0
    # Graph-enriched fields
    related_themes: List[Dict] = field(default_factory=list)
    related_sectors: List[Dict] = field(default_factory=list)
    graph_insights: Dict = field(default_factory=dict)
    # Semantic search fields
    semantic_matches: List[Dict] = field(default_factory=list)


class QueryAnalyzer:
    """Analyzes queries to extract intent and entities."""

    # Intent patterns
    INTENT_PATTERNS = {
        QueryIntent.MARKET_OUTLOOK: [
            r'시장\s*전망', r'증시\s*전망', r'주식\s*전망',
            r'outlook', r'view', r'방향',
            r'강세|약세|상승|하락|조정',
        ],
        QueryIntent.ASSET_ALLOCATION: [
            r'자산\s*배분', r'포트폴리오', r'비중',
            r'allocation', r'overweight|underweight',
            r'주식\s*비중|채권\s*비중|현금\s*비중',
        ],
        QueryIntent.SECTOR_STRATEGY: [
            r'섹터|업종', r'반도체|자동차|바이오|금융|IT|에너지',
            r'sector\s*rotation', r'업종\s*전략',
        ],
        QueryIntent.THEME_PLAY: [
            r'테마|theme', r'AI|인공지능|전기차|2차전지|로봇',
            r'수혜|관련주', r'트렌드',
        ],
        QueryIntent.MACRO_ANALYSIS: [
            r'금리|환율|물가|인플레', r'GDP|경기|경제',
            r'연준|Fed|기준금리', r'달러|원화|엔화',
            r'macro|매크로',
        ],
        QueryIntent.RISK_ASSESSMENT: [
            r'리스크|위험|risk', r'변동성|volatility',
            r'헷지|hedge|방어',
        ],
        QueryIntent.ISSUER_RESEARCH: [
            r'(키움|하나|미래에셋|신한|대신|한화|SK|삼성|IBK).*?(증권|투자)',
            r'애널리스트|analyst', r'리포트|보고서',
        ],
    }

    # Geography mappings
    GEOGRAPHY_MAP = {
        '미국': 'US', '미증시': 'US', '나스닥': 'US', 'S&P': 'US',
        '중국': 'CHINA', '홍콩': 'CHINA', '상해': 'CHINA',
        '일본': 'JAPAN', '니케이': 'JAPAN',
        '유럽': 'EUROPE', '독일': 'EUROPE',
        '한국': 'KOREA', '코스피': 'KOREA', '코스닥': 'KOREA',
        '글로벌': 'GLOBAL', '세계': 'GLOBAL',
        '신흥국': 'EMERGING_MARKETS', '이머징': 'EMERGING_MARKETS',
    }

    # Market outlook mappings
    OUTLOOK_MAP = {
        '강세': 'BULLISH', '상승': 'BULLISH', '긍정': 'BULLISH',
        '약세': 'BEARISH', '하락': 'BEARISH', '부정': 'BEARISH',
        '중립': 'NEUTRAL', '보합': 'NEUTRAL',
        '회복': 'RECOVERY', '반등': 'RECOVERY',
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
        '삼성': '삼성증권',
        '유안타': '유안타증권',
    }

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract intent and entities."""
        result = {
            'intent': QueryIntent.GENERAL_SEARCH,
            'geography': None,
            'outlook': None,
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

        # Extract geography
        for keyword, geo in self.GEOGRAPHY_MAP.items():
            if keyword in query:
                result['geography'] = geo
                break

        # Extract outlook
        for keyword, outlook in self.OUTLOOK_MAP.items():
            if keyword in query:
                result['outlook'] = outlook
                break

        # Extract issuer
        for keyword, issuer in self.ISSUER_MAP.items():
            if keyword in query:
                result['issuer'] = issuer
                break

        # Extract keywords for search
        # Remove common words and keep meaningful terms
        keywords = re.findall(r'[\w가-힣]+', query)
        stopwords = {'의', '을', '를', '이', '가', '은', '는', '에', '로', '와', '과', '어떻게', '뭐', '무엇'}
        result['keywords'] = [k for k in keywords if k not in stopwords and len(k) > 1]

        return result


class SmartQuery:
    """
    Intelligent query router for Investment Strategy reports.

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
        self._recent_queries: List[Dict] = []  # Last N queries for logging
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
        # Evict oldest if at capacity
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
        # Evict oldest if at capacity
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
            cached_result.execution_time_ms = 0.1  # Exact cache hit
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
            semantic_hit = self._get_semantic_cached(query_embedding) if not date_from and not date_to else None
            if semantic_hit:
                result, score, orig_query = semantic_hit
                result.execution_time_ms = 0.2  # Semantic cache hit
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
            if intent == QueryIntent.THEME_PLAY:
                # Search themes table first
                theme_keywords = self._get_theme_keywords(keywords)
                if theme_keywords:
                    reports = self.pg_client.search_by_theme(theme_keywords, limit=max_reports)
                    if verbose:
                        print(f"[Theme Search] Found {len(reports)} reports")

            elif intent == QueryIntent.SECTOR_STRATEGY:
                # Search sector recommendations table
                sector_keywords = self._get_sector_keywords(keywords)
                if sector_keywords:
                    reports = self.pg_client.search_by_sector(sector_keywords, limit=max_reports)
                    if verbose:
                        print(f"[Sector Search] Found {len(reports)} reports")

            elif intent == QueryIntent.ASSET_ALLOCATION:
                # Search allocations table
                asset_keywords = self._get_asset_keywords(keywords)
                if asset_keywords:
                    reports = self.pg_client.search_by_allocation(asset_keywords, limit=max_reports)
                    if verbose:
                        print(f"[Allocation Search] Found {len(reports)} reports")

            # Stage 2: Broad text search if stage 1 didn't find enough
            if len(reports) < max_reports:
                search_params = {
                    'keywords': keywords,
                    'limit': max_reports - len(reports),
                    'broad_search': True
                }

                # Add soft filters based on analysis
                if analysis['geography']:
                    search_params['geography'] = analysis['geography']

                if analysis['issuer']:
                    search_params['issuer'] = analysis['issuer']

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

            # Stage 3: Fallback - relax criteria if still no results
            if not reports and keywords:
                # Try with just the most important keyword
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

            # Stage 4: Last resort - get recent reports if nothing found
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

        # ============================================
        # Neo4j Graph Integration
        # ============================================
        try:
            if self.neo4j_client:
                graph_reports = []
                main_keyword = max(keywords, key=len) if keywords else None

                # Stage 5: Graph-based search for additional results
                if main_keyword:
                    graph_search = self.neo4j_client.search_graph(main_keyword, limit=max_reports)

                    # Collect reports from graph search
                    for source in ['by_theme', 'by_sector', 'by_geography']:
                        graph_reports.extend(graph_search.get(source, []))

                    if verbose:
                        print(f"[Neo4j Search] Found {len(graph_reports)} graph results for '{main_keyword}'")

                # Merge graph reports with existing results (avoid duplicates)
                existing_ids = {r['report_id'] for r in result.reports}
                for gr in graph_reports:
                    if gr.get('report_id') and gr['report_id'] not in existing_ids:
                        # Convert graph result to match PostgreSQL format
                        merged_report = {
                            'report_id': gr['report_id'],
                            'title': gr.get('title', ''),
                            'issuer': gr.get('issuer', ''),
                            'issue_date': gr.get('issue_date', ''),
                            'market_outlook': gr.get('outlook'),
                            'geography': gr.get('geography'),
                            # Graph-specific fields
                            'theme': gr.get('theme'),
                            'sector': gr.get('sector'),
                            'conviction': gr.get('conviction'),
                            'recommendation': gr.get('recommendation'),
                        }
                        result.reports.append(merged_report)
                        existing_ids.add(gr['report_id'])

                # Trim to max_reports
                result.reports = result.reports[:max_reports]

                # Stage 6: Get related themes from graph
                if intent == QueryIntent.THEME_PLAY and main_keyword:
                    related_themes = self.neo4j_client.find_related_themes(main_keyword, limit=10)
                    result.related_themes = related_themes
                    if verbose:
                        print(f"[Neo4j] Found {len(related_themes)} related themes")

                # Stage 7: Get related sectors from graph
                if intent == QueryIntent.SECTOR_STRATEGY and main_keyword:
                    related_sectors = self.neo4j_client.find_related_sectors(main_keyword, limit=10)
                    result.related_sectors = related_sectors
                    if verbose:
                        print(f"[Neo4j] Found {len(related_sectors)} related sectors")

                # Stage 8: Get graph insights for any query
                if main_keyword:
                    result.graph_insights = self._get_graph_insights(main_keyword, intent, verbose)

                result.sources_used.append('neo4j')

        except Exception as e:
            if verbose:
                print(f"[Neo4j] Error: {e}")

        # ============================================
        # Vector/Semantic Search Integration
        # ============================================
        try:
            if self.vector_client:
                # Stage 9: Semantic search for similar reports
                semantic_results = self.vector_client.search_with_scores(
                    query=question,
                    n_results=max_reports,
                    score_threshold=0.4  # Minimum similarity threshold
                )

                if verbose:
                    print(f"[Vector Search] Found {len(semantic_results)} semantic matches")

                # Store semantic matches with scores
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

                # Merge semantic results into reports (avoid duplicates)
                existing_ids = {r['report_id'] for r in result.reports}
                semantic_report_ids = [r['id'] for r in semantic_results if r['id'] not in existing_ids]

                # Fetch full report details for semantic matches
                if semantic_report_ids:
                    for report_id in semantic_report_ids[:max_reports - len(result.reports)]:
                        # Find matching semantic result
                        semantic_match = next(
                            (r for r in semantic_results if r['id'] == report_id),
                            None
                        )
                        if semantic_match:
                            # Add as a minimal report (full details will be fetched if needed)
                            result.reports.append({
                                'report_id': report_id,
                                'title': semantic_match['metadata'].get('title', ''),
                                'issuer': semantic_match['metadata'].get('issuer', ''),
                                'issue_date': semantic_match['metadata'].get('issue_date', ''),
                                'market_outlook': semantic_match['metadata'].get('market_outlook'),
                                'geography': semantic_match['metadata'].get('geography'),
                                'semantic_score': semantic_match['similarity_score'],
                            })
                            existing_ids.add(report_id)

                    if verbose:
                        print(f"[Vector Search] Added {len(semantic_report_ids)} semantic results")

                # Trim to max_reports
                result.reports = result.reports[:max_reports]
                result.sources_used.append('vector')

        except Exception as e:
            if verbose:
                print(f"[Vector Search] Error: {e}")

        # Get themes if relevant
        if intent == QueryIntent.THEME_PLAY and keywords:
            try:
                themes = self.pg_client.get_themes(
                    theme_name=keywords[0],
                    limit=20
                )
                result.themes = themes
                if verbose:
                    print(f"[Themes] Found {len(themes)} themes")
            except Exception as e:
                if verbose:
                    print(f"[Themes] Error: {e}")

        # Synthesize answer
        self._synthesize_answer(result, analysis)

        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log cache miss and record Prometheus metrics
        self._log_query(question, 'miss', result.execution_time_ms)
        QUERY_COUNTER.labels(cache_type='miss', intent=result.intent).inc()
        REPORTS_RETURNED.observe(len(result.reports))

        # Cache the result (exact match)
        self._set_cache(cache_key, result)

        # Also cache for semantic lookup (skip when date filters to avoid pollution)
        if not date_from and not date_to:
            if query_embedding is not None:
                self._set_semantic_cache(query_embedding, result, question)
            elif self.vector_client:
                emb = self.vector_client.embed_text(question)
                self._set_semantic_cache(emb, result, question)

        return result

    def _get_theme_keywords(self, keywords: List[str]) -> List[str]:
        """Extract theme-related keywords."""
        theme_terms = {
            'AI', '인공지능', '전기차', 'EV', '2차전지', '배터리', '로봇', '자율주행',
            '메타버스', '반도체', 'HBM', 'ESG', '친환경', '신재생', '데이터센터',
            '클라우드', '사이버보안', '바이오', '헬스케어', '우주', '방산', '원자력',
        }
        result = []
        for kw in keywords:
            if kw.upper() in [t.upper() for t in theme_terms]:
                result.append(kw)
            elif len(kw) >= 2:
                result.append(kw)
        return result if result else keywords

    def _get_sector_keywords(self, keywords: List[str]) -> List[str]:
        """Extract sector-related keywords."""
        sector_terms = {
            '반도체', 'IT', '자동차', '바이오', '금융', '은행', '증권', '보험',
            '유틸리티', '전력', '에너지', '철강', '화학', '조선', '건설', '운송',
            '유통', '소비재', '제약', '통신', '미디어', '엔터', '게임', '인터넷',
        }
        result = []
        for kw in keywords:
            if kw in sector_terms:
                result.append(kw)
            elif len(kw) >= 2:
                result.append(kw)
        return result if result else keywords

    def _get_asset_keywords(self, keywords: List[str]) -> List[str]:
        """Extract asset allocation keywords."""
        asset_terms = {
            '주식', '채권', '현금', '원자재', '금', '원유', '달러', '부동산',
            'EQUITY', 'BOND', 'CASH', 'COMMODITY', 'REITS', 'equity', 'bond',
            '비중', '배분', '포트폴리오', 'overweight', 'underweight', '확대', '축소',
        }
        result = []
        for kw in keywords:
            if kw in asset_terms or kw.lower() in [t.lower() for t in asset_terms]:
                result.append(kw)
            elif len(kw) >= 2:
                result.append(kw)
        return result if result else keywords

    def _get_graph_insights(self, keyword: str, intent: QueryIntent, verbose: bool = False) -> Dict:
        """Get insights from the knowledge graph."""
        insights = {}

        try:
            if not self.neo4j_client:
                return insights

            # Get top themes related to the keyword
            if intent in [QueryIntent.THEME_PLAY, QueryIntent.GENERAL_SEARCH]:
                top_themes = self.neo4j_client.get_top_themes(limit=5)
                insights['top_themes'] = top_themes

            # Get top sectors related to the keyword
            if intent in [QueryIntent.SECTOR_STRATEGY, QueryIntent.GENERAL_SEARCH]:
                top_sectors = self.neo4j_client.get_top_sectors(limit=5)
                insights['top_sectors'] = top_sectors

            # Get outlook distribution if market outlook query
            if intent == QueryIntent.MARKET_OUTLOOK:
                # Get reports by different outlooks
                for outlook in ['BULL', 'BEAR', 'NEUTRAL']:
                    try:
                        outlook_reports = self.neo4j_client.get_reports_by_outlook(outlook, limit=3)
                        insights[f'outlook_{outlook.lower()}'] = len(outlook_reports)
                    except Exception:
                        pass

            if verbose and insights:
                print(f"[Graph Insights] {list(insights.keys())}")

        except Exception as e:
            if verbose:
                print(f"[Graph Insights] Error: {e}")

        return insights

    def _synthesize_answer(self, result: QueryResult, analysis: Dict):
        """Synthesize answer from results."""
        if not result.reports:
            result.answer = "관련 리포트를 찾지 못했습니다."
            return

        # Extract key information
        issuers = list(set(r['issuer'] for r in result.reports if r.get('issuer')))
        outlooks = list(set(r['market_outlook'] for r in result.reports if r.get('market_outlook')))
        regimes = list(set(r['market_regime'] for r in result.reports if r.get('market_regime')))

        latest = result.reports[0] if result.reports else None

        lines = []

        # Header
        lines.append(f"분석 기관: {', '.join(issuers[:5])}")
        if latest:
            lines.append(f"최근 리포트: {latest['issue_date']}")
        lines.append(f"리포트 {len(result.reports)}건 발견")

        # Market view summary
        if outlooks:
            outlook_kr = {
                'BULLISH': '강세',
                'BEARISH': '약세',
                'NEUTRAL': '중립',
                'RECOVERY': '회복',
            }
            outlook_str = ', '.join(outlook_kr.get(o, o) for o in outlooks[:3])
            lines.append(f"\n시장 전망: {outlook_str}")

        if regimes:
            regime_kr = {
                'RISK_ON': '위험선호',
                'RISK_OFF': '위험회피',
                'NEUTRAL': '중립',
            }
            regime_str = ', '.join(regime_kr.get(r, r) for r in regimes[:3])
            lines.append(f"시장 분위기: {regime_str}")

        # Key content from reports
        lines.append("\n[관련 내용]")
        for i, report in enumerate(result.reports[:5], 1):
            title = report.get('title', '')
            issuer = report.get('issuer', 'Unknown')
            summary = report.get('summary', '')[:200] if report.get('summary') else ''
            lines.append(f"{i}. {issuer}: {title}")
            if summary:
                lines.append(f"   {summary}...")

            # Add graph-enriched fields if present
            if report.get('theme'):
                lines.append(f"   [테마] {report['theme']}")
            if report.get('sector'):
                lines.append(f"   [섹터] {report['sector']}")
            if report.get('conviction'):
                lines.append(f"   [확신도] {report['conviction']}")

            # Add actionable items
            do_list = report.get('do_list', [])
            if do_list and isinstance(do_list, list) and len(do_list) > 0:
                lines.append(f"   [추천] {', '.join(str(d) for d in do_list[:3])}")

            avoid_list = report.get('avoid_list', [])
            if avoid_list and isinstance(avoid_list, list) and len(avoid_list) > 0:
                lines.append(f"   [회피] {', '.join(str(a) for a in avoid_list[:3])}")

        # Add graph insights
        if result.related_themes:
            lines.append("\n[관련 테마]")
            for theme in result.related_themes[:5]:
                lines.append(f"  • {theme.get('related_theme', theme.get('theme', ''))}: {theme.get('co_occurrences', theme.get('mentions', 0))}건")

        if result.related_sectors:
            lines.append("\n[관련 섹터]")
            for sector in result.related_sectors[:5]:
                lines.append(f"  • {sector.get('related_sector', sector.get('sector', ''))}: {sector.get('co_occurrences', sector.get('recommendations', 0))}건")

        if result.graph_insights:
            if result.graph_insights.get('top_themes'):
                lines.append("\n[인기 테마]")
                for theme in result.graph_insights['top_themes'][:3]:
                    lines.append(f"  • {theme['theme']}: {theme['mentions']}건")

            if result.graph_insights.get('top_sectors'):
                lines.append("\n[인기 섹터]")
                for sector in result.graph_insights['top_sectors'][:3]:
                    lines.append(f"  • {sector['sector']}: {sector['recommendations']}건")

            # Market outlook distribution
            outlook_counts = []
            for key in ['outlook_bull', 'outlook_bear', 'outlook_neutral']:
                if key in result.graph_insights:
                    outlook_name = key.replace('outlook_', '').upper()
                    outlook_counts.append(f"{outlook_name}: {result.graph_insights[key]}건")
            if outlook_counts:
                lines.append(f"\n[시장 전망 분포] {' | '.join(outlook_counts)}")

        # Add semantic search highlights
        if result.semantic_matches:
            lines.append("\n[의미 검색 결과]")
            for match in result.semantic_matches[:3]:
                score_pct = int(match['similarity_score'] * 100)
                lines.append(f"  • [{score_pct}%] {match.get('issuer', '')}: {match.get('title', '')[:40]}")

        result.answer = '\n'.join(lines)


class QueryRefiner:
    """
    Refines casual queries into structured financial queries.
    """

    SYNONYM_MAP = {
        # Market terms
        '증시': '주식 시장',
        '코스피': '한국 주식 시장',
        '나스닥': '미국 기술주',
        'S&P': '미국 대형주',

        # Action terms
        '사야': '매수',
        '팔아야': '매도',
        '들어가야': '진입',
        '나와야': '청산',

        # Sentiment
        '좋아': '강세',
        '나빠': '약세',
        '불안': '변동성',

        # Time
        '요즘': '최근',
        '앞으로': '전망',
        '내년': '2026년',
        '올해': '2025년',
    }

    def refine(self, query: str) -> str:
        """Refine casual query to financial terms."""
        refined = query

        for casual, formal in self.SYNONYM_MAP.items():
            refined = refined.replace(casual, formal)

        return refined


if __name__ == "__main__":
    sq = SmartQuery()

    test_queries = [
        "미국 증시 전망",
        "채권 비중 늘려야 하나?",
        "AI 테마 관련 전략",
        "키움증권 최근 리포트",
        "금리 인상 영향",
    ]

    print("=" * 60)
    print("SmartQuery Test")
    print("=" * 60)

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = sq.query(q, max_reports=3, verbose=True)
        print(f"Intent: {result.intent}")
        print(f"Time: {result.execution_time_ms:.0f}ms")
        print(f"Reports: {len(result.reports)}")
        print("-" * 40)
