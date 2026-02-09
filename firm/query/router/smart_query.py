"""
Smart Query

Intelligent query interface that automatically routes queries to appropriate databases.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures

from config.settings import get_settings
from ..postgres_client import PostgresClient
from ..neo4j_client import Neo4jClient
from ..vector_client import VectorClient

from .query_analyzer import QueryAnalyzer, ExtractedEntities
from .signal_detector import SignalDetector, DetectedSignals
from .route_planner import RoutePlanner, RouteDecision, QueryIntent, DatabasePriority


@dataclass
class SmartQueryResult:
    """Result from smart query execution."""
    query: str
    intent: str
    answer: Optional[str] = None
    reports: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)

    # Analysis metadata
    entities: Optional[ExtractedEntities] = None
    routing: Optional[RouteDecision] = None
    execution_time_ms: float = 0

    # Per-database results
    pg_result: Optional[Dict] = None
    neo4j_result: Optional[Dict] = None
    vector_result: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'query': self.query,
            'intent': self.intent,
            'answer': self.answer,
            'reports_count': len(self.reports),
            'claims_count': len(self.claims),
            'sources_used': self.sources_used,
            'execution_time_ms': self.execution_time_ms,
        }


class SmartQuery:
    """
    Intelligent query interface with automatic routing.

    Usage:
        sq = SmartQuery()
        result = sq.query("테슬라 공급망 관련 한국 배터리 기업")
        print(result.answer)
        print(result.reports)
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        # Initialize analyzers
        self.analyzer = QueryAnalyzer()
        self.detector = SignalDetector()
        self.planner = RoutePlanner()

        # Initialize database clients (lazy loading)
        self._pg_client = None
        self._neo4j_client = None
        self._vector_client = None

    @property
    def pg_client(self) -> PostgresClient:
        if self._pg_client is None:
            self._pg_client = PostgresClient(self.settings)
        return self._pg_client

    @property
    def neo4j_client(self) -> Neo4jClient:
        if self._neo4j_client is None:
            try:
                self._neo4j_client = Neo4jClient(self.settings)
            except Exception as e:
                print(f"Neo4j client not available: {e}")
        return self._neo4j_client

    @property
    def vector_client(self) -> VectorClient:
        if self._vector_client is None:
            try:
                self._vector_client = VectorClient(self.settings)
            except Exception as e:
                print(f"Vector client not available: {e}")
        return self._vector_client

    def close(self):
        """Close all database connections."""
        if self._neo4j_client:
            self._neo4j_client.close()

    def query_fast(
        self,
        question: str,
        max_reports: int = 20,
        max_claims: int = 50,
        verbose: bool = False,
        date_from: str = None,
        date_to: str = None
    ) -> SmartQueryResult:
        """
        Execute fast query using only PostgreSQL and Neo4j (no vector DB).

        This is much faster for structured queries like company lookups,
        sector filtering, etc.
        """
        start_time = datetime.now()

        # Step 1: Analyze query
        entities = self.analyzer.analyze(question)

        # Step 2: Create simple result
        result = SmartQueryResult(
            query=question,
            intent="fast_lookup"
        )

        # Step 3: Query PostgreSQL
        try:
            params = {}
            if entities.companies:
                params['company'] = entities.companies[0]
            if entities.issuers:
                params['issuer'] = entities.issuers[0]
            if entities.sectors:
                params['sector'] = entities.sectors[0]
            if entities.valuation_keywords:
                params['valuation_regime'] = entities.valuation_keywords[0]
            if entities.growth_keywords:
                params['growth_regime'] = entities.growth_keywords[0]
            # Explicit date params override entity-extracted dates
            if date_from:
                params['date_from'] = date_from
            elif entities.date_range:
                params['date_from'] = entities.date_range.get('from')
            if date_to:
                params['date_to'] = date_to
            elif entities.date_range:
                params['date_to'] = entities.date_range.get('to')

            # Use text search if no structured filters
            text_query = question if not params else None

            reports = self.pg_client.search_reports(
                query=text_query,
                company=params.get('company'),
                issuer=params.get('issuer'),
                sector=params.get('sector'),
                valuation_regime=params.get('valuation_regime'),
                growth_regime=params.get('growth_regime'),
                date_from=params.get('date_from'),
                date_to=params.get('date_to'),
                limit=max_reports
            )
            result.reports = reports
            result.sources_used.append('postgresql')

            if verbose:
                print(f"[PostgreSQL] Found {len(reports)} reports")
        except Exception as e:
            if verbose:
                print(f"[PostgreSQL] Error: {e}")

        # Step 4: Query Neo4j for claims (optional, based on entities)
        if entities.companies:
            try:
                claims = []
                for company in entities.companies[:3]:
                    company_claims = self.neo4j_client.get_company_claims(
                        company=company,
                        limit=max_claims // max(len(entities.companies), 1)
                    )
                    claims.extend(company_claims)
                result.claims = claims[:max_claims]
                result.sources_used.append('neo4j')

                if verbose:
                    print(f"[Neo4j] Found {len(claims)} claims")
            except Exception as e:
                if verbose:
                    print(f"[Neo4j] Error: {e}")

        # Step 5: Synthesize answer
        self._synthesize_fast_answer(result)

        result.entities = entities
        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _synthesize_fast_answer(self, result: SmartQueryResult):
        """Synthesize answer from PostgreSQL and Neo4j results only."""
        parts = []

        if result.reports:
            companies = set(r['company'] for r in result.reports if r.get('company'))
            issuers = set(r['issuer'] for r in result.reports if r.get('issuer'))
            sectors = set(r['sector'] for r in result.reports if r.get('sector'))
            dates = [r.get('issue_date') for r in result.reports if r.get('issue_date')]

            summary = []
            if companies:
                summary.append(f"관련 기업: {', '.join(sorted(companies)[:10])}")
            if issuers:
                summary.append(f"분석 기관: {', '.join(sorted(issuers)[:5])}")
            if sectors:
                summary.append(f"섹터: {', '.join(sorted(sectors))}")
            if dates:
                latest = max(str(d)[:10] for d in dates if d)
                summary.append(f"최근 리포트: {latest}")
            summary.append(f"리포트 {len(result.reports)}건 발견")

            # Add summaries from reports
            if any(r.get('summary') or r.get('title') for r in result.reports[:5]):
                summary.append("\n[관련 내용]")
                for i, r in enumerate(result.reports[:5], 1):
                    if r.get('summary'):
                        summary.append(f"{i}. {r['company']}: {r['summary'][:150]}...")
                    elif r.get('title'):
                        summary.append(f"{i}. {r['company']}: {r['title']}")

            parts.append("\n".join(summary))

        # Add claim highlights if available
        if result.claims:
            claim_summary = []
            for claim in result.claims[:3]:
                claim_type = claim.get('claim_type', '')
                claim_text = claim.get('claim_text', '')[:100]
                if claim_text:
                    claim_summary.append(f"[{claim_type}] {claim_text}...")
            if claim_summary:
                parts.append("\n주요 분석:\n" + "\n".join(claim_summary))

        result.answer = "\n\n".join(parts) if parts else "관련 정보를 찾을 수 없습니다."

    def query(
        self,
        question: str,
        max_reports: int = 20,
        max_claims: int = 50,
        verbose: bool = False,
        date_from: str = None,
        date_to: str = None
    ) -> SmartQueryResult:
        """
        Execute intelligent query with automatic routing.

        Args:
            question: Natural language query
            max_reports: Maximum reports to return
            max_claims: Maximum claims to return
            verbose: Print debug information

        Returns:
            SmartQueryResult with aggregated results
        """
        start_time = datetime.now()

        # Step 1: Analyze query
        entities = self.analyzer.analyze(question)
        if verbose:
            print(f"[Analyzer] Entities: {entities}")

        # Step 2: Detect signals
        signals = self.detector.detect(question, entities)
        if verbose:
            print(f"[Detector] Signals: {signals.signals}")

        # Step 3: Plan routing
        decision = self.planner.plan(question, entities, signals)
        if verbose:
            print(f"[Planner] Decision: {decision.intent.value}")
            print(f"[Planner] DBs: {decision.get_active_databases()}")

        # Inject explicit date filters into pg_params
        if date_from:
            decision.pg_params['date_from'] = date_from
        if date_to:
            decision.pg_params['date_to'] = date_to

        # Step 4: Execute queries
        result = self._execute_queries(
            question=question,
            decision=decision,
            entities=entities,
            max_reports=max_reports,
            max_claims=max_claims,
            verbose=verbose
        )

        # Set metadata
        result.entities = entities
        result.routing = decision
        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _execute_queries(
        self,
        question: str,
        decision: RouteDecision,
        entities: ExtractedEntities,
        max_reports: int,
        max_claims: int,
        verbose: bool
    ) -> SmartQueryResult:
        """Execute queries according to routing decision."""
        result = SmartQueryResult(
            query=question,
            intent=decision.intent.value
        )

        if decision.parallel and len(decision.sequence) > 1:
            # Parallel execution
            self._execute_parallel(result, decision, max_reports, max_claims, verbose)
        else:
            # Sequential execution
            self._execute_sequential(result, decision, max_reports, max_claims, verbose)

        # Synthesize final answer
        self._synthesize_answer(result, decision)

        return result

    def _execute_sequential(
        self,
        result: SmartQueryResult,
        decision: RouteDecision,
        max_reports: int,
        max_claims: int,
        verbose: bool
    ):
        """Execute queries sequentially according to sequence."""
        for db in decision.sequence:
            try:
                if db == 'postgresql' and decision.postgresql != DatabasePriority.SKIP:
                    self._execute_postgresql(result, decision, max_reports, verbose)
                elif db == 'neo4j' and decision.neo4j != DatabasePriority.SKIP:
                    self._execute_neo4j(result, decision, max_claims, verbose)
                elif db == 'vector_db' and decision.vector_db != DatabasePriority.SKIP:
                    self._execute_vector(result, decision, verbose)
            except Exception as e:
                if verbose:
                    print(f"[{db}] Execution error: {e}")

    def _execute_parallel(
        self,
        result: SmartQueryResult,
        decision: RouteDecision,
        max_reports: int,
        max_claims: int,
        verbose: bool
    ):
        """Execute queries in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if decision.postgresql != DatabasePriority.SKIP:
                futures['postgresql'] = executor.submit(
                    self._execute_postgresql, result, decision, max_reports, verbose
                )

            if decision.neo4j != DatabasePriority.SKIP:
                futures['neo4j'] = executor.submit(
                    self._execute_neo4j, result, decision, max_claims, verbose
                )

            if decision.vector_db != DatabasePriority.SKIP:
                futures['vector_db'] = executor.submit(
                    self._execute_vector, result, decision, verbose
                )

            # Wait for all to complete
            for db, future in futures.items():
                try:
                    future.result(timeout=30)
                except Exception as e:
                    if verbose:
                        print(f"[Error] {db} failed: {e}")

    def _execute_postgresql(
        self,
        result: SmartQueryResult,
        decision: RouteDecision,
        max_reports: int,
        verbose: bool
    ):
        """Execute PostgreSQL query."""
        params = decision.pg_params

        # For SEMANTIC_QA without structured filters, use text search
        text_query = None
        if decision.intent == QueryIntent.SEMANTIC_QA and not params:
            text_query = result.query

        try:
            reports = self.pg_client.search_reports(
                query=text_query,
                company=params.get('company'),
                issuer=params.get('issuer'),
                sector=params.get('sector'),
                valuation_regime=params.get('valuation_regime'),
                growth_regime=params.get('growth_regime'),
                date_from=params.get('date_from'),
                date_to=params.get('date_to'),
                limit=max_reports
            )

            result.reports = reports
            result.sources_used.append('postgresql')
            result.pg_result = {'count': len(reports)}

            if verbose:
                print(f"[PostgreSQL] Found {len(reports)} reports")

        except Exception as e:
            if verbose:
                print(f"[PostgreSQL] Error: {e}")

    def _execute_neo4j(
        self,
        result: SmartQueryResult,
        decision: RouteDecision,
        max_claims: int,
        verbose: bool
    ):
        """Execute Neo4j query."""
        params = decision.neo4j_params
        query_type = params.get('query_type', 'context_expansion')

        try:
            claims = []
            relationships = []

            if query_type == 'supply_chain' and params.get('companies'):
                # Get supply chain relationships
                for company in params['companies'][:3]:
                    related = self.neo4j_client.get_company_claims(
                        company=company,
                        limit=max_claims // 3
                    )
                    claims.extend(related)

            elif query_type == 'context_expansion':
                # Get claims for companies found in PostgreSQL
                companies = set()
                for report in result.reports[:10]:
                    if report.get('company'):
                        companies.add(report['company'])

                for company in list(companies)[:5]:
                    company_claims = self.neo4j_client.get_company_claims(
                        company=company,
                        limit=max_claims // max(len(companies), 1)
                    )
                    claims.extend(company_claims)

            # Sort by date
            claims = sorted(
                claims,
                key=lambda x: x.get('issue_date') or '',
                reverse=True
            )

            result.claims = claims[:max_claims]
            result.relationships = relationships
            result.sources_used.append('neo4j')
            result.neo4j_result = {'claims': len(claims)}

            if verbose:
                print(f"[Neo4j] Found {len(claims)} claims")

        except Exception as e:
            if verbose:
                print(f"[Neo4j] Error: {e}")

    def _execute_vector(
        self,
        result: SmartQueryResult,
        decision: RouteDecision,
        verbose: bool
    ):
        """Execute Vector DB query."""
        if not self.vector_client:
            return

        params = decision.vector_params

        try:
            vector_result = self.vector_client.query(
                question=params.get('question', result.query),
                company=params.get('company'),
                n_results=params.get('n_results', 10)
            )

            result.vector_result = vector_result
            result.sources_used.append('vector_db')

            if verbose:
                print(f"[Vector] Found {vector_result.get('chunks_found', 0)} chunks")

        except Exception as e:
            if verbose:
                print(f"[Vector] Error: {e}")

    def _synthesize_answer(self, result: SmartQueryResult, decision: RouteDecision):
        """Synthesize final answer from all sources."""
        parts = []

        # Check if we have sector-based PostgreSQL results (more reliable for sector queries)
        has_sector_results = (
            result.reports and
            any(r.get('sector') for r in result.reports[:5])
        )

        # For SEMANTIC_QA, always prioritize vector results
        if decision.intent == QueryIntent.SEMANTIC_QA and not has_sector_results:
            # Add vector DB answer (primary source for semantic queries)
            if result.vector_result and result.vector_result.get('answer'):
                parts.append(result.vector_result['answer'])
            elif result.vector_result and result.vector_result.get('sources'):
                # Build answer from vector sources if no pre-built answer
                sources = result.vector_result['sources']
                companies = set(s.get('company', '') for s in sources if s.get('company'))
                issuers = set(s.get('issuer', '') for s in sources if s.get('issuer'))

                summary = []
                if companies:
                    summary.append(f"관련 기업: {', '.join(sorted(companies))}")
                if issuers:
                    summary.append(f"분석 기관: {', '.join(sorted(issuers))}")

                # Add top chunks as content
                if result.vector_result.get('top_chunks'):
                    summary.append("\n[관련 내용]")
                    for i, chunk in enumerate(result.vector_result['top_chunks'][:5], 1):
                        text = chunk.get('text', '')[:200]
                        summary.append(f"{i}. {text}...")

                if summary:
                    parts.append("\n".join(summary))
        elif has_sector_results:
            # Prioritize PostgreSQL results when we have sector-filtered data
            companies = set(r['company'] for r in result.reports if r.get('company'))
            issuers = set(r['issuer'] for r in result.reports if r.get('issuer'))
            sectors = set(r['sector'] for r in result.reports if r.get('sector'))
            dates = [r['issue_date'] for r in result.reports if r.get('issue_date')]

            summary = []
            if companies:
                summary.append(f"관련 기업: {', '.join(sorted(companies)[:10])}")
            if issuers:
                summary.append(f"분석 기관: {', '.join(sorted(issuers)[:5])}")
            if sectors:
                summary.append(f"섹터: {', '.join(sorted(sectors))}")
            if dates:
                latest_date = max(dates) if isinstance(dates[0], str) else max(d.strftime('%Y-%m-%d') if d else '' for d in dates)
                summary.append(f"최근 리포트: {str(latest_date)[:10]}")
            summary.append(f"리포트 {len(result.reports)}건 발견")

            # Add summaries from reports
            summary.append("\n[관련 내용]")
            for i, r in enumerate(result.reports[:5], 1):
                if r.get('summary'):
                    summary.append(f"{i}. {r['company']}: {r['summary'][:150]}...")
                elif r.get('title'):
                    summary.append(f"{i}. {r['company']}: {r['title']}")

            parts.append("\n".join(summary))
        else:
            # For other intents, use existing logic
            if result.vector_result and result.vector_result.get('answer'):
                parts.append(result.vector_result['answer'])

        # Add summary from reports if no answer yet
        if not parts and result.reports:
            companies = set(r['company'] for r in result.reports if r.get('company'))
            issuers = set(r['issuer'] for r in result.reports if r.get('issuer'))

            summary = []
            if companies:
                summary.append(f"관련 기업: {', '.join(list(companies)[:5])}")
            if issuers:
                summary.append(f"분석 기관: {', '.join(list(issuers)[:5])}")
            summary.append(f"리포트 {len(result.reports)}건 발견")

            parts.append("\n".join(summary))

        # Add claim highlights if available
        if result.claims and decision.intent in [QueryIntent.RELATIONSHIP, QueryIntent.RESEARCH]:
            claim_summary = []
            for claim in result.claims[:3]:
                claim_type = claim.get('claim_type', '')
                claim_text = claim.get('claim_text', '')[:100]
                if claim_text:
                    claim_summary.append(f"[{claim_type}] {claim_text}...")

            if claim_summary:
                parts.append("\n주요 분석:\n" + "\n".join(claim_summary))

        result.answer = "\n\n".join(parts) if parts else "관련 정보를 찾을 수 없습니다."


def format_smart_result(result: SmartQueryResult, verbose: bool = False) -> str:
    """Format SmartQueryResult for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Query: {result.query}")
    lines.append(f"Intent: {result.intent}")
    lines.append(f"Sources: {', '.join(result.sources_used)}")
    lines.append(f"Execution: {result.execution_time_ms:.0f}ms")
    lines.append(f"{'='*60}")

    if result.answer:
        lines.append(f"\n📊 Answer:\n{result.answer}")

    if result.reports:
        lines.append(f"\n📋 Reports ({len(result.reports)}):")
        for r in result.reports[:5]:
            lines.append(f"  {r['issue_date']} | {r['issuer']} | {r['company']}")

    if verbose and result.claims:
        lines.append(f"\n💡 Claims ({len(result.claims)}):")
        for c in result.claims[:5]:
            ctype = c.get('claim_type', 'N/A')
            text = (c.get('claim_text') or '')[:60]
            lines.append(f"  [{ctype}] {text}...")

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import sys

    sq = SmartQuery()

    try:
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
        else:
            question = "삼성전자 최근 투자의견"

        result = sq.query(question, verbose=True)
        print(format_smart_result(result, verbose=True))

    finally:
        sq.close()
