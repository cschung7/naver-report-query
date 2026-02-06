"""
Hybrid Query Orchestrator

Coordinates queries across PostgreSQL, Neo4j, and Vector DB for comprehensive results.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import get_settings
from .postgres_client import PostgresClient
from .neo4j_client import Neo4jClient


@dataclass
class QueryResult:
    """Result from hybrid query."""
    query: str
    answer: Optional[str] = None
    reports: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    context_expansion: Dict[str, Any] = field(default_factory=dict)
    sources_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridQuery:
    """
    Hybrid query engine orchestrating:
    1. PostgreSQL for structured filtering and metadata
    2. Neo4j for claim-centric context expansion
    3. Vector DB for semantic search (FREE, local)
    """

    def __init__(self, settings=None, enable_vector: bool = True):
        self.settings = settings or get_settings()
        self.pg_client = PostgresClient(self.settings)
        self.neo4j_client = Neo4jClient(self.settings)

        # Vector DB client (FREE, local)
        self.vector_client = None
        if enable_vector:
            try:
                from .vector_client import VectorClient
                self.vector_client = VectorClient(self.settings)
            except Exception as e:
                print(f"Vector DB not available: {e}")

    def close(self):
        """Close all connections."""
        self.neo4j_client.close()

    def query(
        self,
        question: str,
        company: Optional[str] = None,
        issuer: Optional[str] = None,
        sector: Optional[str] = None,
        valuation_regime: Optional[str] = None,
        growth_regime: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        expand_context: bool = True,
        use_semantic: bool = True,
        max_reports: int = 20,
        max_claims: int = 50
    ) -> QueryResult:
        """
        Execute a hybrid query across all data sources.

        Query Flow:
        1. PostgreSQL: Filter reports by structured criteria
        2. Neo4j: Expand with claim-centric context
        3. Vector DB: FREE local semantic search

        Args:
            question: User query/question
            company: Filter by company name
            issuer: Filter by broker
            sector: Filter by sector
            valuation_regime: Filter by valuation (DEEP_DISCOUNT, etc.)
            growth_regime: Filter by growth (HIGH_GROWTH, etc.)
            date_from: Start date filter
            date_to: End date filter
            expand_context: Whether to expand with Neo4j claims
            use_semantic: Whether to use Vector DB for semantic search
            max_reports: Maximum reports to return
            max_claims: Maximum claims to return

        Returns:
            QueryResult with aggregated results
        """
        result = QueryResult(query=question)
        result.metadata['started_at'] = datetime.now().isoformat()

        # Step 1: PostgreSQL filtering
        reports = self._query_postgres(
            question=question,
            company=company,
            issuer=issuer,
            sector=sector,
            valuation_regime=valuation_regime,
            growth_regime=growth_regime,
            date_from=date_from,
            date_to=date_to,
            limit=max_reports
        )
        result.reports = reports
        result.sources_used.append('postgresql')

        # Step 2: Neo4j context expansion
        if expand_context and reports:
            claims, context = self._expand_with_neo4j(
                reports=reports,
                company=company,
                valuation_regime=valuation_regime,
                growth_regime=growth_regime,
                max_claims=max_claims
            )
            # Sort claims by issue_date descending (most recent first)
            claims = sorted(
                claims,
                key=lambda x: x.get('issue_date') or '',
                reverse=True
            )
            result.claims = claims
            result.context_expansion = context
            result.sources_used.append('neo4j')

        # Step 3: Vector DB semantic search (FREE, local)
        if use_semantic and self.vector_client:
            vector_result = self._query_vector(
                question=question,
                company=company,
                reports=reports
            )
            if vector_result:
                result.answer = vector_result.get('answer')
                result.metadata['vector_sources'] = vector_result.get('sources', [])
                result.metadata['chunks_found'] = vector_result.get('chunks_found', 0)
                result.sources_used.append('vector_db')

        result.metadata['completed_at'] = datetime.now().isoformat()
        result.metadata['report_count'] = len(result.reports)
        result.metadata['claim_count'] = len(result.claims)

        return result

    def _query_postgres(
        self,
        question: str,
        company: Optional[str],
        issuer: Optional[str],
        sector: Optional[str],
        valuation_regime: Optional[str],
        growth_regime: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute PostgreSQL query."""
        # Only use question as text search if no structured filters provided
        has_filters = any([company, issuer, sector, valuation_regime, growth_regime])
        text_query = None if has_filters else question

        return self.pg_client.search_reports(
            query=text_query,
            company=company,
            issuer=issuer,
            sector=sector,
            valuation_regime=valuation_regime,
            growth_regime=growth_regime,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )

    def _expand_with_neo4j(
        self,
        reports: List[Dict[str, Any]],
        company: Optional[str],
        valuation_regime: Optional[str],
        growth_regime: Optional[str],
        max_claims: int
    ) -> tuple:
        """Expand results with Neo4j claims."""
        claims = []
        context = {}

        # Get claims for companies in results
        companies = set(r['company'] for r in reports if r.get('company'))

        for comp in list(companies)[:5]:  # Limit to top 5 companies
            company_claims = self.neo4j_client.get_company_claims(
                company=comp,
                limit=max_claims // max(len(companies), 1)
            )
            claims.extend(company_claims)

        # Get claims by context values
        if valuation_regime:
            context_claims = self.neo4j_client.get_claims_by_context(
                context_value=valuation_regime,
                claim_type='VALUATION',
                limit=20
            )
            context['valuation_claims'] = context_claims

        if growth_regime:
            context_claims = self.neo4j_client.get_claims_by_context(
                context_value=growth_regime,
                claim_type='GROWTH',
                limit=20
            )
            context['growth_claims'] = context_claims

        # Get company context summaries
        if company:
            try:
                summary = self.neo4j_client.get_company_context_summary(company)
                context['company_summary'] = summary
            except Exception:
                pass

        return claims, context

    def _query_vector(
        self,
        question: str,
        company: Optional[str],
        reports: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Execute Vector DB semantic search (FREE, local)."""
        if not self.vector_client:
            return None

        try:
            # Get report IDs from results
            report_ids = [r['report_id'] for r in reports] if reports else None

            # Query vector DB
            result = self.vector_client.query(
                question=question,
                report_ids=report_ids[:20] if report_ids else None,
                company=company,
                n_results=10
            )

            return result
        except Exception as e:
            print(f"Vector query failed: {e}")
            return None

    # Convenience methods for common queries
    def company_analysis(self, company: str) -> QueryResult:
        """Get comprehensive analysis for a company."""
        return self.query(
            question=f"{company} 투자의견 분석",
            company=company,
            expand_context=True
        )

    def sector_screening(
        self,
        sector: str,
        valuation: Optional[str] = None,
        growth: Optional[str] = None
    ) -> QueryResult:
        """Screen companies in a sector."""
        return self.query(
            question=f"{sector} 섹터 종목 스크리닝",
            sector=sector,
            valuation_regime=valuation,
            growth_regime=growth
        )

    def undervalued_screen(self, sector: Optional[str] = None) -> QueryResult:
        """Find undervalued stocks."""
        return self.query(
            question="저평가 종목",
            sector=sector,
            valuation_regime="DEEP_DISCOUNT"
        )

    def high_growth_screen(self, sector: Optional[str] = None) -> QueryResult:
        """Find high growth stocks."""
        return self.query(
            question="고성장 종목",
            sector=sector,
            growth_regime="HIGH_GROWTH"
        )

    def broker_research(
        self,
        issuer: str,
        company: Optional[str] = None
    ) -> QueryResult:
        """Get research from a specific broker."""
        return self.query(
            question=f"{issuer} 리서치",
            issuer=issuer,
            company=company
        )

    def recent_reports(
        self,
        days: int = 7,
        company: Optional[str] = None,
        sector: Optional[str] = None
    ) -> QueryResult:
        """Get recent reports."""
        from datetime import datetime, timedelta
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        return self.query(
            question="최근 리포트",
            company=company,
            sector=sector,
            date_from=date_from
        )


def format_result(result: QueryResult, verbose: bool = False) -> str:
    """Format query result for display (chronological order - most recent first)."""
    lines = []
    lines.append(f"\n=== Query: {result.query} ===")
    lines.append(f"Sources: {', '.join(result.sources_used)}")
    lines.append(f"Reports: {len(result.reports)}, Claims: {len(result.claims)}")

    if result.answer:
        lines.append(f"\n--- Semantic Answer ---")
        lines.append(result.answer)

    if result.reports:
        lines.append(f"\n--- Reports (Most Recent First) ---")
        for r in result.reports[:10]:
            lines.append(
                f"  {r['issue_date']} | {r['issuer']} | {r['company']}"
            )
            if verbose:
                lines.append(f"    {r.get('title', '')[:60]}...")
                if r.get('valuation_regime'):
                    lines.append(f"    Valuation: {r['valuation_regime']}")

    if result.claims and verbose:
        lines.append(f"\n--- Claims (Most Recent First) ---")
        for c in result.claims[:10]:
            date = c.get('issue_date', 'N/A')
            ctype = c.get('claim_type', 'N/A')
            issuer = c.get('issuer', '')
            text = c.get('claim_text', '') or ''
            lines.append(f"  {date} | [{ctype}] ({issuer})")
            lines.append(f"    {text[:70]}...")

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import sys

    hq = HybridQuery()

    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'company':
            company = sys.argv[2] if len(sys.argv) > 2 else "삼성전자"
            result = hq.company_analysis(company)
            print(format_result(result, verbose=True))

        elif len(sys.argv) > 1 and sys.argv[1] == 'undervalued':
            sector = sys.argv[2] if len(sys.argv) > 2 else None
            result = hq.undervalued_screen(sector=sector)
            print(format_result(result, verbose=True))

        elif len(sys.argv) > 1 and sys.argv[1] == 'growth':
            sector = sys.argv[2] if len(sys.argv) > 2 else None
            result = hq.high_growth_screen(sector=sector)
            print(format_result(result, verbose=True))

        elif len(sys.argv) > 1 and sys.argv[1] == 'broker':
            issuer = sys.argv[2] if len(sys.argv) > 2 else "한화투자증권"
            result = hq.broker_research(issuer)
            print(format_result(result, verbose=True))

        elif len(sys.argv) > 1 and sys.argv[1] == 'recent':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            result = hq.recent_reports(days=days)
            print(format_result(result))

        elif len(sys.argv) > 1 and sys.argv[1] == 'query':
            question = " ".join(sys.argv[2:])
            result = hq.query(question)
            print(format_result(result, verbose=True))

        else:
            print("Usage: python -m query.hybrid_query [company <name>|undervalued|growth|broker <name>|recent <days>|query <question>]")

    finally:
        hq.close()
