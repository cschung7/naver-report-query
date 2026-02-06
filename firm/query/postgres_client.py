"""
PostgreSQL Query Client

Query interface for firm reports and extractions.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import get_settings


class PostgresClient:
    """Query client for PostgreSQL database."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = psycopg2.connect(self.settings.postgres_uri)
        try:
            yield conn
        finally:
            conn.close()

    def search_reports(
        self,
        query: Optional[str] = None,
        company: Optional[str] = None,
        issuer: Optional[str] = None,
        sector: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        valuation_regime: Optional[str] = None,
        growth_regime: Optional[str] = None,
        has_extraction: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search firm reports with filters.

        Args:
            query: Text search in title/summary
            company: Filter by company name (partial match)
            issuer: Filter by broker name
            sector: Filter by sector
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            valuation_regime: Filter by valuation regime
            growth_regime: Filter by growth regime
            has_extraction: Filter by extraction availability
            limit: Max results
            offset: Pagination offset

        Returns:
            List of report dictionaries
        """
        conditions = []
        params = []

        if query:
            conditions.append(
                "(r.title ILIKE %s OR r.summary ILIKE %s OR r.company ILIKE %s)"
            )
            params.extend([f"%{query}%"] * 3)

        if company:
            conditions.append("r.company ILIKE %s")
            params.append(f"%{company}%")

        if issuer:
            conditions.append("r.issuer = %s")
            params.append(issuer)

        if sector:
            conditions.append("e.sector = %s")
            params.append(sector)

        if date_from:
            conditions.append("r.issue_date >= %s")
            params.append(date_from)

        if date_to:
            conditions.append("r.issue_date <= %s")
            params.append(date_to)

        if valuation_regime:
            conditions.append("e.valuation_regime = %s")
            params.append(valuation_regime)

        if growth_regime:
            conditions.append("e.growth_regime = %s")
            params.append(growth_regime)

        if has_extraction is not None:
            conditions.append("r.has_extraction = %s")
            params.append(has_extraction)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT
                r.report_id,
                r.company,
                r.title,
                r.issuer,
                r.issue_date,
                r.viewer_count,
                r.summary,
                r.author,
                r.pdf_link,
                r.has_extraction,
                r.has_md_file,
                r.local_md_path,
                e.ticker,
                e.sector,
                e.valuation_regime,
                e.current_per,
                e.current_pbr,
                e.growth_regime,
                e.industry_cycle_stage
            FROM firm_reports r
            LEFT JOIN firm_extraction e ON r.report_id = e.report_id
            WHERE {where_clause}
            ORDER BY r.issue_date DESC
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                results = cur.fetchall()

        return [dict(r) for r in results]

    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get a single report by ID with full extraction data."""
        sql = """
            SELECT
                r.*,
                e.ticker,
                e.sector,
                e.industry,
                e.extraction_confidence,
                e.valuation_regime,
                e.current_per,
                e.current_pbr,
                e.current_ev_ebitda,
                e.dividend_yield,
                e.roe_current,
                e.roe_trend,
                e.growth_regime,
                e.growth_drivers,
                e.industry_cycle_stage,
                e.rerating_catalyst,
                e.raw_json
            FROM firm_reports r
            LEFT JOIN firm_extraction e ON r.report_id = e.report_id
            WHERE r.report_id = %s
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (report_id,))
                result = cur.fetchone()

        return dict(result) if result else None

    def get_company_coverage(
        self,
        company: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all reports for a company."""
        return self.search_reports(company=company, limit=limit)

    def get_broker_coverage(
        self,
        issuer: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all reports from a broker."""
        return self.search_reports(issuer=issuer, limit=limit)

    def get_recent_reports(
        self,
        days: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent reports."""
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self.search_reports(date_from=date_from, limit=limit)

    def get_undervalued_stocks(
        self,
        sector: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get reports with DEEP_DISCOUNT valuation regime."""
        return self.search_reports(
            sector=sector,
            valuation_regime='DEEP_DISCOUNT',
            limit=limit
        )

    def get_high_growth_stocks(
        self,
        sector: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get reports with high growth regime."""
        return self.search_reports(
            sector=sector,
            growth_regime='HIGH_GROWTH',
            limit=limit
        )

    def get_report_ids_for_date_range(
        self,
        date_from: str,
        date_to: str
    ) -> List[str]:
        """Get report IDs in a date range (for Gemini upload)."""
        sql = """
            SELECT report_id
            FROM firm_reports
            WHERE issue_date >= %s AND issue_date <= %s
            ORDER BY issue_date DESC
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (date_from, date_to))
                return [r[0] for r in cur.fetchall()]

    def get_md_paths(
        self,
        report_ids: List[str]
    ) -> Dict[str, str]:
        """Get MD file paths for report IDs."""
        if not report_ids:
            return {}

        sql = """
            SELECT report_id, local_md_path
            FROM firm_reports
            WHERE report_id = ANY(%s)
              AND local_md_path IS NOT NULL
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (report_ids,))
                return {r[0]: r[1] for r in cur.fetchall()}

    def get_distinct_values(self, column: str) -> List[str]:
        """Get distinct values for a column."""
        valid_columns = {
            'issuer': 'firm_reports',
            'company': 'firm_reports',
            'sector': 'firm_extraction',
            'valuation_regime': 'firm_extraction',
            'growth_regime': 'firm_extraction',
            'industry_cycle_stage': 'firm_extraction'
        }

        if column not in valid_columns:
            raise ValueError(f"Invalid column: {column}")

        table = valid_columns[column]
        sql = f"""
            SELECT DISTINCT {column}
            FROM {table}
            WHERE {column} IS NOT NULL
            ORDER BY {column}
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [r[0] for r in cur.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics (total reports, issuers count)."""
        sql = """
            SELECT
                COUNT(*) as total_reports,
                COUNT(DISTINCT issuer) as issuers
            FROM firm_reports
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                row = cur.fetchone()
                return {
                    'total_reports': row['total_reports'],
                    'issuers': row['issuers']
                }


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    client = PostgresClient()

    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else None
        results = client.search_reports(query=query, limit=10)
        print(f"\nSearch results ({len(results)}):")
        for r in results:
            print(f"  {r['issue_date']} | {r['issuer']} | {r['company']}")
            print(f"    {r['title'][:60]}...")

    elif len(sys.argv) > 1 and sys.argv[1] == 'recent':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        results = client.get_recent_reports(days=days, limit=10)
        print(f"\nRecent {days} days ({len(results)}):")
        for r in results:
            print(f"  {r['issue_date']} | {r['issuer']} | {r['company']}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'undervalued':
        results = client.get_undervalued_stocks(limit=10)
        print(f"\nUndervalued stocks ({len(results)}):")
        for r in results:
            print(f"  {r['company']} | PER: {r['current_per']} | PBR: {r['current_pbr']}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'issuers':
        issuers = client.get_distinct_values('issuer')
        print(f"\nBrokers ({len(issuers)}):")
        for issuer in issuers[:20]:
            print(f"  {issuer}")

    else:
        print("Usage: python -m query.postgres_client [search <query>|recent <days>|undervalued|issuers]")
