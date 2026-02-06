"""
PostgreSQL Client for Industry Analysis Reports
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import POSTGRES_URI, PG_TABLE


class PostgresClient:
    """PostgreSQL client for Industry Analysis queries."""

    def __init__(self, uri: str = None):
        self.uri = uri or POSTGRES_URI
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.uri)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()

    def search_reports(
        self,
        query: str = None,
        keywords: List[str] = None,
        issuer: str = None,
        industry: str = None,
        cycle_stage: str = None,
        demand_trend: str = None,
        investment_timing: str = None,
        geography: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 20,
        broad_search: bool = True
    ) -> List[Dict]:
        """
        Search Industry Analysis reports with improved matching.

        Args:
            query: Text search query
            keywords: List of individual keywords to search
            issuer: Filter by broker/issuer
            industry: Filter by industry/sector
            cycle_stage: Filter by cycle stage (UPCYCLE, DOWNCYCLE, etc.)
            demand_trend: Filter by demand trend (GROWING, CONTRACTING, etc.)
            investment_timing: Filter by investment timing (BUY, HOLD, AVOID)
            geography: Filter by geography (GLOBAL, KOREA, US, etc.)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Maximum results
            broad_search: If True, search in all text fields including JSON

        Returns:
            List of matching reports
        """
        conditions = []
        params = []

        # Build text search conditions
        text_conditions = []

        if query:
            like_query = f"%{query}%"
            text_conditions.append("title ILIKE %s")
            params.append(like_query)
            text_conditions.append("summary ILIKE %s")
            params.append(like_query)

            if broad_search:
                # Search in JSON fields as text
                text_conditions.append("cycle_drivers::text ILIKE %s")
                params.append(like_query)
                text_conditions.append("demand_drivers::text ILIKE %s")
                params.append(like_query)
                text_conditions.append("key_themes::text ILIKE %s")
                params.append(like_query)

        # Search individual keywords with OR logic
        if keywords:
            for kw in keywords:
                if len(kw) >= 2:  # Skip very short words
                    like_kw = f"%{kw}%"
                    text_conditions.append("title ILIKE %s")
                    params.append(like_kw)
                    text_conditions.append("summary ILIKE %s")
                    params.append(like_kw)
                    if broad_search:
                        text_conditions.append("cycle_drivers::text ILIKE %s")
                        params.append(like_kw)
                        text_conditions.append("key_themes::text ILIKE %s")
                        params.append(like_kw)

        if text_conditions:
            conditions.append(f"({' OR '.join(text_conditions)})")

        # Filter conditions
        if issuer:
            conditions.append("issuer = %s")
            params.append(issuer)

        if industry:
            conditions.append("industry ILIKE %s")
            params.append(f"%{industry}%")

        if cycle_stage:
            conditions.append("(cycle_stage = %s OR cycle_stage IS NULL)")
            params.append(cycle_stage)

        if demand_trend:
            conditions.append("(demand_trend = %s OR demand_trend IS NULL)")
            params.append(demand_trend)

        if investment_timing:
            conditions.append("(investment_timing = %s OR investment_timing IS NULL)")
            params.append(investment_timing)

        if geography:
            conditions.append("(geography = %s OR geography = 'GLOBAL' OR geography IS NULL)")
            params.append(geography)

        if date_from:
            conditions.append("issue_date >= %s")
            params.append(date_from)

        if date_to:
            conditions.append("issue_date <= %s")
            params.append(date_to)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT
                report_id, title, issuer, issue_date, author, summary,
                industry, sector, geography,
                cycle_stage, cycle_drivers, cycle_duration,
                demand_trend, demand_drivers,
                supply_dynamics, competitive_landscape,
                investment_timing, key_themes,
                extraction_confidence, pdf_link
            FROM {PG_TABLE}
            WHERE {where_clause}
            ORDER BY issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_by_industry(self, industry_keywords: List[str], limit: int = 20) -> List[Dict]:
        """Search reports by industry."""
        conditions = []
        params = []

        for kw in industry_keywords:
            conditions.append("industry ILIKE %s")
            params.append(f"%{kw}%")

        sql = f"""
            SELECT report_id, title, issuer, issue_date, author, summary,
                industry, sector, geography,
                cycle_stage, cycle_drivers, cycle_duration,
                demand_trend, demand_drivers,
                supply_dynamics, competitive_landscape,
                investment_timing, key_themes,
                extraction_confidence, pdf_link
            FROM {PG_TABLE}
            WHERE {' OR '.join(conditions)}
            ORDER BY issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_by_cycle(self, cycle_stage: str, limit: int = 20) -> List[Dict]:
        """Search reports by cycle stage."""
        sql = f"""
            SELECT report_id, title, issuer, issue_date, author, summary,
                industry, sector, geography,
                cycle_stage, cycle_drivers, cycle_duration,
                demand_trend, demand_drivers,
                supply_dynamics, competitive_landscape,
                investment_timing, key_themes,
                extraction_confidence, pdf_link
            FROM {PG_TABLE}
            WHERE cycle_stage = %s
            ORDER BY issue_date DESC
            LIMIT %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (cycle_stage, limit))
            return [dict(row) for row in cur.fetchall()]

    def search_by_demand(self, demand_trend: str, limit: int = 20) -> List[Dict]:
        """Search reports by demand trend."""
        sql = f"""
            SELECT report_id, title, issuer, issue_date, author, summary,
                industry, sector, geography,
                cycle_stage, cycle_drivers, cycle_duration,
                demand_trend, demand_drivers,
                supply_dynamics, competitive_landscape,
                investment_timing, key_themes,
                extraction_confidence, pdf_link
            FROM {PG_TABLE}
            WHERE demand_trend = %s
            ORDER BY issue_date DESC
            LIMIT %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (demand_trend, limit))
            return [dict(row) for row in cur.fetchall()]

    def get_recent_reports(self, days: int = 30, limit: int = 50) -> List[Dict]:
        """Get recent reports."""
        date_from = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.search_reports(date_from=date_from, limit=limit)

    def get_by_investment_timing(self, timing: str, limit: int = 20) -> List[Dict]:
        """Get reports by investment timing."""
        return self.search_reports(investment_timing=timing, limit=limit)

    def get_issuers(self) -> List[Dict]:
        """Get list of issuers with stats."""
        sql = f"""
            SELECT issuer, COUNT(*) as count,
                   MAX(issue_date) as latest_report
            FROM {PG_TABLE}
            GROUP BY issuer
            ORDER BY count DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    def get_industries(self) -> List[Dict]:
        """Get list of industries with stats."""
        sql = f"""
            SELECT industry, COUNT(*) as count,
                   MAX(issue_date) as latest_report
            FROM {PG_TABLE}
            WHERE industry IS NOT NULL
            GROUP BY industry
            ORDER BY count DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) as total FROM {PG_TABLE}")
            total = cur.fetchone()['total']

            cur.execute(f"SELECT MIN(issue_date) as min_date, MAX(issue_date) as max_date FROM {PG_TABLE}")
            dates = cur.fetchone()

            cur.execute(f"SELECT COUNT(DISTINCT issuer) as issuers FROM {PG_TABLE}")
            issuers = cur.fetchone()['issuers']

            cur.execute(f"SELECT COUNT(DISTINCT industry) as industries FROM {PG_TABLE}")
            industries = cur.fetchone()['industries']

            cur.execute(f"""
                SELECT industry, COUNT(*) as count
                FROM {PG_TABLE}
                WHERE industry IS NOT NULL
                GROUP BY industry
                ORDER BY count DESC
                LIMIT 10
            """)
            top_industries = [dict(row) for row in cur.fetchall()]

            cur.execute(f"""
                SELECT cycle_stage, COUNT(*) as count
                FROM {PG_TABLE}
                WHERE cycle_stage IS NOT NULL
                GROUP BY cycle_stage
                ORDER BY count DESC
            """)
            cycle_distribution = [dict(row) for row in cur.fetchall()]

            return {
                'total_reports': total,
                'date_range': {
                    'min': str(dates['min_date']) if dates['min_date'] else None,
                    'max': str(dates['max_date']) if dates['max_date'] else None,
                },
                'issuers': issuers,
                'industries': industries,
                'top_industries': top_industries,
                'cycle_distribution': cycle_distribution,
            }


if __name__ == "__main__":
    client = PostgresClient()
    print("Stats:", client.get_stats())
    print("\nRecent reports:")
    for r in client.get_recent_reports(days=365, limit=5):
        print(f"  - {r['issue_date']} {r['issuer']}: {r['title'][:50]}...")
