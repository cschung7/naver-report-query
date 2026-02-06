"""
PostgreSQL Client for Economic Analysis Reports
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import POSTGRES_URI, PG_TABLE


class PostgresClient:
    """PostgreSQL client for Economic Analysis queries."""

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
        category: str = None,
        region: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 20,
        broad_search: bool = True
    ) -> List[Dict]:
        """
        Search Economic Analysis reports with improved matching.
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
                text_conditions.append("key_themes::text ILIKE %s")
                params.append(like_query)

        # Search individual keywords with OR logic
        if keywords:
            for kw in keywords:
                if len(kw) >= 2:
                    like_kw = f"%{kw}%"
                    text_conditions.append("title ILIKE %s")
                    params.append(like_kw)
                    text_conditions.append("summary ILIKE %s")
                    params.append(like_kw)

        if text_conditions:
            conditions.append(f"({' OR '.join(text_conditions)})")

        # Filter conditions
        if issuer:
            conditions.append("issuer = %s")
            params.append(issuer)

        if category:
            conditions.append("category ILIKE %s")
            params.append(f"%{category}%")

        if region:
            conditions.append("region ILIKE %s")
            params.append(f"%{region}%")

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
                category, region, indicator_type, forecast_period,
                key_metrics, key_themes, pdf_link
            FROM {PG_TABLE}
            WHERE {where_clause}
            ORDER BY issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def get_recent_reports(self, days: int = 30, limit: int = 50) -> List[Dict]:
        """Get recent reports."""
        date_from = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return self.search_reports(date_from=date_from, limit=limit)

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

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) as total FROM {PG_TABLE}")
            total = cur.fetchone()['total']

            cur.execute(f"SELECT MIN(issue_date) as min_date, MAX(issue_date) as max_date FROM {PG_TABLE}")
            dates = cur.fetchone()

            cur.execute(f"SELECT COUNT(DISTINCT issuer) as issuers FROM {PG_TABLE}")
            issuers = cur.fetchone()['issuers']

            return {
                'total_reports': total,
                'date_range': {
                    'min': str(dates['min_date']) if dates['min_date'] else None,
                    'max': str(dates['max_date']) if dates['max_date'] else None,
                },
                'issuers': issuers,
            }


if __name__ == "__main__":
    client = PostgresClient()
    print("Stats:", client.get_stats())
    print("\nRecent reports:")
    for r in client.get_recent_reports(days=365, limit=5):
        print(f"  - {r['issue_date']} {r['issuer']}: {r['title'][:50]}...")
