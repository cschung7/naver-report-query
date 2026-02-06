"""
PostgreSQL Client for Investment Strategy Reports
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
    """PostgreSQL client for Investment Strategy queries."""

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
        strategy_type: str = None,
        market_outlook: str = None,
        market_regime: str = None,
        geography: str = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 20,
        broad_search: bool = True
    ) -> List[Dict]:
        """
        Search Investment Strategy reports with improved matching.

        Args:
            query: Text search query
            keywords: List of individual keywords to search
            issuer: Filter by broker/issuer
            strategy_type: Filter by strategy type (ANNUAL_OUTLOOK, WEEKLY_REVIEW, etc.)
            market_outlook: Filter by market outlook (BULLISH, BEARISH, etc.)
            market_regime: Filter by market regime (RISK_ON, RISK_OFF, etc.)
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
                text_conditions.append("key_thesis::text ILIKE %s")
                params.append(like_query)
                text_conditions.append("do_list::text ILIKE %s")
                params.append(like_query)
                text_conditions.append("avoid_list::text ILIKE %s")
                params.append(like_query)
                text_conditions.append("watch_list::text ILIKE %s")
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
                        text_conditions.append("key_thesis::text ILIKE %s")
                        params.append(like_kw)
                        text_conditions.append("do_list::text ILIKE %s")
                        params.append(like_kw)
                        text_conditions.append("avoid_list::text ILIKE %s")
                        params.append(like_kw)

        if text_conditions:
            conditions.append(f"({' OR '.join(text_conditions)})")

        # Filter conditions (soft matching for geography)
        if issuer:
            conditions.append("issuer = %s")
            params.append(issuer)

        if strategy_type:
            conditions.append("strategy_type = %s")
            params.append(strategy_type)

        if market_outlook:
            # Soft match - also include NULL (not specified)
            conditions.append("(market_outlook = %s OR market_outlook IS NULL)")
            params.append(market_outlook)

        if market_regime:
            conditions.append("(market_regime = %s OR market_regime IS NULL)")
            params.append(market_regime)

        if geography:
            # Soft match - include GLOBAL as fallback
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
                strategy_type, time_horizon, geography,
                market_outlook, conviction_level, market_stage, market_regime,
                key_thesis, equity_allocation, cash_allocation, risk_posture,
                gdp_trend, do_list, avoid_list, watch_list,
                extraction_confidence
            FROM {PG_TABLE}
            WHERE {where_clause}
            ORDER BY issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_by_sector(self, sector_keywords: List[str], limit: int = 20) -> List[Dict]:
        """Search reports by sector recommendations."""
        conditions = []
        params = []

        for kw in sector_keywords:
            conditions.append("s.sector ILIKE %s OR s.rationale ILIKE %s")
            params.extend([f"%{kw}%", f"%{kw}%"])

        sql = f"""
            SELECT DISTINCT r.report_id, r.title, r.issuer, r.issue_date, r.author, r.summary,
                r.strategy_type, r.time_horizon, r.geography,
                r.market_outlook, r.conviction_level, r.market_stage, r.market_regime,
                r.key_thesis, r.equity_allocation, r.cash_allocation, r.risk_posture,
                r.gdp_trend, r.do_list, r.avoid_list, r.watch_list,
                r.extraction_confidence
            FROM {PG_TABLE} r
            JOIN invest_sector_recs s ON r.report_id = s.report_id
            WHERE {' OR '.join(conditions)}
            ORDER BY r.issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_by_theme(self, theme_keywords: List[str], limit: int = 20) -> List[Dict]:
        """Search reports by thematic plays."""
        conditions = []
        params = []

        for kw in theme_keywords:
            conditions.append("t.theme_name ILIKE %s OR t.rationale ILIKE %s")
            params.extend([f"%{kw}%", f"%{kw}%"])

        sql = f"""
            SELECT DISTINCT r.report_id, r.title, r.issuer, r.issue_date, r.author, r.summary,
                r.strategy_type, r.time_horizon, r.geography,
                r.market_outlook, r.conviction_level, r.market_stage, r.market_regime,
                r.key_thesis, r.equity_allocation, r.cash_allocation, r.risk_posture,
                r.gdp_trend, r.do_list, r.avoid_list, r.watch_list,
                r.extraction_confidence
            FROM {PG_TABLE} r
            JOIN invest_themes t ON r.report_id = t.report_id
            WHERE {' OR '.join(conditions)}
            ORDER BY r.issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def search_by_allocation(self, asset_keywords: List[str], limit: int = 20) -> List[Dict]:
        """Search reports by asset allocation."""
        conditions = []
        params = []

        for kw in asset_keywords:
            conditions.append("a.asset_category ILIKE %s OR a.rationale ILIKE %s")
            params.extend([f"%{kw}%", f"%{kw}%"])

        sql = f"""
            SELECT DISTINCT r.report_id, r.title, r.issuer, r.issue_date, r.author, r.summary,
                r.strategy_type, r.time_horizon, r.geography,
                r.market_outlook, r.conviction_level, r.market_stage, r.market_regime,
                r.key_thesis, r.equity_allocation, r.cash_allocation, r.risk_posture,
                r.gdp_trend, r.do_list, r.avoid_list, r.watch_list,
                r.extraction_confidence
            FROM {PG_TABLE} r
            JOIN invest_allocations a ON r.report_id = a.report_id
            WHERE {' OR '.join(conditions)}
            ORDER BY r.issue_date DESC
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

    def get_by_outlook(self, outlook: str, limit: int = 20) -> List[Dict]:
        """Get reports by market outlook."""
        return self.search_reports(market_outlook=outlook, limit=limit)

    def get_by_regime(self, regime: str, limit: int = 20) -> List[Dict]:
        """Get reports by market regime."""
        return self.search_reports(market_regime=regime, limit=limit)

    def get_issuers(self) -> List[Dict]:
        """Get list of issuers with stats."""
        sql = """
            SELECT issuer, COUNT(*) as count,
                   MAX(issue_date) as latest_report
            FROM invest_strategy_reports
            GROUP BY issuer
            ORDER BY count DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    def get_allocations(self, report_id: str) -> List[Dict]:
        """Get asset allocations for a report."""
        sql = """
            SELECT asset_category, allocation_weight, rationale
            FROM invest_allocations
            WHERE report_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (report_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_themes(self, report_id: str = None, theme_name: str = None, limit: int = 50) -> List[Dict]:
        """Get thematic plays."""
        conditions = []
        params = []

        if report_id:
            conditions.append("t.report_id = %s")
            params.append(report_id)

        if theme_name:
            conditions.append("t.theme_name ILIKE %s")
            params.append(f"%{theme_name}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT t.theme_name, t.conviction, t.time_horizon, t.rationale,
                   r.title, r.issuer, r.issue_date
            FROM invest_themes t
            JOIN {PG_TABLE} r ON t.report_id = r.report_id
            WHERE {where_clause}
            ORDER BY r.issue_date DESC
            LIMIT %s
        """
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
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

            cur.execute(f"""
                SELECT strategy_type, COUNT(*) as count
                FROM {PG_TABLE}
                WHERE strategy_type IS NOT NULL
                GROUP BY strategy_type
                ORDER BY count DESC
                LIMIT 10
            """)
            strategy_types = [dict(row) for row in cur.fetchall()]

            return {
                'total_reports': total,
                'date_range': {
                    'min': str(dates['min_date']) if dates['min_date'] else None,
                    'max': str(dates['max_date']) if dates['max_date'] else None,
                },
                'issuers': issuers,
                'strategy_types': strategy_types,
            }


if __name__ == "__main__":
    client = PostgresClient()
    print("Stats:", client.get_stats())
    print("\nRecent reports:")
    for r in client.get_recent_reports(days=365, limit=5):
        print(f"  - {r['issue_date']} {r['issuer']}: {r['title'][:50]}...")
