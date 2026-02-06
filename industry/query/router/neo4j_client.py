"""
Neo4j Client for Industry Analysis Knowledge Graph
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Dict, Optional, Any

try:
    from neo4j import GraphDatabase
    from neo4j.time import Date, DateTime
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    Date = DateTime = None

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def _convert_neo4j_types(record: Dict) -> Dict:
    """Convert Neo4j types to JSON-serializable Python types."""
    result = {}
    _neo4j_types = tuple(t for t in (Date, DateTime) if t is not None)
    for key, value in record.items():
        if _neo4j_types and isinstance(value, _neo4j_types):
            result[key] = str(value)
        elif value is None:
            result[key] = None
        else:
            result[key] = value
    return result


class Neo4jClient:
    """Neo4j client for Industry Analysis graph queries."""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            if not NEO4J_AVAILABLE:
                raise RuntimeError("neo4j package not installed")
            if self.uri and self.uri.lower() == 'none':
                raise RuntimeError("Neo4j disabled (NEO4J_URI=none)")
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        with self.driver.session() as session:
            stats = {}

            # Node counts
            node_types = [
                'IndustryReport', 'Issuer', 'Industry', 'CycleStage',
                'DemandTrend', 'Geography', 'Theme'
            ]
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[node_type.lower() + '_count'] = result.single()['count']

            return stats

    def get_reports_by_industry(self, industry: str, limit: int = 20) -> List[Dict]:
        """Get reports for a specific industry."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:COVERS_INDUSTRY]->(ind:Industry)
                WHERE ind.name CONTAINS $industry OR toLower(ind.name) CONTAINS toLower($industry)
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:HAS_CYCLE]->(c:CycleStage)
                OPTIONAL MATCH (r)-[:HAS_DEMAND]->(d:DemandTrend)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       ind.name as industry,
                       c.name as cycle_stage,
                       d.name as demand_trend
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, industry=industry, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_cycle(self, cycle_stage: str, limit: int = 20) -> List[Dict]:
        """Get reports by cycle stage."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:HAS_CYCLE]->(c:CycleStage)
                WHERE c.name = $cycle_stage OR c.name CONTAINS $cycle_stage
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:COVERS_INDUSTRY]->(ind:Industry)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       ind.name as industry,
                       c.name as cycle_stage
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, cycle_stage=cycle_stage, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_demand(self, demand_trend: str, limit: int = 20) -> List[Dict]:
        """Get reports by demand trend."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:HAS_DEMAND]->(d:DemandTrend)
                WHERE d.name = $demand_trend
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:COVERS_INDUSTRY]->(ind:Industry)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       ind.name as industry,
                       d.name as demand_trend
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, demand_trend=demand_trend, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_issuer(self, issuer: str, limit: int = 20) -> List[Dict]:
        """Get reports from a specific issuer."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:FROM_ISSUER]->(i:Issuer)
                WHERE i.name CONTAINS $issuer
                OPTIONAL MATCH (r)-[:COVERS_INDUSTRY]->(ind:Industry)
                OPTIONAL MATCH (r)-[:HAS_CYCLE]->(c:CycleStage)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       ind.name as industry,
                       c.name as cycle_stage
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, issuer=issuer, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def find_related_industries(self, industry: str, limit: int = 10) -> List[Dict]:
        """Find industries that appear together with the given industry."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i1:Industry)<-[:COVERS_INDUSTRY]-(r:IndustryReport)-[:COVERS_INDUSTRY]->(i2:Industry)
                WHERE i1.name CONTAINS $industry AND i1 <> i2
                RETURN i2.name as related_industry, count(r) as co_occurrences
                ORDER BY co_occurrences DESC
                LIMIT $limit
            """, industry=industry, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_top_industries(self, limit: int = 20) -> List[Dict]:
        """Get most frequently covered industries."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:COVERS_INDUSTRY]->(ind:Industry)
                RETURN ind.name as industry, count(r) as reports
                ORDER BY reports DESC
                LIMIT $limit
            """, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_top_issuers(self, limit: int = 20) -> List[Dict]:
        """Get most active issuers."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:FROM_ISSUER]->(i:Issuer)
                RETURN i.name as issuer, count(r) as reports
                ORDER BY reports DESC
                LIMIT $limit
            """, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_cycle_distribution(self) -> List[Dict]:
        """Get distribution of cycle stages."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:IndustryReport)-[:HAS_CYCLE]->(c:CycleStage)
                RETURN c.name as cycle_stage, count(r) as count
                ORDER BY count DESC
            """)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_issuer_industries(self, issuer: str) -> List[Dict]:
        """Get industries that an issuer frequently writes about."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Issuer)<-[:FROM_ISSUER]-(r:IndustryReport)-[:COVERS_INDUSTRY]->(ind:Industry)
                WHERE i.name CONTAINS $issuer
                RETURN ind.name as industry, count(r) as reports
                ORDER BY reports DESC
                LIMIT 10
            """, issuer=issuer)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def search_graph(self, query: str, limit: int = 20) -> Dict[str, List[Dict]]:
        """
        Comprehensive graph search across all node types.

        Returns reports found via industries, cycles, and direct matches.
        """
        results = {
            'by_industry': [],
            'by_cycle': [],
            'by_issuer': [],
            'related_industries': [],
        }

        # Search industries
        industry_reports = self.get_reports_by_industry(query, limit=limit)
        if industry_reports:
            results['by_industry'] = industry_reports
            # Get related industries
            results['related_industries'] = self.find_related_industries(query, limit=5)

        # Search by cycle stage keywords
        cycle_keywords = {
            '상승': 'UPCYCLE', '업사이클': 'UPCYCLE', '호황': 'UPCYCLE',
            '하락': 'DOWNCYCLE', '다운사이클': 'DOWNCYCLE', '불황': 'DOWNCYCLE',
            '정점': 'PEAK', '피크': 'PEAK',
            '저점': 'TROUGH', '바닥': 'TROUGH',
            '회복': 'RECOVERY',
        }
        for keyword, cycle in cycle_keywords.items():
            if keyword in query.lower():
                cycle_reports = self.get_reports_by_cycle(cycle, limit=limit)
                if cycle_reports:
                    results['by_cycle'] = cycle_reports
                break

        # Search by issuer
        issuer_reports = self.get_reports_by_issuer(query, limit=limit)
        if issuer_reports:
            results['by_issuer'] = issuer_reports

        return results


if __name__ == "__main__":
    client = Neo4jClient()

    print("=" * 60)
    print("Neo4j Client Test")
    print("=" * 60)

    # Test stats
    print("\nGraph Statistics:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Test industry search
    print("\nReports about Semiconductors:")
    reports = client.get_reports_by_industry("반도체", limit=3)
    for r in reports:
        print(f"  - {r['issue_date']}: {r['title'][:50]}...")

    # Test top industries
    print("\nTop Industries:")
    industries = client.get_top_industries(limit=5)
    for ind in industries:
        print(f"  - {ind['industry']}: {ind['reports']} reports")

    client.close()
