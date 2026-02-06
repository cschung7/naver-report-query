"""
Neo4j Client for Investment Strategy Knowledge Graph
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
    """Neo4j client for Investment Strategy graph queries."""

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
                'StrategyReport', 'Issuer', 'Geography', 'Theme',
                'Sector', 'MarketOutlook', 'MarketRegime', 'AssetClass'
            ]
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[node_type.lower() + '_count'] = result.single()['count']

            return stats

    def get_reports_by_theme(self, theme: str, limit: int = 20) -> List[Dict]:
        """Get reports related to a theme."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[rel:HAS_THEME]->(t:Theme)
                WHERE t.name CONTAINS $theme OR toLower(t.name) CONTAINS toLower($theme)
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                OPTIONAL MATCH (r)-[:COVERS]->(g:Geography)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       t.name as theme,
                       rel.conviction as conviction,
                       o.name as outlook,
                       g.name as geography
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, theme=theme, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_sector(self, sector: str, limit: int = 20) -> List[Dict]:
        """Get reports recommending a sector."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[rel:RECOMMENDS_SECTOR]->(s:Sector)
                WHERE s.name CONTAINS $sector OR toLower(s.name) CONTAINS toLower($sector)
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       s.name as sector,
                       rel.recommendation as recommendation,
                       o.name as outlook
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, sector=sector, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_geography(self, geography: str, limit: int = 20) -> List[Dict]:
        """Get reports covering a geography."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:COVERS]->(g:Geography)
                WHERE g.name = $geography OR g.name CONTAINS $geography
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       g.name as geography,
                       o.name as outlook
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, geography=geography, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_outlook(self, outlook: str, limit: int = 20) -> List[Dict]:
        """Get reports with a specific market outlook."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                WHERE o.name = $outlook
                MATCH (r)-[:FROM_ISSUER]->(i:Issuer)
                OPTIONAL MATCH (r)-[:COVERS]->(g:Geography)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       o.name as outlook,
                       g.name as geography
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, outlook=outlook, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_reports_by_issuer(self, issuer: str, limit: int = 20) -> List[Dict]:
        """Get reports from a specific issuer."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:FROM_ISSUER]->(i:Issuer)
                WHERE i.name CONTAINS $issuer
                OPTIONAL MATCH (r)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                OPTIONAL MATCH (r)-[:COVERS]->(g:Geography)
                RETURN r.report_id as report_id,
                       r.title as title,
                       r.issue_date as issue_date,
                       i.name as issuer,
                       o.name as outlook,
                       g.name as geography
                ORDER BY r.issue_date DESC
                LIMIT $limit
            """, issuer=issuer, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def find_related_themes(self, theme: str, limit: int = 10) -> List[Dict]:
        """Find themes that appear together with the given theme."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Theme)<-[:HAS_THEME]-(r:StrategyReport)-[:HAS_THEME]->(t2:Theme)
                WHERE t1.name CONTAINS $theme AND t1 <> t2
                RETURN t2.name as related_theme, count(r) as co_occurrences
                ORDER BY co_occurrences DESC
                LIMIT $limit
            """, theme=theme, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def find_related_sectors(self, sector: str, limit: int = 10) -> List[Dict]:
        """Find sectors that are recommended together with the given sector."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s1:Sector)<-[:RECOMMENDS_SECTOR]-(r:StrategyReport)-[:RECOMMENDS_SECTOR]->(s2:Sector)
                WHERE s1.name CONTAINS $sector AND s1 <> s2
                RETURN s2.name as related_sector, count(r) as co_occurrences
                ORDER BY co_occurrences DESC
                LIMIT $limit
            """, sector=sector, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_issuer_themes(self, issuer: str) -> List[Dict]:
        """Get themes that an issuer frequently writes about."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Issuer)<-[:FROM_ISSUER]-(r:StrategyReport)-[:HAS_THEME]->(t:Theme)
                WHERE i.name CONTAINS $issuer
                RETURN t.name as theme, count(r) as mentions
                ORDER BY mentions DESC
                LIMIT 10
            """, issuer=issuer)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_top_themes(self, limit: int = 20) -> List[Dict]:
        """Get most frequently mentioned themes."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:HAS_THEME]->(t:Theme)
                RETURN t.name as theme, count(r) as mentions
                ORDER BY mentions DESC
                LIMIT $limit
            """, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_top_sectors(self, limit: int = 20) -> List[Dict]:
        """Get most frequently recommended sectors."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:RECOMMENDS_SECTOR]->(s:Sector)
                RETURN s.name as sector, count(r) as recommendations
                ORDER BY recommendations DESC
                LIMIT $limit
            """, limit=limit)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def get_outlook_by_geography(self, geography: str) -> List[Dict]:
        """Get market outlook distribution for a geography."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (r:StrategyReport)-[:COVERS]->(g:Geography)
                WHERE g.name = $geography OR g.name CONTAINS $geography
                MATCH (r)-[:HAS_OUTLOOK]->(o:MarketOutlook)
                RETURN o.name as outlook, count(r) as count
                ORDER BY count DESC
            """, geography=geography)

            return [_convert_neo4j_types(dict(record)) for record in result]

    def search_graph(self, query: str, limit: int = 20) -> Dict[str, List[Dict]]:
        """
        Comprehensive graph search across all node types.

        Returns reports found via themes, sectors, and direct matches.
        """
        results = {
            'by_theme': [],
            'by_sector': [],
            'by_geography': [],
            'related_themes': [],
            'related_sectors': [],
        }

        # Search themes
        theme_reports = self.get_reports_by_theme(query, limit=limit)
        if theme_reports:
            results['by_theme'] = theme_reports
            # Get related themes
            results['related_themes'] = self.find_related_themes(query, limit=5)

        # Search sectors
        sector_reports = self.get_reports_by_sector(query, limit=limit)
        if sector_reports:
            results['by_sector'] = sector_reports
            # Get related sectors
            results['related_sectors'] = self.find_related_sectors(query, limit=5)

        # Search geography
        geo_reports = self.get_reports_by_geography(query, limit=limit)
        if geo_reports:
            results['by_geography'] = geo_reports

        return results


if __name__ == "__main__":
    client = Neo4jClient()

    print("=" * 60)
    print("Neo4j Client Test")
    print("=" * 60)

    # Test stats
    print("\n📊 Graph Statistics:")
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Test theme search
    print("\n🎯 Reports with AI theme:")
    reports = client.get_reports_by_theme("AI", limit=3)
    for r in reports:
        print(f"  - {r['issue_date']}: {r['title'][:50]}...")

    # Test sector search
    print("\n📈 Reports recommending Semiconductors:")
    reports = client.get_reports_by_sector("Semiconductor", limit=3)
    for r in reports:
        print(f"  - {r['issue_date']}: {r['title'][:50]}...")

    # Test related themes
    print("\n🔗 Themes related to AI:")
    related = client.find_related_themes("AI", limit=5)
    for r in related:
        print(f"  - {r['related_theme']}: {r['co_occurrences']} co-occurrences")

    client.close()
