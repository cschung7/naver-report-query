"""
Neo4j Query Client

Query interface for the claim-centric knowledge graph.
"""
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from config.settings import get_settings


class Neo4jClient:
    """Query client for Neo4j knowledge graph."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._driver = None

    @property
    def driver(self):
        """Lazy driver initialization."""
        if self._driver is None:
            if not NEO4J_AVAILABLE:
                raise RuntimeError("neo4j package not installed")
            uri = self.settings.neo4j_uri
            if uri and uri.lower() == 'none':
                raise RuntimeError("Neo4j disabled (NEO4J_URI=none)")
            self._driver = GraphDatabase.driver(
                uri,
                auth=(
                    self.settings.neo4j_user,
                    self.settings.neo4j_password
                )
            )
        return self._driver

    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def get_session(self):
        """Get a session context manager."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def get_company_claims(
        self,
        company: str,
        claim_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get claims about a company.

        Args:
            company: Company name
            claim_types: Filter by claim types
            limit: Max results

        Returns:
            List of claim dictionaries
        """
        type_filter = ""
        params = {'company': company, 'limit': limit}

        if claim_types:
            type_filter = "AND cl.claim_type IN $types"
            params['types'] = claim_types

        query = f"""
            MATCH (cl:Claim)-[:ABOUT]->(c:Company {{name: $company}})
            WHERE cl.claim_text IS NOT NULL
            {type_filter}
            OPTIONAL MATCH (cl)-[:FROM_REPORT]->(r:Report)
            OPTIONAL MATCH (cl)-[:HAS_CONTEXT]->(ctx)
            RETURN
                cl.claim_id AS claim_id,
                cl.claim_type AS claim_type,
                cl.claim_text AS claim_text,
                cl.confidence AS confidence,
                cl.issue_date AS issue_date,
                r.report_id AS report_id,
                r.issuer AS issuer,
                collect(DISTINCT ctx.name) AS context_nodes
            ORDER BY cl.issue_date DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_broker_claims(
        self,
        broker: str,
        company: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get claims from a specific broker."""
        company_filter = ""
        params = {'broker': broker, 'limit': limit}

        if company:
            company_filter = "AND c.name = $company"
            params['company'] = company

        query = f"""
            MATCH (b:Broker {{name: $broker}})-[:PUBLISHED]->(r:Report)
            MATCH (cl:Claim)-[:FROM_REPORT]->(r)
            MATCH (cl)-[:ABOUT]->(c:Company)
            WHERE cl.claim_text IS NOT NULL
            {company_filter}
            OPTIONAL MATCH (cl)-[:HAS_CONTEXT]->(ctx)
            RETURN
                cl.claim_id AS claim_id,
                cl.claim_type AS claim_type,
                cl.claim_text AS claim_text,
                c.name AS company,
                cl.issue_date AS issue_date,
                r.report_id AS report_id,
                collect(DISTINCT ctx.name) AS context_nodes
            ORDER BY cl.issue_date DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_claims_by_context(
        self,
        context_value: str,
        claim_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get claims with a specific context value.

        Args:
            context_value: Context node value (e.g., "DEEP_DISCOUNT", "HIGH_GROWTH")
            claim_type: Optional claim type filter
            limit: Max results
        """
        type_filter = ""
        params = {'context': context_value, 'limit': limit}

        if claim_type:
            type_filter = "AND cl.claim_type = $type"
            params['type'] = claim_type

        query = f"""
            MATCH (cl:Claim)-[:HAS_CONTEXT]->(ctx {{name: $context}})
            WHERE cl.claim_text IS NOT NULL
            {type_filter}
            MATCH (cl)-[:ABOUT]->(c:Company)
            OPTIONAL MATCH (cl)-[:FROM_REPORT]->(r:Report)
            RETURN
                cl.claim_id AS claim_id,
                cl.claim_type AS claim_type,
                cl.claim_text AS claim_text,
                c.name AS company,
                cl.issue_date AS issue_date,
                r.report_id AS report_id,
                r.issuer AS issuer
            ORDER BY cl.issue_date DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_related_claims(
        self,
        claim_id: str,
        depth: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get claims related to a given claim through shared context.

        Args:
            claim_id: Source claim ID
            depth: Max relationship depth
            limit: Max results
        """
        query = """
            MATCH (source:Claim {claim_id: $claim_id})
            MATCH (source)-[:HAS_CONTEXT]->(ctx)<-[:HAS_CONTEXT]-(related:Claim)
            WHERE source <> related
            MATCH (related)-[:ABOUT]->(c:Company)
            OPTIONAL MATCH (related)-[:FROM_REPORT]->(r:Report)
            RETURN DISTINCT
                related.claim_id AS claim_id,
                related.claim_type AS claim_type,
                related.claim_text AS claim_text,
                c.name AS company,
                ctx.name AS shared_context,
                related.issue_date AS issue_date,
                r.report_id AS report_id
            ORDER BY related.issue_date DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, claim_id=claim_id, limit=limit)
            return [dict(r) for r in result]

    def get_company_context_summary(
        self,
        company: str
    ) -> Dict[str, Any]:
        """
        Get a summary of all context nodes for a company.

        Returns counts and recent values for each context type.
        """
        query = """
            MATCH (c:Company {name: $company})
            MATCH (cl:Claim)-[:ABOUT]->(c)
            OPTIONAL MATCH (cl)-[:HAS_CONTEXT]->(ctx)
            WITH cl, ctx, labels(ctx)[0] AS ctx_type
            RETURN
                ctx_type,
                collect(DISTINCT ctx.name) AS values,
                count(DISTINCT cl) AS claim_count
            ORDER BY claim_count DESC
        """

        with self.get_session() as session:
            result = session.run(query, company=company)

            summary = {
                'company': company,
                'context_types': {}
            }

            for r in result:
                ctx_type = r['ctx_type']
                if ctx_type:
                    summary['context_types'][ctx_type] = {
                        'values': r['values'],
                        'claim_count': r['claim_count']
                    }

            return summary

    def find_companies_by_criteria(
        self,
        valuation_regime: Optional[str] = None,
        growth_regime: Optional[str] = None,
        cycle_stage: Optional[str] = None,
        sector: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find companies matching multiple criteria.

        Args:
            valuation_regime: e.g., "DEEP_DISCOUNT", "FAIR_VALUE"
            growth_regime: e.g., "HIGH_GROWTH", "MATURE"
            cycle_stage: e.g., "RECOVERY", "EXPANSION"
            sector: e.g., "반도체", "자동차"
            limit: Max results
        """
        matches = ["MATCH (c:Company)"]
        where_conditions = []
        params = {'limit': limit}

        if valuation_regime:
            matches.append(
                "MATCH (cl1:Claim)-[:ABOUT]->(c), "
                "(cl1)-[:HAS_CONTEXT]->(:ValuationRegime {name: $val_regime})"
            )
            params['val_regime'] = valuation_regime

        if growth_regime:
            matches.append(
                "MATCH (cl2:Claim)-[:ABOUT]->(c), "
                "(cl2)-[:HAS_CONTEXT]->(:GrowthDriver {name: $growth_regime})"
            )
            params['growth_regime'] = growth_regime

        if cycle_stage:
            matches.append(
                "MATCH (cl3:Claim)-[:ABOUT]->(c), "
                "(cl3)-[:HAS_CONTEXT]->(:CycleStage {name: $cycle})"
            )
            params['cycle'] = cycle_stage

        if sector:
            matches.append("MATCH (c)-[:IN_SECTOR]->(:Sector {name: $sector})")
            params['sector'] = sector

        query = f"""
            {' '.join(matches)}
            OPTIONAL MATCH (r:Report)-[:COVERS]->(c)
            RETURN DISTINCT
                c.name AS company,
                c.ticker AS ticker,
                c.sector AS sector,
                count(DISTINCT r) AS report_count
            ORDER BY report_count DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, **params)
            return [dict(r) for r in result]

    def get_analyst_coverage(
        self,
        analyst: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get companies and claims by a specific analyst."""
        query = """
            MATCH (a:Analyst {name: $analyst})-[:AUTHORED]->(r:Report)
            MATCH (r)-[:COVERS]->(c:Company)
            OPTIONAL MATCH (cl:Claim)-[:FROM_REPORT]->(r)
            RETURN DISTINCT
                c.name AS company,
                r.report_id AS report_id,
                r.issue_date AS issue_date,
                count(cl) AS claim_count
            ORDER BY r.issue_date DESC
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(query, analyst=analyst, limit=limit)
            return [dict(r) for r in result]

    def expand_context(
        self,
        report_ids: List[str],
        max_related: int = 10
    ) -> Dict[str, Any]:
        """
        Expand context for a set of reports.

        Returns claims and related context for the given reports.
        """
        query = """
            UNWIND $report_ids AS rid
            MATCH (r:Report {report_id: rid})
            MATCH (cl:Claim)-[:FROM_REPORT]->(r)
            MATCH (cl)-[:ABOUT]->(c:Company)
            OPTIONAL MATCH (cl)-[:HAS_CONTEXT]->(ctx)
            RETURN
                r.report_id AS report_id,
                cl.claim_id AS claim_id,
                cl.claim_type AS claim_type,
                cl.claim_text AS claim_text,
                c.name AS company,
                collect(DISTINCT ctx.name) AS context_nodes
            LIMIT $limit
        """

        with self.get_session() as session:
            result = session.run(
                query,
                report_ids=report_ids,
                limit=max_related * len(report_ids)
            )

            claims_by_report = {}
            for r in result:
                rid = r['report_id']
                if rid not in claims_by_report:
                    claims_by_report[rid] = []
                claims_by_report[rid].append(dict(r))

            return claims_by_report


# CLI for testing
if __name__ == "__main__":
    import sys

    client = Neo4jClient()

    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'company':
            company = sys.argv[2] if len(sys.argv) > 2 else "삼성전자"
            claims = client.get_company_claims(company, limit=10)
            print(f"\nClaims about {company} ({len(claims)}):")
            for c in claims:
                print(f"  [{c['claim_type']}] {c['claim_text'][:60]}...")
                print(f"    context: {c['context_nodes']}")

        elif len(sys.argv) > 1 and sys.argv[1] == 'context':
            context = sys.argv[2] if len(sys.argv) > 2 else "DEEP_DISCOUNT"
            claims = client.get_claims_by_context(context, limit=10)
            print(f"\nClaims with context '{context}' ({len(claims)}):")
            for c in claims:
                print(f"  {c['company']} | {c['claim_text'][:50]}...")

        elif len(sys.argv) > 1 and sys.argv[1] == 'search':
            results = client.find_companies_by_criteria(
                valuation_regime='DEEP_DISCOUNT',
                limit=10
            )
            print(f"\nCompanies with DEEP_DISCOUNT ({len(results)}):")
            for r in results:
                print(f"  {r['company']} | reports: {r['report_count']}")

        elif len(sys.argv) > 1 and sys.argv[1] == 'summary':
            company = sys.argv[2] if len(sys.argv) > 2 else "삼성전자"
            summary = client.get_company_context_summary(company)
            print(f"\nContext summary for {company}:")
            for ctx_type, data in summary['context_types'].items():
                print(f"  {ctx_type}: {data['claim_count']} claims")
                print(f"    values: {data['values'][:5]}")

        else:
            print("Usage: python -m query.neo4j_client [company <name>|context <value>|search|summary <name>]")

    finally:
        client.close()
