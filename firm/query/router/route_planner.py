"""
Route Planner

Makes routing decisions based on detected signals and entities.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .query_analyzer import ExtractedEntities
from .signal_detector import DetectedSignals, SignalType


class QueryIntent(Enum):
    """High-level query intent classification."""
    LOOKUP = "lookup"              # Simple company/report lookup
    FILTER = "filter"              # Structured filtering/screening
    RELATIONSHIP = "relationship"  # Graph/relationship query
    SEMANTIC_QA = "semantic_qa"    # Natural language Q&A
    COMPARISON = "comparison"      # Compare multiple entities
    RESEARCH = "research"          # Comprehensive research (all DBs)


class DatabasePriority(Enum):
    """Database query priority levels."""
    PRIMARY = "primary"      # Must query, main source
    SECONDARY = "secondary"  # Should query for enrichment
    OPTIONAL = "optional"    # Query if time permits
    SKIP = "skip"           # Don't query


@dataclass
class RouteDecision:
    """Container for routing decision."""
    intent: QueryIntent
    postgresql: DatabasePriority = DatabasePriority.SKIP
    neo4j: DatabasePriority = DatabasePriority.SKIP
    vector_db: DatabasePriority = DatabasePriority.SKIP

    # Query parameters for each DB
    pg_params: Dict[str, Any] = field(default_factory=dict)
    neo4j_params: Dict[str, Any] = field(default_factory=dict)
    vector_params: Dict[str, Any] = field(default_factory=dict)

    # Execution strategy
    parallel: bool = False  # Can DBs be queried in parallel?
    sequence: List[str] = field(default_factory=list)  # Query order if sequential

    # Explanation
    reasoning: str = ""

    def get_active_databases(self) -> List[str]:
        """Get list of databases that should be queried."""
        active = []
        if self.postgresql != DatabasePriority.SKIP:
            active.append("postgresql")
        if self.neo4j != DatabasePriority.SKIP:
            active.append("neo4j")
        if self.vector_db != DatabasePriority.SKIP:
            active.append("vector_db")
        return active

    def is_hybrid(self) -> bool:
        """Check if multiple databases will be queried."""
        return len(self.get_active_databases()) > 1


class RoutePlanner:
    """
    Plans query routing based on signals and entities.

    Routing Rules:
    1. Structured filters → PostgreSQL primary
    2. Relationship queries → Neo4j primary
    3. Semantic questions → Vector DB primary
    4. Multiple signals → Hybrid with appropriate priority
    """

    # Thresholds for database activation
    POSTGRESQL_THRESHOLD = 0.3
    NEO4J_THRESHOLD = 0.5
    VECTOR_THRESHOLD = 0.5

    def plan(
        self,
        query: str,
        entities: ExtractedEntities,
        signals: DetectedSignals
    ) -> RouteDecision:
        """
        Create routing decision based on analysis.

        Args:
            query: Original query string
            entities: Extracted entities
            signals: Detected signals

        Returns:
            RouteDecision with database priorities and parameters
        """
        # Calculate scores
        pg_score = signals.postgresql_score()
        neo4j_score = signals.neo4j_score()
        vector_score = signals.vector_score()

        # Classify intent
        intent = self._classify_intent(entities, signals, pg_score, neo4j_score, vector_score)

        # Create decision based on intent
        decision = self._create_decision(intent, entities, signals)

        # Set parameters
        self._set_postgresql_params(decision, entities)
        self._set_neo4j_params(decision, entities, signals)
        self._set_vector_params(decision, query, entities)

        # Determine execution strategy
        self._set_execution_strategy(decision, intent)

        # Add reasoning
        decision.reasoning = self._generate_reasoning(
            intent, pg_score, neo4j_score, vector_score, entities
        )

        return decision

    def _classify_intent(
        self,
        entities: ExtractedEntities,
        signals: DetectedSignals,
        pg_score: float,
        neo4j_score: float,
        vector_score: float
    ) -> QueryIntent:
        """Classify the high-level query intent."""
        # Check for comparison first (multiple entities)
        if SignalType.COMPARISON in signals.signals:
            return QueryIntent.COMPARISON

        # Check for relationship queries
        if neo4j_score > self.NEO4J_THRESHOLD:
            if SignalType.SUPPLY_CHAIN in signals.signals:
                return QueryIntent.RELATIONSHIP
            if SignalType.COMPETITOR in signals.signals:
                return QueryIntent.RELATIONSHIP
            if SignalType.RELATIONSHIP in signals.signals and neo4j_score > pg_score:
                return QueryIntent.RELATIONSHIP

        # Check for semantic Q&A
        if vector_score > self.VECTOR_THRESHOLD:
            if SignalType.SEMANTIC_QUESTION in signals.signals:
                return QueryIntent.SEMANTIC_QA
            if SignalType.CONCEPTUAL_QUERY in signals.signals and vector_score > pg_score:
                return QueryIntent.SEMANTIC_QA

        # Check for screening/filtering
        if pg_score > self.POSTGRESQL_THRESHOLD:
            if entities.valuation_keywords or entities.growth_keywords:
                return QueryIntent.FILTER
            if entities.sectors and not entities.companies:
                return QueryIntent.FILTER

        # Simple lookup
        if entities.companies and pg_score > 0:
            return QueryIntent.LOOKUP

        # Research mode (all DBs) for complex queries
        if pg_score > 0 and (neo4j_score > 0 or vector_score > 0):
            return QueryIntent.RESEARCH

        # Default to semantic if no structured elements
        if not entities.has_structured_filters():
            return QueryIntent.SEMANTIC_QA

        return QueryIntent.LOOKUP

    def _create_decision(
        self,
        intent: QueryIntent,
        entities: ExtractedEntities,
        signals: DetectedSignals
    ) -> RouteDecision:
        """Create RouteDecision based on intent."""
        decision = RouteDecision(intent=intent)

        if intent == QueryIntent.LOOKUP:
            # Simple lookup: PostgreSQL primary
            decision.postgresql = DatabasePriority.PRIMARY
            decision.neo4j = DatabasePriority.SECONDARY  # For context
            decision.vector_db = DatabasePriority.OPTIONAL

        elif intent == QueryIntent.FILTER:
            # Screening: PostgreSQL primary, Neo4j for themes
            decision.postgresql = DatabasePriority.PRIMARY
            decision.neo4j = DatabasePriority.SECONDARY
            decision.vector_db = DatabasePriority.OPTIONAL

        elif intent == QueryIntent.RELATIONSHIP:
            # Relationship: Neo4j primary, PostgreSQL for details
            decision.neo4j = DatabasePriority.PRIMARY
            decision.postgresql = DatabasePriority.SECONDARY
            decision.vector_db = DatabasePriority.OPTIONAL

        elif intent == QueryIntent.SEMANTIC_QA:
            # Semantic Q&A: Vector DB primary
            decision.vector_db = DatabasePriority.PRIMARY
            decision.postgresql = DatabasePriority.SECONDARY  # For filtering
            decision.neo4j = DatabasePriority.OPTIONAL

        elif intent == QueryIntent.COMPARISON:
            # Comparison: All DBs important
            decision.postgresql = DatabasePriority.PRIMARY
            decision.neo4j = DatabasePriority.PRIMARY
            decision.vector_db = DatabasePriority.SECONDARY

        elif intent == QueryIntent.RESEARCH:
            # Research: All DBs
            decision.postgresql = DatabasePriority.PRIMARY
            decision.neo4j = DatabasePriority.PRIMARY
            decision.vector_db = DatabasePriority.PRIMARY

        return decision

    def _set_postgresql_params(
        self,
        decision: RouteDecision,
        entities: ExtractedEntities
    ):
        """Set PostgreSQL query parameters."""
        if decision.postgresql == DatabasePriority.SKIP:
            return

        params = {}

        # Company filter
        if entities.companies:
            if len(entities.companies) == 1:
                params['company'] = entities.companies[0]
            else:
                params['companies'] = entities.companies

        # Issuer filter
        if entities.issuers:
            params['issuer'] = entities.issuers[0]

        # Sector filter
        if entities.sectors:
            params['sector'] = entities.sectors[0]

        # Date filter
        if entities.date_range:
            params['date_from'] = entities.date_range.get('from')
            params['date_to'] = entities.date_range.get('to')

        # Valuation filter
        if entities.valuation_keywords:
            params['valuation_regime'] = entities.valuation_keywords[0]

        # Growth filter
        if entities.growth_keywords:
            params['growth_regime'] = entities.growth_keywords[0]

        decision.pg_params = params

    def _set_neo4j_params(
        self,
        decision: RouteDecision,
        entities: ExtractedEntities,
        signals: DetectedSignals
    ):
        """Set Neo4j query parameters."""
        if decision.neo4j == DatabasePriority.SKIP:
            return

        params = {}

        # Companies for relationship queries
        if entities.companies:
            params['companies'] = entities.companies

        # Query type based on signals
        if SignalType.SUPPLY_CHAIN in signals.signals:
            params['query_type'] = 'supply_chain'
        elif SignalType.COMPETITOR in signals.signals:
            params['query_type'] = 'competitors'
        elif SignalType.THEME_GRAPH in signals.signals:
            params['query_type'] = 'themes'
        else:
            params['query_type'] = 'context_expansion'

        # Themes
        if entities.themes:
            params['themes'] = entities.themes

        decision.neo4j_params = params

    def _set_vector_params(
        self,
        decision: RouteDecision,
        query: str,
        entities: ExtractedEntities
    ):
        """Set Vector DB query parameters."""
        if decision.vector_db == DatabasePriority.SKIP:
            return

        params = {
            'question': query,
            'n_results': 10,
        }

        # Company filter for vector search
        if entities.companies:
            params['company'] = entities.companies[0]

        decision.vector_params = params

    def _set_execution_strategy(self, decision: RouteDecision, intent: QueryIntent):
        """Set execution strategy (parallel vs sequential)."""
        active_dbs = decision.get_active_databases()

        if len(active_dbs) <= 1:
            decision.parallel = False
            decision.sequence = active_dbs
            return

        # Determine if parallel execution is possible
        if intent in [QueryIntent.LOOKUP, QueryIntent.FILTER]:
            # PostgreSQL first, then others can run in parallel
            decision.parallel = False
            decision.sequence = ['postgresql', 'neo4j', 'vector_db']

        elif intent == QueryIntent.RELATIONSHIP:
            # Neo4j first to get related companies, then PostgreSQL
            decision.parallel = False
            decision.sequence = ['neo4j', 'postgresql', 'vector_db']

        elif intent == QueryIntent.SEMANTIC_QA:
            # PostgreSQL first for filtering, then Vector
            decision.parallel = False
            decision.sequence = ['postgresql', 'vector_db', 'neo4j']

        elif intent in [QueryIntent.COMPARISON, QueryIntent.RESEARCH]:
            # All DBs can run in parallel
            decision.parallel = True
            decision.sequence = active_dbs

        # Remove skipped DBs from sequence
        decision.sequence = [db for db in decision.sequence if db in active_dbs]

    def _generate_reasoning(
        self,
        intent: QueryIntent,
        pg_score: float,
        neo4j_score: float,
        vector_score: float,
        entities: ExtractedEntities
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = [f"Intent: {intent.value}"]
        parts.append(f"Scores - PG:{pg_score:.1f} Neo4j:{neo4j_score:.1f} Vector:{vector_score:.1f}")

        if entities.companies:
            parts.append(f"Companies: {', '.join(entities.companies)}")
        if entities.sectors:
            parts.append(f"Sectors: {', '.join(entities.sectors)}")
        if entities.themes:
            parts.append(f"Themes: {', '.join(entities.themes)}")

        return " | ".join(parts)


# Test
if __name__ == "__main__":
    from .query_analyzer import QueryAnalyzer
    from .signal_detector import SignalDetector

    analyzer = QueryAnalyzer()
    detector = SignalDetector()
    planner = RoutePlanner()

    test_queries = [
        "삼성전자 최근 리포트",
        "테슬라 공급망 관련 한국 배터리 기업",
        "SK하이닉스 HBM 전망은 어떤가요?",
        "삼성전자와 SK하이닉스 비교",
        "저평가된 반도체 고성장주",
        "AI 반도체 시장 리스크 분석",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")

        entities = analyzer.analyze(q)
        signals = detector.detect(q, entities)
        decision = planner.plan(q, entities, signals)

        print(f"\nRouting Decision:")
        print(f"  Intent: {decision.intent.value}")
        print(f"  PostgreSQL: {decision.postgresql.value}")
        print(f"  Neo4j: {decision.neo4j.value}")
        print(f"  Vector DB: {decision.vector_db.value}")
        print(f"  Parallel: {decision.parallel}")
        print(f"  Sequence: {decision.sequence}")
        print(f"  Reasoning: {decision.reasoning}")

        if decision.pg_params:
            print(f"  PG Params: {decision.pg_params}")
        if decision.neo4j_params:
            print(f"  Neo4j Params: {decision.neo4j_params}")
        if decision.vector_params:
            print(f"  Vector Params: {decision.vector_params}")
