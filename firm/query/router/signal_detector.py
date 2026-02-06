"""
Signal Detector

Detects signals that indicate which database(s) should be queried.
"""
import re
from typing import Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

from .query_analyzer import ExtractedEntities


class SignalType(Enum):
    """Types of signals detected in queries."""
    # PostgreSQL signals
    COMPANY_LOOKUP = "company_lookup"
    ISSUER_FILTER = "issuer_filter"
    DATE_FILTER = "date_filter"
    SECTOR_FILTER = "sector_filter"
    VALUATION_FILTER = "valuation_filter"
    GROWTH_FILTER = "growth_filter"

    # Neo4j signals
    RELATIONSHIP = "relationship"
    SUPPLY_CHAIN = "supply_chain"
    COMPETITOR = "competitor"
    THEME_GRAPH = "theme_graph"
    CONTEXT_EXPANSION = "context_expansion"

    # Vector DB signals
    SEMANTIC_QUESTION = "semantic_question"
    CONCEPTUAL_QUERY = "conceptual_query"
    OPINION_REQUEST = "opinion_request"
    ANALYSIS_REQUEST = "analysis_request"
    COMPARISON = "comparison"


@dataclass
class DetectedSignals:
    """Container for detected signals with confidence scores."""
    signals: Dict[SignalType, float] = field(default_factory=dict)
    raw_query: str = ""

    def add_signal(self, signal: SignalType, confidence: float = 1.0):
        """Add or update a signal with confidence score."""
        current = self.signals.get(signal, 0)
        self.signals[signal] = max(current, confidence)

    def get_postgresql_signals(self) -> Dict[SignalType, float]:
        """Get signals indicating PostgreSQL should be used."""
        pg_signals = {
            SignalType.COMPANY_LOOKUP,
            SignalType.ISSUER_FILTER,
            SignalType.DATE_FILTER,
            SignalType.SECTOR_FILTER,
            SignalType.VALUATION_FILTER,
            SignalType.GROWTH_FILTER,
        }
        return {k: v for k, v in self.signals.items() if k in pg_signals}

    def get_neo4j_signals(self) -> Dict[SignalType, float]:
        """Get signals indicating Neo4j should be used."""
        neo4j_signals = {
            SignalType.RELATIONSHIP,
            SignalType.SUPPLY_CHAIN,
            SignalType.COMPETITOR,
            SignalType.THEME_GRAPH,
            SignalType.CONTEXT_EXPANSION,
        }
        return {k: v for k, v in self.signals.items() if k in neo4j_signals}

    def get_vector_signals(self) -> Dict[SignalType, float]:
        """Get signals indicating Vector DB should be used."""
        vector_signals = {
            SignalType.SEMANTIC_QUESTION,
            SignalType.CONCEPTUAL_QUERY,
            SignalType.OPINION_REQUEST,
            SignalType.ANALYSIS_REQUEST,
            SignalType.COMPARISON,
        }
        return {k: v for k, v in self.signals.items() if k in vector_signals}

    def postgresql_score(self) -> float:
        """Calculate overall PostgreSQL relevance score."""
        signals = self.get_postgresql_signals()
        return sum(signals.values()) if signals else 0

    def neo4j_score(self) -> float:
        """Calculate overall Neo4j relevance score."""
        signals = self.get_neo4j_signals()
        return sum(signals.values()) if signals else 0

    def vector_score(self) -> float:
        """Calculate overall Vector DB relevance score."""
        signals = self.get_vector_signals()
        return sum(signals.values()) if signals else 0


class SignalDetector:
    """
    Detects signals in queries that indicate which database(s) to use.

    Signal Categories:
    1. PostgreSQL: Structured filters (company, date, sector, etc.)
    2. Neo4j: Relationship queries (supply chain, competitors, themes)
    3. Vector DB: Semantic/conceptual queries (why, how, analysis)
    """

    # Relationship keywords → Neo4j
    RELATIONSHIP_KEYWORDS = [
        "관련", "연결", "연관", "관계",
        "영향", "수혜", "수혜주", "피해주",
        "연동", "상관", "파급",
    ]

    # Supply chain keywords → Neo4j
    SUPPLY_CHAIN_KEYWORDS = [
        "공급망", "공급업체", "협력사", "파트너",
        "납품", "공급", "밸류체인", "가치사슬",
        "upstream", "downstream", "고객사",
    ]

    # Competitor keywords → Neo4j
    COMPETITOR_KEYWORDS = [
        "경쟁사", "경쟁", "라이벌", "대체",
        "경쟁업체", "vs", "대비",
    ]

    # Question words → Vector DB (semantic)
    QUESTION_KEYWORDS = [
        "왜", "어떻게", "무엇", "어떤", "얼마나",
        "이유", "원인", "배경", "근거",
    ]

    # Conceptual/analysis keywords → Vector DB
    CONCEPTUAL_KEYWORDS = [
        "전망", "분석", "의견", "견해", "관점",
        "리스크", "위험", "기회", "전략",
        "트렌드", "추세", "동향", "흐름",
        "예상", "예측", "시나리오",
    ]

    # Opinion keywords → Vector DB
    OPINION_KEYWORDS = [
        "투자의견", "추천", "매수", "매도", "중립",
        "목표가", "타겟", "컨센서스",
        "긍정적", "부정적", "중립적",
    ]

    # Comparison keywords → All DBs
    COMPARISON_KEYWORDS = [
        "비교", "차이", "vs", "대비",
        "어느", "둘 중", "뭐가 더",
    ]

    # Listing/screening keywords → PostgreSQL
    LISTING_KEYWORDS = [
        "목록", "리스트", "종목", "스크리닝",
        "찾아", "보여줘", "알려줘",
    ]

    def detect(self, query: str, entities: ExtractedEntities) -> DetectedSignals:
        """
        Detect all signals in query.

        Args:
            query: Original query string
            entities: Pre-extracted entities from QueryAnalyzer

        Returns:
            DetectedSignals with all detected signals and confidence
        """
        signals = DetectedSignals(raw_query=query)
        query_lower = query.lower()

        # Detect PostgreSQL signals from entities
        self._detect_postgresql_signals(entities, signals)

        # Detect Neo4j signals from keywords
        self._detect_neo4j_signals(query_lower, signals)

        # Detect Vector DB signals from keywords
        self._detect_vector_signals(query_lower, signals)

        # Detect comparison signals
        self._detect_comparison_signals(query_lower, entities, signals)

        return signals

    def _detect_postgresql_signals(
        self,
        entities: ExtractedEntities,
        signals: DetectedSignals
    ):
        """Detect PostgreSQL-related signals from extracted entities."""
        # Company lookup
        if entities.companies:
            confidence = min(1.0, len(entities.companies) * 0.5)
            signals.add_signal(SignalType.COMPANY_LOOKUP, confidence)

        # Issuer filter
        if entities.issuers:
            signals.add_signal(SignalType.ISSUER_FILTER, 1.0)

        # Date filter
        if entities.date_range:
            signals.add_signal(SignalType.DATE_FILTER, 1.0)

        # Sector filter
        if entities.sectors:
            confidence = min(1.0, len(entities.sectors) * 0.5)
            signals.add_signal(SignalType.SECTOR_FILTER, confidence)

        # Valuation filter
        if entities.valuation_keywords:
            signals.add_signal(SignalType.VALUATION_FILTER, 1.0)

        # Growth filter
        if entities.growth_keywords:
            signals.add_signal(SignalType.GROWTH_FILTER, 1.0)

    def _detect_neo4j_signals(self, query: str, signals: DetectedSignals):
        """Detect Neo4j-related signals from keywords."""
        # Relationship signals
        relationship_count = sum(1 for kw in self.RELATIONSHIP_KEYWORDS if kw in query)
        if relationship_count > 0:
            confidence = min(1.0, relationship_count * 0.5)
            signals.add_signal(SignalType.RELATIONSHIP, confidence)
            signals.add_signal(SignalType.CONTEXT_EXPANSION, 0.5)

        # Supply chain signals
        supply_count = sum(1 for kw in self.SUPPLY_CHAIN_KEYWORDS if kw in query)
        if supply_count > 0:
            confidence = min(1.0, supply_count * 0.5)
            signals.add_signal(SignalType.SUPPLY_CHAIN, confidence)
            signals.add_signal(SignalType.RELATIONSHIP, 0.5)

        # Competitor signals
        competitor_count = sum(1 for kw in self.COMPETITOR_KEYWORDS if kw in query)
        if competitor_count > 0:
            signals.add_signal(SignalType.COMPETITOR, 1.0)
            signals.add_signal(SignalType.RELATIONSHIP, 0.5)

    def _detect_vector_signals(self, query: str, signals: DetectedSignals):
        """Detect Vector DB-related signals from keywords."""
        # Question signals
        question_count = sum(1 for kw in self.QUESTION_KEYWORDS if kw in query)
        if question_count > 0:
            signals.add_signal(SignalType.SEMANTIC_QUESTION, 1.0)

        # Conceptual signals
        concept_count = sum(1 for kw in self.CONCEPTUAL_KEYWORDS if kw in query)
        if concept_count > 0:
            confidence = min(1.0, concept_count * 0.3)
            signals.add_signal(SignalType.CONCEPTUAL_QUERY, confidence)
            signals.add_signal(SignalType.ANALYSIS_REQUEST, confidence * 0.5)

        # Opinion signals
        opinion_count = sum(1 for kw in self.OPINION_KEYWORDS if kw in query)
        if opinion_count > 0:
            confidence = min(1.0, opinion_count * 0.4)
            signals.add_signal(SignalType.OPINION_REQUEST, confidence)

        # Check for question mark
        if "?" in query:
            signals.add_signal(SignalType.SEMANTIC_QUESTION, 0.5)

    def _detect_comparison_signals(
        self,
        query: str,
        entities: ExtractedEntities,
        signals: DetectedSignals
    ):
        """Detect comparison-related signals."""
        # Explicit comparison keywords
        comparison_count = sum(1 for kw in self.COMPARISON_KEYWORDS if kw in query)

        # Multiple companies suggest comparison
        if len(entities.companies) >= 2:
            signals.add_signal(SignalType.COMPARISON, 1.0)

        if comparison_count > 0:
            signals.add_signal(SignalType.COMPARISON, 1.0)


# Test
if __name__ == "__main__":
    from .query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    detector = SignalDetector()

    test_queries = [
        "삼성전자 최근 리포트",
        "테슬라 공급망 관련 한국 배터리 기업",
        "SK하이닉스 HBM 전망은 어떤가요?",
        "삼성전자와 SK하이닉스 비교",
        "저평가된 반도체 고성장주 찾아줘",
        "AI 반도체 시장 리스크 분석",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        entities = analyzer.analyze(q)
        signals = detector.detect(q, entities)

        print(f"\nSignals detected:")
        for signal, conf in signals.signals.items():
            print(f"  {signal.value}: {conf:.2f}")

        print(f"\nDB Scores:")
        print(f"  PostgreSQL: {signals.postgresql_score():.2f}")
        print(f"  Neo4j: {signals.neo4j_score():.2f}")
        print(f"  Vector DB: {signals.vector_score():.2f}")
