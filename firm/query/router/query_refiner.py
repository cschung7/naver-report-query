"""
Query Refiner - Sub-agent for converting casual queries to financial jargon

Converts natural language queries from casual investors into
proper financial terminology for better search results.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class RefinementResult:
    """Result of query refinement."""
    original_query: str
    refined_query: str
    transformations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification_options: List[Dict[str, str]] = field(default_factory=list)
    clarification_prompt: Optional[str] = None


class QueryRefiner:
    """
    Refines casual user queries into financial jargon.

    Examples:
        "싼 반도체 주식" → "저평가 반도체주"
        "AI로 돈 버는 회사" → "AI 수혜주"
        "요즘 뜨는 배터리 주식" → "고성장 배터리주"
        "삼전 어때?" → "삼성전자 투자의견"
    """

    # Casual term → Financial jargon mappings
    SYNONYM_MAP = {
        # Valuation terms
        "싼": "저평가",
        "싸다": "저평가",
        "저렴한": "저평가",
        "헐값": "저평가",
        "가성비": "저평가",
        "비싼": "고평가",
        "비싸다": "고평가",
        "거품": "고평가",
        "버블": "고평가",

        # Growth terms
        "뜨는": "고성장",
        "핫한": "고성장",
        "잘나가는": "고성장",
        "급등": "고성장",
        "떡상": "급등",
        "로켓": "급등",
        "대박": "고성장",
        "죽는": "하락",
        "망하는": "역성장",
        "떡락": "급락",

        # Relationship terms
        "돈 버는": "수혜",
        "돈버는": "수혜",
        "이득 보는": "수혜",
        "혜택": "수혜",
        "관련된": "관련",
        "연관된": "관련",
        "엮인": "관련",

        # Action terms
        "사야 할": "매수 추천",
        "사야할": "매수 추천",
        "살만한": "매수 추천",
        "팔아야 할": "매도 추천",
        "팔아야할": "매도 추천",
        "들고 있어도": "보유",
        "존버": "장기 보유",

        # Question patterns
        "어때": "투자의견",
        "어떄": "투자의견",
        "괜찮아": "투자의견",
        "괜찮을까": "투자의견",
        "살까": "매수 의견",
        "팔까": "매도 의견",
        "전망은": "전망",
        "앞으로": "전망",
        "미래": "전망",

        # Sector casual terms
        "반도체 주식": "반도체주",
        "배터리 주식": "배터리주",
        "자동차 주식": "자동차주",
        "바이오 주식": "바이오주",
        "게임 주식": "게임주",
        "조선 주식": "조선주",
        "은행 주식": "금융주",
        "증권 주식": "금융주",

        # Theme casual terms
        "AI 주식": "AI 관련주",
        "전기차 주식": "전기차 관련주",
        "2차전지 주식": "배터리 관련주",
        "친환경 주식": "친환경 관련주",
        "메타버스 주식": "메타버스 관련주",

        # Time expressions
        "요즘": "최근",
        "요새": "최근",
        "며칠 전": "최근",
        "얼마 전": "최근",
        "작년": "2024년",
        "올해": "2025년",

        # Misc
        "주식": "주",
        "종목": "주",
    }

    # Company nickname → Official name
    COMPANY_NICKNAMES = {
        "삼전": "삼성전자",
        "하닉": "SK하이닉스",
        "하이닉스": "SK하이닉스",
        "네버": "네이버",
        "카카오톡": "카카오",
        "톡": "카카오",
        "현차": "현대차",
        "기아차": "기아",
        "엘지": "LG",
        "에스케이": "SK",
        "포스코": "포스코홀딩스",
        "셀트리온": "셀트리온",
        "바로직스": "삼성바이오로직스",
        "삼바": "삼성바이오로직스",
        "엔씨": "엔씨소프트",
        "크래프톤": "크래프톤",
        "하오션": "한화오션",
        "현중": "HD현대중공업",
        "삼중": "삼성중공업",
        "현일렉": "HD현대일렉트릭",
        "엘에스": "LS ELECTRIC",
    }

    # Ambiguous terms that need clarification
    AMBIGUOUS_TERMS = {
        "좋은": {
            "prompt": "'좋은'이 의미하는 것을 선택해주세요:",
            "options": [
                {"label": "저평가 (싼)", "value": "저평가"},
                {"label": "고성장 (성장성 좋은)", "value": "고성장"},
                {"label": "실적 호조 (실적 좋은)", "value": "실적 호조"},
                {"label": "배당 좋은", "value": "고배당"},
            ]
        },
        "추천": {
            "prompt": "어떤 추천을 원하시나요?",
            "options": [
                {"label": "매수 추천 종목", "value": "매수 추천"},
                {"label": "저평가 종목", "value": "저평가"},
                {"label": "고성장 종목", "value": "고성장"},
            ]
        },
        "위험한": {
            "prompt": "'위험'이 의미하는 것을 선택해주세요:",
            "options": [
                {"label": "변동성 높은", "value": "고변동성"},
                {"label": "실적 악화", "value": "역성장"},
                {"label": "리스크 요인", "value": "리스크"},
            ]
        },
    }

    # Query patterns that need specific handling
    QUERY_PATTERNS = [
        # "X 어때?" pattern
        (r"(.+?)\s*(어때|어떄|괜찮|어떻|어떤가)", r"\1 투자의견"),
        # "X 살까?" pattern
        (r"(.+?)\s*(살까|사야|살만)", r"\1 매수 의견"),
        # "X 팔까?" pattern
        (r"(.+?)\s*(팔까|팔아야)", r"\1 매도 의견"),
        # "X vs Y" comparison
        (r"(.+?)\s*(vs|VS|대|랑)\s*(.+?)\s*(비교|차이|뭐가)", r"\1 \3 비교"),
    ]

    def __init__(self):
        # Build reverse lookup for faster processing
        self._build_lookups()

    def _build_lookups(self):
        """Build efficient lookup structures."""
        # Sort by length (longest first) for proper matching
        self.sorted_synonyms = sorted(
            self.SYNONYM_MAP.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self.sorted_nicknames = sorted(
            self.COMPANY_NICKNAMES.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def refine(self, query: str, auto_clarify: bool = False) -> RefinementResult:
        """
        Refine a casual user query into financial jargon.

        Args:
            query: Original user query
            auto_clarify: If True, auto-select first option for ambiguous terms

        Returns:
            RefinementResult with refined query and metadata
        """
        result = RefinementResult(
            original_query=query,
            refined_query=query
        )

        working_query = query

        # Step 1: Apply regex patterns
        working_query, pattern_transforms = self._apply_patterns(working_query)
        result.transformations.extend(pattern_transforms)

        # Step 2: Replace company nicknames
        working_query, nickname_transforms = self._replace_nicknames(working_query)
        result.transformations.extend(nickname_transforms)

        # Step 3: Check for ambiguous terms
        ambiguous = self._check_ambiguous(working_query)
        if ambiguous and not auto_clarify:
            result.needs_clarification = True
            result.clarification_prompt = ambiguous['prompt']
            result.clarification_options = ambiguous['options']
            result.confidence = 0.6

        # Step 4: Apply synonym mappings
        working_query, synonym_transforms = self._apply_synonyms(working_query)
        result.transformations.extend(synonym_transforms)

        # Step 5: Clean up query
        working_query = self._cleanup_query(working_query)

        result.refined_query = working_query

        # Calculate confidence based on transformations
        if result.transformations:
            result.confidence = min(0.95, 0.7 + len(result.transformations) * 0.05)

        return result

    def refine_with_clarification(
        self,
        query: str,
        clarification_response: str
    ) -> RefinementResult:
        """
        Refine query with user's clarification response.

        Args:
            query: Original query
            clarification_response: User's selected clarification

        Returns:
            RefinementResult with clarification applied
        """
        # Find and replace ambiguous term with clarification
        working_query = query

        for term, config in self.AMBIGUOUS_TERMS.items():
            if term in working_query:
                working_query = working_query.replace(term, clarification_response)
                break

        # Now refine normally
        result = self.refine(working_query, auto_clarify=True)
        result.transformations.insert(0, f"Clarified: '{query}' → '{working_query}'")

        return result

    def _apply_patterns(self, query: str) -> Tuple[str, List[str]]:
        """Apply regex patterns for common query structures."""
        transforms = []

        for pattern, replacement in self.QUERY_PATTERNS:
            match = re.search(pattern, query)
            if match:
                new_query = re.sub(pattern, replacement, query)
                if new_query != query:
                    transforms.append(f"Pattern: '{query}' → '{new_query}'")
                    query = new_query
                    break  # Apply only first matching pattern

        return query, transforms

    def _replace_nicknames(self, query: str) -> Tuple[str, List[str]]:
        """Replace company nicknames with official names."""
        transforms = []

        for nickname, official in self.sorted_nicknames:
            if nickname in query:
                query = query.replace(nickname, official)
                transforms.append(f"Company: '{nickname}' → '{official}'")

        return query, transforms

    def _check_ambiguous(self, query: str) -> Optional[Dict]:
        """Check for ambiguous terms that need clarification."""
        for term, config in self.AMBIGUOUS_TERMS.items():
            if term in query:
                return config
        return None

    def _apply_synonyms(self, query: str) -> Tuple[str, List[str]]:
        """Apply synonym mappings."""
        transforms = []

        for casual, formal in self.sorted_synonyms:
            if casual in query:
                query = query.replace(casual, formal)
                transforms.append(f"Synonym: '{casual}' → '{formal}'")

        return query, transforms

    def _cleanup_query(self, query: str) -> str:
        """Clean up the refined query."""
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query).strip()

        # Remove redundant terms
        query = query.replace("주주", "주")  # 반도체주주 → 반도체주

        return query

    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions for autocomplete.

        Args:
            partial_query: Partial user input

        Returns:
            List of suggested complete queries
        """
        suggestions = []

        # Common query templates
        templates = [
            "{company} 투자의견",
            "{company} 전망",
            "{company} 목표가",
            "저평가 {sector}주",
            "고성장 {sector}주",
            "{theme} 관련주",
            "{theme} 수혜주",
        ]

        # If partial matches a company nickname
        for nickname, official in self.COMPANY_NICKNAMES.items():
            if partial_query.lower() in nickname.lower():
                suggestions.extend([
                    f"{official} 투자의견",
                    f"{official} 전망",
                    f"{official} 최근 리포트",
                ])

        # If partial matches a sector
        sectors = ["반도체", "배터리", "바이오", "자동차", "조선", "금융", "게임"]
        for sector in sectors:
            if partial_query in sector:
                suggestions.extend([
                    f"저평가 {sector}주",
                    f"고성장 {sector}주",
                    f"{sector} 관련주 전망",
                ])

        # If partial matches a theme
        themes = ["AI", "전기차", "데이터센터", "로봇", "메타버스"]
        for theme in themes:
            if partial_query.upper() in theme.upper():
                suggestions.extend([
                    f"{theme} 관련주",
                    f"{theme} 수혜주",
                ])

        return suggestions[:5]  # Return top 5


# Test
if __name__ == "__main__":
    refiner = QueryRefiner()

    test_queries = [
        "삼전 어때?",
        "싼 반도체 주식 찾아줘",
        "AI로 돈 버는 회사",
        "요즘 뜨는 배터리 주식",
        "하닉 살까?",
        "현차 vs 기아 비교",
        "좋은 바이오 주식",  # Ambiguous
        "떡상할 게임주",
        "존버해도 될까 네버",
    ]

    print("=" * 70)
    print("Query Refiner Test")
    print("=" * 70)

    for q in test_queries:
        result = refiner.refine(q)
        print(f"\nOriginal:  {result.original_query}")
        print(f"Refined:   {result.refined_query}")
        print(f"Confidence: {result.confidence:.0%}")

        if result.transformations:
            print(f"Transforms: {', '.join(result.transformations[:3])}")

        if result.needs_clarification:
            print(f"⚠️  Needs clarification: {result.clarification_prompt}")
            for opt in result.clarification_options:
                print(f"   - {opt['label']}")
