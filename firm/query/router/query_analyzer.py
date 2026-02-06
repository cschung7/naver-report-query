"""
Query Analyzer

Extracts entities, temporal references, and structured elements from queries.
"""
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ExtractedEntities:
    """Container for extracted entities from a query."""
    companies: List[str] = field(default_factory=list)
    issuers: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None
    valuation_keywords: List[str] = field(default_factory=list)
    growth_keywords: List[str] = field(default_factory=list)
    raw_query: str = ""

    def has_structured_filters(self) -> bool:
        """Check if query has any structured filter elements."""
        return bool(
            self.companies or
            self.issuers or
            self.sectors or
            self.date_range or
            self.valuation_keywords or
            self.growth_keywords
        )


class QueryAnalyzer:
    """
    Analyzes user queries to extract structured elements.

    Extracts:
    - Company names (삼성전자, SK하이닉스, etc.)
    - Broker/Issuer names (한화투자증권, 미래에셋증권, etc.)
    - Sector references (반도체, 배터리, 바이오, etc.)
    - Temporal references (최근, 지난주, 2024년, etc.)
    - Valuation keywords (저평가, 할인, etc.)
    - Growth keywords (고성장, 성장주, etc.)
    - Investment themes (AI, 전기차, 친환경, etc.)
    """

    # Major Korean companies (can be extended from DB)
    KNOWN_COMPANIES = {
        # Tech/Semiconductor
        "삼성전자", "SK하이닉스", "삼성SDI", "LG이노텍", "삼성전기",
        "DB하이텍", "리노공업", "원익IPS", "한미반도체", "이오테크닉스",
        # Battery/EV
        "LG에너지솔루션", "삼성SDI", "SK온", "에코프로", "에코프로비엠",
        "포스코퓨처엠", "엘앤에프", "천보", "코스모신소재",
        # Auto
        "현대차", "기아", "현대모비스", "만도", "한온시스템",
        # Bio/Pharma
        "삼성바이오로직스", "셀트리온", "SK바이오팜", "유한양행", "녹십자",
        # Internet/Platform
        "네이버", "카카오", "엔씨소프트", "크래프톤", "넷마블",
        # Finance
        "삼성생명", "KB금융", "신한지주", "하나금융지주", "우리금융지주",
        # Industrial
        "포스코홀딩스", "현대제철", "고려아연", "삼성물산", "현대건설",
        # Power Equipment / Heavy Industry
        "효성중공업", "HD현대일렉트릭", "현대일렉트릭", "LS일렉트릭", "LS ELECTRIC",
        "일진전기", "대한전선", "제룡전기", "제룡산업", "LS전선", "가온전선",
        # Entertainment
        "하이브", "JYP엔터", "SM엔터테인먼트", "CJ ENM",
        # Foreign companies often mentioned
        "테슬라", "Tesla", "엔비디아", "NVIDIA", "애플", "Apple",
        "마이크로소프트", "Microsoft", "아마존", "Amazon", "구글", "Google",
        "TSMC", "인텔", "Intel", "AMD", "퀄컴", "Qualcomm",
    }

    # Korean securities firms
    KNOWN_ISSUERS = {
        "한화투자증권", "미래에셋증권", "삼성증권", "NH투자증권", "KB증권",
        "신한투자증권", "하나증권", "대신증권", "키움증권", "메리츠증권",
        "유안타증권", "유진투자증권", "하이투자증권", "IBK투자증권",
        "SK증권", "현대차증권", "한국투자증권", "교보증권", "DB금융투자",
        "이베스트투자증권", "케이프투자증권", "부국증권", "BNK투자증권",
    }

    # Sector keywords
    SECTOR_KEYWORDS = {
        "반도체": ["반도체", "칩", "메모리", "HBM", "파운드리", "DRAM", "NAND"],
        "배터리": ["배터리", "2차전지", "이차전지", "양극재", "음극재", "전해질", "분리막"],
        "자동차": ["자동차", "전기차", "EV", "자율주행", "모빌리티"],
        "바이오": ["바이오", "제약", "헬스케어", "신약", "의료기기", "CMO", "CDMO"],
        "인터넷": ["인터넷", "플랫폼", "게임", "메타버스", "콘텐츠"],
        "금융": ["금융", "은행", "보험", "증권", "카드"],
        "화학": ["화학", "정유", "석유화학", "2차전지소재"],
        "철강": ["철강", "금속", "비철금속"],
        "건설": ["건설", "부동산", "인프라"],
        "유틸리티": ["전력", "가스", "에너지", "신재생"],
        "전력기기": ["변압기", "송배전", "전력설비", "전력기기", "중공업", "중전기기", "배전반", "차단기", "개폐기", "데이터센터 전력", "AI 전력", "HVDC"],
        "엔터테인먼트": ["엔터", "미디어", "K-POP", "음악", "영화", "드라마"],
        "통신": ["통신", "5G", "6G", "네트워크"],
        "조선": ["조선", "해운", "LNG선"],
        "항공": ["항공", "여행", "관광"],
        "방산": ["방산", "국방", "방위산업"],
    }

    # Investment themes
    THEME_KEYWORDS = {
        "AI": ["AI", "인공지능", "머신러닝", "딥러닝", "GPT", "LLM", "생성형AI"],
        "전기차": ["전기차", "EV", "xEV", "BEV", "PHEV"],
        "친환경": ["친환경", "ESG", "탄소중립", "그린", "RE100"],
        "우주항공": ["우주", "항공", "위성", "스페이스X", "SpaceX"],
        "로봇": ["로봇", "자동화", "로보틱스"],
        "메타버스": ["메타버스", "VR", "AR", "XR"],
        "클라우드": ["클라우드", "SaaS", "데이터센터"],
        "사이버보안": ["보안", "사이버", "해킹"],
    }

    # Temporal patterns
    TEMPORAL_PATTERNS = {
        "최근": lambda: (datetime.now() - timedelta(days=30), None),
        "오늘": lambda: (datetime.now().replace(hour=0, minute=0), None),
        "이번주": lambda: (datetime.now() - timedelta(days=7), None),
        "이번달": lambda: (datetime.now() - timedelta(days=30), None),
        "지난주": lambda: (datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=7)),
        "지난달": lambda: (datetime.now() - timedelta(days=60), datetime.now() - timedelta(days=30)),
        "올해": lambda: (datetime(datetime.now().year, 1, 1), None),
        "작년": lambda: (datetime(datetime.now().year - 1, 1, 1), datetime(datetime.now().year, 1, 1)),
    }

    # Valuation keywords → valuation_regime mapping
    VALUATION_KEYWORDS = {
        "저평가": "DEEP_DISCOUNT",
        "할인": "DISCOUNT",
        "적정가": "FAIR_VALUE",
        "고평가": "PREMIUM",
        "프리미엄": "PREMIUM",
        "싸다": "DEEP_DISCOUNT",
        "비싸다": "PREMIUM",
        "밸류에이션": None,  # General valuation mention
    }

    # Growth keywords → growth_regime mapping
    GROWTH_KEYWORDS = {
        "고성장": "HIGH_GROWTH",
        "성장주": "HIGH_GROWTH",
        "성장": "GROWTH",
        "정체": "STAGNANT",
        "역성장": "DECLINE",
        "하락": "DECLINE",
        "감소": "DECLINE",
        "턴어라운드": "TURNAROUND",
    }

    def __init__(self, db_companies: Optional[Set[str]] = None):
        """
        Initialize analyzer with optional company list from database.

        Args:
            db_companies: Set of company names from database for better matching
        """
        self.companies = self.KNOWN_COMPANIES.copy()
        if db_companies:
            self.companies.update(db_companies)

    def analyze(self, query: str) -> ExtractedEntities:
        """
        Analyze query and extract all structured elements.

        Args:
            query: User's natural language query

        Returns:
            ExtractedEntities with all extracted elements
        """
        result = ExtractedEntities(raw_query=query)

        # Extract each type
        result.companies = self._extract_companies(query)
        result.issuers = self._extract_issuers(query)
        result.sectors = self._extract_sectors(query)
        result.themes = self._extract_themes(query)
        result.date_range = self._extract_date_range(query)
        result.valuation_keywords = self._extract_valuation(query)
        result.growth_keywords = self._extract_growth(query)

        return result

    def _extract_companies(self, query: str) -> List[str]:
        """Extract company names from query."""
        found = []
        query_lower = query.lower()

        for company in self.companies:
            if company.lower() in query_lower or company in query:
                found.append(company)

        return list(set(found))

    def _extract_issuers(self, query: str) -> List[str]:
        """Extract broker/issuer names from query."""
        found = []

        for issuer in self.KNOWN_ISSUERS:
            if issuer in query:
                found.append(issuer)

        return list(set(found))

    def _extract_sectors(self, query: str) -> List[str]:
        """Extract sector references from query."""
        found = []
        query_lower = query.lower()

        for sector, keywords in self.SECTOR_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    found.append(sector)
                    break

        return list(set(found))

    def _extract_themes(self, query: str) -> List[str]:
        """Extract investment themes from query."""
        found = []
        query_upper = query.upper()
        query_lower = query.lower()

        for theme, keywords in self.THEME_KEYWORDS.items():
            for kw in keywords:
                if kw.upper() in query_upper or kw.lower() in query_lower:
                    found.append(theme)
                    break

        return list(set(found))

    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract temporal references and convert to date range."""
        # Check for explicit year patterns (2024년, 2025년)
        year_match = re.search(r'(\d{4})년', query)
        if year_match:
            year = int(year_match.group(1))
            return {
                "from": f"{year}-01-01",
                "to": f"{year}-12-31"
            }

        # Check for temporal keywords
        for keyword, date_func in self.TEMPORAL_PATTERNS.items():
            if keyword in query:
                start, end = date_func()
                result = {"from": start.strftime("%Y-%m-%d")}
                if end:
                    result["to"] = end.strftime("%Y-%m-%d")
                return result

        return None

    def _extract_valuation(self, query: str) -> List[str]:
        """Extract valuation-related keywords."""
        found = []

        for keyword in self.VALUATION_KEYWORDS:
            if keyword in query:
                regime = self.VALUATION_KEYWORDS[keyword]
                if regime:
                    found.append(regime)

        return list(set(found))

    def _extract_growth(self, query: str) -> List[str]:
        """Extract growth-related keywords."""
        found = []

        for keyword in self.GROWTH_KEYWORDS:
            if keyword in query:
                regime = self.GROWTH_KEYWORDS[keyword]
                if regime:
                    found.append(regime)

        return list(set(found))

    def get_primary_company(self, entities: ExtractedEntities) -> Optional[str]:
        """Get the primary/first company from extracted entities."""
        return entities.companies[0] if entities.companies else None

    def get_primary_issuer(self, entities: ExtractedEntities) -> Optional[str]:
        """Get the primary/first issuer from extracted entities."""
        return entities.issuers[0] if entities.issuers else None

    def get_primary_sector(self, entities: ExtractedEntities) -> Optional[str]:
        """Get the primary/first sector from extracted entities."""
        return entities.sectors[0] if entities.sectors else None


# Test
if __name__ == "__main__":
    analyzer = QueryAnalyzer()

    test_queries = [
        "삼성전자 최근 리포트",
        "테슬라 공급망 관련 한국 배터리 기업",
        "한화투자증권이 분석한 SK하이닉스",
        "저평가된 반도체 고성장주",
        "2024년 AI 관련주 투자전략",
        "HBM 시장 전망과 수혜주",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {q}")
        result = analyzer.analyze(q)
        print(f"  Companies: {result.companies}")
        print(f"  Issuers: {result.issuers}")
        print(f"  Sectors: {result.sectors}")
        print(f"  Themes: {result.themes}")
        print(f"  Date range: {result.date_range}")
        print(f"  Valuation: {result.valuation_keywords}")
        print(f"  Growth: {result.growth_keywords}")
        print(f"  Has structured filters: {result.has_structured_filters()}")
