"""
Configuration settings for NaverReport-Industry
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# Base paths
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/mnt/nas/gpt/Naver"))

# Data source paths
CSV_PATH = DATA_ROOT / "Job/df_files/df_IndustryAnalysis_with_url_2026-01-18.csv"
EXTRACTION_PATH = DATA_ROOT / "Extraction_IndustryAnalysis/extractions"
MARKER_PATH = DATA_ROOT / "Marker/IndustryAnalysis"

# Database settings
POSTGRES_URI = os.getenv("POSTGRES_URI_INDUSTRY", os.getenv("POSTGRES_URI", "postgresql://kg_user:kg_secure_password_2025@localhost:5432/naver_industry"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "naverneo4j")

# Table/collection names (separate from other projects)
PG_TABLE = "industry_reports"
NEO4J_DB = "industryanalysis"

# Vector DB settings
VECTOR_DB_PATH = PROJECT_ROOT / "vectordb"
VECTOR_COLLECTION = "industry_vectors"
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # Korean sentence-BERT model
EMBEDDING_DIMENSION = 768

# Industry categories
INDUSTRIES = [
    "기타", "건설", "자동차", "석유화학", "철강금속", "은행", "반도체",
    "음식료", "게임", "통신", "제약", "전기전자", "유틸리티", "유통",
    "보험", "조선", "디스플레이", "IT", "항공운송", "증권", "바이오",
    "화장품", "금융", "지주회사", "에너지", "휴대폰", "미디어", "섬유의류",
    "인터넷포탈", "기계", "타이어", "여행", "해운", "건자재", "자동차부품",
    "소프트웨어", "교육", "담배",
]

# Cycle stages
CYCLE_STAGES = [
    "UPCYCLE",
    "DOWNCYCLE",
    "PEAK",
    "TROUGH",
    "RECOVERY",
    "EXPANSION",
    "CONTRACTION",
]

# Demand trends
DEMAND_TRENDS = [
    "GROWING",
    "STABLE",
    "CONTRACTING",
    "RECOVERING",
    "PEAKING",
]

# Investment timings
INVESTMENT_TIMINGS = [
    "BUY",
    "HOLD",
    "AVOID",
    "ACCUMULATE",
    "REDUCE",
]

# Geographies
GEOGRAPHIES = [
    "GLOBAL",
    "KOREA",
    "US",
    "CHINA",
    "JAPAN",
    "EUROPE",
    "EMERGING_MARKETS",
    "ASIA",
]

# Top issuers
TOP_ISSUERS = [
    "하나증권",
    "키움증권",
    "한화투자증권",
    "유진투자증권",
    "신한투자증권",
    "하이투자증권",
    "IBK투자증권",
    "교보증권",
    "SK증권",
    "유안타증권",
    "이베스트증권",
    "대신증권",
    "DS투자증권",
    "메리츠증권",
]
