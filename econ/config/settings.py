"""
Configuration settings for NaverReport-EconAnalysis
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # Also load from current directory

# Base paths
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/mnt/nas/gpt/Naver"))

# Data source paths
CSV_PATH = DATA_ROOT / "Job/df_files/df_EconAnalysis_with_url_2026-01-18.csv"
EXTRACTION_PATH = DATA_ROOT / "Extraction_EconAnalysis/extractions"
MARKER_PATH = DATA_ROOT / "Marker/EconAnalysis"

# Database settings
POSTGRES_URI = os.getenv("POSTGRES_URI_ECON", os.getenv("POSTGRES_URI", "postgresql://kg_user:kg_secure_password_2025@localhost:5432/naver_econ"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "naverneo4j")

# Table/collection names (separate from other projects)
PG_TABLE = "econ_reports"
NEO4J_DB = "econanalysis"

# Vector DB settings
VECTOR_DB_PATH = PROJECT_ROOT / "vectordb"
VECTOR_COLLECTION = "econ_vectors"
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # Korean sentence-BERT model
EMBEDDING_DIMENSION = 768

# Economic report categories
ECON_CATEGORIES = [
    "금리",
    "환율",
    "물가",
    "고용",
    "GDP",
    "무역",
    "통화정책",
    "재정정책",
    "글로벌경제",
    "국내경제",
    "미국경제",
    "중국경제",
    "유럽경제",
    "신흥국경제",
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
    "케이프투자증권",
]
