"""
Configuration settings for NaverReport-InvestmentStrategy
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
CSV_PATH = DATA_ROOT / "Job/df_files/df_2024_2023_InvestStrategy_with_author_final_2024-06-15.csv"
EXTRACTION_PATH = DATA_ROOT / "Extraction_InvestmentAnalysis/extraction"
MARKER_PATH = DATA_ROOT / "Marker/InvestmentStrategy"

# Database settings
POSTGRES_URI = os.getenv("POSTGRES_URI_INVEST", os.getenv("POSTGRES_URI", "postgresql://naver:naver@localhost:5432/naver_report"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "naverneo4j")

# Table/collection names (separate from FirmAnalysis)
PG_TABLE = "invest_strategy_reports"
NEO4J_DB = "investstrategy"

# Vector DB settings
VECTOR_DB_PATH = PROJECT_ROOT / "vectordb"
VECTOR_COLLECTION = "invest_strategy_vectors"
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # Korean sentence-BERT model
EMBEDDING_DIMENSION = 768

# Strategy types
STRATEGY_TYPES = [
    "ANNUAL_OUTLOOK",
    "QUARTERLY_OUTLOOK",
    "MONTHLY_OUTLOOK",
    "WEEKLY_REVIEW",
    "DAILY_BRIEF",
    "THEMATIC",
    "SECTOR_ROTATION",
    "ASSET_ALLOCATION",
    "FACTOR_STRATEGY",
    "MACRO_ANALYSIS",
    "MARKET_SENTIMENT",
]

# Market regimes
MARKET_REGIMES = [
    "RISK_ON",
    "RISK_OFF",
    "NEUTRAL",
    "DEFENSIVE",
    "AGGRESSIVE",
]

# Market outlooks
MARKET_OUTLOOKS = [
    "BULLISH",
    "BEARISH",
    "NEUTRAL",
    "RECOVERY",
    "CONSOLIDATION",
]

# Market stages
MARKET_STAGES = [
    "EARLY_CYCLE",
    "MID_CYCLE",
    "LATE_CYCLE",
    "RECESSION",
    "RECOVERY",
]

# Time horizons
TIME_HORIZONS = [
    "1W",
    "1M",
    "3M",
    "6M",
    "1Y",
    "LONG_TERM",
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
    "DEVELOPED_MARKETS",
    "ASIA",
]

# Asset classes
ASSET_CLASSES = [
    "EQUITY",
    "BOND",
    "COMMODITY",
    "CURRENCY",
    "REITS",
    "ALTERNATIVES",
    "CASH",
]

# Allocation weights
ALLOCATION_WEIGHTS = [
    "OVERWEIGHT",
    "NORMAL",
    "UNDERWEIGHT",
    "HIGH",
    "MEDIUM",
    "LOW",
]
