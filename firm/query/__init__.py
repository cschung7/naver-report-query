"""NaverReport-FirmAnalysis Query Module

Query clients for PostgreSQL, Neo4j, and Gemini.
"""
from .postgres_client import PostgresClient
from .neo4j_client import Neo4jClient
from .hybrid_query import HybridQuery, QueryResult, format_result

# Optional Gemini import
try:
    from .gemini_client import GeminiClient
    __all__ = [
        "PostgresClient",
        "Neo4jClient",
        "GeminiClient",
        "HybridQuery",
        "QueryResult",
        "format_result",
    ]
except ImportError:
    __all__ = [
        "PostgresClient",
        "Neo4jClient",
        "HybridQuery",
        "QueryResult",
        "format_result",
    ]
