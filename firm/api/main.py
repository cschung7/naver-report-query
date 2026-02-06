"""
SmartQuery FastAPI Server

Intelligent query API for Korean analyst reports.
Routes queries to PostgreSQL, Neo4j, and Vector DB automatically.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# Global SmartQuery instance (initialized on startup)
smart_query = None


# ============================================================
# Pydantic Models
# ============================================================

class QueryRequest(BaseModel):
    """Request model for smart query."""
    question: str = Field(..., min_length=2, max_length=500, description="Natural language query")
    max_reports: int = Field(default=20, ge=1, le=100, description="Maximum reports to return")
    max_claims: int = Field(default=50, ge=1, le=200, description="Maximum claims to return")
    verbose: bool = Field(default=False, description="Include debug information")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "삼성전자 HBM 전망은?",
                "max_reports": 20,
                "max_claims": 50,
                "verbose": False
            }
        }


class SourceInfo(BaseModel):
    """Source information from vector search."""
    report_id: str
    company: Optional[str] = None
    issuer: Optional[str] = None
    similarity: float


class ReportInfo(BaseModel):
    """Report information."""
    report_id: Optional[str] = None
    issue_date: Optional[str] = None
    company: Optional[str] = None
    issuer: Optional[str] = None
    title: Optional[str] = None


class ClaimInfo(BaseModel):
    """Claim information from Neo4j."""
    claim_type: Optional[str] = None
    claim_text: Optional[str] = None
    company: Optional[str] = None
    issue_date: Optional[str] = None


class RoutingInfo(BaseModel):
    """Query routing information."""
    intent: str
    postgresql: str
    neo4j: str
    vector_db: str
    parallel: bool
    reasoning: str


class QueryResponse(BaseModel):
    """Response model for smart query."""
    success: bool
    query: str
    intent: str
    answer: Optional[str] = None

    # Results
    reports: List[Dict[str, Any]] = []
    claims: List[Dict[str, Any]] = []
    sources_used: List[str] = []

    # Vector search results
    vector_chunks_found: int = 0
    vector_sources: List[SourceInfo] = []

    # Metadata
    execution_time_ms: float
    routing: Optional[RoutingInfo] = None

    # Debug info (if verbose)
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    postgresql: bool
    neo4j: bool
    vector_db: bool
    vector_chunks: int
    indexed_files: int
    timestamp: str


class StatsResponse(BaseModel):
    """Statistics response."""
    vector_db: Dict[str, Any]
    reports_count: int
    claims_count: int


# ============================================================
# Lifespan - Initialize on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SmartQuery on startup, cleanup on shutdown."""
    global smart_query

    print("=" * 60)
    print("Starting SmartQuery API Server")
    print("=" * 60)
    print("Loading SmartQuery (this takes ~2 minutes for Vector DB)...")

    try:
        from query.router import SmartQuery
        smart_query = SmartQuery()
        print("✅ SmartQuery initialized successfully!")

        # Get stats
        if smart_query.vector_client:
            stats = smart_query.vector_client.get_stats()
            print(f"   Vector DB: {stats['total_chunks']:,} chunks")
            print(f"   Indexed files: {stats['indexed_files']:,}")

    except Exception as e:
        print(f"❌ Failed to initialize SmartQuery: {e}")
        raise

    print("=" * 60)
    print("Server ready!")
    print("=" * 60)

    yield

    # Cleanup
    print("Shutting down...")
    if smart_query:
        smart_query.close()
    print("Goodbye!")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="SmartQuery API",
    description="Intelligent query API for Korean analyst reports. "
                "Automatically routes queries to PostgreSQL, Neo4j, and Vector DB.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SmartQuery API",
        "version": "1.0.0",
        "description": "Intelligent query API for Korean analyst reports",
        "endpoints": {
            "POST /query": "Execute smart query",
            "GET /health": "Health check",
            "GET /stats": "Database statistics",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health of all database connections."""
    global smart_query

    if not smart_query:
        raise HTTPException(status_code=503, detail="SmartQuery not initialized")

    # Check connections
    pg_ok = False
    neo4j_ok = False
    vector_ok = False
    vector_chunks = 0
    indexed_files = 0

    try:
        # PostgreSQL
        smart_query.pg_client.search_reports(limit=1)
        pg_ok = True
    except:
        pass

    try:
        # Neo4j
        smart_query.neo4j_client.get_company_claims("test", limit=1)
        neo4j_ok = True
    except:
        pass

    try:
        # Vector DB
        if smart_query.vector_client:
            stats = smart_query.vector_client.get_stats()
            vector_chunks = stats['total_chunks']
            indexed_files = stats['indexed_files']
            vector_ok = True
    except:
        pass

    status = "healthy" if (pg_ok and neo4j_ok and vector_ok) else "degraded"

    return HealthResponse(
        status=status,
        postgresql=pg_ok,
        neo4j=neo4j_ok,
        vector_db=vector_ok,
        vector_chunks=vector_chunks,
        indexed_files=indexed_files,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get database statistics."""
    global smart_query

    if not smart_query:
        raise HTTPException(status_code=503, detail="SmartQuery not initialized")

    vector_stats = {}
    if smart_query.vector_client:
        vector_stats = smart_query.vector_client.get_stats()

    # Get report count
    reports = smart_query.pg_client.search_reports(limit=1)

    return StatsResponse(
        vector_db=vector_stats,
        reports_count=len(reports) if reports else 0,
        claims_count=0  # TODO: Add Neo4j count
    )


def _run_query(question: str, max_reports: int, max_claims: int, verbose: bool):
    """Run SmartQuery synchronously (for thread pool)."""
    global smart_query
    return smart_query.query(
        question=question,
        max_reports=max_reports,
        max_claims=max_claims,
        verbose=verbose
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def execute_query(request: QueryRequest):
    """
    Execute intelligent query with automatic routing.

    The query is analyzed and routed to the appropriate database(s):
    - **PostgreSQL**: Structured filters (company, date, sector)
    - **Neo4j**: Relationship queries (supply chain, competitors)
    - **Vector DB**: Semantic/conceptual questions

    Multiple databases may be queried for comprehensive results.
    """
    global smart_query

    if not smart_query:
        raise HTTPException(status_code=503, detail="SmartQuery not initialized")

    try:
        # Run blocking query in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            _run_query,
            request.question,
            request.max_reports,
            request.max_claims,
            request.verbose
        )

        # Build response
        response = QueryResponse(
            success=True,
            query=result.query,
            intent=result.intent,
            answer=result.answer,
            reports=result.reports[:request.max_reports],
            claims=result.claims[:request.max_claims],
            sources_used=result.sources_used,
            execution_time_ms=result.execution_time_ms,
        )

        # Add vector search info
        if result.vector_result:
            response.vector_chunks_found = result.vector_result.get('chunks_found', 0)
            sources = result.vector_result.get('sources', [])
            response.vector_sources = [
                SourceInfo(
                    report_id=s.get('report_id', ''),
                    company=s.get('company'),
                    issuer=s.get('issuer'),
                    similarity=s.get('similarity', 0)
                )
                for s in sources
            ]

        # Add routing info
        if result.routing:
            response.routing = RoutingInfo(
                intent=result.routing.intent.value,
                postgresql=result.routing.postgresql.value,
                neo4j=result.routing.neo4j.value,
                vector_db=result.routing.vector_db.value,
                parallel=result.routing.parallel,
                reasoning=result.routing.reasoning
            )

        # Add debug info if verbose
        if request.verbose and result.entities:
            response.debug = {
                "entities": {
                    "companies": result.entities.companies,
                    "issuers": result.entities.issuers,
                    "sectors": result.entities.sectors,
                    "themes": result.entities.themes,
                    "date_range": result.entities.date_range,
                    "valuation_keywords": result.entities.valuation_keywords,
                    "growth_keywords": result.entities.growth_keywords,
                }
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def execute_query_get(
    q: str = Query(..., min_length=2, max_length=500, description="Query string"),
    max_reports: int = Query(default=20, ge=1, le=100),
    max_claims: int = Query(default=50, ge=1, le=200),
    verbose: bool = Query(default=False)
):
    """
    Execute query via GET request (for simple testing).

    Example: /query?q=삼성전자 HBM 전망
    """
    request = QueryRequest(
        question=q,
        max_reports=max_reports,
        max_claims=max_claims,
        verbose=verbose
    )
    return await execute_query(request)


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
