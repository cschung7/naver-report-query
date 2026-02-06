"""
Simple IndustryQuery API for Industry Analysis - Sync version

Run with: python api/simple_api.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__, template_folder='templates')

# Global instances
industry_query = None
neo4j_client = None
gemini_client = None


def get_gemini_client():
    """Lazy-load GeminiClient."""
    global gemini_client
    if gemini_client is None:
        try:
            from query.gemini_client import GeminiClient
            gemini_client = GeminiClient()
        except Exception as e:
            print(f"Warning: GeminiClient not available: {e}")
            return None
    return gemini_client


def get_industry_query():
    """Lazy-load IndustryQuery."""
    global industry_query
    if industry_query is None:
        print("Loading IndustryQuery...", flush=True)
        from query.router import IndustryQuery
        industry_query = IndustryQuery()
        print("IndustryQuery ready!", flush=True)
    return industry_query


def get_neo4j_client():
    """Lazy-load Neo4j client."""
    global neo4j_client
    if neo4j_client is None:
        print("Loading Neo4jClient...", flush=True)
        from query.router.neo4j_client import Neo4jClient
        neo4j_client = Neo4jClient()
        print("Neo4jClient ready!", flush=True)
    return neo4j_client


def preload_industry_query():
    """Pre-load IndustryQuery at startup."""
    iq = get_industry_query()

    # Force initialization of PostgreSQL
    print("Pre-loading PostgresClient...", flush=True)
    if iq.pg_client:
        try:
            stats = iq.pg_client.get_stats()
            print(f"  PostgresClient ready - {stats.get('total_reports', 0)} reports", flush=True)
        except Exception as e:
            print(f"  PostgresClient error: {e}", flush=True)

    # Pre-load Neo4j client
    print("Pre-loading Neo4jClient...", flush=True)
    try:
        nc = get_neo4j_client()
        graph_stats = nc.get_stats()
        print(f"  Neo4jClient ready - {graph_stats.get('industryreport_count', 0)} nodes", flush=True)
    except Exception as e:
        print(f"  Neo4jClient error: {e}", flush=True)

    # Pre-load Vector client and embedding model
    print("Pre-loading VectorClient...", flush=True)
    try:
        if iq.vector_client:
            vector_stats = iq.vector_client.get_stats()
            print(f"  VectorClient ready - {vector_stats.get('total_documents', 0)} vectors", flush=True)
            # Force-load embedding model for faster first query
            iq.vector_client.warmup()
    except Exception as e:
        print(f"  VectorClient error: {e}", flush=True)

    return iq


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.route('/api')
def api_info():
    """API information endpoint."""
    return jsonify({
        "name": "Industry Analysis IndustryQuery API",
        "version": "1.0.0",
        "description": "Multi-source search: PostgreSQL + Neo4j + ChromaDB + Semantic Cache + Prometheus",
        "endpoints": {
            "POST /query": "Execute industry query (keyword + graph + semantic search)",
            "GET /query?q=...": "Query via GET",
            "GET /health": "Health check (all data sources)",
            "GET /stats": "Database stats (PostgreSQL + Vector + Cache)",
            "GET /metrics": "Prometheus metrics export",
            "GET /cache/metrics": "Cache hit/miss metrics and rates",
            "GET /cache/history": "Recent query history",
            "POST /cache/clear": "Clear query cache",
            "POST /cache/reset": "Reset metrics (keep cache)",
            "GET /issuers": "List all issuers",
            "GET /industries": "Get industries from PostgreSQL",
            "GET /graph/stats": "Neo4j graph statistics",
            "GET /graph/industries": "Top industries from graph",
            "GET /graph/issuers": "Top issuers from graph",
            "GET /graph/search?q=...": "Search knowledge graph",
            "GET /graph/industry/<name>": "Reports for an industry",
            "GET /graph/issuer/<name>": "Reports from an issuer",
            "GET /graph/cycle/<stage>": "Reports by cycle stage"
        }
    })


@app.route('/health')
def health():
    try:
        iq = get_industry_query()
        return jsonify({
            "status": "healthy",
            "postgresql": iq._pg_client is not None,
            "neo4j": iq._neo4j_client is not None,
            "vector": iq._vector_client is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/stats')
def stats():
    """Get database stats."""
    iq = get_industry_query()
    try:
        pg_stats = iq.pg_client.get_stats()
        vector_stats = iq.vector_client.get_stats() if iq.vector_client else {}
        cache_stats = iq.get_cache_stats()
        return jsonify({
            "postgresql": pg_stats,
            "vector": vector_stats,
            "cache": cache_stats,
            "status": "ready"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate AI research summary."""
    data = request.get_json() or {}
    question = data.get('question', '')
    reports = data.get('reports', [])
    max_reports = data.get('max_reports', 10)

    if not question:
        return jsonify({"success": False, "error": "Missing 'question'"}), 400

    try:
        if not reports:
            iq = get_industry_query()
            result = iq.query(question, max_reports=max_reports)
            reports = result.reports

        client = get_gemini_client()
        if not client:
            return jsonify({"success": False, "error": "Gemini API not configured"}), 503

        return jsonify(client.summarize_research(question, reports[:max_reports], []))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear query cache."""
    iq = get_industry_query()
    iq.clear_cache()
    return jsonify({"status": "cache cleared"})


@app.route('/cache/metrics')
def cache_metrics():
    """Get cache hit/miss metrics."""
    iq = get_industry_query()
    return jsonify({
        "metrics": iq.get_cache_metrics(),
        "cache": iq.get_cache_stats()
    })


@app.route('/cache/reset', methods=['POST'])
def reset_metrics():
    """Reset cache metrics (keep cache entries)."""
    iq = get_industry_query()
    iq.reset_metrics()
    return jsonify({"status": "metrics reset"})


@app.route('/cache/history')
def cache_history():
    """Get recent query history."""
    limit = int(request.args.get('limit', 20))
    iq = get_industry_query()
    return jsonify({
        "recent_queries": iq.get_recent_queries(limit),
        "total_logged": len(iq._recent_queries)
    })


@app.route('/metrics')
def prometheus_metrics():
    """Export Prometheus metrics."""
    # Ensure gauges are updated
    iq = get_industry_query()
    iq._update_prometheus_gauges()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/query', methods=['GET', 'POST'])
def query():
    # Get parameters
    if request.method == 'POST':
        data = request.get_json() or {}
        question = data.get('question', '')
        max_reports = data.get('max_reports', 20)
        verbose = data.get('verbose', False)
    else:
        question = request.args.get('q', '')
        max_reports = int(request.args.get('max_reports', 20))
        verbose = request.args.get('verbose', 'false').lower() == 'true'

    if not question:
        return jsonify({"error": "Missing 'question' or 'q' parameter"}), 400

    try:
        iq = get_industry_query()

        result = iq.query(
            question=question,
            max_reports=max_reports,
            verbose=verbose
        )

        response = {
            "success": True,
            "query": result.query,
            "intent": result.intent,
            "answer": result.answer,
            "sources_used": result.sources_used,
            "execution_time_ms": result.execution_time_ms,
            "reports": result.reports[:max_reports],
            "reports_count": len(result.reports),
            # Graph-enriched fields
            "related_industries": result.related_industries,
            "graph_insights": result.graph_insights,
            # Semantic search fields
            "semantic_matches": result.semantic_matches,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/issuers')
def issuers():
    """Get list of issuers."""
    iq = get_industry_query()
    try:
        issuers = iq.pg_client.get_issuers()
        return jsonify({"issuers": issuers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/industries')
def industries():
    """Get industries."""
    limit = int(request.args.get('limit', 50))

    iq = get_industry_query()
    try:
        industries = iq.pg_client.get_industries()
        return jsonify({"industries": industries[:limit]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Neo4j Graph Endpoints
# ============================================================

@app.route('/graph/stats')
def graph_stats():
    """Get Neo4j graph statistics."""
    try:
        nc = get_neo4j_client()
        stats = nc.get_stats()
        return jsonify({
            "status": "connected",
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/industries')
def graph_industries():
    """Get top industries from graph."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        industries = nc.get_top_industries(limit=limit)
        return jsonify({"industries": industries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/issuers')
def graph_issuers():
    """Get top issuers from graph."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        issuers = nc.get_top_issuers(limit=limit)
        return jsonify({"issuers": issuers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/cycles')
def graph_cycles():
    """Get cycle stage distribution from graph."""
    try:
        nc = get_neo4j_client()
        cycles = nc.get_cycle_distribution()
        return jsonify({"cycles": cycles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/search')
def graph_search():
    """Search the knowledge graph."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify({"error": "Missing 'q' parameter"}), 400

    try:
        nc = get_neo4j_client()
        results = nc.search_graph(query, limit=limit)
        return jsonify({
            "query": query,
            "results": results,
            "total_by_industry": len(results.get('by_industry', [])),
            "total_by_cycle": len(results.get('by_cycle', [])),
            "total_by_issuer": len(results.get('by_issuer', [])),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/industry/<industry>')
def graph_industry_reports(industry):
    """Get reports for a specific industry."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_industry(industry, limit=limit)
        related = nc.find_related_industries(industry, limit=5)
        return jsonify({
            "industry": industry,
            "reports": reports,
            "reports_count": len(reports),
            "related_industries": related
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/cycle/<cycle_stage>')
def graph_cycle_reports(cycle_stage):
    """Get reports for a specific cycle stage."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_cycle(cycle_stage, limit=limit)
        return jsonify({
            "cycle_stage": cycle_stage,
            "reports": reports,
            "reports_count": len(reports)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/demand/<demand_trend>')
def graph_demand_reports(demand_trend):
    """Get reports for a specific demand trend."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_demand(demand_trend, limit=limit)
        return jsonify({
            "demand_trend": demand_trend,
            "reports": reports,
            "reports_count": len(reports)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/issuer/<issuer>')
def graph_issuer_reports(issuer):
    """Get reports from a specific issuer."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_issuer(issuer, limit=limit)
        industries = nc.get_issuer_industries(issuer)
        return jsonify({
            "issuer": issuer,
            "reports": reports,
            "reports_count": len(reports),
            "top_industries": industries
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/theme-explorer')
def serve_theme_explorer():
    """Serve the standalone theme explorer page."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        theme_path = os.path.join(base_dir, '..', 'shared', 'theme-explorer', 'index.html')
        with open(theme_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error loading theme explorer: {e}", 500


if __name__ == '__main__':
    print("=" * 60)
    print("Industry Analysis IndustryQuery API Server (Flask)")
    print("=" * 60)
    print("Pre-loading models at startup...")
    print("=" * 60)

    # Pre-load IndustryQuery
    preload_industry_query()

    print("=" * 60)
    print("Server ready for queries!")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8002, debug=False, threaded=True)
