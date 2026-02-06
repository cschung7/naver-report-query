"""
Simple SmartQuery API for Investment Strategy - Sync version

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
smart_query = None
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


def get_smart_query():
    """Lazy-load SmartQuery."""
    global smart_query
    if smart_query is None:
        print("Loading SmartQuery...", flush=True)
        from query.router import SmartQuery
        smart_query = SmartQuery()
        print("SmartQuery ready!", flush=True)
    return smart_query


def get_neo4j_client():
    """Lazy-load Neo4j client."""
    global neo4j_client
    if neo4j_client is None:
        print("Loading Neo4jClient...", flush=True)
        from query.router.neo4j_client import Neo4jClient
        neo4j_client = Neo4jClient()
        print("Neo4jClient ready!", flush=True)
    return neo4j_client


def preload_smart_query():
    """Pre-load SmartQuery at startup."""
    sq = get_smart_query()

    # Force initialization of PostgreSQL
    print("Pre-loading PostgresClient...", flush=True)
    if sq.pg_client:
        try:
            stats = sq.pg_client.get_stats()
            print(f"  PostgresClient ready - {stats.get('total_reports', 0)} reports", flush=True)
        except Exception as e:
            print(f"  PostgresClient error: {e}", flush=True)

    # Pre-load Neo4j client
    print("Pre-loading Neo4jClient...", flush=True)
    try:
        nc = get_neo4j_client()
        graph_stats = nc.get_stats()
        print(f"  Neo4jClient ready - {graph_stats.get('strategyreport_count', 0)} nodes", flush=True)
    except Exception as e:
        print(f"  Neo4jClient error: {e}", flush=True)

    # Pre-load Vector client and embedding model
    print("Pre-loading VectorClient...", flush=True)
    try:
        if sq.vector_client:
            vector_stats = sq.vector_client.get_stats()
            print(f"  VectorClient ready - {vector_stats.get('total_documents', 0)} vectors", flush=True)
            # Force-load embedding model for faster first query
            sq.vector_client.warmup()
    except Exception as e:
        print(f"  VectorClient error: {e}", flush=True)

    return sq


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.route('/api')
def api_info():
    """API information endpoint."""
    return jsonify({
        "name": "Investment Strategy SmartQuery API",
        "version": "1.5.0",
        "description": "Multi-source search: PostgreSQL + Neo4j + ChromaDB + Semantic Cache + Prometheus",
        "endpoints": {
            "POST /query": "Execute smart query (keyword + graph + semantic search)",
            "GET /query?q=...": "Query via GET",
            "GET /health": "Health check (all data sources)",
            "GET /stats": "Database stats (PostgreSQL + Vector + Cache)",
            "GET /metrics": "Prometheus metrics export",
            "GET /cache/metrics": "Cache hit/miss metrics and rates",
            "GET /cache/history": "Recent query history",
            "POST /cache/clear": "Clear query cache",
            "POST /cache/reset": "Reset metrics (keep cache)",
            "GET /issuers": "List all issuers",
            "GET /themes": "Get themes from PostgreSQL",
            "GET /graph/stats": "Neo4j graph statistics",
            "GET /graph/themes": "Top themes from graph",
            "GET /graph/sectors": "Top sectors from graph",
            "GET /graph/search?q=...": "Search knowledge graph",
            "GET /graph/theme/<name>": "Reports for a theme",
            "GET /graph/sector/<name>": "Reports for a sector",
            "GET /graph/issuer/<name>": "Reports from an issuer",
            "GET /graph/outlook/<outlook>": "Reports by market outlook"
        }
    })


@app.route('/health')
def health():
    try:
        sq = get_smart_query()
        return jsonify({
            "status": "healthy",
            "postgresql": sq._pg_client is not None,
            "neo4j": sq._neo4j_client is not None,
            "vector": sq._vector_client is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/stats')
def stats():
    """Get database stats."""
    sq = get_smart_query()
    try:
        pg_stats = sq.pg_client.get_stats()
        vector_stats = sq.vector_client.get_stats() if sq.vector_client else {}
        cache_stats = sq.get_cache_stats()
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
    claims = data.get('claims', [])
    max_reports = data.get('max_reports', 10)

    if not question:
        return jsonify({"success": False, "error": "Missing 'question'"}), 400

    try:
        if not reports:
            sq = get_smart_query()
            result = sq.query(question, max_reports=max_reports)
            reports = result.reports
            claims = getattr(result, 'claims', [])

        client = get_gemini_client()
        if not client:
            return jsonify({"success": False, "error": "Gemini API not configured"}), 503

        return jsonify(client.summarize_research(question, reports[:max_reports], claims))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear query cache."""
    sq = get_smart_query()
    sq.clear_cache()
    return jsonify({"status": "cache cleared"})


@app.route('/cache/metrics')
def cache_metrics():
    """Get cache hit/miss metrics."""
    sq = get_smart_query()
    return jsonify({
        "metrics": sq.get_cache_metrics(),
        "cache": sq.get_cache_stats()
    })


@app.route('/cache/reset', methods=['POST'])
def reset_metrics():
    """Reset cache metrics (keep cache entries)."""
    sq = get_smart_query()
    sq.reset_metrics()
    return jsonify({"status": "metrics reset"})


@app.route('/cache/history')
def cache_history():
    """Get recent query history."""
    limit = int(request.args.get('limit', 20))
    sq = get_smart_query()
    return jsonify({
        "recent_queries": sq.get_recent_queries(limit),
        "total_logged": len(sq._recent_queries)
    })


@app.route('/metrics')
def prometheus_metrics():
    """Export Prometheus metrics."""
    # Ensure gauges are updated
    sq = get_smart_query()
    sq._update_prometheus_gauges()
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
        sq = get_smart_query()

        result = sq.query(
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
            "themes_count": len(result.themes),
            # Graph-enriched fields
            "related_themes": result.related_themes,
            "related_sectors": result.related_sectors,
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
    sq = get_smart_query()
    try:
        issuers = sq.pg_client.get_issuers()
        return jsonify({"issuers": issuers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/themes')
def themes():
    """Get themes."""
    theme_name = request.args.get('name', '')
    limit = int(request.args.get('limit', 50))

    sq = get_smart_query()
    try:
        themes = sq.pg_client.get_themes(theme_name=theme_name if theme_name else None, limit=limit)
        return jsonify({"themes": themes})
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


@app.route('/graph/themes')
def graph_themes():
    """Get top themes from graph."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        themes = nc.get_top_themes(limit=limit)
        return jsonify({"themes": themes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/sectors')
def graph_sectors():
    """Get top sectors from graph."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        sectors = nc.get_top_sectors(limit=limit)
        return jsonify({"sectors": sectors})
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
            "total_by_theme": len(results.get('by_theme', [])),
            "total_by_sector": len(results.get('by_sector', [])),
            "total_by_geography": len(results.get('by_geography', [])),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/theme/<theme>')
def graph_theme_reports(theme):
    """Get reports for a specific theme."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_theme(theme, limit=limit)
        related = nc.find_related_themes(theme, limit=5)
        return jsonify({
            "theme": theme,
            "reports": reports,
            "reports_count": len(reports),
            "related_themes": related
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/sector/<sector>')
def graph_sector_reports(sector):
    """Get reports for a specific sector."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_sector(sector, limit=limit)
        related = nc.find_related_sectors(sector, limit=5)
        return jsonify({
            "sector": sector,
            "reports": reports,
            "reports_count": len(reports),
            "related_sectors": related
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
        themes = nc.get_issuer_themes(issuer)
        return jsonify({
            "issuer": issuer,
            "reports": reports,
            "reports_count": len(reports),
            "top_themes": themes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/graph/outlook/<outlook>')
def graph_outlook_reports(outlook):
    """Get reports with a specific market outlook."""
    limit = int(request.args.get('limit', 20))
    try:
        nc = get_neo4j_client()
        reports = nc.get_reports_by_outlook(outlook, limit=limit)
        return jsonify({
            "outlook": outlook,
            "reports": reports,
            "reports_count": len(reports)
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
    print("Investment Strategy SmartQuery API Server (Flask)")
    print("=" * 60)
    print("Pre-loading models at startup...")
    print("=" * 60)

    # Pre-load SmartQuery
    preload_smart_query()

    print("=" * 60)
    print("Server ready for queries!")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8001, debug=False, threaded=True)
