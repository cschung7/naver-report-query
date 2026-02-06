"""
Economic Analysis Query API Server (Flask)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from flask import Flask, request, jsonify, render_template
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global instances
eq = None
pg_client = None
vector_client = None
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


def get_eq():
    """Lazy load EconQuery."""
    global eq
    if eq is None:
        print("Loading EconQuery...")
        from query.router import EconQuery
        eq = EconQuery()
        print("EconQuery ready!")
    return eq


def get_pg():
    """Lazy load PostgresClient."""
    global pg_client
    if pg_client is None:
        print("Loading PostgresClient...")
        from query.router import PostgresClient
        pg_client = PostgresClient()
        print("PostgresClient ready!")
    return pg_client


def get_vector():
    """Lazy load VectorClient."""
    global vector_client
    if vector_client is None:
        print("Loading VectorClient...")
        from query.router import VectorClient
        vector_client = VectorClient()
        print("VectorClient ready!")
    return vector_client


@app.route('/')
def index():
    """Web UI."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    pg_ok = False
    vector_ok = False

    try:
        pg = get_pg()
        pg.conn.cursor().execute("SELECT 1")
        pg_ok = True
    except:
        pass

    try:
        vc = get_vector()
        vc.collection.count()
        vector_ok = True
    except:
        pass

    return jsonify({
        'status': 'healthy' if pg_ok else 'degraded',
        'postgresql': pg_ok,
        'vector': vector_ok,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/query', methods=['GET', 'POST'])
def query():
    """Execute economic analysis query."""
    if request.method == 'POST':
        data = request.get_json() or {}
        question = data.get('question', '')
        max_reports = data.get('max_reports', 10)
    else:
        question = request.args.get('q', '')
        max_reports = int(request.args.get('limit', 10))

    if not question:
        return jsonify({'success': False, 'error': 'No question provided'}), 400

    try:
        eq = get_eq()
        result = eq.query(question, max_reports=max_reports)

        cache_type = 'HIT' if 'HIT' in result.cache_hit else 'MISS'
        logger.info(f"[CACHE] {result.cache_hit} | {result.execution_time_ms}ms | {question}")

        return jsonify({
            'success': True,
            'query': result.query,
            'intent': result.intent,
            'reports': result.reports,
            'reports_count': len(result.reports),
            'answer': result.answer,
            'sources_used': result.sources_used,
            'execution_time_ms': result.execution_time_ms,
            'cache_hit': result.cache_hit,
        })

    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get database statistics."""
    try:
        pg = get_pg()
        pg_stats = pg.get_stats()

        vc = get_vector()
        vector_stats = vc.get_stats()

        eq = get_eq()
        cache_stats = eq.get_cache_metrics()

        return jsonify({
            'postgresql': pg_stats,
            'vector': vector_stats,
            'cache': cache_stats,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
            econ = get_eq()
            result = econ.query(question, max_reports=max_reports)
            reports = result.reports

        client = get_gemini_client()
        if not client:
            return jsonify({"success": False, "error": "Gemini API not configured"}), 503

        return jsonify(client.summarize_research(question, reports[:max_reports], []))
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/cache/metrics')
def cache_metrics():
    """Get cache metrics."""
    eq = get_eq()
    return jsonify(eq.get_cache_metrics())


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    """Clear cache."""
    eq = get_eq()
    eq.clear_cache()
    return jsonify({'status': 'cache cleared'})


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
    print("Economic Analysis EconQuery API Server (Flask)")
    print("=" * 60)

    # Pre-load models
    print("Pre-loading models at startup...")

    print("=" * 60)
    print("Loading EconQuery...")
    get_eq()

    print("Pre-loading PostgresClient...")
    pg = get_pg()
    try:
        stats = pg.get_stats()
        print(f"  PostgresClient ready - {stats['total_reports']} reports")
    except Exception as e:
        print(f"  PostgresClient error: {e}")

    print("Pre-loading VectorClient...")
    vc = get_vector()
    try:
        stats = vc.get_stats()
        print(f"  VectorClient ready - {stats['total_vectors']} vectors")
    except Exception as e:
        print(f"  VectorClient error: {e}")

    print("Loading embedding model: jhgan/ko-sbert-nli...")
    _ = vc.model
    print("Embedding model ready!")

    print("=" * 60)
    print("Server ready for queries!")
    print("=" * 60)

    app.run(host='0.0.0.0', port=8003, debug=False)
