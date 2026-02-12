"""
SmartQuery Unified Gateway API

Single entry point for all SmartQuery services:
- /firm/*     -> FirmAnalysis (8080)
- /invest/*   -> InvestmentStrategy (8001)
- /industry/* -> Industry (8002)
- /econ/*     -> EconAnalysis (8003)
- /themes/*   -> FirmAnalysis (theme data)
- /chart/*    -> FirmAnalysis (chart data)

Run with: python api/gateway.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import requests
from datetime import datetime

app = Flask(__name__, template_folder='templates')
CORS(app)

# Gateway port - uses Railway's PORT env var
GATEWAY_PORT = int(os.environ.get('PORT', 8000))

# Backend endpoints (internal services on localhost)
BACKENDS = {
    'firm': 'http://localhost:8080',
    'invest': 'http://localhost:8001',
    'industry': 'http://localhost:8002',
    'econ': 'http://localhost:8003'
}

DOMAIN_NAMES = {
    'firm': '기업분석',
    'invest': '투자전략',
    'industry': '산업분석',
    'econ': '경제분석'
}

# ============================================================
# Main Routes
# ============================================================

@app.route('/')
def index():
    """Serve unified UI."""
    return render_template('index.html')


@app.route('/analysis')
@app.route('/analysis/<path:path>')
def proxy_analysis(path=''):
    """Proxy to FirmAnalysis /analysis page and sub-routes."""
    try:
        url = f"{BACKENDS['firm']}/analysis"
        if path:
            url += f"/{path}"
        resp = requests.get(url, params=request.args, timeout=30)
        return Response(resp.content, status=resp.status_code,
                       content_type=resp.headers.get('content-type', 'text/html'))
    except Exception as e:
        return f"Error: {e}", 502


@app.route('/industry')
@app.route('/industry/<path:path>')
def proxy_industry_page(path=''):
    """Proxy to FirmAnalysis /industry page and sub-routes."""
    try:
        url = f"{BACKENDS['firm']}/industry"
        if path:
            url += f"/{path}"
        resp = requests.get(url, params=request.args, timeout=30)
        return Response(resp.content, status=resp.status_code,
                       content_type=resp.headers.get('content-type', 'text/html'))
    except Exception as e:
        return f"Error: {e}", 502


@app.route('/strategy')
@app.route('/strategy/<path:path>')
def proxy_strategy_page(path=''):
    """Proxy to FirmAnalysis /strategy page and sub-routes."""
    try:
        url = f"{BACKENDS['firm']}/strategy"
        if path:
            url += f"/{path}"
        resp = requests.get(url, params=request.args, timeout=30)
        return Response(resp.content, status=resp.status_code,
                       content_type=resp.headers.get('content-type', 'text/html'))
    except Exception as e:
        return f"Error: {e}", 502


@app.route('/economy')
@app.route('/economy/<path:path>')
def proxy_economy_page(path=''):
    """Proxy to FirmAnalysis /economy page and sub-routes."""
    try:
        url = f"{BACKENDS['firm']}/economy"
        if path:
            url += f"/{path}"
        resp = requests.get(url, params=request.args, timeout=30)
        return Response(resp.content, status=resp.status_code,
                       content_type=resp.headers.get('content-type', 'text/html'))
    except Exception as e:
        return f"Error: {e}", 502


@app.route('/api')
def api_info():
    """API information endpoint."""
    return jsonify({
        "name": "SmartQuery Unified Gateway",
        "version": "1.0.0",
        "description": "Unified gateway for all SmartQuery services",
        "domains": DOMAIN_NAMES,
        "endpoints": {
            "GET /": "Unified Web UI",
            "GET /health": "Gateway and backend health status",
            "GET /api/<domain>/query": "Query specific domain (firm/invest/industry/econ)",
            "POST /api/<domain>/query": "Query specific domain with JSON body",
            "GET /api/<domain>/stats": "Get domain statistics",
            "GET /themes/stock": "Get themes for a stock (proxied to firm)",
            "GET /themes/stocks": "Get stocks for a theme (proxied to firm)",
            "GET /chart/ohlcv": "Get OHLCV chart data (proxied to firm)",
            "GET /theme-explorer": "Theme analysis mindmap UI"
        }
    })


@app.route('/health')
def health():
    """Check health of gateway and all backends."""
    results = {"gateway": "healthy", "timestamp": datetime.now().isoformat()}

    for domain, url in BACKENDS.items():
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                results[domain] = resp.json()
            else:
                results[domain] = {"status": "error", "code": resp.status_code}
        except Exception as e:
            results[domain] = {"status": "unreachable", "error": str(e)}

    return jsonify(results)


# ============================================================
# Domain Query Proxy
# ============================================================

@app.route('/api/<domain>/query', methods=['GET', 'POST'])
def proxy_query(domain):
    """Proxy query requests to appropriate backend."""
    backend = BACKENDS.get(domain)
    if not backend:
        return jsonify({"error": f"Unknown domain: {domain}", "valid_domains": list(BACKENDS.keys())}), 400

    try:
        if request.method == 'POST':
            resp = requests.post(
                f"{backend}/query",
                json=request.get_json(),
                timeout=120
            )
        else:
            resp = requests.get(
                f"{backend}/query",
                params=request.args,
                timeout=120
            )

        result = resp.json()
        result['_gateway'] = {"domain": domain, "backend": backend}
        return jsonify(result)

    except requests.Timeout:
        return jsonify({"error": "Backend timeout", "domain": domain}), 504
    except Exception as e:
        return jsonify({"error": str(e), "domain": domain}), 502


@app.route('/api/<domain>/stats')
def proxy_stats(domain):
    """Proxy stats requests to appropriate backend."""
    backend = BACKENDS.get(domain)
    if not backend:
        return jsonify({"error": f"Unknown domain: {domain}"}), 400

    try:
        resp = requests.get(f"{backend}/stats", timeout=30)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route('/api/<domain>/summarize', methods=['POST'])
def proxy_summarize(domain):
    """Proxy summarize requests to appropriate backend."""
    backend = BACKENDS.get(domain)
    if not backend:
        return jsonify({"error": f"Unknown domain: {domain}"}), 400

    try:
        resp = requests.post(
            f"{backend}/summarize",
            json=request.get_json(),
            timeout=180
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ============================================================
# Theme Endpoints (Proxied to FirmAnalysis)
# ============================================================

@app.route('/themes/stock')
def proxy_themes_stock():
    """Get themes for a specific stock."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/themes/stock",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 502


@app.route('/themes/stocks')
def proxy_themes_stocks():
    """Get stocks for a specific theme."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/themes/stocks",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 502


@app.route('/themes/search')
def proxy_themes_search():
    """Search themes."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/themes/search",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 502


# ============================================================
# Chart Endpoints (Proxied to FirmAnalysis)
# ============================================================

@app.route('/chart/ohlcv')
def proxy_chart_ohlcv():
    """Get OHLCV chart data."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/chart/ohlcv",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 502


# ============================================================
# Theme Explorer
# ============================================================

@app.route('/theme-explorer')
def serve_theme_explorer():
    """Serve the standalone theme explorer page."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        theme_path = os.path.join(base_dir, '..', 'shared', 'theme-explorer', 'index.html')
        with open(theme_path, 'r') as f:
            content = f.read()
            content = content.replace('const API_PORT = 8080;', f'const API_PORT = {GATEWAY_PORT};')
            return content
    except Exception as e:
        return f"Error loading theme explorer: {e}", 500


# ============================================================
# Stock Info (Proxied to FirmAnalysis)
# ============================================================

@app.route('/stock')
def proxy_stock():
    """Get stock information page."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/stock",
            params=request.args,
            timeout=30
        )
        return Response(resp.content, content_type=resp.headers.get('content-type', 'text/html'))
    except Exception as e:
        return f"Error: {e}", 502


@app.route('/stock/themes')
def proxy_stock_themes():
    """Get stock themes."""
    try:
        resp = requests.get(
            f"{BACKENDS['firm']}/stock/themes",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ============================================================
# Graph Endpoints (Proxied by domain)
# ============================================================

@app.route('/api/<domain>/graph/<path:path>')
def proxy_graph(domain, path):
    """Proxy graph requests to appropriate backend."""
    backend = BACKENDS.get(domain)
    if not backend:
        return jsonify({"error": f"Unknown domain: {domain}"}), 400

    try:
        resp = requests.get(
            f"{backend}/graph/{path}",
            params=request.args,
            timeout=30
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == '__main__':
    print("=" * 60)
    print("SmartQuery Unified Gateway")
    print("=" * 60)
    print(f"Port: {GATEWAY_PORT}")
    print(f"Backends: {BACKENDS}")
    print("=" * 60)

    app.run(host='0.0.0.0', port=GATEWAY_PORT, debug=False, threaded=True)
