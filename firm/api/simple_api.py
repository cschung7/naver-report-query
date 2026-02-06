"""
Simple SmartQuery API - Sync version

Run with: python api/simple_api.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, render_template_string
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import json

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes (needed for theme-explorer cross-port access)

# ============================================================
# Data paths - use bundled data/ directory with NAS fallback
# ============================================================
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_ROOT)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

def _data_path(filename, nas_fallback=None):
    """Resolve data file path: bundled data/ dir first, NAS fallback second."""
    bundled = os.path.join(_DATA_DIR, filename)
    if os.path.exists(bundled):
        return bundled
    if nas_fallback and os.path.exists(nas_fallback):
        return nas_fallback
    return bundled  # Return bundled path even if missing (for error messages)

# ============================================================
# Fundamental Data & Company Overview Cache
# ============================================================
_fundamental_data = None
_company_overview = None
_ticker_name_map = None


def get_fundamental_data():
    """Load fundamental data from JSON file."""
    global _fundamental_data
    if _fundamental_data is None:
        try:
            with open(_data_path('fundamental_data.json', '/mnt/nas/AutoGluon/AutoML_Krx/DB/fundamental_data.json'), 'r') as f:
                _fundamental_data = json.load(f)
            print(f"  Fundamental data loaded: {len(_fundamental_data)} entries")
        except Exception as e:
            print(f"  Warning: Could not load fundamental data: {e}")
            _fundamental_data = {}
    return _fundamental_data


def get_company_overview():
    """Load company overview from CSV file."""
    global _company_overview
    if _company_overview is None:
        try:
            df = pd.read_csv(_data_path('company_overview.csv', '/mnt/nas/AutoGluon/AutoML_Krx/DB/company_overview.csv'))
            # Create dict mapping ticker to overview
            _company_overview = {}
            for _, row in df.iterrows():
                ticker = str(row['tickers']).zfill(6)
                _company_overview[ticker] = row['company overview']
            print(f"  Company overview loaded: {len(_company_overview)} entries")
        except Exception as e:
            print(f"  Warning: Could not load company overview: {e}")
            _company_overview = {}
    return _company_overview


def get_ticker_name_map():
    """Load ticker to name mapping from db_final.csv."""
    global _ticker_name_map
    if _ticker_name_map is None:
        try:
            df = pd.read_csv(_data_path('db_final.csv', '/mnt/nas/AutoGluon/AutoML_Krx/DB/db_final.csv'))
            _ticker_name_map = {}
            for _, row in df.iterrows():
                ticker = str(row['tickers']).zfill(6)
                name = row['name']
                _ticker_name_map[name] = ticker
                _ticker_name_map[ticker] = name
            print(f"  Ticker-name map loaded: {len(_ticker_name_map)//2} stocks")
        except Exception as e:
            print(f"  Warning: Could not load ticker-name map: {e}")
            _ticker_name_map = {}
    return _ticker_name_map


def get_stock_fundamental(stock_name):
    """Get fundamental data for a stock by name or ticker."""
    fundamentals = get_fundamental_data()
    overview_data = get_company_overview()
    name_map = get_ticker_name_map()

    # Determine ticker
    ticker = None
    if stock_name in name_map:
        if stock_name.isdigit() or (len(stock_name) == 6 and stock_name[0].isdigit()):
            ticker = stock_name.zfill(6)
        else:
            ticker = name_map.get(stock_name, '').zfill(6) if name_map.get(stock_name) else None

    if not ticker:
        return None

    result = {
        "ticker": ticker,
        "fundamental": fundamentals.get(ticker, {}),
        "overview": overview_data.get(ticker, "")
    }
    return result


# ============================================================
# ETF Data Cache
# ============================================================
_ticker_to_etf_map = None
_etf_details = None


def get_ticker_to_etf_map():
    """Load ticker to ETF mapping."""
    global _ticker_to_etf_map
    if _ticker_to_etf_map is None:
        try:
            with open(_data_path('domestic_ticker_to_etfs_map.json', '/mnt/nas/AutoGluon/AutoML_KrxETF/DB/domestic_ticker_to_etfs_map.json'), 'r') as f:
                _ticker_to_etf_map = json.load(f)
            print(f"  Ticker-ETF map loaded: {len(_ticker_to_etf_map)} stocks")
        except Exception as e:
            print(f"  Warning: Could not load ticker-ETF map: {e}")
            _ticker_to_etf_map = {}
    return _ticker_to_etf_map


def get_etf_details():
    """Load ETF details from Research DB."""
    global _etf_details
    if _etf_details is None:
        try:
            with open(_data_path('etf_by_code.json', '/mnt/nas/AutoGluon/AutoML_KrxETF/Research/DB/etf_by_code.json'), 'r') as f:
                _etf_details = json.load(f)
            print(f"  ETF details loaded: {len(_etf_details)} ETFs")
        except Exception as e:
            print(f"  Warning: Could not load ETF details: {e}")
            _etf_details = {}
    return _etf_details


def get_stock_etfs(stock_name):
    """Get all ETFs containing a specific stock."""
    ticker_etf_map = get_ticker_to_etf_map()
    etf_details = get_etf_details()

    etf_list = ticker_etf_map.get(stock_name, [])
    if not etf_list:
        return []

    # Get unique ETFs with details
    unique_etfs = {}
    for etf in etf_list:
        code = etf.get('itemcode', '')
        if code and code not in unique_etfs:
            details = etf_details.get(code, {})
            unique_etfs[code] = {
                'code': code,
                'name': etf.get('itemname', ''),
                'description': details.get('description', '')[:100] if details else '',
                'issuing_date': details.get('issuingDate', '') if details else ''
            }

    return list(unique_etfs.values())


# Global instances
smart_query = None
query_refiner = None
gemini_client = None
naver_theme_service = None


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


def get_naver_theme_service():
    """Lazy-load NaverThemeService."""
    global naver_theme_service
    if naver_theme_service is None:
        try:
            from query.naver_theme import NaverThemeService
            naver_theme_service = NaverThemeService
        except Exception as e:
            print(f"Warning: NaverThemeService not available: {e}")
            return None
    return naver_theme_service


def get_query_refiner():
    """Lazy-load QueryRefiner."""
    global query_refiner
    if query_refiner is None:
        from query.router.query_refiner import QueryRefiner
        query_refiner = QueryRefiner()
    return query_refiner


def get_smart_query():
    """Lazy-load SmartQuery."""
    global smart_query
    if smart_query is None:
        print("Loading SmartQuery (~2min)...", flush=True)
        from query.router import SmartQuery
        smart_query = SmartQuery()
        print("✅ SmartQuery ready!", flush=True)
    return smart_query


def preload_smart_query():
    """Pre-load SmartQuery at startup (called by gunicorn preload)."""
    sq = get_smart_query()

    # Force initialization of PostgreSQL and Neo4j only (Vector is too slow)
    print("Pre-loading PostgresClient...", flush=True)
    if sq.pg_client:
        try:
            sq.pg_client.search_reports(company='삼성전자', limit=1)
            print("  PostgresClient ready", flush=True)
        except Exception as e:
            print(f"  PostgresClient error: {e}", flush=True)

    print("Pre-loading Neo4jClient...", flush=True)
    if sq.neo4j_client:
        try:
            sq.neo4j_client.get_company_claims(company='삼성전자', limit=1)
            print("  Neo4jClient ready", flush=True)
        except Exception as e:
            print(f"  Neo4jClient error: {e}", flush=True)

    # Note: VectorClient is loaded lazily on first use_vector=true request
    print("VectorClient: lazy-loaded on demand (use_vector=true)", flush=True)

    # Pre-load fundamental data
    print("Pre-loading Fundamental Data...", flush=True)
    get_fundamental_data()
    get_company_overview()
    get_ticker_name_map()

    # Pre-load ETF data
    print("Pre-loading ETF Data...", flush=True)
    get_ticker_to_etf_map()
    get_etf_details()

    return sq


# Pre-load when running with gunicorn --preload or gunicorn_config.py
# Note: Set PRELOAD_SMARTQUERY=false if using gunicorn with multiple workers
if os.environ.get('PRELOAD_SMARTQUERY', 'false').lower() == 'true':
    print("Pre-loading SmartQuery at import time...", flush=True)
    preload_smart_query()
    print("All clients pre-loaded. Server ready for queries.", flush=True)


@app.route('/')
def index():
    """Serve the web UI."""
    return render_template('index.html')


@app.route('/stock')
def stock_detail():
    """Serve the stock detail page."""
    return render_template('stock_detail.html')


@app.route('/stock/etfs')
def stock_etfs():
    """
    Get all ETFs containing a specific stock.

    Example: /stock/etfs?name=삼성전자

    Response: {
        "success": true,
        "stock_name": "삼성전자",
        "etf_count": 80,
        "etfs": [
            {"code": "457930", "name": "BNK 미래전략기술액티브", "description": "..."},
            ...
        ]
    }
    """
    stock_name = request.args.get('name', '')
    limit = int(request.args.get('limit', 20))

    if not stock_name:
        return jsonify({"success": False, "error": "Missing 'name' parameter"}), 400

    try:
        etfs = get_stock_etfs(stock_name)

        return jsonify({
            "success": True,
            "stock_name": stock_name,
            "etf_count": len(etfs),
            "etfs": etfs[:limit]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/stock/info')
def stock_info():
    """
    Get comprehensive stock information including fundamentals and overview.

    Example: /stock/info?name=삼성전자

    Response: {
        "success": true,
        "stock_name": "삼성전자",
        "ticker": "005930",
        "fundamental": {
            "PER": 75.36,
            "PBR": 0.92,
            "EPS": 12.00,
            "BPS": 0.93,
            "MarketCap": 31729.26
        },
        "overview": "동사는 ..."
    }
    """
    stock_name = request.args.get('name', '')

    if not stock_name:
        return jsonify({"success": False, "error": "Missing 'name' parameter"}), 400

    try:
        result = get_stock_fundamental(stock_name)

        if not result:
            return jsonify({
                "success": False,
                "error": f"Stock not found: {stock_name}"
            }), 404

        return jsonify({
            "success": True,
            "stock_name": stock_name,
            "ticker": result["ticker"],
            "fundamental": result["fundamental"],
            "overview": result["overview"]
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/chart/ohlcv')
def chart_ohlcv():
    """Get OHLCV chart data for a stock with technical indicators."""
    stock_name = request.args.get('name', '')
    days = int(request.args.get('days', 90))  # Default 90 days

    if not stock_name:
        return jsonify({"success": False, "error": "Stock name required"}), 400

    # CSV file path - chart data (NAS only)
    krx_data_dir = os.environ.get('KRX_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_Krx/KRXNOTTRAINED')
    csv_path = f"{krx_data_dir}/{stock_name}.csv"

    try:
        df_full = None

        # Try local CSV first
        if os.path.exists(csv_path):
            df_full = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            # Fallback: Yahoo Finance
            df_full = _fetch_yahoo_ohlcv(stock_name, days)

        if df_full is None or df_full.empty:
            return jsonify({"success": False, "error": f"Data not found for {stock_name}"}), 404

        # Ensure lowercase column names
        df_full.columns = [c.lower() for c in df_full.columns]

        # Calculate indicators on full data, then slice
        df_full['ma10'] = df_full['close'].rolling(window=10).mean()
        df_full['ma20'] = df_full['close'].rolling(window=20).mean()
        df_full['ma60'] = df_full['close'].rolling(window=60).mean()

        # RSI (14 period)
        delta = df_full['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_full['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2 std)
        df_full['bb_middle'] = df_full['close'].rolling(window=20).mean()
        bb_std = df_full['close'].rolling(window=20).std()
        df_full['bb_upper'] = df_full['bb_middle'] + (bb_std * 2)
        df_full['bb_lower'] = df_full['bb_middle'] - (bb_std * 2)

        # Get last N days
        df = df_full.tail(days)

        # Format data for charting (lightweight-charts format)
        data = []
        ma10_data = []
        ma20_data = []
        ma60_data = []
        rsi_data = []
        bb_upper_data = []
        bb_middle_data = []
        bb_lower_data = []

        for date, row in df.iterrows():
            time_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

            data.append({
                "time": time_str,
                "open": round(float(row['open']), 0),
                "high": round(float(row['high']), 0),
                "low": round(float(row['low']), 0),
                "close": round(float(row['close']), 0),
                "volume": int(row['volume'])
            })

            if pd.notna(row['ma10']):
                ma10_data.append({"time": time_str, "value": round(float(row['ma10']), 0)})
            if pd.notna(row['ma20']):
                ma20_data.append({"time": time_str, "value": round(float(row['ma20']), 0)})
            if pd.notna(row['ma60']):
                ma60_data.append({"time": time_str, "value": round(float(row['ma60']), 0)})

            if pd.notna(row['rsi']):
                rsi_data.append({"time": time_str, "value": round(float(row['rsi']), 2)})

            if pd.notna(row.get('bb_upper', float('nan'))):
                bb_upper_data.append({"time": time_str, "value": round(float(row['bb_upper']), 0)})
                bb_middle_data.append({"time": time_str, "value": round(float(row['bb_middle']), 0)})
                bb_lower_data.append({"time": time_str, "value": round(float(row['bb_lower']), 0)})

        return jsonify({
            "success": True,
            "symbol": stock_name,
            "market": "KRX",
            "count": len(data),
            "ohlcv": data,
            "indicators": {
                "ma10": ma10_data,
                "ma20": ma20_data,
                "ma60": ma60_data,
                "rsi": rsi_data,
                "bb_upper": bb_upper_data,
                "bb_middle": bb_middle_data,
                "bb_lower": bb_lower_data
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _fetch_yahoo_ohlcv(stock_name, days):
    """Fetch OHLCV data from Yahoo Finance as fallback.

    Maps Korean stock name to KRX ticker (e.g. 005930.KS) using db_final.csv.
    Fetches extra data for indicator calculation.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed, cannot fetch chart data")
        return None

    # Resolve stock name to ticker code
    name_map = get_ticker_name_map()
    ticker_code = None

    if stock_name in name_map:
        val = name_map[stock_name]
        # name_map stores both name->code and code->name; we need the 6-digit code
        if val.isdigit() or (len(val) == 6 and val[0].isdigit()):
            ticker_code = val.zfill(6)
        else:
            # stock_name is already a ticker code
            ticker_code = stock_name.zfill(6)

    if not ticker_code:
        # Try treating input as ticker directly
        if stock_name.isdigit():
            ticker_code = stock_name.zfill(6)
        else:
            print(f"  Cannot resolve '{stock_name}' to KRX ticker")
            return None

    yahoo_ticker = f"{ticker_code}.KS"
    # Fetch extra days for indicator warmup (60-day MA needs history)
    fetch_days = days + 120

    try:
        period = f"{fetch_days}d"
        df = yf.download(yahoo_ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            # Try KOSDAQ suffix
            yahoo_ticker = f"{ticker_code}.KQ"
            df = yf.download(yahoo_ticker, period=period, progress=False, auto_adjust=True)

        if df.empty:
            print(f"  Yahoo Finance: no data for {yahoo_ticker}")
            return None

        # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]
        print(f"  Yahoo Finance: fetched {len(df)} rows for {yahoo_ticker}")
        return df

    except Exception as e:
        print(f"  Yahoo Finance error for {yahoo_ticker}: {e}")
        return None


@app.route('/api')
def api_info():
    """API information endpoint."""
    return jsonify({
        "name": "SmartQuery API",
        "version": "1.2.0",
        "endpoints": {
            "POST /query": "Execute smart query (auto-refines casual queries)",
            "GET /query?q=...": "Query via GET",
            "POST /refine": "Refine casual query to financial jargon",
            "POST /refine/clarify": "Refine with clarification response",
            "GET /suggest?q=...": "Get query suggestions for autocomplete",
            "GET /stock/info?name=...": "Get stock fundamentals and overview",
            "GET /stock/etfs?name=...": "Get ETFs containing a stock",
            "GET /chart/ohlcv?name=...&days=...": "Get OHLCV chart data with indicators",
            "GET /health": "Health check",
            "GET /stats": "Vector DB stats"
        },
        "features": {
            "auto_refine": "Automatically converts casual queries",
            "clarification": "Asks for clarification when query is ambiguous",
            "technical_indicators": "MA10/20/60, RSI, Bollinger Bands"
        }
    })


@app.route('/health')
def health():
    try:
        sq = get_smart_query()
        # Don't load VectorClient for health check (too slow)
        return jsonify({
            "status": "healthy",
            "postgresql": sq._pg_client is not None,
            "neo4j": sq._neo4j_client is not None,
            "vector_db": sq._vector_client is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/stats')
def stats():
    """Get stats including report counts."""
    sq = get_smart_query()
    result = {}

    # PostgreSQL stats with counts
    if sq._pg_client:
        try:
            pg_stats = sq.pg_client.get_stats()
            result["postgresql"] = pg_stats
        except Exception as e:
            result["postgresql"] = {"status": "error", "error": str(e)}
    else:
        result["postgresql"] = {"status": "not_loaded"}

    # Neo4j status
    result["neo4j"] = "ready" if sq._neo4j_client else "not_loaded"

    # Vector DB stats (only if loaded)
    if sq._vector_client:
        result["vector_db"] = sq.vector_client.get_stats()
    else:
        result["vector_db"] = "not_loaded"

    return jsonify(result)


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Generate a research summary from search results using Gemini.

    Request: {
        "question": "삼성전자 HBM 전망",
        "reports": [...],  # Optional: pre-fetched reports
        "claims": [...],   # Optional: pre-fetched claims
        "max_reports": 5   # Optional: limit reports to summarize
    }

    Response: {
        "success": true,
        "summary": "...",
        "references": [...],
        "model": "gemini-1.5-flash"
    }
    """
    data = request.get_json() or {}
    question = data.get('question', '')
    reports = data.get('reports', [])
    claims = data.get('claims', [])
    max_reports = data.get('max_reports', 10)

    if not question:
        return jsonify({"success": False, "error": "Missing 'question' parameter"}), 400

    try:
        # If no reports provided, fetch them first
        if not reports:
            sq = get_smart_query()
            result = sq.query(question, max_reports=max_reports)
            reports = result.reports
            claims = result.claims if hasattr(result, 'claims') else []

        # Get Gemini client and generate summary
        client = get_gemini_client()
        if not client:
            return jsonify({
                "success": False,
                "error": "Gemini API not configured. Set GEMINI_API_KEY environment variable."
            }), 503

        summary_result = client.summarize_research(
            query=question,
            reports=reports[:max_reports],
            claims=claims
        )

        return jsonify(summary_result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/refine', methods=['POST'])
def refine():
    """
    Refine a casual query into financial jargon.

    Request: {"question": "삼전 어때?"}
    Response: {
        "original": "삼전 어때?",
        "refined": "삼성전자 투자의견",
        "transformations": [...],
        "needs_clarification": false,
        "clarification_prompt": null,
        "clarification_options": []
    }
    """
    data = request.get_json() or {}
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Missing 'question' parameter"}), 400

    try:
        refiner = get_query_refiner()
        result = refiner.refine(question)

        return jsonify({
            "original": result.original_query,
            "refined": result.refined_query,
            "transformations": result.transformations,
            "confidence": result.confidence,
            "needs_clarification": result.needs_clarification,
            "clarification_prompt": result.clarification_prompt,
            "clarification_options": result.clarification_options,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/refine/clarify', methods=['POST'])
def refine_clarify():
    """
    Refine query with user's clarification response.

    Request: {"question": "좋은 바이오 주식", "clarification": "저평가"}
    """
    data = request.get_json() or {}
    question = data.get('question', '')
    clarification = data.get('clarification', '')

    if not question or not clarification:
        return jsonify({"error": "Missing 'question' or 'clarification' parameter"}), 400

    try:
        refiner = get_query_refiner()
        result = refiner.refine_with_clarification(question, clarification)

        return jsonify({
            "original": result.original_query,
            "refined": result.refined_query,
            "transformations": result.transformations,
            "confidence": result.confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/suggest', methods=['GET'])
def suggest():
    """Get query suggestions for autocomplete."""
    partial = request.args.get('q', '')

    if not partial or len(partial) < 2:
        return jsonify({"suggestions": []})

    try:
        refiner = get_query_refiner()
        suggestions = refiner.get_suggestions(partial)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        return jsonify({"suggestions": [], "error": str(e)})


@app.route('/query', methods=['GET', 'POST'])
def query():
    # Get parameters
    if request.method == 'POST':
        data = request.get_json() or {}
        question = data.get('question', '')
        max_reports = data.get('max_reports', 20)
        max_claims = data.get('max_claims', 50)
        verbose = data.get('verbose', False)
        auto_refine = data.get('auto_refine', True)  # Auto-refine by default
        use_vector = data.get('use_vector', False)  # Vector DB disabled by default (slow)
    else:
        question = request.args.get('q', '')
        max_reports = int(request.args.get('max_reports', 20))
        max_claims = int(request.args.get('max_claims', 50))
        verbose = request.args.get('verbose', 'false').lower() == 'true'
        auto_refine = request.args.get('auto_refine', 'true').lower() == 'true'
        use_vector = request.args.get('use_vector', 'false').lower() == 'true'

    if not question:
        return jsonify({"error": "Missing 'question' or 'q' parameter"}), 400

    # Auto-refine the query
    refined_query = question
    refinement_info = None
    if auto_refine:
        try:
            refiner = get_query_refiner()
            refinement = refiner.refine(question, auto_clarify=True)
            refined_query = refinement.refined_query

            # Check if clarification is needed
            if refinement.needs_clarification:
                return jsonify({
                    "success": True,
                    "needs_clarification": True,
                    "original_query": question,
                    "clarification_prompt": refinement.clarification_prompt,
                    "clarification_options": refinement.clarification_options,
                })

            if refinement.transformations:
                refinement_info = {
                    "original": question,
                    "refined": refined_query,
                    "transformations": refinement.transformations,
                }
        except Exception as e:
            # If refiner fails, use original query
            if verbose:
                print(f"Refiner error: {e}")

    try:
        sq = get_smart_query()

        # Fast path: skip vector DB for speed
        if not use_vector:
            result = sq.query_fast(
                question=refined_query,
                max_reports=max_reports,
                max_claims=max_claims,
                verbose=verbose
            )
        else:
            result = sq.query(
                question=refined_query,
                max_reports=max_reports,
                max_claims=max_claims,
                verbose=verbose
            )

        response = {
            "success": True,
            "query": result.query,
            "original_query": question if refinement_info else None,
            "intent": result.intent,
            "answer": result.answer,
            "sources_used": result.sources_used,
            "execution_time_ms": result.execution_time_ms,
            "reports": result.reports[:max_reports],
            "reports_count": len(result.reports),
            "claims_count": len(result.claims),
        }

        # Add refinement info if query was refined
        if refinement_info:
            response["refinement"] = refinement_info

        # Add vector results
        if result.vector_result:
            response["vector_chunks_found"] = result.vector_result.get('chunks_found', 0)
            response["vector_sources"] = result.vector_result.get('sources', [])[:5]

        # Add routing info
        if result.routing:
            response["routing"] = {
                "intent": result.routing.intent.value,
                "postgresql": result.routing.postgresql.value,
                "neo4j": result.routing.neo4j.value,
                "vector_db": result.routing.vector_db.value,
                "parallel": result.routing.parallel,
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# NaverTheme Integration Endpoints
# ============================================================

@app.route('/themes/stock', methods=['GET'])
def get_stock_themes():
    """
    Get all NaverThemes for a specific stock.

    Example: /themes/stock?name=삼성전자

    Response: {
        "success": true,
        "stock_name": "삼성전자",
        "themes": ["반도체", "IT", ...],
        "theme_count": 30,
        "scores": {"bearish": 0.2, "neutral": 0.3, "bullish": 0.5}
    }
    """
    stock_name = request.args.get('name', '')

    if not stock_name:
        return jsonify({"success": False, "error": "Missing 'name' parameter"}), 400

    try:
        service = get_naver_theme_service()
        if not service:
            return jsonify({
                "success": False,
                "error": "NaverTheme service not available"
            }), 503

        result = service.get_stock_themes(stock_name)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/themes/stocks', methods=['GET'])
def get_theme_stocks():
    """
    Get all stocks belonging to a specific NaverTheme.

    Example: /themes/stocks?theme=반도체&limit=20

    Response: {
        "success": true,
        "theme": "반도체",
        "stock_count": 50,
        "stocks": [
            {"name": "삼성전자", "ticker": "005930", "scores": {...}, "total_score": 0.3},
            ...
        ]
    }
    """
    theme_name = request.args.get('theme', '')
    limit = int(request.args.get('limit', 20))

    if not theme_name:
        return jsonify({"success": False, "error": "Missing 'theme' parameter"}), 400

    try:
        service = get_naver_theme_service()
        if not service:
            return jsonify({
                "success": False,
                "error": "NaverTheme service not available"
            }), 503

        result = service.get_theme_stocks(theme_name, limit=limit)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/themes/search', methods=['GET'])
def search_themes():
    """
    Search for themes by name.

    Example: /themes/search?q=반도

    Response: {
        "themes": ["반도체", "반도체 장비", ...]
    }
    """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    if not query or len(query) < 2:
        return jsonify({"themes": []})

    try:
        service = get_naver_theme_service()
        if not service:
            return jsonify({"themes": []})

        themes = service.search_themes(query, limit=limit)
        return jsonify({"themes": themes})

    except Exception as e:
        return jsonify({"themes": [], "error": str(e)})


# =============================================================================
# Multi-Market Stock Search & Chart Integration
# =============================================================================

# Market data paths for OHLCV CSV files (NAS only, charts disabled on Railway)
MARKET_PATHS = {
    "KRX": os.environ.get('KRX_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_Krx/KRXNOTTRAINED'),
    "USA": os.environ.get('USA_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_Usa/USANOTTRAINED'),
    "JAPAN": os.environ.get('JAPAN_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_Japan/JAPANNOTTRAINED'),
    "INDIA": os.environ.get('INDIA_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_India/INDIANOTTRAINED'),
    "HONGKONG": os.environ.get('HONGKONG_DATA_PATH', '/mnt/nas/AutoGluon/AutoML_Hongkong/HONGKONGNOTTRAINED'),
}

# KRX theme data cache
_krx_themes_data = None

def get_krx_themes():
    """Load KRX themes from db_final.csv."""
    global _krx_themes_data
    if _krx_themes_data is None:
        try:
            import ast
            df = pd.read_csv(_data_path('db_final.csv', '/mnt/nas/AutoGluon/AutoML_Krx/DB/db_final.csv'))
            _krx_themes_data = {}
            for _, row in df.iterrows():
                name = row['name']
                naver_theme = row.get('naverTheme', '')
                if pd.notna(naver_theme) and naver_theme:
                    try:
                        themes_list = ast.literal_eval(naver_theme)
                        if isinstance(themes_list, list):
                            _krx_themes_data[name] = themes_list
                    except:
                        _krx_themes_data[name] = []
                else:
                    _krx_themes_data[name] = []
        except Exception as e:
            print(f"Warning: Could not load KRX themes: {e}")
            _krx_themes_data = {}
    return _krx_themes_data


@app.route('/stock/search')
def stock_search():
    """Search stocks by name or ticker for autocomplete."""
    market = request.args.get('market', 'krx').upper()
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 100))

    if market == "KRX":
        # Read directly from db_final.csv for accurate name->ticker mapping
        try:
            df = pd.read_csv(_data_path('db_final.csv', '/mnt/nas/AutoGluon/AutoML_Krx/DB/db_final.csv'))
            stocks = []
            for _, row in df.iterrows():
                name = row['name']
                ticker = str(row['tickers']).zfill(6)
                stocks.append({"name": name, "ticker": ticker})
            stocks.sort(key=lambda x: x["name"])
            total = len(stocks)
            stocks = stocks[:limit]
            return jsonify({"success": True, "stocks": stocks, "total": total})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # Other markets: read CSV filenames
    if market in MARKET_PATHS:
        data_path = MARKET_PATHS[market]
        if os.path.exists(data_path):
            files = [f.replace(".csv", "") for f in os.listdir(data_path) if f.endswith(".csv")]
            stocks = [{"name": f, "ticker": f} for f in sorted(files)]
            total = len(stocks)
            stocks = stocks[:limit]
            return jsonify({"success": True, "stocks": stocks, "total": total})

    return jsonify({"success": True, "stocks": [], "total": 0})


@app.route('/stock/overview')
def stock_overview():
    """Get company overview."""
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    if market == "KRX":
        overview_data = get_company_overview()
        name_map = get_ticker_name_map()
        ticker = name_map.get(symbol, symbol)
        if ticker:
            ticker = str(ticker).zfill(6)
        overview = overview_data.get(ticker, "")
        if overview:
            return jsonify({"success": True, "overview": overview})

    return jsonify({"success": True, "overview": f"Company overview for {symbol} ({market})"})


@app.route('/stock/themes')
def stock_themes():
    """Get stock themes (KRX only)."""
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    if market == "KRX":
        themes_data = get_krx_themes()
        themes = themes_data.get(symbol, [])
        return jsonify({"success": True, "themes": themes})

    return jsonify({"success": True, "themes": []})


@app.route('/stock/reports')
def stock_reports():
    """Get stock reports (KRX only)."""
    symbol = request.args.get('symbol', '')
    limit = int(request.args.get('limit', 5))

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    import glob
    df_files_dir = os.environ.get('DF_FILES_PATH', '/mnt/nas/gpt/Naver/Job/df_files')
    files = sorted(glob.glob(os.path.join(df_files_dir, 'df_firm_*.csv')))
    if not files:
        return jsonify({"success": True, "reports": []})

    try:
        df = pd.read_csv(files[-1], index_col=0)
        reports = df[df['company'] == symbol].head(limit)

        result = []
        for _, row in reports.iterrows():
            result.append({
                "title": row.get('title', ''),
                "issuer": row.get('issuer', ''),
                "issue_date": str(row.get('issue_date', '')),
                "pdf_link": row.get('pdf_links', '')
            })

        return jsonify({"success": True, "reports": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/stock/signal')
def stock_signal():
    """Get signal scores (KRX only)."""
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    # Get signal data from NaverThemeService
    try:
        service = get_naver_theme_service()
        if service:
            result = service.get_stock_themes(symbol)
            if result and result.get('success'):
                return jsonify({
                    "success": True,
                    "scores": result.get('scores', {}),
                    "momentum": result.get('momentum', ''),
                    "theme_count": result.get('theme_count', 0)
                })
            else:
                return jsonify({"success": True, "scores": None, "momentum": "", "theme_count": 0,
                                "debug": f"service_ok=True, result_success={result.get('success') if result else 'None'}"})
        else:
            return jsonify({"success": True, "scores": None, "momentum": "", "theme_count": 0,
                            "debug": "service=None"})
    except Exception as e:
        return jsonify({"success": True, "scores": None, "momentum": "", "theme_count": 0,
                        "debug": f"exception: {str(e)}"})


@app.route('/chart/ohlcv/multi')
def chart_ohlcv_multi():
    """
    Unified chart endpoint for all markets.
    /chart/ohlcv/multi?market=usa&symbol=AAPL&days=180
    """
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '') or request.args.get('name', '')
    days = int(request.args.get('days', 180))

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol parameter"}), 400

    if market not in MARKET_PATHS:
        return jsonify({"success": False, "error": f"Unknown market: {market}"}), 400

    csv_path = f"{MARKET_PATHS[market]}/{symbol}.csv"

    try:
        if not os.path.exists(csv_path):
            return jsonify({"success": False, "error": f"Data not found for {symbol}"}), 404

        df_full = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Calculate indicators
        df_full['ma10'] = df_full['close'].rolling(window=10).mean()
        df_full['ma20'] = df_full['close'].rolling(window=20).mean()
        df_full['ma60'] = df_full['close'].rolling(window=60).mean()

        # RSI
        delta = df_full['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_full['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df_full['bb_middle'] = df_full['close'].rolling(window=20).mean()
        bb_std = df_full['close'].rolling(window=20).std()
        df_full['bb_upper'] = df_full['bb_middle'] + (bb_std * 2)
        df_full['bb_lower'] = df_full['bb_middle'] - (bb_std * 2)

        df = df_full.tail(days)

        data = []
        ma10_data, ma20_data, ma60_data = [], [], []
        rsi_data = []
        bb_upper_data, bb_middle_data, bb_lower_data = [], [], []

        for date, row in df.iterrows():
            time_str = date.strftime('%Y-%m-%d')
            data.append({
                "time": time_str,
                "open": round(float(row['open']), 2),
                "high": round(float(row['high']), 2),
                "low": round(float(row['low']), 2),
                "close": round(float(row['close']), 2),
                "volume": int(row['volume'])
            })

            if pd.notna(row['ma10']):
                ma10_data.append({"time": time_str, "value": round(float(row['ma10']), 2)})
            if pd.notna(row['ma20']):
                ma20_data.append({"time": time_str, "value": round(float(row['ma20']), 2)})
            if pd.notna(row['ma60']):
                ma60_data.append({"time": time_str, "value": round(float(row['ma60']), 2)})
            if pd.notna(row['rsi']):
                rsi_data.append({"time": time_str, "value": round(float(row['rsi']), 2)})
            if pd.notna(row['bb_upper']):
                bb_upper_data.append({"time": time_str, "value": round(float(row['bb_upper']), 2)})
                bb_middle_data.append({"time": time_str, "value": round(float(row['bb_middle']), 2)})
                bb_lower_data.append({"time": time_str, "value": round(float(row['bb_lower']), 2)})

        return jsonify({
            "success": True,
            "stock_name": symbol,
            "market": market,
            "count": len(data),
            "data": data,
            "indicators": {
                "ma10": ma10_data, "ma20": ma20_data, "ma60": ma60_data,
                "rsi": rsi_data,
                "bb_upper": bb_upper_data, "bb_middle": bb_middle_data, "bb_lower": bb_lower_data
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Global Stock Dashboard (Demo)
# =============================================================================

DEMO_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Global Stock Dashboard</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --bullish: #10b981;
            --bearish: #ef4444;
            --neutral: #f59e0b;
            --border: #334155;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg-primary); color: var(--text-secondary); }
        .container { max-width: 1600px; margin: 0 auto; padding: 1rem; }
        .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .back-link { color: var(--text-muted); text-decoration: none; padding: 0.5rem 1rem; background: var(--bg-secondary); border-radius: 6px; font-size: 0.875rem; }
        .back-link:hover { background: var(--bg-tertiary); }
        .market-tabs { display: flex; gap: 0.5rem; }
        .market-tab { padding: 0.5rem 1rem; background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-secondary); cursor: pointer; border-radius: 6px; font-size: 0.875rem; }
        .market-tab:hover { background: var(--bg-tertiary); }
        .market-tab.active { background: var(--accent); border-color: var(--accent); color: white; }
        .stock-header { display: none; margin-bottom: 1rem; }
        .stock-header.visible { display: block; }
        .stock-title-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
        .stock-name { font-size: 1.75rem; font-weight: 700; color: var(--text-primary); }
        .stock-ticker { font-size: 1rem; color: var(--text-muted); background: var(--bg-secondary); padding: 0.25rem 0.5rem; border-radius: 4px; font-family: monospace; }
        .stock-market { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; background: var(--bg-tertiary); color: var(--text-secondary); }
        .stock-market.kospi { background: #1e40af; color: white; }
        .stock-market.kosdaq { background: #7c3aed; color: white; }
        .signal-badge { padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.875rem; font-weight: 600; }
        .signal-badge.buy { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .momentum-badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; background: var(--bg-tertiary); color: var(--text-muted); }
        .momentum-badge.up { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .momentum-badge.down { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .stock-meta { font-size: 0.8rem; color: var(--text-muted); }
        .search-box { display: flex; gap: 0.5rem; margin-bottom: 1rem; position: relative; }
        .search-wrapper { flex: 1; position: relative; }
        .search-box input { width: 100%; padding: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary); border-radius: 6px; font-size: 1rem; }
        .search-box input:focus { outline: none; border-color: var(--accent); }
        .search-box button { padding: 0.75rem 1.5rem; background: var(--accent); border: none; color: white; cursor: pointer; border-radius: 6px; font-weight: 500; }
        .autocomplete-list { position: absolute; top: 100%; left: 0; right: 0; background: var(--bg-secondary); border: 1px solid var(--border); border-top: none; border-radius: 0 0 6px 6px; max-height: 300px; overflow-y: auto; z-index: 1000; display: none; }
        .autocomplete-list.show { display: block; }
        .autocomplete-item { padding: 0.75rem 1rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--bg-tertiary); }
        .autocomplete-item:last-child { border-bottom: none; }
        .autocomplete-item:hover, .autocomplete-item.selected { background: var(--bg-tertiary); }
        .autocomplete-item .name { color: var(--text-primary); }
        .autocomplete-item .ticker { color: var(--text-muted); font-size: 0.8rem; font-family: monospace; }
        .autocomplete-item .match { color: var(--accent); font-weight: 600; }
        .main-layout { display: grid; grid-template-columns: 1fr 320px; gap: 1rem; }
        @media (max-width: 1024px) { .main-layout { grid-template-columns: 1fr; } }
        .main-content { display: flex; flex-direction: column; gap: 1rem; }
        .sidebar { display: flex; flex-direction: column; gap: 1rem; }
        .card { background: var(--bg-secondary); border-radius: 8px; overflow: hidden; }
        .card-header { padding: 0.75rem 1rem; background: var(--bg-tertiary); font-weight: 600; color: var(--text-primary); font-size: 0.875rem; }
        .card-body { padding: 1rem; }
        .chart-section iframe { width: 100%; height: 450px; border: none; }
        .info-tabs { display: flex; border-bottom: 1px solid var(--border); }
        .info-tab { flex: 1; padding: 0.75rem; background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.875rem; transition: all 0.2s; }
        .info-tab:hover { background: var(--bg-tertiary); }
        .info-tab.active { background: var(--accent); color: white; }
        .tab-content { display: none; padding: 1rem; }
        .tab-content.active { display: block; }
        .overview-table { width: 100%; margin-bottom: 1rem; }
        .overview-table tr { border-bottom: 1px solid var(--border); }
        .overview-table td { padding: 0.5rem 0; font-size: 0.8rem; }
        .overview-table td:first-child { color: var(--text-muted); width: 80px; }
        .overview-table td:last-child { color: var(--text-primary); }
        .overview-text { font-size: 0.8rem; line-height: 1.6; color: var(--text-secondary); background: var(--bg-primary); padding: 1rem; border-radius: 6px; max-height: 150px; overflow-y: auto; }
        .fund-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
        .fund-item { text-align: center; padding: 0.75rem; background: var(--bg-primary); border-radius: 6px; }
        .fund-label { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.25rem; }
        .fund-value { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); }
        .theme-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .theme-tag { padding: 0.35rem 0.75rem; background: var(--bg-primary); border-radius: 4px; font-size: 0.75rem; color: var(--text-secondary); }
        .report-item { padding: 0.75rem 0; border-bottom: 1px solid var(--border); }
        .report-item:last-child { border-bottom: none; }
        .report-link { color: var(--text-primary); text-decoration: none; font-size: 0.85rem; }
        .report-link:hover { color: var(--accent); }
        .report-meta { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; }
        .signal-bars { display: flex; gap: 0.5rem; }
        .signal-bar { flex: 1; text-align: center; padding: 1rem 0.5rem; border-radius: 6px; }
        .signal-bar.bullish { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--bullish); }
        .signal-bar.neutral { background: rgba(245, 158, 11, 0.15); border: 1px solid var(--neutral); }
        .signal-bar.bearish { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--bearish); }
        .signal-label { font-size: 0.65rem; color: var(--text-muted); margin-bottom: 0.25rem; }
        .signal-value { font-size: 1.25rem; font-weight: 700; }
        .signal-bar.bullish .signal-value { color: var(--bullish); }
        .signal-bar.neutral .signal-value { color: var(--neutral); }
        .signal-bar.bearish .signal-value { color: var(--bearish); }
        .etf-count { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.75rem; }
        .etf-list { max-height: 300px; overflow-y: auto; }
        .etf-item { padding: 0.5rem; background: var(--bg-primary); border-radius: 4px; margin-bottom: 0.5rem; }
        .etf-name { font-size: 0.8rem; color: var(--text-primary); }
        .etf-code { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.15rem; }
        .etf-more { text-align: center; font-size: 0.75rem; color: var(--text-muted); padding: 0.5rem; }
        .placeholder { text-align: center; padding: 2rem; color: var(--text-muted); }
        .placeholder-icon { font-size: 2rem; margin-bottom: 0.5rem; }
        .krx-badge { font-size: 0.6rem; padding: 0.15rem 0.4rem; background: var(--bullish); color: white; border-radius: 3px; margin-left: 0.5rem; }
    </style>
</head>
<body>
<div class="container">
    <div class="top-bar">
        <a href="/" class="back-link">← SmartQuery로 돌아가기</a>
        <div class="market-tabs">
            <button class="market-tab active" data-market="krx">🇰🇷 KRX</button>
            <button class="market-tab" data-market="usa">🇺🇸 USA</button>
            <button class="market-tab" data-market="japan">🇯🇵 Japan</button>
            <button class="market-tab" data-market="india">🇮🇳 India</button>
            <button class="market-tab" data-market="hongkong">🇭🇰 HongKong</button>
        </div>
    </div>

    <div class="search-box">
        <div class="search-wrapper">
            <input type="text" id="symbol" placeholder="종목명 입력 (예: 삼성전자, AAPL, 7203.T)" autocomplete="off">
            <div class="autocomplete-list" id="autocomplete-list"></div>
        </div>
        <button onclick="loadStock()">검색</button>
    </div>

    <div class="stock-header" id="stock-header">
        <div class="stock-title-row">
            <span class="stock-name" id="stock-name"></span>
            <span class="stock-ticker" id="stock-ticker-display"></span>
            <span class="stock-market" id="stock-market-display"></span>
            <span class="signal-badge buy" id="signal-badge">🔴 매수</span>
            <span class="momentum-badge" id="momentum-badge">하락</span>
        </div>
        <div class="stock-meta" id="theme-count"></div>
    </div>

    <div class="main-layout" id="main-layout" style="display: none;">
        <div class="main-content">
            <div class="card chart-section">
                <div class="card-header">📈 차트</div>
                <iframe id="chart-iframe" src=""></iframe>
            </div>
            <div class="card">
                <div class="info-tabs">
                    <button class="info-tab active" data-tab="overview">개요</button>
                    <button class="info-tab" data-tab="fundamentals">펀더멘탈</button>
                    <button class="info-tab" data-tab="themes">테마</button>
                    <button class="info-tab" data-tab="reports">리포트</button>
                </div>
                <div class="tab-content active" id="tab-overview">
                    <table class="overview-table">
                        <tr><td>종목명</td><td id="info-name"></td></tr>
                        <tr><td>종목코드</td><td id="info-ticker"></td></tr>
                        <tr><td>시장</td><td id="info-market"></td></tr>
                        <tr><td>모멘텀</td><td id="info-momentum"></td></tr>
                        <tr><td>테마 수</td><td id="info-theme-count"></td></tr>
                    </table>
                    <div class="card-header" style="margin: 0 -1rem; padding: 0.5rem 1rem;">📋 기업 개요</div>
                    <div class="overview-text" id="overview-text"></div>
                </div>
                <div class="tab-content" id="tab-fundamentals">
                    <div class="fund-grid" id="fund-grid"></div>
                </div>
                <div class="tab-content" id="tab-themes">
                    <div class="theme-tags" id="theme-tags"></div>
                </div>
                <div class="tab-content" id="tab-reports">
                    <div id="reports-list"></div>
                </div>
            </div>
        </div>
        <div class="sidebar" id="sidebar">
            <div class="card">
                <div class="card-header">📊 시그널 스코어</div>
                <div class="card-body">
                    <div class="signal-bars" id="signal-bars"></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">📦 관련 ETF<span class="krx-badge">KRX Only</span></div>
                <div class="card-body">
                    <div class="etf-count" id="etf-count"></div>
                    <div class="etf-list" id="etf-list"></div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let currentMarket = 'krx';
let stockList = [];
let selectedIndex = -1;
const placeholders = {
    krx: '삼성전자, SK하이닉스, 현대차...',
    usa: 'AAPL, MSFT, GOOGL, TSLA...',
    japan: '7203.T (Toyota), 6758.T (Sony)...',
    india: 'TCS.NS, RELIANCE.NS, INFY.NS...',
    hongkong: '0700.HK (Tencent), 9988.HK (Alibaba)...'
};

document.querySelectorAll('.market-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.market-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentMarket = tab.dataset.market;
        document.getElementById('symbol').placeholder = placeholders[currentMarket];
        loadStockList();
    });
});

document.querySelectorAll('.info-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.info-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
});

async function loadStockList() {
    try {
        const resp = await fetch(`/stock/search?market=${currentMarket}&limit=3000`);
        const data = await resp.json();
        if (data.success) stockList = data.stocks || [];
    } catch (e) { console.error('Failed to load stock list:', e); }
}

function highlightMatch(text, query) {
    if (!query) return text;
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx === -1) return text;
    return text.substring(0, idx) + '<span class="match">' + text.substring(idx, idx + query.length) + '</span>' + text.substring(idx + query.length);
}

function showAutocomplete(query) {
    const list = document.getElementById('autocomplete-list');
    if (!query || query.length < 1) { list.classList.remove('show'); return; }
    const filtered = stockList.filter(s => s.name.toLowerCase().includes(query.toLowerCase()) || s.ticker.includes(query)).slice(0, 15);
    if (filtered.length === 0) { list.classList.remove('show'); return; }
    selectedIndex = -1;
    list.innerHTML = filtered.map((s, i) => `<div class="autocomplete-item" data-index="${i}" data-name="${s.name}"><span class="name">${highlightMatch(s.name, query)}</span><span class="ticker">${s.ticker}</span></div>`).join('');
    list.querySelectorAll('.autocomplete-item').forEach(item => {
        item.addEventListener('click', () => {
            document.getElementById('symbol').value = item.dataset.name;
            list.classList.remove('show');
            loadStock();
        });
    });
    list.classList.add('show');
}

document.getElementById('symbol').addEventListener('input', (e) => showAutocomplete(e.target.value));
document.getElementById('symbol').addEventListener('keydown', (e) => {
    const list = document.getElementById('autocomplete-list');
    const items = list.querySelectorAll('.autocomplete-item');
    if (e.key === 'ArrowDown') { e.preventDefault(); selectedIndex = Math.min(selectedIndex + 1, items.length - 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); selectedIndex = Math.max(selectedIndex - 1, -1); }
    else if (e.key === 'Enter' && selectedIndex >= 0) { e.preventDefault(); items[selectedIndex].click(); return; }
    else if (e.key === 'Escape') { list.classList.remove('show'); return; }
    items.forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
});
document.addEventListener('click', (e) => { if (!e.target.closest('.search-wrapper')) document.getElementById('autocomplete-list').classList.remove('show'); });

async function loadStock() {
    const symbol = document.getElementById('symbol').value.trim();
    if (!symbol) return;
    document.getElementById('stock-header').classList.add('visible');
    document.getElementById('main-layout').style.display = 'grid';
    document.getElementById('autocomplete-list').classList.remove('show');

    // Load info
    try {
        const resp = await fetch(`/stock/info?market=${currentMarket}&name=${encodeURIComponent(symbol)}`);
        const data = await resp.json();
        if (data.success && data.info) {
            document.getElementById('stock-name').textContent = data.info.name || symbol;
            document.getElementById('stock-ticker-display').textContent = data.info.ticker || '';
            document.getElementById('stock-market-display').textContent = data.info.market || currentMarket.toUpperCase();
            document.getElementById('info-name').textContent = data.info.name || symbol;
            document.getElementById('info-ticker').textContent = data.info.ticker || '';
            document.getElementById('info-market').textContent = data.info.market || currentMarket.toUpperCase();
        }
    } catch (e) {}

    // Load chart - use /chart/ohlcv for KRX, /chart/ohlcv/multi for others
    const chartUrl = currentMarket === 'krx'
        ? `/chart/ohlcv?name=${encodeURIComponent(symbol)}&days=180`
        : `/chart/ohlcv/multi?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}&days=180`;
    document.getElementById('chart-iframe').src = `/shared/chart/chart_component.html?api=${encodeURIComponent(chartUrl)}`;

    // Load overview
    try {
        const resp = await fetch(`/stock/overview?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
        const data = await resp.json();
        document.getElementById('overview-text').textContent = data.success ? data.overview : 'No overview available';
    } catch (e) { document.getElementById('overview-text').textContent = 'Failed to load'; }

    // Load fundamentals (KRX only)
    if (currentMarket === 'krx') {
        try {
            const resp = await fetch(`/stock/info?name=${encodeURIComponent(symbol)}`);
            const data = await resp.json();
            if (data.success && data.fundamental) {
                const f = data.fundamental;
                document.getElementById('fund-grid').innerHTML = `
                    <div class="fund-item"><div class="fund-label">시가총액</div><div class="fund-value">${f.marketCap || '-'}</div></div>
                    <div class="fund-item"><div class="fund-label">PER</div><div class="fund-value">${f.per || '-'}</div></div>
                    <div class="fund-item"><div class="fund-label">PBR</div><div class="fund-value">${f.pbr || '-'}</div></div>
                    <div class="fund-item"><div class="fund-label">ROE</div><div class="fund-value">${f.roe || '-'}</div></div>
                    <div class="fund-item"><div class="fund-label">EPS</div><div class="fund-value">${f.eps || '-'}</div></div>
                    <div class="fund-item"><div class="fund-label">배당률</div><div class="fund-value">${f.dividendYield || '-'}</div></div>`;
            } else {
                document.getElementById('fund-grid').innerHTML = '<div class="placeholder">No data</div>';
            }
        } catch (e) { document.getElementById('fund-grid').innerHTML = '<div class="placeholder">Failed to load</div>'; }
    } else {
        document.getElementById('fund-grid').innerHTML = '<div class="placeholder"><div class="placeholder-icon">📊</div><div>KRX Only</div></div>';
    }

    // Load themes
    try {
        const resp = await fetch(`/stock/themes?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
        const data = await resp.json();
        if (data.success && data.themes && data.themes.length > 0) {
            document.getElementById('theme-tags').innerHTML = data.themes.map(t => `<span class="theme-tag">${t}</span>`).join('');
            document.getElementById('theme-count').textContent = `테마: ${data.themes.length}개`;
            document.getElementById('info-theme-count').textContent = `${data.themes.length}개`;
        } else {
            document.getElementById('theme-tags').innerHTML = '<div class="placeholder"><div class="placeholder-icon">🏷️</div><div>No themes</div></div>';
            document.getElementById('theme-count').textContent = '';
            document.getElementById('info-theme-count').textContent = '-';
        }
    } catch (e) { document.getElementById('theme-tags').innerHTML = '<div class="placeholder">Failed to load</div>'; }

    // Load signal (KRX only)
    if (currentMarket === 'krx') {
        try {
            const resp = await fetch(`/stock/signal?symbol=${encodeURIComponent(symbol)}`);
            const data = await resp.json();
            if (data.success && data.scores) {
                const s = data.scores;
                document.getElementById('signal-bars').innerHTML = `
                    <div class="signal-bar bullish"><div class="signal-label">BULLISH</div><div class="signal-value">${s.bullish || '0'}%</div></div>
                    <div class="signal-bar neutral"><div class="signal-label">NEUTRAL</div><div class="signal-value">${s.neutral || '0'}%</div></div>
                    <div class="signal-bar bearish"><div class="signal-label">BEARISH</div><div class="signal-value">${s.bearish || '0'}%</div></div>`;
                document.getElementById('info-momentum').textContent = data.momentum || '-';
                document.getElementById('momentum-badge').textContent = data.momentum || '-';
                document.getElementById('momentum-badge').className = 'momentum-badge ' + (data.momentum === '상승' ? 'up' : 'down');
            } else {
                document.getElementById('signal-bars').innerHTML = '<div class="placeholder">No signal data</div>';
            }
        } catch (e) { document.getElementById('signal-bars').innerHTML = '<div class="placeholder">Failed to load</div>'; }
    } else {
        document.getElementById('signal-bars').innerHTML = '<div class="placeholder"><div class="placeholder-icon">📊</div><div>KRX Only</div></div>';
        document.getElementById('info-momentum').textContent = '-';
        document.getElementById('momentum-badge').textContent = '-';
    }

    // Load ETFs (KRX only)
    if (currentMarket === 'krx') {
        try {
            const resp = await fetch(`/stock/etfs?name=${encodeURIComponent(symbol)}&limit=10`);
            const data = await resp.json();
            if (data.success && data.etfs && data.etfs.length > 0) {
                document.getElementById('etf-count').textContent = `총 ${data.etf_count || data.etfs.length}개 ETF에 포함`;
                document.getElementById('etf-list').innerHTML = data.etfs.map(e => `<div class="etf-item"><div class="etf-name">${e.name}</div><div class="etf-code">${e.code}</div></div>`).join('') +
                    (data.etf_count > 10 ? `<div class="etf-more">외 ${data.etf_count - 10}개 ETF</div>` : '');
            } else {
                document.getElementById('etf-count').textContent = '';
                document.getElementById('etf-list').innerHTML = '<div class="placeholder"><div class="placeholder-icon">📦</div><div>No ETF data</div></div>';
            }
        } catch (e) { document.getElementById('etf-list').innerHTML = '<div class="placeholder">Failed to load</div>'; }
    } else {
        document.getElementById('etf-count').textContent = '';
        document.getElementById('etf-list').innerHTML = '<div class="placeholder"><div class="placeholder-icon">📦</div><div>KRX Only</div></div>';
    }

    // Load reports (KRX only)
    if (currentMarket === 'krx') {
        try {
            const resp = await fetch(`/stock/reports?symbol=${encodeURIComponent(symbol)}&limit=5`);
            const data = await resp.json();
            const el = document.getElementById('reports-list');
            if (data.success && data.reports && data.reports.length > 0) {
                el.innerHTML = data.reports.map(r => {
                    const nid = r.pdf_link?.match(/_company_(\\d+)\\.pdf/)?.[1];
                    const url = nid ? `https://finance.naver.com/research/company_read.naver?nid=${nid}` : '#';
                    return `<div class="report-item"><a href="${url}" target="_blank" class="report-link">${r.title || 'No title'}</a><div class="report-meta">${r.issuer || ''} · ${r.issue_date || ''}</div></div>`;
                }).join('');
            } else {
                el.innerHTML = '<div class="placeholder"><div class="placeholder-icon">📋</div><div>No reports</div></div>';
            }
        } catch (e) { document.getElementById('reports-list').innerHTML = '<div class="placeholder">Failed to load</div>'; }
    } else {
        document.getElementById('reports-list').innerHTML = '<div class="placeholder"><div class="placeholder-icon">📋</div><div>KRX Only</div></div>';
    }
}

// Initial load
loadStockList();
document.getElementById('symbol').value = '삼성전자';
loadStock();
</script>
</body>
</html>'''


@app.route('/demo')
def demo():
    """Global Stock Dashboard - Multi-market chart demo."""
    return render_template_string(DEMO_HTML)


@app.route('/shared/chart/chart_component.html')
def serve_chart_component():
    """Serve the chart component HTML."""
    try:
        chart_path = os.path.join(_PROJECT_ROOT, 'shared', 'chart', 'chart_component.html')
        with open(chart_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error loading chart component: {e}", 500


@app.route('/theme-explorer')
def serve_theme_explorer():
    """Serve the standalone theme explorer page."""
    try:
        theme_path = os.path.join(_PROJECT_ROOT, 'shared', 'theme-explorer', 'index.html')
        with open(theme_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error loading theme explorer: {e}", 500


# ============================================================
# /api/network/* compatibility endpoints for theme-graph.html
# ============================================================

@app.route('/api/network/graph-data')
def network_graph_data():
    """Build vis.js graph data from theme endpoints.

    Supports:
      - No params: returns top themes overview
      - ?stock=NAME: stock-centered graph with its themes
      - ?theme=NAME: theme-centered graph with its stocks
    """
    stock = request.args.get('stock', '')
    theme = request.args.get('theme', '')

    service = get_naver_theme_service()
    if not service:
        return jsonify({"success": False, "detail": "Theme service unavailable"}), 503

    graph_nodes = []
    graph_edges = []

    try:
        if stock:
            # Stock-centered view
            data = service.get_stock_themes(stock)
            if not data.get('success'):
                return jsonify({"success": False, "detail": f"Stock '{stock}' not found"}), 404

            bullish = data['scores'].get('bullish', 0)
            buy_pct = bullish * 100

            # Center stock node
            graph_nodes.append({
                "id": f"stock_{data['stock_name']}",
                "label": data['stock_name'],
                "type": "stock",
                "ticker": data.get('ticker', ''),
                "score": buy_pct,
                "signal": "buy" if buy_pct >= 50 else "hold" if buy_pct >= 30 else "avoid",
                "buy_pct": buy_pct,
                "signal_probability": {
                    "buy": bullish * 100,
                    "neutral": data['scores'].get('neutral', 0) * 100,
                    "sell": data['scores'].get('bearish', 0) * 100,
                },
                "isCenter": True
            })

            # Theme nodes
            for theme_name in data.get('themes', []):
                tid = f"theme_{theme_name}"
                graph_nodes.append({
                    "id": tid,
                    "label": theme_name,
                    "type": "theme",
                    "color": "#3b82f6",
                    "fiedler": 1.0,
                    "size": 25
                })
                graph_edges.append({
                    "id": f"edge_{data['stock_name']}_{theme_name}",
                    "from": f"stock_{data['stock_name']}",
                    "to": tid
                })

        elif theme:
            # Theme-centered view
            data = service.get_theme_stocks(theme, limit=20)
            if not data.get('success'):
                return jsonify({"success": False, "detail": f"Theme '{theme}' not found"}), 404

            # Center theme node
            tid = f"theme_{theme}"
            graph_nodes.append({
                "id": tid,
                "label": theme,
                "type": "theme",
                "color": "#8b5cf6",
                "fiedler": 1.0,
                "size": 35
            })

            # Stock nodes
            for s in data.get('stocks', []):
                buy_pct = s['scores'].get('bullish', 0) * 100
                sid = f"stock_{s['name']}"
                graph_nodes.append({
                    "id": sid,
                    "label": s['name'],
                    "type": "stock",
                    "ticker": s.get('ticker', ''),
                    "score": buy_pct,
                    "buy_pct": buy_pct,
                    "signal": "buy" if buy_pct >= 50 else "hold" if buy_pct >= 30 else "avoid",
                    "signal_probability": {
                        "buy": buy_pct,
                        "neutral": s['scores'].get('neutral', 0) * 100,
                        "sell": s['scores'].get('bearish', 0) * 100,
                    },
                    "isCenter": False
                })
                graph_edges.append({
                    "id": f"edge_{theme}_{s['name']}",
                    "from": tid,
                    "to": sid
                })

        else:
            # Default: overview with popular themes
            # Search for some well-known themes
            popular = ['반도체', '2차전지', 'AI(인공지능)', '방위산업', '로봇', '자율주행차']
            for theme_name in popular:
                tid = f"theme_{theme_name}"
                graph_nodes.append({
                    "id": tid,
                    "label": theme_name,
                    "type": "theme",
                    "color": "#3b82f6",
                    "fiedler": 1.0,
                    "size": 30
                })

        stock_count = len([n for n in graph_nodes if n['type'] == 'stock'])
        theme_count = len([n for n in graph_nodes if n['type'] == 'theme'])

        return jsonify({
            "success": True,
            "nodes": graph_nodes,
            "edges": graph_edges,
            "stats": {
                "stock_count": stock_count,
                "theme_count": theme_count,
                "edge_count": len(graph_edges)
            }
        })

    except Exception as e:
        return jsonify({"success": False, "detail": str(e)}), 500


@app.route('/api/network/search')
def network_search():
    """Search stocks and themes for autocomplete."""
    q = request.args.get('q', '')
    limit = int(request.args.get('limit', 15))

    if len(q) < 2:
        return jsonify({"success": True, "stocks": [], "themes": []})

    service = get_naver_theme_service()
    if not service:
        return jsonify({"success": False, "stocks": [], "themes": []})

    try:
        # Search themes
        theme_results = service.search_themes(q, limit=limit)
        themes_out = [{"theme": t, "fiedler": 1.0, "cohesion_level": "strong"} for t in theme_results]

        # Search stocks by name from the ticker map
        name_map = get_ticker_name_map()
        stock_results = []
        for name, val in name_map.items():
            if q.lower() in name.lower() and not name.isdigit() and len(name) > 1:
                # name_map has both name->ticker and ticker->name; filter to name entries
                if val.isdigit() or (len(val) <= 6 and val[0].isdigit()):
                    stock_data = service.get_stock_themes(name)
                    buy_pct = stock_data.get('scores', {}).get('bullish', 0) * 100 if stock_data.get('success') else 0
                    stock_results.append({
                        "name": name,
                        "ticker": val.zfill(6),
                        "buy_pct": buy_pct,
                        "signal": "buy" if buy_pct >= 50 else "hold" if buy_pct >= 30 else "avoid"
                    })
                    if len(stock_results) >= limit:
                        break

        return jsonify({
            "success": True,
            "stocks": stock_results[:limit],
            "themes": themes_out[:limit]
        })

    except Exception as e:
        return jsonify({"success": False, "stocks": [], "themes": [], "error": str(e)})


@app.route('/api/network/stock-themes')
def network_stock_themes():
    """Get themes for a stock (sidebar list format)."""
    name = request.args.get('name', '')
    if not name:
        return jsonify({"success": False, "themes": []})

    service = get_naver_theme_service()
    if not service:
        return jsonify({"success": False, "themes": []})

    try:
        data = service.get_stock_themes(name)
        if not data.get('success'):
            return jsonify({"success": False, "themes": []})

        themes_out = [{"theme": t, "fiedler": 1.0, "cohesion_level": "strong"} for t in data.get('themes', [])]
        return jsonify({"success": True, "themes": themes_out})

    except Exception as e:
        return jsonify({"success": False, "themes": [], "error": str(e)})


@app.route('/api/network/theme-stocks')
def network_theme_stocks():
    """Get stocks in a theme (sidebar list format)."""
    theme = request.args.get('theme', '')
    limit = int(request.args.get('limit', 20))

    if not theme:
        return jsonify({"success": False, "stocks": []})

    service = get_naver_theme_service()
    if not service:
        return jsonify({"success": False, "stocks": []})

    try:
        data = service.get_theme_stocks(theme, limit=limit)
        if not data.get('success'):
            return jsonify({"success": False, "stocks": []})

        stocks_out = []
        for s in data.get('stocks', []):
            buy_pct = s['scores'].get('bullish', 0) * 100
            stocks_out.append({
                "name": s['name'],
                "ticker": s.get('ticker', ''),
                "market": s.get('market', ''),
                "buy_pct": buy_pct,
                "total_score": s.get('total_score', 0),
                "signal": "buy" if buy_pct >= 50 else "hold" if buy_pct >= 30 else "avoid"
            })

        return jsonify({
            "success": True,
            "theme": theme,
            "fiedler": 1.0,
            "stocks": stocks_out
        })

    except Exception as e:
        return jsonify({"success": False, "stocks": [], "error": str(e)})


@app.route('/theme-graph.html')
def serve_theme_graph():
    """Serve the theme network graph page."""
    try:
        path = os.path.join(_PROJECT_ROOT, 'shared', 'theme-graph.html')
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error loading theme graph: {e}", 500


if __name__ == '__main__':
    print("=" * 60)
    print("SmartQuery API Server (Flask)")
    print("=" * 60)
    print("Pre-loading models at startup...")
    print("=" * 60)

    # Pre-load SmartQuery and all clients at startup
    preload_smart_query()

    print("=" * 60)
    print("Server ready for queries!")
    print("=" * 60)

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
