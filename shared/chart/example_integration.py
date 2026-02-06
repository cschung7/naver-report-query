"""
Example: How to integrate Universal Chart into your Flask project
==================================================================

This shows how to add chart endpoints to any market-specific API.
"""

from flask import Flask, jsonify, request, render_template_string
import sys
sys.path.insert(0, '/mnt/nas/WWAI/NaverReport')

from shared.chart.universal_chart import ChartDataProvider, create_chart_blueprint

app = Flask(__name__)

# =============================================================================
# Option 1: Quick integration with Blueprint (Recommended)
# =============================================================================

# For USA market
usa_chart = create_chart_blueprint("USA", url_prefix="/usa/chart")
app.register_blueprint(usa_chart)
# Endpoints: /usa/chart/ohlcv?symbol=AAPL&days=180
#            /usa/chart/symbols?limit=100

# For Japan market
japan_chart = create_chart_blueprint("JAPAN", url_prefix="/japan/chart")
app.register_blueprint(japan_chart)

# For India market
india_chart = create_chart_blueprint("INDIA", url_prefix="/india/chart")
app.register_blueprint(india_chart)


# =============================================================================
# Option 2: Custom endpoint with more control
# =============================================================================

krx_provider = ChartDataProvider(market_name="KRX")

@app.route('/krx/chart/ohlcv')
def krx_ohlcv():
    """Custom endpoint for KRX with name parameter (Korean stock names)."""
    # KRX uses 'name' instead of 'symbol' for Korean names
    name = request.args.get('name', '')
    days = int(request.args.get('days', 180))

    if not name:
        return jsonify({"success": False, "error": "Missing 'name' parameter"}), 400

    data = krx_provider.get_ohlcv(name, days=days, include_indicators=True)
    return jsonify(data)


# =============================================================================
# Option 3: Unified endpoint for all markets
# =============================================================================

@app.route('/chart/ohlcv')
def unified_ohlcv():
    """
    Unified chart endpoint for all markets.

    Example:
        /chart/ohlcv?market=usa&symbol=AAPL&days=180
        /chart/ohlcv?market=krx&symbol=삼성전자&days=90
    """
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '') or request.args.get('name', '')
    days = int(request.args.get('days', 180))

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol parameter"}), 400

    try:
        provider = ChartDataProvider(market_name=market)
        data = provider.get_ohlcv(symbol, days=days, include_indicators=True)
        return jsonify(data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Stock Info APIs (Overview, Fundamentals, Themes, Reports)
# =============================================================================

# Load KRX-specific data
import json
import os

def load_krx_data():
    """Load KRX fundamental and overview data."""
    import pandas as pd

    fundamental_path = "/mnt/nas/AutoGluon/AutoML_Krx/DB/fundamental_data.json"
    overview_path = "/mnt/nas/AutoGluon/AutoML_Krx/DB/company_overview.csv"
    db_path = "/mnt/nas/AutoGluon/AutoML_Krx/DB/db_final.csv"

    data = {
        "fundamentals": {},  # ticker -> fund data
        "overviews": {},     # ticker -> overview text
        "themes": {},        # name -> themes list
        "name_to_ticker": {},  # name -> ticker mapping
        "ticker_to_name": {},  # ticker -> name mapping
        "stock_info": {}     # name -> {ticker, market, name}
    }

    # Load ticker-name-market-themes mapping from db_final.csv
    if os.path.exists(db_path):
        import ast
        df = pd.read_csv(db_path, encoding='utf-8')
        if 'tickers' in df.columns and 'name' in df.columns:
            # Ensure ticker is string and zero-padded
            df['tickers'] = df['tickers'].astype(str).str.zfill(6)
            data["name_to_ticker"] = dict(zip(df['name'], df['tickers']))
            data["ticker_to_name"] = dict(zip(df['tickers'], df['name']))

            # Stock info (name -> {ticker, market, name}) and themes
            for _, row in df.iterrows():
                name = row['name']
                data["stock_info"][name] = {
                    "name": name,
                    "ticker": row['tickers'],
                    "market": row.get('market', 'KRX')
                }

                # Parse naverTheme column (string representation of list)
                naver_theme = row.get('naverTheme', '')
                if pd.notna(naver_theme) and naver_theme:
                    try:
                        themes_list = ast.literal_eval(naver_theme)
                        if isinstance(themes_list, list):
                            data["themes"][name] = themes_list
                    except:
                        data["themes"][name] = []
                else:
                    data["themes"][name] = []

    # Fundamentals (keyed by ticker)
    if os.path.exists(fundamental_path):
        with open(fundamental_path, 'r', encoding='utf-8') as f:
            data["fundamentals"] = json.load(f)

    # Overview (keyed by ticker)
    if os.path.exists(overview_path):
        df = pd.read_csv(overview_path, encoding='utf-8')
        if 'tickers' in df.columns and 'company overview' in df.columns:
            df['tickers'] = df['tickers'].astype(str).str.zfill(6)
            data["overviews"] = dict(zip(df['tickers'], df['company overview']))

    return data

KRX_DATA = None

def get_krx_data():
    global KRX_DATA
    if KRX_DATA is None:
        KRX_DATA = load_krx_data()
    return KRX_DATA


@app.route('/stock/info')
def stock_info():
    """Get basic stock info (name, ticker, market)."""
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    if market == "KRX":
        data = get_krx_data()
        info = data["stock_info"].get(symbol)
        if info:
            return jsonify({"success": True, "info": info})

    # For other markets, return basic info
    return jsonify({
        "success": True,
        "info": {
            "name": symbol,
            "ticker": symbol,
            "market": market
        }
    })


@app.route('/stock/search')
def stock_search():
    """Search stocks by name or ticker for autocomplete."""
    market = request.args.get('market', 'krx').upper()
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 100))

    if market == "KRX":
        data = get_krx_data()
        stocks = [
            {"name": name, "ticker": info["ticker"]}
            for name, info in data["stock_info"].items()
        ]
        # Sort by name
        stocks.sort(key=lambda x: x["name"])
        # Limit
        stocks = stocks[:limit]
        return jsonify({"success": True, "stocks": stocks, "total": len(data["stock_info"])})

    # Other markets: read CSV filenames from data directory
    market_paths = {
        "USA": "/mnt/nas/AutoGluon/AutoML_Usa/USANOTTRAINED",
        "JAPAN": "/mnt/nas/AutoGluon/AutoML_Japan/JAPANNOTTRAINED",
        "INDIA": "/mnt/nas/AutoGluon/AutoML_India/INDIANOTTRAINED",
        "HONGKONG": "/mnt/nas/AutoGluon/AutoML_Hongkong/HONGKONGNOTTRAINED"
    }

    if market in market_paths:
        data_path = market_paths[market]
        if os.path.exists(data_path):
            # Get all CSV files (symbol is filename without .csv)
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
        data = get_krx_data()
        # Get ticker from name
        ticker = data["name_to_ticker"].get(symbol, symbol)
        overview = data["overviews"].get(ticker)
        if overview:
            return jsonify({"success": True, "overview": overview})

    # For other markets, return placeholder (can be extended later)
    return jsonify({"success": True, "overview": f"Company overview for {symbol} ({market})"})


@app.route('/stock/fundamentals')
def stock_fundamentals():
    """Get stock fundamentals."""
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    if market == "KRX":
        data = get_krx_data()
        # Get ticker from name
        ticker = data["name_to_ticker"].get(symbol, symbol)
        fund = data["fundamentals"].get(ticker)
        if fund:
            return jsonify({"success": True, "fundamentals": fund})

    # For other markets, return empty (can be extended later with yfinance etc.)
    return jsonify({"success": True, "fundamentals": None})


@app.route('/stock/themes')
def stock_themes():
    """Get stock themes."""
    market = request.args.get('market', 'krx').upper()
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    if market == "KRX":
        data = get_krx_data()
        themes = data["themes"].get(symbol, [])
        return jsonify({"success": True, "themes": themes})

    # Other markets - no themes
    return jsonify({"success": True, "themes": []})


@app.route('/stock/reports')
def stock_reports():
    """Get stock reports (KRX only)."""
    symbol = request.args.get('symbol', '')
    limit = int(request.args.get('limit', 5))

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    import glob
    import pandas as pd

    # Find latest firm report file
    files = sorted(glob.glob('/mnt/nas/gpt/Naver/Job/df_files/df_firm_*.csv'))
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


# =============================================================================
# Signal Score & ETF APIs (KRX only, proxy to main server)
# =============================================================================

@app.route('/stock/signal')
def stock_signal():
    """Get signal scores (KRX only) - proxy to main SmartQuery server."""
    symbol = request.args.get('symbol', '')

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    import urllib.request
    import urllib.parse

    try:
        # Proxy to main SmartQuery server
        url = f"http://127.0.0.1:8080/themes/stock?name={urllib.parse.quote(symbol)}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            import json as json_module
            data = json_module.loads(resp.read().decode('utf-8'))
            if data.get('success'):
                return jsonify({
                    "success": True,
                    "scores": data.get('scores', {}),
                    "momentum": data.get('momentum', ''),
                    "theme_count": data.get('theme_count', 0)
                })
    except Exception as e:
        pass

    # Fallback - no signal data available
    return jsonify({"success": True, "scores": None, "momentum": "", "theme_count": 0})


@app.route('/stock/etfs')
def stock_etfs():
    """Get related ETFs (KRX only) - proxy to main SmartQuery server."""
    symbol = request.args.get('symbol', '')
    limit = int(request.args.get('limit', 10))

    if not symbol:
        return jsonify({"success": False, "error": "Missing symbol"}), 400

    import urllib.request
    import urllib.parse

    try:
        # Proxy to main SmartQuery server
        url = f"http://127.0.0.1:8080/stock/etfs?name={urllib.parse.quote(symbol)}&limit={limit}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            import json as json_module
            data = json_module.loads(resp.read().decode('utf-8'))
            if data.get('success'):
                return jsonify(data)
    except Exception as e:
        pass

    # Fallback - no ETF data available
    return jsonify({"success": True, "etfs": [], "etf_count": 0})


# =============================================================================
# Demo page with embedded chart
# =============================================================================

DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Universal Stock Dashboard</title>
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

        /* Header & Market Tabs */
        .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .back-link { color: var(--text-muted); text-decoration: none; padding: 0.5rem 1rem; background: var(--bg-secondary); border-radius: 6px; font-size: 0.875rem; }
        .back-link:hover { background: var(--bg-tertiary); }
        .market-tabs { display: flex; gap: 0.5rem; }
        .market-tab { padding: 0.5rem 1rem; background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-secondary); cursor: pointer; border-radius: 6px; font-size: 0.875rem; }
        .market-tab:hover { background: var(--bg-tertiary); }
        .market-tab.active { background: var(--accent); border-color: var(--accent); color: white; }

        /* Stock Header */
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
        .signal-badge.hold { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
        .momentum-badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; background: var(--bg-tertiary); color: var(--text-muted); }
        .momentum-badge.up { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .momentum-badge.down { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .stock-meta { font-size: 0.8rem; color: var(--text-muted); }

        /* Search */
        .search-box { display: flex; gap: 0.5rem; margin-bottom: 1rem; position: relative; }
        .search-wrapper { flex: 1; position: relative; }
        .search-box input { width: 100%; padding: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border); color: var(--text-primary); border-radius: 6px; font-size: 1rem; }
        .search-box input:focus { outline: none; border-color: var(--accent); }
        .search-box button { padding: 0.75rem 1.5rem; background: var(--accent); border: none; color: white; cursor: pointer; border-radius: 6px; font-weight: 500; }

        /* Autocomplete */
        .autocomplete-list { position: absolute; top: 100%; left: 0; right: 0; background: var(--bg-secondary); border: 1px solid var(--border); border-top: none; border-radius: 0 0 6px 6px; max-height: 300px; overflow-y: auto; z-index: 1000; display: none; }
        .autocomplete-list.show { display: block; }
        .autocomplete-item { padding: 0.75rem 1rem; cursor: pointer; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--bg-tertiary); }
        .autocomplete-item:last-child { border-bottom: none; }
        .autocomplete-item:hover, .autocomplete-item.selected { background: var(--bg-tertiary); }
        .autocomplete-item .name { color: var(--text-primary); }
        .autocomplete-item .ticker { color: var(--text-muted); font-size: 0.8rem; font-family: monospace; }
        .autocomplete-item .match { color: var(--accent); font-weight: 600; }

        /* Two Column Layout */
        .main-layout { display: grid; grid-template-columns: 1fr 320px; gap: 1rem; }
        @media (max-width: 1024px) { .main-layout { grid-template-columns: 1fr; } }

        /* Left: Main Content */
        .main-content { display: flex; flex-direction: column; gap: 1rem; }

        /* Right: Sidebar */
        .sidebar { display: flex; flex-direction: column; gap: 1rem; }

        /* Cards */
        .card { background: var(--bg-secondary); border-radius: 8px; overflow: hidden; }
        .card-header { padding: 0.75rem 1rem; background: var(--bg-tertiary); font-weight: 600; color: var(--text-primary); font-size: 0.875rem; }
        .card-body { padding: 1rem; }

        /* Chart Section */
        .chart-section iframe { width: 100%; height: 450px; border: none; }

        /* Info Tabs */
        .info-tabs { display: flex; border-bottom: 1px solid var(--border); }
        .info-tab { flex: 1; padding: 0.75rem; background: transparent; border: none; color: var(--text-muted); cursor: pointer; font-size: 0.875rem; transition: all 0.2s; }
        .info-tab:hover { background: var(--bg-tertiary); }
        .info-tab.active { background: var(--accent); color: white; }
        .tab-content { display: none; padding: 1rem; }
        .tab-content.active { display: block; }

        /* Overview */
        .overview-table { width: 100%; margin-bottom: 1rem; }
        .overview-table tr { border-bottom: 1px solid var(--border); }
        .overview-table td { padding: 0.5rem 0; font-size: 0.8rem; }
        .overview-table td:first-child { color: var(--text-muted); width: 80px; }
        .overview-table td:last-child { color: var(--text-primary); }
        .overview-text { font-size: 0.8rem; line-height: 1.6; color: var(--text-secondary); background: var(--bg-primary); padding: 1rem; border-radius: 6px; max-height: 150px; overflow-y: auto; }

        /* Fundamentals */
        .fund-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
        .fund-item { text-align: center; padding: 0.75rem; background: var(--bg-primary); border-radius: 6px; }
        .fund-label { font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.25rem; }
        .fund-value { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); }

        /* Themes */
        .theme-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .theme-tag { padding: 0.35rem 0.75rem; background: var(--bg-primary); border-radius: 4px; font-size: 0.75rem; color: var(--text-secondary); }

        /* Reports */
        .report-item { padding: 0.75rem 0; border-bottom: 1px solid var(--border); }
        .report-item:last-child { border-bottom: none; }
        .report-link { color: var(--text-primary); text-decoration: none; font-size: 0.85rem; }
        .report-link:hover { color: var(--accent); }
        .report-meta { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; }

        /* Signal Score */
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

        /* ETF List */
        .etf-count { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.75rem; }
        .etf-list { max-height: 300px; overflow-y: auto; }
        .etf-item { padding: 0.5rem; background: var(--bg-primary); border-radius: 4px; margin-bottom: 0.5rem; }
        .etf-name { font-size: 0.8rem; color: var(--text-primary); }
        .etf-code { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.15rem; }
        .etf-more { text-align: center; font-size: 0.75rem; color: var(--text-muted); padding: 0.5rem; }

        /* Placeholder & Loading */
        .placeholder { text-align: center; padding: 2rem; color: var(--text-muted); }
        .placeholder-icon { font-size: 2rem; margin-bottom: 0.5rem; }
        .loading { text-align: center; padding: 1rem; color: var(--text-muted); }

        /* KRX Only Badge */
        .krx-badge { font-size: 0.6rem; padding: 0.15rem 0.4rem; background: var(--bullish); color: white; border-radius: 3px; margin-left: 0.5rem; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Bar -->
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

        <!-- Search Box -->
        <div class="search-box">
            <div class="search-wrapper">
                <input type="text" id="symbol" placeholder="종목명 입력 (예: 삼성전자, AAPL, 7203.T)" autocomplete="off">
                <div class="autocomplete-list" id="autocomplete-list"></div>
            </div>
            <button onclick="loadStock()">검색</button>
        </div>

        <!-- Stock Header -->
        <div class="stock-header" id="stock-header">
            <div class="stock-title-row">
                <span class="stock-name" id="stock-name">-</span>
                <span class="stock-ticker" id="stock-ticker">-</span>
                <span class="stock-market" id="stock-market">-</span>
                <span class="signal-badge" id="signal-badge" style="display:none"></span>
                <span class="momentum-badge" id="momentum-badge" style="display:none"></span>
            </div>
            <div class="stock-meta" id="stock-meta"></div>
        </div>

        <!-- Main Layout: Two Columns -->
        <div class="main-layout">
            <!-- Left: Main Content -->
            <div class="main-content">
                <!-- Chart -->
                <div class="card chart-section">
                    <div class="card-header">📈 차트</div>
                    <iframe id="chartFrame" src="about:blank"></iframe>
                </div>

                <!-- Info Tabs Card -->
                <div class="card">
                    <div class="info-tabs">
                        <button class="info-tab active" data-tab="overview">개요</button>
                        <button class="info-tab" data-tab="fundamentals">펀더멘탈</button>
                        <button class="info-tab" data-tab="themes">테마</button>
                        <button class="info-tab" data-tab="reports" id="reports-tab">리포트</button>
                    </div>

                    <!-- Overview Tab -->
                    <div class="tab-content active" id="tab-overview">
                        <table class="overview-table" id="overview-table">
                            <tr><td>종목명</td><td id="ov-name">-</td></tr>
                            <tr><td>종목코드</td><td id="ov-ticker">-</td></tr>
                            <tr><td>시장</td><td id="ov-market">-</td></tr>
                            <tr><td>모멘텀</td><td id="ov-momentum">-</td></tr>
                            <tr><td>테마 수</td><td id="ov-themes">-</td></tr>
                        </table>
                        <div class="card-header" style="margin: 1rem -1rem -1rem -1rem; margin-top: 1rem;">📋 기업 개요</div>
                        <div id="overview-content" class="overview-text" style="margin-top: 1rem;">Loading...</div>
                    </div>

                    <!-- Fundamentals Tab -->
                    <div class="tab-content" id="tab-fundamentals">
                        <div id="fundamentals-content" class="fund-grid">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>

                    <!-- Themes Tab -->
                    <div class="tab-content" id="tab-themes">
                        <div id="themes-content" class="theme-tags">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>

                    <!-- Reports Tab -->
                    <div class="tab-content" id="tab-reports">
                        <div id="reports-content">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right: Sidebar -->
            <div class="sidebar">
                <!-- Signal Score Card -->
                <div class="card" id="signal-card">
                    <div class="card-header">📊 시그널 스코어</div>
                    <div class="card-body">
                        <div id="signal-content" class="signal-bars">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>
                </div>

                <!-- Related ETF Card -->
                <div class="card" id="etf-card">
                    <div class="card-header">📦 관련 ETF<span class="krx-badge">KRX Only</span></div>
                    <div class="card-body">
                        <div id="etf-content">
                            <div class="loading">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentMarket = 'krx';
        let currentSymbol = '';

        // Market tab handlers
        document.querySelectorAll('.market-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('.market-tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                currentMarket = this.dataset.market;

                // Show/hide KRX-only elements
                const isKRX = currentMarket === 'krx';
                document.getElementById('reports-tab').style.display = isKRX ? 'block' : 'none';
                document.getElementById('signal-card').style.display = isKRX ? 'block' : 'none';
                document.getElementById('etf-card').style.display = isKRX ? 'block' : 'none';

                // Update placeholder
                const placeholders = {
                    'krx': '삼성전자, SK하이닉스, 현대차...',
                    'usa': 'AAPL, MSFT, GOOGL, TSLA...',
                    'japan': '7203.T (Toyota), 6758.T (Sony)...',
                    'india': 'TCS.NS, RELIANCE.NS, INFY.NS...',
                    'hongkong': '0700.HK (Tencent), 9988.HK (Alibaba)...'
                };
                document.getElementById('symbol').placeholder = placeholders[currentMarket];

                // Reload stock list for new market
                loadStockList();
            });
        });

        // Info tab handlers
        document.querySelectorAll('.info-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('.info-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                this.classList.add('active');
                document.getElementById('tab-' + this.dataset.tab).classList.add('active');
            });
        });

        // Autocomplete functionality
        let stockList = [];
        let selectedIndex = -1;

        async function loadStockList() {
            try {
                const resp = await fetch(`/stock/search?market=${currentMarket}&limit=3000`);
                const data = await resp.json();
                if (data.success) {
                    stockList = data.stocks || [];
                }
            } catch (e) {
                console.error('Failed to load stock list:', e);
            }
        }

        function highlightMatch(text, query) {
            if (!query) return text;
            const lowerText = text.toLowerCase();
            const lowerQuery = query.toLowerCase();
            const idx = lowerText.indexOf(lowerQuery);
            if (idx === -1) return text;
            return text.substring(0, idx) +
                   '<span class="match">' + text.substring(idx, idx + query.length) + '</span>' +
                   text.substring(idx + query.length);
        }

        function showAutocomplete(query) {
            const list = document.getElementById('autocomplete-list');
            if (!query || query.length < 1) {
                list.classList.remove('show');
                return;
            }

            const filtered = stockList.filter(s =>
                s.name.toLowerCase().includes(query.toLowerCase()) ||
                s.ticker.includes(query)
            ).slice(0, 15);

            if (filtered.length === 0) {
                list.classList.remove('show');
                return;
            }

            list.innerHTML = filtered.map((s, i) => `
                <div class="autocomplete-item${i === selectedIndex ? ' selected' : ''}"
                     data-name="${s.name}" data-index="${i}">
                    <span class="name">${highlightMatch(s.name, query)}</span>
                    <span class="ticker">${highlightMatch(s.ticker, query)}</span>
                </div>
            `).join('');

            list.classList.add('show');

            // Click handlers
            list.querySelectorAll('.autocomplete-item').forEach(item => {
                item.addEventListener('click', function() {
                    document.getElementById('symbol').value = this.dataset.name;
                    list.classList.remove('show');
                    selectedIndex = -1;
                    loadStock();
                });
            });
        }

        // Input event handlers
        const symbolInput = document.getElementById('symbol');
        symbolInput.addEventListener('input', function() {
            selectedIndex = -1;
            showAutocomplete(this.value.trim());
        });

        symbolInput.addEventListener('keydown', function(e) {
            const list = document.getElementById('autocomplete-list');
            const items = list.querySelectorAll('.autocomplete-item');

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, items.length - 1);
                items.forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
                if (items[selectedIndex]) items[selectedIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, 0);
                items.forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
                if (items[selectedIndex]) items[selectedIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'Enter') {
                if (selectedIndex >= 0 && items[selectedIndex]) {
                    e.preventDefault();
                    this.value = items[selectedIndex].dataset.name;
                    list.classList.remove('show');
                    selectedIndex = -1;
                    loadStock();
                }
            } else if (e.key === 'Escape') {
                list.classList.remove('show');
                selectedIndex = -1;
            }
        });

        // Close autocomplete when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.search-wrapper')) {
                document.getElementById('autocomplete-list').classList.remove('show');
                selectedIndex = -1;
            }
        });

        // Load stock list on init and market change
        loadStockList();

        // Load stock
        async function loadStock() {
            const symbol = document.getElementById('symbol').value.trim();
            if (!symbol) return;
            currentSymbol = symbol;

            // Load all data
            loadStockInfo(symbol);
            loadChart(symbol);
            loadOverview(symbol);
            loadFundamentals(symbol);
            loadThemes(symbol);

            if (currentMarket === 'krx') {
                loadSignal(symbol);
                loadETFs(symbol);
                loadReports(symbol);
            }
        }

        function loadChart(symbol) {
            const apiUrl = `/chart/ohlcv?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}&days=180`;
            document.getElementById('chartFrame').src = `/shared/chart/chart_component.html?api=${encodeURIComponent(apiUrl)}`;
        }

        async function loadStockInfo(symbol) {
            const header = document.getElementById('stock-header');
            try {
                const resp = await fetch(`/stock/info?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();
                if (data.success && data.info) {
                    document.getElementById('stock-name').textContent = data.info.name;
                    document.getElementById('stock-ticker').textContent = data.info.ticker;
                    const marketEl = document.getElementById('stock-market');
                    marketEl.textContent = data.info.market;
                    marketEl.className = 'stock-market';
                    if (data.info.market === 'KOSPI') marketEl.classList.add('kospi');
                    else if (data.info.market === 'KOSDAQ') marketEl.classList.add('kosdaq');

                    // Update overview table
                    document.getElementById('ov-name').textContent = data.info.name;
                    document.getElementById('ov-ticker').textContent = data.info.ticker;
                    document.getElementById('ov-market').textContent = data.info.market;
                }
                header.classList.add('visible');
            } catch (e) {
                document.getElementById('stock-name').textContent = symbol;
                header.classList.add('visible');
            }
        }

        async function loadSignal(symbol) {
            const el = document.getElementById('signal-content');
            const signalBadge = document.getElementById('signal-badge');
            const momentumBadge = document.getElementById('momentum-badge');

            try {
                const resp = await fetch(`/stock/signal?symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();

                if (data.success && data.scores) {
                    const bullish = (data.scores.bullish * 100).toFixed(1);
                    const neutral = (data.scores.neutral * 100).toFixed(1);
                    const bearish = (data.scores.bearish * 100).toFixed(1);

                    el.innerHTML = `
                        <div class="signal-bar bullish"><div class="signal-label">BULLISH</div><div class="signal-value">${bullish}%</div></div>
                        <div class="signal-bar neutral"><div class="signal-label">NEUTRAL</div><div class="signal-value">${neutral}%</div></div>
                        <div class="signal-bar bearish"><div class="signal-label">BEARISH</div><div class="signal-value">${bearish}%</div></div>
                    `;

                    // Update header badges
                    const isBuy = data.scores.bullish > data.scores.bearish;
                    signalBadge.textContent = isBuy ? '🔴 매수' : '🔵 관망';
                    signalBadge.className = 'signal-badge ' + (isBuy ? 'buy' : 'hold');
                    signalBadge.style.display = 'inline-block';

                    // Momentum badge
                    const momentum = data.momentum || '';
                    let momLabel = '중립', momClass = '';
                    if (momentum.includes('up')) { momLabel = '상승'; momClass = 'up'; }
                    else if (momentum.includes('dn')) { momLabel = '하락'; momClass = 'down'; }
                    momentumBadge.textContent = momLabel;
                    momentumBadge.className = 'momentum-badge ' + momClass;
                    momentumBadge.style.display = 'inline-block';

                    // Update overview
                    document.getElementById('ov-momentum').textContent = momLabel;
                    document.getElementById('ov-themes').textContent = (data.theme_count || 0) + '개';
                    document.getElementById('stock-meta').textContent = `테마: ${data.theme_count || 0}개`;
                } else {
                    el.innerHTML = '<div class="placeholder"><div class="placeholder-icon">📊</div><div>시그널 데이터 없음</div></div>';
                }
            } catch (e) {
                el.innerHTML = '<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>로딩 실패</div></div>';
            }
        }

        async function loadETFs(symbol) {
            const el = document.getElementById('etf-content');
            try {
                const resp = await fetch(`/stock/etfs?symbol=${encodeURIComponent(symbol)}&limit=10`);
                const data = await resp.json();

                if (data.success && data.etfs && data.etfs.length > 0) {
                    const count = data.etf_count || data.etfs.length;
                    const etfHtml = data.etfs.map(e => `<div class="etf-item"><div class="etf-name">${e.name}</div><div class="etf-code">${e.code}</div></div>`).join('');
                    const more = count > 10 ? `<div class="etf-more">외 ${count - 10}개 ETF</div>` : '';
                    el.innerHTML = `<div class="etf-count">총 ${count}개 ETF에 포함</div><div class="etf-list">${etfHtml}</div>${more}`;
                } else {
                    el.innerHTML = '<div class="placeholder"><div class="placeholder-icon">📦</div><div>관련 ETF 없음</div></div>';
                }
            } catch (e) {
                el.innerHTML = '<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>로딩 실패</div></div>';
            }
        }

        async function loadOverview(symbol) {
            const el = document.getElementById('overview-content');
            el.innerHTML = 'Loading...';
            try {
                const resp = await fetch(`/stock/overview?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();

                if (data.success && data.overview) {
                    el.innerHTML = data.overview;
                } else {
                    el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">📄</div><div>No overview available</div></div>`;
                }
            } catch (e) {
                el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>Failed to load</div></div>`;
            }
        }

        async function loadFundamentals(symbol) {
            const el = document.getElementById('fundamentals-content');
            el.innerHTML = '<div class="loading">Loading...</div>';

            try {
                const resp = await fetch(`/stock/fundamentals?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();

                if (data.success && data.fundamentals) {
                    const f = data.fundamentals;
                    el.innerHTML = `
                        <div class="fund-item"><div class="fund-label">PER</div><div class="fund-value">${f.PER?.toFixed(2) || '-'}</div></div>
                        <div class="fund-item"><div class="fund-label">PBR</div><div class="fund-value">${f.PBR?.toFixed(2) || '-'}</div></div>
                        <div class="fund-item"><div class="fund-label">EPS</div><div class="fund-value">${f.EPS?.toFixed(2) || '-'}</div></div>
                        <div class="fund-item"><div class="fund-label">BPS</div><div class="fund-value">${f.BPS?.toFixed(2) || '-'}</div></div>
                        <div class="fund-item"><div class="fund-label">Market Cap</div><div class="fund-value">${f.MarketCap ? (f.MarketCap/1000).toFixed(1) + 'T' : '-'}</div></div>
                        <div class="fund-item"><div class="fund-label">Div Yield</div><div class="fund-value">${f.DivYield ? f.DivYield.toFixed(2) + '%' : '-'}</div></div>
                    `;
                } else {
                    el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">📊</div><div>No data</div></div>`;
                }
            } catch (e) {
                el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>Failed to load</div></div>`;
            }
        }

        async function loadThemes(symbol) {
            const el = document.getElementById('themes-content');
            el.innerHTML = '<div class="loading">Loading...</div>';

            try {
                const resp = await fetch(`/stock/themes?market=${currentMarket}&symbol=${encodeURIComponent(symbol)}`);
                const data = await resp.json();

                if (data.success && data.themes && data.themes.length > 0) {
                    el.innerHTML = data.themes.map(t => `<span class="theme-tag">${t}</span>`).join('');
                } else {
                    el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">🏷️</div><div>No themes</div></div>`;
                }
            } catch (e) {
                el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>Failed to load</div></div>`;
            }
        }

        async function loadReports(symbol) {
            const el = document.getElementById('reports-content');
            el.innerHTML = '<div class="loading">Loading...</div>';

            try {
                const resp = await fetch(`/stock/reports?symbol=${encodeURIComponent(symbol)}&limit=5`);
                const data = await resp.json();

                if (data.success && data.reports && data.reports.length > 0) {
                    el.innerHTML = data.reports.map(r => {
                        const nid = r.pdf_link?.match(/_company_(\\d+)\\.pdf/)?.[1];
                        const url = nid ? `https://finance.naver.com/research/company_read.naver?nid=${nid}` : '#';
                        return `
                            <div class="report-item">
                                <a href="${url}" target="_blank" class="report-link">${r.title || 'No title'}</a>
                                <div class="report-meta">${r.issuer || ''} · ${r.issue_date || ''}</div>
                            </div>
                        `;
                    }).join('');
                } else {
                    el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">📋</div><div>No reports</div></div>`;
                }
            } catch (e) {
                el.innerHTML = `<div class="placeholder"><div class="placeholder-icon">⚠️</div><div>Failed to load</div></div>`;
            }
        }

        // Initial load
        document.getElementById('symbol').value = '삼성전자';
        loadStock();
    </script>
</body>
</html>
"""

@app.route('/demo')
def demo():
    return render_template_string(DEMO_HTML)


@app.route('/shared/chart/chart_component.html')
def serve_chart_component():
    """Serve the chart component HTML."""
    with open('/mnt/nas/WWAI/NaverReport/shared/chart/chart_component.html', 'r') as f:
        return f.read()


if __name__ == '__main__':
    print("=" * 60)
    print("Universal Chart Demo Server")
    print("=" * 60)
    print("Endpoints:")
    print("  /demo                    - Demo page with all markets")
    print("  /chart/ohlcv             - Unified endpoint (all markets)")
    print("  /usa/chart/ohlcv         - USA endpoint")
    print("  /japan/chart/ohlcv       - Japan endpoint")
    print("  /india/chart/ohlcv       - India endpoint")
    print("  /krx/chart/ohlcv         - KRX endpoint (custom)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8081, debug=True)
