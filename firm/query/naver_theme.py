"""
NaverTheme Integration Module
============================
Provides access to NaverTheme data for FirmAnalysis integration.
"""
import os
import ast
import pandas as pd
from typing import Dict, List, Any, Optional

# Path to NaverTheme data - use bundled data/ or NAS fallback
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

THEME_DATA_DIR = os.environ.get('THEME_DATA_DIR', '/mnt/nas/WWAI/NaverTheme/webapp/backend/data')
_bundled_db_final = os.path.join(_DATA_DIR, 'db_final.csv')
CSV_PATH = _bundled_db_final if os.path.exists(_bundled_db_final) else os.path.join(THEME_DATA_DIR, "db_final.csv")

# Path to Buy_Or_Not database (NAS only, graceful skip on Railway)
BUY_OR_NOT_PATH = os.environ.get('BUY_OR_NOT_PATH', '/mnt/nas/AutoGluon/AutoML_Krx/DB/Buy_Or_Not/latest_buy_or_not.csv')


class NaverThemeService:
    """Service for accessing NaverTheme data."""

    _df: Optional[pd.DataFrame] = None
    _buy_or_not_df: Optional[pd.DataFrame] = None

    @classmethod
    def _load_data(cls) -> pd.DataFrame:
        """Load and cache the theme data."""
        if cls._df is None:
            if os.path.exists(CSV_PATH):
                cls._df = pd.read_csv(CSV_PATH)
            else:
                cls._df = pd.DataFrame()
        return cls._df

    @classmethod
    def _load_buy_or_not(cls) -> pd.DataFrame:
        """Load and cache the buy_or_not data."""
        if cls._buy_or_not_df is None:
            if os.path.exists(BUY_OR_NOT_PATH):
                cls._buy_or_not_df = pd.read_csv(BUY_OR_NOT_PATH)
                # Ensure ticker is string and zero-padded to 6 digits
                cls._buy_or_not_df['ticker'] = cls._buy_or_not_df['ticker'].astype(str).str.zfill(6)
            else:
                cls._buy_or_not_df = pd.DataFrame()
        return cls._buy_or_not_df

    @classmethod
    def get_stock_themes(cls, stock_name: str) -> Dict[str, Any]:
        """
        Get all themes for a specific stock.

        Args:
            stock_name: Name of the stock (e.g., "삼성전자")

        Returns:
            Dictionary with themes and stock info
        """
        df = cls._load_data()
        if df.empty:
            return {"success": False, "error": "Data not available"}

        # Find the stock
        stock_row = df[df['name'] == stock_name]

        if stock_row.empty:
            # Try partial match
            stock_row = df[df['name'].str.contains(stock_name, na=False)]

        if stock_row.empty:
            return {
                "success": False,
                "error": f"Stock '{stock_name}' not found",
                "themes": []
            }

        row = stock_row.iloc[0]

        # Parse themes from string representation of list
        themes_str = row.get('naverTheme', '[]')
        try:
            if isinstance(themes_str, str):
                themes = ast.literal_eval(themes_str)
            else:
                themes = []
        except:
            themes = []

        return {
            "success": True,
            "stock_name": row['name'],
            "ticker": str(row.get('tickers', row.get('티커', ''))),
            "market": row.get('market', ''),
            "momentum": row.get('mmt', ''),
            "themes": themes,
            "theme_count": len(themes),
            "scores": {
                "bearish": float(row.get('-1', 0)) if pd.notna(row.get('-1')) else 0,
                "neutral": float(row.get('0', 0)) if pd.notna(row.get('0')) else 0,
                "bullish": float(row.get('1', 0)) if pd.notna(row.get('1')) else 0
            }
        }

    @classmethod
    def get_theme_stocks(cls, theme_name: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get all stocks belonging to a specific theme.

        Args:
            theme_name: Name of the theme (e.g., "반도체")
            limit: Maximum number of stocks to return

        Returns:
            Dictionary with stocks in the theme
        """
        df = cls._load_data()
        if df.empty:
            return {"success": False, "error": "Data not available"}

        # Find stocks containing this theme
        def contains_theme(themes_str):
            if pd.isna(themes_str):
                return False
            try:
                themes = ast.literal_eval(themes_str) if isinstance(themes_str, str) else []
                return any(theme_name in t for t in themes)
            except:
                return False

        mask = df['naverTheme'].apply(contains_theme)
        theme_stocks = df[mask]

        if theme_stocks.empty:
            return {
                "success": False,
                "error": f"Theme '{theme_name}' not found or has no stocks",
                "stocks": []
            }

        # Calculate total score and sort
        def calc_score(row):
            try:
                bullish = float(row.get('1', 0)) if pd.notna(row.get('1')) else 0
                bearish = float(row.get('-1', 0)) if pd.notna(row.get('-1')) else 0
                return bullish - bearish
            except:
                return 0

        theme_stocks = theme_stocks.copy()
        theme_stocks['_score'] = theme_stocks.apply(calc_score, axis=1)
        theme_stocks = theme_stocks.sort_values('_score', ascending=False)

        # Load buy_or_not data
        buy_or_not_df = cls._load_buy_or_not()
        buy_or_not_dict = {}
        if not buy_or_not_df.empty:
            buy_or_not_dict = dict(zip(buy_or_not_df['ticker'], buy_or_not_df['buyable']))

        stocks = []
        for _, row in theme_stocks.head(limit).iterrows():
            # Parse market cap (시가총액) - already in 조원 (trillion won)
            market_cap = row.get('시가총액', 0)
            if pd.notna(market_cap):
                market_cap = float(market_cap)  # Already in 조원
            else:
                market_cap = 0

            # Handle NaN values for momentum
            momentum = row.get('mmt', '')
            if pd.isna(momentum):
                momentum = ''

            # Get ticker and lookup buyable status from buy_or_not DB
            ticker = str(row.get('tickers', row.get('티커', '')))
            ticker_padded = ticker.zfill(6) if ticker.isdigit() else ticker
            buyable = buy_or_not_dict.get(ticker_padded, None)  # None if not found

            stocks.append({
                "name": row['name'],
                "ticker": ticker,
                "market": row.get('market', ''),
                "momentum": momentum,
                "market_cap": market_cap,  # 시가총액 in 조원
                "buyable": buyable,  # True/False from buy_or_not DB, None if not found
                "scores": {
                    "bearish": float(row.get('-1', 0)) if pd.notna(row.get('-1')) else 0,
                    "neutral": float(row.get('0', 0)) if pd.notna(row.get('0')) else 0,
                    "bullish": float(row.get('1', 0)) if pd.notna(row.get('1')) else 0
                },
                "total_score": float(row['_score'])
            })

        return {
            "success": True,
            "theme": theme_name,
            "stock_count": len(theme_stocks),
            "stocks": stocks
        }

    @classmethod
    def search_themes(cls, query: str, limit: int = 10) -> List[str]:
        """Search for themes by name."""
        df = cls._load_data()
        if df.empty:
            return []

        all_themes = set()
        for themes_str in df['naverTheme'].dropna():
            try:
                themes = ast.literal_eval(themes_str) if isinstance(themes_str, str) else []
                all_themes.update(themes)
            except:
                pass

        matching = [t for t in all_themes if query.lower() in t.lower()]
        return sorted(matching)[:limit]


# CLI for testing
if __name__ == "__main__":
    import json

    # Test get_stock_themes
    result = NaverThemeService.get_stock_themes("삼성전자")
    print("=== Stock Themes for 삼성전자 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Test get_theme_stocks
    if result.get("themes"):
        first_theme = result["themes"][0]
        print(f"\n=== Stocks in theme: {first_theme} ===")
        theme_result = NaverThemeService.get_theme_stocks(first_theme, limit=5)
        print(json.dumps(theme_result, ensure_ascii=False, indent=2))
