"""
Universal Chart Data Provider
==============================
Provides OHLCV data with technical indicators for any market.
Each market project imports this module and configures its data path.

Usage:
    from shared.chart.universal_chart import ChartDataProvider

    # Configure for your market
    provider = ChartDataProvider(
        data_path="/mnt/nas/AutoGluon/AutoML_Usa/USANOTTRAINED",
        market_name="USA"
    )

    # Get chart data
    data = provider.get_ohlcv("AAPL", days=180)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class ChartDataProvider:
    """Universal chart data provider for any market."""

    # Market configurations
    MARKET_CONFIGS = {
        "KRX": {
            "path": "/mnt/nas/AutoGluon/AutoML_Krx/KRXNOTTRAINED",
            "file_pattern": "{symbol}.csv",
            "date_column": None,  # Uses index
            "encoding": "utf-8"
        },
        "USA": {
            "path": "/mnt/nas/AutoGluon/AutoML_Usa/USANOTTRAINED",
            "file_pattern": "{symbol}.csv",
            "date_column": "Date",
            "encoding": "utf-8"
        },
        "JAPAN": {
            "path": "/mnt/nas/AutoGluon/AutoML_Japan/JAPANNOTTRAINED",
            "file_pattern": "{symbol}.csv",
            "date_column": None,
            "encoding": "utf-8"
        },
        "INDIA": {
            "path": "/mnt/nas/AutoGluon/AutoML_India/INDIANOTTRAINED",
            "file_pattern": "{symbol}.csv",
            "date_column": None,
            "encoding": "utf-8"
        },
        "HONGKONG": {
            "path": "/mnt/nas/AutoGluon/AutoML_Hongkong/HONGKONGNOTTRAINED",
            "file_pattern": "{symbol}.csv",
            "date_column": None,
            "encoding": "utf-8"
        }
    }

    def __init__(self, data_path: str = None, market_name: str = None):
        """
        Initialize chart data provider.

        Args:
            data_path: Path to OHLCV CSV files (optional if market_name provided)
            market_name: Market name (KRX, USA, JAPAN, INDIA, HONGKONG)
        """
        if market_name and market_name.upper() in self.MARKET_CONFIGS:
            config = self.MARKET_CONFIGS[market_name.upper()]
            self.data_path = config["path"]
            self.date_column = config["date_column"]
            self.encoding = config["encoding"]
            self.market_name = market_name.upper()
        else:
            self.data_path = data_path
            self.date_column = None
            self.encoding = "utf-8"
            self.market_name = market_name or "CUSTOM"

    def get_ohlcv(
        self,
        symbol: str,
        days: int = 180,
        include_indicators: bool = True
    ) -> Dict[str, Any]:
        """
        Get OHLCV data with optional technical indicators.

        Args:
            symbol: Stock symbol or name
            days: Number of days to return
            include_indicators: Whether to include MA, RSI, Bollinger Bands

        Returns:
            Dict with success status, OHLCV data, and indicators
        """
        csv_path = os.path.join(self.data_path, f"{symbol}.csv")

        if not os.path.exists(csv_path):
            return {
                "success": False,
                "error": f"Data file not found: {symbol}",
                "symbol": symbol,
                "market": self.market_name
            }

        try:
            # Load CSV
            if self.date_column:
                df = pd.read_csv(csv_path, encoding=self.encoding, parse_dates=[self.date_column])
                df.set_index(self.date_column, inplace=True)
            else:
                df = pd.read_csv(csv_path, encoding=self.encoding, index_col=0, parse_dates=True)

            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()

            # Sort by date ascending
            df.sort_index(inplace=True)

            # Calculate indicators on full data
            if include_indicators:
                df = self._add_indicators(df)

            # Get last N days
            df_recent = df.tail(days).copy()

            # Format response
            result = {
                "success": True,
                "symbol": symbol,
                "market": self.market_name,
                "total_records": len(df),
                "returned_records": len(df_recent),
                "date_range": {
                    "start": df_recent.index[0].strftime("%Y-%m-%d") if len(df_recent) > 0 else None,
                    "end": df_recent.index[-1].strftime("%Y-%m-%d") if len(df_recent) > 0 else None
                },
                "ohlcv": self._format_ohlcv(df_recent),
            }

            if include_indicators:
                result["indicators"] = {
                    "ma10": self._format_series(df_recent, "ma10"),
                    "ma20": self._format_series(df_recent, "ma20"),
                    "ma60": self._format_series(df_recent, "ma60"),
                    "rsi": self._format_series(df_recent, "rsi"),
                    "bb_upper": self._format_series(df_recent, "bb_upper"),
                    "bb_middle": self._format_series(df_recent, "bb_middle"),
                    "bb_lower": self._format_series(df_recent, "bb_lower"),
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "market": self.market_name
            }

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""

        # Moving Averages
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()

        # RSI (14 period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20 period, 2 std)
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

        return df

    def _format_ohlcv(self, df: pd.DataFrame) -> List[Dict]:
        """Format OHLCV data for chart consumption."""
        records = []
        for idx, row in df.iterrows():
            record = {
                "time": idx.strftime("%Y-%m-%d"),
                "open": round(row["open"], 4) if pd.notna(row["open"]) else None,
                "high": round(row["high"], 4) if pd.notna(row["high"]) else None,
                "low": round(row["low"], 4) if pd.notna(row["low"]) else None,
                "close": round(row["close"], 4) if pd.notna(row["close"]) else None,
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0
            }
            records.append(record)
        return records

    def _format_series(self, df: pd.DataFrame, column: str) -> List[Dict]:
        """Format a single series for chart consumption."""
        if column not in df.columns:
            return []

        records = []
        for idx, value in df[column].items():
            if pd.notna(value):
                records.append({
                    "time": idx.strftime("%Y-%m-%d"),
                    "value": round(value, 4)
                })
        return records

    def list_symbols(self, limit: int = 100) -> List[str]:
        """List available symbols in this market."""
        if not os.path.exists(self.data_path):
            return []

        files = [f.replace(".csv", "") for f in os.listdir(self.data_path) if f.endswith(".csv")]
        return sorted(files)[:limit]


# Convenience functions for Flask/FastAPI integration
def create_chart_blueprint(market_name: str, url_prefix: str = "/chart"):
    """
    Create a Flask blueprint for chart endpoints.

    Usage:
        from shared.chart.universal_chart import create_chart_blueprint

        chart_bp = create_chart_blueprint("USA", url_prefix="/usa/chart")
        app.register_blueprint(chart_bp)
    """
    from flask import Blueprint, jsonify, request

    bp = Blueprint(f"chart_{market_name.lower()}", __name__, url_prefix=url_prefix)
    provider = ChartDataProvider(market_name=market_name)

    @bp.route("/ohlcv")
    def ohlcv():
        symbol = request.args.get("symbol", "")
        days = int(request.args.get("days", 180))
        indicators = request.args.get("indicators", "true").lower() == "true"

        if not symbol:
            return jsonify({"success": False, "error": "Missing symbol parameter"}), 400

        data = provider.get_ohlcv(symbol, days=days, include_indicators=indicators)
        return jsonify(data)

    @bp.route("/symbols")
    def symbols():
        limit = int(request.args.get("limit", 100))
        return jsonify({
            "success": True,
            "market": market_name,
            "symbols": provider.list_symbols(limit=limit)
        })

    return bp


# Test
if __name__ == "__main__":
    # Test each market
    for market in ["KRX", "USA", "JAPAN", "INDIA", "HONGKONG"]:
        provider = ChartDataProvider(market_name=market)
        symbols = provider.list_symbols(limit=3)
        print(f"\n{market}: {symbols}")

        if symbols:
            data = provider.get_ohlcv(symbols[0], days=5, include_indicators=True)
            if data["success"]:
                print(f"  Records: {data['returned_records']}, Date: {data['date_range']}")
            else:
                print(f"  Error: {data['error']}")
