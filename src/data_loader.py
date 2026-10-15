
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf

from src import config


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Bereinigte Schlusskurse von start bis einschliesslich end (eine Spalte je Ticker)."""
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end_exclusive, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Keine Kursdaten erhalten (Internetverbindung oder Ticker pruefen).")
    prices = raw["Close"]
    if isinstance(prices, pd.Series):            # nur ein Ticker
        prices = prices.to_frame(tickers[0])
    prices = prices[tickers]                      # feste Spaltenreihenfolge
    missing = [t for t in tickers if prices[t].dropna().empty]
    if missing:
        raise RuntimeError(f"Keine Daten fuer: {', '.join(missing)}")
    return prices.dropna(how="any")


def cache_path() -> str:
    return os.path.join(config.DATA_DIR, f"kurse_bis_{config.END_DATE}.csv")


def load_prices(use_cache: bool = True) -> pd.DataFrame:
    path = cache_path()
    if use_cache and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    prices = download_prices(list(config.TICKERS), config.START_DATE, config.END_DATE)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    prices.to_csv(path)
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")