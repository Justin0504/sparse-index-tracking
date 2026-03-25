"""S&P 500 data loader using yfinance."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger

log = get_logger(__name__)

# Top 100 S&P 500 by market cap (fallback if Wikipedia scrape fails)
_TOP100_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO",
    "JPM", "UNH", "V", "XOM", "MA", "COST", "PG", "JNJ", "HD", "ABBV",
    "WMT", "NFLX", "BAC", "CRM", "ORCL", "CVX", "MRK", "KO", "AMD", "PEP",
    "TMO", "CSCO", "ACN", "LIN", "ADBE", "MCD", "ABT", "WFC", "PM", "IBM",
    "GE", "DHR", "NOW", "ISRG", "DIS", "TXN", "CAT", "QCOM", "INTU", "VZ",
    "GS", "AMGN", "BKNG", "AMAT", "AXP", "PFE", "T", "SPGI", "LOW", "CMCSA",
    "BLK", "RTX", "UNP", "SYK", "NEE", "HON", "TJX", "SCHW", "BA", "DE",
    "PLD", "VRTX", "MMC", "COP", "BSX", "BMY", "LRCX", "ADP", "FI", "PANW",
    "MDLZ", "ADI", "SBUX", "GILD", "KLAC", "CB", "CI", "SO", "MO", "SHW",
    "AMT", "DUK", "CME", "ICE", "ZTS", "SNPS", "CDNS", "PGR", "MCK", "USB",
]


def _fetch_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia, with fallback."""
    try:
        req = Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (sparse-index-tracking research)"},
        )
        html = urlopen(req, timeout=10).read().decode("utf-8")
        table = pd.read_html(html)[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Fetched {len(tickers)} tickers from Wikipedia")
        return sorted(tickers)
    except Exception as e:
        log.warning(f"Wikipedia scrape failed ({e}), using hardcoded top-100 fallback")
        return sorted(_TOP100_FALLBACK)


class SP500DataLoader:
    """Downloads and caches S&P 500 adjusted close prices and index returns."""

    def __init__(
        self,
        start_date: str = "2006-01-01",
        end_date: str = "2025-12-31",
        top_n: int = 100,
        cache_dir: str = "data/cache",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.top_n = top_n
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self) -> str:
        key = f"{self.start_date}_{self.end_date}_{self.top_n}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def load(self) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Returns
        -------
        prices : DataFrame  (dates × stocks) adjusted close prices
        index_prices : Series  S&P 500 index prices
        tickers : list[str]  selected tickers (top N by market cap)
        """
        cache_path = self.cache_dir / f"sp500_{self._cache_key()}.pkl"
        if cache_path.exists():
            log.info(f"Loading cached data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        log.info("Fetching S&P 500 tickers...")
        all_tickers = _fetch_sp500_tickers()

        log.info(f"Downloading prices for {len(all_tickers)} stocks ({self.start_date} → {self.end_date})...")
        raw = yf.download(
            all_tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            threads=True,
        )
        prices = raw["Close"].dropna(axis=1, how="all")

        # Pre-screen to top N by market cap
        log.info(f"Pre-screening to top {self.top_n} stocks by market cap proxy...")
        tickers_with_data = [t for t in all_tickers if t in prices.columns]

        # Use last price as market-cap proxy (avoids slow per-ticker API calls)
        last_prices = prices.iloc[-1]
        ranked = last_prices.sort_values(ascending=False).index.tolist()
        selected = [t for t in ranked if t in tickers_with_data][:self.top_n]
        prices = prices[selected].dropna()

        # Download S&P 500 index
        log.info("Downloading S&P 500 index (^GSPC)...")
        idx = yf.download("^GSPC", start=self.start_date, end=self.end_date, auto_adjust=True)
        index_prices = idx["Close"].squeeze()

        # Align dates
        common_dates = prices.index.intersection(index_prices.index)
        prices = prices.loc[common_dates]
        index_prices = index_prices.loc[common_dates]

        log.info(f"Final dataset: {len(common_dates)} trading days, {len(selected)} stocks")

        result = (prices, index_prices, selected)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
