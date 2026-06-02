"""Unified multi-market data loader for global index tracking experiments."""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.logging import get_logger

log = get_logger(__name__)

UNIVERSE_CONFIGS = {
    "sp500": {
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_idx": 0,
        "ticker_col": "Symbol",
        "suffix": "",
        "index_ticker": "^GSPC",
        "fix_dots": True,
    },
    "sp400": {
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "table_idx": 0,
        "ticker_col": "Symbol",
        "suffix": "",
        "index_ticker": "^GSPC",
        "fix_dots": True,
    },
    "sp600": {
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "table_idx": 0,
        "ticker_col": "Symbol",
        "suffix": "",
        "index_ticker": "^GSPC",
        "fix_dots": True,
    },
    "ftse100": {
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_idx": 6,
        "ticker_col": "Ticker",
        "suffix": ".L",
        "index_ticker": "^FTSE",
        "fix_dots": False,
    },
    "nikkei225": {
        "wiki_url": "https://en.wikipedia.org/wiki/Nikkei_225",
        "table_idx": None,
        "ticker_col": None,
        "suffix": ".T",
        "index_ticker": "^N225",
        "fix_dots": False,
        "custom_fetcher": "_fetch_nikkei225",
    },
    "eurostoxx50": {
        "wiki_url": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
        "table_idx": 3,
        "ticker_col": "Ticker",
        "suffix": "",
        "index_ticker": "^STOXX50E",
        "fix_dots": False,
    },
    "asx200": {
        "wiki_url": "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        "table_idx": 2,
        "ticker_col": "Code",
        "suffix": ".AX",
        "index_ticker": "^AXJO",
        "fix_dots": False,
    },
    "hangseng": {
        "wiki_url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "table_idx": 6,
        "ticker_col": "Ticker",
        "suffix": ".HK",
        "index_ticker": "^HSI",
        "fix_dots": False,
    },
}


def _fetch_nikkei225() -> list[str]:
    """Fetch Nikkei 225 tickers — primary: topforeignstocks, fallback: Wikipedia regex."""
    import re

    try:
        req = Request(
            "https://topforeignstocks.com/indices/the-components-of-the-nikkei-225-index/",
            headers={"User-Agent": "Mozilla/5.0 (research)"},
        )
        html = urlopen(req, timeout=15).read().decode("utf-8")
        matches = re.findall(r"(\d{4})\.T", html)
        tickers = sorted(set(m + ".T" for m in matches if len(m) == 4))
        if len(tickers) >= 200:
            return tickers
    except Exception:
        pass

    req = Request(
        "https://en.wikipedia.org/wiki/Nikkei_225",
        headers={"User-Agent": "Mozilla/5.0 (research)"},
    )
    html = urlopen(req, timeout=15).read().decode("utf-8")
    matches = re.findall(r"(\d{4})\.T", html)
    tickers = sorted(set(m + ".T" for m in matches if len(m) == 4))
    if len(tickers) < 100:
        tables = pd.read_html(io.StringIO(html))
        for t in tables:
            for col in t.columns:
                vals = t[col].astype(str)
                nums = vals[vals.str.match(r"^\d{4}$")]
                if len(nums) > 50:
                    tickers = sorted(set(n + ".T" for n in nums.tolist()))
                    break
            if len(tickers) > 100:
                break
    return tickers


def _fetch_tickers(config: dict) -> list[str]:
    if config.get("custom_fetcher") == "_fetch_nikkei225":
        return _fetch_nikkei225()

    req = Request(config["wiki_url"], headers={"User-Agent": "Mozilla/5.0 (research)"})
    html = urlopen(req, timeout=15).read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    table = tables[config["table_idx"]]

    col = config["ticker_col"]
    if col not in table.columns:
        for c in table.columns:
            if "ticker" in str(c).lower() or "symbol" in str(c).lower() or "code" in str(c).lower():
                col = c
                break

    tickers = table[col].astype(str).tolist()
    suffix = config["suffix"]

    cleaned = []
    for t in tickers:
        t = t.strip()
        # Remove exchange prefixes like "SEHK: " or "TSE: "
        if ":" in t:
            t = t.split(":")[-1].strip().replace("\xa0", "")
        if config.get("fix_dots"):
            t = t.replace(".", "-")
        if suffix:
            t_no_suffix = t.replace(suffix, "") if t.endswith(suffix) else t
            # Zero-pad Hong Kong tickers (e.g., "1" → "0001")
            if suffix == ".HK" and t_no_suffix.isdigit():
                t_no_suffix = t_no_suffix.zfill(4)
            t = t_no_suffix + suffix if not t.endswith(suffix) else t
        cleaned.append(t)

    return sorted(set(cleaned))


def load_universe(
    name: str,
    start_date: str = "2015-01-01",
    end_date: str = "2025-12-31",
    min_coverage: float = 0.90,
    cache_dir: str = "data/cache",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load price data for a market universe.

    Parameters
    ----------
    name : one of sp500, sp400, sp600, sp1500, ftse100, nikkei225, eurostoxx50, asx200, hangseng
    start_date, end_date : date range
    min_coverage : minimum fraction of non-NaN trading days
    cache_dir : directory for pickle cache

    Returns
    -------
    prices : DataFrame (dates × N stocks)
    index_prices : Series (index level)
    tickers : list[str]
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{name}_{start_date}_{end_date}_{min_coverage}".encode()).hexdigest()[:12]
    pkl = cache_path / f"{name}_{key}.pkl"

    if pkl.exists():
        log.info(f"Loading cached {name} data from {pkl}")
        with open(pkl, "rb") as f:
            return pickle.load(f)

    if name == "sp1500":
        return _load_sp1500(start_date, end_date, min_coverage, pkl)

    config = UNIVERSE_CONFIGS[name]
    log.info(f"Fetching {name} tickers from Wikipedia...")
    tickers = _fetch_tickers(config)
    log.info(f"Got {len(tickers)} {name} tickers")

    log.info(f"Downloading {len(tickers)} stocks ({start_date} → {end_date})...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(axis=1, how="all")
    else:
        prices = raw.dropna(axis=1, how="all")

    coverage = prices.notna().mean()
    good = coverage[coverage >= min_coverage].index.tolist()
    log.info(f"Stocks with >= {min_coverage:.0%} coverage: {len(good)}/{len(prices.columns)}")
    prices = prices[good].ffill().bfill().dropna()

    idx_ticker = config["index_ticker"]
    log.info(f"Downloading index {idx_ticker}...")
    idx = yf.download(idx_ticker, start=start_date, end=end_date, auto_adjust=True)
    index_prices = idx["Close"].squeeze()

    common = prices.index.intersection(index_prices.index)
    prices = prices.loc[common]
    index_prices = index_prices.loc[common]
    final_tickers = list(prices.columns)

    log.info(f"Final {name}: {len(final_tickers)} stocks, {len(prices)} trading days")

    result = (prices, index_prices, final_tickers)
    with open(pkl, "wb") as f:
        pickle.dump(result, f)
    return result


def _load_sp1500(
    start_date: str, end_date: str, min_coverage: float, cache_path: Path
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load S&P 1500 = S&P 500 + S&P 400 + S&P 600."""
    all_tickers = set()
    for idx_name in ["sp500", "sp400", "sp600"]:
        config = UNIVERSE_CONFIGS[idx_name]
        try:
            tickers = _fetch_tickers(config)
            all_tickers.update(tickers)
            log.info(f"  {idx_name}: {len(tickers)} tickers")
        except Exception as e:
            log.warning(f"  {idx_name} fetch failed: {e}")

    all_tickers = sorted(all_tickers)
    log.info(f"S&P 1500 total unique tickers: {len(all_tickers)}")

    batch_size = 200
    all_prices = []
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        log.info(f"  Downloading batch {i // batch_size + 1}/{(len(all_tickers) + batch_size - 1) // batch_size} ({len(batch)} tickers)...")
        raw = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            batch_prices = raw["Close"].dropna(axis=1, how="all")
        else:
            batch_prices = raw.dropna(axis=1, how="all")
        all_prices.append(batch_prices)

    prices = pd.concat(all_prices, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    coverage = prices.notna().mean()
    good = coverage[coverage >= min_coverage].index.tolist()
    log.info(f"Stocks with >= {min_coverage:.0%} coverage: {len(good)}/{len(prices.columns)}")
    prices = prices[good].ffill().bfill().dropna()

    log.info("Downloading S&P 1500 proxy index (^GSPC)...")
    idx = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    index_prices = idx["Close"].squeeze()

    common = prices.index.intersection(index_prices.index)
    prices = prices.loc[common]
    index_prices = index_prices.loc[common]
    final_tickers = list(prices.columns)

    log.info(f"Final S&P 1500: {len(final_tickers)} stocks, {len(prices)} trading days")

    result = (prices, index_prices, final_tickers)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
