"""NOAA ASOS weather station data loader via Iowa Environmental Mesonet."""

from __future__ import annotations

import pickle
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def _fetch_station_list() -> pd.DataFrame:
    """Fetch ASOS station metadata from IEM (per-state networks)."""
    import requests
    all_dfs = []
    for state in _US_STATES:
        url = f"https://mesonet.agron.iastate.edu/sites/networks.php?network={state}_ASOS&format=csv&nohtml=on"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 100:
                df = pd.read_csv(StringIO(resp.text))
                df["state"] = state
                all_dfs.append(df)
        except Exception:
            continue
    if not all_dfs:
        raise RuntimeError("No ASOS station data fetched from IEM")
    result = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Fetched {len(result)} ASOS stations from {len(all_dfs)} states")
    return result


def _download_asos_data(
    stations: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily average temperature for stations from IEM."""
    import requests

    params = {
        "data": "tmpf",
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": [3, 4],
        "year1": start_date[:4],
        "month1": start_date[5:7],
        "day1": start_date[8:10],
        "year2": end_date[:4],
        "month2": end_date[5:7],
        "day2": end_date[8:10],
    }

    all_data = []
    batch_size = 50
    for i in range(0, len(stations), batch_size):
        batch = stations[i:i + batch_size]
        log.info(f"  Downloading batch {i // batch_size + 1}/"
                 f"{(len(stations) + batch_size - 1) // batch_size} ({len(batch)} stations)...")
        batch_params = dict(params)
        batch_params["station"] = batch

        resp = requests.get(IEM_BASE, params=batch_params, timeout=120)
        if resp.status_code == 200 and len(resp.text) > 100:
            df = pd.read_csv(StringIO(resp.text), na_values=["M", "T", ""])
            all_data.append(df)

    if not all_data:
        raise RuntimeError("No data downloaded from IEM")

    return pd.concat(all_data, ignore_index=True)


class WeatherStationLoader:
    """Load NOAA ASOS weather station temperature data."""

    def __init__(
        self,
        n_stations: int = 500,
        start_date: str = "2010-01-01",
        end_date: str = "2023-12-31",
        min_coverage: float = 0.90,
        cache_dir: str = "data/cache",
    ):
        self.n_stations = n_stations
        self.start_date = start_date
        self.end_date = end_date
        self.min_coverage = min_coverage
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame | None]:
        """Load weather station data.

        Returns
        -------
        anomalies : DataFrame (dates × N stations) — daily temperature anomalies
        field_avg : Series — area-weighted field average
        station_ids : list[str]
        metadata : DataFrame with lat/lon for region assignment (or None)
        """
        cache = self.cache_dir / f"weather_n{self.n_stations}.pkl"
        if cache.exists():
            log.info(f"Loading cached weather data from {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)

        log.info("Fetching ASOS station list...")
        meta = _fetch_station_list()
        meta = meta[meta["country"] == "US"] if "country" in meta.columns else meta

        if "lat" in meta.columns and "lon" in meta.columns:
            meta = meta.dropna(subset=["lat", "lon"])
            meta = meta[(meta["lat"] > 24) & (meta["lat"] < 50)]
            meta = meta[(meta["lon"] > -130) & (meta["lon"] < -65)]

        id_col = "stid" if "stid" in meta.columns else ("id" if "id" in meta.columns else meta.columns[0])
        station_ids_all = meta[id_col].tolist()
        log.info(f"Found {len(station_ids_all)} CONUS ASOS stations")

        log.info(f"Downloading temperature data ({self.start_date} → {self.end_date})...")
        raw = _download_asos_data(station_ids_all, self.start_date, self.end_date)

        raw["valid"] = pd.to_datetime(raw["valid"])
        raw["date"] = raw["valid"].dt.date
        raw["tmpf"] = pd.to_numeric(raw["tmpf"], errors="coerce")
        raw = raw.dropna(subset=["tmpf"])

        daily = raw.groupby(["station", "date"])["tmpf"].mean().reset_index()
        pivot = daily.pivot(index="date", columns="station", values="tmpf")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()

        coverage = pivot.notna().mean()
        good = coverage[coverage >= self.min_coverage].index.tolist()
        log.info(f"Stations with >= {self.min_coverage:.0%} coverage: {len(good)}")
        pivot = pivot[good].ffill(limit=3).bfill(limit=3).dropna(axis=1)

        if len(pivot.columns) > self.n_stations:
            rng = np.random.RandomState(42)
            selected = rng.choice(pivot.columns.tolist(), self.n_stations, replace=False)
            pivot = pivot[sorted(selected)]

        climatology = pivot.groupby(pivot.index.dayofyear).transform("mean")
        anomalies = pivot - climatology

        field_avg = anomalies.mean(axis=1)
        station_ids = list(anomalies.columns)

        meta_out = None
        if id_col in meta.columns:
            meta_out = meta[meta[id_col].isin(station_ids)].copy()

        log.info(f"Final: {len(station_ids)} stations, {len(anomalies)} days")

        result = (anomalies, field_avg, station_ids, meta_out)
        with open(cache, "wb") as f:
            pickle.dump(result, f)
        return result

    def assign_regions(self, metadata: pd.DataFrame, grid_size: int = 3) -> np.ndarray:
        """Assign stations to spatial regions based on lat/lon grid."""
        lats = metadata["lat"].values
        lons = metadata["lon"].values

        lat_bins = np.linspace(lats.min(), lats.max() + 1e-6, grid_size + 1)
        lon_bins = np.linspace(lons.min(), lons.max() + 1e-6, grid_size + 1)

        lat_idx = np.digitize(lats, lat_bins) - 1
        lon_idx = np.digitize(lons, lon_bins) - 1

        regions = lat_idx * grid_size + lon_idx
        return regions.astype(np.int64)
