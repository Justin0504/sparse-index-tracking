"""EPA AQS PM2.5 air quality monitoring data loader."""

from __future__ import annotations

import pickle
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

EPA_URL_TEMPLATE = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"


class EPAAirQualityLoader:
    """Load EPA AQS PM2.5 daily monitoring data."""

    def __init__(
        self,
        n_stations: int = 500,
        start_year: int = 2010,
        end_year: int = 2023,
        min_coverage: float = 0.90,
        cache_dir: str = "data/cache",
    ):
        self.n_stations = n_stations
        self.start_year = start_year
        self.end_year = end_year
        self.min_coverage = min_coverage
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame | None]:
        """Load PM2.5 monitoring data.

        Returns
        -------
        anomalies : DataFrame (dates × N stations) — daily PM2.5 anomalies
        field_avg : Series — equal-weighted field average
        station_ids : list[str]
        metadata : DataFrame with Latitude/Longitude for region assignment
        """
        cache = self.cache_dir / f"epa_n{self.n_stations}.pkl"
        if cache.exists():
            log.info(f"Loading cached EPA data from {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)

        import requests

        all_data = []
        for year in range(self.start_year, self.end_year + 1):
            url = EPA_URL_TEMPLATE.format(year=year)
            log.info(f"  Downloading EPA PM2.5 {year}...")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, usecols=[
                            "State Code", "County Code", "Site Num",
                            "Date Local", "Arithmetic Mean",
                            "Latitude", "Longitude",
                        ])
                        all_data.append(df)
            except Exception as e:
                log.warning(f"  {year} download failed: {e}")

        if not all_data:
            raise RuntimeError("No EPA data downloaded")

        raw = pd.concat(all_data, ignore_index=True)
        raw["station_id"] = (
            raw["State Code"].astype(str).str.zfill(2) + "-"
            + raw["County Code"].astype(str).str.zfill(3) + "-"
            + raw["Site Num"].astype(str).str.zfill(4)
        )
        raw["date"] = pd.to_datetime(raw["Date Local"])
        raw["pm25"] = pd.to_numeric(raw["Arithmetic Mean"], errors="coerce")
        raw = raw.dropna(subset=["pm25"])

        raw = raw[
            (raw["Latitude"] > 24) & (raw["Latitude"] < 50)
            & (raw["Longitude"] > -130) & (raw["Longitude"] < -65)
        ]

        meta = raw.groupby("station_id").agg(
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
        ).reset_index()

        daily = raw.groupby(["station_id", "date"])["pm25"].mean().reset_index()
        pivot = daily.pivot(index="date", columns="station_id", values="pm25")
        pivot = pivot.sort_index()

        coverage = pivot.notna().mean()
        good = coverage[coverage >= self.min_coverage].index.tolist()
        log.info(f"Stations with >= {self.min_coverage:.0%} coverage: {len(good)}")
        pivot = pivot[good].ffill(limit=7).bfill(limit=7).dropna(axis=1)

        if len(pivot.columns) > self.n_stations:
            rng = np.random.RandomState(42)
            selected = rng.choice(pivot.columns.tolist(), self.n_stations, replace=False)
            pivot = pivot[sorted(selected)]

        climatology = pivot.groupby(pivot.index.month).transform("mean")
        anomalies = pivot - climatology

        field_avg = anomalies.mean(axis=1)
        station_ids = list(anomalies.columns)

        meta_out = meta[meta["station_id"].isin(station_ids)].copy()

        log.info(f"Final: {len(station_ids)} stations, {len(anomalies)} days")

        result = (anomalies, field_avg, station_ids, meta_out)
        with open(cache, "wb") as f:
            pickle.dump(result, f)
        return result

    def assign_regions(self, metadata: pd.DataFrame, grid_size: int = 3) -> np.ndarray:
        """Assign stations to spatial regions based on lat/lon grid."""
        lats = metadata["Latitude"].values
        lons = metadata["Longitude"].values

        lat_bins = np.linspace(lats.min(), lats.max() + 1e-6, grid_size + 1)
        lon_bins = np.linspace(lons.min(), lons.max() + 1e-6, grid_size + 1)

        lat_idx = np.digitize(lats, lat_bins) - 1
        lon_idx = np.digitize(lons, lon_bins) - 1

        regions = lat_idx * grid_size + lon_idx
        return regions.astype(np.int64)
