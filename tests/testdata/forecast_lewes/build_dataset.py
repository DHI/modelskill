# /// script
# requires-python = ">=3.10"
# dependencies = ["xarray", "netCDF4", "pandas"]
# ///
"""Rebuild lewes_dbofs_forecast.csv from NOAA sources.

Pairs NOAA Delaware Bay OFS (DBOFS) water level forecasts at multiple lead times with
observations from CO-OPS station 8557380 (Lewes, DE), plus harmonic tide predictions as a
reference forecast.

Requires network access. The AWS bucket keeps only a trailing 30-day window, so DATES must
be recent; see README.md for the NCEI archive if you need an older period.

    uv run build_dataset.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import xarray as xr

DATES = [f"202607{d:02d}" for d in range(26, 32)] + ["20260801"]
CYCLES = ["00", "06", "12", "18"]

STATION_ID = "8557380"  # CO-OPS Lewes, DE
STATION_IX = 5  # nearest DBOFS station; see "Known limitations" in README.md

S3 = "https://noaa-ofs-pds.s3.amazonaws.com"
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

OUT = Path(__file__).parent / "lewes_dbofs_forecast.csv"


def read_cycle(date: str, cycle: str) -> pd.DataFrame:
    """Read one forecast cycle and return its hourly water level by lead time."""
    url = f"{S3}/dbofs.{date}/dbofs.t{cycle}z.{date}.stations.forecast.nc"
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        urlretrieve(url, tmp.name)  # noqa: S310 - fixed https host
        with xr.open_dataset(tmp.name, decode_timedelta=False) as ds:
            time = pd.DatetimeIndex(ds.ocean_time.values)
            zeta = pd.Series(ds.zeta.isel(station=STATION_IX).values, index=time)

    zeta = zeta[zeta.index.minute == 0]  # 6-min output -> hourly
    reference_time = time[0]

    return pd.DataFrame(
        {
            "reference_time": reference_time,
            "valid_time": zeta.index,
            "lead_time": ((zeta.index - reference_time).total_seconds() / 3600).astype(
                int
            ),
            "dbofs": zeta.values,
        }
    )


def read_coops(product: str, column: str, **extra: str) -> pd.DataFrame:
    """Read a CO-OPS product for the observation station."""
    params = {
        "product": product,
        "application": "modelskill",
        "begin_date": DATES[0],
        "end_date": "20260804",
        "datum": "MSL",
        "station": STATION_ID,
        "time_zone": "gmt",
        "units": "metric",
        "format": "csv",
        **extra,
    }
    url = API + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    df = pd.read_csv(url, parse_dates=["Date Time"])
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df.rename(columns={"Date Time": "valid_time"})[["valid_time", column]]


def main() -> None:
    forecasts = pd.concat(
        [read_cycle(date, cycle) for date in DATES for cycle in CYCLES],
        ignore_index=True,
    )

    observed = read_coops("water_level", "Water Level").rename(
        columns={"Water Level": "observed"}
    )
    observed = observed[observed.valid_time.dt.minute == 0]
    tide = read_coops("predictions", "Prediction", interval="h").rename(
        columns={"Prediction": "tide"}
    )

    df = forecasts.merge(observed, on="valid_time").merge(tide, on="valid_time")
    df = df.sort_values(["reference_time", "lead_time"]).reset_index(drop=True)
    df[["dbofs", "observed", "tide"]] = df[["dbofs", "observed", "tide"]].round(3)

    df.to_csv(OUT, index=False)
    print(
        f"wrote {OUT} - {len(df)} rows, {df.reference_time.nunique()} cycles, "
        f"lead times {df.lead_time.min()}-{df.lead_time.max()} h"
    )


if __name__ == "__main__":
    main()
