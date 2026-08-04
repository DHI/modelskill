# Forecast lead-time example data: Lewes, Delaware

Water level forecasts at multiple lead times, paired with observations, for use in the
forecast lead-time example. All data originate from NOAA and are in the public domain.

## `lewes_dbofs_forecast.csv`

Long format, one row per (forecast cycle, valid time):

| column | description |
| --- | --- |
| `reference_time` | when the forecast was issued (the model cycle, 4 per day) |
| `valid_time` | the time the forecast is *for* |
| `lead_time` | forecast horizon in hours, `valid_time - reference_time`, 0–48 |
| `dbofs` | forecast water level from NOAA Delaware Bay OFS \[m] |
| `observed` | observed water level, CO-OPS station 8557380 \[m, MSL] |
| `tide` | harmonic tide prediction for the same station \[m, MSL] — a reference forecast with no initialisation, so its error does not grow with lead time |

28 cycles covering 2026-07-26 to 2026-08-01, hourly, 1372 rows.

## Sources

- **Model**: NOAA Delaware Bay Operational Forecast System (DBOFS) station files, from the
  [NOAA OFS data on AWS](https://registry.opendata.aws/noaa-ofs-pds/)
  (`s3://noaa-ofs-pds/dbofs.<date>/dbofs.t<cycle>z.<date>.stations.forecast.nc`).
  The station files contain time series at 64 predefined stations rather than the full model
  grid, which is why they are small enough to subset quickly.
  The AWS bucket holds only a trailing 30-day window; longer archives (back to 2014) are at
  [NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/co-ops-operational-forecast).
- **Observations and tide predictions**: NOAA CO-OPS station 8557380 (Lewes, DE) via the
  [CO-OPS data API](https://api.tidesandcurrents.noaa.gov/api/prod/).

Both are works of the US federal government and are in the public domain.

## Known limitations

These are real caveats, not artefacts of the export — an example built on this data should
acknowledge them rather than paper over them.

- **The model station is identified by position, not by name.** DBOFS station files carry only
  `lon_rho`/`lat_rho`, with no station identifiers. Station index 5 was selected as the nearest
  to the Lewes gauge, about 1.4 km away. This has not been verified against a published DBOFS
  station list.
- **Datums are not reconciled in the data.** Model `zeta` is free-surface relative to the model's
  own reference level; observations are relative to the station MSL datum. Both model columns
  carry a large constant bias as a result — about -0.08 m for `dbofs` and -0.26 m for `tide`. The
  example corrects this with `remove_bias()`, which matters: with the offsets left in, the
  tide-only prediction looks worse than the forecast, and with them removed it is clearly better.
- **No storm event.** Observed water level spans -0.48 to 1.12 m over this week — ordinary
  tidal conditions. A surge event would show the value of the forecast over the tide-only
  reference far more clearly.
- **The 48 h lead time has only 28 points** (one per cycle), so the end of any skill-vs-lead-time
  curve is noisier than the rest.

## Regenerating

`build_dataset.py` re-downloads and rebuilds the CSV. It declares its own dependencies inline
(PEP 723), so it runs standalone. It requires network access, and will only work for dates still
inside the AWS 30-day window — the committed CSV is the durable artefact.

```bash
uv run tests/testdata/forecast_lewes/build_dataset.py
```
