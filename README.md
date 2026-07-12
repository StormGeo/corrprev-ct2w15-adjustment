# CorrPrev CT2W15 Adjustment

Generate corrected operational CSV files from explicit NetCDF and meteorologist JSON inputs.

Each execution processes one variable and writes one CSV with this schema:

```csv
date,data,city,lat,lon
```

- `date`: local forecast timestamp;
- `data`: corrected value;
- `city`: station/city id from `configs/stations.yaml`;
- `lat`, `lon`: station coordinates from `configs/stations.yaml`.

The output path is generated as:

```text
{output_dir}/{variable}/{year}/{julian}/meteorologist_{variable}_M000_{YYYYMMDDHH}.csv
```

`YYYYMMDDHH`, `year`, and `julian` are based on the local execution time.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Examples

Temperature:

```bash
python run_correction.py \
  --input-nc /path/to/temperature.nc \
  --nc-variable 2m_air_temperature \
  --met-json /path/to/weather-export.json \
  --forecast-start "2026-06-16 00:00" \
  --output-dir /path/to/outputs
```

Total precipitation:

```bash
python run_correction.py \
  --input-nc /path/to/total_precipitation.nc \
  --nc-variable total_precipitation \
  --met-json /path/to/weather-export.json \
  --forecast-start "2026-06-16 00:00" \
  --output-dir /path/to/outputs
```

If `--cape-nc` is omitted, the script tries to infer it by replacing
`total_precipitation` with `cape_index` in `--input-nc`.

Weather icon:

```bash
python run_correction.py \
  --nc-variable weather_icon \
  --met-json /path/to/weather-export.json \
  --forecast-start "2026-06-16 00:00" \
  --output-dir /path/to/outputs
```

`weather_icon` uses the JSON daily period icons and distributes them hourly by dawn,
morning, afternoon, and night. It does not require `--input-nc`.

## Defaults

If `--forecast-start` is omitted, the script uses today's 00:00 in the configured
timezone. `--forecast-hours` defaults to `processing.forecast_hours`, currently 72.

The end time is inclusive, so 72 hours produces 73 hourly timestamps.

## Missing Cities

Default behavior is:

```bash
--missing-city-policy skip
```

Stations missing from the JSON are skipped. If the JSON also contains cities
that are not configured in `configs/stations.yaml`, they are ignored and a warning
is printed asking you to contact support.
