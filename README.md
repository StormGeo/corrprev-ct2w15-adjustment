# CorrPrev CT2W15 Adjustment

Operational utilities to adjust CT2W15 forecast fields using meteorologist-provided daily JSON data.

The pipeline currently corrects:

- hourly 2 m air temperature;
- hourly precipitation distribution;
- weather icon values derived from the meteorologist JSON;
- optional point updates in existing meteorologist NetCDF files.

## Repository Layout

```text
configs/
  paths.yaml        # Local paths and processing parameters
  stations.yaml     # Station latitude/longitude definitions
src/weather_correction/
  core.py           # Correction and NetCDF I/O logic
  cli.py            # Command-line interface
run_correction.py   # Thin local runner
requirements.txt
pyproject.toml
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package usage:

```bash
pip install -e .
```

## Configuration

Edit `configs/paths.yaml` for your local data layout. The file currently keeps local paths for operational testing.

Important sections:

- `paths`: input and output directories;
- `files`: file-name templates for model and meteorologist NetCDFs;
- `model_inputs`: variable names read from model NetCDFs;
- `netcdf_update`: file-name tokens and internal variable names for target NetCDF updates;
- `processing`: forecast window and JSON/date alignment parameters.

`*_file_variable` is the token used in the NetCDF path and file name. `*_variable` is the variable inside the NetCDF file. They can differ, for example:

```yaml
netcdf_update:
  icon_file_variable: weather_icon
  icon_variable: weather_icon_smoothed
```

## Usage

Generate corrected CSV files:

```bash
python run_correction.py --date-rod 20260615 --run-hour 00 --output-mode csv
```

Update NetCDF files only:

```bash
python run_correction.py --date-rod 20260615 --run-hour 00 --output-mode netcdf
```

Generate CSV files and update NetCDF files:

```bash
python run_correction.py --date-rod 20260615 --run-hour 00 --output-mode both --save-validation true
```

The JSON file is selected automatically from `paths.json_dir` using `files.json_template`. The current default layout is `weather/exports/{year}/{julian}/*.json`, and the file name must contain the target date `date_rod + processing.start_day_offset`. A flat `paths.json_dir/*.json` folder is still supported as a fallback.

## Notes

The correction window is controlled by:

```yaml
processing:
  start_day_offset: 1
  forecast_hours: 72
```

The end time is inclusive, so `forecast_hours: 72` produces 73 hourly timestamps.
