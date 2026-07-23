from __future__ import annotations

import argparse
from pathlib import Path

from weather_correction.core import (
    build_output_path,
    execution_timestamp,
    load_yaml,
    parse_forecast_start,
    run_single_variable_correction,
)


def resolve_output_path(
    output_path_or_dir: str | Path,
    variable: str,
    run_timestamp,
) -> Path:
    output_path_or_dir = Path(output_path_or_dir)
    if output_path_or_dir.suffix.lower() == ".csv":
        return output_path_or_dir
    return build_output_path(output_path_or_dir, variable, run_timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an operational corrected CSV for one forecast variable."
    )
    parser.add_argument(
        "--input-nc",
        default=None,
        help="Input NetCDF file path for the requested variable. Not required for weather_icon.",
    )
    parser.add_argument(
        "--nc-variable",
        required=True,
        choices=["2m_air_temperature", "total_precipitation", "weather_icon"],
        help="Variable to process. This is also used in the output directory and file name.",
    )
    parser.add_argument(
        "--met-json",
        required=True,
        help="Meteorologist JSON file path.",
    )
    parser.add_argument(
        "--cape-nc",
        default=None,
        help=(
            "CAPE NetCDF path, required for total_precipitation when it cannot be "
            "inferred by replacing total_precipitation with cape_index in --input-nc."
        ),
    )
    parser.add_argument(
        "--forecast-start",
        default=None,
        help=(
            "Forecast start in local time, for example '2026-06-16 00:00'. "
            "Defaults to today's 00:00 in the configured timezone."
        ),
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=None,
        help="Forecast horizon in hours. Defaults to processing.forecast_hours or 72.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Base output directory or full CSV path. Defaults to paths.output_dir "
            "from YAML."
        ),
    )
    parser.add_argument(
        "--missing-city-policy",
        choices=["skip", "error"],
        default="skip",
        help=(
            "How to handle stations missing from the JSON. Extra JSON cities are "
            "skipped with a warning. Default: skip."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="YAML file with runtime parameters.",
    )
    parser.add_argument(
        "--stations",
        default="configs/stations.yaml",
        help="YAML file with station metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    stations_cfg = load_yaml(args.stations)
    timezone = cfg["run"].get("timezone", "America/Sao_Paulo")

    forecast_start = parse_forecast_start(args.forecast_start, timezone)
    forecast_hours = int(
        args.forecast_hours
        if args.forecast_hours is not None
        else cfg.get("processing", {}).get("forecast_hours", 72)
    )
    output_dir = args.output_dir or cfg.get("paths", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Pass --output-dir or configure paths.output_dir in the YAML file.")

    if args.nc_variable != "weather_icon" and not args.input_nc:
        raise ValueError(f"--input-nc is required for {args.nc_variable}.")

    output = run_single_variable_correction(
        nc_variable=args.nc_variable,
        input_nc=args.input_nc,
        met_json=args.met_json,
        stations_cfg=stations_cfg,
        cfg=cfg,
        forecast_start=forecast_start,
        forecast_hours=forecast_hours,
        missing_city_policy=args.missing_city_policy,
        cape_nc=args.cape_nc,
    )

    run_timestamp = execution_timestamp(timezone)
    output_path = resolve_output_path(output_dir, args.nc_variable, run_timestamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Variable: {args.nc_variable}")
    print(f"Forecast start: {forecast_start}")
    print(f"Forecast hours: {forecast_hours}")
    print(f"Rows written: {len(output)}")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
