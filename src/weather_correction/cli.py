from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from weather_correction.core import (
    load_meteorologist_daily_for_run,
    load_yaml,
    run_rain_correction,
    run_temperature_correction,
    update_meteorologist_netcdfs,
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Use true/false, 1/0, or yes/no.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run operational temperature and precipitation/icon corrections."
    )
    parser.add_argument(
        "--date-rod",
        dest="date_rod",
        required=True,
        help="Run date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--run-hour",
        "--rod",
        dest="run_hour",
        required=True,
        help="Run hour, for example 00.",
    )
    parser.add_argument(
        "--json-dir",
        default=None,
        help="Directory containing meteorologist JSON files. Defaults to paths.json_dir from YAML.",
    )
    parser.add_argument(
        "--save-validation",
        "--save-checks",
        dest="save_validation",
        type=str_to_bool,
        default=False,
        help="Save validation CSV files. Default: false.",
    )
    parser.add_argument(
        "--config",
        "--paths-yaml",
        dest="paths_yaml",
        default="configs/paths.yaml",
        help="YAML file with paths and processing parameters.",
    )
    parser.add_argument(
        "--stations",
        "--stations-yaml",
        dest="stations_yaml",
        default="configs/stations.yaml",
        help="YAML file with station metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override paths.output_dir from YAML.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["csv", "netcdf", "both"],
        default="csv",
        help="Write CSVs only, update NetCDFs only, or do both. Default: csv.",
    )
    parser.add_argument(
        "--meteorologist-base",
        default=None,
        help="Override paths.meteorologist_base from YAML.",
    )
    parser.add_argument(
        "--nc-temp-var",
        default=None,
        help="Temperature variable name inside the target NetCDF.",
    )
    parser.add_argument(
        "--nc-temp-file-var",
        default=None,
        help="Variable token used in the temperature NetCDF path/file name.",
    )
    parser.add_argument(
        "--nc-precip-var",
        default=None,
        help="Precipitation variable name inside the target NetCDF.",
    )
    parser.add_argument(
        "--nc-precip-file-var",
        default=None,
        help="Variable token used in the precipitation NetCDF path/file name.",
    )
    parser.add_argument(
        "--nc-icon-var",
        default=None,
        help="Icon variable name inside the target NetCDF.",
    )
    parser.add_argument(
        "--nc-icon-file-var",
        default=None,
        help="Variable token used in the icon NetCDF path/file name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = deepcopy(load_yaml(args.paths_yaml))
    stations_cfg = load_yaml(args.stations_yaml)

    cfg["run"]["date_rod"] = args.date_rod
    cfg["run"]["rod"] = args.run_hour
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.meteorologist_base:
        cfg["paths"]["meteorologist_base"] = args.meteorologist_base

    json_dir = args.json_dir or cfg.get("paths", {}).get("json_dir")
    if not json_dir:
        raise ValueError("Pass --json-dir or configure paths.json_dir in the YAML file.")

    met_daily, json_path, shift_days = load_meteorologist_daily_for_run(
        cfg=cfg,
        stations_cfg=stations_cfg,
        json_dir=json_dir,
    )
    temperature_output, temperature_validation = run_temperature_correction(
        cfg=cfg,
        stations_cfg=stations_cfg,
        met_daily=met_daily,
    )
    rain_output, rain_validation = run_rain_correction(
        cfg=cfg,
        stations_cfg=stations_cfg,
        met_daily=met_daily,
    )

    run_id = f"{args.date_rod}{args.run_hour}"

    print(f"JSON used: {json_path}")
    print(f"date_shift_days used: {shift_days}")

    if args.output_mode in {"csv", "both"}:
        output_dir = Path(cfg["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        temperature_csv = output_dir / f"temperature_corrected_{run_id}.csv"
        rain_csv = output_dir / f"icon_rain_corrected_{run_id}.csv"
        temperature_output.to_csv(temperature_csv, index=False)
        rain_output.to_csv(rain_csv, index=False)

        print(f"Temperature saved to: {temperature_csv}")
        print(f"Precipitation/icons saved to: {rain_csv}")

        if args.save_validation:
            temperature_validation_csv = output_dir / f"temperature_validation_{run_id}.csv"
            rain_validation_csv = output_dir / f"rain_validation_{run_id}.csv"
            temperature_validation.to_csv(temperature_validation_csv, index=False)
            rain_validation.to_csv(rain_validation_csv, index=False)
            print(f"Temperature validation saved to: {temperature_validation_csv}")
            print(f"Rain validation saved to: {rain_validation_csv}")

    if args.output_mode in {"netcdf", "both"}:
        netcdf_cfg = cfg.get("netcdf_update", {})
        temperature_variable = args.nc_temp_var or netcdf_cfg.get("temperature_variable")
        precipitation_variable = args.nc_precip_var or netcdf_cfg.get("precipitation_variable")
        icon_variable = args.nc_icon_var or netcdf_cfg.get("icon_variable")
        temperature_file_variable = (
            args.nc_temp_file_var
            or netcdf_cfg.get("temperature_file_variable")
            or temperature_variable
        )
        precipitation_file_variable = (
            args.nc_precip_file_var
            or netcdf_cfg.get("precipitation_file_variable")
            or precipitation_variable
        )
        icon_file_variable = (
            args.nc_icon_file_var
            or netcdf_cfg.get("icon_file_variable")
            or icon_variable
        )

        missing = [
            name for name, value in [
                ("--nc-temp-var", temperature_variable),
                ("--nc-precip-var", precipitation_variable),
                ("--nc-icon-var", icon_variable),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                "Provide NetCDF variable names via CLI or netcdf_update in YAML: "
                + ", ".join(missing)
            )

        results = update_meteorologist_netcdfs(
            cfg=cfg,
            stations_cfg=stations_cfg,
            temperature_output=temperature_output,
            rain_output=rain_output,
            temperature_variable=temperature_variable,
            precipitation_variable=precipitation_variable,
            icon_variable=icon_variable,
            temperature_file_variable=temperature_file_variable,
            precipitation_file_variable=precipitation_file_variable,
            icon_file_variable=icon_file_variable,
            meteorologist_base=cfg["paths"].get("meteorologist_base"),
        )
        for result in results:
            if result.get("skipped"):
                print(
                    "NetCDF skipped: "
                    f"{result['path']} | variable={result['variable']} | "
                    f"reason={result['error']}"
                )
            else:
                print(
                    "NetCDF updated: "
                    f"{result['path']} | variable={result['variable']} | "
                    f"column={result['value_col']} | points={result['updates']}"
                )


if __name__ == "__main__":
    main()
