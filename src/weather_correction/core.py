from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import yaml


RAIN_ICON_CONFIG = {
    "alpha": 0.15,
    "cape_exponent": 0.8,
    "cape_divisor": 2000.0,
    "afternoon_start_hour": 14,
    "afternoon_end_hour": 19,
    "icon_factor": {
        7: 0.8,
        8: 1.0,
        9: 1.25,
        10: 1.1,
        11: 1.15,
    },
    "afternoon_factor": 1.25,
}

RAINY_SYMBOLS = {7, 8, 9, 10, 11}

JSON_ICON_TO_FINAL_SYMBOL = {
    "7": 1,
    "1": 2,
    "2": 3,
    "2r": 4,
    "9": 5,
    "3": 7,
    "4": 8,
    "4r": 8,
    "5": 9,
    "4t": 10,
    "6": 11,
}

PERIOD_TO_JSON_COLUMN = {
    "dawn": "dawn_icon",
    "morning": "morning_icon",
    "afternoon": "afternoon_icon",
    "night": "night_icon",
}

AUX_NC_VARIABLES = {
    "total_precipitation": {
        "output_col": "prec01_model_nc",
        "required": True,
        "round_decimals": 1,
    },
    "cape_index": {
        "output_col": "cape_model_nc",
        "required": True,
        "round_decimals": 1,
    },
    "weather_icon_smoothed": {
        "output_col": "symbol_model_nc",
        "required": False,
        "round_decimals": None,
    },
}


def get_model_input_variables(cfg: dict) -> dict:
    model_inputs = cfg.get("model_inputs", {})
    return {
        "temperature": model_inputs.get("temperature_variable", "2m_air_temperature"),
        "precipitation": model_inputs.get("precipitation_variable", "total_precipitation"),
        "cape": model_inputs.get("cape_variable", "cape_index"),
        "icon": model_inputs.get("icon_variable", "weather_icon_smoothed"),
    }


def get_aux_nc_variables(cfg: dict) -> dict:
    variables = get_model_input_variables(cfg)
    return {
        variables["precipitation"]: {
            "output_col": "prec01_model_nc",
            "required": True,
            "round_decimals": 1,
        },
        variables["cape"]: {
            "output_col": "cape_model_nc",
            "required": True,
            "round_decimals": 1,
        },
        variables["icon"]: {
            "output_col": "symbol_model_nc",
            "required": False,
            "round_decimals": None,
        },
    }


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_city_name(name: str) -> str:
    name = str(name)
    name = name.replace("Prev ", "")
    name = name.split(" - ")[0]
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return name.strip()


def get_year_julian(date_rod: str) -> tuple[str, str]:
    dt = datetime.strptime(date_rod, "%Y%m%d")
    return dt.strftime("%Y"), dt.strftime("%j")


def build_nc_path(cfg: dict, variable: str) -> Path:
    date_rod = str(cfg["run"]["date_rod"])
    rod = str(cfg["run"]["rod"])
    year, julian = get_year_julian(date_rod)
    base = cfg["paths"]["chimera_as_base"]
    template = cfg.get("files", {}).get(
        "nc_template",
        "{base}/{variable}/{year}/{julian}/chimera_as_{variable}_M000_{date_rod}{rod}.nc",
    )
    return Path(
        template.format(
            base=base,
            variable=variable,
            year=year,
            julian=julian,
            date_rod=date_rod,
            rod=rod,
        )
    )


def open_nc_as_dataarray(
    path: str | Path,
    preferred_variable: str | None = None,
) -> xr.DataArray:
    path = Path(path)
    try:
        return xr.open_dataarray(path)
    except Exception:
        ds = xr.open_dataset(path)

    if preferred_variable and preferred_variable in ds.data_vars:
        return ds[preferred_variable]
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]

    raise ValueError(
        "The NetCDF file has more than one variable. "
        f"Pass preferred_variable. Variables found: {list(ds.data_vars)}"
    )


def infer_coord_names(da: xr.DataArray) -> dict:
    candidates = {
        "time": ["time", "valid_time"],
        "lat": ["latitude", "lat", "y"],
        "lon": ["longitude", "lon", "x"],
    }
    found = {}

    for key, options in candidates.items():
        for option in options:
            if option in da.coords or option in da.dims:
                found[key] = option
                break

    missing = [key for key in candidates if key not in found]
    if missing:
        raise ValueError(
            f"Coordinates not found in NetCDF: {missing}. "
            f"Available coords: {list(da.coords)}. Available dims: {list(da.dims)}"
        )
    return found


def nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values)
    return int(np.abs(values - target).argmin())


def build_time_table(
    da: xr.DataArray,
    cfg: dict,
    coord_names: dict,
) -> pd.DataFrame:
    time_name = coord_names["time"]
    tz_name = cfg["run"]["timezone"]
    time_utc = pd.to_datetime(da[time_name].values, utc=True)
    time_local = time_utc.tz_convert(tz_name)
    return pd.DataFrame(
        {
            "time_index": np.arange(len(time_utc)),
            "time_utc": time_utc,
            "time_local": time_local,
            "local_date": time_local.date,
            "hour_local": time_local.hour,
        }
    )


def filter_forecast_period(
    time_table: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    date_rod = str(cfg["run"]["date_rod"])
    tz_name = cfg["run"]["timezone"]
    start_day_offset = int(cfg["processing"]["start_day_offset"])
    forecast_hours = int(cfg["processing"]["forecast_hours"])

    start_local = (
        pd.Timestamp(datetime.strptime(date_rod, "%Y%m%d"))
        .tz_localize(tz_name)
        + pd.Timedelta(days=start_day_offset)
    )
    end_local = start_local + pd.Timedelta(hours=forecast_hours)

    selected = time_table[
        (time_table["time_local"] >= start_local)
        & (time_table["time_local"] <= end_local)
    ].copy()
    selected["corr_date"] = selected["local_date"]
    return selected.reset_index(drop=True)


def read_meteorologist_json(
    path: str | Path,
    date_shift_days: int = 0,
) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []

    for location_name, location in raw["locations"].items():
        city_key = normalize_city_name(location_name)
        for item in location["data"]:
            date = pd.to_datetime(item["date"]) + pd.Timedelta(days=date_shift_days)
            records.append(
                {
                    "city_json": location_name,
                    "city_key": city_key,
                    "corr_date": date.date(),
                    "tmin_met": float(item["tmin"]),
                    "tmax_met": float(item["tmax"]),
                    "rain_met": float(item["rain"]),
                    "dawn_icon": item.get("dawn"),
                    "morning_icon": item.get("morning"),
                    "afternoon_icon": item.get("afternoon"),
                    "night_icon": item.get("night"),
                    "inverted_minimum": item.get("inverted_minimum") or "",
                }
            )
    return pd.DataFrame.from_records(records)


def find_meteorologist_json(
    json_dir: str | Path,
    date_rod: str,
    start_day_offset: int = 1,
    json_template: str | None = None,
) -> Path:
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise NotADirectoryError(json_dir)

    target_date = (
        datetime.strptime(str(date_rod), "%Y%m%d")
        + pd.Timedelta(days=int(start_day_offset))
    ).strftime("%Y%m%d")
    year, julian = get_year_julian(target_date)

    search_patterns = []
    if json_template:
        search_patterns.append(
            json_template.format(
                base=json_dir,
                year=year,
                julian=julian,
                date=target_date,
            )
        )

    # Backward-compatible fallbacks for local tests and older flat folders.
    search_patterns.extend(
        [
            str(json_dir / "weather" / "exports" / year / julian / "*.json"),
            str(json_dir / "*.json"),
        ]
    )

    candidates = []
    for pattern in search_patterns:
        matches = sorted(Path().glob(pattern) if not Path(pattern).is_absolute() else Path(pattern).parent.glob(Path(pattern).name))
        candidates.extend(path for path in matches if target_date in path.name)
        if candidates:
            break

    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = "\n".join(str(path) for path in candidates)
        raise ValueError(f"More than one JSON file found for {target_date}:\n{names}")

    searched = "\n".join(search_patterns)
    raise FileNotFoundError(
        f"No JSON file containing date {target_date} was found. Searched patterns:\n{searched}"
    )


def infer_date_shift_days(
    met_daily_raw: pd.DataFrame,
    model_points: pd.DataFrame,
) -> int:
    first_met_date = pd.to_datetime(met_daily_raw["corr_date"].min())
    first_model_date = pd.to_datetime(model_points["local_date"].min())
    return int((first_model_date - first_met_date).days)


def extract_temperature_points_from_nc(
    da: xr.DataArray,
    stations_cfg: dict,
    period_table: pd.DataFrame,
    cfg: dict,
    coord_names: dict,
) -> pd.DataFrame:
    time_name = coord_names["time"]
    lat_name = coord_names["lat"]
    lon_name = coord_names["lon"]
    lat_values = da[lat_name].values
    lon_values = da[lon_name].values
    time_indices = period_table["time_index"].to_numpy()
    records = []

    for station_key, station in stations_cfg["stations"].items():
        lat = float(station["lat"])
        lon = float(station["lon"])
        lat_idx = nearest_index(lat_values, lat)
        lon_idx = nearest_index(lon_values, lon)
        point_da = da.isel(
            {time_name: time_indices, lat_name: lat_idx, lon_name: lon_idx}
        )

        tmp = period_table.copy()
        tmp["station_key"] = station_key
        tmp["station_name"] = station.get("name", station["city"])
        tmp["city"] = station["city"]
        tmp["city_key"] = tmp["city"].apply(normalize_city_name)
        tmp["station_lat"] = lat
        tmp["station_lon"] = lon
        tmp["grid_lat"] = float(lat_values[lat_idx])
        tmp["grid_lon"] = float(lon_values[lon_idx])
        tmp["lat_idx"] = lat_idx
        tmp["lon_idx"] = lon_idx
        tmp["temperature_model_nc"] = point_da.values.astype(float)
        records.append(tmp)

    df = pd.concat(records, ignore_index=True)
    if cfg["temperature"]["round_model_before_correction"]:
        decimals = int(cfg["temperature"]["model_round_decimals"])
        df["temperature_model_for_correction"] = df["temperature_model_nc"].round(
            decimals
        )
    else:
        df["temperature_model_for_correction"] = df["temperature_model_nc"]
    return df


def extract_temperature_model_points_from_nc(
    cfg: dict,
    stations_cfg: dict,
) -> pd.DataFrame:
    variable = get_model_input_variables(cfg)["temperature"]
    nc_path = build_nc_path(cfg, variable)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    da = open_nc_as_dataarray(nc_path, preferred_variable=variable)
    coord_names = infer_coord_names(da)
    time_table = build_time_table(da, cfg, coord_names)
    period_table = filter_forecast_period(time_table, cfg)
    return extract_temperature_points_from_nc(
        da=da,
        stations_cfg=stations_cfg,
        period_table=period_table,
        cfg=cfg,
        coord_names=coord_names,
    )


def attach_trailing_midnight_to_previous_day_from_met(
    model_points: pd.DataFrame,
    met_daily: pd.DataFrame,
) -> pd.DataFrame:
    df = model_points.copy()
    df["is_trailing_midnight"] = False
    met_index = pd.MultiIndex.from_frame(
        met_daily[["city_key", "corr_date"]].drop_duplicates()
    )
    current_index = pd.MultiIndex.from_frame(df[["city_key", "corr_date"]])
    has_current_met = current_index.isin(met_index)
    previous_date = (pd.to_datetime(df["corr_date"]) - pd.Timedelta(days=1)).dt.date
    previous_index = pd.MultiIndex.from_arrays([df["city_key"], previous_date])
    has_previous_met = previous_index.isin(met_index)
    mask = df["hour_local"].eq(0) & ~has_current_met & has_previous_met
    df.loc[mask, "corr_date"] = previous_date[mask]
    df.loc[mask, "is_trailing_midnight"] = True
    return df


def compute_model_daily_minmax(model_points: pd.DataFrame) -> pd.DataFrame:
    keys = ["city_key", "corr_date"]
    return (
        model_points
        .groupby(keys, as_index=False)
        .agg(
            tmin_model=("temperature_model_for_correction", "min"),
            tmax_model=("temperature_model_for_correction", "max"),
            n_hours=("temperature_model_for_correction", "size"),
        )
    )


def apply_temperature_correction_from_daily_values(
    model_points: pd.DataFrame,
    met_daily: pd.DataFrame,
    model_daily_stats: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    keys = ["city_key", "corr_date"]
    df = (
        model_points
        .merge(model_daily_stats, on=keys, how="left")
        .merge(met_daily, on=keys, how="left", indicator=True)
    )

    missing = df["_merge"].ne("both")
    if missing.any():
        missing_report = (
            df.loc[missing]
            .groupby(["city", "city_key", "corr_date"], as_index=False)
            .agg(
                first_time=("time_local", "min"),
                last_time=("time_local", "max"),
                n_rows=("time_local", "size"),
            )
        )
        raise ValueError(
            "Hourly rows without a matching JSON record were found.\n"
            f"{missing_report.to_string(index=False)}"
        )

    span_model = df["tmax_model"] - df["tmin_model"]
    span_met = df["tmax_met"] - df["tmin_met"]
    norm = np.where(
        span_model.abs() > 1e-9,
        (df["temperature_model_for_correction"] - df["tmin_model"]) / span_model,
        0.5,
    )
    df["temperature_corrected_raw"] = df["tmin_met"] + norm * span_met

    rounding = cfg["temperature"]["output_rounding"]
    decimals = int(cfg["temperature"]["output_round_decimals"])
    if rounding == "python":
        df["temperature_corrected"] = df["temperature_corrected_raw"].round(decimals)
    elif rounding == "none":
        df["temperature_corrected"] = df["temperature_corrected_raw"]
    else:
        raise ValueError("Use output_rounding: 'python' or 'none'.")

    df["temperature_delta"] = (
        df["temperature_corrected"] - df["temperature_model_for_correction"]
    )
    return df.drop(columns=["_merge"])


def validate_temperature_result(df: pd.DataFrame) -> pd.DataFrame:
    validation = (
        df
        .groupby(["city", "city_key", "corr_date"], as_index=False)
        .agg(
            n_hours=("temperature_corrected", "size"),
            min_corr=("temperature_corrected", "min"),
            max_corr=("temperature_corrected", "max"),
            min_corr_raw=("temperature_corrected_raw", "min"),
            max_corr_raw=("temperature_corrected_raw", "max"),
            tmin_met=("tmin_met", "first"),
            tmax_met=("tmax_met", "first"),
            tmin_model=("tmin_model", "first"),
            tmax_model=("tmax_model", "first"),
        )
    )
    validation["min_error"] = validation["min_corr"] - validation["tmin_met"]
    validation["max_error"] = validation["max_corr"] - validation["tmax_met"]
    validation["raw_min_error"] = validation["min_corr_raw"] - validation["tmin_met"]
    validation["raw_max_error"] = validation["max_corr_raw"] - validation["tmax_met"]
    return validation


def normalize_json_icon_token(icon) -> str | None:
    if icon is None or pd.isna(icon):
        return None
    token = str(icon).strip().lower()
    try:
        value = float(token)
        if value.is_integer():
            token = str(int(value))
    except ValueError:
        pass
    if token.endswith("n"):
        token = token[:-1]
    return token


def json_icon_to_final_symbol(icon) -> int:
    token = normalize_json_icon_token(icon)
    if token not in JSON_ICON_TO_FINAL_SYMBOL:
        raise ValueError(f"Unmapped icon: {icon!r} -> token {token!r}")
    return JSON_ICON_TO_FINAL_SYMBOL[token]


def period_from_hour(hour: int, is_trailing_midnight: bool = False) -> str:
    hour = int(hour)
    if is_trailing_midnight:
        return "night"
    if 0 <= hour <= 5:
        return "dawn"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 23:
        return "night"
    raise ValueError(f"Invalid hour: {hour}")


def extract_nc_variable_at_stations(
    variable: str,
    output_col: str,
    cfg: dict,
    stations_cfg: dict,
    round_decimals: int | None = None,
) -> pd.DataFrame:
    nc_path = build_nc_path(cfg, variable)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    da = open_nc_as_dataarray(nc_path, preferred_variable=variable)
    coord_names = infer_coord_names(da)
    period_table = filter_forecast_period(build_time_table(da, cfg, coord_names), cfg)
    time_name = coord_names["time"]
    lat_name = coord_names["lat"]
    lon_name = coord_names["lon"]
    lat_values = da[lat_name].values
    lon_values = da[lon_name].values
    time_indices = period_table["time_index"].to_numpy()
    records = []

    for station_key, station in stations_cfg["stations"].items():
        lat = float(station["lat"])
        lon = float(station["lon"])
        lat_idx = nearest_index(lat_values, lat)
        lon_idx = nearest_index(lon_values, lon)
        point_da = da.isel(
            {time_name: time_indices, lat_name: lat_idx, lon_name: lon_idx}
        )

        tmp = period_table.copy()
        tmp = tmp.rename(columns={"time_index": f"time_index_{output_col}"})
        tmp["station_key"] = station_key
        tmp["station_name"] = station.get("name", station["city"])
        tmp["city"] = station["city"]
        tmp["city_key"] = tmp["city"].apply(normalize_city_name)
        tmp[f"lat_idx_{output_col}"] = lat_idx
        tmp[f"lon_idx_{output_col}"] = lon_idx
        tmp[f"grid_lat_{output_col}"] = float(lat_values[lat_idx])
        tmp[f"grid_lon_{output_col}"] = float(lon_values[lon_idx])

        values = point_da.values.astype(float)
        if output_col == "symbol_model_nc":
            tmp[output_col] = values.astype(int)
        else:
            tmp[output_col] = values
            if round_decimals is not None:
                tmp[output_col] = tmp[output_col].round(round_decimals)
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


def extract_rain_icon_model_inputs_from_nc(
    cfg: dict,
    stations_cfg: dict,
) -> pd.DataFrame:
    dfs = []
    skipped = []
    merge_keys = [
        "station_key",
        "station_name",
        "city",
        "city_key",
        "time_utc",
        "time_local",
        "local_date",
        "hour_local",
    ]

    for variable, info in get_aux_nc_variables(cfg).items():
        output_col = info["output_col"]
        try:
            df_var = extract_nc_variable_at_stations(
                variable=variable,
                output_col=output_col,
                cfg=cfg,
                stations_cfg=stations_cfg,
                round_decimals=info["round_decimals"],
            )
        except FileNotFoundError as exc:
            if info["required"]:
                raise
            skipped.append((variable, str(exc)))
            continue

        keep_cols = merge_keys + [
            col for col in df_var.columns
            if col.startswith("time_index_")
            or col.startswith("lat_idx_")
            or col.startswith("lon_idx_")
            or col.startswith("grid_lat_")
            or col.startswith("grid_lon_")
            or col == output_col
        ]
        dfs.append(df_var[keep_cols])

    if not dfs:
        raise ValueError("No auxiliary NetCDF variable was extracted.")

    df = dfs[0]
    for other in dfs[1:]:
        df = df.merge(other, on=merge_keys, how="inner")

    required_cols = ["prec01_model_nc", "cape_model_nc"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df["prec01_model_for_correction"] = pd.to_numeric(
        df["prec01_model_nc"],
        errors="coerce",
    ).round(1)
    df["cape_model_for_correction"] = pd.to_numeric(
        df["cape_model_nc"],
        errors="coerce",
    ).round(1)
    df["corr_date"] = df["local_date"]
    df["is_trailing_midnight"] = False

    if skipped:
        for variable, path in skipped:
            print(f"Optional variable not found: {variable}: {path}")
    return df


def mark_trailing_midnight_for_icon_rain(
    hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
) -> pd.DataFrame:
    df = hourly.copy()
    df["corr_date"] = df["local_date"]
    df["is_trailing_midnight"] = False
    met_index = pd.MultiIndex.from_frame(
        met_daily[["city_key", "corr_date"]].drop_duplicates()
    )
    current_index = pd.MultiIndex.from_frame(df[["city_key", "corr_date"]])
    has_current_met = current_index.isin(met_index)
    previous_date = (pd.to_datetime(df["local_date"]) - pd.Timedelta(days=1)).dt.date
    previous_index = pd.MultiIndex.from_arrays([df["city_key"], previous_date])
    has_previous_met = previous_index.isin(met_index)
    mask = df["hour_local"].eq(0) & ~has_current_met & has_previous_met
    df.loc[mask, "corr_date"] = previous_date[mask]
    df.loc[mask, "is_trailing_midnight"] = True
    return df


def apply_icon_rain_correction_from_json(
    model_inputs: pd.DataFrame,
    met_daily: pd.DataFrame,
    rain_icon_config: dict = RAIN_ICON_CONFIG,
    output_rounding: str = "python",
    output_round_decimals: int = 1,
) -> pd.DataFrame:
    keys = ["city_key", "corr_date"]
    df = model_inputs.copy()
    if "met_window" not in df.columns:
        df["met_window"] = 0

    df = df.merge(
        met_daily[
            [
                "city_key",
                "corr_date",
                "rain_met",
                "dawn_icon",
                "morning_icon",
                "afternoon_icon",
                "night_icon",
            ]
        ],
        on=keys,
        how="left",
        indicator=True,
    )
    missing = df["_merge"].ne("both")
    if missing.any():
        missing_report = (
            df.loc[missing]
            .groupby(["city", "city_key", "corr_date"], as_index=False)
            .agg(
                first_time=("time_local", "min"),
                last_time=("time_local", "max"),
                n_rows=("time_local", "size"),
            )
        )
        raise ValueError(
            "Hourly rows without a matching JSON record were found.\n"
            f"{missing_report.to_string(index=False)}"
        )
    df = df.drop(columns=["_merge"])

    df["period_name"] = df.apply(
        lambda row: period_from_hour(
            row["hour_local"],
            bool(row["is_trailing_midnight"]),
        ),
        axis=1,
    )
    df["icon_json_raw"] = df.apply(
        lambda row: row[PERIOD_TO_JSON_COLUMN[row["period_name"]]],
        axis=1,
    )
    df["icon_json_normalized"] = df["icon_json_raw"].apply(normalize_json_icon_token)
    df["symbol_corrected"] = df["icon_json_raw"].apply(json_icon_to_final_symbol)
    df["is_rainy_symbol"] = df["symbol_corrected"].isin(RAINY_SYMBOLS)
    df["has_met_window_daily"] = (
        df.groupby(keys)["met_window"].transform(lambda s: s.eq(1).any())
    )
    df["mask_meteorologist"] = np.where(
        df["has_met_window_daily"],
        df["met_window"].eq(1),
        True,
    )
    df["rain_mask"] = (
        df["is_rainy_symbol"]
        & df["mask_meteorologist"]
        & ~df["is_trailing_midnight"]
    )

    alpha = rain_icon_config["alpha"]
    cape_u = (
        np.maximum(df["cape_model_for_correction"] - 100, 0)
        ** rain_icon_config["cape_exponent"]
    ) / rain_icon_config["cape_divisor"]
    df["cape_u"] = cape_u
    df["icon_factor"] = df["symbol_corrected"].map(
        rain_icon_config["icon_factor"]
    ).fillna(1.0)
    df["afternoon_factor"] = np.where(
        (
            df["symbol_corrected"].eq(10)
            & df["hour_local"].between(
                rain_icon_config["afternoon_start_hour"],
                rain_icon_config["afternoon_end_hour"],
            )
        ),
        rain_icon_config["afternoon_factor"],
        1.0,
    )
    df["rain_score"] = np.where(
        df["rain_mask"],
        alpha
        + (1 - alpha)
        * df["icon_factor"]
        * df["afternoon_factor"]
        * (df["prec01_model_for_correction"] + df["cape_u"]),
        0.0,
    )
    df["rain_score_sum_daily"] = df.groupby(keys)["rain_score"].transform("sum")
    df["rain_corrected_raw"] = np.where(
        df["rain_mask"] & df["rain_score_sum_daily"].gt(0),
        df["rain_met"] * df["rain_score"] / df["rain_score_sum_daily"],
        0.0,
    )

    if output_rounding == "python":
        df["rain_corrected"] = df["rain_corrected_raw"].round(output_round_decimals)
    elif output_rounding == "none":
        df["rain_corrected"] = df["rain_corrected_raw"]
    else:
        raise ValueError("Use output_rounding='python' or output_rounding='none'.")

    df["rain_corrected_text"] = df["rain_corrected"].map(
        lambda value: f"{float(value):.{output_round_decimals}f}"
    )
    return df


def validate_icon_rain_result(df: pd.DataFrame) -> pd.DataFrame:
    validation = (
        df
        .groupby(["city", "city_key", "corr_date"], as_index=False)
        .agg(
            n_hours=("time_local", "size"),
            n_trailing=("is_trailing_midnight", "sum"),
            n_rainy_symbols=("is_rainy_symbol", "sum"),
            n_rain_distribution_hours=("rain_mask", "sum"),
            rain_met=("rain_met", "first"),
            rain_sum_raw=("rain_corrected_raw", "sum"),
            rain_sum_rounded=("rain_corrected", "sum"),
            rain_score_sum_daily=("rain_score", "sum"),
            first_time=("time_local", "min"),
            last_time=("time_local", "max"),
        )
    )
    validation["rain_raw_error"] = validation["rain_sum_raw"] - validation["rain_met"]
    validation["rain_rounded_error"] = (
        validation["rain_sum_rounded"] - validation["rain_met"]
    )
    validation["status"] = np.where(
        validation["rain_met"].eq(0),
        "no_meteorologist_rain",
        np.where(
            validation["n_rain_distribution_hours"].eq(0),
            "meteorologist_rain_without_rainy_icon",
            "ok",
        ),
    )
    return validation


def prepare_temperature_output(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "time_local",
            "city",
            "city_key",
            "temperature_model_nc",
            "temperature_model_for_correction",
            "temperature_corrected_raw",
            "temperature_corrected",
            "temperature_delta",
            "tmin_model",
            "tmax_model",
            "tmin_met",
            "tmax_met",
            "is_trailing_midnight",
        ]
    ].copy()


def prepare_rain_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "time_local",
            "city",
            "city_key",
            "symbol_corrected",
            "rain_corrected",
            "rain_corrected_text",
            "rain_corrected_raw",
            "icon_json_raw",
            "period_name",
            "prec01_model_for_correction",
            "cape_model_for_correction",
            "rain_met",
            "rain_score",
            "rain_mask",
            "is_trailing_midnight",
        ]
    ].copy()
    return out.rename(
        columns={
            "symbol_corrected": "symbol_final",
            "rain_corrected": "prec01_final",
            "rain_corrected_text": "prec01_final_text",
        }
    )


def load_meteorologist_daily_for_run(
    cfg: dict,
    stations_cfg: dict,
    json_dir: str | Path,
) -> tuple[pd.DataFrame, Path, int]:
    json_path = find_meteorologist_json(
        json_dir=json_dir,
        date_rod=str(cfg["run"]["date_rod"]),
        start_day_offset=int(cfg["processing"]["start_day_offset"]),
        json_template=cfg.get("files", {}).get("json_template"),
    )
    met_daily_raw = read_meteorologist_json(json_path, date_shift_days=0)

    if cfg["processing"]["infer_date_shift_days"]:
        model_points = extract_temperature_model_points_from_nc(cfg, stations_cfg)
        shift_days = infer_date_shift_days(met_daily_raw, model_points)
    else:
        shift_days = int(cfg["processing"]["date_shift_days"])

    met_daily = read_meteorologist_json(json_path, date_shift_days=shift_days)
    return met_daily, json_path, shift_days


def get_meteorologist_target_date(cfg: dict) -> str:
    date_rod = str(cfg["run"]["date_rod"])
    start_day_offset = int(cfg["processing"]["start_day_offset"])
    return (
        datetime.strptime(date_rod, "%Y%m%d")
        + pd.Timedelta(days=start_day_offset)
    ).strftime("%Y%m%d")


def build_meteorologist_nc_path(
    cfg: dict,
    variable: str,
    base: str | Path | None = None,
) -> Path:
    target_date = get_meteorologist_target_date(cfg)
    rod = str(cfg["run"]["rod"])
    year, julian = get_year_julian(target_date)
    nc_base = str(base or cfg["paths"]["meteorologist_base"])
    template = cfg.get("files", {}).get(
        "meteorologist_nc_template",
        "{base}/{variable}/{year}/{julian}/meteorologist_{variable}_M000_{date_rod}{rod}.nc",
    )
    path = Path(
        template.format(
            base=nc_base,
            variable=variable,
            year=year,
            julian=julian,
            date_rod=target_date,
            rod=rod,
        )
    )
    if path.exists():
        return path

    fallback = Path(
        "{base}/{variable}/{julian}/meteorologist_{variable}_M000_{date_rod}{rod}.nc".format(
            base=nc_base,
            variable=variable,
            julian=julian,
            date_rod=target_date,
            rod=rod,
        )
    )
    if fallback.exists():
        return fallback

    return path


def _local_naive_times(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values)
    if getattr(times.dt, "tz", None) is not None:
        return times.dt.tz_localize(None)
    return times


def _build_station_lookup(stations_cfg: dict) -> dict:
    lookup = {}
    for station_key, station in stations_cfg["stations"].items():
        city_key = normalize_city_name(station["city"])
        lookup[city_key] = {
            "station_key": station_key,
            "lat": float(station["lat"]),
            "lon": float(station["lon"]),
        }
    return lookup


def update_netcdf_points_from_dataframe(
    nc_path: str | Path,
    variable_name: str,
    corrected_df: pd.DataFrame,
    value_col: str,
    stations_cfg: dict,
    dry_run: bool = False,
) -> dict:
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)
    if value_col not in corrected_df.columns:
        raise ValueError(f"Coluna ausente no DataFrame corrigido: {value_col}")

    df = corrected_df.copy()
    df["datetime_local"] = _local_naive_times(df["time_local"])
    df[value_col] = pd.to_numeric(df[value_col], errors="raise")

    station_lookup = _build_station_lookup(stations_cfg)
    missing_stations = sorted(set(df["city_key"]) - set(station_lookup))
    if missing_stations:
        raise ValueError(f"Cities missing from stations YAML: {missing_stations}")

    updates = 0
    station_points = {}

    with xr.open_dataset(nc_path) as meta_ds:
        if variable_name not in meta_ds.variables:
            raise ValueError(
                f"Variable {variable_name!r} does not exist em {nc_path}. "
                f"Available variables: {list(meta_ds.variables)}"
            )

        coord_names = infer_coord_names(meta_ds[variable_name])
        var_dims = meta_ds[variable_name].dims
        dim_sizes = dict(meta_ds.sizes)
        time_values = pd.to_datetime(meta_ds[coord_names["time"]].values)
        time_index = {pd.Timestamp(value): idx for idx, value in enumerate(time_values)}
        lat_values = np.asarray(meta_ds[coord_names["lat"]].values, dtype=float)
        lon_values = np.asarray(meta_ds[coord_names["lon"]].values, dtype=float)

        missing_times = sorted(
            set(pd.Timestamp(value) for value in df["datetime_local"])
            - set(time_index)
        )
        if missing_times:
            sample = ", ".join(str(value) for value in missing_times[:10])
            raise ValueError(f"Missing timestamps in {nc_path}: {sample}")

        for city_key, station in station_lookup.items():
            lat_idx = nearest_index(lat_values, station["lat"])
            lon_idx = nearest_index(lon_values, station["lon"])
            station_points[city_key] = {
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "grid_lat": float(lat_values[lat_idx]),
                "grid_lon": float(lon_values[lon_idx]),
            }

    if dry_run:
        return {
            "path": str(nc_path),
            "variable": variable_name,
            "value_col": value_col,
            "updates": len(df),
            "station_points": station_points,
        }

    try:
        h5_file = h5py.File(nc_path, "r+")
    except OSError as exc:
        raise OSError(
            f"Could not open {nc_path} em modo escrita. "
            "Check that the file is not open in another program and that the directory has write permission."
        ) from exc

    with h5_file as ds:
        if variable_name not in ds:
            raise ValueError(
                f"Variable {variable_name!r} does not exist no HDF5 {nc_path}. "
                f"Available variables: {list(ds.keys())}"
            )

        nc_var = ds[variable_name]
        for _, row in df.iterrows():
            city_key = row["city_key"]
            point = station_points[city_key]
            t_idx = time_index[pd.Timestamp(row["datetime_local"])]
            indexer = []

            for dim_name in var_dims:
                if dim_name == coord_names["time"]:
                    indexer.append(t_idx)
                elif dim_name == coord_names["lat"]:
                    indexer.append(point["lat_idx"])
                elif dim_name == coord_names["lon"]:
                    indexer.append(point["lon_idx"])
                elif dim_sizes[dim_name] == 1:
                    indexer.append(0)
                else:
                    raise ValueError(
                        f"Unsupported dimension em {variable_name}: {dim_name}"
                    )

            nc_var[tuple(indexer)] = row[value_col]
            updates += 1

    return {
        "path": str(nc_path),
        "variable": variable_name,
        "value_col": value_col,
        "updates": updates,
        "station_points": station_points,
    }


def update_meteorologist_netcdfs(
    cfg: dict,
    stations_cfg: dict,
    temperature_output: pd.DataFrame,
    rain_output: pd.DataFrame,
    temperature_variable: str,
    precipitation_variable: str,
    icon_variable: str,
    temperature_file_variable: str | None = None,
    precipitation_file_variable: str | None = None,
    icon_file_variable: str | None = None,
    meteorologist_base: str | Path | None = None,
    continue_on_icon_error: bool = True,
) -> list[dict]:
    targets = [
        {
            "file_variable": temperature_file_variable or temperature_variable,
            "variable": temperature_variable,
            "df": temperature_output,
            "value_col": "temperature_corrected",
            "kind": "temperature",
            "required": True,
        },
        {
            "file_variable": precipitation_file_variable or precipitation_variable,
            "variable": precipitation_variable,
            "df": rain_output,
            "value_col": "prec01_final_text",
            "kind": "precipitation",
            "required": True,
        },
        {
            "file_variable": icon_file_variable or icon_variable,
            "variable": icon_variable,
            "df": rain_output,
            "value_col": "symbol_final",
            "kind": "icon",
            "required": not continue_on_icon_error,
        },
    ]

    required_targets = [target for target in targets if target["required"]]
    optional_targets = [target for target in targets if not target["required"]]

    for target in required_targets:
        nc_path = build_meteorologist_nc_path(
            cfg=cfg,
            variable=target["file_variable"],
            base=meteorologist_base,
        )
        update_netcdf_points_from_dataframe(
            nc_path=nc_path,
            variable_name=target["variable"],
            corrected_df=target["df"],
            value_col=target["value_col"],
            stations_cfg=stations_cfg,
            dry_run=True,
        )

    results = []
    for target in required_targets:
        nc_path = build_meteorologist_nc_path(
            cfg=cfg,
            variable=target["file_variable"],
            base=meteorologist_base,
        )
        results.append(
            update_netcdf_points_from_dataframe(
                nc_path=nc_path,
                variable_name=target["variable"],
                corrected_df=target["df"],
                value_col=target["value_col"],
                stations_cfg=stations_cfg,
            )
        )

    for target in optional_targets:
        nc_path = build_meteorologist_nc_path(
            cfg=cfg,
            variable=target["file_variable"],
            base=meteorologist_base,
        )
        try:
            update_netcdf_points_from_dataframe(
                nc_path=nc_path,
                variable_name=target["variable"],
                corrected_df=target["df"],
                value_col=target["value_col"],
                stations_cfg=stations_cfg,
                dry_run=True,
            )
            results.append(
                update_netcdf_points_from_dataframe(
                    nc_path=nc_path,
                    variable_name=target["variable"],
                    corrected_df=target["df"],
                    value_col=target["value_col"],
                    stations_cfg=stations_cfg,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "path": str(nc_path),
                    "variable": target["variable"],
                    "value_col": target["value_col"],
                    "updates": 0,
                    "skipped": True,
                    "error": str(exc),
                }
            )
    return results


def run_temperature_correction(
    cfg: dict,
    stations_cfg: dict,
    met_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_points = extract_temperature_model_points_from_nc(cfg, stations_cfg)
    model_points = attach_trailing_midnight_to_previous_day_from_met(
        model_points,
        met_daily,
    )
    model_daily_stats = compute_model_daily_minmax(model_points)
    corrected = apply_temperature_correction_from_daily_values(
        model_points=model_points,
        met_daily=met_daily,
        model_daily_stats=model_daily_stats,
        cfg=cfg,
    )
    return prepare_temperature_output(corrected), validate_temperature_result(corrected)


def run_rain_correction(
    cfg: dict,
    stations_cfg: dict,
    met_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_inputs = extract_rain_icon_model_inputs_from_nc(cfg, stations_cfg)
    model_inputs = mark_trailing_midnight_for_icon_rain(model_inputs, met_daily)
    corrected = apply_icon_rain_correction_from_json(
        model_inputs=model_inputs,
        met_daily=met_daily,
        rain_icon_config=RAIN_ICON_CONFIG,
        output_rounding="python",
        output_round_decimals=1,
    )
    return prepare_rain_output(corrected), validate_icon_rain_result(corrected)
