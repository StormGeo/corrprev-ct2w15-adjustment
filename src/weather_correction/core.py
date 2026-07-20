from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
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

SUPPORTED_VARIABLES = {"2m_air_temperature", "total_precipitation", "weather_icon"}


class CorrectionError(ValueError):
    """Raised when input data cannot be aligned for correction."""


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


def get_year_julian(date_value: str | pd.Timestamp | datetime) -> tuple[str, str]:
    if isinstance(date_value, str):
        dt = datetime.strptime(date_value[:8], "%Y%m%d")
    else:
        dt = pd.Timestamp(date_value).to_pydatetime()
    return dt.strftime("%Y"), dt.strftime("%j")


def parse_forecast_start(
    value: str | None,
    timezone: str,
) -> pd.Timestamp:
    if value:
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp.now(tz=timezone).normalize()

    if ts.tzinfo is None:
        return ts.tz_localize(timezone)
    return ts.tz_convert(timezone)


def execution_timestamp(timezone: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=timezone)


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
        raise CorrectionError(
            f"Coordinates not found in NetCDF: {missing}. "
            f"Available coords: {list(da.coords)}. Available dims: {list(da.dims)}"
        )
    return found


def nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values)
    return int(np.abs(values - target).argmin())


def open_nc_as_dataarray(
    path: str | Path,
    variable: str,
) -> xr.DataArray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        da = xr.open_dataarray(path)
        if da.name == variable or da.name is None:
            return da
    except Exception:
        pass

    ds = xr.open_dataset(path)
    if variable in ds.data_vars:
        return ds[variable]

    if len(ds.data_vars) == 1:
        fallback = list(ds.data_vars)[0]
        print(
            f"Variable {variable!r} was not found in {path}; using only available "
            f"data variable {fallback!r}."
        )
        return ds[fallback]

    raise CorrectionError(
        f"Variable {variable!r} not found in {path}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def stations_table(stations_cfg: dict) -> pd.DataFrame:
    records = []
    for station_key, station in stations_cfg["stations"].items():
        records.append(
            {
                "station_key": station_key,
                "city_key": normalize_city_name(station["city"]),
                "city_name": station["city"],
                "city_id": int(station["id"]),
                "lat": float(station["lat"]),
                "lon": float(station["lon"]),
                "station_order": len(records),
            }
        )
    return pd.DataFrame.from_records(records)


def build_time_table(
    da: xr.DataArray,
    timezone: str,
    forecast_start: pd.Timestamp,
    forecast_hours: int,
    coord_names: dict,
) -> pd.DataFrame:
    time_name = coord_names["time"]
    time_utc = pd.to_datetime(da[time_name].values, utc=True)
    time_local = time_utc.tz_convert(timezone)
    end_local = forecast_start + pd.Timedelta(hours=int(forecast_hours))

    table = pd.DataFrame(
        {
            "time_index": np.arange(len(time_utc)),
            "time_utc": time_utc,
            "time_local": time_local,
            "local_date": time_local.date,
            "hour_local": time_local.hour,
        }
    )
    selected = table[
        (table["time_local"] >= forecast_start)
        & (table["time_local"] <= end_local)
    ].copy()

    if selected.empty:
        raise CorrectionError(
            f"No NetCDF timestamps found between {forecast_start} and {end_local}."
        )

    selected["corr_date"] = selected["local_date"]
    return selected.reset_index(drop=True)


def build_hourly_template_from_stations(
    stations_cfg: dict,
    forecast_start: pd.Timestamp,
    forecast_hours: int,
) -> pd.DataFrame:
    end_local = forecast_start + pd.Timedelta(hours=int(forecast_hours))
    time_local = pd.date_range(start=forecast_start, end=end_local, freq="h")
    if len(time_local) == 0:
        raise CorrectionError(
            f"No timestamps generated between {forecast_start} and {end_local}."
        )

    period_table = pd.DataFrame(
        {
            "time_local": time_local,
            "time_utc": time_local.tz_convert("UTC"),
            "local_date": time_local.date,
            "hour_local": time_local.hour,
        }
    )
    period_table["corr_date"] = period_table["local_date"]

    station_df = stations_table(stations_cfg)
    records = []
    for station in station_df.to_dict("records"):
        tmp = period_table.copy()
        tmp["station_key"] = station["station_key"]
        tmp["city_key"] = station["city_key"]
        tmp["city_name"] = station["city_name"]
        tmp["city_id"] = station["city_id"]
        tmp["lat"] = station["lat"]
        tmp["lon"] = station["lon"]
        tmp["station_order"] = station["station_order"]
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


def extract_variable_at_stations(
    nc_path: str | Path,
    nc_variable: str,
    stations_cfg: dict,
    forecast_start: pd.Timestamp,
    forecast_hours: int,
    timezone: str,
    output_col: str = "model_value",
) -> pd.DataFrame:
    da = open_nc_as_dataarray(nc_path, nc_variable)
    coord_names = infer_coord_names(da)
    period_table = build_time_table(
        da=da,
        timezone=timezone,
        forecast_start=forecast_start,
        forecast_hours=forecast_hours,
        coord_names=coord_names,
    )

    time_name = coord_names["time"]
    lat_name = coord_names["lat"]
    lon_name = coord_names["lon"]
    lat_values = da[lat_name].values
    lon_values = da[lon_name].values
    time_indices = period_table["time_index"].to_numpy()
    station_df = stations_table(stations_cfg)
    records = []

    for station in station_df.to_dict("records"):
        lat_idx = nearest_index(lat_values, station["lat"])
        lon_idx = nearest_index(lon_values, station["lon"])
        point_da = da.isel(
            {time_name: time_indices, lat_name: lat_idx, lon_name: lon_idx}
        )

        tmp = period_table.copy()
        tmp["station_key"] = station["station_key"]
        tmp["city_key"] = station["city_key"]
        tmp["city_name"] = station["city_name"]
        tmp["city_id"] = station["city_id"]
        tmp["lat"] = station["lat"]
        tmp["lon"] = station["lon"]
        tmp["station_order"] = station["station_order"]
        tmp["grid_lat"] = float(lat_values[lat_idx])
        tmp["grid_lon"] = float(lon_values[lon_idx])
        tmp["lat_idx"] = lat_idx
        tmp["lon_idx"] = lon_idx
        tmp[output_col] = point_da.values.astype(float)
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


def read_meteorologist_json(path: str | Path) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []

    for location_name, location in raw["locations"].items():
        city_key = normalize_city_name(location_name)
        for item in location["data"]:
            date = pd.to_datetime(item["date"])
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


def validate_json_cities(
    met_daily: pd.DataFrame,
    stations_cfg: dict,
    missing_city_policy: str,
) -> dict[str, list[str]]:
    station_keys = set(stations_table(stations_cfg)["city_key"])
    json_keys = set(met_daily["city_key"].unique())

    extra_json = sorted(json_keys - station_keys)
    valid_json = sorted(json_keys & station_keys)
    if not valid_json:
        raise CorrectionError(
            "JSON does not contain any cities configured in stations.yaml. "
            "Please contact support."
        )

    missing_json = sorted(station_keys - json_keys)
    if missing_json and missing_city_policy == "error":
        raise CorrectionError(
            "stations.yaml contains cities missing from JSON: "
            + ", ".join(missing_json)
        )

    return {
        "extra_json": extra_json,
        "missing_json": missing_json,
        "valid_json": valid_json,
    }


def filter_to_json_cities(
    hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
    missing_city_policy: str,
) -> pd.DataFrame:
    if missing_city_policy == "skip":
        json_city_keys = set(met_daily["city_key"].unique())
        return hourly[hourly["city_key"].isin(json_city_keys)].copy()
    return hourly.copy()


def mark_trailing_midnight(
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
        raise CorrectionError(f"Unmapped icon: {icon!r} -> token {token!r}")
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
    raise CorrectionError(f"Invalid hour: {hour}")


def merge_with_meteorologist(
    hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
    required_cols: list[str],
    missing_city_policy: str,
) -> pd.DataFrame:
    df = filter_to_json_cities(hourly, met_daily, missing_city_policy)
    df = mark_trailing_midnight(df, met_daily)
    keys = ["city_key", "corr_date"]
    df = df.merge(
        met_daily[["city_key", "corr_date", *required_cols]],
        on=keys,
        how="left",
        indicator=True,
    )

    missing = df["_merge"].ne("both")
    if missing.any():
        if missing_city_policy == "skip":
            df = df.loc[~missing].copy()
        else:
            report = (
                df.loc[missing]
                .groupby(["city_key", "corr_date"], as_index=False)
                .agg(first_time=("time_local", "min"), n_rows=("time_local", "size"))
            )
            raise CorrectionError(
                "Hourly rows without matching JSON data were found:\n"
                + report.to_string(index=False)
            )
    return df.drop(columns=["_merge"])


def correct_temperature(
    hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
    cfg: dict,
    missing_city_policy: str,
) -> pd.DataFrame:
    df = merge_with_meteorologist(
        hourly=hourly,
        met_daily=met_daily,
        required_cols=["tmin_met", "tmax_met"],
        missing_city_policy=missing_city_policy,
    )

    if df.empty:
        return df.assign(corrected_value=pd.Series(dtype=float))

    if cfg["temperature"].get("round_model_before_correction", True):
        decimals = int(cfg["temperature"].get("model_round_decimals", 1))
        df["model_for_correction"] = df["model_value"].round(decimals)
    else:
        df["model_for_correction"] = df["model_value"]

    keys = ["city_key", "corr_date"]
    stats = (
        df.groupby(keys, as_index=False)
        .agg(
            tmin_model=("model_for_correction", "min"),
            tmax_model=("model_for_correction", "max"),
        )
    )
    df = df.merge(stats, on=keys, how="left")

    span_model = df["tmax_model"] - df["tmin_model"]
    span_met = df["tmax_met"] - df["tmin_met"]
    norm = np.where(
        span_model.abs() > 1e-9,
        (df["model_for_correction"] - df["tmin_model"]) / span_model,
        0.5,
    )
    raw = df["tmin_met"] + norm * span_met

    rounding = cfg["temperature"].get("output_rounding", "python")
    decimals = int(cfg["temperature"].get("output_round_decimals", 1))
    if rounding == "python":
        df["corrected_value"] = raw.round(decimals)
    elif rounding == "none":
        df["corrected_value"] = raw
    else:
        raise CorrectionError("Use temperature.output_rounding as 'python' or 'none'.")
    return df


def apply_json_symbols(
    hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
    missing_city_policy: str,
) -> pd.DataFrame:
    df = merge_with_meteorologist(
        hourly=hourly,
        met_daily=met_daily,
        required_cols=["dawn_icon", "morning_icon", "afternoon_icon", "night_icon"],
        missing_city_policy=missing_city_policy,
    )

    if df.empty:
        return df.assign(corrected_value=pd.Series(dtype=float))

    df["period_name"] = df.apply(
        lambda row: period_from_hour(row["hour_local"], bool(row["is_trailing_midnight"])),
        axis=1,
    )
    df["icon_json_raw"] = df.apply(
        lambda row: row[PERIOD_TO_JSON_COLUMN[row["period_name"]]],
        axis=1,
    )
    df["corrected_value"] = df["icon_json_raw"].apply(json_icon_to_final_symbol)
    return df


def correct_precipitation(
    precipitation_hourly: pd.DataFrame,
    cape_hourly: pd.DataFrame,
    met_daily: pd.DataFrame,
    cfg: dict,
    missing_city_policy: str,
) -> pd.DataFrame:
    merge_keys = [
        "station_key",
        "city_key",
        "city_id",
        "time_utc",
        "time_local",
        "local_date",
        "hour_local",
        "station_order",
        "lat",
        "lon",
    ]
    cape_keep = merge_keys + ["model_value"]
    df = precipitation_hourly.merge(
        cape_hourly[cape_keep].rename(columns={"model_value": "cape_model"}),
        on=merge_keys,
        how="inner",
    ).rename(columns={"model_value": "precip_model"})

    df = merge_with_meteorologist(
        hourly=df,
        met_daily=met_daily,
        required_cols=[
            "rain_met",
            "dawn_icon",
            "morning_icon",
            "afternoon_icon",
            "night_icon",
        ],
        missing_city_policy=missing_city_policy,
    )

    if df.empty:
        return df.assign(corrected_value=pd.Series(dtype=float))

    df["period_name"] = df.apply(
        lambda row: period_from_hour(row["hour_local"], bool(row["is_trailing_midnight"])),
        axis=1,
    )
    df["icon_json_raw"] = df.apply(
        lambda row: row[PERIOD_TO_JSON_COLUMN[row["period_name"]]],
        axis=1,
    )
    df["symbol_final"] = df["icon_json_raw"].apply(json_icon_to_final_symbol)
    df["is_rainy_symbol"] = df["symbol_final"].isin(RAINY_SYMBOLS)
    df["met_window"] = 0
    keys = ["city_key", "corr_date"]
    df["rain_mask"] = df["is_rainy_symbol"] & ~df["is_trailing_midnight"]

    rain_cfg = RAIN_ICON_CONFIG
    alpha = rain_cfg["alpha"]
    cape_u = (
        np.maximum(df["cape_model"] - 100, 0) ** rain_cfg["cape_exponent"]
    ) / rain_cfg["cape_divisor"]
    df["icon_factor"] = df["symbol_final"].map(rain_cfg["icon_factor"]).fillna(1.0)
    df["afternoon_factor"] = np.where(
        (
            df["symbol_final"].eq(10)
            & df["hour_local"].between(
                rain_cfg["afternoon_start_hour"],
                rain_cfg["afternoon_end_hour"],
            )
        ),
        rain_cfg["afternoon_factor"],
        1.0,
    )
    df["rain_score"] = np.where(
        df["rain_mask"],
        alpha
        + (1 - alpha)
        * df["icon_factor"]
        * df["afternoon_factor"]
        * (df["precip_model"].round(1) + cape_u),
        0.0,
    )
    df["rain_score_sum_daily"] = df.groupby(keys)["rain_score"].transform("sum")
    raw = np.where(
        df["rain_mask"] & df["rain_score_sum_daily"].gt(0),
        df["rain_met"] * df["rain_score"] / df["rain_score_sum_daily"],
        0.0,
    )

    rounding = cfg.get("rain", {}).get("output_rounding", "python")
    decimals = int(cfg.get("rain", {}).get("output_round_decimals", 1))
    if rounding == "python":
        df["corrected_value"] = pd.Series(raw, index=df.index).round(decimals)
    elif rounding == "none":
        df["corrected_value"] = raw
    else:
        raise CorrectionError("Use rain.output_rounding as 'python' or 'none'.")
    return df


def default_cape_path(precipitation_path: str | Path) -> Path:
    path_text = str(precipitation_path)
    if "total_precipitation" not in path_text:
        raise CorrectionError(
            "--cape-nc is required because total_precipitation was not found in --input-nc."
        )
    return Path(path_text.replace("total_precipitation", "cape_index"))


def build_output_path(
    output_dir: str | Path,
    variable: str,
    run_timestamp: pd.Timestamp,
) -> Path:
    
    if run_timestamp.tzinfo is None:
        run_timestamp = run_timestamp.tz_localize("UTC")
    else:
        run_timestamp = run_timestamp.tz_convert("UTC")
        
    year = run_timestamp.strftime("%Y")
    julian = run_timestamp.strftime("%j")
    run_id = run_timestamp.strftime("%Y%m%d%H")
    return (
        Path(output_dir)
        / variable
        / year
        / julian
        / f"meteorologist_{variable}_M000_{run_id}.csv"
    )


def build_operational_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "data", "city", "lat", "lon"])

    out = df[["time_local", "corrected_value", "city_id", "lat", "lon", "station_order"]].copy()
    out["date"] = pd.to_datetime(out["time_local"]).dt.strftime("%d/%m/%Y %H:%M")
    out = out.rename(columns={"corrected_value": "data", "city_id": "city"})
    out = out.sort_values(["station_order", "date"])
    return out[["date", "data", "city", "lat", "lon"]]


def run_single_variable_correction(
    nc_variable: str,
    input_nc: str | Path | None,
    met_json: str | Path,
    stations_cfg: dict,
    cfg: dict,
    forecast_start: pd.Timestamp,
    forecast_hours: int,
    missing_city_policy: str = "skip",
    cape_nc: str | Path | None = None,
) -> pd.DataFrame:
    if nc_variable not in SUPPORTED_VARIABLES:
        raise CorrectionError(
            f"Unsupported --nc-variable {nc_variable!r}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_VARIABLES))}."
        )

    met_daily = read_meteorologist_json(met_json)
    city_validation = validate_json_cities(met_daily, stations_cfg, missing_city_policy)
    if city_validation["extra_json"]:
        print(
            "Warning: JSON contains cities not registered in stations.yaml and they will be skipped: "
            + ", ".join(city_validation["extra_json"])
            + ". Please contact support."
        )
    timezone = cfg["run"].get("timezone", "America/Sao_Paulo")

    if nc_variable == "weather_icon":
        base_hourly = build_hourly_template_from_stations(
            stations_cfg=stations_cfg,
            forecast_start=forecast_start,
            forecast_hours=forecast_hours,
        )
        corrected = apply_json_symbols(base_hourly, met_daily, missing_city_policy)
    else:
        if not input_nc:
            raise CorrectionError(f"--input-nc is required for {nc_variable}.")

        base_hourly = extract_variable_at_stations(
            nc_path=input_nc,
            nc_variable=nc_variable,
            stations_cfg=stations_cfg,
            forecast_start=forecast_start,
            forecast_hours=forecast_hours,
            timezone=timezone,
            output_col="model_value",
        )

        if nc_variable == "2m_air_temperature":
            corrected = correct_temperature(base_hourly, met_daily, cfg, missing_city_policy)
        elif nc_variable == "total_precipitation":
            cape_path = Path(cape_nc) if cape_nc else default_cape_path(input_nc)
            cape_hourly = extract_variable_at_stations(
                nc_path=cape_path,
                nc_variable="cape_index",
                stations_cfg=stations_cfg,
                forecast_start=forecast_start,
                forecast_hours=forecast_hours,
                timezone=timezone,
                output_col="model_value",
            )
            corrected = correct_precipitation(
                precipitation_hourly=base_hourly,
                cape_hourly=cape_hourly,
                met_daily=met_daily,
                cfg=cfg,
                missing_city_policy=missing_city_policy,
            )
        else:
            raise AssertionError("unreachable")

    return build_operational_output(corrected)
