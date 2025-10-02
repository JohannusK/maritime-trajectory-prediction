"""Utility for extracting documentation metrics from AIS datasets.

This script computes the statistics referenced in the manuscript's data
collection section, including message counts, unique vessel totals, channel
breakdowns, received-power dynamics, and FSPL alignment factors. It can be run
against either a Pandas-friendly table (Parquet/CSV/Feather) or an xarray Zarr
store that contains per-message AIS records.

Example
-------
python scripts/extract_dataset_metrics.py \
    --input data/processed/ais_positions.parquet \
    --site-lat 37.7749 --site-lon -122.4194 \
    --output dataset_metrics.json

The resulting JSON file includes ready-to-copy values for the documentation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

KM_PER_NAUTICAL_MILE = 1.852
EARTH_RADIUS_KM = 6371.0
DEFAULT_FREQ_MHZ = {"A": 161.975, "B": 162.025}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract descriptive metrics from processed AIS datasets."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to AIS position dataset (Parquet, Feather, CSV, or Zarr store).",
    )
    parser.add_argument(
        "--site-lat", type=float, required=True, help="Latitude of the receive site."
    )
    parser.add_argument(
        "--site-lon", type=float, required=True, help="Longitude of the receive site."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_metrics.json"),
        help="Where to write the metrics summary (JSON).",
    )
    parser.add_argument(
        "--power-column",
        default="rssi",
        help="Column name that stores received signal power in dB.",
    )
    parser.add_argument(
        "--freq-column",
        default="freq_offset_ppm",
        help="Column name that stores frequency offset in parts-per-million.",
    )
    parser.add_argument(
        "--channel-column",
        default="channel",
        help="Column containing channel identifiers (e.g., 'A' or 'B').",
    )
    parser.add_argument(
        "--msg-type-column",
        default="msg_type",
        help="Column that stores raw AIS message type IDs.",
    )
    parser.add_argument(
        "--lat-column",
        default=None,
        help="Column storing latitude values. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--lon-column",
        default=None,
        help="Column storing longitude values. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column name for message timestamps (ISO8601).",
    )
    parser.add_argument(
        "--mmsi-column",
        default="mmsi",
        help="Column name for vessel identifiers (MMSI).",
    )
    parser.add_argument(
        "--fspl-fit-min",
        type=float,
        default=10.0,
        help="Lower distance bound (km) for FSPL offset fitting.",
    )
    parser.add_argument(
        "--fspl-fit-max",
        type=float,
        default=40.0,
        help="Upper distance bound (km) for FSPL offset fitting.",
    )
    parser.add_argument(
        "--distance-bin-km",
        type=float,
        default=1.0,
        help="Distance bin size in kilometres for power summarisation.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Fraction of highest-power samples to average per distance bin.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".zarr" or path.is_dir():
        # Treat directories as potential Zarr stores.
        try:
            dataset = xr.open_zarr(path)
        except Exception as err:
            raise ValueError(f"Failed to open '{path}' as Zarr store: {err}") from err
        return dataset.to_dataframe().reset_index()
    raise ValueError(
        f"Unsupported input format '{suffix}'. Provide Parquet/Feather/CSV or a Zarr store."
    )


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Vectorised haversine distance in kilometres."""

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def auto_detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def summarise_message_types(df: pd.DataFrame, msg_type_col: str) -> Dict[str, int]:
    if msg_type_col not in df.columns:
        return {}
    counts = df[msg_type_col].dropna().astype(int).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def summarise_channels(df: pd.DataFrame, channel_col: str) -> Dict[str, int]:
    if channel_col not in df.columns:
        return {}
    counts = df[channel_col].fillna("Unknown").value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def compute_power_stats(df: pd.DataFrame, power_col: str) -> Optional[Dict[str, float]]:
    if power_col not in df.columns:
        return None
    valid = df[power_col].dropna()
    if valid.empty:
        return None
    power_min = float(valid.min())
    power_max = float(valid.max())
    return {
        "min_db": power_min,
        "max_db": power_max,
        "dynamic_range_db": power_max - power_min,
    }


def compute_power_distance_summary(
    df: pd.DataFrame,
    args: argparse.Namespace,
    distance_km: np.ndarray,
) -> Dict[str, Any]:
    power_col = args.power_column
    channel_col = args.channel_column

    if power_col not in df.columns:
        return {}

    data = df[[power_col]].copy()
    data["distance_km"] = distance_km

    if channel_col in df.columns:
        data["channel"] = df[channel_col].fillna("Unknown")
    else:
        data["channel"] = "Unknown"

    # Discard non-positive distances or missing power values.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data["distance_km"] > 0]
    if data.empty:
        return {}

    bin_size = args.distance_bin_km
    top_fraction = args.top_fraction

    def summarise_group(group: pd.DataFrame) -> List[Dict[str, float]]:
        result: List[Dict[str, float]] = []
        group = group.copy()
        group["distance_bin"] = np.floor(group["distance_km"] / bin_size) * bin_size

        for bin_value, bin_df in group.groupby("distance_bin"):
            quantile_threshold = bin_df[power_col].quantile(1.0 - top_fraction)
            top_slice = bin_df[bin_df[power_col] >= quantile_threshold]
            if top_slice.empty:
                continue
            result.append(
                {
                    "distance_km": float(bin_value + bin_size / 2.0),
                    "mean_power_db": float(top_slice[power_col].mean()),
                    "sample_count": int(len(top_slice)),
                }
            )
        return sorted(result, key=lambda item: item["distance_km"])

    summary: Dict[str, Any] = {}
    for channel, channel_df in data.groupby("channel"):
        summary[channel] = summarise_group(channel_df)

    summary["fspl_offsets_db"] = compute_fspl_offsets(summary, args)
    return summary


def compute_fspl_offsets(
    power_summary: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, float]:
    offsets: Dict[str, float] = {}
    for channel, entries in power_summary.items():
        if channel == "fspl_offsets_db" or not isinstance(entries, list):
            continue
        if not entries:
            continue

        freq_mhz = DEFAULT_FREQ_MHZ.get(channel, DEFAULT_FREQ_MHZ["A"])
        filtered = [
            row
            for row in entries
            if args.fspl_fit_min <= row["distance_km"] <= args.fspl_fit_max
        ]
        if not filtered:
            continue

        distances = np.array([row["distance_km"] for row in filtered])
        measured = np.array([row["mean_power_db"] for row in filtered])

        fspl = 32.44 + 20.0 * np.log10(distances) + 20.0 * np.log10(freq_mhz)
        # Received power ≈ constant - FSPL. The best-fit constant is the mean of (measured + FSPL).
        offsets[channel] = float(np.mean(measured + fspl))
    return offsets


def compute_frequency_metrics(
    df: pd.DataFrame, freq_col: str, distance_km: np.ndarray, timestamp_col: str, mmsi_col: str
) -> Dict[str, Any]:
    if freq_col not in df.columns:
        return {}
    freq_series = df[freq_col].replace([np.inf, -np.inf], np.nan).dropna()
    if freq_series.empty:
        return {}

    metrics: Dict[str, Any] = {}
    metrics["mean_ppm"] = float(freq_series.mean())
    metrics["std_ppm"] = float(freq_series.std())

    # Estimate quantisation step by examining consecutive unique values.
    unique_values = np.sort(freq_series.unique())
    if len(unique_values) > 1:
        diffs = np.diff(unique_values)
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            rounded = pd.Series(np.round(diffs, 3))
            most_common = rounded.mode().iloc[0]
            metrics["dominant_step_ppm"] = float(most_common)

    # Compute radial velocity versus frequency offset regression.
    if timestamp_col in df.columns and mmsi_col in df.columns:
        temp = df[[timestamp_col, mmsi_col, freq_col]].copy()
        temp["distance_km"] = distance_km
        temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
        if not temp.empty:
            temp[timestamp_col] = pd.to_datetime(temp[timestamp_col], utc=True, errors="coerce")
            temp = temp.dropna(subset=[timestamp_col])
            temp = temp.sort_values([mmsi_col, timestamp_col])

            radial_velocities: List[float] = []
            ppm_values: List[float] = []

            for _, vessel_df in temp.groupby(mmsi_col):
                if len(vessel_df) < 2:
                    continue
                dist = vessel_df["distance_km"].values
                times = (
                    vessel_df[timestamp_col]
                    .values.astype("datetime64[ns]")
                    .astype("int64")
                )
                deltas = np.diff(times) / 3.6e12  # convert ns to hours
                radial = np.diff(dist) / deltas
                valid = (deltas > 0) & np.isfinite(radial)
                if not np.any(valid):
                    continue
                radial = radial[valid]
                ppm = vessel_df[freq_col].values[1:][valid]
                if len(radial) == 0:
                    continue
                radial_velocities.extend(radial.tolist())
                ppm_values.extend(ppm.tolist())

            if radial_velocities:
                slope, intercept, rvalue, pvalue, stderr = linregress(
                    radial_velocities, ppm_values
                )
                metrics["radial_velocity_slope_ppm_per_kmh"] = float(slope)
                metrics["radial_velocity_p_value"] = float(pvalue)
                metrics["radial_velocity_r"] = float(rvalue)
                metrics["radial_velocity_stderr"] = float(stderr)
    return metrics


def compute_distance_stats(distance_km: np.ndarray) -> Dict[str, float]:
    if distance_km.size == 0:
        return {}
    return {
        "max_distance_km": float(distance_km.max()),
        "max_distance_nm": float(distance_km.max() / KM_PER_NAUTICAL_MILE),
        "median_distance_km": float(np.median(distance_km)),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path '{input_path}' does not exist")

    df = load_dataset(input_path)
    if df.empty:
        raise ValueError("Loaded dataset is empty; cannot compute metrics")

    lat_col = args.lat_column or auto_detect_column(df, ("latitude", "lat", "y"))
    lon_col = args.lon_column or auto_detect_column(df, ("longitude", "lon", "x"))
    if lat_col is None or lon_col is None:
        raise ValueError("Could not detect latitude/longitude columns")

    distance_km = haversine_distance(
        df[lat_col].to_numpy(), df[lon_col].to_numpy(), args.site_lat, args.site_lon
    )

    metrics: Dict[str, Any] = {}
    metrics["total_messages"] = int(len(df))

    if args.mmsi_column in df.columns:
        metrics["unique_vessels"] = int(df[args.mmsi_column].dropna().nunique())

    if args.timestamp_column in df.columns:
        timestamps = pd.to_datetime(df[args.timestamp_column], utc=True, errors="coerce")
        timestamps = timestamps.dropna()
        if not timestamps.empty:
            metrics["time_range"] = {
                "start": timestamps.min().isoformat(),
                "end": timestamps.max().isoformat(),
            }

    metrics["message_type_distribution"] = summarise_message_types(
        df, args.msg_type_column
    )
    metrics["channel_distribution"] = summarise_channels(df, args.channel_column)

    power_stats = compute_power_stats(df, args.power_column)
    if power_stats:
        metrics["power_stats_db"] = power_stats

    metrics["distance_stats"] = compute_distance_stats(distance_km)

    power_distance = compute_power_distance_summary(df, args, distance_km)
    if power_distance:
        metrics["power_vs_distance"] = power_distance

    freq_metrics = compute_frequency_metrics(
        df, args.freq_column, distance_km, args.timestamp_column, args.mmsi_column
    )
    if freq_metrics:
        metrics["frequency_offset_metrics"] = freq_metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
