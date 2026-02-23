from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(r"C:\Users\Omen\Downloads\Data (1)")
YEAR_FOLDERS = ["Documents-2024-2025", "Documents-2025-2026"]

SIGNAL_FILES: Dict[str, str] = {
    "active_power_kw": "Active Power_data_export.csv",
    "daily_yield_kwh": "Daily Yield_data_export.csv",
    "inv1_ac_kw": "Inverter 1 AC Output_data_export.csv",
    "inv1_dc_kw": "Inverter 1 DC Inputs_data_export.csv",
    "inv1_eff": "Inverter 1 Efficiency_data_export.csv",
    "inv2_ac_kw": "Inverter 2 AC Output_data_export.csv",
    "inv2_dc_kw": "Inverter 2 DC Inputs_data_export.csv",
    "inv2_eff": "Inverter 2 Efficiency_data_export.csv",
    "inv3_ac_kw": "Inverter 3 AC Output_data_export.csv",
    "inv3_dc_kw": "Inverter 3 DC Inputs_data_export.csv",
    "inv3_eff": "Inverter 3 Efficiency_data_export.csv",
    "inv4_ac_kw": "Inverter 4 AC Output_data_export.csv",
    "inv4_dc_kw": "Inverter 4 DC Inputs_data_export.csv",
    "inv4_eff": "Inverter 4 Efficiency_data_export.csv",
    "inv5_ac_kw": "Inverter 5 AC Output_data_export.csv",
    "inv5_dc_kw": "Inverter 5 DC Inputs_data_export.csv",
    "inv5_eff": "Inverter 5 Efficiency_data_export.csv",
    "inv6_ac_kw": "Inverter 6 AC Output_data_export.csv",
    "inv6_dc_kw": "Inverter 6 DC Inputs_data_export.csv",
    "inv6_eff": "Inverter 6 Efficiency_data_export.csv",
}


def load_signal_csv(path: Path, value_col: str) -> pd.DataFrame:
    print(f"Loading {value_col} from {path}")
    df = pd.read_csv(path)
    date_col = df.columns[0]
    value_cols = df.columns[1:]
    df_long = df.melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="minute_offset",
        value_name=value_col,
    )
    df_long[date_col] = pd.to_datetime(
        df_long[date_col],
        format="%Y-%m-%d",
        errors="coerce",
    )
    df_long["minute_offset"] = (
        pd.to_numeric(df_long["minute_offset"], errors="coerce").fillna(0).astype(int)
    )
    df_long[value_col] = pd.to_numeric(df_long[value_col], errors="coerce")
    df_long["timestamp_local"] = df_long[date_col] + pd.to_timedelta(
        df_long["minute_offset"], unit="m"
    )
    df_long["timestamp"] = (
        df_long["timestamp_local"]
        .dt.tz_localize("Asia/Kolkata")
        .dt.tz_convert("UTC")
    )
    return df_long[["timestamp", value_col]].copy()


def normalize_signal_df(df_sig: pd.DataFrame) -> pd.DataFrame:
    df_sig = df_sig.copy()
    df_sig["timestamp"] = pd.to_datetime(df_sig["timestamp"], utc=True)
    df_sig = df_sig.dropna(subset=["timestamp"])
    df_sig = df_sig.groupby("timestamp", as_index=False).mean(numeric_only=True)
    return df_sig


def build_full_dataset() -> pd.DataFrame:
    merged_per_signal: Dict[str, pd.DataFrame] = {}
    for value_col, filename in SIGNAL_FILES.items():
        parts = []
        for folder in YEAR_FOLDERS:
            year_path = ROOT / folder / filename
            if year_path.exists():
                parts.append(load_signal_csv(year_path, value_col))
        if parts:
            df_cat = pd.concat(parts, ignore_index=True)
            merged_per_signal[value_col] = normalize_signal_df(df_cat)

    merged: pd.DataFrame | None = None
    for value_col, df_sig in merged_per_signal.items():
        if merged is None:
            merged = df_sig
        else:
            merged = merged.merge(df_sig, on="timestamp", how="outer")

    if merged is None:
        raise RuntimeError("No data found for any signal")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    if merged.shape[1] > 1:
        mask = merged.drop(columns=["timestamp"]).notna().any(axis=1)
        merged = merged.loc[mask].reset_index(drop=True)
    return merged


def main() -> None:
    df = build_full_dataset()
    df.to_csv("wiztric_real_timeseries_2024_2026.csv", index=False)
    if "active_power_kw" in df.columns:
        df_prophet = df[["timestamp", "active_power_kw"]].dropna().copy()
        df_prophet = df_prophet.rename(columns={"timestamp": "ds", "active_power_kw": "y"})
        df_prophet.to_csv("wiztric_prophet_power_5min.csv", index=False)


if __name__ == "__main__":
    main()
