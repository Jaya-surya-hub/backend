import pandas as pd

REQUIRED = [
    "timestamp",
    "active_power_kw",
    "irradiance_wm2",
    "temp_c",
    "humidity",
    "wind_mps",
    "rain_prob",
    "inv1_ac_kw",
    "inv1_dc_kw",
    "inv1_eff",
]

def main():
    inp = r"C:\Users\Omen\Documents\trae_projects\project1wiztric\backend\app\ml_train\solar_week_5min.csv"
    out = r"C:\Users\Omen\Documents\trae_projects\project1wiztric\backend\app\ml_train\baseline_training.csv"

    df = pd.read_csv(inp)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED].copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Convert numeric columns
    for c in REQUIRED:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean
    df = df.dropna(subset=["timestamp", "active_power_kw", "irradiance_wm2"])
    df = df.sort_values("timestamp")

    # Optional sanity filters (safe defaults)
    df = df[df["active_power_kw"] >= 0]
    df = df[(df["humidity"].isna()) | ((df["humidity"] >= 0) & (df["humidity"] <= 100))]
    df = df[(df["inv1_eff"].isna()) | ((df["inv1_eff"] >= 0) & (df["inv1_eff"] <= 1.2))]

    df.to_csv(out, index=False)
    print(f"✅ Exported: {out}")
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

if __name__ == "__main__":
    main()