import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# ==============================
# COLUMN STANDARDIZATION
# ==============================
def standardize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    rename_map = {
        "linear acceleration x (m/s^2)": "Linear Acceleration x (m/s^2)",
        "linear acceleration y (m/s^2)": "Linear Acceleration y (m/s^2)",
        "linear acceleration z (m/s^2)": "Linear Acceleration z (m/s^2)",
        "gyroscope x (rad/s)": "Gyroscope x (rad/s)",
        "gyroscope y (rad/s)": "Gyroscope y (rad/s)",
        "gyroscope z (rad/s)": "Gyroscope z (rad/s)",
        "time (s)": "Time (s)",
        "time": "Time (s)",
        "timestamp": "Time (s)",
        "t": "Time (s)"
    }

    return df.rename(columns=rename_map)


# ==============================
# CLEANING
# ==============================
def clean_sensor_data(df):
    df = df.copy()

    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    df = standardize_columns(df)

    required_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Gyroscope x (rad/s)",
        "Gyroscope y (rad/s)",
        "Gyroscope z (rad/s)"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    if "Time (s)" not in df.columns:
        df["Time (s)"] = np.arange(len(df)) * 0.02

    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
    df["Time (s)"] = df["Time (s)"].interpolate(limit_direction="both")
    df["Time (s)"] = df["Time (s)"].ffill().bfill()

    if df["Time (s)"].isna().any():
        fallback_time = pd.Series(np.arange(len(df)) * 0.02, index=df.index)
        df["Time (s)"] = df["Time (s)"].fillna(fallback_time)

    df = df.sort_values("Time (s)").reset_index(drop=True)

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(limit_direction="both")

    df[required_cols] = df[required_cols].fillna(0)

    # time delta
    df["dt"] = df["Time (s)"].diff().fillna(0)
    df["dt"] = df["dt"].clip(0.001, 1)

    # estimate sampling rate
    df["fs"] = 1 / df["dt"].replace(0, np.nan)
    df["fs"] = df["fs"].fillna(df["fs"].median())
    df["fs"] = df["fs"].fillna(50.0)

    return df


# ==============================
# FEATURE ENGINEERING
# ==============================
def create_features(df):
    df = df.copy()

    # magnitude features
    df["acc_mag"] = np.sqrt(
        df["Linear Acceleration x (m/s^2)"] ** 2 +
        df["Linear Acceleration y (m/s^2)"] ** 2 +
        df["Linear Acceleration z (m/s^2)"] ** 2
    )

    df["gyro_mag"] = np.sqrt(
        df["Gyroscope x (rad/s)"] ** 2 +
        df["Gyroscope y (rad/s)"] ** 2 +
        df["Gyroscope z (rad/s)"] ** 2
    )

    # jerk
    df["jerk"] = df["acc_mag"].diff().fillna(0) / df["dt"]

    # time-based windows
    def seconds_to_window(sec):
        median_dt = df["dt"].median()
        if pd.isna(median_dt) or median_dt <= 0:
            median_dt = 0.02
        return max(1, int(sec / median_dt))

    short_w = seconds_to_window(0.5)
    mid_w = seconds_to_window(2)
    long_w = seconds_to_window(5)

    # rolling stats
    df["acc_mean"] = df["acc_mag"].rolling(mid_w, min_periods=1).mean()
    df["acc_std"] = df["acc_mag"].rolling(mid_w, min_periods=1).std().fillna(0)

    df["gyro_mean"] = df["gyro_mag"].rolling(mid_w, min_periods=1).mean()
    df["gyro_std"] = df["gyro_mag"].rolling(mid_w, min_periods=1).std().fillna(0)

    df["jerk_mean"] = df["jerk"].rolling(short_w, min_periods=1).mean()

    # motion proxy
    df["motion_energy"] = (
        0.5 * df["acc_mean"] +
        0.3 * df["gyro_mean"] +
        0.2 * df["jerk_mean"].abs()
    )

    # trend features
    df["energy_short"] = df["motion_energy"].rolling(short_w, min_periods=1).mean()
    df["energy_long"] = df["motion_energy"].rolling(long_w, min_periods=1).mean()
    df["energy_trend"] = df["energy_short"] - df["energy_long"]

    # cumulative load
    df["load"] = df["motion_energy"] * df["dt"]
    df["cumulative_load"] = df["load"].rolling(long_w, min_periods=1).sum()

    # instability
    df["instability"] = (
        0.4 * df["acc_std"] +
        0.3 * df["gyro_std"] +
        0.3 * df["jerk_mean"].abs()
    )

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ==============================
# DYNAMIC THRESHOLDS
# ==============================
def compute_thresholds(df):
    thresholds = {
        "low_motion": df["motion_energy"].quantile(0.25),
        "medium_motion": df["motion_energy"].quantile(0.6),
        "high_motion": df["motion_energy"].quantile(0.85),
        "fatigue_low": df["cumulative_load"].quantile(0.3),
        "fatigue_high": df["cumulative_load"].quantile(0.7)
    }
    return thresholds


# ==============================
# LABEL GENERATION
# ==============================
def generate_labels(df):
    df = df.copy()
    th = compute_thresholds(df)

    # activity
    df["activity"] = np.select(
        [
            df["motion_energy"] < th["low_motion"],
            df["gyro_mean"] > df["acc_mean"],
            df["motion_energy"] < th["medium_motion"],
            df["motion_energy"] >= th["medium_motion"]
        ],
        ["stationary", "rotation", "walking", "running"],
        default="walking"
    )

    # intensity
    df["intensity"] = np.select(
        [
            df["motion_energy"] < th["low_motion"],
            df["motion_energy"] < th["high_motion"],
            df["motion_energy"] >= th["high_motion"]
        ],
        ["low", "medium", "high"],
        default="low"
    )

    # fatigue
    fatigue_score = (
        0.4 * df["cumulative_load"] +
        0.3 * df["instability"] +
        0.3 * (-df["energy_trend"])
    )

    df["fatigue"] = np.select(
        [
            fatigue_score < th["fatigue_low"],
            fatigue_score < th["fatigue_high"],
            fatigue_score >= th["fatigue_high"]
        ],
        ["fresh", "moderate", "fatigued"],
        default="fresh"
    )

    return df


# ==============================
# FEATURES
# ==============================
def get_feature_columns():
    return [
        "acc_mag",
        "gyro_mag",
        "jerk",
        "acc_mean",
        "acc_std",
        "gyro_mean",
        "gyro_std",
        "jerk_mean",
        "motion_energy",
        "energy_short",
        "energy_long",
        "energy_trend",
        "cumulative_load",
        "instability"
    ]


# ==============================
# TARGETS
# ==============================
def get_target_columns():
    return ["activity", "intensity", "fatigue"]


# ==============================
# MODEL
# ==============================
def get_model():
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return MultiOutputClassifier(base)