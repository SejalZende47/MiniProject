import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        "t": "Time (s)",
        "activity": "activity",
        "intensity": "intensity",
        "fatigue": "fatigue"
    }

    return df.rename(columns=rename_map)


# ==============================
# DATA CLEANING
# ==============================
def clean_sensor_data(df):
    df = df.copy()

    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    df = standardize_columns(df)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    numeric_candidates = [
        "Time (s)",
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Gyroscope x (rad/s)",
        "Gyroscope y (rad/s)",
        "Gyroscope z (rad/s)"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_sensor_cols = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Gyroscope x (rad/s)",
        "Gyroscope y (rad/s)",
        "Gyroscope z (rad/s)"
    ]

    for col in required_sensor_cols:
        if col not in df.columns:
            df[col] = 0.0

    if "Time (s)" not in df.columns:
        df["Time (s)"] = np.arange(len(df), dtype=float) * 0.02  # fallback ~50 Hz

    df = df.sort_values("Time (s)").reset_index(drop=True)

    sensor_present = [c for c in required_sensor_cols if c in df.columns]
    if sensor_present:
        df = df.dropna(subset=sensor_present, how="all")

    for col in required_sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df[required_sensor_cols] = df[required_sensor_cols].fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    for col in required_sensor_cols:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)

    # Time delta
    df["dt"] = df["Time (s)"].diff().fillna(0)
    df["dt"] = df["dt"].replace([np.inf, -np.inf], 0)
    df["dt"] = df["dt"].clip(lower=0.0, upper=1.0)

    # If repeated / broken timestamps happen
    median_dt = df["dt"][df["dt"] > 0].median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 0.02
    df.loc[df["dt"] <= 0, "dt"] = median_dt

    return df.reset_index(drop=True)


# ==============================
# FEATURE ENGINEERING
# ==============================
def create_features(df):
    df = df.copy()

    # Raw magnitudes
    df["acc_mag"] = np.sqrt(
        df["Linear Acceleration x (m/s^2)"]**2 +
        df["Linear Acceleration y (m/s^2)"]**2 +
        df["Linear Acceleration z (m/s^2)"]**2
    )

    df["gyro_mag"] = np.sqrt(
        df["Gyroscope x (rad/s)"]**2 +
        df["Gyroscope y (rad/s)"]**2 +
        df["Gyroscope z (rad/s)"]**2
    )

    # Smoothed acceleration to reduce integration drift
    smooth_window = 5
    df["acc_x_smooth"] = df["Linear Acceleration x (m/s^2)"].rolling(smooth_window, min_periods=1).mean()
    df["acc_y_smooth"] = df["Linear Acceleration y (m/s^2)"].rolling(smooth_window, min_periods=1).mean()
    df["acc_z_smooth"] = df["Linear Acceleration z (m/s^2)"].rolling(smooth_window, min_periods=1).mean()

    # Velocity estimation by integration
    df["vel_x"] = (df["acc_x_smooth"] * df["dt"]).cumsum()
    df["vel_y"] = (df["acc_y_smooth"] * df["dt"]).cumsum()
    df["vel_z"] = (df["acc_z_smooth"] * df["dt"]).cumsum()

    # Drift control: rolling mean subtraction
    drift_window = 25
    df["vel_x"] = df["vel_x"] - df["vel_x"].rolling(drift_window, min_periods=1).mean()
    df["vel_y"] = df["vel_y"] - df["vel_y"].rolling(drift_window, min_periods=1).mean()
    df["vel_z"] = df["vel_z"] - df["vel_z"].rolling(drift_window, min_periods=1).mean()

    df["vel_mag"] = np.sqrt(
        df["vel_x"]**2 +
        df["vel_y"]**2 +
        df["vel_z"]**2
    )

    # Rotation vs translation hint
    df["motion_ratio"] = df["gyro_mag"] / (df["vel_mag"] + 1e-3)

    # Rolling stats
    window = 5
    df["acc_mean"] = df["acc_mag"].rolling(window, min_periods=1).mean()
    df["acc_std"] = df["acc_mag"].rolling(window, min_periods=1).std().fillna(0)

    df["gyro_mean"] = df["gyro_mag"].rolling(window, min_periods=1).mean()
    df["gyro_std"] = df["gyro_mag"].rolling(window, min_periods=1).std().fillna(0)

    df["vel_mean"] = df["vel_mag"].rolling(window, min_periods=1).mean()
    df["vel_std"] = df["vel_mag"].rolling(window, min_periods=1).std().fillna(0)

    # Energy-style movement score
    df["movement_score"] = (
        0.45 * df["acc_mean"] +
        0.35 * df["vel_mean"] +
        0.20 * df["gyro_mean"]
    )

    # --------------------------------
    # FATIGUE-FOCUSED TEMPORAL FEATURES
    # --------------------------------
    short_window = 25
    long_window = 100
    fatigue_window = 250

    # Velocity change features
    df["vel_diff"] = df["vel_mag"].diff().fillna(0)
    df["vel_diff_abs"] = df["vel_diff"].abs()

    # Long-term speed behaviour
    df["vel_mean_long"] = df["vel_mag"].rolling(long_window, min_periods=1).mean()
    df["vel_std_long"] = df["vel_mag"].rolling(long_window, min_periods=1).std().fillna(0)

    # Movement efficiency:
    # higher velocity produced for lower motion effort => fresher
    df["movement_efficiency"] = df["vel_mean"] / (df["acc_mean"] + df["gyro_mean"] + 1e-6)

    df["efficiency_short"] = df["movement_efficiency"].rolling(short_window, min_periods=1).mean()
    df["efficiency_long"] = df["movement_efficiency"].rolling(long_window, min_periods=1).mean()

    # Positive when recent efficiency is worse than long-term trend
    df["efficiency_drop"] = np.maximum(df["efficiency_long"] - df["efficiency_short"], 0)

    # Practical running thresholds
    sustainable_velocity = 4.5
    high_velocity = 6.0

    df["vel_excess"] = np.maximum(df["vel_mean"] - sustainable_velocity, 0)
    df["vel_high_excess"] = np.maximum(df["vel_mean"] - high_velocity, 0)

    # Time spent above thresholds
    df["above_4_5"] = (df["vel_mean"] > sustainable_velocity).astype(float)
    df["above_6_0"] = (df["vel_mean"] > high_velocity).astype(float)

    df["time_above_4_5"] = (df["above_4_5"] * df["dt"]).rolling(fatigue_window, min_periods=1).sum()
    df["time_above_6_0"] = (df["above_6_0"] * df["dt"]).rolling(fatigue_window, min_periods=1).sum()

    # Load accumulated over time
    df["load_score"] = (
        0.45 * df["acc_mean"] +
        0.35 * df["vel_mean"] +
        0.20 * df["gyro_mean"]
    ) * df["dt"]

    df["cumulative_load"] = df["load_score"].rolling(fatigue_window, min_periods=1).sum()

    # Higher penalty when effort is above sustainable pace
    df["effort_per_step"] = (
        0.50 * df["vel_excess"] +
        0.30 * df["vel_high_excess"] +
        0.20 * df["acc_mean"]
    ) * df["dt"]

    df["cumulative_effort"] = df["effort_per_step"].rolling(fatigue_window, min_periods=1).sum()

    # Instability / form breakdown
    df["instability_score"] = (
        0.40 * df["acc_std"] +
        0.35 * df["gyro_std"] +
        0.25 * df["vel_std"]
    )

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df = df.fillna(0)

    return df


# ==============================
# FEATURE COLUMNS
# ==============================
def get_feature_columns():
    return [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)",
        "Gyroscope x (rad/s)",
        "Gyroscope y (rad/s)",
        "Gyroscope z (rad/s)",
        "dt",
        "acc_mag",
        "gyro_mag",
        "vel_x",
        "vel_y",
        "vel_z",
        "vel_mag",
        "motion_ratio",
        "acc_mean",
        "acc_std",
        "gyro_mean",
        "gyro_std",
        "vel_mean",
        "vel_std",
        "movement_score",
        "vel_diff",
        "vel_diff_abs",
        "vel_mean_long",
        "vel_std_long",
        "movement_efficiency",
        "efficiency_short",
        "efficiency_long",
        "efficiency_drop",
        "vel_excess",
        "vel_high_excess",
        "above_4_5",
        "above_6_0",
        "time_above_4_5",
        "time_above_6_0",
        "load_score",
        "cumulative_load",
        "effort_per_step",
        "cumulative_effort",
        "instability_score"
    ]


# ==============================
# TARGET COLUMNS
# ==============================
def get_target_columns():
    return ["activity", "intensity", "fatigue"]


# ==============================
# LABEL GENERATION
# ==============================
def generate_activity(df):
    """
    Better separation:
    - stationary: low acceleration + low velocity + low rotation
    - rotation: low translation velocity but noticeable gyro
    - walking: moderate velocity / acceleration
    - running: high velocity / acceleration
    """
    conditions = [
        (df["vel_mean"] < 0.08) & (df["gyro_mean"] < 0.25) & (df["acc_mean"] < 0.35),
        (df["vel_mean"] < 0.12) & (df["gyro_mean"] >= 0.25),
        (df["vel_mean"] >= 0.12) & (df["vel_mean"] < 1.2) & (df["acc_mean"] < 3.2),
        (df["vel_mean"] >= 1.2) | (df["acc_mean"] >= 3.2)
    ]
    choices = ["stationary", "rotation", "walking", "running"]
    return np.select(conditions, choices, default="walking")


def generate_intensity(df):
    conditions = [
        (df["movement_score"] < 0.35),
        (df["movement_score"] >= 0.35) & (df["movement_score"] < 1.5),
        (df["movement_score"] >= 1.5)
    ]
    choices = ["low", "medium", "high"]
    return np.select(conditions, choices, default="low")


def generate_fatigue(df):
    """
    Fatigue is estimated from:
    - present effort
    - instability
    - cumulative load
    - efficiency drop
    - sustained high-speed exposure

    Output:
    - fresh
    - moderate
    - fatigued
    """
    df = df.copy()

    fatigue_score = (
        0.16 * df["acc_mean"] +
        0.08 * df["acc_std"] +
        0.10 * df["vel_mean"] +
        0.06 * df["vel_std"] +
        0.07 * df["gyro_mean"] +
        0.14 * df["cumulative_load"] +
        0.16 * df["cumulative_effort"] +
        0.10 * df["instability_score"] +
        0.09 * df["efficiency_drop"] +
        0.02 * df["time_above_4_5"] +
        0.02 * df["time_above_6_0"]
    )

    conditions = [
        (fatigue_score < 1.3),
        (fatigue_score >= 1.3) & (fatigue_score < 3.2),
        (fatigue_score >= 3.2)
    ]
    choices = ["fresh", "moderate", "fatigued"]

    return np.select(conditions, choices, default="fresh")


def add_generated_labels(df):
    df = df.copy()

    if "activity" not in df.columns:
        df["activity"] = generate_activity(df)

    if "intensity" not in df.columns:
        df["intensity"] = generate_intensity(df)

    if "fatigue" not in df.columns:
        df["fatigue"] = generate_fatigue(df)

    return df


# ==============================
# MODEL PIPELINE
# ==============================
def get_model():
    base_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model = MultiOutputClassifier(base_model)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    return pipeline