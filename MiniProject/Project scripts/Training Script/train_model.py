import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from model_utils import (
    clean_sensor_data,
    create_features,
    generate_labels,
    get_feature_columns,
    get_target_columns,
    get_model
)


# ==========================================
# SETTINGS
# ==========================================
MODEL_OUTPUT_PATH = "Activity_Model.pkl"
METADATA_OUTPUT_PATH = "Activity_Model_Metadata.json"

LABELED_ACTIVITY_FILES = [
    "Walking.csv",            # input file name to import here
    "Running.csv",            # input file name to import here
    "Rotation.csv",           # input file name to import here
    "IntensiveWalking.csv",   # input file name to import here
    "IntensiveRunning.csv",   # input file name to import here
    "Rotation Fatigue.csv",   # input file name to import here
]

RAW_ACCEL_FILE = "Accelerometer.csv"   # input file name to import here
RAW_GYRO_FILE = "Gyroscope.csv"        # input file name to import here

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================
# HELPERS
# ==========================================
def safe_read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded: {file_path} -> {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


def standardize_xyz_time_for_raw_sensor(df, sensor_type="acc"):
    df = df.copy()
    original_cols = list(df.columns)

    normalized = {col: str(col).strip().lower() for col in df.columns}
    rename_map = {}

    for old_col, low_col in normalized.items():
        if low_col in ["time", "time (s)", "timestamp", "t"]:
            rename_map[old_col] = "Time (s)"

        if sensor_type == "acc":
            if low_col in ["x", "x (m/s^2)", "linear acceleration x (m/s^2)"]:
                rename_map[old_col] = "Linear Acceleration x (m/s^2)"
            elif low_col in ["y", "y (m/s^2)", "linear acceleration y (m/s^2)"]:
                rename_map[old_col] = "Linear Acceleration y (m/s^2)"
            elif low_col in ["z", "z (m/s^2)", "linear acceleration z (m/s^2)"]:
                rename_map[old_col] = "Linear Acceleration z (m/s^2)"

        elif sensor_type == "gyro":
            if low_col in ["x", "x (rad/s)", "gyroscope x (rad/s)"]:
                rename_map[old_col] = "Gyroscope x (rad/s)"
            elif low_col in ["y", "y (rad/s)", "gyroscope y (rad/s)"]:
                rename_map[old_col] = "Gyroscope y (rad/s)"
            elif low_col in ["z", "z (rad/s)", "gyroscope z (rad/s)"]:
                rename_map[old_col] = "Gyroscope z (rad/s)"

    df = df.rename(columns=rename_map)

    print(f"\nStandardized raw {sensor_type} columns:")
    print(f"  Original: {original_cols}")
    print(f"  New     : {list(df.columns)}")

    return df


def merge_accel_gyro_by_time(acc_df, gyro_df):
    acc_df = standardize_xyz_time_for_raw_sensor(acc_df, sensor_type="acc")
    gyro_df = standardize_xyz_time_for_raw_sensor(gyro_df, sensor_type="gyro")

    required_acc = [
        "Time (s)",
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    required_gyro = [
        "Time (s)",
        "Gyroscope x (rad/s)",
        "Gyroscope y (rad/s)",
        "Gyroscope z (rad/s)"
    ]

    missing_acc = [c for c in required_acc if c not in acc_df.columns]
    missing_gyro = [c for c in required_gyro if c not in gyro_df.columns]

    if missing_acc:
        raise ValueError(f"Accelerometer file missing columns: {missing_acc}")
    if missing_gyro:
        raise ValueError(f"Gyroscope file missing columns: {missing_gyro}")

    acc_df = acc_df[required_acc].copy()
    gyro_df = gyro_df[required_gyro].copy()

    acc_df["Time (s)"] = pd.to_numeric(acc_df["Time (s)"], errors="coerce")
    gyro_df["Time (s)"] = pd.to_numeric(gyro_df["Time (s)"], errors="coerce")

    acc_df = acc_df.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)
    gyro_df = gyro_df.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)

    if acc_df.empty or gyro_df.empty:
        raise ValueError("One or both raw sensor files are empty after cleaning timestamps.")

    merged = pd.merge_asof(
        acc_df,
        gyro_df,
        on="Time (s)",
        direction="nearest",
        tolerance=0.05
    )

    return merged


def prepare_training_dataframe_from_file(file_path):
    raw_df = safe_read_csv(file_path)
    if raw_df is None or raw_df.empty:
        return None

    cleaned = clean_sensor_data(raw_df)
    featured = create_features(cleaned)
    labeled = generate_labels(featured)
    labeled["source_file"] = os.path.basename(file_path)

    print(f"Prepared training frame from {file_path}: {labeled.shape}")
    return labeled


def prepare_training_dataframe_from_raw_pair(acc_file, gyro_file):
    if not (os.path.exists(acc_file) and os.path.exists(gyro_file)):
        print("Raw accelerometer/gyroscope pair not fully found. Skipping raw pair.")
        return None

    acc_df = safe_read_csv(acc_file)
    gyro_df = safe_read_csv(gyro_file)

    if acc_df is None or gyro_df is None:
        return None

    try:
        merged = merge_accel_gyro_by_time(acc_df, gyro_df)
        cleaned = clean_sensor_data(merged)
        featured = create_features(cleaned)
        labeled = generate_labels(featured)
        labeled["source_file"] = f"{os.path.basename(acc_file)} + {os.path.basename(gyro_file)}"

        print(f"Prepared raw merged pair: {labeled.shape}")
        return labeled

    except Exception as e:
        print(f"Failed to process raw sensor pair: {e}")
        return None


def print_target_distributions(df, target_cols, title="Class distribution"):
    print(f"\n{title}")
    print("=" * len(title))
    for col in target_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))


def extract_feature_importance(model, feature_cols, target_cols):
    output = {}

    if not hasattr(model, "estimators_"):
        return output

    for idx, target_name in enumerate(target_cols):
        estimator = model.estimators_[idx]
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values("importance", ascending=False)

            output[target_name] = importance_df.to_dict(orient="records")

    return output


def evaluate_model(model, X_test, y_test, target_cols):
    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT SET")
    print("=" * 60)

    y_pred = model.predict(X_test)

    if isinstance(y_pred, np.ndarray):
        y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)
    else:
        y_pred_df = pd.DataFrame(np.array(y_pred), columns=target_cols, index=y_test.index)

    metrics_summary = {}

    for col in target_cols:
        print(f"\n--- {col.upper()} ---")
        acc = accuracy_score(y_test[col], y_pred_df[col])
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test[col], y_pred_df[col], zero_division=0))

        labels = sorted(set(y_test[col].astype(str)) | set(y_pred_df[col].astype(str)))
        cm = confusion_matrix(y_test[col], y_pred_df[col], labels=labels)

        print("Labels:", labels)
        print("Confusion Matrix:")
        print(cm)

        metrics_summary[col] = {
            "accuracy": float(acc),
            "true_distribution": y_test[col].value_counts().to_dict(),
            "pred_distribution": y_pred_df[col].value_counts().to_dict(),
            "confusion_matrix_labels": labels,
            "confusion_matrix": cm.tolist()
        }

    exact_match = (y_pred_df[target_cols] == y_test[target_cols]).all(axis=1).mean()
    print(f"\nExact-match accuracy across all outputs: {exact_match:.4f}")
    metrics_summary["exact_match_accuracy"] = float(exact_match)

    return metrics_summary


def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata: {path}")


# ==========================================
# MAIN TRAINING
# ==========================================
def main():
    all_frames = []

    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    for file_name in LABELED_ACTIVITY_FILES:
        if not os.path.exists(file_name):
            print(f"Not found, skipping: {file_name}")
            continue

        df = prepare_training_dataframe_from_file(file_name)
        if df is not None and not df.empty:
            all_frames.append(df)

    raw_pair_df = prepare_training_dataframe_from_raw_pair(RAW_ACCEL_FILE, RAW_GYRO_FILE)
    if raw_pair_df is not None and not raw_pair_df.empty:
        all_frames.append(raw_pair_df)

    if not all_frames:
        raise ValueError("No usable CSV files found for training.")

    combined_df = pd.concat(all_frames, ignore_index=True)
    combined_df = combined_df.replace([np.inf, -np.inf], 0).fillna(0)

    feature_cols = get_feature_columns()
    target_cols = get_target_columns()

    missing_features = [col for col in feature_cols if col not in combined_df.columns]
    missing_targets = [col for col in target_cols if col not in combined_df.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns after preparation: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns after preparation: {missing_targets}")

    X = combined_df[feature_cols].copy()
    y = combined_df[target_cols].copy()
    groups = combined_df["source_file"].copy()

    print("\nCombined training shape:", combined_df.shape)
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    print("Unique source groups:", groups.nunique())

    print_target_distributions(combined_df, target_cols, title="Overall class distribution")

    unique_groups = groups.nunique()

    if unique_groups >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        train_groups = groups.iloc[train_idx]
        test_groups = groups.iloc[test_idx]

        print("\nTrain groups:", sorted(train_groups.unique().tolist()))
        print("Test groups :", sorted(test_groups.unique().tolist()))
    else:
        print("\nOnly one source group found. Using all data for training; held-out evaluation skipped.")
        X_train, y_train = X, y
        X_test, y_test = None, None

    print("\nTraining set shape:", X_train.shape)
    if X_test is not None:
        print("Test set shape    :", X_test.shape)

    model = get_model()
    model.fit(X_train, y_train)

    metrics_summary = {}
    if X_test is not None and len(X_test) > 0:
        metrics_summary = evaluate_model(model, X_test, y_test, target_cols)

    final_model = get_model()
    final_model.fit(X, y)

    feature_importance_summary = extract_feature_importance(
        final_model,
        feature_cols,
        target_cols
    )

    joblib.dump(final_model, MODEL_OUTPUT_PATH)

    metadata = {
        "model_output_path": MODEL_OUTPUT_PATH,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "training_files_used": [
            df["source_file"].iloc[0]
            for df in all_frames
            if "source_file" in df.columns and not df.empty
        ],
        "total_rows": int(len(combined_df)),
        "total_features": int(len(feature_cols)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "evaluation": metrics_summary,
        "feature_importance": feature_importance_summary
    }
    save_metadata(metadata, METADATA_OUTPUT_PATH)

    print("\n" + "=" * 60)
    print(f"FINAL MODEL TRAINED AND SAVED AS: {MODEL_OUTPUT_PATH}")
    print(f"METADATA SAVED AS: {METADATA_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()