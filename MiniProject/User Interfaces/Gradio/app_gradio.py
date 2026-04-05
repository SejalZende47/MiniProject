import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from model_utils import (
    clean_sensor_data,
    create_features,
    generate_labels,
    get_feature_columns,
    get_target_columns,
    get_model
)


MODEL_PATH = "Activity_Model.pkl"
METADATA_PATH = "Activity_Model_Metadata.json"
BACKUP_MODEL_PATH = "Activity_Model_backup.pkl"


# ==============================
# LOAD MODEL / METADATA
# ==============================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


model = load_model()
metadata = load_metadata()


# ==============================
# COLUMN / INPUT HELPERS
# ==============================
def standardize_raw_sensor_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_matching_column(df, candidates):
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def prepare_accelerometer_df(acc_df):
    acc_df = standardize_raw_sensor_columns(acc_df)

    time_col = find_matching_column(
        acc_df, ["Time (s)", "time", "timestamp", "t", "Time", "Time(s)", "Time [s]"]
    )
    if time_col is None:
        raise ValueError(
            f"No time column found in accelerometer file. Found columns: {list(acc_df.columns)}"
        )

    x_col = find_matching_column(
        acc_df,
        [
            "Linear Acceleration x (m/s^2)", "X (m/s^2)", "x", "ax",
            "Acceleration x (m/s^2)", "acceleration x (m/s^2)",
            "acc_x", "lin_acc_x"
        ]
    )
    y_col = find_matching_column(
        acc_df,
        [
            "Linear Acceleration y (m/s^2)", "Y (m/s^2)", "y", "ay",
            "Acceleration y (m/s^2)", "acceleration y (m/s^2)",
            "acc_y", "lin_acc_y"
        ]
    )
    z_col = find_matching_column(
        acc_df,
        [
            "Linear Acceleration z (m/s^2)", "Z (m/s^2)", "z", "az",
            "Acceleration z (m/s^2)", "acceleration z (m/s^2)",
            "acc_z", "lin_acc_z"
        ]
    )

    if None in [x_col, y_col, z_col]:
        raise ValueError(
            "Accelerometer file does not contain the required columns.\n"
            f"Found columns: {list(acc_df.columns)}"
        )

    acc_df = acc_df.rename(columns={
        time_col: "Time (s)",
        x_col: "Linear Acceleration x (m/s^2)",
        y_col: "Linear Acceleration y (m/s^2)",
        z_col: "Linear Acceleration z (m/s^2)"
    })

    return acc_df[
        [
            "Time (s)",
            "Linear Acceleration x (m/s^2)",
            "Linear Acceleration y (m/s^2)",
            "Linear Acceleration z (m/s^2)"
        ]
    ].copy()


def prepare_gyroscope_df(gyro_df):
    gyro_df = standardize_raw_sensor_columns(gyro_df)

    time_col = find_matching_column(
        gyro_df, ["Time (s)", "time", "timestamp", "t", "Time", "Time(s)", "Time [s]"]
    )
    if time_col is None:
        raise ValueError(
            f"No time column found in gyroscope file. Found columns: {list(gyro_df.columns)}"
        )

    x_col = find_matching_column(
        gyro_df,
        [
            "Gyroscope x (rad/s)", "X (rad/s)", "x", "gx",
            "Angular velocity x (rad/s)", "gyroscope x (rad/s)", "gyro_x"
        ]
    )
    y_col = find_matching_column(
        gyro_df,
        [
            "Gyroscope y (rad/s)", "Y (rad/s)", "y", "gy",
            "Angular velocity y (rad/s)", "gyroscope y (rad/s)", "gyro_y"
        ]
    )
    z_col = find_matching_column(
        gyro_df,
        [
            "Gyroscope z (rad/s)", "Z (rad/s)", "z", "gz",
            "Angular velocity z (rad/s)", "gyroscope z (rad/s)", "gyro_z"
        ]
    )

    if None in [x_col, y_col, z_col]:
        raise ValueError(
            "Gyroscope file does not contain the required columns.\n"
            f"Found columns: {list(gyro_df.columns)}"
        )

    gyro_df = gyro_df.rename(columns={
        time_col: "Time (s)",
        x_col: "Gyroscope x (rad/s)",
        y_col: "Gyroscope y (rad/s)",
        z_col: "Gyroscope z (rad/s)"
    })

    return gyro_df[
        [
            "Time (s)",
            "Gyroscope x (rad/s)",
            "Gyroscope y (rad/s)",
            "Gyroscope z (rad/s)"
        ]
    ].copy()


def combine_accel_gyro(acc_df, gyro_df):
    acc_df = prepare_accelerometer_df(acc_df)
    gyro_df = prepare_gyroscope_df(gyro_df)

    acc_df["Time (s)"] = pd.to_numeric(acc_df["Time (s)"], errors="coerce")
    gyro_df["Time (s)"] = pd.to_numeric(gyro_df["Time (s)"], errors="coerce")

    acc_df = acc_df.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)
    gyro_df = gyro_df.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)

    if acc_df.empty:
        raise ValueError("Accelerometer data is empty after time conversion.")
    if gyro_df.empty:
        raise ValueError("Gyroscope data is empty after time conversion.")

    combined_df = pd.merge_asof(
        acc_df,
        gyro_df,
        on="Time (s)",
        direction="nearest",
        tolerance=0.05
    )

    return combined_df


def toggle_input_mode(mode):
    is_separate = (mode == "Separate Accelerometer + Gyroscope")
    return (
        gr.update(visible=is_separate),
        gr.update(visible=is_separate),
        gr.update(visible=not is_separate),
    )


def load_input_data(mode, acc_file, gyro_file, combined_file):
    if mode == "Separate Accelerometer + Gyroscope":
        if acc_file is None or gyro_file is None:
            raise ValueError("Please upload both accelerometer and gyroscope CSV files.")
        raw_acc_df = pd.read_csv(acc_file.name, sep=None, engine="python")
        raw_gyro_df = pd.read_csv(gyro_file.name, sep=None, engine="python")
        df_source = combine_accel_gyro(raw_acc_df, raw_gyro_df)
        input_mode = "Separate Accelerometer + Gyroscope CSVs"
    else:
        if combined_file is None:
            raise ValueError("Please upload the combined CSV file.")
        df_source = pd.read_csv(combined_file.name, sep=None, engine="python")
        input_mode = "Single Combined CSV"

    return df_source, input_mode


# ==============================
# MODEL HELPERS
# ==============================
def backup_existing_model():
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_MODEL_PATH)
        return BACKUP_MODEL_PATH
    return None


def reload_model_and_metadata():
    global model, metadata
    model = load_model()
    metadata = load_metadata()


def get_training_safeguards(df, min_rows=50, min_class_count=5):
    target_cols = get_target_columns()

    if df is None or df.empty:
        return False, "Training skipped: dataset is empty."

    if len(df) < min_rows:
        return False, f"Training skipped: need at least {min_rows} rows, found {len(df)}."

    for col in target_cols:
        if col not in df.columns:
            return False, f"Training skipped: missing target column '{col}'."

        counts = df[col].value_counts(dropna=False)

        if counts.shape[0] < 2:
            return False, f"Training skipped: '{col}' has only one class."

        too_small = counts[counts < min_class_count]
        if not too_small.empty:
            details = ", ".join([f"{idx}={val}" for idx, val in too_small.items()])
            return False, (
                f"Training skipped: '{col}' has classes with too few samples "
                f"(minimum {min_class_count} each). Problem: {details}"
            )

    return True, "Training safeguards passed."


def retrain_model(df, overwrite_model=False, min_rows=50, min_class_count=5):
    ok, safeguard_message = get_training_safeguards(
        df,
        min_rows=min_rows,
        min_class_count=min_class_count
    )

    if not ok:
        return safeguard_message

    X = df[get_feature_columns()]
    y = df[get_target_columns()]

    if X.empty:
        return "Training skipped: feature matrix is empty."

    if not overwrite_model and os.path.exists(MODEL_PATH):
        return (
            "Training ready, but overwrite protection is ON. "
            "Enable overwrite to replace the saved model."
        )

    backup_path = backup_existing_model()

    new_model = get_model()
    new_model.fit(X, y)
    joblib.dump(new_model, MODEL_PATH)

    reload_model_and_metadata()

    if backup_path:
        return (
            f"Model trained on {len(X)} rows and saved. "
            f"Previous model backed up as: {backup_path}"
        )

    return f"Model trained on {len(X)} rows and saved."


def evaluate_model_grouped(df, test_size=0.2, random_state=42):
    target_cols = get_target_columns()

    if df is None or df.empty:
        return "Evaluation skipped: dataset is empty."

    if len(df) < 30:
        return "Evaluation skipped: need at least 30 rows."

    X = df[get_feature_columns()]
    y = df[target_cols]

    if X.empty:
        return "Evaluation skipped: feature matrix is empty."

    if "source_file" not in df.columns or df["source_file"].nunique() < 2:
        return "Evaluation skipped: need at least 2 source groups for grouped evaluation."

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["source_file"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    eval_model = get_model()
    eval_model.fit(X_train, y_train)

    y_pred = eval_model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

    lines = [
        f"Training rows: {len(X_train)}",
        f"Testing rows: {len(X_test)}",
        f"Train groups: {df['source_file'].iloc[train_idx].nunique()}",
        f"Test groups: {df['source_file'].iloc[test_idx].nunique()}",
        ""
    ]

    for col in target_cols:
        acc = accuracy_score(y_test[col], y_pred_df[col])
        lines.append(f"{col} Accuracy: {acc * 100:.2f}%")
        lines.append("")

    exact_match = (y_pred_df[target_cols] == y_test[target_cols]).all(axis=1).mean()
    lines.append(f"Exact Match Accuracy: {exact_match * 100:.2f}%")

    return "\n".join(lines)


# ==============================
# PREDICTION HELPERS
# ==============================
def get_model_confidence_scores(trained_model, X):
    target_cols = get_target_columns()

    try:
        probas = trained_model.predict_proba(X)
    except Exception:
        return pd.DataFrame(index=X.index)

    confidence_data = {}

    if isinstance(probas, list) and len(probas) == len(target_cols):
        for target_name, target_proba in zip(target_cols, probas):
            if isinstance(target_proba, np.ndarray) and target_proba.ndim == 2:
                confidence_data[f"{target_name}_confidence"] = target_proba.max(axis=1) * 100.0

    return pd.DataFrame(confidence_data, index=X.index)


def smooth_final_state(pred_df, column_name, window=25):
    if pred_df is None or pred_df.empty or column_name not in pred_df.columns:
        return None

    recent = pred_df[column_name].tail(window)
    if recent.empty:
        return None

    return recent.mode().iloc[0]


def create_count_df(pred_df, col_name):
    if pred_df is None or pred_df.empty or col_name not in pred_df.columns:
        return pd.DataFrame(columns=[col_name, "count"])

    counts = pred_df[col_name].value_counts().reset_index()
    counts.columns = [col_name, "count"]
    return counts


def create_window_distribution(pred_df, col_name, window_size=50):
    if pred_df is None or pred_df.empty or col_name not in pred_df.columns:
        return pd.DataFrame(columns=[col_name, "count"])

    values = pred_df[col_name].reset_index(drop=True)
    window_labels = []

    for i in range(0, len(values), window_size):
        chunk = values.iloc[i:i + window_size]
        if not chunk.empty:
            window_labels.append(chunk.mode().iloc[0])

    if not window_labels:
        return pd.DataFrame(columns=[col_name, "count"])

    result = pd.Series(window_labels).value_counts().reset_index()
    result.columns = [col_name, "count"]
    return result


def create_pie_chart(pred_df, col_name, title):
    if pred_df is None or pred_df.empty or col_name not in pred_df.columns:
        return None

    counts = pred_df[col_name].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title(title)
    ax.axis("equal")
    plt.close(fig)
    return fig


def create_time_series_df(df, y_col, label):
    if df is None or df.empty or y_col not in df.columns:
        return pd.DataFrame(columns=["Index", label])

    return pd.DataFrame({
        "Index": np.arange(len(df)),
        label: df[y_col].reset_index(drop=True)
    })


def create_feature_diagnostics(df):
    if df is None or df.empty:
        return {
            "avg_motion_energy": 0.0,
            "avg_acc": 0.0,
            "avg_gyro": 0.0,
            "avg_jerk": 0.0,
            "avg_instability": 0.0,
            "avg_cumulative_load": 0.0,
            "avg_energy_trend": 0.0
        }

    return {
        "avg_motion_energy": float(df["motion_energy"].mean()) if "motion_energy" in df.columns else 0.0,
        "avg_acc": float(df["acc_mean"].mean()) if "acc_mean" in df.columns else 0.0,
        "avg_gyro": float(df["gyro_mean"].mean()) if "gyro_mean" in df.columns else 0.0,
        "avg_jerk": float(df["jerk_mean"].mean()) if "jerk_mean" in df.columns else 0.0,
        "avg_instability": float(df["instability"].mean()) if "instability" in df.columns else 0.0,
        "avg_cumulative_load": float(df["cumulative_load"].mean()) if "cumulative_load" in df.columns else 0.0,
        "avg_energy_trend": float(df["energy_trend"].mean()) if "energy_trend" in df.columns else 0.0,
    }


def create_fatigue_reason(diag, final_fatigue):
    reasons = []

    if diag["avg_motion_energy"] > 1.5:
        reasons.append("high motion energy")
    elif diag["avg_motion_energy"] > 0.6:
        reasons.append("moderate motion energy")

    if diag["avg_cumulative_load"] > 2.0:
        reasons.append("high cumulative load")

    if diag["avg_instability"] > 0.5:
        reasons.append("movement instability")

    if diag["avg_jerk"] > 0.8:
        reasons.append("rapid movement changes")

    if diag["avg_energy_trend"] < -0.2:
        reasons.append("declining recent energy trend")

    if not reasons:
        if final_fatigue == "fresh":
            return "low sustained load"
        return "mixed fatigue indicators"

    return ", ".join(reasons[:3])


def create_motion_hint(diag):
    if diag["avg_motion_energy"] > 1.5:
        return "High motion energy"
    if diag["avg_motion_energy"] > 0.6:
        return "Moderate motion energy"
    return "Low motion energy"


def build_prediction_timeline(pred_df, base_df):
    timeline_df = pred_df.copy()

    if "Time (s)" in base_df.columns:
        timeline_df["Time"] = pd.to_numeric(base_df["Time (s)"], errors="coerce").reset_index(drop=True)
        timeline_df["Time"] = timeline_df["Time"].ffill().bfill()
    else:
        timeline_df["Time"] = np.arange(len(timeline_df))

    activity_order = ["stationary", "rotation", "walking", "running"]
    activity_map = {label: idx for idx, label in enumerate(activity_order)}

    for label in timeline_df["activity"].astype(str).unique():
        if label not in activity_map:
            activity_map[label] = len(activity_map)

    timeline_df["Activity Code"] = timeline_df["activity"].map(activity_map)

    plot_df = pd.DataFrame({
        "Time": timeline_df["Time"],
        "Activity Code": timeline_df["Activity Code"]
    })

    mapping_text = " | ".join([f"{k} = {v}" for k, v in activity_map.items()])
    return plot_df, mapping_text


# ==============================
# MODEL INSIGHT HELPERS
# ==============================
def empty_insight_return(message="No metadata found. Run train_model.py first."):
    return (
        message,
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        None, None, None
    )


def plot_feature_importance_from_df(fi_df, title):
    if fi_df is None or fi_df.empty or "feature" not in fi_df.columns or "importance" not in fi_df.columns:
        return None

    plot_df = fi_df.sort_values("importance", ascending=False).head(10).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_title(title)
    plt.tight_layout()
    plt.close(fig)
    return fig


def get_saved_model_insights():
    global metadata
    metadata = load_metadata()

    if not metadata:
        return empty_insight_return()

    evaluation = metadata.get("evaluation", {})
    fi = metadata.get("feature_importance", {})

    summary_lines = [
        f"Model Path: {metadata.get('model_output_path', 'N/A')}",
        f"Total Rows: {metadata.get('total_rows', 'N/A')}",
        f"Total Features: {metadata.get('total_features', 'N/A')}",
        f"Test Size: {metadata.get('test_size', 'N/A')}",
        f"Random State: {metadata.get('random_state', 'N/A')}",
        f"Exact Match Accuracy: {evaluation.get('exact_match_accuracy', 'N/A')}"
    ]
    summary_text = "\n".join(summary_lines)

    cm_tables = []
    fi_tables = []
    fi_plots = []

    for target in get_target_columns():
        info = evaluation.get(target, {})
        labels = info.get("confusion_matrix_labels", [])
        cm = info.get("confusion_matrix", [])

        if labels and cm:
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        else:
            cm_df = pd.DataFrame()

        fi_rows = fi.get(target, [])
        fi_df = pd.DataFrame(fi_rows)

        cm_tables.append(cm_df)
        fi_tables.append(fi_df)
        fi_plots.append(plot_feature_importance_from_df(fi_df, f"Top Features for {target}"))

    return (
        summary_text,
        cm_tables[0], cm_tables[1], cm_tables[2],
        fi_tables[0], fi_tables[1], fi_tables[2],
        fi_plots[0], fi_plots[1], fi_plots[2]
    )


# ==============================
# MAIN ACTIONS
# ==============================
def process_files(mode, acc_file, gyro_file, combined_file):
    global model, metadata

    try:
        df_source, input_mode = load_input_data(mode, acc_file, gyro_file, combined_file)

        df = clean_sensor_data(df_source)
        df = create_features(df)

        if df.empty:
            raise ValueError("No valid data found after cleaning and feature creation.")

        generated_label_df = generate_labels(df.copy())

        if model is None:
            raise ValueError("No trained model found. Run train_model.py first or retrain.")

        feature_cols = get_feature_columns()
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns for prediction: {missing_features}")

        X = df[feature_cols]
        predictions = model.predict(X)
        pred_df = pd.DataFrame(predictions, columns=get_target_columns(), index=df.index)

        conf_df = get_model_confidence_scores(model, X)
        if not conf_df.empty:
            pred_df = pd.concat([pred_df, conf_df], axis=1)

        final_activity = smooth_final_state(pred_df, "activity", window=25)
        final_intensity = smooth_final_state(pred_df, "intensity", window=25)
        final_fatigue = smooth_final_state(pred_df, "fatigue", window=25)

        activity_conf = pred_df["activity_confidence"].tail(25).mean() if "activity_confidence" in pred_df.columns else None
        intensity_conf = pred_df["intensity_confidence"].tail(25).mean() if "intensity_confidence" in pred_df.columns else None
        fatigue_conf = pred_df["fatigue_confidence"].tail(25).mean() if "fatigue_confidence" in pred_df.columns else None

        diag = create_feature_diagnostics(df)
        fatigue_reason = create_fatigue_reason(diag, final_fatigue)
        motion_hint = create_motion_hint(diag)

        summary = [
            f"Input mode: {input_mode}",
            f"Rows processed: {len(df)}",
            f"Final Activity: {final_activity}" + (f" ({activity_conf:.2f}% confidence)" if activity_conf is not None else ""),
            f"Final Intensity: {final_intensity}" + (f" ({intensity_conf:.2f}% confidence)" if intensity_conf is not None else ""),
            f"Final Fatigue State: {final_fatigue}" + (f" ({fatigue_conf:.2f}% confidence)" if fatigue_conf is not None else ""),
            f"Motion Hint: {motion_hint}",
            f"Fatigue Basis: {fatigue_reason}",
            f"Avg Motion Energy: {diag['avg_motion_energy']:.4f}",
            f"Avg Acceleration Mean: {diag['avg_acc']:.4f}",
            f"Avg Gyroscope Mean: {diag['avg_gyro']:.4f}",
            f"Avg Jerk Mean: {diag['avg_jerk']:.4f}",
            f"Avg Cumulative Load: {diag['avg_cumulative_load']:.4f}",
            f"Avg Instability: {diag['avg_instability']:.4f}",
            f"Avg Energy Trend: {diag['avg_energy_trend']:.4f}",
        ]

        if metadata:
            summary.append(f"Saved Model Features: {len(metadata.get('feature_columns', []))}")

        summary_text = "\n".join(summary)

        raw_preview = df.head(10)
        pred_preview = pred_df.head(10)
        label_dist = generated_label_df[get_target_columns()].apply(lambda col: col.value_counts()).fillna(0)

        acc_plot_df = create_time_series_df(df, "acc_mag", "Acceleration Magnitude")
        gyro_plot_df = create_time_series_df(df, "gyro_mag", "Gyroscope Magnitude")
        motion_energy_plot_df = create_time_series_df(df, "motion_energy", "Motion Energy")
        cumulative_load_plot_df = create_time_series_df(df, "cumulative_load", "Cumulative Load")

        timeline_plot_df, mapping_text = build_prediction_timeline(pred_df, df)

        activity_bar_df = create_count_df(pred_df, "activity")
        activity_window_df = create_window_distribution(pred_df, "activity", window_size=50)
        fatigue_bar_df = create_count_df(pred_df, "fatigue")
        activity_pie_fig = create_pie_chart(pred_df, "activity", "Activity Distribution (Pie)")

        return (
            summary_text,
            raw_preview,
            pred_preview,
            label_dist,
            acc_plot_df,
            gyro_plot_df,
            motion_energy_plot_df,
            cumulative_load_plot_df,
            timeline_plot_df,
            activity_bar_df,
            activity_window_df,
            fatigue_bar_df,
            activity_pie_fig,
            mapping_text
        )

    except Exception as e:
        return (
            f"Error: {e}",
            None, None, None, None, None, None, None,
            None, None, None, None, None, ""
        )


def retrain_from_files(mode, acc_file, gyro_file, combined_file, overwrite_model):
    try:
        df_source, input_mode = load_input_data(mode, acc_file, gyro_file, combined_file)
        df = clean_sensor_data(df_source)
        df = create_features(df)
        training_df = generate_labels(df.copy())
        training_df["source_file"] = input_mode

        return retrain_model(training_df, overwrite_model=overwrite_model)

    except Exception as e:
        return f"Error: {e}"


def evaluate_from_files(mode, acc_file, gyro_file, combined_file):
    try:
        df_source, input_mode = load_input_data(mode, acc_file, gyro_file, combined_file)
        df = clean_sensor_data(df_source)
        df = create_features(df)
        training_df = generate_labels(df.copy())

        n = len(training_df)
        if n >= 100:
            segment_size = max(25, n // 5)
            groups = []
            for i in range(n):
                groups.append(f"{input_mode}_segment_{i // segment_size}")
            training_df["source_file"] = groups
        else:
            training_df["source_file"] = input_mode

        return evaluate_model_grouped(training_df)

    except Exception as e:
        return f"Error: {e}"


# ==============================
# UI
# ==============================
with gr.Blocks(title="Human Activity + Fatigue Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Human Activity + Fatigue Detection")
    gr.Markdown(
        "Upload sensor files and predict whether the person is "
        "**stationary / rotation / walking / running** and whether they are "
        "**fresh / moderate / fatigued**."
    )

    input_mode = gr.Radio(
        choices=[
            "Separate Accelerometer + Gyroscope",
            "Single Combined CSV"
        ],
        value="Separate Accelerometer + Gyroscope",
        label="Input Mode"
    )

    with gr.Row():
        acc_file = gr.File(label="Accelerometer CSV", file_types=[".csv"], visible=True)
        gyro_file = gr.File(label="Gyroscope CSV", file_types=[".csv"], visible=True)
        combined_file = gr.File(label="Combined CSV", file_types=[".csv"], visible=False)

    input_mode.change(
        fn=toggle_input_mode,
        inputs=input_mode,
        outputs=[acc_file, gyro_file, combined_file]
    )

    with gr.Tabs():
        with gr.Tab("Prediction"):
            predict_btn = gr.Button("Run Prediction", variant="primary")
            summary_output = gr.Textbox(label="Summary", lines=14)
            mapping_output = gr.Textbox(label="Activity Labels Mapping", lines=2)

            with gr.Row():
                raw_preview_output = gr.Dataframe(label="Processed Data Preview")
                pred_preview_output = gr.Dataframe(label="Prediction Preview")

            label_dist_output = gr.Dataframe(label="Generated Label Distribution")

            acc_plot_output = gr.LinePlot(
                label="Acceleration Magnitude",
                x="Index",
                y="Acceleration Magnitude"
            )

            gyro_plot_output = gr.LinePlot(
                label="Gyroscope Magnitude",
                x="Index",
                y="Gyroscope Magnitude"
            )

            motion_energy_plot_output = gr.LinePlot(
                label="Motion Energy",
                x="Index",
                y="Motion Energy"
            )

            cumulative_load_plot_output = gr.LinePlot(
                label="Cumulative Load",
                x="Index",
                y="Cumulative Load"
            )

            timeline_plot_output = gr.LinePlot(
                label="Activity Timeline",
                x="Time",
                y="Activity Code"
            )

            activity_bar_output = gr.BarPlot(
                x="activity",
                y="count",
                title="Activity Distribution (Bar)"
            )

            window_activity_output = gr.BarPlot(
                x="activity",
                y="count",
                title="Window-Based Activity Distribution"
            )

            fatigue_bar_output = gr.BarPlot(
                x="fatigue",
                y="count",
                title="Fatigue State Distribution"
            )

            activity_pie_output = gr.Plot(label="Activity Distribution (Pie)")

            predict_btn.click(
                fn=process_files,
                inputs=[input_mode, acc_file, gyro_file, combined_file],
                outputs=[
                    summary_output,
                    raw_preview_output,
                    pred_preview_output,
                    label_dist_output,
                    acc_plot_output,
                    gyro_plot_output,
                    motion_energy_plot_output,
                    cumulative_load_plot_output,
                    timeline_plot_output,
                    activity_bar_output,
                    window_activity_output,
                    fatigue_bar_output,
                    activity_pie_output,
                    mapping_output
                ]
            )

        with gr.Tab("Retrain"):
            overwrite_checkbox = gr.Checkbox(
                label="Overwrite existing saved model",
                value=False
            )
            retrain_btn = gr.Button("Retrain Model")
            retrain_output = gr.Textbox(label="Retrain Status", lines=5)

            retrain_btn.click(
                fn=retrain_from_files,
                inputs=[input_mode, acc_file, gyro_file, combined_file, overwrite_checkbox],
                outputs=retrain_output
            )

        with gr.Tab("Evaluation"):
            eval_btn = gr.Button("Evaluate Model")
            eval_output = gr.Textbox(label="Evaluation Status", lines=10)

            eval_btn.click(
                fn=evaluate_from_files,
                inputs=[input_mode, acc_file, gyro_file, combined_file],
                outputs=eval_output
            )

        with gr.Tab("Model Insights"):
            insight_btn = gr.Button("Load Saved Model Insights", variant="primary")
            insight_summary = gr.Textbox(label="Saved Metadata Summary", lines=8)

            gr.Markdown("## Confusion Matrices")
            cm_activity = gr.Dataframe(label="Activity Confusion Matrix")
            cm_intensity = gr.Dataframe(label="Intensity Confusion Matrix")
            cm_fatigue = gr.Dataframe(label="Fatigue Confusion Matrix")

            gr.Markdown("## Feature Importance Tables")
            fi_activity = gr.Dataframe(label="Activity Feature Importance")
            fi_intensity = gr.Dataframe(label="Intensity Feature Importance")
            fi_fatigue = gr.Dataframe(label="Fatigue Feature Importance")

            gr.Markdown("## Feature Importance Plots")
            fi_plot_activity = gr.Plot(label="Activity Feature Importance Plot")
            fi_plot_intensity = gr.Plot(label="Intensity Feature Importance Plot")
            fi_plot_fatigue = gr.Plot(label="Fatigue Feature Importance Plot")

            insight_btn.click(
                fn=get_saved_model_insights,
                inputs=[],
                outputs=[
                    insight_summary,
                    cm_activity, cm_intensity, cm_fatigue,
                    fi_activity, fi_intensity, fi_fatigue,
                    fi_plot_activity, fi_plot_intensity, fi_plot_fatigue
                ]
            )


if __name__ == "__main__":
    demo.launch()