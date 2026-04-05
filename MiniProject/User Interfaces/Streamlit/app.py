import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import streamlit as st
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

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "Activity_Model.pkl"
METADATA_PATH = "Activity_Model_Metadata.json"
BACKUP_MODEL_PATH = "Activity_Model_backup.pkl"

st.set_page_config(
    page_title="Human Activity + Fatigue Detection",
    page_icon="🧠",
    layout="wide"
)

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


if "model" not in st.session_state:
    st.session_state.model = load_model()

if "metadata" not in st.session_state:
    st.session_state.metadata = load_metadata()

# ==============================
# INPUT HELPERS
# ==============================
def standardize_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def find_col(df, candidates):
    normalized = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in normalized:
            return normalized[c.lower()]
    return None


def prepare_acc(df):
    df = standardize_columns(df)

    time = find_col(df, ["time", "time (s)", "timestamp", "t"])
    x = find_col(df, ["x", "x (m/s^2)", "linear acceleration x (m/s^2)"])
    y = find_col(df, ["y", "y (m/s^2)", "linear acceleration y (m/s^2)"])
    z = find_col(df, ["z", "z (m/s^2)", "linear acceleration z (m/s^2)"])

    if None in [time, x, y, z]:
        raise ValueError(f"Invalid Accelerometer file. Found columns: {list(df.columns)}")

    return df.rename(columns={
        time: "Time (s)",
        x: "Linear Acceleration x (m/s^2)",
        y: "Linear Acceleration y (m/s^2)",
        z: "Linear Acceleration z (m/s^2)"
    })


def prepare_gyro(df):
    df = standardize_columns(df)

    time = find_col(df, ["time", "time (s)", "timestamp", "t"])
    x = find_col(df, ["x", "x (rad/s)", "gyroscope x (rad/s)"])
    y = find_col(df, ["y", "y (rad/s)", "gyroscope y (rad/s)"])
    z = find_col(df, ["z", "z (rad/s)", "gyroscope z (rad/s)"])

    if None in [time, x, y, z]:
        raise ValueError(f"Invalid Gyroscope file. Found columns: {list(df.columns)}")

    return df.rename(columns={
        time: "Time (s)",
        x: "Gyroscope x (rad/s)",
        y: "Gyroscope y (rad/s)",
        z: "Gyroscope z (rad/s)"
    })


def combine(acc, gyro):
    acc = prepare_acc(acc)
    gyro = prepare_gyro(gyro)

    acc["Time (s)"] = pd.to_numeric(acc["Time (s)"], errors="coerce")
    gyro["Time (s)"] = pd.to_numeric(gyro["Time (s)"], errors="coerce")

    acc = acc.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)
    gyro = gyro.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)

    if acc.empty:
        raise ValueError("Accelerometer data is empty after time conversion.")
    if gyro.empty:
        raise ValueError("Gyroscope data is empty after time conversion.")

    return pd.merge_asof(
        acc,
        gyro,
        on="Time (s)",
        direction="nearest",
        tolerance=0.05
    )


def load_input_dataframe(mode, combined_file, acc_file, gyro_file):
    if mode == "Combined CSV":
        if combined_file is None:
            raise ValueError("Please upload a combined CSV file.")
        return pd.read_csv(combined_file)

    if acc_file is None or gyro_file is None:
        raise ValueError("Please upload both accelerometer and gyroscope CSV files.")

    acc_df = pd.read_csv(acc_file)
    gyro_df = pd.read_csv(gyro_file)
    return combine(acc_df, gyro_df)

# ==============================
# MODEL HELPERS
# ==============================
def reload_model_and_metadata():
    st.session_state.model = load_model()
    st.session_state.metadata = load_metadata()


def backup_existing_model():
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, BACKUP_MODEL_PATH)
        return BACKUP_MODEL_PATH
    return None


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


def retrain(df, overwrite=False):
    ok, message = get_training_safeguards(df)
    if not ok:
        return message

    X = df[get_feature_columns()]
    y = df[get_target_columns()]

    if X.empty:
        return "Training skipped: feature matrix is empty."

    if not overwrite and os.path.exists(MODEL_PATH):
        return "Enable overwrite to replace existing saved model."

    backup_path = backup_existing_model()

    model = get_model()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    reload_model_and_metadata()

    if backup_path:
        return f"Model retrained successfully. Previous model backed up as: {backup_path}"

    return "Model retrained successfully."

# ==============================
# EVALUATION
# ==============================
def evaluate(df):
    X = df[get_feature_columns()]
    y = df[get_target_columns()]

    if len(df) < 30:
        return "Evaluation skipped: need at least 30 rows."

    if X.empty:
        return "Evaluation skipped: feature matrix is empty."

    if "source_file" not in df.columns or df["source_file"].nunique() < 2:
        return "Evaluation skipped: need at least 2 source groups."

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, df["source_file"]))

    model = get_model()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    preds = model.predict(X.iloc[test_idx])
    preds = pd.DataFrame(preds, columns=get_target_columns(), index=y.iloc[test_idx].index)

    out = []
    out.append(f"Training rows: {len(train_idx)}")
    out.append(f"Testing rows: {len(test_idx)}")
    out.append(f"Train groups: {df['source_file'].iloc[train_idx].nunique()}")
    out.append(f"Test groups: {df['source_file'].iloc[test_idx].nunique()}")
    out.append("")

    for col in get_target_columns():
        acc = accuracy_score(y.iloc[test_idx][col], preds[col])
        out.append(f"{col} Accuracy: {acc * 100:.2f}%")

    exact_match = (preds[get_target_columns()] == y.iloc[test_idx][get_target_columns()]).all(axis=1).mean()
    out.append("")
    out.append(f"Exact Match Accuracy: {exact_match * 100:.2f}%")

    return "\n".join(out)

# ==============================
# PREDICTION
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


def process(df):
    df = clean_sensor_data(df)
    df = create_features(df)

    if st.session_state.model is None:
        raise ValueError("No trained model found. Run train_model.py first or retrain.")

    X = df[get_feature_columns()]
    preds = st.session_state.model.predict(X)
    pred_df = pd.DataFrame(preds, columns=get_target_columns(), index=df.index)

    conf_df = get_model_confidence_scores(st.session_state.model, X)
    if not conf_df.empty:
        pred_df = pd.concat([pred_df, conf_df], axis=1)

    final = {
        col: pred_df[col].tail(25).mode().iloc[0]
        for col in get_target_columns()
    }

    return df, pred_df, final


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

# ==============================
# MODEL INSIGHTS
# ==============================
def display_saved_confusion_matrices():
    metadata = st.session_state.metadata
    evaluation = metadata.get("evaluation", {})

    shown_any = False
    for target in get_target_columns():
        info = evaluation.get(target, {})
        labels = info.get("confusion_matrix_labels", [])
        cm = info.get("confusion_matrix", [])

        if labels and cm:
            shown_any = True
            st.write(f"### Confusion Matrix: {target}")
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            st.dataframe(cm_df, use_container_width=True)

    if not shown_any:
        st.info("No saved confusion matrices found. Run the updated train_model.py first.")


def display_feature_importance():
    metadata = st.session_state.metadata
    fi = metadata.get("feature_importance", {})

    shown_any = False
    for target in get_target_columns():
        rows = fi.get(target, [])
        if rows:
            shown_any = True
            st.write(f"### Feature Importance: {target}")
            fi_df = pd.DataFrame(rows).sort_values("importance", ascending=False)

            top_df = fi_df.head(10).sort_values("importance", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_df["feature"], top_df["importance"])
            ax.set_title(f"Top Features for {target}")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(fi_df, use_container_width=True)

    if not shown_any:
        st.info("No saved feature importance found. Run the updated train_model.py first.")


def show_metadata_summary():
    metadata = st.session_state.metadata
    if not metadata:
        st.info("No metadata found. Run the updated train_model.py first.")
        return

    evaluation = metadata.get("evaluation", {})
    summary = {
        "Model Path": metadata.get("model_output_path", "N/A"),
        "Total Rows": metadata.get("total_rows", "N/A"),
        "Total Features": metadata.get("total_features", "N/A"),
        "Test Size": metadata.get("test_size", "N/A"),
        "Random State": metadata.get("random_state", "N/A"),
        "Exact Match Accuracy": evaluation.get("exact_match_accuracy", "N/A")
    }
    st.json(summary)

# ==============================
# UI
# ==============================
st.title("🧠 Human Activity + Fatigue Detection")

mode = st.radio(
    "Input Mode",
    ["Combined CSV", "Separate Accelerometer + Gyroscope"],
    horizontal=True
)

combined_file = None
acc_file = None
gyro_file = None

if mode == "Combined CSV":
    combined_file = st.file_uploader("Upload Combined CSV", type=["csv"])
else:
    acc_file = st.file_uploader("Upload Accelerometer CSV", type=["csv"])
    gyro_file = st.file_uploader("Upload Gyroscope CSV", type=["csv"])

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Retrain", "Evaluation", "Model Insights"])

# ==============================
# PREDICTION TAB
# ==============================
with tab1:
    if st.button("Run Prediction", use_container_width=True):
        try:
            df_source = load_input_dataframe(mode, combined_file, acc_file, gyro_file)
            df, pred_df, final = process(df_source)
            diag = create_feature_diagnostics(df)
            fatigue_reason = create_fatigue_reason(diag, final["fatigue"])

            st.success("Prediction completed.")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Final Activity", final["activity"])
            col_b.metric("Final Intensity", final["intensity"])
            col_c.metric("Final Fatigue", final["fatigue"])

            st.write(f"**Fatigue Basis:** {fatigue_reason}")

            st.write("### Prediction Preview")
            st.dataframe(pred_df.head(20), use_container_width=True)

            st.write("### Generated Label Distribution")
            label_dist = generate_labels(df.copy())[get_target_columns()].apply(lambda col: col.value_counts()).fillna(0)
            st.dataframe(label_dist, use_container_width=True)

            st.write("### Activity Distribution")
            counts = pred_df["activity"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Activity Distribution")
            ax.axis("equal")
            st.pyplot(fig)

            st.write("### Fatigue Distribution")
            fatigue_counts = pred_df["fatigue"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(fatigue_counts.index, fatigue_counts.values)
            ax2.set_title("Fatigue State Distribution")
            st.pyplot(fig2)

            st.write("### Acceleration Magnitude")
            st.line_chart(df["acc_mag"])

            st.write("### Gyroscope Magnitude")
            st.line_chart(df["gyro_mag"])

            st.write("### Motion Energy")
            st.line_chart(df["motion_energy"])

            st.write("### Cumulative Load")
            st.line_chart(df["cumulative_load"])

        except Exception as e:
            st.error(str(e))

# ==============================
# RETRAIN TAB
# ==============================
with tab2:
    overwrite = st.checkbox("Overwrite existing saved model", value=False)

    if st.button("Retrain Model", use_container_width=True):
        try:
            df_source = load_input_dataframe(mode, combined_file, acc_file, gyro_file)
            df = clean_sensor_data(df_source)
            df = create_features(df)
            df = generate_labels(df)
            df["source_file"] = "uploaded_training_data"

            msg = retrain(df, overwrite=overwrite)
            st.success(msg)

        except Exception as e:
            st.error(str(e))

# ==============================
# EVALUATION TAB
# ==============================
with tab3:
    if st.button("Evaluate Model", use_container_width=True):
        try:
            df_source = load_input_dataframe(mode, combined_file, acc_file, gyro_file)
            df = clean_sensor_data(df_source)
            df = create_features(df)
            df = generate_labels(df)

            n = len(df)
            if n >= 100:
                segment_size = max(25, n // 5)
                df["source_file"] = [f"segment_{i // segment_size}" for i in range(n)]
            else:
                df["source_file"] = "single_segment"

            result = evaluate(df)
            st.text(result)

        except Exception as e:
            st.error(str(e))

# ==============================
# MODEL INSIGHTS TAB
# ==============================
with tab4:
    st.subheader("Saved Model Metadata")
    if st.button("Refresh Metadata", use_container_width=True):
        reload_model_and_metadata()
        st.success("Metadata reloaded.")

    show_metadata_summary()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Show Confusion Matrices", use_container_width=True):
            display_saved_confusion_matrices()

    with col2:
        if st.button("Show Feature Importance", use_container_width=True):
            display_feature_importance()