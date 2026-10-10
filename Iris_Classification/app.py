
from pathlib import Path
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)


# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #172554;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    h1 {
        color: #172554;
        font-weight: 800;
    }

    h2, h3 {
        color: #1e3a8a;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dbeafe;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.06);
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        border: none;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stFileUploader"] {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px dashed #93c5fd;
    }

    hr {
        border-color: #dbeafe;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODELS_PATH = BASE_DIR / "all_models.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


# --------------------------------------------------
# LOAD TRAINED MODELS AND SCALER
# --------------------------------------------------

@st.cache_resource
def load_models():
    missing_files = [
        str(path.name)
        for path in [MODELS_PATH, SCALER_PATH]
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required file(s) were not found "
            f"beside app.py: {', '.join(missing_files)}. "
            "Please upload these files to the same folder as app.py."
        )

    with MODELS_PATH.open("rb") as file:
        loaded_models = pickle.load(file)

    with SCALER_PATH.open("rb") as file:
        loaded_scaler = pickle.load(file)

    if not isinstance(loaded_models, dict) or not loaded_models:
        raise ValueError(
            "all_models.pkl must contain a non-empty dictionary "
            "of trained models."
        )

    return loaded_models, loaded_scaler


try:
    models, scaler = load_models()

except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.stop()


# --------------------------------------------------
# PREDICTION HELPER
# --------------------------------------------------

def prepare_input(data):
    """
    Prepare measurements for models trained on scaled features.

    This assumes the saved models in all_models.pkl were trained
    using the same scaler saved in scaler.pkl.
    """
    return scaler.transform(data[FEATURE_COLUMNS])


def predict_with_model(selected_model, measurements):
    """Predict species and return probabilities when supported."""

    prediction_input = prepare_input(measurements)

    predictions = selected_model.predict(prediction_input)

    probabilities = None
    confidence = None
    classes = None

    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(
            prediction_input
        )

        classes = getattr(selected_model, "classes_", None)

        if classes is not None:
            confidence_values = []

            for prediction, row in zip(predictions, probabilities):
                matching_indices = [
                    index
                    for index, class_name in enumerate(classes)
                    if class_name == prediction
                ]

                if matching_indices:
                    confidence_values.append(
                        float(row[matching_indices[0]]) * 100
                    )
                else:
                    confidence_values.append(None)

            confidence = confidence_values

    return predictions, probabilities, classes, confidence


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# --------------------------------------------------
# HOME PAGE INTRODUCTION
# --------------------------------------------------

st.title("🌸 Iris Flower Classification")

st.markdown("### An Interactive Machine Learning Application")

st.write(
    "Explore the Iris dataset, compare machine learning models, "
    "predict flower species, and analyze prediction results "
    "through an interactive dashboard."
)

st.divider()

st.subheader("🚀 Project Overview")

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown("### 🌼")
    st.markdown("**Flower Prediction**")
    st.caption(
        "Predict a flower species using its four measurements."
    )

with card2:
    st.markdown("### 📊")
    st.markdown("**Model Evaluation**")
    st.caption(
        "Compare six machine learning algorithms."
    )

with card3:
    st.markdown("### 📁")
    st.markdown("**Data Analysis**")
    st.caption(
        "Explore dataset patterns and export prediction results."
    )

st.divider()

st.subheader("🔬 How It Works")

st.markdown("""
1. **Enter measurements:** Sepal length, sepal width, petal length, and petal width.
2. **Choose a model:** Select one of the available machine learning algorithms.
3. **Get a prediction:** View the predicted species and estimated confidence.
4. **Explore results:** Review prediction history or analyze multiple flowers using CSV upload.
""")

st.divider()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("🤖 Select a Machine Learning Model")

model_name = st.selectbox(
    "Choose a Machine Learning Model",
    options=list(models.keys())
)

# Always use the model selected by the user.
model = models[model_name]

st.caption(f"Selected model: {model_name}")


# --------------------------------------------------
# INPUT FLOWER MEASUREMENTS
# --------------------------------------------------

st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2,
        step=0.1
    )


# --------------------------------------------------
# SINGLE FLOWER PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Species", type="primary"):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=FEATURE_COLUMNS
    )

    try:
        predictions, probabilities, classes, confidence_values = (
            predict_with_model(model, input_data)
        )

        prediction = predictions[0]

        confidence = (
            confidence_values[0]
            if confidence_values is not None
            else None
        )

        record = {
            "Date & Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Model": model_name,
            "Sepal Length (cm)": sepal_length,
            "Sepal Width (cm)": sepal_width,
            "Petal Length (cm)": petal_length,
            "Petal Width (cm)": petal_width,
            "Predicted Species": str(prediction),
            "Confidence (%)": (
                round(confidence, 2)
                if confidence is not None
                else None
            )
        }

        st.session_state.prediction_history.append(record)

        st.subheader("🌼 Prediction Result")

        st.success(f"Predicted Species: {prediction}")

        if confidence is not None:
            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )
        else:
            st.info(
                "This model does not provide prediction probabilities."
            )

        if probabilities is not None and classes is not None:

            probability_df = pd.DataFrame({
                "Species": classes,
                "Probability (%)": probabilities[0] * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Prediction Probabilities")

            for _, row in probability_df.iterrows():
                st.write(
                    f"{row['Species']}: "
                    f"{row['Probability (%)']:.2f}%"
                )

                st.progress(
                    min(
                        1.0,
                        max(
                            0.0,
                            float(row["Probability (%)"]) / 100
                        )
                    )
                )

            fig, ax = plt.subplots()

            ax.bar(
                probability_df["Species"].astype(str),
                probability_df["Probability (%)"]
            )

            ax.set_ylabel("Probability (%)")
            ax.set_xlabel("Species")
            ax.set_ylim(0, 100)
            ax.set_title(
                f"{model_name} - Prediction Probabilities"
            )

            plt.xticks(rotation=15)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        st.info("Prediction added to your session history.")

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()
st.header("📚 Prediction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    total, unique, confident = st.columns(3)

    total.metric(
        "Total Predictions",
        len(history_df)
    )

    unique.metric(
        "Species Predicted",
        history_df["Predicted Species"].nunique()
    )

    available_confidences = history_df[
        "Confidence (%)"
    ].dropna()

    if not available_confidences.empty:
        confident.metric(
            "Average Confidence",
            f"{available_confidences.mean():.2f}%"
        )
    else:
        confident.metric(
            "Average Confidence",
            "N/A"
        )

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="iris_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No predictions recorded yet. Enter flower measurements "
        "and click 'Predict Species' to start your history."
    )


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

st.divider()
st.header("📂 Batch Prediction")

st.write(
    "Upload a CSV file containing multiple flower measurements "
    "to predict their species together."
)

st.caption(
    "Required columns: sepal_length, sepal_width, "
    "petal_length, petal_width"
)

uploaded_file = st.file_uploader(
    "Upload flower measurements CSV",
    type=["csv"],
    key="batch_csv"
)

if uploaded_file is not None:

    try:
        batch_df = pd.read_csv(uploaded_file)

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "Missing columns: "
                + ", ".join(missing_columns)
            )

        elif batch_df.empty:

            st.warning(
                "The uploaded CSV contains no flower data."
            )

        else:

            measurements = batch_df[
                FEATURE_COLUMNS
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if measurements.isna().any().any():

                st.error(
                    "Some measurements are missing or invalid. "
                    "Please correct your CSV and upload it again."
                )

            elif (measurements < 0).any().any():

                st.error(
                    "Measurements cannot be negative."
                )

            else:

                predictions, probabilities, classes, confidence_values = (
                    predict_with_model(model, measurements)
                )

                results_df = batch_df.copy()

                results_df["Predicted Species"] = predictions

                if confidence_values is not None:
                    results_df["Confidence (%)"] = [
                        round(value, 2)
                        if value is not None
                        else None
                        for value in confidence_values
                    ]

                st.success(
                    f"Predictions completed for "
                    f"{len(results_df)} flowers using {model_name}!"
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                csv_results = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Batch Results (CSV)",
                    data=csv_results,
                    file_name="iris_batch_predictions.csv",
                    mime="text/csv",
                    key="batch_download"
                )

    except Exception as error:
        st.error(
            f"Error processing the CSV: {error}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Iris Classification | Machine Learning Project"
)
