
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Professional UI styling
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

# Load trained models and scaler
@st.cache_resource
def load_models():
    with open("all_models.pkl", "rb") as file:
        models = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return models, scaler

try:
    models, scaler = load_models()
except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.stop()
# ADD THE NEW CODE HERE
from pathlib import Path

BEST_MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

@st.cache_resource
def load_best_model():
    with open(BEST_MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    best_model_details = load_best_model()
except Exception as error:
    st.warning(f"Could not load the saved best model: {error}")
    best_model_details = None

# Store history for the current app session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# Home page introduction
st.title("🌸 Iris Flower Classification")
st.markdown(
    "### An Interactive Machine Learning Application"
)

st.write(
    "Explore the Iris dataset, compare machine learning models, "
    "predict flower species, and analyze prediction results "
    "through an interactive dashboard."
)

st.divider()

# Project overview cards
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

# Model selection
model_name = st.selectbox(
    "Choose a Machine Learning Model",
    list(models.keys())
)
model = models[model_name]

# Input measurements
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

# Predict

# Predict
if st.button("🔍 Predict Species", type="primary"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    try:
        if best_model_details is not None:
            prediction_model = best_model_details["model"]
            prediction_model_name = best_model_details["model_name"]
            prediction_input = input_data
        else:
            prediction_model = model
            prediction_model_name = model_name
            prediction_input = scaler.transform(input_data)

        prediction = prediction_model.predict(prediction_input)[0]

        confidence = None
        probabilities = None

        if hasattr(prediction_model, "predict_proba"):
            probabilities = prediction_model.predict_proba(
                prediction_input
            )[0]
            classes = prediction_model.classes_
            predicted_index = list(classes).index(prediction)
            confidence = float(probabilities[predicted_index]) * 100

        record = {
            "Date & Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Model": prediction_model_name,
            "Sepal Length (cm)": sepal_length,
            "Sepal Width (cm)": sepal_width,
            "Petal Length (cm)": petal_length,
            "Petal Width (cm)": petal_width,
            "Predicted Species": str(prediction),
            "Confidence (%)": (
                round(confidence, 2)
                if confidence is not None else None
            )
        }

        st.session_state.prediction_history.append(record)

        st.subheader("🌼 Prediction Result")
        st.success(f"Predicted Species: {prediction}")

        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

        if probabilities is not None:
            probability_df = pd.DataFrame({
                "Species": classes,
                "Probability (%)": probabilities * 100
            }).sort_values("Probability (%)", ascending=False)

            st.subheader("Prediction Probabilities")

            for _, row in probability_df.iterrows():
                st.write(
                    f"{row['Species']}: "
                    f"{row['Probability (%)']:.2f}%"
                )
                st.progress(
                    min(1.0, max(
                        0.0,
                        float(row["Probability (%)"]) / 100
                    ))
                )

            fig, ax = plt.subplots()
            ax.bar(
                probability_df["Species"],
                probability_df["Probability (%)"]
            )
            ax.set_ylabel("Probability (%)")
            ax.set_xlabel("Species")
            ax.set_ylim(0, 100)
            ax.set_title(
                f"{prediction_model_name} - Prediction Probabilities"
            )
            plt.xticks(rotation=15)
            st.pyplot(fig)
            plt.close(fig)

        st.info("Prediction added to your session history.")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
# Prediction history
st.divider()
st.header("📚 Prediction History")

history = st.session_state.prediction_history

if history:
    history_df = pd.DataFrame(history)

    total, unique, confident = st.columns(3)
    total.metric("Total Predictions", len(history))
    unique.metric(
        "Species Predicted",
        history_df["Predicted Species"].nunique()
    )

    available_confidences = history_df["Confidence (%)"].dropna()
    if not available_confidences.empty:
        confident.metric(
            "Average Confidence",
            f"{available_confidences.mean():.2f}%"
        )
    else:
        confident.metric("Average Confidence", "N/A")

    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

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

st.divider()
st.caption(
    "Iris Classification | Machine Learning Project"
)

# Batch Prediction Feature
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

        required_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in batch_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing columns: " + ", ".join(missing_columns)
            )

        elif batch_df.empty:
            st.warning("The uploaded CSV contains no flower data.")

        else:
            measurements = batch_df[required_columns].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if measurements.isna().any().any():
                st.error(
                    "Some measurements are missing or invalid. "
                    "Please correct your CSV."
                )

            elif (measurements < 0).any().any():
                st.error("Measurements cannot be negative.")

            
            else:
                if best_model_details is not None:
                    prediction_model = best_model_details["model"]
                    prediction_model_name = best_model_details["model_name"]
                    prediction_input = measurements
                else:
                    prediction_model = model
                    prediction_model_name = model_name
                    prediction_input = scaler.transform(measurements)

                predictions = prediction_model.predict(prediction_input)

                results_df = batch_df.copy()
                results_df["Predicted Species"] = predictions

                if hasattr(prediction_model, "predict_proba"):
                    probabilities = prediction_model.predict_proba(
                        prediction_input
                    )
                    results_df["Confidence (%)"] = (
                        probabilities.max(axis=1) * 100
                    ).round(2)
                st.success(
                    f"Predictions completed for {len(results_df)} flowers!"
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
        st.error(f"Error processing the CSV: {error}")
