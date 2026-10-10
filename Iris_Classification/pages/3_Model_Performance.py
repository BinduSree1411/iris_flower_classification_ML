import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path
import time

st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")
st.write(
    "Evaluate six machine learning models on the Iris dataset "
    "and compare their performance."
)

# Locate the dataset
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    # Support common target column names
    target_candidates = ["species", "variety", "class", "target"]
    target_column = next(
        (col for col in target_candidates if col in data.columns),
        None
    )

    if target_column is None:
        raise ValueError(
            "Could not find a target column named species, "
            "variety, class, or target."
        )

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError(
            "Missing measurement columns: " + ", ".join(missing)
        )

    data = data[features + [target_column]].copy()
    data = data.dropna()

    X = data[features].apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows]
    y = data.loc[valid_rows, target_column]

    return X, y

try:
    X, y = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

st.caption(f"Dataset: {DATA_PATH.name} | Valid rows: {len(X)}")

# Settings
st.sidebar.header("Evaluation Settings")

test_size = st.sidebar.slider(
    "Test data percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=42
)

if y.nunique() < 2:
    st.error("The dataset must contain at least two species.")
    st.stop()

if y.value_counts().min() < 2:
    st.error(
        "Each species needs at least two rows for a stratified split."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_seed),
    stratify=y
)

# Define six models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=int(random_seed)),
    "Bagging": BaggingClassifier(
        n_estimators=50,
        random_state=int(random_seed)
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        random_state=int(random_seed)
    )
}

if st.button("🚀 Evaluate Models", type="primary"):
    results = []
    predictions_by_model = {}

    progress = st.progress(0)

    for index, (name, model) in enumerate(models.items()):
        start_time = time.perf_counter()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        elapsed = time.perf_counter() - start_time

        results.append({
            "Model": name,
            "Accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Precision (%)": round(
                precision_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ) * 100, 2
            ),
            "Recall (%)": round(
                recall_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ) * 100, 2
            ),
            "F1-score (%)": round(
                f1_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                ) * 100, 2
            ),
            "Training Time (seconds)": round(elapsed, 4)
        })

        predictions_by_model[name] = predictions
        progress.progress((index + 1) / len(models))

    results_df = pd.DataFrame(results).sort_values(
        "F1-score (%)", ascending=False
    ).reset_index(drop=True)
    
    # Save the best-performing model
    best_model_name = results_df.iloc[0]["Model"]
    best_model_object = models[best_model_name]

    best_model_path = (
        Path(__file__).resolve().parent.parent / "best_model.pkl"
    )

    best_model_details = {
        "model_name": best_model_name,
        "model": best_model_object,
        "f1_score": float(results_df.iloc[0]["F1-score (%)"]),
        "accuracy": float(results_df.iloc[0]["Accuracy (%)"])
    }

    with open(best_model_path, "wb") as file:
        pickle.dump(best_model_details, file)

    st.success(
        f"Best model saved successfully: {best_model_name}"
    )

    # Display results for this Streamlit session
    st.session_state["performance_results"] = results_df
    st.session_state["performance_predictions"] = predictions_by_model
    st.session_state["performance_y_test"] = y_test.reset_index(drop=True)

if "performance_results" in st.session_state:
    results_df = st.session_state["performance_results"]

    st.subheader("🏆 Performance Summary")

    best_model = results_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Top Model by F1-score", best_model["Model"])
    c2.metric("Top Model F1-score", f'{best_model["F1-score (%)"]:.2f}%')
    c3.metric("Top Model Accuracy", f'{best_model["Accuracy (%)"]:.2f}%')

    st.caption(
        "The ranking uses weighted F1-score on the selected test split. "
        "Results can vary with the random seed and split."
    )

    st.subheader("📋 All Model Metrics")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("📈 Compare Model Metrics")

    metric = st.selectbox(
        "Choose a metric",
        [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-score (%)"
        ]
    )

    chart_data = results_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(chart_data["Model"], chart_data[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📥 Export Results")
    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Performance Metrics CSV",
        data=csv_data,
        file_name="iris_model_performance.csv",
        mime="text/csv"
    )

else:
    st.info("Click 'Evaluate Models' to calculate and display metrics.")
st.subheader("💾 Saved Best Model")

if st.button("Check Saved Model"):
    best_model_path = (
        Path(__file__).resolve().parent.parent / "best_model.pkl"
    )

    if best_model_path.exists():
        with open(best_model_path, "rb") as file:
            saved_model = pickle.load(file)

        st.success("Best model loaded successfully!")
        st.write("Model name:", saved_model["model_name"])
        st.write("F1-score:", saved_model["f1_score"], "%")
        st.write("Accuracy:", saved_model["accuracy"], "%")
    else:
        st.error("best_model.pkl was not found.")