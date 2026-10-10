
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
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
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Machine Learning Model Comparison")
st.write(
    "Compare six classification algorithms using the same "
    "train-test split and evaluation metrics."
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

@st.cache_data
def load_dataset(path):
    data = pd.read_csv(path)

    # Handle common Iris dataset column-name variations
    target_candidates = [
        col for col in data.columns
        if col.strip().lower() in ["species", "variety", "class", "target"]
    ]

    if not target_candidates:
        raise ValueError(
            "Could not find the target column. "
            f"Available columns: {data.columns.tolist()}"
        )

    target = target_candidates[0]
    data = data.dropna(subset=[target]).copy()

    X = data.drop(columns=[target])
    X = X.select_dtypes(include="number")

    # Remove identifier columns, if present
    X = X.drop(
        columns=[
            col for col in X.columns
            if col.strip().lower() in ["id", "index", "unnamed: 0"]
        ],
        errors="ignore"
    )

    valid_rows = X.notna().all(axis=1)
    X = X.loc[valid_rows]
    y = data.loc[valid_rows, target].astype(str)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found.")

    return X, y

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    X, y = load_dataset(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

if y.nunique() < 2:
    st.error("The dataset must contain at least two classes.")
    st.stop()

st.subheader("Evaluation Settings")

c1, c2 = st.columns(2)

with c1:
    test_size = st.slider(
        "Test data percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

with c2:
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42
    )

if st.button("🚀 Train and Compare All Models", type="primary"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=int(random_seed),
            stratify=y
        )

        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42)
            ),
            "KNN": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5)
            ),
            "SVM": make_pipeline(
                StandardScaler(),
                SVC(random_state=42)
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "Bagging": BaggingClassifier(
                random_state=42,
                n_estimators=50
            ),
            "Boosting": AdaBoostClassifier(
                random_state=42,
                n_estimators=50
            )
        }

        results = []
        fitted_models = {}
        predictions = {}

        progress = st.progress(0, text="Training models...")

        for i, (name, model) in enumerate(models.items()):
            start = perf_counter()
            model.fit(X_train, y_train)
            training_time = perf_counter() - start

            pred = model.predict(X_test)

            fitted_models[name] = model
            predictions[name] = pred

            results.append({
                "Model": name,
                "Accuracy (%)": accuracy_score(
                    y_test, pred
                ) * 100,
                "Precision (%)": precision_score(
                    y_test, pred,
                    average="weighted",
                    zero_division=0
                ) * 100,
                "Recall (%)": recall_score(
                    y_test, pred,
                    average="weighted",
                    zero_division=0
                ) * 100,
                "F1-Score (%)": f1_score(
                    y_test, pred,
                    average="weighted",
                    zero_division=0
                ) * 100,
                "Training Time (ms)": training_time * 1000
            })

            progress.progress(
                (i + 1) / len(models),
                text=f"Evaluating {name}..."
            )

        progress.empty()

        results_df = pd.DataFrame(results).sort_values(
            "Accuracy (%)",
            ascending=False
        ).reset_index(drop=True)

        st.session_state["comparison_results"] = results_df
        st.session_state["comparison_models"] = fitted_models
        st.session_state["comparison_predictions"] = predictions
        st.session_state["comparison_y_test"] = y_test
        st.session_state["comparison_classes"] = sorted(y.unique())

    except Exception as error:
        st.error(f"Model evaluation failed: {error}")

if "comparison_results" in st.session_state:
    results_df = st.session_state["comparison_results"]
    fitted_models = st.session_state["comparison_models"]
    predictions = st.session_state["comparison_predictions"]
    y_test = st.session_state["comparison_y_test"]
    classes = st.session_state["comparison_classes"]

    st.divider()
    st.header("1. Performance Leaderboard")

    best = results_df.iloc[0]

    a, b, c = st.columns(3)
    a.metric("Top Model", best["Model"])
    b.metric("Best Accuracy", f'{best["Accuracy (%)"]:.2f}%')
    c.metric("Models Compared", len(results_df))

    st.dataframe(
        results_df.style.format({
            "Accuracy (%)": "{:.2f}",
            "Precision (%)": "{:.2f}",
            "Recall (%)": "{:.2f}",
            "F1-Score (%)": "{:.2f}",
            "Training Time (ms)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Metrics as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="iris_model_comparison.csv",
        mime="text/csv"
    )

    st.header("2. Compare Model Scores")

    metric = st.selectbox(
        "Choose a metric",
        [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-Score (%)"
        ]
    )

    chart_data = results_df.sort_values(metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(chart_data["Model"], chart_data[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison by {metric}")
    if metric != "Training Time (ms)":
        ax.set_xlim(0, 105)
    st.pyplot(fig)
    plt.close(fig)

    st.header("3. Confusion Matrix")

    chosen_model = st.selectbox(
        "Select a model to inspect",
        results_df["Model"].tolist()
    )

    cm = confusion_matrix(
        y_test,
        predictions[chosen_model],
        labels=classes
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    ax.set_xlabel("Predicted Species")
    ax.set_ylabel("Actual Species")
    ax.set_title(f"{chosen_model} — Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.header("4. Detailed Classification Report")

    report = classification_report(
        y_test,
        predictions[chosen_model],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.header("5. Model Interpretation")

    st.write(
        f"**Highest accuracy in this run:** {best['Model']} "
        f"({best['Accuracy (%)']:.2f}%)."
    )
    st.write(
        "Accuracy measures overall correctness. Precision measures "
        "the reliability of positive predictions, recall measures "
        "how many actual examples are identified, and F1-score "
        "balances precision and recall."
    )
    st.info(
        "These results come from freshly trained models using the "
        "selected split. They may differ from your original notebook. "
        "The top model is best only for this evaluation, not necessarily "
        "for every dataset or future prediction."
    )

else:
    st.info(
        "Choose your evaluation settings and click "
        "'Train and Compare All Models' to see the results."
    )