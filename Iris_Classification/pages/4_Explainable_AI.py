
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Explainable AI",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explainable AI")
st.markdown("### Understanding Why a Model Makes Predictions")

st.write(
    "This page explains which flower measurements are most useful "
    "for predicting Iris species. We use a Random Forest model and "
    "permutation importance to measure how prediction performance "
    "changes when each feature is shuffled."
)

st.divider()

# Load the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

feature_names = [
    "Sepal length",
    "Sepal width",
    "Petal length",
    "Petal width"
]

X.columns = feature_names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

col1, col2 = st.columns(2)

with col1:
    st.metric("Test Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Test Samples", len(X_test))

st.divider()

st.subheader("📊 Feature Importance")

st.write(
    "A higher importance score means that shuffling that measurement "
    "caused a larger drop in model accuracy. It does not prove that "
    "the feature causes a particular species."
)

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring="accuracy"
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance.importances_mean,
    "Standard Deviation": importance.importances_std
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    xerr=importance_df["Standard Deviation"],
    capsize=4
)

ax.set_xlabel("Decrease in accuracy after shuffling")
ax.set_ylabel("Flower measurement")
ax.set_title("Permutation Feature Importance")
ax.grid(axis="x", linestyle="--", alpha=0.4)

st.pyplot(fig)
plt.close(fig)

st.subheader("📋 Importance Scores")
st.dataframe(
    importance_df.sort_values("Importance", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="⬇️ Download Feature Importance CSV",
    data=importance_df.sort_values(
        "Importance", ascending=False
    ).to_csv(index=False).encode("utf-8"),
    file_name="iris_feature_importance.csv",
    mime="text/csv"
)

st.info(
    "Note: This page trains its own Random Forest model for explanation. "
    "Its results may differ from predictions made by other models "
    "on the main page."
)