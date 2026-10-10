
import streamlit as st

st.set_page_config(
    page_title="Project Information",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Iris Flower Classification")
st.markdown("### Project Documentation")

st.write(
    "An interactive machine learning application that predicts "
    "Iris flower species from four flower measurements."
)

st.divider()

st.header("🎯 Project Objectives")

st.markdown("""
- Predict the species of an Iris flower using machine learning.
- Compare the performance of six classification algorithms.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Explain the importance of input features.
- Support predictions for individual flowers and uploaded CSV files.
- Maintain prediction history and allow results to be downloaded.
""")

st.header("📊 Dataset Description")

st.write(
    "The Iris dataset contains measurements of three flower species: "
    "Iris setosa, Iris versicolor, and Iris virginica."
)

st.markdown("""
**Input features**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target:** Iris flower species
""")

st.header("🤖 Machine Learning Algorithms")

st.markdown("""
1. **Logistic Regression** — estimates class probabilities using a linear model.
2. **K-Nearest Neighbors (KNN)** — classifies a flower using nearby examples.
3. **Support Vector Machine (SVM)** — finds decision boundaries between classes.
4. **Decision Tree** — predicts using a sequence of feature-based decisions.
5. **Bagging** — combines predictions from multiple models trained on sampled data.
6. **AdaBoost** — combines weak learners to improve classification performance.
""")

st.header("⚙️ Project Workflow")

st.markdown("""
1. Load and validate the dataset.
2. Explore the data and visualize feature patterns.
3. Split data into training and testing sets.
4. Train and evaluate the classification models.
5. Select and save the top model based on weighted F1-score.
6. Predict species from individual measurements or CSV uploads.
7. Display results, confidence estimates, and feature importance.
""")

st.header("🛠️ Technologies Used")

st.markdown("""
- **Python** — programming language
- **Pandas** — data manipulation
- **Scikit-learn** — machine learning and evaluation
- **Matplotlib** — charts and visualizations
- **Streamlit** — interactive web application
- **Pickle** — model serialization
""")

st.header("⚠️ Limitations")

st.markdown("""
- Predictions depend on the quality and representativeness of the dataset.
- Confidence values are model estimates, not guarantees of correctness.
- Feature importance describes model behavior; it does not establish causation.
- Evaluation scores can vary depending on the train-test split.
""")

st.success(
    "The project demonstrates an end-to-end machine learning workflow "
    "from data exploration to prediction and model evaluation."
)