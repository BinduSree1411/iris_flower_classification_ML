# Iris Flower Classification – Machine Learning Project

## 1. Project Overview

The Iris Flower Classification project is a machine learning application that predicts the species of an Iris flower based on its sepal and petal measurements.

The project uses Python and machine learning algorithms to classify flowers into three species: *Iris Setosa*, *Iris Versicolor*, and *Iris Virginica*. It provides an interactive web application built using Streamlit, allowing users to enter flower measurements, view predictions, explore the dataset, and evaluate different machine learning models.

## 2. Objectives

* Build a machine learning model to classify Iris flower species.
* Compare multiple machine learning algorithms.
* Evaluate model performance using standard classification metrics.
* Visualize patterns and relationships in the dataset.
* Provide an interactive interface for making predictions.
* Support predictions for multiple flowers using CSV file uploads.
* Explore feature importance to understand which measurements influence predictions.

## 3. Dataset

The project uses the Iris dataset, a commonly used dataset for learning and experimenting with machine learning classification.

The dataset contains four numerical features:

* **Sepal Length:** Length of the sepal.
* **Sepal Width:** Width of the sepal.
* **Petal Length:** Length of the petal.
* **Petal Width:** Width of the petal.

The target variable is the flower species, which belongs to one of three classes:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

## 4. Technologies Used

* **Python:** Main programming language.
* **Pandas:** Data manipulation and analysis.
* **NumPy:** Numerical operations.
* **Scikit-learn:** Machine learning algorithms, preprocessing, and model evaluation.
* **Streamlit:** Interactive web application.
* **Matplotlib and Seaborn:** Data visualization, where used.
* **Pickle:** Saving and loading trained models and related objects.
* **Git and GitHub:** Version control and project sharing.

## 5. Machine Learning Algorithms

The project compares the following six classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Bagging Classifier
6. AdaBoost Classifier

These algorithms are evaluated to understand and compare their classification performance.

## 6. Project Features

### Home and Prediction Page

* Enter the four flower measurements.
* Predict the flower species.
* View prediction confidence and class probabilities when supported by the selected model.
* View prediction history during the current session.
* Download prediction history as a CSV file.
* Clear the prediction history.

### Batch Prediction

* Upload a CSV file containing flower measurements.
* Generate predictions for multiple flowers.
* Export the prediction results as a CSV file.

### Exploratory Data Analysis (EDA) Dashboard

* Filter data by flower species.
* View dataset samples and data-quality information.
* Explore missing values and duplicate records.
* Visualize feature distributions, boxplots, scatterplots, and correlations.

### Model Comparison

* Compare the six classification algorithms.
* Examine evaluation metrics and visualizations.
* View confusion matrices and classification reports.
* Download comparison results where supported.

### Model Performance

* Evaluate models using a train-test split.
* Calculate accuracy, precision, recall, and weighted F1-score.
* Rank models based on weighted F1-score.
* Visualize and download performance metrics.
* Save the best evaluated model for use by the application when available.

### Explainable AI

* Examine feature importance using permutation feature importance.
* Understand how changes in input features may affect model performance.
* Explore which flower measurements contribute useful predictive information.

**Note:** The Explainable AI page trains a separate Random Forest model for its feature-importance analysis. Its results may differ from those of the model selected for predictions.

## 7. Project Structure

```text
Iris_Classification/
│
├── app.py
├── pages/
│   ├── 1_EDA_Dashboard.py
│   ├── 2_Model_Comparison.py
│   ├── 3_Model_Performance.py
│   ├── 4_Explainable_AI.py
│   └── 5_Project_Information.py
│
├── data/
│   └── iris.csv
│
├── outputs/
├── iris.csv
├── model.pkl
├── scaler.pkl
├── all_models.pkl
└── requirements.txt
```

The exact files may vary depending on the saved models and outputs available in the project folder.

## 8. Installation and Execution

### Prerequisites

* Python installed on your computer.
* pip package manager.
* A terminal or command prompt.

### Step 1: Clone the Repository

Replace the placeholder URL with the URL of your GitHub repository.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Iris_Classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL that you can open in your web browser.

## 9. Sample Input

The application accepts four numerical input features. For batch prediction, create a CSV file with the following column names:

```csv
sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9
```

Use the same column names and measurement units expected by the application.

## 10. Model Evaluation

The project evaluates classification models using the following metrics:

* **Accuracy:** Proportion of predictions that are correct.
* **Precision:** Proportion of positive predictions that are correct, calculated according to the selected averaging method.
* **Recall:** Proportion of actual class instances correctly identified.
* **Weighted F1-score:** Weighted combination of precision and recall across classes.

Actual evaluation results should be recorded after running the model evaluation page. No performance scores are stated here because they should reflect the results obtained from the application.

## 11. Limitations

* The application is based on the Iris dataset and may not generalize to other flower species.
* Predictions depend on the quality and range of the input measurements.
* Model performance may vary depending on the algorithm and evaluation split.
* Prediction confidence should not automatically be interpreted as a guarantee that a prediction is correct.
* The application is intended for educational and experimental purposes.

## 12. Future Enhancements

* Deploy the application online.
* Improve input validation and error handling.
* Add additional flower datasets and classification models.
* Improve model selection and reproducibility.
* Add automated testing and continuous integration.
* Enhance the user interface and visualizations.

## 13. Conclusion

The Iris Flower Classification project demonstrates the application of machine learning to a multi-class classification problem. It combines data exploration, model comparison, performance evaluation, prediction, batch processing, and feature-importance analysis in an interactive Streamlit application.

The project provides practical experience with Python, machine learning workflows, data visualization, model evaluation, and application development.

## 14. Author

**Project:** Iris Flower Classification – Machine Learning Project

**Domain:** Machine Learning

**Application Framework:** Streamlit

**Repository:** https://github.com/BinduSree1411/iris_flower_classification_ML.git

## 🚀 Live Demo
Try the application here: [Iris Flower Classification](https://iris-ml-classifier-bindusree.streamlit.app/)


