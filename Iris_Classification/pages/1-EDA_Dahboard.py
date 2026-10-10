
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Iris EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Iris Dataset — EDA Dashboard")
st.write(
    "Explore the dataset, understand species distribution, "
    "analyze feature relationships, and identify data-quality issues."
)

# Locate the dataset relative to this file
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Please place iris.csv inside the project's data folder.")
    st.stop()

df = load_data(str(DATA_PATH))

st.sidebar.header("Dashboard Controls")
st.sidebar.write("Choose which species to explore.")

if "species" in df.columns:
    species_options = sorted(df["species"].dropna().unique().tolist())
    selected_species = st.sidebar.multiselect(
        "Filter by species",
        species_options,
        default=species_options
    )
    filtered_df = df[df["species"].isin(selected_species)]
else:
    st.error("A 'species' column was not found. Check your CSV column names.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

# Dataset overview
st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Duplicate Rows", int(df.duplicated().sum()))

with st.expander("View dataset preview"):
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("View column information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

# Species distribution
st.header("2. Species Distribution")

fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="species", hue="species", legend=False, ax=ax)
ax.set_title("Number of Flowers per Species")
ax.set_xlabel("Species")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

# Numerical feature selection
numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

if numeric_cols:
    st.header("3. Feature Analysis")

    feature = st.selectbox(
        "Choose a numerical feature",
        numeric_cols
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{filtered_df[feature].mean():.2f}")
    c2.metric("Median", f"{filtered_df[feature].median():.2f}")
    c3.metric("Standard Deviation", f"{filtered_df[feature].std():.2f}")

    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x=feature, hue="species",
                 kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="species", y=feature, hue="species",
                legend=False, ax=ax)
    ax.set_title(f"{feature} by Species")
    st.pyplot(fig)
    plt.close(fig)

# Feature relationships
st.header("4. Feature Relationships")

if len(numeric_cols) >= 2:
    c1, c2 = st.columns(2)

    with c1:
        x_feature = st.selectbox(
            "X-axis feature",
            numeric_cols,
            index=0
        )

    with c2:
        y_feature = st.selectbox(
            "Y-axis feature",
            numeric_cols,
            index=min(2, len(numeric_cols) - 1)
        )

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=filtered_df,
        x=x_feature,
        y=y_feature,
        hue="species",
        s=80,
        ax=ax
    )
    ax.set_title(f"{x_feature} vs {y_feature}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm",
                fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)
    plt.close(fig)

# Data quality
st.header("5. Data Quality")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isna().sum().values
})
st.dataframe(missing_df, use_container_width=True)

if df.duplicated().sum() > 0:
    st.warning(
        f"The dataset contains {df.duplicated().sum()} duplicate rows. "
        "Review them before deciding whether to remove them."
    )
else:
    st.success("No duplicate rows detected.")

# Download filtered data
st.header("6. Download Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Filtered Dataset (CSV)",
    data=csv_data,
    file_name="iris_filtered.csv",
    mime="text/csv"
)

st.caption("Iris Classification Project | Exploratory Data Analysis")