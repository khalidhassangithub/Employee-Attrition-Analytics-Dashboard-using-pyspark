# =============================================================
# ⭐ Professional Streamlit Dashboard for Employee Attrition Analytics (Live Filters)
# =============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stSidebar {background-color: #F3F6FA;}
    .big-font {font-size:24px !important; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 1️⃣ Load Data (PySpark → Pandas)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    spark = SparkSession.builder.appName("AttritionDashboard").getOrCreate()
    data = spark.read.csv(
        "/Users/khalidoscar/Downloads/Employers_data.csv",
        header=True,
        inferSchema=True
    ).dropna()
    df = data.toPandas()
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

df = load_data()

# ------------------------------------------------------------
# 2️⃣ Sidebar – LIVE Filters
# ------------------------------------------------------------
st.sidebar.header("🔍 Live Data Filters")

def apply_filters(df):
    department = st.sidebar.multiselect(
        "Department",
        options=sorted(df["Department"].unique()),
        default=sorted(df["Department"].unique())
    )

    gender = st.sidebar.multiselect(
        "Gender",
        options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique())
    )

    age_range = st.sidebar.slider(
        "Age Range",
        int(df["Age"].min()), int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max()))
    )

    salary_range = st.sidebar.slider(
        "Salary Range",
        int(df["Salary"].min()), int(df["Salary"].max()),
        (int(df["Salary"].min()), int(df["Salary"].max()))
    )

    exp_range = st.sidebar.slider(
        "Experience (Years)",
        int(df["Experience_Years"].min()), int(df["Experience_Years"].max()),
        (int(df["Experience_Years"].min()), int(df["Experience_Years"].max()))
    )

    # ❗ Reset Filters Button
    if st.sidebar.button("🔄 Reset Filters"):
        st.experimental_rerun()

    # LIVE FILTERING
    filtered = df[
        (df["Department"].isin(department)) &
        (df["Gender"].isin(gender)) &
        (df["Age"].between(age_range[0], age_range[1])) &
        (df["Salary"].between(salary_range[0], salary_range[1])) &
        (df["Experience_Years"].between(exp_range[0], exp_range[1]))
    ]

    return filtered


filtered_df = apply_filters(df)

# ------------------------------------------------------------
# 3️⃣ Page Title
# ------------------------------------------------------------
st.title("📊 Employee Attrition Analytics Dashboard")
st.markdown("<p class='big-font'>Live and interactive analysis of employee attrition patterns.</p>", unsafe_allow_html=True)
st.write("---")

# ------------------------------------------------------------
# 4️⃣ KPIs Section (Auto Updates)
# ------------------------------------------------------------
st.subheader("📌 Key Performance Indicators (LIVE)")

col1, col2, col3, col4 = st.columns(4)

total_employees = len(filtered_df)
left_employees = len(filtered_df[filtered_df["Attrition"] == "Yes"])
attrition_rate = round((left_employees / total_employees) * 100, 2)
avg_salary = round(filtered_df["Salary"].mean(), 2)

col1.metric("Total Employees", total_employees)
col2.metric("Employees Left", left_employees)
col3.metric("Attrition Rate", f"{attrition_rate}%")
col4.metric("Average Salary", f"${avg_salary}")

st.write("---")

# ------------------------------------------------------------
# 5️⃣ Visualizations (LIVE Charts)
# ------------------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.write("### 🔥 Attrition Distribution")

    attr_count = filtered_df["Attrition"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(attr_count.values, labels=attr_count.index, autopct="%1.1f%%")
    st.pyplot(fig)

with right_col:
    st.write("### 🏢 Department-wise Attrition")

    dept_attr = filtered_df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    dept_attr.plot(kind="bar", ax=ax)
    plt.title("Attrition by Department")
    st.pyplot(fig)

st.write("---")

# ------------------------------------------------------------
# 6️⃣ Age + Salary Distribution (LIVE)
# ------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.write("### 👤 Age Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(filtered_df["Age"], bins=15)
    st.pyplot(fig)

with colB:
    st.write("### 💰 Salary vs Attrition")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered_df, x="Attrition", y="Salary", ax=ax)
    st.pyplot(fig)

st.write("---")

# ------------------------------------------------------------
# 7️⃣ Experience vs Salary Scatter Plot (LIVE)
# ------------------------------------------------------------
st.write("### 📈 Experience vs Salary")

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(
    data=filtered_df,
    x="Experience_Years", y="Salary",
    hue="Attrition", ax=ax
)
st.pyplot(fig)

st.write("---")

# ------------------------------------------------------------
# 8️⃣ Correlation Heatmap (LIVE)
# ------------------------------------------------------------
st.write("### 📊 Correlation Heatmap")

numeric_cols = filtered_df.select_dtypes(include=["int64", "float64"])
if numeric_cols.shape[1] > 0:
    corr_matrix = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠ No numeric columns available.")