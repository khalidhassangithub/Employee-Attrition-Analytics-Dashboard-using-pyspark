# fraud_dashboard.py

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------
# Spark Session
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("Fraud Detection Dashboard") \
    .getOrCreate()

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
df = spark.read.csv(
    "/Users/khalidoscar/Downloads/creditcard.csv",
    header=True,
    inferSchema=True
)

pdf_sample = df.sample(fraction=0.10).toPandas()

# ---------------------------------------------------------
# Streamlit Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#555;'>💳 Credit Card Fraud Detection Dashboard</h1>
<p style='text-align:center;font-size:17px;color:#777;'>PySpark + Streamlit + ML Model Evaluation</p>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Train Model", "Predict Fraud"])

# ---------------------------------------------------------
# PAGE 1 — DASHBOARD
# ---------------------------------------------------------
if page == "Dashboard":

    st.markdown("### 📊 Key Metrics")

    total_txn = len(pdf_sample)
    fraud_txn = len(pdf_sample[pdf_sample["Class"] == 1])
    fraud_rate = round(fraud_txn / total_txn * 100, 4)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", total_txn)
    col2.metric("Fraud Transactions", fraud_txn)
    col3.metric("Fraud Rate (%)", fraud_rate)

    st.markdown("---")
    st.markdown("### 💰 Transaction Amount Distribution")

    fig_amount = px.histogram(
        pdf_sample,
        x="Amount",
        color="Class",
        nbins=50,
        color_discrete_map={0: "#4B8BFF", 1: "#FF4B4B"},
    )
    fig_amount.update_layout(template="simple_white")
    st.plotly_chart(fig_amount, use_container_width=True)

    st.markdown("### ⏱️ Time vs Amount Trend")

    fig_time = px.scatter(
        pdf_sample,
        x="Time",
        y="Amount",
        color="Class",
        opacity=0.6
    )
    fig_time.update_layout(template="simple_white")
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("### 📦 Fraud vs Non-Fraud Count")

    fig_bar = go.Figure(
        data=[
            go.Bar(name="Normal", x=["Normal"], y=[total_txn - fraud_txn], marker_color="#4B8BFF"),
            go.Bar(name="Fraud", x=["Fraud"], y=[fraud_txn], marker_color="#FF4B4B"),
        ]
    )
    fig_bar.update_layout(template="simple_white")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2 — TRAIN MODEL
# ---------------------------------------------------------
elif page == "Train Model":

    st.markdown("## 🤖 Train ML Models (PySpark)")

    feature_cols = [c for c in df.columns if c != "Class"]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    assembled_df = assembler.transform(df).select("features", "Class")
    train, test = assembled_df.randomSplit([0.8, 0.2], seed=42)

    model_type = st.selectbox(
        "Choose a Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosted Trees"]
    )

    if st.button("🚀 Train Model"):

        if model_type == "Logistic Regression":
            model = LogisticRegression(featuresCol="features", labelCol="Class")
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(featuresCol="features", labelCol="Class")
        elif model_type == "Random Forest":
            model = RandomForestClassifier(featuresCol="features", labelCol="Class")
        else:
            model = GBTClassifier(featuresCol="features", labelCol="Class")

        trained_model = model.fit(train)

        preds = trained_model.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol="Class")
        auc_score = evaluator.evaluate(preds)

        st.success(f"Model Trained Successfully! ✅ AUC = {round(auc_score, 4)}")

        pdf_pred = preds.select("Class", "prediction", "probability").toPandas()

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix, roc_curve

        y_true = pdf_pred["Class"]
        y_pred = pdf_pred["prediction"]

        cm = confusion_matrix(y_true, y_pred)

        st.markdown("### 🔷 Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        y_prob = pdf_pred["probability"].apply(lambda x: float(x[1]))

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))
        fig_roc.update_layout(title="ROC Curve", template="simple_white")
        st.markly_chart(fig_roc, use_container_width=True)

        st.success("Model Evaluation Completed ✔")

# ---------------------------------------------------------
# PAGE 3 — PREDICT FRAUD
# ---------------------------------------------------------
elif page == "Predict Fraud":

    st.markdown("## 🔍 Real-Time Fraud Prediction")

    amount = st.number_input("Transaction Amount", min_value=0.0)
    time = st.number_input("Time", min_value=0.0)

    if st.button("Predict Fraud"):

        sample = df.limit(1).toPandas()
        sample["Amount"] = amount
        sample["Time"] = time

        sample_df = spark.createDataFrame(sample)

        assembler = VectorAssembler(inputCols=[c for c in df.columns if c != "Class"], outputCol="features")
        vectorized = assembler.transform(sample_df)

        model = LogisticRegression(featuresCol="features", labelCol="Class").fit(
            assembler.transform(df).select("features", "Class")
        )

        prob = model.transform(vectorized).select("probability").collect()[0][0][1]

        st.info(f"Fraud Probability: **{round(prob * 100, 2)}%**")

