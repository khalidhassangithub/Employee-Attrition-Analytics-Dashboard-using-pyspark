import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload data → Explore stats → Train models → Evaluate fraud risk")

uploaded_file = st.sidebar.file_uploader("📁 Upload Credit Card Fraud CSV", type=["csv"])

df = None
if uploaded_file:
    pdf = pd.read_csv(uploaded_file)
    df = spark.createDataFrame(pdf).cache()
    st.success("CSV Loaded Successfully!")

if df:
    st.subheader("📊 Dataset Overview")
    pdf = df.toPandas()
    st.write(pdf.head())

    total = len(pdf)
    fraud = pdf['Class'].sum()
    fraud_rate = round((fraud / total) * 100, 3)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Cases", fraud)
    col3.metric("Fraud Rate", f"{fraud_rate}%")

    st.write("---")
    st.subheader("📈 Visual Explorations")
    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots()
        ax.hist(pdf['Amount'], bins=40)
        plt.title("Transaction Amount Distribution")
        st.pyplot(fig)

    with colB:
        fig, ax = plt.subplots()
        sns.countplot(data=pdf, x='Class')
        plt.title("Fraud vs Non-Fraud Count")
        st.pyplot(fig)

    st.write("---")
    st.subheader("🤖 Train Machine Learning Models (PySpark)")

    feature_cols = [c for c in pdf.columns if c not in ['Class']]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled = assembler.transform(df).select("features", "Class")

    train, test = assembled.randomSplit([0.8, 0.2], seed=42)

    model_choice = st.selectbox("Select ML Model", [
        "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosted Trees"
    ])

    if st.button("🚀 Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(featuresCol="features", labelCol="Class")
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(featuresCol="features", labelCol="Class")
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(featuresCol="features", labelCol="Class")
        else:
            model = GBTClassifier(featuresCol="features", labelCol="Class")

        trained_model = model.fit(train)
        predictions = trained_model.transform(test)

        evaluator = BinaryClassificationEvaluator(labelCol="Class")
        auc_score = evaluator.evaluate(predictions)

        st.success(f"Model Trained! AUC = {auc_score}")

        pdf_pred = predictions.select("Class", "prediction", "probability").toPandas()
        y_true = pdf_pred['Class']
        y_pred = pdf_pred['prediction']

        cm = confusion_matrix(y_true, y_pred)

        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(fig)

        st.subheader("📉 ROC Curve")
        probs = pdf_pred['probability'].apply(lambda x: float(x[1]))
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        st.pyplot(fig)

        st.write("---")
        st.subheader("🔍 Real-Time Fraud Probability Check")
        amount = st.number_input("Transaction Amount", min_value=0.0)

        if st.button("Predict Fraud"):
            dummy = pdf.iloc[0:1].copy()
            dummy['Amount'] = amount
            x = spark.createDataFrame(dummy)
            vec = assembler.transform(x)
            pred = trained_model.transform(vec).select("probability").collect()[0][0][1]
            st.write(f"Fraud Probability: {round(pred*100, 2)}%")
