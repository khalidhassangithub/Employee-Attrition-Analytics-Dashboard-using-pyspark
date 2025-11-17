# =============================================================
# 🔹 Credit Card Fraud Detection using PySpark
# =============================================================

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------------------------------------------
# 1️⃣ Start Spark Session
# -------------------------------------------------------------
spark = SparkSession.builder.appName("CreditCardFraud").getOrCreate()

# -------------------------------------------------------------
# 2️⃣ Load Dataset
# -------------------------------------------------------------
file_path = "/Users/khalidoscar/Downloads/creditcard.csv"   # Change path if needed

data = spark.read.csv(file_path, header=True, inferSchema=True).cache()

print("Data Loaded Successfully!")
data.printSchema()
print("Total rows:", data.count())

# -------------------------------------------------------------
# 3️⃣ Feature Engineering
# -------------------------------------------------------------
feature_cols = [c for c in data.columns if c != "Class"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
assembled_data = assembler.transform(data)

# Normalize features
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaled_data = scaler.fit(assembled_data).transform(assembled_data)

final_data = scaled_data.select("features", "Class")

# -------------------------------------------------------------
# 4️⃣ Train-Test Split
# -------------------------------------------------------------
train, test = final_data.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows: {train.count()} | Test rows: {test.count()}")

# -------------------------------------------------------------
# 5️⃣ Choose Model
# -------------------------------------------------------------
model_name = "Random Forest"   # Change manually: LR, DT, RF, GBT

if model_name == "Logistic Regression":
    model = LogisticRegression(featuresCol="features", labelCol="Class")
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(featuresCol="features", labelCol="Class")
elif model_name == "Random Forest":
    model = RandomForestClassifier(featuresCol="features", labelCol="Class")
else:
    model = GBTClassifier(featuresCol="features", labelCol="Class")

# -------------------------------------------------------------
# 6️⃣ Train Model
# -------------------------------------------------------------
print(f"Training Model: {model_name} ...")
trained_model = model.fit(train)

# -------------------------------------------------------------
# 7️⃣ Predictions
# -------------------------------------------------------------
predictions = trained_model.transform(test)
predictions.select("Class", "prediction", "probability").show(10)

# -------------------------------------------------------------
# 8️⃣ Evaluation — AUC Score
# -------------------------------------------------------------
evaluator = BinaryClassificationEvaluator(labelCol="Class")
auc_score = evaluator.evaluate(predictions)

print(f"\n✅ {model_name} AUC Score: {auc_score}")

# -------------------------------------------------------------
# 9️⃣ Confusion Matrix
# -------------------------------------------------------------
pdf_pred = predictions.select("Class", "prediction").toPandas()

cm = confusion_matrix(pdf_pred["Class"], pdf_pred["prediction"])

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"{model_name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------------------------
# 🔟 ROC Curve
# -------------------------------------------------------------
pdf_prob = predictions.select("Class", "probability").toPandas()
pdf_prob["prob"] = pdf_prob["probability"].apply(lambda x: float(x[1]))

fpr, tpr, _ = roc_curve(pdf_prob["Class"], pdf_prob["prob"])

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.title(f"{model_name} - ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()

print("\n🎉 Fraud Detection Pipeline Completed Successfully!")
