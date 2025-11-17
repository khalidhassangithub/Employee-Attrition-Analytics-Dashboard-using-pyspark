# ============================================================
# Project: Employee Attrition Prediction using PySpark
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

# 1️⃣ Create Spark Session
spark = SparkSession.builder.appName("EmployeeAttritionPrediction").getOrCreate()

# 2️⃣ Load Dataset
data = spark.read.csv(
    "/Users/khalidoscar/Downloads/Employers_data.csv",
    header=True,
    inferSchema=True
)

print("Schema:")
data.printSchema()
data.show(5)

# 3️⃣ Drop missing values
data = data.dropna()

# 4️⃣ Encode Categorical Variables
categorical_cols = ["Gender", "Department", "Job_Title", "Location", "Education_Level", "Attrition"]

indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "Index")
    for col_name in categorical_cols
]

pipeline = Pipeline(stages=indexers)
data = pipeline.fit(data).transform(data)

# Rename AttritionIndex → label
data = data.withColumnRenamed("AttritionIndex", "label")

# 5️⃣ Feature Assembler
feature_cols = ["Age", "Experience_Years", "Salary",
                "GenderIndex", "DepartmentIndex", "Job_TitleIndex",
                "LocationIndex", "Education_LevelIndex"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data).select("features", "label")

# 6️⃣ Train-Test Split
train_data, test_data = final_data.randomSplit([0.7, 0.3], seed=42)

# 7️⃣ Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(train_data)

# 8️⃣ Predictions
predictions = model.transform(test_data)
predictions.select("label", "prediction").show(10)

# 9️⃣ Evaluation Metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")

# ------------------------------------------------------------
# 10️⃣ Visualization: Attrition Count
# ------------------------------------------------------------
attrition_trend = data.groupBy("Attrition").count().toPandas()

plt.figure(figsize=(6,4))
plt.bar(attrition_trend["Attrition"], attrition_trend["count"], color=["green", "red"])
plt.title("Employee Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 11️⃣ Visualization: Department-wise Attrition
# ------------------------------------------------------------
dept_trend = data.groupBy("Department", "Attrition").count().toPandas()
dept_pivot = dept_trend.pivot(index="Department", columns="Attrition", values="count").fillna(0)

dept_pivot.plot(kind="bar", figsize=(8,5))
plt.title("Department-wise Attrition")
plt.xlabel("Department")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 12️⃣ Chart: Age Distribution (Histogram)
# ------------------------------------------------------------
age_df = data.select("Age").toPandas()

plt.figure(figsize=(6,4))
plt.hist(age_df["Age"], bins=15)
plt.title("Age Distribution of Employees")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------------------------
# 13️⃣ Chart: Salary vs Attrition (Boxplot)
# ------------------------------------------------------------
salary_attr = data.select("Salary", "Attrition").toPandas()

plt.figure(figsize=(6,4))
salary_attr.boxplot(by="Attrition", column=["Salary"])
plt.title("Salary Distribution by Attrition")
plt.suptitle("")  # remove extra title
plt.xlabel("Attrition (No / Yes)")
plt.ylabel("Salary")
plt.show()

# ------------------------------------------------------------
# 14️⃣ Chart: Experience vs Salary (Scatter Plot)
# ------------------------------------------------------------
exp_salary = data.select("Experience_Years", "Salary").toPandas()

plt.figure(figsize=(6,4))
plt.scatter(exp_salary["Experience_Years"], exp_salary["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.show()

# ============================================================
# End of PySpark Project
# ============================================================
