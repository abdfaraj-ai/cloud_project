# app_colab.py - Colab-Friendly Version (No Streamlit)

# -----------------------------
# Install Required Packages
# -----------------------------
# !pip install pyspark pymongo pdfplumber

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import io
import time
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, avg, isnan, split
from pymongo import MongoClient
from urllib.parse import quote_plus
import pdfplumber
from google.colab import files

# -----------------------------
# MongoDB Connection
# -----------------------------
username = quote_plus("alaa")
password = quote_plus("a120142001@A")
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.weqkj6j.mongodb.net/cloud_project?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["cloud_project"]
collection = db["results"]

# -----------------------------
# File Upload in Colab
# -----------------------------
print("Upload your dataset file (CSV, XLSX, TXT, or PDF)")
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
print("File uploaded:", file_path)

# -----------------------------
# Load File into Pandas
# -----------------------------
df = None
ext = file_path.split(".")[-1].lower()

try:
    if ext == "csv":
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except:
            df = pd.read_csv(file_path, encoding="latin1")
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file_path)
    elif ext == "txt":
        df = pd.read_csv(file_path, delimiter="\t", encoding="utf-8", engine="python")
    elif ext == "pdf":
        text_data = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_data += extracted + "\n"
        df = pd.DataFrame({"text": text_data.splitlines()})
except Exception as e:
    print("Error reading file:", e)

if df is None:
    raise ValueError("File could not be loaded. Exiting.")

print("\nFirst 5 rows of your dataset:")
display(df.head())

# -----------------------------
# Spark Session
# -----------------------------
spark = SparkSession.builder.appName("CloudProjectApp").getOrCreate()
spark_df = spark.createDataFrame(df)

# -----------------------------
# Column Info and Missing Values
# -----------------------------
print("\nColumn types:")
display(pd.DataFrame(spark_df.dtypes, columns=["Column", "Type"]))

print("\nMissing values % per column:")
missing_list = []
total_rows = spark_df.count()
for c in spark_df.columns:
    missing_count = spark_df.filter(col(c).isNull() | isnan(col(c))).count()
    missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
    missing_list.append([c, missing_pct])
display(pd.DataFrame(missing_list, columns=["Column", "Missing (%)"]))

# -----------------------------
# ML Performance
# -----------------------------
ml_results = []
nodes_list = [1, 2, 4, 8]
base_times = {}
numeric_cols_list = list(df.select_dtypes(include=["number"]).columns)

# -------- KMeans --------
if len(numeric_cols_list) >= 2:
    assembler = VectorAssembler(inputCols=numeric_cols_list[:2], outputCol="features")
    spark_df_km = assembler.transform(spark_df)

    for nodes in nodes_list:
        start_time = time.time()
        kmeans = KMeans(k=3, featuresCol="features")
        kmeans.fit(spark_df_km)
        elapsed = time.time() - start_time
        ml_results.append({"task": f"KMeans_{nodes}_nodes", "nodes": nodes, "time": elapsed})
        if nodes == 1:
            base_times["KMeans"] = elapsed

# -------- Linear Regression --------
if len(numeric_cols_list) >= 2:
    assembler = VectorAssembler(inputCols=[numeric_cols_list[0]], outputCol="features")
    spark_df_lr = assembler.transform(spark_df)
    target_col = numeric_cols_list[1]

    for nodes in nodes_list:
        start_time = time.time()
        lr = LinearRegression(featuresCol="features", labelCol=target_col)
        lr.fit(spark_df_lr)
        elapsed = time.time() - start_time
        ml_results.append({"task": f"LinearRegression_{nodes}_nodes", "nodes": nodes, "time": elapsed})
        if nodes == 1:
            base_times["LinearRegression"] = elapsed

# -------- FP-Growth (Text) --------
if "text" in df.columns:
    spark_df_fp = spark_df.withColumn("items", split(col("text"), " "))
    for nodes in nodes_list:
        start_time = time.time()
        fp = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.5)
        try:
            fp.fit(spark_df_fp)
        except:
            pass
        elapsed = time.time() - start_time
        ml_results.append({"task": f"FPGrowth_{nodes}_nodes", "nodes": nodes, "time": elapsed})
        if nodes == 1:
            base_times["FPGrowth"] = elapsed

# -------- Time Series --------
date_cols = df.select_dtypes(include=["datetime"]).columns
if len(date_cols) > 0 and len(numeric_cols_list) > 0:
    date_col = date_cols[0]
    for nodes in nodes_list:
        start_time = time.time()
        spark_df.groupBy(date_col).agg(avg(numeric_cols_list[0]).alias("daily_avg")).collect()
        elapsed = time.time() - start_time
        ml_results.append({"task": f"TimeSeries_{nodes}_nodes", "nodes": nodes, "time": elapsed})
        if nodes == 1:
            base_times["TimeSeries"] = elapsed

# -------- Speedup & Efficiency --------
for r in ml_results:
    task_name = r["task"].split("_")[0]
    r["speedup"] = base_times.get(task_name, r["time"]) / r["time"] if r["time"] > 0 else None
    r["efficiency"] = r["speedup"] / r["nodes"] if r["nodes"] > 0 else None

results_df = pd.DataFrame(ml_results)

# -----------------------------
# Show and Save Results
# -----------------------------
print("\nML Performance Results:")
display(results_df)

# Save to MongoDB
collection.insert_many(results_df.to_dict(orient="records"))
print("\nResults saved to MongoDB!")

# Save CSV locally
results_df.to_csv("/content/ml_results.csv", index=False)
print("\nCSV file saved as ml_results.csv")

# Optional: download in Colab
files.download("/content/ml_results.csv")

spark.stop()
print("\nProcessing complete!")
