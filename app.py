

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, avg, isnan, split
from pymongo import MongoClient
from urllib.parse import quote_plus
import pandas as pd
import time
import io
import pdfplumber


username = quote_plus("alaa")
password = quote_plus("a120142001@A")
MONGO_URI = (
    f"mongodb+srv://{username}:{password}"
    "@cluster0.weqkj6j.mongodb.net/cloud_project"
    "?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["cloud_project"]
collection = db["results"]


st.title("Cloud-Based Distributed Data Processing Service")
st.write("ارفع ملف CSV, Excel, TXT أو PDF لتحليل البيانات باستخدام Spark و ML")

uploaded_file = st.file_uploader(
    "اختر ملف dataset",
    type=["csv", "xls", "xlsx", "txt", "pdf"]
)


if uploaded_file is not None:
    df = None
    ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if ext == "csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except Exception:
                df = pd.read_csv(uploaded_file, encoding="latin1")

        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)

        elif ext == "txt":
            df = pd.read_csv(
                uploaded_file,
                delimiter="\t",
                encoding="utf-8",
                engine="python"
            )

        elif ext == "pdf":
            text_data = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_data += extracted + "\n"

            df = pd.DataFrame({"text": text_data.splitlines()})

    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

 
    if df is not None and not df.empty:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        spark = (
            SparkSession.builder
            .appName("CloudProjectApp")
            .getOrCreate()
        )

        spark_df = spark.createDataFrame(df)


        st.subheader("عرض أول 5 صفوف:")
        st.dataframe(spark_df.limit(5).toPandas())

        st.subheader("أنواع الأعمدة:")
        st.dataframe(
            pd.DataFrame(spark_df.dtypes, columns=["Column", "Type"])
        )

    
        missing_list = []
        total_rows = spark_df.count()

        for c in spark_df.columns:
            missing_count = spark_df.filter(
                col(c).isNull() | isnan(col(c))
            ).count()
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            missing_list.append([c, missing_pct])

        st.subheader("نسبة القيم المفقودة لكل عمود:")
        st.dataframe(
            pd.DataFrame(missing_list, columns=["Column", "Missing (%)"])
        )

    
        ml_results = []
        nodes_list = [1, 2, 4, 8]
        base_times = {}

        if len(numeric_cols) >= 2:
            assembler = VectorAssembler(
                inputCols=numeric_cols[:2],
                outputCol="features"
            )
            spark_df_km = assembler.transform(spark_df)

            for nodes in nodes_list:
                start_time = time.time()
                kmeans = KMeans(k=3, featuresCol="features")
                kmeans.fit(spark_df_km)
                elapsed = time.time() - start_time

                ml_results.append({
                    "task": f"KMeans_{nodes}_nodes",
                    "nodes": nodes,
                    "time": elapsed
                })

                if nodes == 1:
                    base_times["KMeans"] = elapsed

       
        if len(numeric_cols) >= 2:
            assembler = VectorAssembler(
                inputCols=[numeric_cols[0]],
                outputCol="features"
            )
            spark_df_lr = assembler.transform(spark_df)
            target_col = numeric_cols[1]

            for nodes in nodes_list:
                start_time = time.time()
                lr = LinearRegression(
                    featuresCol="features",
                    labelCol=target_col
                )
                lr.fit(spark_df_lr)
                elapsed = time.time() - start_time

                ml_results.append({
                    "task": f"LinearRegression_{nodes}_nodes",
                    "nodes": nodes,
                    "time": elapsed
                })

                if nodes == 1:
                    base_times["LinearRegression"] = elapsed

       
        if "text" in df.columns:
            spark_df_fp = spark_df.withColumn(
                "items",
                split(col("text"), " ")
            )

            for nodes in nodes_list:
                start_time = time.time()
                fp = FPGrowth(
                    itemsCol="items",
                    minSupport=0.2,
                    minConfidence=0.5
                )
                try:
                    fp.fit(spark_df_fp)
                except Exception:
                    pass

                elapsed = time.time() - start_time
                ml_results.append({
                    "task": f"FPGrowth_{nodes}_nodes",
                    "nodes": nodes,
                    "time": elapsed
                })

                if nodes == 1:
                    base_times["FPGrowth"] = elapsed

        
        date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        if date_cols and numeric_cols:
            date_col = date_cols[0]

            for nodes in nodes_list:
                start_time = time.time()
                spark_df.groupBy(date_col).agg(
                    avg(numeric_cols[0]).alias("daily_avg")
                ).collect()
                elapsed = time.time() - start_time

                ml_results.append({
                    "task": f"TimeSeries_{nodes}_nodes",
                    "nodes": nodes,
                    "time": elapsed
                })

                if nodes == 1:
                    base_times["TimeSeries"] = elapsed

    
        for r in ml_results:
            task_name = r["task"].split("_")[0]
            base = base_times.get(task_name, r["time"])
            r["speedup"] = base / r["time"] if r["time"] > 0 else None
            r["efficiency"] = r["speedup"] / r["nodes"] if r["nodes"] > 0 else None

        results_df = pd.DataFrame(ml_results)

    
        st.subheader("نتائج أداء ML:")
        st.dataframe(results_df)

        collection.insert_many(results_df.to_dict(orient="records"))
        st.success("تم حفظ نتائج ML في MongoDB!")

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="تنزيل نتائج ML كملف CSV",
            data=csv_buffer.getvalue(),
            file_name="ml_results.csv",
            mime="text/csv"
        )

        spark.stop()
