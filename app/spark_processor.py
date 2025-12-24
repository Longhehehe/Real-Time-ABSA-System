"""
Spark Processor Module
Demonstrates distributed text processing using PySpark Pandas UDF.
"""
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

def get_spark_session(app_name="PhoBERT Processor"):
    """Initialize Spark Session with Arrow optimization."""
    return SparkSession.builder \
        .appName(app_name) \
        .master("spark://spark-master:7077") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

# Example Pandas UDF for text preprocessing
# SCALAR_ITER UDF handles batches of data as Pandas Series
@pandas_udf("string")
def clean_text_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Distributed text cleaning using Pandas UDF.
    Input: Iterator of Pandas Series (batch of texts)
    Output: Iterator of Pandas Series (batch of cleaned texts)
    
    This function runs in parallel on Spark Workers.
    """
    import re
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special chars but keep Vietnamese chars
        # (This is a simple regex, for production use more robust ones)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.lower().strip()

    # Process each batch
    for series in batch_iter:
        yield series.apply(clean_text)

if __name__ == "__main__":
    print("🚀 Starting Spark Processor...")
    
    spark = get_spark_session()
    
    # Mock data creation for testing (Create 1000 rows)
    print("📊 Generating mock data...")
    data = [
        ("Sản phẩm tuyệt vời! <br> Giao hàng nhanh.",), 
        ("Chất lượng KÉM... không đáng tiền!!!",), 
        ("Shop phục vụ tốt, 5 sao *****",)
    ] * 10
    
    df = spark.createDataFrame(data, ["raw_text"])
    
    # Repartition to simulate distributed processing across workers
    df = df.repartition(2)
    
    # Apply UDF in parallel
    print("⚡ Processing data with Pandas UDF...")
    df_clean = df.withColumn("clean_text", clean_text_udf("raw_text"))
    
    # Show results
    print("=== Result Sample ===")
    df_clean.show(truncate=False)
    
    # Verify execution plan
    df_clean.explain()
    
    spark.stop()
    print("✅ Processing complete.")
