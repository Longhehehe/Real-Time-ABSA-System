"""
True Spark Structured Streaming pipeline for realtime ABSA.

Spark owns Kafka ingestion through readStream, processes each micro-batch with
Pandas UDF inference, writes prediction events back to Kafka, and preserves the
existing JSON files used by Airflow/API.
"""
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    coalesce,
    col,
    concat_ws,
    current_timestamp,
    from_json,
    lit,
    pandas_udf,
    sha2,
    struct,
    to_json,
)
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


PROJECT_DIR = os.environ.get("PROJECT_DIR", "/app")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
INPUT_TOPIC = os.environ.get("KAFKA_INPUT_TOPIC", "raw_reviews")
OUTPUT_TOPIC = os.environ.get("KAFKA_OUTPUT_TOPIC", "predictions")
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
CHECKPOINT_DIR = os.environ.get(
    "SPARK_CHECKPOINT_DIR",
    os.path.join(PROJECT_DIR, "data", "spark_checkpoints", "absa_kafka_stream"),
)
PREDICTIONS_DIR = os.environ.get(
    "PREDICTIONS_DIR",
    os.path.join(PROJECT_DIR, "data", "predictions"),
)
STARTING_OFFSETS = os.environ.get("KAFKA_STARTING_OFFSETS", "latest")
MAX_OFFSETS_PER_TRIGGER = os.environ.get("KAFKA_MAX_OFFSETS_PER_TRIGGER")
TRIGGER_INTERVAL = os.environ.get("SPARK_TRIGGER_INTERVAL", "5 seconds")
SPARK_TARGET_ROWS_PER_PARTITION = int(os.environ.get("SPARK_TARGET_ROWS_PER_PARTITION", "32"))
ENABLE_KAFKA_OUTPUT = os.environ.get("ENABLE_KAFKA_OUTPUT", "true").lower() == "true"

ASPECTS = [
    "Chất lượng sản phẩm",
    "Hiệu năng & Trải nghiệm",
    "Đúng mô tả",
    "Giá cả & Khuyến mãi",
    "Vận chuyển",
    "Đóng gói",
    "Dịch vụ & Thái độ Shop",
    "Bảo hành & Đổi trả",
    "Tính xác thực",
]


def get_spark_session() -> SparkSession:
    """Create Spark session with Kafka connector and Arrow enabled."""
    return (
        SparkSession.builder.appName("ABSAKafkaStructuredStreaming")
        .master(SPARK_MASTER)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", "2g"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "1g"))
        .config(
            "spark.jars.packages",
            os.environ.get(
                "SPARK_KAFKA_PACKAGE",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
            ),
        )
        .getOrCreate()
    )


def _preprocess_text_logic(texts: pd.Series) -> pd.Series:
    """Vectorized review cleaning executed inside Spark Python workers."""
    import re

    def clean(text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return texts.apply(clean)


def _empty_multipolarity_prediction() -> Dict[str, Dict]:
    return {aspect: {"mentioned": False, "sentiments": None} for aspect in ASPECTS}


def _normalize_ollama_result(raw_result: Dict[str, str]) -> Dict[str, Dict]:
    normalized = {}
    for aspect in ASPECTS:
        sentiment = raw_result.get(aspect)
        if sentiment in {"POS", "NEG", "NEU"}:
            normalized[aspect] = {"mentioned": True, "sentiments": [sentiment]}
        else:
            normalized[aspect] = {"mentioned": False, "sentiments": None}
    return normalized


def _predict_model_logic(texts: pd.Series) -> pd.Series:
    """
    Vectorized model inference executed by Spark workers.

    A predictor singleton is kept per Python worker process so the model is not
    reloaded for every Arrow batch.
    """
    import os
    import sys

    if PROJECT_DIR not in sys.path:
        sys.path.append(PROJECT_DIR)

    try:
        from app.absa_predictor import PhoBERTPredictor
        from app.ollama_predictor import OllamaPredictor
    except Exception as exc:
        error = {"error": f"ImportError on Spark worker: {exc}"}
        return pd.Series([json.dumps(error, ensure_ascii=False)] * len(texts))

    if not hasattr(_predict_model_logic, "predictor"):
        _predict_model_logic.predictor = None
        _predict_model_logic.model_type = None
        _predict_model_logic.last_config_check = 0

    config_path = os.path.join(PROJECT_DIR, "model_config.json")
    now = time.time()
    if now - _predict_model_logic.last_config_check > 30:
        target_model = "phobert"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    target_model = json.load(f).get("active_model", "phobert")
            except Exception:
                target_model = "phobert"

        if target_model != _predict_model_logic.model_type or _predict_model_logic.predictor is None:
            if target_model == "ollama":
                _predict_model_logic.predictor = OllamaPredictor()
            else:
                _predict_model_logic.predictor = PhoBERTPredictor()
            _predict_model_logic.model_type = target_model
            print(f"Spark worker {os.getpid()}: loaded model backend {target_model}")

        _predict_model_logic.last_config_check = now

    predictor = _predict_model_logic.predictor
    if predictor is None:
        return pd.Series(
            [json.dumps(_empty_multipolarity_prediction(), ensure_ascii=False)] * len(texts)
        )

    text_list = ["" if text is None else str(text) for text in texts.tolist()]

    try:
        if _predict_model_logic.model_type == "ollama":
            batch_results = predictor.predict_batch(text_list)
            normalized = [_normalize_ollama_result(result) for result in batch_results]
        else:
            normalized = predictor.predict_batch(text_list, format="multipolarity")
    except Exception as exc:
        print(f"Spark worker {os.getpid()}: prediction failed: {exc}")
        normalized = [_empty_multipolarity_prediction() for _ in text_list]

    return pd.Series([json.dumps(result, ensure_ascii=False) for result in normalized])


def save_predictions(product_id: str, new_predictions: List[Dict]) -> None:
    """Append prediction rows to the existing product JSON file atomically."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    file_path = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    temp_path = f"{file_path}.tmp"

    existing_data = []
    if os.path.exists(file_path):
        for attempt in range(3):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                break
            except json.JSONDecodeError:
                if attempt == 2:
                    existing_data = []
                else:
                    time.sleep(0.1)

    existing_ids = {item.get("review_id") for item in existing_data if item.get("review_id")}
    added_count = 0
    for pred in new_predictions:
        review_id = pred.get("review_id")
        if not review_id or review_id not in existing_ids:
            existing_data.append(pred)
            if review_id:
                existing_ids.add(review_id)
            added_count += 1

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, file_path)
    try:
        os.chmod(file_path, 0o666)
    except Exception as exc:
        print(f"Could not chmod {file_path}: {exc}")

    print(f"Saved {len(existing_data)} predictions for {product_id} ({added_count} new)")


def process_microbatch(batch_df, batch_id: int) -> None:
    """Sink one Spark streaming micro-batch to Kafka predictions and JSON files."""
    cached = batch_df.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        rows = cached.select(
            "product_id",
            "review_id",
            "original_text",
            "cleaned_text",
            "sentiment_json",
            "rating",
        ).collect()

        if not rows:
            print(f"Micro-batch {batch_id}: no rows")
            return

        if ENABLE_KAFKA_OUTPUT:
            kafka_rows = cached.select(
                col("product_id").cast("string").alias("key"),
                to_json(
                    struct(
                        col("product_id"),
                        col("review_id"),
                        col("original_text"),
                        col("cleaned_text"),
                        col("sentiment_json"),
                        col("rating"),
                        col("processed_at"),
                    )
                ).alias("value"),
            )
            kafka_rows.write.format("kafka").option(
                "kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS
            ).option("topic", OUTPUT_TOPIC).save()

        grouped = defaultdict(list)
        processed_at = time.time()
        for row in rows:
            product_id = row["product_id"] or "unknown"
            try:
                sentiment = json.loads(row["sentiment_json"])
            except Exception:
                sentiment = {"error": row["sentiment_json"]}

            grouped[product_id].append(
                {
                    "review_id": row["review_id"],
                    "original_text": row["original_text"],
                    "cleaned_text": row["cleaned_text"],
                    "sentiment": sentiment,
                    "rating": row["rating"],
                    "processed_at": processed_at,
                }
            )

        for product_id, predictions in grouped.items():
            save_predictions(product_id, predictions)

        print(
            f"Micro-batch {batch_id}: processed {len(rows)} reviews "
            f"for {len(grouped)} product(s)"
        )
    finally:
        cached.unpersist()


def build_stream(spark: SparkSession):
    """Build Kafka readStream -> parsed review stream -> prediction stream."""
    review_schema = StructType(
        [
            StructField("product_id", StringType(), True),
            StructField("review_content", StringType(), True),
            StructField("review_text", StringType(), True),
            StructField("reviewContent", StringType(), True),
            StructField("content", StringType(), True),
            StructField("rating", StringType(), True),
            StructField("review_id", StringType(), True),
            StructField("timestamp", DoubleType(), True),
        ]
    )

    reader = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", STARTING_OFFSETS)
        .option("failOnDataLoss", "false")
    )
    if MAX_OFFSETS_PER_TRIGGER:
        reader = reader.option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)

    raw_stream = reader.load()

    parsed = (
        raw_stream.selectExpr("CAST(value AS STRING) AS json_value")
        .select(from_json(col("json_value"), review_schema).alias("data"), col("json_value"))
        .select("data.*", "json_value")
        .withColumn("product_id", coalesce(col("product_id"), lit("unknown")))
        .withColumn(
            "original_text",
            coalesce(col("review_content"), col("review_text"), col("reviewContent"), col("content"), lit("")),
        )
        .withColumn(
            "review_id",
            coalesce(
                col("review_id"),
                sha2(concat_ws("||", col("product_id"), col("original_text"), col("json_value")), 256),
            ),
        )
    )

    preprocess_udf = pandas_udf(StringType())(_preprocess_text_logic)
    predict_model_udf = pandas_udf(StringType())(_predict_model_logic)

    target_partitions = max(1, int(os.environ.get("SPARK_STREAM_REPARTITION", "4")))
    return (
        parsed.repartition(target_partitions, col("product_id"))
        .withColumn("cleaned_text", preprocess_udf(col("original_text")))
        .withColumn("sentiment_json", predict_model_udf(col("cleaned_text")))
        .withColumn("processed_at", current_timestamp())
        .select(
            "product_id",
            "review_id",
            "original_text",
            "cleaned_text",
            "sentiment_json",
            "rating",
            "processed_at",
        )
    )


def main() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    spark = get_spark_session()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))

    print("Starting true Spark Structured Streaming ABSA pipeline")
    print(f"Kafka input: {KAFKA_BOOTSTRAP_SERVERS}/{INPUT_TOPIC}")
    print(f"Kafka output: {KAFKA_BOOTSTRAP_SERVERS}/{OUTPUT_TOPIC}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Prediction files: {PREDICTIONS_DIR}")
    print(f"Starting offsets: {STARTING_OFFSETS}")

    prediction_stream = build_stream(spark)
    query = (
        prediction_stream.writeStream.foreachBatch(process_microbatch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .queryName("absa_kafka_structured_stream")
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
