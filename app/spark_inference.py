"""
Spark Inference Module (Distributed Prediction)
Demonstrates distributed inference using PySpark Pandas UDF and PhoBERT.
"""
import os
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, DoubleType
import pandas as pd
from typing import Iterator

# Define schema for output (Sentiment scores for 9 aspects)
ASPECTS = [
    'Chất lượng sản phẩm',       # Quality, durability, materials
    'Hiệu năng & Trải nghiệm',   # Performance, user experience  
    'Đúng mô tả',                # Accuracy of description
    'Giá cả & Khuyến mãi',       # Price, discounts, value
    'Vận chuyển',                # Shipping speed, delivery
    'Đóng gói',                  # Packaging quality
    'Dịch vụ & Thái độ Shop',    # Customer service, seller attitude
    'Bảo hành & Đổi trả',        # Warranty, returns
    'Tính xác thực',             # Authenticity (fake/genuine)
]


def get_spark_session(app_name="PhoBERT Inference"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("spark://spark-master:7077") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

# Load model globally on worker (Broadcast variable optimization/Singleton pattern)
model = None
tokenizer = None
device = None

def load_model_on_worker():
    """Singleton model loading on worker node."""
    global model, tokenizer, device
    if model is None:
        import sys
        sys.path.insert(0, '/app')  # Ensure app modules are visible
        from transformers import AutoTokenizer
        from model.phobert_trainer import PhoBERTForABSA
        
        device = torch.device('cpu')  # Use CPU on Spark workers (unless GPU config)
        
        # Load tokenizer
        tokenizer_path = "/app/models/phobert_absa/tokenizer"
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
        # Load Model with new multi-task architecture
        model_path = "/app/models/phobert_absa/phobert_absa.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            model = PhoBERTForABSA(num_aspects=12)  # New architecture without num_labels
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

@pandas_udf("string")
def predict_batch_udf(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Distributed Inference UDF with multi-task model.
    Input: Iterator of text batches
    Output: Iterator of prediction results (JSON string)
    """
    # Initialize model once per executor/python worker
    load_model_on_worker()
    
    import json
    
    # Label map for display
    label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    for texts in batch_iter:
        # Preprocessing batch
        inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Inference with multi-task model
        with torch.no_grad():
            logits_m, logits_s = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Mention predictions (binary)
            preds_m = (torch.sigmoid(logits_m) > 0.5).cpu().numpy()
            
            # Sentiment predictions (0=NEG, 1=POS, 2=NEU)
            preds_s = torch.argmax(logits_s, dim=-1).cpu().numpy()
            
        # Format results
        results = []
        
        for i in range(len(texts)):
            row_res = {}
            for j, aspect in enumerate(ASPECTS):
                if preds_m[i][j]:  # Aspect is mentioned
                    row_res[aspect] = label_map[preds_s[i][j]]
            results.append(json.dumps(row_res, ensure_ascii=False))
            
        yield pd.Series(results)

if __name__ == "__main__":
    spark = get_spark_session()
    
    # Mock Big Data (1000 rows)
    raw_data = [
        ("Sản phẩm xài ổn, giao hàng hơi lâu",),
        ("Tuyệt vời ông mặt trời",),
        ("Máy nóng quá, pin tụt nhanh",),
        ("Shop đóng gói cẩn thận, cho 5 sao",)
    ] * 250
    
    df = spark.createDataFrame(raw_data, ["review_text"])
    df = df.repartition(4)  # Simulate multiple partitions
    
    print("🚀 Starting Distributed Inference for 1000 reviews...")
    
    # Apply Inference UDF
    df_pred = df.withColumn("prediction", predict_batch_udf("review_text"))
    
    # Show results
    df_pred.select("review_text", "prediction").show(10, truncate=False)
    
    spark.stop()
    print("✅ Inference complete.")
