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

ASPECTS = [
    'Chất lượng sản phẩm',                                       
    'Hiệu năng & Trải nghiệm',                                   
    'Đúng mô tả',                                         
    'Giá cả & Khuyến mãi',                                
    'Vận chuyển',                                          
    'Đóng gói',                                     
    'Dịch vụ & Thái độ Shop',                                       
    'Bảo hành & Đổi trả',                           
    'Tính xác thực',                                          
]

def get_spark_session(app_name="PhoBERT Inference"):
    return SparkSession.builder        .appName(app_name)        .master("spark://spark-master:7077")        .config("spark.sql.execution.arrow.pyspark.enabled", "true")        .config("spark.driver.memory", "4g")        .config("spark.executor.memory", "4g")        .getOrCreate()

model = None
tokenizer = None
device = None

def load_model_on_worker():
    """Singleton model loading on worker node."""
    global model, tokenizer, device
    if model is None:
        import sys
        sys.path.insert(0, '/app')                                  
        from transformers import AutoTokenizer
        from phobert_trainer import PhoBERTForABSA
        
        device = torch.device('cpu')                                                
        
        tokenizer_path = "/app/models/phobert_absa/tokenizer"
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
        model_path = "/app/models/phobert_absa/phobert_absa.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            model = PhoBERTForABSA(num_aspects=12)                                       
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
                                                      
    load_model_on_worker()
    
    import json
    
    label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    
    for texts in batch_iter:
                             
        inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits_m, logits_s = model(inputs['input_ids'], inputs['attention_mask'])
            
            preds_m = (torch.sigmoid(logits_m) > 0.5).cpu().numpy()
            
            preds_s = torch.argmax(logits_s, dim=-1).cpu().numpy()
            
        results = []
        
        for i in range(len(texts)):
            row_res = {}
            for j, aspect in enumerate(ASPECTS):
                if preds_m[i][j]:                       
                    row_res[aspect] = label_map[preds_s[i][j]]
            results.append(json.dumps(row_res, ensure_ascii=False))
            
        yield pd.Series(results)

if __name__ == "__main__":
    spark = get_spark_session()
    
    raw_data = [
        ("Sản phẩm xài ổn, giao hàng hơi lâu",),
        ("Tuyệt vời ông mặt trời",),
        ("Máy nóng quá, pin tụt nhanh",),
        ("Shop đóng gói cẩn thận, cho 5 sao",)
    ] * 250
    
    df = spark.createDataFrame(raw_data, ["review_text"])
    df = df.repartition(4)                                
    
    print("🚀 Starting Distributed Inference for 1000 reviews...")
    
    df_pred = df.withColumn("prediction", predict_batch_udf("review_text"))
    
    df_pred.select("review_text", "prediction").show(10, truncate=False)
    
    spark.stop()
    print("✅ Inference complete.")
