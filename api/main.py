"""
FastAPI Backend for ABSA Predictions
Serves prediction data to React frontend
Supports multi-polarity sentiment predictions
"""
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from pydantic import BaseModel
import sys

# Add project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'app'))

# Import the shared predictor from app/
try:
    from absa_predictor import PhoBERTPredictor, ASPECTS
except ImportError:
    # Fallback if run from different cwd
    from app.absa_predictor import PhoBERTPredictor, ASPECTS

# Paths
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup using the shared Predictor class"""
    global predictor
    print(f"🚀 Starting ABSA API...")
    
    predictor = PhoBERTPredictor()
    # It will automatically find the best model in ./models/
    success = predictor.load_model()
    
    if success:
        print("✅ Shared PhoBERT Predictor loaded successfully!")
    else:
        print("❌ Failed to load Shared PhoBERT Predictor.")
    
    yield
    
    print("🛑 Shutting down ABSA API...")
    predictor = None


app = FastAPI(title="ABSA Prediction API", version="2.1.0", lifespan=lifespan)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DATA MODELS ===

class PredictionFile(BaseModel):
    product_id: str
    file_name: str
    review_count: int
    last_modified: float


class AspectScore(BaseModel):
    aspect: str
    score: float
    review_count: int
    sentiment_distribution: Optional[Dict[str, float]] = None
    has_mixed: Optional[bool] = False


class ProductPrediction(BaseModel):
    product_id: str
    reviews: List[Dict]
    aspect_scores: List[AspectScore]
    total_reviews: int


class TextPredictionRequest(BaseModel):
    text: str


class TriggerRequest(BaseModel):
    product_url: str
    max_reviews: int = 50


class TriggerResponse(BaseModel):
    success: bool
    message: str
    dag_run_id: Optional[str] = None
    product_id: Optional[str] = None


# === ENDPOINTS ===

@app.get("/")
def root():
    model_status = predictor.model_loaded if predictor else False
    return {
        "status": "ok", 
        "message": "ABSA Prediction API v2.1 (Shared Predictor)", 
        "model_loaded": model_status
    }


@app.get("/api/model-info")
def get_model_info():
    """Get ABSA model configuration"""
    return {
        "aspects": ASPECTS,
        "sentiment_classes": ["NEG", "POS", "NEU"],
        "multi_polarity": True,
        "description": "PhoBERT ABSA Multi-Polarity Model",
        "loaded": predictor.model_loaded if predictor else False
    }


@app.post("/api/evaluate-model")
def run_model_evaluation():
    """Trigger model evaluation on validation set"""
    try:
        from phobert_trainer_multipolarity import evaluate_model
        
        DATA_PATH = os.path.join(BASE_DIR, 'labeled')
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'phobert_absa_multipolarity')
        
        if not os.path.exists(DATA_PATH):
             raise HTTPException(status_code=404, detail=f"Labeled data path not found: {DATA_PATH}")
             
        if not os.path.exists(MODEL_PATH):
             raise HTTPException(status_code=404, detail=f"Model path not found: {MODEL_PATH}")

        print(f"Starting evaluation on {DATA_PATH} using model at {MODEL_PATH}")
        results = evaluate_model(data_path=DATA_PATH, model_path=MODEL_PATH)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
            
        return results
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-text")
def predict_text(request: TextPredictionRequest):
    """Predict aspects and sentiments for a single text using shared predictor"""
    if not predictor or not predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please verify ./models folder.")
        
    try:
        # Use simple legacy prediction for now or clean up result
        # The frontend expects { text: str, aspects: { name: { mentioned: bool, sentiments: [] } } }
        
        raw_result = predictor.predict_single(request.text)
        multipolarity_result = raw_result.get('multipolarity', {})
        
        return {
            "text": request.text,
            "aspects": multipolarity_result
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
def search_products(keyword: str, limit: int = 20):
    """Search Lazada products"""
    try:
        from lazada_search import search_lazada
        
        cookies_path = None
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'lazada_cookies.json'),
            '/app/api/lazada_cookies.json',
            '/app/cookie/lazada_cookies.json',
            os.path.join(BASE_DIR, 'lazada_crawler', 'lazada_cookies.json'),
            os.path.join(BASE_DIR, 'cookie', 'lazada_cookies.json'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cookies_path = path
                print(f"Using cookies from: {path}")
                break
        
        products = search_lazada(keyword, limit=limit, cookies_path=cookies_path)
        
        results = []
        for p in products:
            url = p.get('url', '')
            item_id = p.get('item_id', '')
            if not url and item_id:
                url = f"https://www.lazada.vn/products/-i{item_id}.html"
            
            results.append({
                "title": p.get('name', 'Unknown'),
                "price": p.get('price', 'N/A'),
                "link": url,
                "reviews": p.get('reviews', 0),
                "image": p.get('image', ''),
                "item_id": item_id,
                "rating": p.get('rating', 0),
                "sold": p.get('sold', '')
            })
        
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "products": []}


@app.get("/api/aspects", response_model=List[str])
def get_aspects():
    """Get list of aspects"""
    return ASPECTS


@app.delete("/api/predictions/clear")
def clear_predictions():
    """Clear all prediction files"""
    import glob
    files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.json")) + glob.glob(os.path.join(PREDICTIONS_DIR, "*.done"))
    
    deleted_count = 0
    errors = []
    
    for file_path in files:
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            errors.append(f"{os.path.basename(file_path)}: {str(e)}")
    
    return {
        "success": len(errors) == 0,
        "deleted_count": deleted_count,
        "errors": errors
    }


@app.get("/api/predictions", response_model=List[PredictionFile])
def list_predictions():
    """List all prediction files"""
    if not os.path.exists(PREDICTIONS_DIR):
        return []
    
    files = []
    for f in os.listdir(PREDICTIONS_DIR):
        if f.endswith('.json'):
            file_path = os.path.join(PREDICTIONS_DIR, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                files.append(PredictionFile(
                    product_id=f.replace('.json', ''),
                    file_name=f,
                    review_count=len(data) if isinstance(data, list) else 0,
                    last_modified=os.path.getmtime(file_path)
                ))
            except:
                pass
    
    return sorted(files, key=lambda x: x.last_modified, reverse=True)


@app.get("/api/predictions/{product_id}", response_model=ProductPrediction)
def get_prediction(product_id: str):
    """Get predictions for a specific product"""
    file_path = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    
    # Calculate aspect scores using robust logic
    from app.absa_predictor import aggregate_multipolarity_scores
    
    # Pre-process sentiments to ensure compatibility with aggregate_multipolarity_scores
    formatted_preds = []
    
    for r in reviews:
        if 'sentiment' not in r:
            continue
            
        raw_sentiment = r['sentiment']
        processed = {}
        
        # Check if legacy format (integers)
        is_legacy = False
        for k, v in raw_sentiment.items():
            if isinstance(v, int):
                is_legacy = True
                break
        
        if is_legacy:
            # Convert legacy int to multipolarity dict
            # 1 -> POS, -1 -> NEG, 0 -> NEU, 2 -> Not mentioned
            label_map = {1: ['POS'], -1: ['NEG'], 0: ['NEU']}
            
            for aspect, val in raw_sentiment.items():
                if val == 2:
                    processed[aspect] = {'mentioned': False, 'sentiments': []}
                else:
                    processed[aspect] = {
                        'mentioned': True,
                        'sentiments': label_map.get(val, ['NEU'])
                    }
            formatted_preds.append(processed)
        else:
            # Already in multipolarity format
            formatted_preds.append(raw_sentiment)
             
    aspect_scores_dict = aggregate_multipolarity_scores(formatted_preds)
    
    # Convert dict to AspectScore objects
    final_scores = []
    for aspect, data in aspect_scores_dict.items():
        final_scores.append(AspectScore(
            aspect=aspect,
            score=data['score'],
            review_count=data['review_count'],
            sentiment_distribution=data['sentiment_distribution'],
            has_mixed=data['has_mixed']
        ))
    
    return ProductPrediction(
        product_id=product_id,
        reviews=reviews,
        aspect_scores=final_scores,
        total_reviews=len(reviews)
    )

@app.post("/api/trigger-absa", response_model=TriggerResponse)
async def trigger_absa_pipeline(request: TriggerRequest):
    """Trigger the ABSA pipeline via Airflow REST API."""
    import re
    import httpx
    import base64
    from datetime import datetime
    
    match = re.search(r'-i(\d+)', request.product_url)
    if not match:
        match = re.search(r'products/.*-i(\d+)', request.product_url)
    
    product_id = match.group(1) if match else f"product_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    airflow_url = os.environ.get('AIRFLOW_URL', 'http://airflow-webserver:8080')
    dag_id = 'realtime_absa_pipeline'
    run_id = f"react_trigger_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        credentials = base64.b64encode(b"admin:admin").decode("ascii")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {credentials}"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns",
                json={
                    "dag_run_id": run_id,
                    "conf": {
                        "product_url": request.product_url,
                        "product_id": product_id,
                        "max_reviews": request.max_reviews,
                        "source": "react_frontend"
                    }
                },
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                return TriggerResponse(
                    success=True,
                    message=f"Pipeline triggered successfully for product {product_id}",
                    dag_run_id=run_id,
                    product_id=product_id
                )
            else:
                raise Exception(f"Airflow API failed with {response.status_code}")
                
    except Exception as e:
        trigger_dir = os.path.join(BASE_DIR, 'data', 'triggers')
        os.makedirs(trigger_dir, exist_ok=True)
        trigger_file = os.path.join(trigger_dir, f"{product_id}.json")
        with open(trigger_file, 'w', encoding='utf-8') as f:
            json.dump({
                "product_url": request.product_url,
                "product_id": product_id,
                "max_reviews": request.max_reviews,
                "triggered_at": datetime.now().isoformat(),
                "error": str(e)
            }, f, ensure_ascii=False)
        
        return TriggerResponse(
            success=True,
            message=f"Trigger saved to file locally (Airflow offline): {product_id}",
            product_id=product_id
        )

@app.get("/api/trigger-status/{product_id}")
def get_trigger_status(product_id: str):
    """Check if predictions exist"""
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    trigger_file = os.path.join(BASE_DIR, 'data', 'triggers', f"{product_id}.json")
    done_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.done")
    
    if os.path.exists(pred_file):
        with open(pred_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
            except:
                count = 0
        status = "completed" if os.path.exists(done_file) else "processing"
        return {"status": status, "review_count": count, "last_updated": os.path.getmtime(pred_file)}
    elif os.path.exists(trigger_file):
        return {"status": "processing", "message": "Pipeline is running..."}
    else:
        return {"status": "not_found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
