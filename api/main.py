"""
FastAPI Backend for ABSA Predictions
Serves prediction data to React frontend
Supports multi-polarity sentiment predictions
"""
import os
import json
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from pydantic import BaseModel
import sys

# Add project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import the shared predictor from app/
from app.absa_predictor import PhoBERTPredictor, ASPECTS, MODEL_PATH, MODEL_PATH_OLD

# Paths
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'data', 'predictions')
SENTIMENT_NAMES = ['NEG', 'POS', 'NEU']

ML_MODEL_PATHS = {
    'logistic_regression': os.path.join(BASE_DIR, 'models', 'logistic_regression_absa', 'logistic_regression_model.pkl'),
    'naive_bayes': os.path.join(BASE_DIR, 'models', 'naive_bayes_absa', 'naive_bayes_model.pkl'),
}

# Global predictor instance
predictor = None
_ml_predictors = {}
_torch_predictors = {}
_phobert_predictors = {}

TORCH_MODEL_SPECS = {
    'bilstm': {
        'dir': os.path.join(BASE_DIR, 'models', 'bilstm_absa'),
        'tokenizer': 'vinai/phobert-base',
        'class_name': 'BiLSTMForABSA',
        'needs_vocab': True,
        'label': 'BiLSTM',
    },
    'cnn_bilstm': {
        'dir': os.path.join(BASE_DIR, 'models', 'cnn_bilstm_absa'),
        'tokenizer': 'vinai/phobert-base',
        'class_name': 'CNNBiLSTMForABSA',
        'needs_vocab': True,
        'label': 'CNN-BiLSTM',
    },
    'xlm_roberta': {
        'dir': os.path.join(BASE_DIR, 'models', 'xlm_roberta_absa'),
        'tokenizer': 'xlm-roberta-base',
        'class_name': 'XLMRoBERTaForABSA',
        'needs_vocab': False,
        'label': 'XLM-RoBERTa',
    },
}


def _get_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for this model") from exc
    return torch


def _get_device() -> str:
    torch = _get_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_ml_predictor(model_key: str):
    from methods import LogisticRegressionABSA, NaiveBayesABSA

    if model_key in _ml_predictors:
        return _ml_predictors[model_key]

    model_path = ML_MODEL_PATHS.get(model_key)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        payload = pickle.load(f)

    if model_key == 'naive_bayes':
        model = NaiveBayesABSA()
    else:
        model = LogisticRegressionABSA()

    model.tfidf = payload['tfidf']
    model.mention_clfs = payload['mention_clfs']
    model.sentiment_clfs = payload['sentiment_clfs']

    _ml_predictors[model_key] = model
    return model


def _find_checkpoint(model_dir: str) -> Optional[str]:
    if not os.path.isdir(model_dir):
        return None
    candidates = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(model_dir, candidates[0])


def _load_torch_predictor(model_key: str, device: str):
    torch = _get_torch()
    from transformers import AutoTokenizer

    if model_key in _torch_predictors and _torch_predictors[model_key]['device'] == device:
        return _torch_predictors[model_key]

    spec = TORCH_MODEL_SPECS[model_key]
    checkpoint_path = _find_checkpoint(spec['dir'])
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found for {model_key} in {spec['dir']}")

    from methods import BiLSTMForABSA, CNNBiLSTMForABSA, XLMRoBERTaForABSA
    class_map = {
        'BiLSTMForABSA': BiLSTMForABSA,
        'CNNBiLSTMForABSA': CNNBiLSTMForABSA,
        'XLMRoBERTaForABSA': XLMRoBERTaForABSA,
    }
    model_class = class_map[spec['class_name']]

    tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer'])
    if spec['needs_vocab']:
        model = model_class(vocab_size=tokenizer.vocab_size)
    else:
        model = model_class(num_aspects=len(ASPECTS))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    payload = {
        'model': model,
        'tokenizer': tokenizer,
        'checkpoint': checkpoint_path,
        'device': device,
    }
    _torch_predictors[model_key] = payload
    return payload


def _load_phobert_predictor(model_key: str):
    if model_key in _phobert_predictors:
        return _phobert_predictors[model_key]

    if model_key == 'phobert_legacy':
        model_path = MODEL_PATH_OLD
    else:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    predictor_instance = PhoBERTPredictor(model_path=model_path, use_multipolarity=model_key != 'phobert_legacy')
    if not predictor_instance.model_loaded:
        raise RuntimeError("PhoBERT model failed to load")

    _phobert_predictors[model_key] = predictor_instance
    return predictor_instance


def _format_aspect_predictions(mention_scores, sentiment_scores, threshold: float = 0.5) -> Dict[str, Dict]:
    results = {}
    for idx, aspect in enumerate(ASPECTS):
        mentioned = mention_scores[idx] > threshold
        sentiments = []
        if mentioned:
            for s_idx, score in enumerate(sentiment_scores[idx]):
                if score > threshold:
                    sentiments.append(SENTIMENT_NAMES[s_idx])
            if not sentiments:
                sentiments = ['NEU']

        results[aspect] = {
            'mentioned': bool(mentioned),
            'sentiments': sentiments,
        }
    return results


def _predict_with_ml(model_key: str, text: str) -> Dict[str, Dict]:
    model = _load_ml_predictor(model_key)
    X = model.tfidf.transform([text])
    pred_m, pred_s, _, _ = model.predict(X)
    return _format_aspect_predictions(pred_m[0], pred_s[0])


def _predict_with_torch(model_key: str, text: str, device: str, max_length: int = 256) -> Dict[str, Dict]:
    torch = _get_torch()
    predictor_payload = _load_torch_predictor(model_key, device)
    tokenizer = predictor_payload['tokenizer']
    model = predictor_payload['model']

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits_m, logits_s = model(input_ids, attention_mask)
        probs_m = torch.sigmoid(logits_m)[0].cpu().numpy()
        probs_s = torch.sigmoid(logits_s)[0].cpu().numpy()

    return _format_aspect_predictions(probs_m, probs_s)


def _list_available_models() -> List[Dict[str, str]]:
    available = []

    if os.path.exists(MODEL_PATH):
        available.append({'id': 'phobert_multipolarity', 'label': 'PhoBERT (Multi-Polarity)'})
    if os.path.exists(MODEL_PATH_OLD):
        available.append({'id': 'phobert_legacy', 'label': 'PhoBERT (Legacy)'})

    if os.path.exists(ML_MODEL_PATHS['logistic_regression']):
        available.append({'id': 'logistic_regression', 'label': 'Logistic Regression (TF-IDF)'})
    if os.path.exists(ML_MODEL_PATHS['naive_bayes']):
        available.append({'id': 'naive_bayes', 'label': 'Naive Bayes (TF-IDF)'})

    for key, spec in TORCH_MODEL_SPECS.items():
        if _find_checkpoint(spec['dir']):
            available.append({'id': key, 'label': spec['label']})

    return available

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup using the shared Predictor class"""
    global predictor
    print(f"🚀 Starting ABSA API...")
    
    predictor = PhoBERTPredictor()
    print("✅ Predictor initialized (model loads on first request)")
    
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
    model: Optional[str] = None


class TriggerRequest(BaseModel):
    product_url: str
    max_reviews: int = 50
    model: Optional[str] = None


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
        "description": "ABSA model registry",
        "available_models": _list_available_models(),
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
    model_choice = (request.model or "phobert").lower()

    try:
        if model_choice == "phobert":
            if not predictor:
                raise HTTPException(status_code=503, detail="Predictor is not available.")

            if not predictor.model_loaded:
                if not predictor.load_model():
                    raise HTTPException(status_code=503, detail="Model is not loaded. Please verify ./models folder.")

            raw_result = predictor.predict_single(request.text)
            multipolarity_result = raw_result.get('multipolarity', {})

            return {
                "text": request.text,
                "model": "phobert",
                "aspects": multipolarity_result
            }

        if model_choice in ["phobert_multipolarity", "phobert_legacy"]:
            local_predictor = _load_phobert_predictor(model_choice)
            raw_result = local_predictor.predict_single(request.text)
            multipolarity_result = raw_result.get('multipolarity', {})
            return {
                "text": request.text,
                "model": model_choice,
                "aspects": multipolarity_result
            }

        if model_choice in ["logistic_regression", "naive_bayes"]:
            aspects = _predict_with_ml(model_choice, request.text)
            return {
                "text": request.text,
                "model": model_choice,
                "aspects": aspects
            }

        if model_choice in TORCH_MODEL_SPECS:
            device = _get_device()
            aspects = _predict_with_torch(model_choice, request.text, device)
            return {
                "text": request.text,
                "model": model_choice,
                "aspects": aspects
            }

        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_choice}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
def search_products(keyword: str, limit: int = 20):
    """Search Lazada products"""
    try:
        from app.lazada_search import search_lazada
        
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
                        "source": "react_frontend",
                        "model": request.model
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
                "model": request.model,
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
