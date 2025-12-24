
import requests
import json
import re
from typing import Dict, List, Optional
import os

# Aspects definition (MUST match absa_predictor.py for UI compatibility)
ASPECTS = [
    'Chất lượng sản phẩm',
    'Trải nghiệm sử dụng',
    'Đúng mô tả sản phẩm',
    'Hiệu năng sản phẩm',
    'Giá cả',
    'Khuyến mãi & voucher',
    'Vận chuyển & giao hàng',
    'Đóng gói & bao bì',
    'Uy tín & thái độ shop',
    'Dịch vụ chăm sóc khách hàng',
    'Lỗi & bảo hành & hàng giả',
    'Đổi trả & bảo hành'
]

class OllamaPredictor:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        # Docker internal host if running in container, else localhost
        self.api_base = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        if os.name == 'nt': # Windows/Local
             self.api_base = "http://localhost:11434"
             
        self.api_url = f"{self.api_base}/api/generate"
        print(f"🤖 Ollama initialized with model: {model_name} at {self.api_base}")

    def _construct_prompt(self, text: str) -> str:
        aspects_str = ", ".join([f'"{a}"' for a in ASPECTS])
        return f"""
        Analyze the sentiment of the following Vietnamese product review for specific aspects.
        Review: "{text}"

        Aspects to analyze: {aspects_str}
        Sentiments: POS (Positive), NEG (Negative), NEU (Neutral), None (Not mentioned).

        Return ONLY a JSON object where keys are aspects and values are sentiments. Do not include markdown formatting or explanations.
        Example: {{"Mùi hương": "POS", "Giá cả": "NEG"}}
        """

    def predict_single(self, text: str) -> Dict[str, str]:
        """Predict sentiment for a single review."""
        if not text or not text.strip():
            return {a: None for a in ASPECTS}

        prompt = self._construct_prompt(text)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json" # Enforce JSON mode if supported
        }

        # Use session for connection pooling
        if not hasattr(self, '_session'):
            self._session = requests.Session()
            
        # Retry logic for connection stability
        max_retries = 2  # Reduced retries
        for attempt in range(max_retries):
            try:
                response = self._session.post(self.api_url, json=payload, timeout=30)  # Reduced timeout
                if response.status_code == 200:
                    result = response.json()
                    raw_response = result.get("response", "")
                    return self._parse_response(raw_response)
                else:
                    print(f"⚠️ Ollama API Error (Attempt {attempt+1}/{max_retries}): {response.status_code}")
            except Exception as e:
                print(f"⚠️ Ollama Connection Error (Attempt {attempt+1}/{max_retries}): {e}")
                import time
                time.sleep(1)  # Reduced backoff
        
        # If all retries fail
        print(f"❌ Failed to connect to Ollama after {max_retries} attempts.")
        return {a: None for a in ASPECTS}

    def _parse_response(self, raw_response: str) -> Dict[str, str]:
        """Parse strict JSON from LLM response."""
        try:
            # Try direct JSON parse
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from potential markdown
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except:
                    return {a: None for a in ASPECTS}
            else:
                return {a: None for a in ASPECTS}

        # Normalize and fill missing
        result = {}
        for aspect in ASPECTS:
            val = data.get(aspect)
            if val in ["POS", "NEG", "NEU"]:
                result[aspect] = val
            else:
                result[aspect] = None # Map 'None' or missing to valid Python None
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Predict a batch of reviews using concurrent threads for better throughput."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_workers = 4  # Parallel requests to Ollama
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.predict_single, text): idx 
                             for idx, text in enumerate(texts)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"⚠️ Prediction failed for index {idx}: {e}")
                    results[idx] = {a: None for a in ASPECTS}
        
        return results

# Test
if __name__ == "__main__":
    predictor = OllamaPredictor()
    sample = "Dầu gội này thơm nhưng giá hơi chát."
    print("Input:", sample)
    print("Output:", predictor.predict_single(sample))
