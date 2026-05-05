
import requests
import json
import re
from typing import Dict, List, Optional
import os

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

class OllamaPredictor:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
                                                                      
        self.api_base = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        if os.name == 'nt':                
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
            "format": "json"                                 
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                return self._parse_response(raw_response)
            else:
                print(f" Ollama API Error: {response.status_code} - {response.text}")
                return {a: None for a in ASPECTS}
        except Exception as e:
            print(f" Ollama Connection Error: {e}")
                                
            return {a: None for a in ASPECTS}

    def _parse_response(self, raw_response: str) -> Dict[str, str]:
        """Parse strict JSON from LLM response."""
        try:
                                   
            data = json.loads(raw_response)
        except json.JSONDecodeError:
                                                         
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except:
                    return {a: None for a in ASPECTS}
            else:
                return {a: None for a in ASPECTS}

        result = {}
        for aspect in ASPECTS:
            val = data.get(aspect)
            if val in ["POS", "NEG", "NEU"]:
                result[aspect] = val
            else:
                result[aspect] = None                                             
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """Predict a batch of reviews (Sequential for Ollama to avoid OOM)."""
        return [self.predict_single(t) for t in texts]

if __name__ == "__main__":
    predictor = OllamaPredictor()
    sample = "Dầu gội này thơm nhưng giá hơi chát."
    print("Input:", sample)
    print("Output:", predictor.predict_single(sample))
