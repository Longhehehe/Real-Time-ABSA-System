import pandas as pd
from tqdm import tqdm
import json
import requests

# ============================
# 1. Cấu hình Ollama
# ============================
MODEL_NAME = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

print(f"Sử dụng Ollama model: {MODEL_NAME}")
print("Đảm bảo Ollama đang chạy (ollama serve) và model đã được pull (ollama pull qwen2.5:7b)")

# ============================
# 2. Dữ liệu
# ============================
INPUT_FILE = "test_flow_reviews_1.xlsx"
COL_REVIEW = "reviewContent"
OUTPUT_FILE = "full_reviews_1.xlsx"

CANONICAL = [
    "Chất lượng sản phẩm",
    "Giá cả",
    "Vận chuyển & Giao hàng",
    "Đóng gói & Bao bì",
    "Dịch vụ & CSKH",
    "Mô tả & Hình ảnh",
    "Lỗi, bảo hành, hàng giả",
    "Trải nghiệm sử dụng",
    "Uy tín & thái độ shop",
    "Khuyến mãi & voucher"
]

df = pd.read_excel(INPUT_FILE)
for asp in CANONICAL:
    df[asp] = 2

# ============================
# 3. Prompt
# ============================
SYSTEM_PROMPT = """
Bạn là AI phân tích cảm xúc cho đánh giá sản phẩm thương mại điện tử tiếng Việt.

Nhiệm vụ:
1. Trích xuất các "target" (khía cạnh) mà người dùng đang khen/chê trong câu review.
2. Chuẩn hóa mỗi target về một trong CÁC NHÓM CỐ ĐỊNH dưới đây (canonical aspect):

- "Chất lượng sản phẩm"
- "Giá cả"
- "Vận chuyển & Giao hàng"
- "Đóng gói & Bao bì"
- "Dịch vụ & CSKH"
- "Mô tả & Hình ảnh"
- "Lỗi, bảo hành, hàng giả"
- "Trải nghiệm sử dụng"
- "Uy tín & thái độ shop"
- "Khuyến mãi & voucher"

3. Gán score cho từng target: "1": "tích cực", "-1": "tiêu cực", "0": "trung lập" hoặc "2": "không đề cập".

Yêu cầu:
- Chỉ trả về JSON hợp lệ.
- Không giải thích, không thêm text ngoài JSON.
- Trả về JSON gồm đúng 10 keys.

Format JSON OUTPUT:

{
  "targets": [
    {
      "raw_target": "<target chi tiết, ví dụ: 'pin', 'camera', 'độ thoải mái'>",
      "canonical_aspect": "<một trong 10 nhóm canonical>",
      "sentiment": "<tích cực | tiêu cực | trung lập>"
    }
  ]
}
""".strip()


def build_prompt(review: str) -> str:
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Review: \"{review}\"\n"
        "Hãy trích xuất target và trả về JSON đúng format đã quy định."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


sentiment_map = {
    "tích cực": 1,
    "tiêu cực": -1,
    "trung lập": 0,
    "không đề cập": 2
}
# ============================
# 4. Hàm gọi Ollama API
# ============================
def call_ollama(prompt: str) -> str:
    """Gọi Ollama API để lấy response"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 128,
                    "top_p": 1.0
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Lỗi khi gọi Ollama: {e}")
        return ""


# ============================
# 5. Xử lý từng review
# ============================
print("Đang xử lý reviews...")
for i, review in enumerate(tqdm(df[COL_REVIEW].tolist())):
    prompt = build_prompt(str(review))
    text = call_ollama(prompt)

    # Lấy JSON gán vào cột
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
    except:
        data = {}

    # reset tất cả canonical = 2 ở mỗi dòng
    # for asp in CANONICAL:
    #     df.at[idx, asp] = 2

    # Nếu JSON hợp lệ
    if isinstance(data, dict) and "targets" in data:
        for item in data["targets"]:
            canonical = item.get("canonical_aspect")
            senti = item.get("sentiment")

            if canonical in CANONICAL:
                df.at[i, canonical] = sentiment_map.get(senti, 2)     # 1, -1, 0, 2

# ============================
# 6. Lưu kết quả
# ============================

df.to_excel(OUTPUT_FILE, index=False)
print(f"Đã lưu file: {OUTPUT_FILE}")
