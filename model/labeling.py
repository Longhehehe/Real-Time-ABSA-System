import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json

# ============================
# 1. Chọn model vLLM
# ============================
# Nếu GPU T4: nên dùng bản AWQ cho đỡ nặng
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

print("Đang load model bằng vLLM...")
llm = LLM(model=MODEL_NAME)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    top_p=1.0
)

# ============================
# 2. Dữ liệu
# ============================
INPUT_FILE = "processed_reviews_1.xlsx"
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
llm = LLM(model=MODEL_NAME)
sampling = SamplingParams(temperature=0.0, max_tokens=128)

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


# ============================
# 4. Chuẩn bị text cho vLLM
# ============================

prompts = [build_prompt(str(r)) for r in df[COL_REVIEW].tolist()]
outputs = llm.generate(prompts, sampling)

for i, out in enumerate(outputs):
    text = out.outputs[0].text

    # Lấy JSON
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
    except:
        data = {}

    # Gán vào cột
    for asp in CANONICAL:
        if asp in data:
            df.at[i, asp] = data[asp]     # 1, -1, 0, 2

# ============================
# 6. Lưu kết quả
# ============================

df.to_excel(OUTPUT_FILE, index=False)
print(f"Đã lưu file: {OUTPUT_FILE}")
