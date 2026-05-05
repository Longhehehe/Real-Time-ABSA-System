"""
ABSA Dataset Relabeling Script using Mistral Medium 3.5 128B (via NVIDIA API)
==============================================================================
Reads all .xlsx files from the 'labeled' folder, sends each review to Mistral
for aspect-based sentiment relabeling according to ABSA_Labeling_Guideline.txt,
and saves the relabeled files to the 'New database' folder.
"""

import os
import re
import json
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
NVIDIA_API_KEY = "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"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "mistralai/mistral-medium-3.5-128b"

INPUT_DIR = Path("tesst_Data")
OUTPUT_DIR = Path("True_Test_Data")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many reviews to send per API call (batch)
BATCH_SIZE = 5
# Number of concurrent threads
MAX_WORKERS = 5
# Delay between API calls (seconds) to avoid rate limiting
API_DELAY = 1.0
# Max retries on API failure
MAX_RETRIES = 3

ASPECT_COLUMNS = [
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

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — The core labeling instructions for Mistral
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia đánh nhãn dữ liệu (Data Annotator) cho bài toán Aspect-Based Sentiment Analysis (ABSA) trên các đánh giá sản phẩm e-commerce tiếng Việt. Hệ thống này hỗ trợ MULTI-POLARITY, tức một khía cạnh có thể mang NHIỀU cảm xúc trái ngược cùng lúc.

═══════════════════════════════════════════════════════════════
NHIỆM VỤ CỦA BẠN
═══════════════════════════════════════════════════════════════
Với mỗi câu review, bạn phải phân tích và gán nhãn cảm xúc cho ĐÚNG 9 khía cạnh sau:

1. "Chất lượng sản phẩm" — Chất liệu, độ bền, độ hoàn thiện, thiết kế, màu sắc, kích thước vật lý của sản phẩm.
2. "Hiệu năng & Trải nghiệm" — Sản phẩm hoạt động có tốt không, mượt không, pin, tốc độ, trải nghiệm sử dụng thực tế.
3. "Đúng mô tả" — Hàng nhận được có giống hình ảnh/mô tả trên trang bán hàng không. Sai màu, sai size, sai mẫu, khác hình.
4. "Giá cả & Khuyến mãi" — Giá đắt/rẻ, có mã giảm giá, đáng tiền, xứng đáng, giá hời.
5. "Vận chuyển" — Giao hàng nhanh/chậm, phí ship, thái độ shipper, thời gian giao.
6. "Đóng gói" — Bọc kỹ, hộp móp méo, chống sốc, cách đóng hàng.
7. "Dịch vụ & Thái độ Shop" — Shop tư vấn nhiệt tình, phản hồi chậm, chu đáo, thái độ bán hàng.
8. "Bảo hành & Đổi trả" — Chính sách hậu mãi, quá trình trả hàng, hỗ trợ sau mua, bảo hành.
9. "Tính xác thực" — Hàng real/fake, chính hãng, hàng nhái, hàng giả.

═══════════════════════════════════════════════════════════════
CÁC GIÁ TRỊ NHÃN HỢP LỆ
═══════════════════════════════════════════════════════════════
Mỗi khía cạnh PHẢI được gán MỘT trong các giá trị sau:
•  1      → Tích cực (khen ngợi, hài lòng, đánh giá tốt)
• -1      → Tiêu cực (chê bai, phàn nàn, lỗi, không hài lòng)
•  0      → Trung tính (chỉ nhắc đến/trần thuật, không bộc lộ rõ khen hay chê)
•  2      → KHÔNG được nhắc đến trong câu review (khía cạnh này hoàn toàn vắng mặt)
• "1, -1" → Multi-Polarity: câu review có CẢ ý khen VÀ ý chê cho CÙNG MỘT khía cạnh

═══════════════════════════════════════════════════════════════
QUY TẮC ĐÁNH NHÃN BẮT BUỘC
═══════════════════════════════════════════════════════════════
1. ĐÁNH GIÁ ĐỘC LẬP TỪNG CỘT: Khách chê Giá đắt (-1) KHÔNG có nghĩa là Chất lượng cũng -1. Mỗi cột phải được phân tích riêng biệt dựa trên nội dung thực tế trong câu.

2. BẮT BUỘC DÙNG MULTI-POLARITY khi câu review có ý "quay xe" cho cùng một khía cạnh:
   - "Chất vải ok nhưng chỉ thừa nhiều" → Chất lượng: "1, -1"
   - "Đóng gói đẹp nhưng hộp bị rách" → Đóng gói: "1, -1"
   - "Máy chạy mượt nhưng thỉnh thoảng giật lag" → Hiệu năng: "1, -1"

3. PHÂN BIỆT RÕ RÀNG giữa các khía cạnh:
   - Giao sai hàng/sai màu/sai size → thuộc "Đúng mô tả" (KHÔNG phải Chất lượng)
   - Giao hàng nhanh/chậm → thuộc "Vận chuyển" (KHÔNG phải Dịch vụ Shop)
   - Vải mỏng/dày, màu đẹp/xấu → thuộc "Chất lượng sản phẩm"
   - Giá rẻ/đắt, đáng tiền → thuộc "Giá cả & Khuyến mãi"

4. NẾU KHÍA CẠNH KHÔNG ĐƯỢC NHẮC ĐẾN → gán giá trị 2. Đừng suy diễn hay đoán mò.

5. CÂU NGẮN/MƠ HỒ: Nếu câu review quá ngắn hoặc không rõ ý, chỉ gán nhãn cho những gì thực sự được đề cập. Ưu tiên gán 2 (không nhắc đến) hơn là đoán sai.

6. EMOJI VÀ NGÔN NGỮ KHÔNG CHUẨN: Nhiều review dùng emoji, viết tắt, tiếng lóng. Hãy hiểu ngữ cảnh để đánh nhãn chính xác.

═══════════════════════════════════════════════════════════════
VÍ DỤ MINH HỌA
═══════════════════════════════════════════════════════════════
Review: "Shop giao hàng siêu nhanh, cơ mà đóng gói bọc xốp cẩn thận nhưng hộp vẫn bị móp móp."
→ Chất lượng sản phẩm: 2 | Hiệu năng & Trải nghiệm: 2 | Đúng mô tả: 2 | Giá cả & Khuyến mãi: 2 | Vận chuyển: 1 | Đóng gói: "1, -1" | Dịch vụ & Thái độ Shop: 2 | Bảo hành & Đổi trả: 2 | Tính xác thực: 2

Review: "Điện thoại cầm đầm tay, màn hình đẹp nhưng pin thì tụt nhanh, giá 5 triệu là quá đắt."
→ Chất lượng sản phẩm: 1 | Hiệu năng & Trải nghiệm: -1 | Đúng mô tả: 2 | Giá cả & Khuyến mãi: -1 | Vận chuyển: 2 | Đóng gói: 2 | Dịch vụ & Thái độ Shop: 2 | Bảo hành & Đổi trả: 2 | Tính xác thực: 2

Review: "Tư vấn nhiệt tình, mua áo màu xanh mà giao nhầm màu đỏ nhưng mặc vẫn đẹp."
→ Chất lượng sản phẩm: 1 | Hiệu năng & Trải nghiệm: 2 | Đúng mô tả: -1 | Giá cả & Khuyến mãi: 2 | Vận chuyển: 2 | Đóng gói: 2 | Dịch vụ & Thái độ Shop: 1 | Bảo hành & Đổi trả: 2 | Tính xác thực: 2

═══════════════════════════════════════════════════════════════
ĐỊNH DẠNG OUTPUT BẮT BUỘC
═══════════════════════════════════════════════════════════════
Trả về ĐÚNG một JSON array, mỗi phần tử là một object chứa key "labels" với 9 giá trị tương ứng 9 khía cạnh theo THỨ TỰ trên. Giá trị Multi-Polarity phải là string (ví dụ "1, -1"). Giá trị đơn là số nguyên (1, -1, 0, 2).

KHÔNG giải thích, KHÔNG thêm text ngoài JSON. Chỉ trả về JSON array thuần túy.

Ví dụ output cho 2 reviews:
[
  {"labels": [1, 2, -1, 2, 2, 2, 1, 2, 2]},
  {"labels": [2, -1, 2, "1, -1", 1, 2, 2, 2, 2]}
]"""


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def call_mistral_api(reviews: list[str]) -> str:
    """Send a batch of reviews to Mistral and return the raw response text."""
    # Build user message with numbered reviews
    review_lines = []
    for i, review in enumerate(reviews, 1):
        review_lines.append(f"[Review {i}]: \"{review}\"")
    user_content = "\n".join(review_lines)
    user_content += f"\n\nHãy đánh nhãn {len(reviews)} reviews trên. Trả về JSON array với {len(reviews)} phần tử."

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.15,  # Low temperature for consistent labeling
        "top_p": 0.95,
        "stream": False,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  ⚠ Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠ API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    return ""


def parse_labels_from_response(response_text: str, expected_count: int) -> list[list]:
    """Parse Mistral's JSON response into a list of label lists."""
    # Try to extract JSON array from the response
    # Sometimes the model wraps it in markdown code blocks
    text = response_text.strip()
    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            print(f"  ✗ Failed to parse JSON from response: {text[:200]}...")
            return None

    if not isinstance(parsed, list) or len(parsed) != expected_count:
        print(f"  ✗ Expected {expected_count} items, got {len(parsed) if isinstance(parsed, list) else 'non-list'}")
        return None

    results = []
    for item in parsed:
        if isinstance(item, dict) and "labels" in item:
            results.append(item["labels"])
        elif isinstance(item, list):
            results.append(item)
        else:
            print(f"  ✗ Unexpected item format: {item}")
            return None

    return results


def format_label(val):
    """Convert a label value to the proper format for the DataFrame."""
    if isinstance(val, str):
        # Multi-polarity like "1, -1"
        return val
    elif isinstance(val, (int, float)):
        return int(val)
    return val


def process_batch(batch_idx, num_batches, start, end, batch_reviews):
    batch_size_actual = len(batch_reviews)
    print(f"   📦 [Thread] Batch {batch_idx+1}/{num_batches} (rows {start+1}-{end}) started")
    results = []
    
    try:
        response_text = call_mistral_api(batch_reviews)
        labels_list = parse_labels_from_response(response_text, batch_size_actual)

        if labels_list is None:
            print(f"   🔄 [Thread] Batch {batch_idx+1} Retrying individually...")
            for i, review in enumerate(batch_reviews):
                try:
                    resp = call_mistral_api([review])
                    single_labels = parse_labels_from_response(resp, 1)
                    if single_labels and len(single_labels[0]) == 9:
                        results.append((start + i, single_labels[0]))
                    else:
                        print(f"      ✗ [Thread] Row {start + i + 1} failed")
                except Exception as e:
                    print(f"      ✗ [Thread] Row {start + i + 1} error: {e}")
                time.sleep(API_DELAY)
        else:
            for i, labels in enumerate(labels_list):
                if len(labels) == 9:
                    results.append((start + i, labels))
                else:
                    print(f"      ✗ [Thread] Row {start + i + 1}: expected 9 labels, got {len(labels)}")
            print(f"      ✓ [Thread] Labeled {batch_size_actual} reviews successfully in Batch {batch_idx+1}")
            
    except Exception as e:
        print(f"      ✗ [Thread] Batch {batch_idx+1} failed: {e}")

    # Optionally sleep to respect rate limits if threads overlap too much
    time.sleep(API_DELAY)
    return results


def process_file(filepath: Path):
    """Process a single Excel file: relabel all reviews and save to output dir."""
    print(f"\n{'='*70}")
    print(f"📂 Processing: {filepath.name}")
    print(f"{'='*70}")

    df = pd.read_excel(filepath)
    total_rows = len(df)
    print(f"   Total reviews: {total_rows}")

    # Clear existing labels
    for col in ASPECT_COLUMNS:
        if col in df.columns:
            df[col] = None

    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    
    batches = []
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total_rows)
        batch_reviews = df["reviewContent"].iloc[start:end].tolist()
        batch_reviews = [str(r) if pd.notna(r) else "" for r in batch_reviews]
        batches.append((batch_idx, num_batches, start, end, batch_reviews))

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, *b): b for b in batches}
        
        for future in as_completed(futures):
            b = futures[future]
            try:
                results = future.result()
                for row_idx, labels in results:
                    for j, col in enumerate(ASPECT_COLUMNS):
                        df.at[row_idx, col] = format_label(labels[j])
                    success_count += 1
                fail_count += (len(b[4]) - len(results))
            except Exception as exc:
                print(f"      ✗ Batch {b[0]+1} generated an exception: {exc}")
                fail_count += len(b[4])

    # Save to output directory
    output_path = OUTPUT_DIR / filepath.name
    df.to_excel(output_path, index=False)
    print(f"\n   💾 Saved to: {output_path}")
    print(f"   ✓ Success: {success_count} | ✗ Failed: {fail_count}")
    return success_count, fail_count


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🚀 ABSA RELABELING TOOL — Mistral Medium 3.5 128B via NVIDIA API")
    print("=" * 70)

    # Find all xlsx files in the labeled folder
    files = sorted(INPUT_DIR.glob("*.xlsx"))
    print(f"\n📁 Found {len(files)} files in '{INPUT_DIR}':")
    for f in files:
        print(f"   - {f.name}")

    if not files:
        print("❌ No .xlsx files found in the 'labeled' folder!")
        return

    total_success = 0
    total_fail = 0

    for filepath in files:
        s, f = process_file(filepath)
        total_success += s
        total_fail += f

    print(f"\n{'='*70}")
    print(f"🏁 COMPLETED — All files processed!")
    print(f"   Total Success: {total_success}")
    print(f"   Total Failed:  {total_fail}")
    print(f"   Output folder: '{OUTPUT_DIR}'")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
