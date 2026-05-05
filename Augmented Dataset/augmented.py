"""
augmentation.py — ABSA Data Augmentation Pipeline (Multi-threaded)
====================================================================
Mục tiêu: Tăng cường dữ liệu Multi-polarity [1, -1] từ ~567 → ~2,000 samples
Strategies:
    1. Synonym Replacement  (offline — dùng ThreadPoolExecutor)
    2. Back-Translation     (I/O-bound, dùng ThreadPoolExecutor)
    3. LLM Paraphrase       (I/O-bound API calls, dùng ThreadPoolExecutor)
    4. LLM Synthetic        (I/O-bound API calls, dùng ThreadPoolExecutor)

Threading design:
  - Offline strategies (1, 2): workers = min(os.cpu_count(), MAX_OFFLINE_WORKERS)
  - LLM strategies    (3, 4): workers = MAX_LLM_WORKERS  (cap để tránh rate-limit)
  - Strategies chạy song song với nhau (outer parallelism)
  - Mỗi strategy xử lý tasks song song (inner parallelism)
  - Thread-safe progress bar (tqdm + Lock)
  - Rate limiter cho LLM (token bucket — đảm bảo không vượt RPS)

Usage:
    python augmentation.py --input  data/reviews.csv \\
                           --output data/reviews_augmented.csv \\
                           --target 2000 \\
                           --strategies all \\
                           --api-key  <NVIDIA_API_KEY> \\
                           --llm-workers 8 \\
                           --offline-workers auto
"""

# ─────────────────────────── Imports ───────────────────────────
from __future__ import annotations

import argparse
import ast
import logging
import os
import random
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from tqdm import tqdm

# ─────────────────────────── Logging ───────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────── Configuration ─────────────────────────
ASPECT_COLUMNS: list[str] = [
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

REVIEW_COLUMN   = "reviewContent"
LABEL_NONE      = 2
LABEL_NEG       = -1
LABEL_POS       = 1

# Mistral / NVIDIA API
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL   = "mistralai/mistral-large-3-675b-instruct-2512"

# Worker caps
MAX_LLM_WORKERS     = 4    # giới hạn LLM threads để tránh 429
MAX_OFFLINE_WORKERS = 4    # giới hạn offline threads

# Rate limiter mặc định cho LLM
DEFAULT_LLM_RPS = 5.0      # requests per second

SYNONYM_DICT: dict[str, list[str]] = {
    "tốt": ["ổn", "tuyệt", "hay", "được", "ok", "xuất sắc"],
    "xấu": ["tệ", "kém", "dở", "không tốt", "thất vọng"],
    "nhanh": ["mau", "lẹ", "kịp thời", "tốc độ"],
    "chậm": ["trễ", "muộn", "lâu", "chờ lâu"],
    "đẹp": ["xinh", "bắt mắt", "sang", "tinh tế", "chất lượng"],
    "rẻ": ["giá tốt", "hợp lý", "phải chăng", "tiết kiệm"],
    "đắt": ["mắc", "cao", "pricey", "không rẻ"],
    "hài lòng": ["thích", "vừa ý", "ok", "ưng", "mãn nguyện"],
    "thất vọng": ["buồn", "không hài lòng", "chán", "tiếc"],
    "giao hàng": ["vận chuyển", "ship", "giao", "nhận hàng"],
    "đóng gói": ["bao bì", "packaging", "hộp", "túi"],
    "shop": ["cửa hàng", "người bán", "seller"],
    "sản phẩm": ["hàng", "món", "item", "đồ"],
    "chất lượng": ["chất", "quality", "độ bền"],
    "dịch vụ": ["support", "hỗ trợ", "CSKH", "phục vụ"],
    "bảo hành": ["warranty", "đổi trả", "hoàn tiền", "bảo đảm"],
    "mô tả": ["hình ảnh", "thông tin", "quảng cáo", "listing"],
    "thật": ["chính hãng", "authentic", "real", "genuine"],
    "fake": ["giả", "nhái", "hàng giả", "không thật"],
    "tuyệt vời": ["tuyệt", "quá tốt", "hoàn hảo", "xuất sắc", "5 sao"],
    "ổn": ["được", "bình thường", "không tệ", "tạm"],
    "nhân viên": ["staff", "người bán", "shipper", "tư vấn"],
    "khuyến mãi": ["sale", "giảm giá", "ưu đãi", "voucher", "deal"],
}


# ═══════════════════════════════════════════════════════════════
#  RATE LIMITER — Token Bucket (thread-safe)
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token bucket rate limiter — thread-safe.
    Đảm bảo tổng số requests/giây không vượt `rps` dù chạy đa luồng.
    """
    def __init__(self, rps: float):
        self.min_gap  = 1.0 / rps
        self._lock    = threading.Lock()
        self._last_ts = 0.0

    def acquire(self) -> None:
        """Block cho đến khi được phép gửi request tiếp theo."""
        with self._lock:
            now  = time.monotonic()
            wait = self._last_ts + self.min_gap - now
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()


# ═══════════════════════════════════════════════════════════════
#  THREAD-SAFE PROGRESS BAR
# ═══════════════════════════════════════════════════════════════

class SafeBar:
    """tqdm wrapper an toàn khi cập nhật từ nhiều thread."""
    def __init__(self, total: int, desc: str):
        self._pbar = tqdm(total=total, desc=desc, dynamic_ncols=True)
        self._lock = threading.Lock()

    def update(self, n: int = 1) -> None:
        with self._lock:
            self._pbar.update(n)

    def close(self) -> None:
        self._pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ═══════════════════════════════════════════════════════════════
#  HELPER: Parse & Detect Multi-polarity
# ═══════════════════════════════════════════════════════════════

def parse_label(value: Any) -> list[int] | int | None:
    """Chuyển giá trị ô label sang Python native."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except ValueError:
        pass # Bỏ qua lỗi mảng/list không check được pd.isna()

    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    try:
        parsed = ast.literal_eval(s)
        return [int(v) for v in parsed] if isinstance(parsed, list) else int(parsed)
    except Exception:
        pass
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def is_multi_polarity(label: Any) -> bool:
    parsed = parse_label(label)
    if not isinstance(parsed, list):
        return False
    vals = set(parsed)
    return LABEL_POS in vals and LABEL_NEG in vals


def has_multi_polarity_row(row: pd.Series) -> bool:
    return any(
        col in row.index and is_multi_polarity(row[col])
        for col in ASPECT_COLUMNS
    )


def label_to_str(label: Any) -> str:
    parsed = parse_label(label)
    if parsed is None:
        return "null"
    return str(parsed) if isinstance(parsed, list) else str(parsed)


def label_summary(row: pd.Series) -> str:
    parts = [
        f"  - {col}: {label_to_str(row[col])}"
        for col in ASPECT_COLUMNS
        if col in row.index and parse_label(row[col]) not in (None, 0)
    ]
    return "\n".join(parts) if parts else "(không có label)"


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 1: Synonym Replacement (multi-threaded)
# ═══════════════════════════════════════════════════════════════

def synonym_replacement(text: str, rng: random.Random, n: int = 2) -> str:
    """Thay thế n từ trong text bằng từ đồng nghĩa (thread-safe vì dùng rng riêng)."""
    words = text.split()
    replaceable = [(i, w) for i, w in enumerate(words) if w.lower() in SYNONYM_DICT]
    if not replaceable:
        return text
    rng.shuffle(replaceable)
    replaced, count = list(words), 0
    for i, w in replaceable:
        if count >= n:
            break
        chosen = rng.choice(SYNONYM_DICT[w.lower()])
        replaced[i] = chosen.capitalize() if w[0].isupper() else chosen
        count += 1
    return " ".join(replaced)


def _synonym_task(pool: list[dict], task_id: int) -> dict | None:
    rng = random.Random(task_id)          # seed riêng mỗi task → reproducible
    row = deepcopy(rng.choice(pool))
    original = str(row[REVIEW_COLUMN])
    new_text = synonym_replacement(original, rng, n=rng.randint(1, 3))
    if new_text == original:
        return None
    row[REVIEW_COLUMN]  = new_text
    row["aug_strategy"] = "synonym"
    return row


def augment_synonym(df_minority: pd.DataFrame, n_needed: int, n_workers: int) -> pd.DataFrame:
    log.info(f"[S1] Synonym Replacement — {n_needed} samples | {n_workers} workers")
    pool   = df_minority.to_dict("records")
    result : list[dict] = []
    lock   = threading.Lock()
    task_counter = 0

    with SafeBar(n_needed, "S1 Synonym") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            pending: deque[Future] = deque()
            max_tasks = n_needed * 6

            # Submit initial batch
            batch = min(n_workers * 4, max_tasks)
            for _ in range(batch):
                pending.append(exe.submit(_synonym_task, pool, task_counter))
                task_counter += 1

            while pending:
                fut = pending.popleft()
                row_out = fut.result()

                with lock:
                    done = len(result)
                    if row_out is not None and done < n_needed:
                        result.append(row_out)
                        pbar.update(1)
                        done += 1

                if done < n_needed and task_counter < max_tasks:
                    pending.append(exe.submit(_synonym_task, pool, task_counter))
                    task_counter += 1

    if len(result) < n_needed:
        log.warning(f"[S1] Chỉ tạo được {len(result)}/{n_needed} (từ điển hạn chế)")
    return pd.DataFrame(result)


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 2: Back-Translation (multi-threaded)
# ═══════════════════════════════════════════════════════════════

# Thread-local Translator instance (mỗi thread giữ 1 riêng)
_tl = threading.local()

def _get_translator():
    if not hasattr(_tl, "tr"):
        try:
            from googletrans import Translator  # type: ignore
            _tl.tr = Translator()
        except ImportError:
            log.warning("[S2] googletrans chưa cài. pip install googletrans==4.0.0-rc1")
            _tl.tr = None
    return _tl.tr


def _back_translate_task(row: dict) -> dict:
    row = deepcopy(row)
    original = str(row[REVIEW_COLUMN])
    tr = _get_translator()
    new_text = original

    if tr is not None:
        for attempt in range(3):
            try:
                en = tr.translate(original, src="vi", dest="en")
                time.sleep(0.15)
                vi = tr.translate(en.text, src="en", dest="vi")
                time.sleep(0.15)
                new_text = vi.text
                break
            except Exception as e:
                log.debug(f"[S2] translate attempt {attempt+1}: {e}")
                time.sleep(0.3 * (attempt + 1))

    if new_text == original:
        # Fallback: synonym với RNG mới
        new_text = synonym_replacement(original, random.Random(), n=2)

    row[REVIEW_COLUMN]  = new_text
    row["aug_strategy"] = "back_translation" if new_text != original else "back_translation_fallback"
    return row


def augment_back_translation(df_minority: pd.DataFrame, n_needed: int, n_workers: int) -> pd.DataFrame:
    log.info(f"[S2] Back-Translation — {n_needed} samples | {n_workers} workers")
    pool  = df_minority.to_dict("records")
    rng   = random.Random(2024)
    tasks = [deepcopy(rng.choice(pool)) for _ in range(n_needed)]

    result: list[dict] = []
    with SafeBar(n_needed, "S2 BackTrans") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            for fut in as_completed(exe.submit(_back_translate_task, t) for t in tasks):
                try:
                    result.append(fut.result())
                    pbar.update(1)
                except Exception as e:
                    log.debug(f"[S2] Worker error: {e}")

    return pd.DataFrame(result)


# ═══════════════════════════════════════════════════════════════
#  LLM Client — thread-safe, rate-limited, connection-pooled
# ═══════════════════════════════════════════════════════════════

class MistralClient:
    """
    Thread-safe wrapper cho NVIDIA / Mistral API.
    - Shared RateLimiter: tất cả threads dùng chung 1 token bucket
    - Per-thread requests.Session: tận dụng HTTP keep-alive
    - Exponential backoff khi gặp 429 / 5xx
    """
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, rps: float = DEFAULT_LLM_RPS):
        self.api_key  = api_key
        self.model    = model
        self.url      = f"{NVIDIA_BASE_URL}/chat/completions"
        self.headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        self._limiter = RateLimiter(rps)
        self._tl      = threading.local()

    def _session(self) -> requests.Session:
        """Thread-local session để tái sử dụng kết nối TCP."""
        if not hasattr(self._tl, "sess"):
            s = requests.Session()
            s.headers.update(self.headers)
            self._tl.sess = s
        return self._tl.sess

    def complete(self, prompt: str, temperature: float = 0.85, max_tokens: int = 300) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        for attempt in range(4):
            self._limiter.acquire()          # chặn nếu vượt RPS
            try:
                resp = self._session().post(self.url, json=payload, timeout=45)
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    log.warning(f"429 Rate-limit — chờ {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as e:
                log.debug(f"HTTP {resp.status_code} attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            except Exception as e:
                log.debug(f"API error attempt {attempt+1}: {e}")
                time.sleep(1)
        return ""


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 3: LLM Paraphrase (multi-threaded)
# ═══════════════════════════════════════════════════════════════

PARAPHRASE_PROMPT = """\
Bạn là chuyên gia viết review sản phẩm tiếng Việt.

Hãy viết lại review bên dưới bằng cách khác (paraphrase) nhưng phải:
- Giữ nguyên ý nghĩa và cảm xúc
- Giữ nguyên mọi aspect được đề cập và sentiment của chúng
- Dùng từ ngữ/cấu trúc câu khác
- Tự nhiên như người Việt thật sự viết
- KHÔNG thêm/bớt thông tin

Review gốc:
"{review}"

Aspect & Sentiment cần giữ nguyên:
{label_info}

Chỉ trả về review mới, không giải thích, không dấu ngoặc kép."""


def _paraphrase_task(row: dict, client: MistralClient) -> dict | None:
    original   = str(row[REVIEW_COLUMN])
    label_info = label_summary(pd.Series(row))
    prompt     = PARAPHRASE_PROMPT.format(review=original, label_info=label_info)
    new_text   = client.complete(prompt, temperature=0.85)
    if not new_text or new_text == original or len(new_text) < 10:
        return None
    row = deepcopy(row)
    row[REVIEW_COLUMN]  = new_text
    row["aug_strategy"] = "llm_paraphrase"
    return row


def augment_paraphrase(
    df_minority: pd.DataFrame, n_needed: int, client: MistralClient, n_workers: int
) -> pd.DataFrame:
    log.info(f"[S3] LLM Paraphrase — {n_needed} samples | {n_workers} workers")
    pool   = df_minority.to_dict("records")
    result : list[dict] = []
    lock   = threading.Lock()
    rng    = random.Random(3333)
    submitted = 0
    max_tasks = n_needed * 3

    with SafeBar(n_needed, "S3 Paraphrase") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            pending: deque[Future] = deque()

            init = min(n_workers * 2, max_tasks)
            for _ in range(init):
                pending.append(exe.submit(_paraphrase_task, deepcopy(rng.choice(pool)), client))
                submitted += 1

            while pending:
                fut     = pending.popleft()
                row_out = fut.result()

                with lock:
                    done = len(result)
                    if row_out is not None and done < n_needed:
                        result.append(row_out)
                        pbar.update(1)
                        done += 1

                if done < n_needed and submitted < max_tasks:
                    pending.append(exe.submit(_paraphrase_task, deepcopy(rng.choice(pool)), client))
                    submitted += 1

    if len(result) < n_needed:
        log.warning(f"[S3] Chỉ tạo được {len(result)}/{n_needed}")
    return pd.DataFrame(result)


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 4: LLM Synthetic Generation (multi-threaded)
# ═══════════════════════════════════════════════════════════════


SYNTHETIC_PROMPT = """\
Bạn là khách hàng đang viết đánh giá (review) thực tế trên Shopee/Lazada.
Hãy viết 1 review ngắn gọn, đi thẳng vào vấn đề, thể hiện **đúng** các cảm xúc sau:
{label_info}

Phong cách viết:
- Ngôn ngữ đời thường, tự nhiên, KHÔNG văn chương.
- Có thể viết tắt (vd: sp, hđ, ok, ko, đc, shop, tks, bthuong...), dùng icon nếu muốn.
- Viết ngắn gọn, không cần quá rõ ràng, có thể lồng ghép các khía cạnh vào nhau.

Quy ước sentiment:
  1      = tích cực (positive)
  -1     = tiêu cực (negative)
  1, -1  = vừa tích cực vừa tiêu cực (mixed)

Chỉ trả về review, không giải thích, không dấu ngoặc kép."""


def _synthetic_task(combo: dict[str, Any], client: MistralClient) -> dict | None:
    label_info = "\n".join(f"  - {k}: {v}" for k, v in combo.items())
    prompt     = SYNTHETIC_PROMPT.format(label_info=label_info)
    new_text   = client.complete(prompt, temperature=0.9, max_tokens=250)
    if not new_text or len(new_text.split()) < 5:
        return None
    row: dict[str, Any] = {REVIEW_COLUMN: new_text, "aug_strategy": "llm_synthetic"}
    for col in ASPECT_COLUMNS:
        row[col] = combo.get(col, LABEL_NONE)
    return row


def augment_synthetic(n_needed: int, client: MistralClient, n_workers: int) -> pd.DataFrame:
    log.info(f"[S4] LLM Synthetic — {n_needed} samples | {n_workers} workers")
    result    : list[dict]  = []
    lock      = threading.Lock()
    def _next_combo() -> dict[str, Any]:
        c = {}
        # 1. Chọn số lượng khía cạnh từ 2 đến 5
        num_aspects = random.randint(2, 5)
        chosen_aspects = random.sample(ASPECT_COLUMNS, num_aspects)
        
        # 2. Chọn mục tiêu chính
        goal = random.choices(["mp", "warranty", "auth"], weights=[40, 30, 30])[0]
        
        mp_count = 0
        max_mp_per_sample = 2 # Giới hạn tối đa 2 khía cạnh Multi-polarity mỗi câu

        # Xử lý khía cạnh đầu tiên dựa trên mục tiêu
        primary = chosen_aspects[0]
        if goal == "mp":
            c[primary] = "1, -1"
            mp_count += 1
        elif goal == "warranty":
            target_aspect = "Bảo hành & Đổi trả"
            c[target_aspect] = random.choice([1, -1, "1, -1"])
            if c[target_aspect] == "1, -1": mp_count += 1
            if target_aspect in chosen_aspects: chosen_aspects.remove(target_aspect)
        elif goal == "auth":
            target_aspect = "Tính xác thực"
            c[target_aspect] = random.choice([1, -1, "1, -1"])
            if c[target_aspect] == "1, -1": mp_count += 1
            if target_aspect in chosen_aspects: chosen_aspects.remove(target_aspect)

        # 3. Gán nhãn cho các khía cạnh còn lại
        for aspect in chosen_aspects:
            if aspect in c: continue
            
            # Chỉ cho phép MP nếu chưa đạt giới hạn và xác suất thấp (15%)
            if mp_count < max_mp_per_sample and random.random() < 0.15:
                c[aspect] = "1, -1"
                mp_count += 1
            else:
                c[aspect] = random.choice([1, -1])
        return c

    submitted = 0
    max_tasks = n_needed * 3

    with SafeBar(n_needed, "S4 Synthetic") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            pending: deque[Future] = deque()

            init = min(n_workers * 2, max_tasks)
            for _ in range(init):
                pending.append(exe.submit(_synthetic_task, _next_combo(), client))
                submitted += 1

            while pending:
                fut     = pending.popleft()
                row_out = fut.result()

                with lock:
                    done = len(result)
                    if row_out is not None and done < n_needed:
                        result.append(row_out)
                        pbar.update(1)
                        done += 1

                if done < n_needed and submitted < max_tasks:
                    pending.append(exe.submit(_synthetic_task, _next_combo(), client))
                    submitted += 1

    if len(result) < n_needed:
        log.warning(f"[S4] Chỉ tạo được {len(result)}/{n_needed}")
    return pd.DataFrame(result)


# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def analyze_distribution(df: pd.DataFrame) -> dict[str, int]:
    mp_mask = df.apply(has_multi_polarity_row, axis=1)
    stats: dict[str, int] = {
        "total_multi_polarity": int(mp_mask.sum()),
        "total_rows":           len(df),
    }
    for col in ASPECT_COLUMNS:
        if col in df.columns:
            stats[f"mp_{col}"] = int(df[col].apply(is_multi_polarity).sum())
    return stats


def _resolve_workers(val: Any, cap: int) -> int:
    n = os.cpu_count() or 4 if str(val).lower() == "auto" else int(val)
    return max(1, min(n, cap))


def run_augmentation(
    df:              pd.DataFrame,
    target:          int,
    strategies:      list[int],
    api_key:         str | None,
    model:           str,
    llm_workers:     int,
    offline_workers: int,
    llm_rps:         float,
) -> pd.DataFrame:
    stats      = analyze_distribution(df)
    current_mp = stats["total_multi_polarity"]
    log.info(f"Hiện tại : {current_mp:,} Multi-polarity / {len(df):,} rows")
    log.info(f"Mục tiêu : {target:,} Multi-polarity rows")

    n_needed = max(0, target - current_mp)
    if n_needed == 0:
        log.info("Dataset đã đủ. Không cần augment.")
        return df

    log.info(f"Cần sinh : {n_needed:,} samples")
    log.info(f"Workers  : offline={offline_workers}, llm={llm_workers}, rps={llm_rps}")

    # Lọc minority pool
    mp_mask     = df.apply(has_multi_polarity_row, axis=1)
    df_minority = df[mp_mask].copy()
    if len(df_minority) == 0:
        log.error("Không tìm thấy row Multi-polarity nào!")
        return df
    log.info(f"Minority pool: {len(df_minority):,} rows")

    # Kiểm tra API key
    client: MistralClient | None = None
    if any(s in strategies for s in (3, 4)):
        if not api_key:
            log.warning("Strategies 3/4 cần --api-key. Bỏ qua.")
            strategies = [s for s in strategies if s not in (3, 4)]
        else:
            client = MistralClient(api_key=api_key, model=model, rps=llm_rps)

    if not strategies:
        log.error("Không còn strategy nào!")
        return df

    # Phân bổ quota đều
    n_s       = len(strategies)
    quota_per = n_needed // n_s
    remainder = n_needed % n_s
    quotas    = {s: quota_per + (1 if i < remainder else 0) for i, s in enumerate(strategies)}
    log.info(f"Quota phân bổ: {quotas}")

    # Map strategy → callable
    strategy_fns: dict[int, Callable[[], pd.DataFrame]] = {}
    if 1 in strategies:
        q = quotas[1]
        strategy_fns[1] = lambda q=q: augment_synonym(df_minority, q, offline_workers)
    if 2 in strategies:
        q = quotas[2]
        strategy_fns[2] = lambda q=q: augment_back_translation(df_minority, q, offline_workers)
    if 3 in strategies and client:
        q = quotas[3]
        strategy_fns[3] = lambda q=q: augment_paraphrase(df_minority, q, client, llm_workers)
    if 4 in strategies and client:
        q = quotas[4]
        strategy_fns[4] = lambda q=q: augment_synthetic(q, client, llm_workers)

    # ── Outer parallelism: chạy tất cả strategies đồng thời ──
    log.info(f"Chạy {len(strategy_fns)} strategies song song...")
    augmented_dfs: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=len(strategy_fns)) as exe:
        fut_map = {exe.submit(fn): s for s, fn in strategy_fns.items()}
        for fut in as_completed(fut_map):
            s = fut_map[fut]
            try:
                aug_df = fut.result()
                augmented_dfs.append(aug_df)
                log.info(f"[S{s}] ✓ Hoàn thành — {len(aug_df):,} samples")
            except Exception as e:
                log.error(f"[S{s}] ✗ Thất bại: {e}")

    if not augmented_dfs:
        log.warning("Không có augmented data nào!")
        return df

    df_aug   = pd.concat(augmented_dfs, ignore_index=True)
    df_final = pd.concat([df, df_aug], ignore_index=True)

    after = analyze_distribution(df_final)
    log.info("─" * 55)
    log.info("Kết quả augmentation:")
    log.info(f"  Tổng rows   : {len(df):,} → {len(df_final):,}  (+{len(df_aug):,})")
    log.info(f"  Multi-polar : {current_mp:,} → {after['total_multi_polarity']:,}")
    log.info("─" * 55)

    return df_final


# ═══════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> bool:
    ok = True
    missing = [c for c in [REVIEW_COLUMN] + ASPECT_COLUMNS if c not in df.columns]
    if missing:
        log.warning(f"Thiếu cột: {missing}")

    empty = int(df[REVIEW_COLUMN].isna().sum()) + int((df[REVIEW_COLUMN] == "").sum())
    if empty > 0:
        log.warning(f"{empty} review rỗng")
        ok = False

    valid_single = {-1, 0, 1, 2}  # Thêm 2 vì LABEL_NONE = 2
    
    def is_invalid(v):
        val = parse_label(v)
        if isinstance(val, list):
            return not all(x in valid_single for x in val)
        return val not in valid_single

    for col in ASPECT_COLUMNS:
        if col not in df.columns:
            continue
        bad = df[col].apply(is_invalid).sum()
        if bad > 0:
            log.warning(f"Cột '{col}': {bad} label không hợp lệ")
            ok = False

    log.info("✅ Validation passed!" if ok else "⚠️  Validation có cảnh báo")
    return ok


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ABSA Multi-polarity Augmentation Pipeline (Multi-threaded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tất cả strategies, 2000 samples, auto workers
  python augmentation.py --input data/reviews.csv --target 2000 --api-key sk-xxx

  # Chỉ offline, 8 workers
  python augmentation.py --input data/reviews.csv --target 2000 \\
      --strategies 1,2 --offline-workers 8

  # Chỉ LLM, 6 threads, giới hạn 3 req/s
  python augmentation.py --input data/reviews.csv --target 2000 \\
      --strategies 3,4 --api-key sk-xxx --llm-workers 6 --llm-rps 3
        """,
    )
    p.add_argument("--input",           required=True,               help="Path CSV gốc")
    p.add_argument("--output",          default="reviews_augmented.csv")
    p.add_argument("--target",          type=int,   default=50,     help="Số Multi-polarity mục tiêu")
    p.add_argument("--strategies",      default="4",                 help="'all' hoặc '1,2,3,4'")
    p.add_argument("--api-key",         default=None,                help="NVIDIA API key")
    p.add_argument("--model",           default=DEFAULT_MODEL)
    p.add_argument("--llm-workers",     default=8,                   help="Số thread LLM (int | 'auto')")
    p.add_argument("--offline-workers", default="auto",              help="Số thread offline (int | 'auto')")
    p.add_argument("--llm-rps",         type=float, default=DEFAULT_LLM_RPS,
                   help=f"LLM requests/giây (default: {DEFAULT_LLM_RPS})")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--no-validate",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    llm_w     = _resolve_workers(args.llm_workers,     MAX_LLM_WORKERS)
    offline_w = _resolve_workers(args.offline_workers, MAX_OFFLINE_WORKERS)
    strategies = (
        [1, 2, 3, 4] if args.strategies.lower() == "all"
        else [int(s.strip()) for s in args.strategies.split(",")]
    )

    log.info(f"Input      : {args.input}")
    log.info(f"Output     : {args.output}")
    log.info(f"Target     : {args.target:,} Multi-polarity samples")
    log.info(f"Strategies : {strategies}")
    log.info(f"Workers    : offline={offline_w}, llm={llm_w}, rps={args.llm_rps}")

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Đường dẫn không tồn tại: {args.input}")
        return

    if input_path.is_dir():
        log.info(f"Đang quét thư mục: {input_path}")
        # Lấy tất cả file excel/csv và sắp xếp theo tên
        all_files = sorted(list(input_path.glob("*.xlsx")) + list(input_path.glob("*.csv")))
        # Lấy 5 file đầu theo yêu cầu
        files_to_process = all_files[:5]
        
        if not files_to_process:
            log.error("Không tìm thấy file .xlsx hoặc .csv nào trong thư mục!")
            return
            
        log.info(f"Sẽ xử lý {len(files_to_process)} file đầu tiên:")
        for f in files_to_process:
            log.info(f"  - {f.name}")
            
        dfs = []
        for f in files_to_process:
            dfs.append(pd.read_excel(f) if f.suffix == '.xlsx' else pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Xử lý file đơn lẻ
        df = pd.read_excel(input_path) if input_path.suffix == '.xlsx' else pd.read_csv(input_path)
    
    log.info(f"Tổng cộng nạp được {len(df):,} dòng dữ liệu.")

    df_final = run_augmentation(
        df              = df,
        target          = args.target,
        strategies      = strategies,
        api_key         = args.api_key,
        model           = args.model,
        llm_workers     = llm_w,
        offline_workers = offline_w,
        llm_rps         = args.llm_rps,
    )

    if not args.no_validate:
        validate_dataset(df_final)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    if out.suffix == '.xlsx':
        df_final.to_excel(out, index=False)
    else:
        df_final.to_csv(out, index=False, encoding="utf-8-sig")
    
    log.info(f"✅ Saved {len(df_final):,} rows → {out}")


if __name__ == "__main__":
    main()