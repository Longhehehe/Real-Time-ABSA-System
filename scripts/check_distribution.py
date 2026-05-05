import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import ast
from typing import Any

# Re-use constants from the project structure
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

ASPECT_MAP_EN = {
    'Chất lượng sản phẩm': 'Product Quality',
    'Hiệu năng & Trải nghiệm': 'Performance & Exp',
    'Đúng mô tả': 'Correct Desc',
    'Giá cả & Khuyến mãi': 'Price & Promo',
    'Vận chuyển': 'Shipping',
    'Đóng gói': 'Packaging',
    'Dịch vụ & Thái độ Shop': 'Service & Attitude',
    'Bảo hành & Đổi trả': 'Warranty & Returns',
    'Tính xác thực': 'Authenticity'
}

def parse_label(value: Any) -> list[int] | int | None:
    """Convert label cell value to Python native."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except ValueError:
        pass

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
        if len(parts) > 1:
            return [int(float(p)) for p in parts]
        return int(float(parts[0]))
    except ValueError:
        return None

def analyze_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"\n[ANALYZING]: {path.name}")
    print("="*70)

    if path.suffix == '.xlsx':
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    total_rows = len(df)
    print(f"Total samples: {total_rows:,}")

    stats = []

    for col in ASPECT_COLUMNS:
        if col not in df.columns:
            stats.append({"Aspect": col, "Status": "Missing"})
            continue
        
        parsed_col = df[col].apply(parse_label)
        
        pos = 0
        neg = 0
        neu = 0 
        none = 0 
        multi = 0 

        for val in parsed_col:
            if val is None or val == 2:
                none += 1
            elif isinstance(val, list):
                if 1 in val and -1 in val:
                    multi += 1
                elif 1 in val: pos += 1
                elif -1 in val: neg += 1
            elif val == 1:
                pos += 1
            elif val == -1:
                neg += 1
            elif val == 0:
                neu += 1

        mention_count = total_rows - none
        mention_pct = (mention_count / total_rows) * 100 if total_rows > 0 else 0
        
        stats.append({
            "Aspect": col,
            "Mentions": mention_count,
            "Pct": f"{mention_pct:.1f}%",
            "Pos": pos,
            "Neg": neg,
            "Neu": neu,
            "Multi": multi,
            "None": none
        })

    def has_mp(row):
        for col in ASPECT_COLUMNS:
            if col in row:
                val = parse_label(row[col])
                if isinstance(val, list) and 1 in val and -1 in val:
                    return True
        return False
    
    total_mp_sentences = df.apply(has_mp, axis=1).sum()

    print("\nLabel Distribution Details:")
    header = f"{'Aspect':<25} | {'Mentions':<12} | {'Pos':<6} | {'Neg':<6} | {'Neu':<6} | {'Multi':<6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for s in stats:
        aspect_en = ASPECT_MAP_EN.get(s['Aspect'], s['Aspect'])
        if s.get("Status") == "Missing":
            print(f"{aspect_en:<25} | MISSING")
        else:
            mention_str = f"{s['Mentions']:<4} ({s['Pct']:>5})"
            print(f"{aspect_en:<25} | {mention_str:<12} | {s['Pos']:<6} | {s['Neg']:<6} | {s['Neu']:<6} | {s['Multi']:<6}")
    
    print("-" * len(header))
    print(f"\nTOTAL MULTI-POLARITY SENTENCES: {total_mp_sentences}")
    print(f"Multi-polarity rate: {(total_mp_sentences/total_rows)*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="File path (.xlsx or .csv)")
    args = parser.parse_args()
    
    analyze_file(args.file)
