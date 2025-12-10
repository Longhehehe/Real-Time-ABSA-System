import pandas as pd

def debug_load_data(file_path):
    print(f"Reading {file_path}")
    try:
        # Match utils.py logic exactly
        df_preview = pd.read_excel(file_path, header=None, nrows=5)
        print("Preview (First 3 rows):")
        print(df_preview.iloc[:3].to_string())
        
        header_row_idx = 0
        found = False
        for idx, row in df_preview.iterrows():
            # Check for 'Chất lượng sản phẩm' specifically
            values = [str(v).strip() for v in row.values]
            if 'Chất lượng sản phẩm' in values:
                header_row_idx = idx
                found = True
                print(f"Found header at index {idx}")
                break
        
        if not found:
            print("Could not find 'Chất lượng sản phẩm' in first 5 rows.")
            
        # Try loading with found header
        df = pd.read_excel(file_path, header=header_row_idx)
        print("Columns after loading:", df.columns.tolist())
        
        expected_aspects = [
            'Chất lượng sản phẩm', 'Trải nghiệm sử dụng', 'Đúng mô tả sản phẩm', 'Hiệu năng sản phẩm',
            'Giá cả', 'Khuyến mãi & voucher',
            'Vận chuyển & giao hàng', 'Đóng gói & bao bì',
            'Uy tín & thái độ shop', 'Dịch vụ chăm sóc khách hàng', 'Lỗi & bảo hành & hàng giả', 'Đổi trả & bảo hành'
        ]
        
        existing = [col for col in expected_aspects if col in df.columns]
        print("Existing aspects found:", existing)
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    debug_load_data(r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx')
