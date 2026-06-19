# 🛒 Lazada Review Crawler

Công cụ crawl đánh giá sản phẩm từ Lazada Vietnam với giao diện Streamlit.

## ✨ Tính năng

- **Bulk Crawl**: Crawl tự động 80+ từ khóa sản phẩm
- **Balanced Mode**: Ưu tiên đánh giá 1,2,3 sao trước, sau đó cân bằng với 4,5 sao
- **Anti-Detection**: Random User-Agent, ẩn webdriver flags
- **Real-time Save**: Lưu dữ liệu sau mỗi sản phẩm (không mất dữ liệu khi dừng)
- **Data View**: Xem và tải xuống dữ liệu ngay trong app

## 📋 Yêu cầu

- Python 3.8+
- Google Chrome (phiên bản mới nhất)

## 🚀 Hướng dẫn cài đặt

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy ứng dụng

**Cách 1: Dùng file batch (Windows)**
```bash
run_lazada_crawler.bat
```

**Cách 2: Chạy trực tiếp**
```bash
cd lazada_crawler
streamlit run app.py
```

### Bước 3: Mở trình duyệt

Truy cập: http://localhost:8501

## 📖 Hướng dẫn sử dụng

### 1. Đăng nhập Lazada

1. Bấm **"Open Browser to Login"** ở sidebar
2. Đăng nhập tài khoản Lazada trong cửa sổ Chrome mở ra
3. Bấm **"I have Logged In - Save Cookies"**

### 2. Bulk Crawl (Crawl nhiều từ khóa)

1. Cuộn xuống phần **"🔄 Bulk Keywords Auto-Crawl"**
2. Nhập danh sách từ khóa (mỗi dòng 1 từ khóa)
3. Chọn **Số sản phẩm/từ khóa** (mặc định: 3)
4. Bấm **"🚀 Bắt đầu Bulk Crawl"**

### 3. Xem dữ liệu

1. Chuyển sang tab **"Data View"**
2. Xem bảng dữ liệu đã crawl
3. Bấm **"📥 Download CSV"** để tải về

## 📁 Cấu trúc thư mục

```
lazada_crawler/
├── app.py              # Giao diện Streamlit
├── crawler.py          # Logic crawl chính
├── utils.py            # Hàm tiện ích
├── requirements.txt    # Thư viện cần thiết
├── run_lazada_crawler.bat  # File chạy nhanh
└── bulk_crawl_progress.csv # Dữ liệu đang crawl mới (auto-save)
```

## ⚙️ Cấu hình

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| Số sản phẩm/từ khóa | 3 | Top N sản phẩm có nhiều review nhất |
| Balanced Mode | ✅ BẬT | Cân bằng số lượng đánh giá thấp/cao |
| Random delay | 5-15s | Delay giữa các sản phẩm |

## 📊 Định dạng output

| Cột | Mô tả |
|-----|-------|
| reviewContent | Nội dung đánh giá |
| rating | Số sao (1-5) |
| buyerName | Tên người mua |
| reviewTime | Thời gian đánh giá |
| likeCount | Số lượt thích |
| skuInfo | Thông tin variant |
| images | Ảnh đính kèm |
| has_text_content | True nếu có nội dung thực |
| keyword | Từ khóa tìm kiếm |
| product_title | Tên sản phẩm |

## ⚠️ Lưu ý

- **Không đóng cửa sổ Chrome** khi đang crawl
- Dữ liệu được **tự động lưu** vào `bulk_crawl_progress.csv` sau mỗi sản phẩm
- Các output lịch sử đã được gom vào `data/crawl_outputs/lazada_crawler/`
- Nếu bị chặn, **đợi 10-30 phút** rồi thử lại
- Cookie sẽ được **lưu tự động** cho lần sau

## 🛡️ Anti-Detection

Crawler đã tích hợp các biện pháp chống phát hiện:
- Random User-Agent (6 loại browser)
- Ẩn `navigator.webdriver`
- Random delay 5-15 giây
- Vietnamese language headers

---

📧 Nếu gặp lỗi, kiểm tra:
1. Chrome đã cài đặt chưa?
2. Đã đăng nhập Lazada chưa?
3. Kết nối internet ổn định?
