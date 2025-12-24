# 🐳 Hướng Dẫn Khởi Tạo Hệ Thống với Docker

Tài liệu này hướng dẫn chi tiết cách thiết lập và chạy dự án Real-Time ABSA System sử dụng Docker Compose.

## 1. Yêu Cầu Hệ Thống

*   **Docker Desktop**: Đã cài đặt và đang chạy (trên Windows khuyến nghị dùng WSL 2 backend).
*   **RAM**: Tối thiểu 8GB (khuyến nghị 16GB do chạy Spark và PhoBERT model).
*   **Disk**: Trống ít nhất 10GB.

## 2. Các Bước Cài Đặt

### Bước 1: Chuẩn Bị File `.env`
Tạo file `.env` tại thư mục gốc của dự án (`c:\Users\Long\Documents\Hoc_Tap\SE363 (1)\`) để cấu hình user ID cho Airflow, tránh lỗi permission.

Chạy lệnh sau trong terminal (PowerShell) tại thư mục dự án:
```powershell
echo "AIRFLOW_UID=50000" > .env
```
*(Nếu bạn chạy trên Linux/Mac, dùng `echo "AIRFLOW_UID=$(id -u)" > .env`)*

### Bước 2: Khởi Tạo Database cho Airflow
Lần đầu tiên chạy, bạn cần khởi tạo database và tài khoản admin cho Airflow.

```bash
docker-compose up airflow-init
```
*   **Đợi một chút**: Quá trình này sẽ pull images và setup DB.
*   **Thành công khi**: Bạn thấy thông báo `User "admin" created with role "Admin"` và container `airflow-init` dừng lại với code 0.

### Bước 3: Khởi Chạy Toàn Bộ Hệ Thống
Sau khi init xong, chạy lệnh sau để dựng toàn bộ các services:

```bash
docker-compose up -d
```
*   `-d`: Chạy ngầm (detached mode).

## 3. Kiểm Tra Trạng Thái

Sau khoảng 2-5 phút để các services khởi động hoàn tất, bạn có thể truy cập các dịch vụ tại:

| Dịch Vụ | URL / Port | Tài khoản (nếu có) |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | - |
| **Airflow UI** | [http://localhost:8080](http://localhost:8080) | `airflow` / `airflow` |
| **Spark Master UI** | [http://localhost:8081](http://localhost:8081) | - |
| **Kafka** | `localhost:9092` | - |

### Kiểm tra bằng lệnh:
```bash
docker-compose ps
```
Đảm bảo tất cả các containers đang ở trạng thái `Up` (hoặc `running`).

## 4. Quản Lý Hệ Thống

### Xem Logs
Để xem logs của một service cụ thể (ví dụ `kafka-consumer` hoặc `streamlit-app`):
```bash
docker-compose logs -f kafka-consumer
docker-compose logs -f streamlit-app
```

### Dừng Hệ Thống
Để dừng các containers nhưng giữ lại dữ liệu (trong volumes):
```bash
docker-compose stop
```

Để dừng và xóa toàn bộ containers & networks (nhưng giữ volumes):
```bash
docker-compose down
```

Để xóa sạch toàn bộ (bao gồm cả volumes database - **Cẩn thận mất dữ liệu**):
```bash
docker-compose down --volumes --remove-orphans
```

## 5. Troubleshooting (Sửa Lỗi Thường Gặp)

*   **Service cứ restart liên tục (Exited)**: Thường do thiếu RAM. Hãy tăng RAM cho Docker trong Settings -> Resources.
*   **Lỗi permission volume**: Đảm bảo bạn đã set `AIRFLOW_UID` trong file `.env`.
*   **Spark Worker không kết nối được Master**: Kiểm tra logs `docker-compose logs spark-worker`.
*   **Không vào được Streamlit**: Đợi thêm chút, model PhoBERT khá nặng nên service app khởi động lâu hơn bình thường.

---
**Lưu ý:**
Hệ thống sử dụng **PhoBERT** cho việc training và prediction, model này sẽ được download tự động trong lần chạy đầu tiên (hoặc load từ thư mục `models/`), vui lòng đảm bảo kết nối internet ổn định.
