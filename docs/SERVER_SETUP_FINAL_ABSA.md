# Hướng dẫn triển khai `final_absa` trên Ubuntu Server

Tài liệu này áp dụng cho active system trong nhánh `final_absa`. Nó thay thế
`legacy/system/HUONG_DAN_SETUP_TRAIN_TEST_SERVER.md`, vốn dành cho pipeline
Docker/Airflow/Kafka và experiment runner cũ.

Phạm vi của bộ triển khai mới:

- clone hoặc cập nhật đúng nhánh `final_absa`;
- tạo virtual environment Python riêng;
- cài dependency collector/model;
- kiểm tra PyTorch/CUDA và tài nguyên server;
- kiểm tra checksum, schema và leakage isolation của model-ready release;
- tải/cache `vinai/phobert-base`;
- chạy capacity pilot hoặc full training;
- validate sealed training artifact sau khi train.

Script không cài NVIDIA driver, không tải raw/private annotation archive,
không đưa checkpoint vào Git và không chứa password/token.

## 1. Thành phần

```text
docs/SERVER_SETUP_FINAL_ABSA.md
scripts/setup_final_absa_server.sh
scripts/deploy_final_absa_server.ps1
```

- `setup_final_absa_server.sh`: chạy trên Linux server.
- `deploy_final_absa_server.ps1`: chạy trên máy Windows, upload script Bash
  tạm qua SSH rồi gọi nó trên server.
- Dataset dùng trực tiếp đã nằm trong Git:
  `data/model_ready/absa_pseudo_v1_2_20260729/`.

## 2. Yêu cầu server

Khuyến nghị:

- Ubuntu 24.04 x86_64;
- Python 3.11 trở lên;
- 8 CPU core;
- RAM 16 GB tối thiểu, 32 GB khuyến nghị;
- ít nhất 20 GB dung lượng trống;
- NVIDIA GPU 12 GB VRAM trở lên được khuyến nghị cho cấu hình chính;
- NVIDIA driver hoạt động và `nvidia-smi` chạy được;
- internet ở lần setup đầu để cài package và tải PhoBERT.

GPU/VRAM tối thiểu chưa được tuyên bố trước capacity pilot. Nếu GPU ít VRAM,
chạy pilot với batch size 1 trước. Không dùng full CPU training trừ khi chấp
nhận thời gian rất dài và truyền cờ xác nhận.

PyTorch yêu cầu chọn wheel phù hợp driver/compute platform của server. Lấy
lệnh/index URL từ trang chính thức:

<https://pytorch.org/get-started/locally/>

Không sao chép một CUDA wheel URL cũ chỉ vì nó từng chạy trên máy khác.

Transformers tải model vào Hugging Face cache. Chế độ offline chỉ hoạt động
khi model/tokenizer đã được tải trước:

<https://huggingface.co/docs/transformers/installation>

## 3. Kiểm tra server trước khi setup

SSH vào server:

```bash
ssh <user>@<server-ip>
```

Kiểm tra:

```bash
cat /etc/os-release
uname -m
df -h
free -h
python3 --version
nvidia-smi
```

Nếu `nvidia-smi` lỗi, quản trị viên phải cài/sửa NVIDIA driver trước. Script
không tự sửa driver vì đây là thay đổi cấp hệ điều hành và có thể cần reboot.

Ubuntu 22.04 thường có Python mặc định thấp hơn 3.11. Hãy cài một interpreter
3.11+ theo chính sách server rồi truyền, ví dụ:

```bash
--python-bin python3.11
```

Không dùng `sudo pip install`.

## 4. Cách nhanh nhất từ Windows

Mở PowerShell tại repository local:

```powershell
cd C:\Users\Luc\Real-Time-ABSA-System
git switch final_absa
git pull --ff-only
```

Nếu Windows chặn script bởi Execution Policy, mở PowerShell cho session hiện
tại bằng:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4.1. Setup server

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action setup `
  -Device cuda
```

Nếu SSH dùng private key hoặc port riêng:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Port 2222 `
  -IdentityFile "C:\Keys\server_ed25519" `
  -Action setup `
  -Device cuda
```

Nếu cần chỉ định official PyTorch wheel index đã chọn cho server:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action setup `
  -Device cuda `
  -TorchIndexUrl "https://download.pytorch.org/whl/<OFFICIAL-CUDA-INDEX>"
```

Không ghi URL placeholder trên vào lệnh thật. Thay nó bằng URL được PyTorch
selector cung cấp tại thời điểm setup.

Script Windows chỉ upload bootstrap Bash, không upload dataset ZIP vì final
model-ready v1.2 đã nằm trong nhánh Git.

### 4.2. Dry run phía Windows

Dry run chỉ in lệnh `ssh`/`scp`, không kết nối:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action setup `
  -Device cuda `
  -DryRun
```

## 5. Chạy trực tiếp trên server

Nếu muốn tự thao tác:

```bash
git clone --branch final_absa --single-branch \
  https://github.com/Longhehehe/Real-Time-ABSA-System.git

cd Real-Time-ABSA-System

bash scripts/setup_final_absa_server.sh \
  --action setup \
  --device cuda
```

Với official PyTorch index:

```bash
bash scripts/setup_final_absa_server.sh \
  --action setup \
  --device cuda \
  --torch-index-url \
  "https://download.pytorch.org/whl/<OFFICIAL-CUDA-INDEX>"
```

Setup mặc định:

1. cài `git`, build tools, Python venv support và `tmux` bằng `apt`;
2. clone/update `final_absa` bằng fast-forward;
3. tạo `.venv-model`;
4. cài `.[ml]`;
5. in phiên bản NumPy/PyTorch/Transformers và trạng thái CUDA;
6. validate 28.266 model-ready records;
7. tải hoặc xác minh `vinai/phobert-base`.

Nếu system packages đã được quản trị viên chuẩn bị:

```bash
bash scripts/setup_final_absa_server.sh \
  --action setup \
  --device cuda \
  --skip-system-packages
```

Nếu muốn cài cả Selenium collector dependency:

```bash
bash scripts/setup_final_absa_server.sh \
  --action setup \
  --device cuda \
  --with-browser
```

Chrome/ChromeDriver không được script cài tự động. `--with-browser` chỉ cài
Python Selenium package.

## 6. Validate trước khi train

Từ Windows:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action validate `
  -Device cuda `
  -SkipInstall
```

Trực tiếp trên server:

```bash
bash scripts/setup_final_absa_server.sh \
  --action validate \
  --device cuda \
  --skip-system-packages \
  --skip-install \
  --no-update
```

Điều kiện pass:

- Python từ 3.11;
- import `numpy`, `torch`, `transformers` thành công;
- nếu yêu cầu CUDA: `torch.cuda.is_available() == True`;
- data validator trả `VALID`;
- split là 22.508/2.861/2.897;
- group là 7.383/876/877;
- PhoBERT tokenizer/model tải hoặc đọc được từ cache.

Không train nếu checksum/schema/group validation lỗi.

## 7. Capacity pilot bắt buộc

Pilot dùng:

- 1.000 train;
- 200 dev;
- 200 test;
- một epoch;
- max length và batch size mặc định từ `configs/training_v1.json`, trừ khi có
  runtime override.

Chạy detached để SSH mất kết nối không làm dừng train:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action pilot `
  -Device cuda `
  -Detach `
  -TmuxSession "absa-pilot"
```

GPU ít VRAM:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action pilot `
  -Device cuda `
  -BatchSize 1 `
  -GradientAccumulationSteps 16 `
  -Detach `
  -TmuxSession "absa-pilot-b1"
```

Trực tiếp trên server:

```bash
bash scripts/setup_final_absa_server.sh \
  --action pilot \
  --device cuda \
  --detach \
  --tmux-session absa-pilot
```

Theo dõi:

```bash
tmux ls
tmux attach -t absa-pilot
```

Tách khỏi tmux: `Ctrl+B`, sau đó `D`.

Log:

```text
artifacts/server_logs/<RUN_NAME>.log
```

Run artifact:

```text
artifacts/models/<RUN_NAME>/
```

Pilot pass khi:

- không CUDA OOM;
- train/dev/test path hoàn tất;
- có `model.pt`, `run.json`, `thresholds.json`, `manifest.json`,
  `SHA256SUMS`;
- `validate-run` trả `VALID`.

Metric pilot không được dùng làm kết quả paper.

## 8. Full training

Chỉ chạy sau khi pilot pass:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action full `
  -Device cuda `
  -RunName "absa_phobert_full_seed20260729_v1" `
  -Detach `
  -TmuxSession "absa-full-v1"
```

Hoặc trên server:

```bash
bash scripts/setup_final_absa_server.sh \
  --action full \
  --device cuda \
  --run-name absa_phobert_full_seed20260729_v1 \
  --detach \
  --tmux-session absa-full-v1
```

Không tái sử dụng một `run-name` đã tồn tại. Training fail-closed thay vì ghi
đè artifact.

Sau khi train foreground, script tự chạy:

```bash
python -m absa_system validate-run artifacts/models/<RUN_NAME>
```

Detached run cũng nối validation sau training bằng `&&`; nếu train lỗi,
sealed-run validation không chạy.

## 9. Chế độ offline

Chỉ dùng sau khi dependency và PhoBERT đã nằm trên server:

```bash
bash scripts/setup_final_absa_server.sh \
  --action validate \
  --device cuda \
  --offline \
  --skip-install \
  --no-update
```

Script đặt:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Nếu cache thiếu file, validation phải fail thay vì âm thầm truy cập mạng.

Có thể chỉ định cache riêng:

```bash
--hf-home /mnt/fast-cache/huggingface
```

## 10. Cập nhật code

Script mặc định cập nhật existing checkout bằng:

```bash
git fetch origin final_absa
git switch final_absa
git pull --ff-only origin final_absa
```

Nếu có tracked modification, script dừng và không reset. Kiểm tra thủ công:

```bash
cd ~/Real-Time-ABSA-System
git status
```

Không dùng `git reset --hard` để giải quyết tự động.

Muốn giữ nguyên commit hiện có:

```bash
--no-update
```

## 11. Xử lý lỗi

### `torch.cuda.is_available()` là `False`

Kiểm tra:

```bash
nvidia-smi
.venv-model/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Nếu `nvidia-smi` pass nhưng PyTorch không thấy CUDA, khả năng cao đã cài sai
wheel. Chọn lại official index URL và setup lại environment; không cài CUDA
wheel ngẫu nhiên.

### CUDA out-of-memory

Thử pilot với:

```text
--batch-size 1
--gradient-accumulation-steps 16
```

Chỉ giảm `--max-length` sau khi ghi thành một experiment config khác, vì thay
đổi max length làm thay đổi bài toán/truncation và không còn đúng frozen
baseline.

### SSH mất kết nối

Foreground process có thể bị dừng tùy shell/server. Vì vậy full training phải
chạy với `--detach`/tmux.

### Server reboot

Sau reboot:

```bash
cd ~/Real-Time-ABSA-System
git status
nvidia-smi
bash scripts/setup_final_absa_server.sh \
  --action validate \
  --device cuda \
  --skip-system-packages \
  --skip-install \
  --no-update
```

Training CLI hiện chưa hỗ trợ resume optimizer/mid-epoch. Best checkpoint của
run lỗi có thể tồn tại nhưng run chưa seal thì không được gọi là completed.
Tạo run name mới và chạy lại; không sửa artifact dở dang để giả hoàn tất.

## 12. Bảo mật và dữ liệu

- Không commit hoặc upload SSH private key.
- Không đưa cookie Lazada lên model-training server nếu không chạy collector.
- Không đưa `.env`, API key hoặc browser profile vào Git.
- Không sửa `data/model_ready/...` trên server; validator dựa vào checksum.
- Không đưa `artifacts/models/` vào Git.
- Raw/private annotation data không nằm trong `final_absa`; nếu sau này cần
  rebuild dataset, chuyển qua research storage có access control và checksum,
  không dùng Git thông thường.

## 13. Lệnh trợ giúp

Linux:

```bash
bash scripts/setup_final_absa_server.sh --help
```

Windows:

```powershell
.\scripts\deploy_final_absa_server.ps1 -Help
```
