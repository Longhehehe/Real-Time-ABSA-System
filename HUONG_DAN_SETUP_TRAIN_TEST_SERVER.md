# Hướng dẫn setup, train, test và vận hành ABSA trên Ubuntu Server

Tài liệu này đi từ thời điểm vừa SSH vào một Ubuntu Server trống đến khi cài NVIDIA driver, Python, pip, PyTorch, dependency của dự án; chuyển dữ liệu từ máy local lên server; giải nén ZIP/RAR; rồi train, test, evaluate và khởi động stack Docker.

Các lệnh được viết cho máy local Windows dùng PowerShell và Ubuntu Server 22.04/24.04 x86_64. Ngày rà soát: 2026-07-21.

> Trạng thái quan trọng: pipeline **train và evaluate mô hình** có thể chạy độc lập. Stack end-to-end Streamlit → Airflow → Kafka → Spark hiện còn các lỗi tích hợp được liệt kê ở cuối tài liệu. Không xem việc `docker compose up` thành công là bằng chứng toàn bộ pipeline đã hoạt động.

## 1. Cấu hình server khuyến nghị

Để train PhoBERT hoặc XLM-RoBERTa:

- Ubuntu Server 22.04 hoặc 24.04 64-bit.
- 8 CPU cores trở lên.
- RAM tối thiểu 16 GB, khuyến nghị 32 GB.
- NVIDIA GPU tối thiểu 12 GB VRAM; khuyến nghị 16–24 GB.
- Ổ đĩa trống tối thiểu 40 GB cho benchmark 72 run; khuyến nghị 60 GB.
- Internet để tải package, Docker image và model/tokenizer từ Hugging Face.

Nếu server chỉ có CPU, Logistic Regression và Naive Bayes vẫn chạy được. Train transformer bằng CPU sẽ rất chậm.

## 2. SSH vào server

Từ máy cá nhân:

```bash
ssh <username>@<server-ip>
```

Nếu dùng private key:

```bash
ssh -i /path/to/private_key <username>@<server-ip>
```

Kiểm tra hệ điều hành và tài nguyên:

```bash
cat /etc/os-release
uname -m
df -h
free -h
lscpu | head -30
```

Nên dùng `tmux` để tiến trình train không chết khi SSH mất kết nối:

```bash
sudo apt-get update
sudo apt-get install -y tmux
tmux new -s absa
```

- Tách khỏi session: nhấn `Ctrl+B`, sau đó nhấn `D`.
- Vào lại: `tmux attach -t absa`.
- Liệt kê session: `tmux ls`.

## 3. Cập nhật hệ thống và cài package nền

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

sudo apt-get install -y \
  git curl wget ca-certificates gnupg \
  build-essential pkg-config \
  python3 python3-dev python3-pip python3-venv \
  libgomp1 jq zip unzip p7zip-full rsync htop nvme-cli
```

Cài công cụ giải nén RAR. Trên một số bản Ubuntu, package `unrar` cần repository `multiverse`; nếu package không có thì dùng `7z` từ `p7zip-full` đã cài ở trên:

```bash
sudo apt-get install -y unrar || \
  echo "unrar không có trong repository hiện tại; sẽ dùng lệnh 7z"

command -v unzip
command -v 7z
command -v unrar || true
```

Kiểm tra Python. Dự án cần Python 3.9 trở lên:

```bash
python3 --version
python3 -m pip --version
```

Không chạy `sudo pip install ...`. Luôn dùng virtual environment.

## 4. Cài NVIDIA driver

### 4.1. Kiểm tra driver có sẵn

```bash
lspci | grep -i nvidia || true
nvidia-smi
```

Nếu `nvidia-smi` hiển thị GPU bình thường, chuyển đến mục 5.

### 4.2. Cài driver nếu cần

```bash
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

`reboot` sẽ ngắt SSH. Chờ server khởi động rồi vào lại:

```bash
ssh <username>@<server-ip>
nvidia-smi
```

Không cần cài full CUDA Toolkit để chạy PyTorch wheel. Host chủ yếu cần NVIDIA driver tương thích; wheel mang theo CUDA runtime.

## 5. Clone repository

Pipeline thực nghiệm bốn profile nằm trên nhánh `experiments`. Clone trực tiếp nhánh này trên server:

```bash
cd ~
git clone --branch experiments --single-branch \
  https://github.com/Longhehehe/Real-Time-ABSA-System.git
cd Real-Time-ABSA-System
```

Kiểm tra:

```bash
git status
git remote -v
git branch --show-current
git log -1 --oneline
```

`git branch --show-current` phải in ra `experiments`. Nếu repository đã được clone từ trước ở nhánh khác:

```bash
cd ~/Real-Time-ABSA-System
git status
git fetch origin
git switch experiments
git pull --ff-only origin experiments
```

Không switch hoặc pull nếu `git status` đang có thay đổi local mà chưa xác định cần giữ hay bỏ.

Cập nhật code ở lần sau:

```bash
cd ~/Real-Time-ABSA-System
git status
git pull --ff-only
```

Không `git pull` khi có thay đổi local chưa rõ cần giữ hay bỏ.

### 5.1. Cấu trúc dữ liệu cần chuyển lên server

Raw data thực nghiệm không nằm trong Git. Trên server, pipeline mong đợi đúng cấu trúc sau bên trong repository:

```text
Real-Time-ABSA-System/
  absa data/
    UIT-ViSD4SA-main/UIT-ViSD4SA-main/data/
      train.jsonl
      dev.jsonl
      test.jsonl
    absa-vlsp-2018-main/absa-vlsp-2018-main/datasets/
      vlsp2018_hotel/*.txt
      vlsp2018_restaurant/*.txt
    vlsp2016/
      SA-2016.train
      SA-2016.dev
      SA-2016.test
      SA-2016.dev_test
```

Không đổi tên file/thư mục và không đặt thành `absa data/absa data/...`. Thư mục `absa data/` đã bị Git ignore nên raw dataset không được push lên GitHub.

### 5.2. Đóng gói dữ liệu trên máy local Windows

Mở **PowerShell trên máy local**, vào đúng repository và kiểm tra dữ liệu:

```powershell
Set-Location "C:\Users\Luc\Real-Time-ABSA-System"
Get-ChildItem -LiteralPath ".\absa data" -Force
```

Cách khuyến nghị là đóng gói thành ZIP trước khi upload. Lệnh sau giữ nguyên thư mục gốc `absa data`:

```powershell
tar.exe -a -c -f ".\absa-data.zip" ".\absa data"
Get-Item -LiteralPath ".\absa-data.zip"
Get-FileHash -LiteralPath ".\absa-data.zip" -Algorithm SHA256
```

Ghi lại chuỗi SHA-256 để đối chiếu trên server. Nếu muốn dùng RAR và máy đã cài WinRAR:

```powershell
& "C:\Program Files\WinRAR\Rar.exe" a -r ".\absa-data.rar" ".\absa data\*"
Get-FileHash -LiteralPath ".\absa-data.rar" -Algorithm SHA256
```

ZIP được khuyến nghị vì Ubuntu hỗ trợ sẵn và ít phụ thuộc hơn RAR.

### 5.3. Upload từ máy local lên server

Vẫn chạy trên **PowerShell của máy local**. Thay `<server-ip>` bằng IP/domain thật; ví dụ user hiện tại là `islabworker1`:

```powershell
scp ".\absa-data.zip" "islabworker1@<server-ip>:/home/islabworker1/"
```

Nếu SSH dùng port khác 22, `scp` dùng cờ `-P` viết hoa:

```powershell
scp -P <ssh-port> ".\absa-data.zip" "islabworker1@<server-ip>:/home/islabworker1/"
```

Nếu đăng nhập bằng private key:

```powershell
scp -i "C:\path\to\private_key" ".\absa-data.zip" `
  "islabworker1@<server-ip>:/home/islabworker1/"
```

Nếu upload file lớn và cần tiếp tục khi mạng gián đoạn, dùng `rsync` trong **WSL/Git Bash** thay cho PowerShell thuần:

```bash
rsync -avP -e "ssh -p 22" \
  "/mnt/c/Users/Luc/Real-Time-ABSA-System/absa-data.zip" \
  "islabworker1@<server-ip>:/home/islabworker1/"
```

Có thể chép thẳng cả thư mục, nhưng thường chậm hơn vì có nhiều file nhỏ:

```powershell
scp -r ".\absa data" \
  "islabworker1@<server-ip>:/home/islabworker1/Real-Time-ABSA-System/"
```

Nếu đã chép thẳng thư mục bằng lệnh cuối thì bỏ qua bước giải nén, nhưng vẫn phải kiểm tra cấu trúc ở mục 5.5.

### 5.4. Kiểm tra checksum và giải nén trên server

SSH vào server, kiểm tra dung lượng và checksum file vừa nhận:

```bash
ssh islabworker1@<server-ip>
ls -lh ~/absa-data.zip
df -h ~
sha256sum ~/absa-data.zip
```

Chuỗi từ `sha256sum` phải giống SHA-256 đã in trên PowerShell. Nếu khác, file upload bị lỗi và cần gửi lại.

Nên xem nội dung archive và giải nén vào thư mục staging trước để tránh tạo nhầm `absa data/absa data` hoặc ghi đè dữ liệu đang có:

```bash
unzip -l ~/absa-data.zip | head -50
mkdir -p ~/absa-data-staging
unzip -n ~/absa-data.zip -d ~/absa-data-staging
find ~/absa-data-staging -maxdepth 5 -type f | head -50
```

Với file RAR:

```bash
unrar l ~/absa-data.rar | head -50
mkdir -p ~/absa-data-staging
unrar x -o- ~/absa-data.rar ~/absa-data-staging/
```

Nếu `unrar` không có hoặc không đọc được định dạng, dùng `7z`:

```bash
7z l ~/absa-data.rar
mkdir -p ~/absa-data-staging
7z x ~/absa-data.rar -o"$HOME/absa-data-staging" -aos
```

Các cờ `-n`, `-o-` và `-aos` đều yêu cầu bỏ qua file đã tồn tại, giúp tránh ghi đè âm thầm. Sau khi xác nhận staging có đúng một thư mục `absa data`, đồng bộ nó vào repository:

```bash
mkdir -p "$HOME/Real-Time-ABSA-System/absa data"
rsync -av --info=progress2 \
  "$HOME/absa-data-staging/absa data/" \
  "$HOME/Real-Time-ABSA-System/absa data/"
```

Nếu archive RAR được tạo bằng `"absa data\*"`, staging có thể chứa trực tiếp ba thư mục dataset thay vì thư mục `absa data`. Khi đó dùng:

```bash
rsync -av --info=progress2 \
  "$HOME/absa-data-staging/" \
  "$HOME/Real-Time-ABSA-System/absa data/"
```

Chỉ chạy **một** trong hai lệnh `rsync`, dựa trên kết quả `find` thực tế.

### 5.5. Xác nhận dữ liệu đã nằm đúng vị trí

```bash
cd ~/Real-Time-ABSA-System

test -f "absa data/UIT-ViSD4SA-main/UIT-ViSD4SA-main/data/train.jsonl" \
  && echo "OK: UIT train"
test -f "absa data/vlsp2016/SA-2016.train" \
  && echo "OK: VLSP 2016 train"

find "absa data" -type f | sort | head -80
du -sh "absa data"
git status --short --ignored "absa data"
```

Dòng cuối phải cho thấy raw data bị ignore, thường là `!! absa data/`. Không `git add -f` thư mục này. Có thể giữ file ZIP/RAR trong home để backup; nếu xóa thì chỉ xóa sau khi checksum, giải nén và chạy data audit thành công.

### 5.6. Chạy tự động bằng hai script

Các mục 5.1–11 vẫn là tài liệu tham chiếu để xử lý lỗi hoặc chạy thủ công. Trong trường hợp thông thường, có thể tự động hóa toàn bộ bằng hai file:

```text
scripts/upload_and_setup_server.ps1   # chạy trên máy local Windows
scripts/setup_experiment_server.sh    # chạy trên Ubuntu Server
```

Từ PowerShell máy local, cho phép chạy script trong riêng process hiện tại rồi thực hiện setup, upload data, audit, smoke test và khởi động full 72 run:

```powershell
Set-Location "C:\Users\Luc\Real-Time-ABSA-System"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\upload_and_setup_server.ps1 `
  -Server "islabworker1@<server-ip>" `
  -FullRun
```

Nếu SSH dùng private key hoặc port khác:

```powershell
.\scripts\upload_and_setup_server.ps1 `
  -Server "islabworker1@<server-ip>" `
  -Port <ssh-port> `
  -IdentityFile "C:\path\to\private_key" `
  -FullRun
```

PowerShell launcher sẽ tuần tự:

1. Kiểm tra kết nối SSH.
2. Upload Bash bootstrap mới nhất.
3. Đóng gói `absa data` thành ZIP tạm.
4. Upload ZIP và so sánh SHA-256 hai phía.
5. Gọi Bash bootstrap trên server với pseudo-terminal để nhập mật khẩu `sudo` nếu cần.
6. Xóa ZIP tạm ở local sau khi hoàn tất upload; ZIP trên server được giữ lại.

Bash bootstrap sẽ cài system package, clone/update nhánh `experiments`, cài Miniconda nếu thiếu, tạo env `absa`, cài PyTorch/requirements, giải nén data, audit, test, smoke và chạy full matrix bằng `--resume`. Khi có `-FullRun`, ma trận 72 run được khởi động trong detached `tmux` tên `absa-experiments`; mất kết nối SSH không làm chết job.

Nếu server cần cài NVIDIA driver, chạy lượt đầu:

```powershell
.\scripts\upload_and_setup_server.ps1 `
  -Server "islabworker1@<server-ip>" `
  -InstallDriver
```

Script sẽ dừng sau khi cài driver. Reboot server, SSH lại, rồi dùng file data đã upload sẵn để tránh upload lần hai:

```powershell
.\scripts\upload_and_setup_server.ps1 `
  -Server "islabworker1@<server-ip>" `
  -SkipUpload `
  -FullRun
```

Các chế độ hữu ích:

| Cờ PowerShell | Tác dụng |
|---|---|
| `-FullRun` | Sau setup/smoke, chạy 72 run trong `tmux` |
| `-SkipUpload` | Dùng lại `~/absa-data.zip` đã có trên server |
| `-SkipSystemPackages` | Bỏ qua `apt-get` khi server đã setup |
| `-SkipSmoke` | Bỏ smoke test; chỉ dùng khi smoke đã thành công trước đó |
| `-Cpu` | Cài/chạy bản CPU; transformer sẽ rất chậm |
| `-DryRun` | Chỉ in các lệnh local/SSH, không thay đổi server |
| `-KeepArchive` | Giữ lại file ZIP tạm trên máy local |

Nếu data đã nằm trên server và chỉ muốn chạy trực tiếp sau khi SSH:

```bash
cd ~/Real-Time-ABSA-System
bash scripts/setup_experiment_server.sh \
  --data-archive ~/absa-data.zip \
  --full-run
```

Xem toàn bộ tùy chọn:

```powershell
Get-Help .\scripts\upload_and_setup_server.ps1 -Detailed
```

```bash
bash scripts/setup_experiment_server.sh --help
```

## 6. Tạo môi trường Python riêng cho train/evaluate

Chỉ chọn **một** trong hai cách dưới đây. Không cài dependency hoặc chạy train trực tiếp trong Conda `(base)`. Sau khi đã activate môi trường riêng, luôn dùng `python` hoặc `python -m pip`; không gọi `/usr/bin/python3` hay `/usr/bin/pip3`.

### 6.1. Conda — khuyến nghị khi server đã có Anaconda/Miniconda

Các lệnh tạo môi trường `absa` dưới đây chỉ cần chạy **một lần**:

```bash
cd ~/Real-Time-ABSA-System

# Với Anaconda cài tại ~/anaconda3
source ~/anaconda3/etc/profile.d/conda.sh

conda create -n absa python=3.11 pip -y
conda activate absa

# Tránh package từ môi trường khác bị chèn vào đường dẫn import
unset PYTHONPATH
hash -r

python -m pip install --upgrade pip setuptools wheel
which python
python --version
python -m pip --version
```

Nếu dùng Miniconda, thay `~/anaconda3` bằng `~/miniconda3`. Có thể tìm thư mục Conda bằng:

```bash
command -v conda
conda info --base
```

Kết quả `which python` phải tương tự:

```text
/home/<username>/anaconda3/envs/absa/bin/python
```

Nó **không được** là `/usr/bin/python3` hoặc `/home/<username>/anaconda3/bin/python` của môi trường base.

Sau **mỗi lần SSH mới**, activate lại môi trường trước khi cài package, train, test hoặc evaluate:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH
hash -r
which python
```

Prompt thường đổi thành `(absa)`. Nếu muốn bật lệnh `conda activate` tự động cho Bash ở các phiên SSH sau:

```bash
~/anaconda3/bin/conda init bash
exec bash
conda activate absa
```

Khi làm việc xong:

```bash
conda deactivate
```

### 6.2. venv — dùng khi server không có Conda

Tạo một lần:

```bash
cd ~/Real-Time-ABSA-System
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
which python
python --version
python -m pip --version
```

Sau mỗi lần SSH mới:

```bash
cd ~/Real-Time-ABSA-System
source .venv/bin/activate
which python
```

Kết quả phải là `~/Real-Time-ABSA-System/.venv/bin/python`; prompt thường là `(.venv)`. Thoát bằng lệnh `deactivate`.

### 6.3. Chặn sớm việc dùng nhầm Python

Chạy kiểm tra này sau khi activate. Script sẽ dừng nếu interpreter không nằm trong Conda env `absa` hoặc `.venv`:

```bash
python - <<'PY'
import sys
from pathlib import Path

executable = str(Path(sys.executable).resolve())
print("Python executable:", executable)

if "/envs/absa/" not in executable and "/.venv/" not in executable:
    raise SystemExit(
        "ERROR: đang dùng sai Python. Hãy activate env absa hoặc .venv trước."
    )
PY
```

## 7. Cài PyTorch

Đảm bảo `(absa)` hoặc `(.venv)` đang active, rồi kiểm tra lại `which python` và `python -m pip --version`. Chỉ chọn **một** phương án GPU hoặc CPU; các lệnh `python -m pip` dưới đây sẽ cài vào đúng môi trường đang active.

### 7.1. NVIDIA GPU — khuyến nghị

```bash
python -m pip install \
  torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

Kiểm tra:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA runtime của wheel:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("VRAM GB:", round(memory, 2))
else:
    raise SystemExit("PyTorch chưa nhận GPU")
PY
```

Nếu CUDA 12.6 không phù hợp với GPU/driver cụ thể, lấy lệnh mới từ trang chính thức thay vì tự ghép phiên bản:

- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/

### 7.2. CPU only

```bash
python -m pip install \
  torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cpu

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 8. Cài requirements của code

File dùng cho train và evaluate:

```bash
python -m pip install -r requirements_training.txt
```

Các artifact `.pkl` hiện có trên máy phát triển được tạo bằng scikit-learn 1.6.1. Nếu copy artifact cũ lên server, đồng bộ phiên bản:

```bash
python -m pip install scikit-learn==1.6.1
```

Nếu train mới hoàn toàn, điều quan trọng nhất là train và evaluate trong cùng Conda env hoặc virtual environment.

Kiểm tra dependency:

```bash
python -m pip check

python - <<'PY'
import numpy
import pandas
import sklearn
import torch
import transformers

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
PY
```

Lưu môi trường chính xác:

```bash
mkdir -p server_artifacts
python -m pip freeze > server_artifacts/requirements-server.lock.txt
```

Vai trò của từng requirements file:

| File | Mục đích |
|---|---|
| `requirements_training.txt` | Train, test, evaluate native |
| `app_requirements.txt` | Streamlit app |
| `airflow_requirements.txt` | Airflow image |
| `spark_requirements.txt` | Spark/Kafka consumer image |
| `lazada_crawler/requirements.txt` | Crawler độc lập |

Không nên cài tất cả vào cùng `.venv` train. Stack đầy đủ nên chạy bằng Docker Compose.

## 9. Kiểm tra code và CLI

```bash
python -m compileall -q .
python train_all_methods.py --help
python scripts/evaluate_true_test.py --help
python scripts/check_distribution.py --help
python scripts/prepare_experiment_data.py --help
python scripts/run_experiments.py --help
python scripts/validate_experiment_artifacts.py --help
python scripts/publish_experiment_results.py --help
```

`train_all_methods.py` tự chọn `cuda` khi `torch.cuda.is_available()` là `True`. CLI hiện không có `--device`; không chạy `setup.sh` nguyên trạng vì script đó đang truyền `--device cuda`.

## 10. Chuẩn bị và audit bốn profile thực nghiệm

Phần này và mục 11 là workflow chính của nhánh `experiments`. Bốn profile gồm:

- `uit_visd4sa`: ACSA cấp review, 10 aspect gốc.
- `vlsp2018_hotel`: ACSA, 34 aspect gốc.
- `vlsp2018_restaurant`: ACSA, 12 aspect gốc.
- `vlsp2016`: sentiment toàn câu, ba lớp.

Không ánh xạ aspect của các dataset này sang 9 aspect Lazada. Pipeline chỉ chuẩn hóa thứ tự sentiment thành `NEG`, `POS`, `NEU`.

Sau mỗi lần SSH mới, luôn bắt đầu bằng block sau:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH
hash -r

git branch --show-current
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả phải là nhánh `experiments`, Python trong `envs/absa`, và `True` nếu server dùng GPU.

### 10.1. Tạo processed cache và audit

```bash
python scripts/prepare_experiment_data.py --profiles all
```

Script đọc raw data từ `absa data/`, ghi processed JSONL và audit vào `.experiment_cache/processed/`. Thư mục này bị Git ignore. Nó cũng tự động:

- chuẩn hóa Unicode/khoảng trắng mà không đổi ngữ nghĩa;
- giữ taxonomy aspect gốc của từng profile;
- gom các span UIT cùng aspect thành multi-hot polarity ở cấp review;
- báo nhãn lạ, text rỗng, span sai biên và duplicate nội bộ;
- loại khỏi train mọi exact-text duplicate xuất hiện trong dev/test;
- không xóa hoặc dùng dev/test để fit model.

Kiểm tra bốn báo cáo audit và dừng nếu train vẫn giao với dev/test:

```bash
python - <<'PY'
import json
from pathlib import Path

profiles = [
    "uit_visd4sa",
    "vlsp2018_hotel",
    "vlsp2018_restaurant",
    "vlsp2016",
]

for profile in profiles:
    path = Path(".experiment_cache/processed") / profile / "dataset_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    overlaps = audit["clean_overlaps"]
    print(
        profile,
        "train/dev=", overlaps["train_dev_unique_texts"],
        "train/test=", overlaps["train_test_unique_texts"],
        "clean_train=", audit["clean_train_rows"],
    )
    assert overlaps["train_dev_unique_texts"] == 0
    assert overlaps["train_test_unique_texts"] == 0

print("OK: không còn exact-text leakage từ train sang dev/test")
PY
```

Kích thước sạch dự kiến với bộ raw data hiện tại:

| Profile | Train | Dev | Test | Native aspects |
|---|---:|---:|---:|---:|
| `uit_visd4sa` | 7.782 | 1.112 | 2.225 | 10 |
| `vlsp2018_hotel` | 2.999 | 2.000 | 600 | 34 |
| `vlsp2018_restaurant` | 2.923 | 1.290 | 500 | 12 |
| `vlsp2016` | 4.983 | 100 | 1.050 | không áp dụng |

`SA-2016.dev_test` chỉ được audit, không tham gia train, model selection hoặc final test.

### 10.2. Kiểm tra manifest và dung lượng

```bash
find .experiment_cache/processed -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
jq '
  .profiles
  | to_entries[]
  | .key as $profile
  | .value.source_manifest
  | to_entries[]
  | {
      profile: $profile,
      split: .key,
      path: .value.path,
      bytes: .value.bytes,
      sha256: .value.sha256
    }
' \
  .experiment_cache/processed/source_manifest.json
df -h .
```

Giữ ít nhất 40 GB trống trước full run để chứa checkpoint tạm, Hugging Face cache, log và artifact của 72 lượt chạy.

## 11. Train, test và evaluate bốn profile trên server

`scripts/run_experiments.py` thực hiện toàn bộ protocol chính trong một lệnh:

1. Fit model và TF-IDF chỉ trên clean train.
2. Early stopping và threshold tuning chỉ trên dev.
3. Load best epoch theo dev rồi evaluate official test đúng một lần cho mỗi seed.
4. Tổng hợp test mean ± std qua seed `42`, `52`, `62`.
5. Chọn best seed theo dev score, không theo test score.

Vì vậy không có một lệnh test thủ công riêng cho protocol bốn profile. Bước test/evaluate đã được thực hiện bên trong runner sau khi train của từng seed hoàn tất.

### 11.1. Thiết lập cache và thư mục log

```bash
cd ~/Real-Time-ABSA-System
mkdir -p .cache/huggingface logs

export HF_HOME="$PWD/.cache/huggingface"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

### 11.2. Dry-run: xác nhận đúng 72 lượt

```bash
python scripts/run_experiments.py --dry-run
```

Ma trận đúng phải là `4 profile × 6 model × 3 seed = 72 run`, với sáu model:

- `logistic_regression`
- `naive_bayes`
- `bilstm`
- `cnn_bilstm`
- `phobert`
- `xlm_roberta`

### 11.3. Smoke test trước khi chạy thật

Smoke Logistic Regression trên cả bốn profile:

```bash
python scripts/run_experiments.py \
  --profiles all \
  --models logistic_regression \
  --seeds 42 \
  --smoke \
  --fail-fast \
  --artifact-root .experiment_cache/smoke_artifacts
```

Smoke PhoBERT một epoch trên tập nhỏ của cả bốn profile để kiểm tra GPU/tokenizer/model download:

```bash
python scripts/run_experiments.py \
  --profiles all \
  --models phobert \
  --seeds 42 \
  --smoke \
  --fail-fast \
  --artifact-root .experiment_cache/smoke_artifacts
```

Smoke artifact không phải kết quả nghiên cứu chính và không được publish. Nếu hai lệnh này lỗi, xử lý lỗi trước khi chạy 72 lượt.

### 11.4. Chạy full train + test + evaluate

Tạo một session riêng để job không chết khi mất SSH:

```bash
tmux new -s absa-experiments
```

Bên trong `tmux`, activate lại Conda và chạy full matrix:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH

export HF_HOME="$PWD/.cache/huggingface"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
set -o pipefail

python scripts/run_experiments.py \
  --profiles uit_visd4sa vlsp2018_hotel vlsp2018_restaurant vlsp2016 \
  --models logistic_regression naive_bayes bilstm cnn_bilstm phobert xlm_roberta \
  --seeds 42 52 62 \
  --resume \
  2>&1 | tee logs/experiments_full.log

run_status=${PIPESTATUS[0]}
echo "experiment exit code: $run_status"
test "$run_status" -eq 0
```

Runner chạy tuần tự trên một GPU. Với transformer, fallback OOM là tự động:

1. physical batch 16;
2. batch 8 + gradient accumulation 2;
3. batch 4 + gradient accumulation 4.

Effective batch vẫn là 16 và cấu hình thực tế được ghi trong artifact. Không tự giảm `max_length` hoặc đổi hyperparameter giữa các profile nếu mục tiêu là so sánh protocol đã chốt.

Tách khỏi `tmux`: `Ctrl+B`, rồi `D`. Vào lại:

```bash
tmux attach -t absa-experiments
```

Nếu server reboot hoặc process dừng, tạo/attach `tmux`, activate lại env rồi chạy lại đúng lệnh full có `--resume`. Các run hoàn tất và tương thích sẽ được bỏ qua.

### 11.5. Theo dõi tiến trình

Mở SSH session thứ hai:

```bash
cd ~/Real-Time-ABSA-System
tail -f logs/experiments_full.log
```

Ở terminal khác có thể xem GPU, ổ đĩa và số seed đã hoàn tất:

```bash
watch -n 2 nvidia-smi
watch -n 30 'du -sh artifacts/experiments .cache/huggingface; df -h .'

find artifacts/experiments \
  -path '*/runs/seed_*/status.json' \
  -exec grep -l '"completed"' {} \; | wc -l
```

Khi hoàn tất, lệnh đếm phải ra `72`. Kiểm tra chi tiết bằng Python:

```bash
python - <<'PY'
import json
from pathlib import Path

files = sorted(Path("artifacts/experiments").glob("*/*/runs/seed_*/results.json"))
completed = []
failed = []
for path in files:
    data = json.loads(path.read_text(encoding="utf-8"))
    (completed if data.get("status") == "completed" else failed).append(str(path))

print("completed:", len(completed))
print("failed:", len(failed))
for path in failed:
    print("-", path)

if len(completed) != 72 or failed:
    raise SystemExit("Experiment matrix chưa hoàn tất")
PY
```

### 11.6. Validate và đọc kết quả evaluate

Chỉ validate strict sau khi đủ 72 run:

```bash
python scripts/validate_experiment_artifacts.py
```

Kết quả chính của mỗi cặp profile/model nằm ở:

```text
artifacts/experiments/<profile>/<model>/results.json
```

Ví dụ xem XLM-RoBERTa trên UIT. Primary metric ACSA là end-to-end `Aspect#Polarity` micro-F1:

```bash
jq '{
  profile_id,
  model,
  primary_metric,
  best_seed,
  best_dev_score,
  test_mean: .avg_metrics.end_to_end_f1_micro,
  test_std: .avg_metrics.end_to_end_f1_micro_std
}' artifacts/experiments/uit_visd4sa/xlm_roberta/results.json
```

Ví dụ xem XLM-RoBERTa trên VLSP 2016. Primary metric là sentiment macro-F1:

```bash
jq '{
  profile_id,
  model,
  primary_metric,
  best_seed,
  best_dev_score,
  test_mean: .avg_metrics.f1_macro,
  test_std: .avg_metrics.f1_macro_std
}' artifacts/experiments/vlsp2016/xlm_roberta/results.json
```

Metric test của từng seed nằm trong `seed_results[].test_metrics`; test không được dùng để chọn best seed:

```bash
jq '.seed_results[] | {
  seed,
  dev_primary: .dev_metrics,
  test_metrics: .test_metrics
}' artifacts/experiments/uit_visd4sa/xlm_roberta/results.json
```

### 11.7. Artifact được tạo trên server

```text
artifacts/experiments/<profile>/<model>/
  config.json
  results.json
  metrics_comparison.png
  seed_scores.png
  training_loss_by_epoch.png   # neural model
  confusion_matrix.png         # VLSP 2016
  thresholds.json              # ACSA
  best_model.pt|pkl
  runs/
    seed_42/
    seed_52/
    seed_62/
```

Mỗi profile còn có `dataset_audit.json`, `all_models_comparison.json` và `all_models_metrics_comparison.png`. Checkpoint, prediction chi tiết, cache và log là artifact nặng, đã bị Git ignore.

### 11.8. Publish báo cáo nhẹ

Sau khi validator thành công:

```bash
python scripts/publish_experiment_results.py

find reports/experiments -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
head -5 reports/experiments/summary_acsa.csv
head -5 reports/experiments/summary_sentiment.csv
```

Output publish gồm:

```text
reports/experiments/
  uit_visd4sa/
  vlsp2018_hotel/
  vlsp2018_restaurant/
  vlsp2016/
  summary_acsa.csv
  summary_sentiment.csv
  run_manifest.json
```

Hai bảng ACSA và global sentiment được tách riêng vì chúng là hai task khác nhau, không so trực tiếp cùng một metric.

### 11.9. Tải báo cáo từ server về máy local

Đóng gói **báo cáo nhẹ** trên server:

```bash
tar -czf ~/absa-experiment-reports.tar.gz \
  -C "$HOME/Real-Time-ABSA-System" reports/experiments
sha256sum ~/absa-experiment-reports.tar.gz
```

Sau đó chạy trên PowerShell máy local:

```powershell
Set-Location "C:\Users\Luc\Real-Time-ABSA-System"
scp "islabworker1@<server-ip>:/home/islabworker1/absa-experiment-reports.tar.gz" ".\"
Get-FileHash ".\absa-experiment-reports.tar.gz" -Algorithm SHA256
tar.exe -xzf ".\absa-experiment-reports.tar.gz"
```

So sánh SHA-256 hai phía. Không nên tải toàn bộ `artifacts/experiments/` nếu chỉ cần bảng và hình kết quả, vì checkpoint có thể rất lớn. Nếu thực sự cần backup checkpoint, dùng `rsync -avP` từ WSL để có resume.

## 12. Chuẩn bị thư mục và dữ liệu Lazada hiện tại (workflow cũ)

Từ mục 12 đến mục 20 là workflow cũ dùng `Augmented Dataset`, K-Fold và taxonomy 9 aspect Lazada. Không dùng các lệnh này để benchmark bốn profile ở mục 10–11.

```bash
mkdir -p \
  data/predictions \
  data/triggers \
  data/label \
  airflow/logs \
  airflow/plugins \
  logs \
  server_artifacts
```

Dataset train:

```text
Augmented Dataset/*.xlsx
```

Tập evaluate:

```text
True_Test_Data/dev_augmented_500.xlsx
```

Kiểm tra:

```bash
find "Augmented Dataset" -maxdepth 1 -type f -name '*.xlsx' -printf '%f\n' | sort
find True_Test_Data -maxdepth 1 -type f -printf '%f\n' | sort
du -sh "Augmented Dataset" True_Test_Data
```

Kiểm tra phân phối nhãn:

```bash
python scripts/check_distribution.py \
  --file "Augmented Dataset/augmented_result.xlsx"

python scripts/check_distribution.py \
  --file "True_Test_Data/dev_augmented_500.xlsx"
```

Kiểm tra mọi file train:

```bash
for file in "Augmented Dataset"/*.xlsx; do
  echo "===== $file ====="
  python scripts/check_distribution.py --file "$file"
done
```

### 12.1. Data augmentation — tùy chọn

Xem toàn bộ tham số:

```bash
python "Augmented Dataset/augmented.py" --help
```

Strategy 1 dùng thay thế từ đồng nghĩa. Strategy 2 dùng back-translation và cần package bổ sung:

```bash
python -m pip install googletrans==4.0.0-rc1
```

Ví dụ augment offline một file riêng, không ghi đè dữ liệu gốc:

```bash
python "Augmented Dataset/augmented.py" \
  --input "Old Dataset/test_flow_reviews_part1_labeled.xlsx" \
  --output "server_artifacts/augmented_offline.xlsx" \
  --target 200 \
  --strategies 1,2 \
  --offline-workers 4 \
  --seed 42
```

Strategy 3/4 gọi NVIDIA-hosted Mistral API và cần API key. Không commit key vào Git hoặc ghi key thật vào tài liệu:

```bash
read -s -p "NVIDIA API key: " NVIDIA_API_KEY
echo

python "Augmented Dataset/augmented.py" \
  --input "Old Dataset/test_flow_reviews_part1_labeled.xlsx" \
  --output "server_artifacts/augmented_llm.xlsx" \
  --target 200 \
  --strategies 3,4 \
  --api-key "$NVIDIA_API_KEY" \
  --llm-workers 4 \
  --llm-rps 3 \
  --seed 42

unset NVIDIA_API_KEY
```

Script nhận API key qua command-line nên key có thể xuất hiện tạm thời trong process list. Chỉ chạy trên server được kiểm soát. Luôn kiểm tra chất lượng/nhãn của output trước khi đưa vào train; tuyệt đối không augment dev/test.

## 13. Tải trước model/tokenizer từ Hugging Face

Không bắt buộc, nhưng giúp phát hiện sớm lỗi Internet/dung lượng:

```bash
mkdir -p .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"

python - <<'PY'
from transformers import AutoModel, AutoTokenizer

for name in ["vinai/phobert-base", "xlm-roberta-base"]:
    print("Downloading:", name)
    AutoTokenizer.from_pretrained(name)
    AutoModel.from_pretrained(name)
    print("OK:", name)
PY
```

Biến môi trường nên đặt trước khi train:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
```

## 14. Smoke test workflow Lazada trước khi train lớn

Train Logistic Regression với 2 folds:

```bash
set -o pipefail
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model logistic_regression \
  --folds 2 \
  2>&1 | tee logs/train_lr_smoke.log
```

Kiểm tra artifact:

```bash
find models/logistic_regression_absa -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
jq '.model, .avg_metrics.combined_score' models/logistic_regression_absa/results.json
```

Evaluate smoke model:

```bash
python scripts/evaluate_true_test.py \
  --model lr \
  --data dev_augmented_500.xlsx \
  2>&1 | tee logs/evaluate_lr_smoke.log
```

## 15. Train model Lazada chính

### 15.1. XLM-RoBERTa — model tốt nhất hiện tại

5 folds nghĩa là model được train 5 lần:

```bash
set -o pipefail
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model xlm_roberta \
  --folds 5 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_length 256 \
  --label_smoothing 0.1 \
  --gamma 2.0 \
  --sentiment_weight 5.0 \
  --patience 3 \
  --threshold_min 0.1 \
  --threshold_max 0.9 \
  --threshold_steps 17 \
  2>&1 | tee logs/train_xlm_roberta.log
```

Nếu hết VRAM, giảm `--batch_size` xuống `8`, `4` hoặc `2`.

### 15.2. PhoBERT

```bash
set -o pipefail
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model phobert \
  --folds 5 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_length 256 \
  --label_smoothing 0.1 \
  --gamma 2.0 \
  --sentiment_weight 5.0 \
  --patience 3 \
  2>&1 | tee logs/train_phobert.log
```

### 15.3. BiLSTM và CNN-BiLSTM

```bash
set -o pipefail
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model bilstm cnn_bilstm \
  --folds 5 \
  --epochs 5 \
  --batch_size 16 \
  --max_length 256 \
  --gamma 2.0 \
  --sentiment_weight 5.0 \
  --patience 3 \
  2>&1 | tee logs/train_deep_models.log
```

### 15.4. Logistic Regression và Naive Bayes

```bash
set -o pipefail
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model logistic_regression naive_bayes \
  --folds 5 \
  2>&1 | tee logs/train_ml_models.log
```

### 15.5. Train cả 6 model

Bỏ `--model` để train toàn bộ:

```bash
set -o pipefail
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --folds 5 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_length 256 \
  --label_smoothing 0.1 \
  --gamma 2.0 \
  --sentiment_weight 5.0 \
  --patience 3 \
  2>&1 | tee logs/train_all_models.log
```

Lệnh này rất tốn thời gian. Luôn smoke test và train từng transformer riêng trước.

## 16. Theo dõi train workflow Lazada

Mở SSH session thứ hai:

```bash
cd ~/Real-Time-ABSA-System
watch -n 2 nvidia-smi
```

Log và tài nguyên:

```bash
tail -f logs/train_xlm_roberta.log
htop
watch -n 10 'free -h; df -h .'
```

Sau lệnh có `tee`:

```bash
echo $?
```

Với `set -o pipefail`, kết quả khác `0` nghĩa là train lỗi.

## 17. Artifact Lazada sau train

Đường dẫn mặc định:

```text
models/logistic_regression_absa/logistic_regression_model.pkl
models/naive_bayes_absa/naive_bayes_model.pkl
models/bilstm_absa/bilstmforabsa_absa.pt
models/cnn_bilstm_absa/cnnbilstmforabsa_absa.pt
models/phobert_absa/phobertforabsamultipolarity_absa.pt
models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt
```

Kiểm tra:

```bash
find models -maxdepth 2 -type f \
  \( -name '*.pt' -o -name '*.pkl' -o -name 'results.json' \) \
  -printf '%p %s bytes\n' | sort
```

`.pt` và `.pkl` bị `.gitignore` loại trừ. Server khác cần train lại hoặc chép artifact.

Tải checkpoint về máy cá nhân:

```bash
scp <username>@<server-ip>:~/Real-Time-ABSA-System/models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt .
```

## 18. Evaluate workflow Lazada

### XLM-RoBERTa

```bash
set -o pipefail
python scripts/evaluate_true_test.py \
  --model xlm \
  --data dev_augmented_500.xlsx \
  2>&1 | tee logs/evaluate_xlm.log
```

### PhoBERT

```bash
python scripts/evaluate_true_test.py \
  --model phobert \
  --data dev_augmented_500.xlsx \
  2>&1 | tee logs/evaluate_phobert.log
```

### Tất cả model

```bash
set -o pipefail
for model in lr nb bilstm cnn_bilstm phobert xlm; do
  echo "===== EVALUATE: $model ====="
  python scripts/evaluate_true_test.py \
    --model "$model" \
    --data dev_augmented_500.xlsx \
    2>&1 | tee "logs/evaluate_${model}.log"
done
```

Xem kết quả:

```bash
find test_results -maxdepth 1 -type f -name '*_test_results.json' -printf '%f\n' | sort
jq '.model, .num_samples, .metrics.combined_score' test_results/xlm_test_results.json
```

In bảng so sánh:

```bash
python - <<'PY'
import glob
import json

rows = []
for path in glob.glob("test_results/*_test_results.json"):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    metrics = data["metrics"]
    rows.append((
        data["model"],
        metrics["combined_score"],
        metrics["mention_f1_macro"],
        metrics["sentiment_f1_macro"],
    ))

print(f"{'Model':32} {'Combined':>10} {'MentionF1':>10} {'SentF1':>10}")
for model, combined, mention, sentiment in sorted(rows, key=lambda row: row[1], reverse=True):
    print(f"{model:32} {combined:10.4f} {mention:10.4f} {sentiment:10.4f}")
PY
```

## 19. Test inference một câu

`predict_example.py` đang dùng interface predictor cũ. Dùng `GeneralABSAPredictor` trực tiếp.

### XLM-RoBERTa

```bash
python - <<'PY'
from app.absa_predictor import GeneralABSAPredictor

model_path = "models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt"
predictor = GeneralABSAPredictor(model_path)

text = "Sản phẩm đẹp, chạy mượt nhưng giao hàng hơi chậm"
result = predictor.predict_single(text)

print("Text:", text)
for aspect, info in result["multipolarity"].items():
    if info["mentioned"]:
        print(f"- {aspect}: {info['sentiments']}")
PY
```

### Logistic Regression

```bash
python - <<'PY'
from app.absa_predictor import GeneralABSAPredictor

model_path = "models/logistic_regression_absa/logistic_regression_model.pkl"
predictor = GeneralABSAPredictor(model_path, device="cpu")

result = predictor.predict_single("Đóng gói đẹp nhưng giao hàng quá chậm")
for aspect, info in result["multipolarity"].items():
    if info["mentioned"]:
        print(aspect, info["sentiments"])
PY
```

## 20. Cảnh báo về kết quả đánh giá workflow Lazada hiện tại

Các lệnh trên tái tạo workflow hiện tại, nhưng benchmark chưa sạch:

- 500/1.721 dòng test trùng chính xác với text trong train.
- Dataset train có text trùng và KFold chia theo dòng, không group theo review nguồn.
- TF-IDF được fit trên toàn bộ text trước khi chia KFold.
- Threshold được tối ưu và chấm trên cùng validation fold.
- Dữ liệu augment chưa có `source_review_id`.

Quy trình nghiên cứu đúng nên là:

1. Deduplicate dữ liệu gốc.
2. Chia train/dev/test từ dữ liệu gốc trước.
3. Gắn `source_review_id`.
4. Chỉ augmentation trên train.
5. Group mọi biến thể cùng nguồn vào một split.
6. Fit TF-IDF chỉ trên train fold.
7. Tune threshold trên dev/inner validation.
8. Evaluate test một lần sau khi chốt model.

## 21. Cài Docker Engine — tùy chọn

Train/evaluate không cần Docker. Phần này dùng cho full stack.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
sudo docker run --rm hello-world
docker compose version
```

Cho phép user hiện tại chạy Docker:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker version
```

Tài liệu chính thức: https://docs.docker.com/engine/install/ubuntu/

## 22. Cài NVIDIA Container Toolkit

Chỉ cần khi Docker dùng GPU:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi
```

Tài liệu chính thức: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## 23. Chạy Docker Compose

```bash
cd ~/Real-Time-ABSA-System

mkdir -p \
  data/predictions data/triggers data/label \
  airflow/logs airflow/plugins cookie

cat > .env <<EOF
AIRFLOW_UID=$(id -u)
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=CHANGE_THIS_PASSWORD
EOF
```

Thay `CHANGE_THIS_PASSWORD`.

```bash
docker compose config --quiet
docker compose build
docker compose up airflow-init
docker compose up -d
docker compose ps
```

Log:

```bash
docker compose logs -f --tail=200 \
  airflow-webserver airflow-scheduler kafka-consumer
```

URL:

| Dịch vụ | URL |
|---|---|
| Streamlit | `http://<server-ip>:8501` |
| FastAPI | `http://<server-ip>:8000/docs` |
| Airflow | `http://<server-ip>:8080` |
| Spark UI | `http://<server-ip>:8081` |

Nên dùng SSH tunnel:

```bash
ssh -L 8501:localhost:8501 \
    -L 8000:localhost:8000 \
    -L 8080:localhost:8080 \
    -L 8081:localhost:8081 \
    <username>@<server-ip>
```

Dừng và giữ database volume:

```bash
docker compose down
```

Không dùng `docker compose down -v` trừ khi muốn xóa database Airflow/Postgres.

## 24. Lỗi tích hợp hiện tại của full stack

Cần sửa trước khi production:

1. API gọi `PhoBERTPredictor()` thiếu model path, rồi gọi `load_model()` và `model_loaded` không tồn tại.
2. API import `aggregate_multipolarity_scores` không tồn tại.
3. Trang sản phẩm import `get_product_info` và `create_session` không tồn tại.
4. Trang so sánh import `aggregate_scores` và `SENTIMENT_MAP` không tồn tại.
5. Consumer đọc `/app/model_config.json`, file thật ở `/app/app/model_config.json`.
6. Consumer gọi `PhoBERTPredictor()` thiếu model path.
7. Consumer truyền `format` cho `OllamaPredictor.predict_batch`, nhưng hàm không nhận tham số này.
8. Spark session local được tạo trước Spark master và dữ liệu bị `repartition(1)`.
9. Dashboard đòi file simulation trước khi có thể chọn Live Predictions.
10. DAG training cũ tham chiếu các module không tồn tại.

Ở trạng thái hiện tại, ưu tiên tài liệu này cho **native training/evaluation**. Docker dùng kiểm tra infrastructure cho đến khi code tích hợp được sửa.

## 25. Security checklist

- Xóa cookie Lazada thật khỏi Git và rotate phiên đăng nhập.
- Không dùng `admin/admin` cho Airflow.
- Thay webserver secret hard-code bằng secret từ environment.
- Không để FastAPI CORS `*` trong production.
- Thêm authentication cho endpoint xóa prediction và trigger pipeline.
- Không public Kafka, Zookeeper, Spark hoặc Airflow ra Internet.
- Không commit `.env`, private key, cookie hoặc token.

UFW cơ bản, thay IP quản trị thật:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from <your-public-ip> to any port 22 proto tcp
sudo ufw enable
sudo ufw status verbose
```

Kiểm tra rule SSH trước khi bật UFW để tránh tự khóa server.

## 26. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Lỗi này thường xảy ra khi chạy bằng `/usr/bin/python3` trong khi PyTorch được cài ở Conda env hoặc `.venv`. Kiểm tra:

```bash
which python
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip show torch
```

Với Conda, activate lại đúng env rồi chạy bằng `python`, không dùng `/usr/bin/python3`:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH
hash -r

which python
python -m pip show torch
python train_all_methods.py --help
```

Nếu `python -m pip show torch` vẫn báo không tìm thấy package, cài PyTorch bằng đúng lệnh ở mục 7 trong khi `(absa)` đang active.

### `ImportError` khi import `torch._C` từ Conda `(base)`

Nếu traceback chứa đường dẫn dạng sau thì tiến trình đang nạp PyTorch từ Conda base, không phải env `absa`:

```text
/home/<username>/anaconda3/lib/python3.x/site-packages/torch/...
```

Không cần sửa hoặc tiếp tục cài package vào `(base)`. Nếu env `absa` đã được tạo, chuyển sang đúng env:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH
hash -r

which python
python -m pip --version
python -m pip show torch

python - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("Torch file:", torch.__file__)
print("Torch version:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

`Python` và `Torch file` đều phải chứa `/envs/absa/`. Nếu env chưa tồn tại, quay lại mục 6.1 để tạo env, sau đó cài PyTorch và requirements trong env đó.

### PyTorch không nhận GPU

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Nếu vừa nâng driver, reboot.

### CUDA out of memory

- Với pipeline bốn profile, runner tự thử batch `16`, `8 + accumulation 2`, rồi `4 + accumulation 4`. Xem `config.json` để biết cấu hình thực tế; chỉ can thiệp nếu cả ba mức đều thất bại.
- Với workflow Lazada cũ, giảm `--batch_size`: 16 → 8 → 4 → 2.
- Chỉ giảm `--max_length` khi chấp nhận thay đổi protocol thực nghiệm: 256 → 192 → 128.
- Kiểm tra process khác bằng `nvidia-smi`.

### Checksum ZIP/RAR trên local và server không giống nhau

Không giải nén hoặc train từ file này. Upload lại bằng `scp`, hoặc dùng `rsync -avP` để tiếp tục file lớn. Sau đó chạy lại:

```bash
sha256sum ~/absa-data.zip
```

và so sánh với:

```powershell
Get-FileHash ".\absa-data.zip" -Algorithm SHA256
```

### Data bị lồng thành `absa data/absa data`

Không sửa `configs/datasets.json` để chạy theo đường dẫn lồng sai. Kiểm tra staging và đồng bộ đúng lớp thư mục theo mục 5.4:

```bash
find ~/absa-data-staging -maxdepth 4 -type d | sort
find "$HOME/Real-Time-ABSA-System/absa data" -maxdepth 4 -type d | sort
```

Đường dẫn cuối phải khớp chính xác cấu trúc ở mục 5.1.

### Hết RAM

```bash
free -h
dmesg -T | grep -i -E 'killed process|out of memory|oom' | tail -30
```

### Hết dung lượng

```bash
df -h
du -sh .cache models logs 2>/dev/null
docker system df
```

Không xóa model/Docker data khi chưa sao lưu.

### Hugging Face không tải được

```bash
curl -I https://huggingface.co
export HF_HOME="$PWD/.cache/huggingface"
```

Kiểm tra proxy, DNS, outbound firewall và dung lượng.

### Pickle sai phiên bản scikit-learn

```bash
python -c "import sklearn; print(sklearn.__version__)"
python -m pip install scikit-learn==1.6.1
```

Không load pickle từ nguồn không tin cậy.

### Xác nhận train thành công

```bash
grep -i -E 'traceback|error|exception|results saved' logs/train_xlm_roberta.log | tail -50
test -f models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt && echo OK || echo MISSING
jq '.avg_metrics.combined_score' models/xlm_roberta_absa/results.json
```

## 27. Checklist hoàn tất

### Setup và truyền dữ liệu

- [ ] `nvidia-smi` nhận GPU.
- [ ] Conda env `(absa)` hoặc `(.venv)` đã active.
- [ ] `sys.executable` trỏ vào `/envs/absa/` hoặc `/.venv/`, không phải `/usr/bin/python3`/Conda base.
- [ ] `torch.cuda.is_available()` là `True` trên GPU server.
- [ ] `pip check` không báo lỗi.
- [ ] Nhánh hiện tại là `experiments`.
- [ ] SHA-256 của ZIP/RAR trên local và server giống nhau.
- [ ] Raw data có đúng cấu trúc mục 5.1 và `git status --ignored` cho thấy `absa data/` bị ignore.
- [ ] Ổ đĩa còn ít nhất 40 GB trước full run.
- [ ] Đã lưu `requirements-server.lock.txt`.

### Benchmark bốn profile

- [ ] `prepare_experiment_data.py --profiles all` chạy thành công.
- [ ] Audit xác nhận train/dev và train/test overlap đều bằng 0.
- [ ] Dry-run liệt kê đúng 72 run.
- [ ] Smoke Logistic Regression và PhoBERT thành công trên cả bốn profile.
- [ ] Full runner có `--resume` hoàn tất 72/72 run, không có run failed.
- [ ] `validate_experiment_artifacts.py` thành công.
- [ ] `publish_experiment_results.py` tạo hai summary CSV riêng.
- [ ] SHA-256 của gói report tải về local giống phía server.
- [ ] Raw data, `.pt`, `.pkl`, log và prediction chi tiết không bị stage vào Git.

### Workflow Lazada cũ và vận hành

- [ ] Nếu chạy workflow cũ: dataset train/test tồn tại.
- [ ] Nếu chạy workflow cũ: smoke Logistic Regression thành công.
- [ ] Nếu chạy workflow cũ: XLM-R/PhoBERT tạo checkpoint `.pt` và evaluate tạo JSON trong `test_results/`.
- [ ] Đã sao lưu checkpoint cần giữ ra ngoài server.
- [ ] Cookie và mật khẩu mặc định đã được xử lý.

## 28. Tài liệu chính thức

- Ubuntu NVIDIA drivers: https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/
- PyTorch Start Locally: https://pytorch.org/get-started/locally/
- PyTorch Previous Versions: https://pytorch.org/get-started/previous-versions/
- Docker Engine Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- Docker Compose: https://docs.docker.com/compose/install/linux/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
