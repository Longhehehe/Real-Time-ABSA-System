#!/usr/bin/env bash
set -e

# ══════════════════════════════════════════════════════════════════════════════
#  Setup & Training cho NVIDIA A10 (Ampere, Compute 8.6, 24GB VRAM)
#  Dùng uv (Astral) để quản lý venv + packages
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Cài uv (Astral) ───────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo ">>> Cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Load uv vào PATH
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo ">>> uv $(uv --version)"

# ── 2. Kiểm tra NVIDIA driver ────────────────────────────────────────────────
echo ">>> Kiểm tra GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi không tìm thấy — thiếu NVIDIA driver!"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
echo ">>> CUDA Driver Version: $CUDA_VER"

# ── 3. Tạo venv ──────────────────────────────────────────────────────────────
echo ">>> Tạo .venv..."
uv venv .venv
source .venv/bin/activate
echo ">>> Python: $(which python)"

# ── 4. Cài PyTorch CUDA (A10 → cu121) ────────────────────────────────────────
echo ">>> Cài PyTorch với CUDA 12.1..."
uv pip install "torch>=2.1.0" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ── 5. Cài dependencies còn lại (bỏ dòng torch tránh ghi đè CPU build) ──────
echo ">>> Cài packages từ requirements_training.txt..."
grep -v -E "^torch|^#|^$" requirements_training.txt | uv pip install -r /dev/stdin

# ── 6. Xác nhận CUDA hoạt động ───────────────────────────────────────────────
echo ">>> Xác nhận CUDA..."
python -c "
import torch
print(f'  torch   : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
    
else:
    print('  [ERROR] CUDA không khả dụng!')
    exit(1)
"

# ── 7. Chạy training ─────────────────────────────────────────────────────────
echo ">>> Bắt đầu training..."
python train_all_methods.py \
    --data "Augmented Dataset" \
    --output models \
    --model xlm_roberta phobert \
    --device cuda \
    --gamma 2.0 \
    --sentiment_weight 7.0 \
    --label_smoothing 0.1 \
    --epochs 1 \
    --batch_size 16

echo ">>> Hoàn tất!"
