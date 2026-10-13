#!/usr/bin/env bash
# One-shot environment setup on the GPU server. Run from the repository root:
#   bash scripts/setup_server.sh            # CUDA 12.8 wheels (RTX 4090)
#   TORCH_INDEX=cpu bash scripts/setup_server.sh
# Creates .venv, installs the pinned stack, downloads the COCO-pretrained YOLO26 weights and prints the
# environment check (ultralytics/torch versions must match revision/configs/protocol.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_INDEX="${TORCH_INDEX:-cu128}"
PY="${PYTHON:-python3}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  PIP="uv pip install --python .venv/bin/python"
else
  "$PY" -m venv .venv
  PIP=".venv/bin/python -m pip install"
  .venv/bin/python -m pip install --upgrade pip
fi

$PIP torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
$PIP -r requirements-lock.txt

.venv/bin/python -c "import cv2" || {
  echo "OpenCV cannot load (usually libGL.so.1 missing on headless servers)."
  echo "Fix: sudo apt-get install -y libgl1 libglib2.0-0   or   $PIP opencv-python-headless"
  exit 1
}

.venv/bin/python - <<'EOF'
from ultralytics import YOLO
for m in ("yolo26n.pt", "yolo26m.pt", "yolo26l.pt"):
    YOLO(m)  # downloads the COCO checkpoint into the working directory if missing
print("COCO weights ready")
EOF

.venv/bin/python -m revision.protocol --check
echo "setup complete"
