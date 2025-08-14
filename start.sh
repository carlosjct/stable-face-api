#!/bin/bash
set -euo pipefail

VENV=/opt/venv
REPO_URL=${GIT_REPO:-https://github.com/carlosjct/stable-face-api.git}
BRANCH=${GIT_BRANCH:-main}

apt-get update -y && apt-get install -y git wget unzip ca-certificates && update-ca-certificates && rm -rf /var/lib/apt/lists/*

# 0) venv + deps
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
  "$VENV/bin/pip" install --no-cache-dir \
    diffusers==0.24.0 transformers==4.35.2 huggingface_hub==0.19.4 accelerate==0.24.1 \
    safetensors==0.3.1 gunicorn flask insightface==0.7.3 onnxruntime==1.16.3 opencv-python-headless pillow
fi

# 1) Clonar limpio
mkdir -p /workspace
find /workspace -mindepth 1 -maxdepth 1 ! -name ".cache" -exec rm -rf {} +
git clone --depth=1 --branch "$BRANCH" "$REPO_URL" /workspace

# 2) ip_adapter
PKG_ROOT=""
if [ -d /workspace/IP-Adapter/ip_adapter ]; then
  PKG_ROOT="/workspace/IP-Adapter"
else
  echo "⚠️ ip_adapter no está en repo. Descargando..."
  mkdir -p /workspace/_vendor && cd /workspace/_vendor
  wget -q --https-only --no-check-certificate https://github.com/tencent-ailab/IP-Adapter/archive/refs/heads/main.zip -O ipadapter.zip
  unzip -q ipadapter.zip && rm ipadapter.zip
  cp -r IP-Adapter-main/ip_adapter /workspace/IP-Adapter/
  PKG_ROOT="/workspace/IP-Adapter"
fi
echo "$PKG_ROOT" > "$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')/ip_adapter.pth"

# 3) Preflight
cd /workspace
"$VENV/bin/python" - <<'PY'
import sys, pkgutil, importlib.util, pathlib
print("sys.path head:", sys.path[:5])
print("ip_adapter found?", pkgutil.find_loader("ip_adapter") is not None)
p = pathlib.Path("api.py")
spec = importlib.util.spec_from_file_location("api", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("✅ api.py import OK")
PY

# 4) Run
MOD=$(ls | grep -E "^(api|app|main)\.py$" | head -n1 | sed s/.py$//)
exec "$VENV/bin/gunicorn" -w 1 -k gthread --threads 8 -b 0.0.0.0:3000 "${MOD}:app"
