#!/usr/bin/env bash
set -euo pipefail

# ===== Config por variables de entorno (puedes sobreescribir en el Pod) =====
export SDXL_MODEL="${SDXL_MODEL:-SG161222/RealVisXL_V5.0}"
export SDXL_REFINER="${SDXL_REFINER:-stabilityai/stable-diffusion-xl-refiner-1.0}"
export USE_REFINER="${USE_REFINER:-1}"                     # 1 = activar refiner
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" # cache HF
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export FACEID_CKPT="${FACEID_CKPT:-/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin}"
export DOWNLOAD_FACEID="${DOWNLOAD_FACEID:-0}"             # 1 = descargar ckpt FaceID
export GIT_REPO="${GIT_REPO:-https://github.com/carlosjct/stable-face-api.git}"
export GIT_BRANCH="${GIT_BRANCH:-main}"

APP=/srv/app
VENV=/opt/venv

echo "== CUDA =="
nvidia-smi || true

echo "== APT =="
apt-get update -y
apt-get install -y --no-install-recommends ca-certificates git wget unzip
update-ca-certificates

# ===== Python venv + dependencias =====
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  # Torch CUDA 11.8
  "$VENV/bin/pip" install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
  # Paquetes base
  "$VENV/bin/pip" install --no-cache-dir \
    numpy==1.26.4 pillow==10.4.0 opencv-python-headless==4.10.0.84 \
    diffusers==0.24.0 transformers==4.35.2 accelerate==0.24.1 safetensors==0.3.1 \
    einops==0.7.0 timm==0.9.16 \
    flask==2.3.3 gunicorn==23.0.0 \
    onnxruntime==1.16.3 rembg==2.0.50
fi

# ===== Clonar SIEMPRE la app a /srv/app (recibe tus pushes) =====
rm -rf "$APP"
git clone --depth=1 --branch "$GIT_BRANCH" "$GIT_REPO" "$APP"

# ===== IP-Adapter: .pth para import =====
PKG_ROOT=""
if [ -d "$APP/IP-Adapter/ip_adapter" ]; then
  PKG_ROOT="$APP/IP-Adapter"
elif [ -d "$APP/IP-Adapter/src/ip_adapter" ]; then
  PKG_ROOT="$APP/IP-Adapter/src"
fi

if [ -n "$PKG_ROOT" ]; then
  SP=$("$VENV/bin/python" - <<'PY'
import site
c=[p for p in site.getsitepackages() if p.endswith("site-packages")]
print(c[0] if c else site.getusersitepackages())
PY
)
  echo "$PKG_ROOT" > "$SP/ip_adapter.pth"
  chmod 644 "$SP/ip_adapter.pth"
  echo "ip_adapter vendorizado en: $PKG_ROOT"
else
  echo "ip_adapter no encontrado en el repo (opcional)."
fi

# ===== (Opcional) Descargar FaceID ckpt si así lo pides =====
if [ "${DOWNLOAD_FACEID}" = "1" ]; then
  mkdir -p /workspace/models/ipadapter
  if [ ! -f "$FACEID_CKPT" ]; then
    echo "Descargando FaceID ckpt a $FACEID_CKPT"
    wget -O "$FACEID_CKPT" "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin?download=true"
  fi
fi

# ===== Mostrar entorno =====
echo "PATH=$PATH"
echo "which python: $(which python || true)"
echo "which gunicorn: $(which gunicorn || true)"
echo "python: $("$VENV/bin/python" -V)"
echo "numpy: $("$VENV/bin/python" - <<'PY'
import numpy as np; print(np.__version__)
PY
)"
echo "torch: $("$VENV/bin/python" - <<'PY'
import torch; print(torch.__version__)
PY
)"

# ===== Lanzar API =====
cd "$APP"
echo "🚀 arrancando api:app (modelo=$SDXL_MODEL, refiner=$SDXL_REFINER, USE_REFINER=$USE_REFINER)"
exec "$VENV/bin/gunicorn" -w 1 -k gthread --threads 8 -b 0.0.0.0:3000 api:app
