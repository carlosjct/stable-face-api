#!/usr/bin/env bash
set -euo pipefail

echo "=========="
echo "== BOOT =="
echo "=========="

# --- Paquetes base ---
apt-get update -y
apt-get install -y --no-install-recommends git wget unzip ca-certificates
update-ca-certificates

APP_DIR=${APP_DIR:-/srv/app}
REPO_URL=${GIT_REPO:-https://github.com/carlosjct/stable-face-api.git}
REPO_BRANCH=${GIT_BRANCH:-main}
VENV=/opt/venv

# ==== MODELO FOTORREALISTA XL ====
# Puedes cambiarlo por otro luego vía env MODEL_NAME
MODEL_NAME=${MODEL_NAME:-"SG161222/RealVisXL_V4.0"}

# --- venv + deps pinneadas (Numpy<2, Torch cu118) ---
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install --no-cache-dir "numpy==1.26.4"
  "$VENV/bin/pip" install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
fi

# --- Código de la app ---
rm -rf "$APP_DIR"
git clone --depth=1 --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR"

# --- Python deps (si no hay requirements, instala mínimos necesarios) ---
REQ="$APP_DIR/requirements.txt"
if [ -f "$REQ" ]; then
  "$VENV/bin/pip" install --no-cache-dir -r "$REQ"
else
  "$VENV/bin/pip" install --no-cache-dir \
    diffusers==0.24.0 transformers==4.35.2 huggingface_hub==0.19.4 accelerate==0.24.1 safetensors==0.3.1 \
    flask gunicorn pillow opencv-python-headless \
    einops==0.7.0 timm==0.9.16 insightface==0.7.3 onnxruntime==1.16.3 rembg==2.0.50
fi

# --- IP-Adapter (vendor o descarga oficial) ---
PKG_ROOT=""
if [ -d "$APP_DIR/IP-Adapter/ip_adapter" ]; then
  PKG_ROOT="$APP_DIR/IP-Adapter"
elif [ -d "$APP_DIR/IP-Adapter/src/ip_adapter" ]; then
  PKG_ROOT="$APP_DIR/IP-Adapter/src"
else
  echo "⚠️  No hay IP-Adapter en el repo, descargando…"
  mkdir -p "$APP_DIR/_vendor"
  cd "$APP_DIR/_vendor"
  wget -q https://github.com/tencent-ailab/IP-Adapter/archive/refs/heads/main.zip -O ipadapter.zip
  unzip -q ipadapter.zip && rm ipadapter.zip
  mkdir -p "$APP_DIR/IP-Adapter"
  if [ -d IP-Adapter-main/ip_adapter ]; then
    cp -r IP-Adapter-main/ip_adapter "$APP_DIR/IP-Adapter/"
  elif [ -d IP-Adapter-main/src/ip_adapter ]; then
    cp -r IP-Adapter-main/src/ip_adapter "$APP_DIR/IP-Adapter/"
  fi
  PKG_ROOT="$APP_DIR/IP-Adapter"
  cd "$APP_DIR"
fi

# --- Registrar 'ip_adapter' en site-packages ---
SP=$("$VENV/bin/python" - <<'PY'
import site
c=[p for p in site.getsitepackages() if p.endswith("site-packages")]
print(c[0] if c else site.getusersitepackages())
PY
)
if [ -n "$PKG_ROOT" ] && [ -d "$PKG_ROOT/ip_adapter" ]; then
  echo "$PKG_ROOT" > "$SP/ip_adapter.pth"
  chmod 644 "$SP/ip_adapter.pth"
  echo "✔ ip_adapter registrado en $SP/ip_adapter.pth"
else
  echo "⚠️  ip_adapter no encontrado (FaceID podría no funcionar)"
fi

# --- Descargar FaceID ckpt si no existe ---
FACE_DIR=/workspace/models/ipadapter
FACE_BIN=${FACEID_CKPT:-$FACE_DIR/ip-adapter-faceid_sdxl.bin}
mkdir -p "$FACE_DIR"
if [ ! -s "$FACE_BIN" ]; then
  echo "⬇️  Bajando FaceID SDXL .bin…"
  wget -q -O "$FACE_BIN" "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin?download=true"
fi
export FACEID_CKPT="$FACE_BIN"

# --- Exportar el modelo fotorrealista al API ---
export MODEL_NAME

# --- Arrancar API ---
cd "$APP_DIR"
MOD=$(ls | grep -E "^(api|app|main)\.py$" | head -n1 | sed 's/\.py$//')
OBJ=$("$VENV/bin/python" - <<PY
import importlib.util, pathlib
p = pathlib.Path("$MOD.py"); s = importlib.util.spec_from_file_location("$MOD", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("app" if hasattr(m,"app") else ("application" if hasattr(m,"application") else ""))
PY
)
[ -z "$OBJ" ] && { echo "❌ No Flask app en $MOD.py"; exit 1; }
echo "🚀 arrancando $MOD:$OBJ (modelo: $MODEL_NAME)"
exec "$VENV/bin/gunicorn" -w 1 -k gthread --threads 8 -b 0.0.0.0:3000 "$MOD:$OBJ"
