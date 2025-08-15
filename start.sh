#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/app}"
VENV="${VENV:-/opt/venv}"
HOST="0.0.0.0"
PORT="${PORT:-3000}"

# ---------- venv ----------
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---------- constraints globales ----------
# Fuerza numpy<2 en TODO lo que se instale después
CONSTR="/tmp/constraints.txt"
echo "numpy<2" > "$CONSTR"

# Limpia restos posibles
pip uninstall -y numpy onnxruntime onnxruntime-gpu || true

# Pin inicial de numpy 1.x
pip install --no-cache-dir --constraint "$CONSTR" "numpy==1.26.4"

# ---------- PyTorch CUDA 11.8 ----------
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118

# ---------- Resto dependencias base ----------
pip install --no-cache-dir --constraint "$CONSTR" \
  diffusers==0.24.0 transformers==4.35.2 huggingface_hub==0.19.4 \
  accelerate==0.24.1 safetensors==0.3.1 flask gunicorn \
  einops==0.7.0 timm==0.9.16 opencv-python-headless pillow

# ---------- ORT GPU + rembg GPU (compatibles con numpy 1.26.x) ----------
pip install --no-cache-dir --constraint "$CONSTR" \
  "onnxruntime-gpu==1.15.1" "rembg[gpu]==2.0.50"

# ---------- requirements del repo (si existen) ----------
if [[ -f "$APP_DIR/requirements.txt" ]]; then
  # APLICAMOS TAMBIÉN LA CONSTRAINT AQUÍ, para impedir que suba numpy
  pip install --no-cache-dir --constraint "$CONSTR" -r "$APP_DIR/requirements.txt"
fi

# ---------- preflight ----------
python - <<'PY'
import importlib
for mod in ("numpy","torch","diffusers","onnxruntime","rembg"):
    try:
        m = importlib.import_module(mod)
        print(f"[ok] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"[ERR] {mod}: {e}")
PY

# ---------- launch ----------
cd "$APP_DIR"
MOD=$(ls | grep -E '^(api|app|main)\.py$' | head -n1 | sed 's/\.py$//')
OBJ=$(python - <<PY
import importlib.util, pathlib
p = pathlib.Path("$MOD.py")
s = importlib.util.spec_from_file_location("$MOD", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("app" if hasattr(m,"app") else ("application" if hasattr(m,"application") else ""))
PY
)
if [[ -z "$OBJ" ]]; then
  echo "❌ No Flask app (esperado app/application)"
  exit 1
fi

echo "🚀 arrancando ${MOD}:${OBJ}"
exec gunicorn -w 1 -k gthread --threads 8 -b "${HOST}:${PORT}" "${MOD}:${OBJ}"
