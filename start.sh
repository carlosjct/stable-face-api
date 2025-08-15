#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/app}"
ALT_DIR="/workspace"                     # fallback si usas /workspace
VENV="${VENV:-/opt/venv}"
HOST="0.0.0.0"
PORT="${PORT:-3000}"

echo "== Boot =="
python3 --version || true

# 1) VENV + pip
if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">> creating venv at $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# 2) PyTorch CUDA 11.8 (como ya vienes usando)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118

# 3) Dependencias del repo (si hay requirements.txt)
if [[ -f "$APP_DIR/requirements.txt" ]]; then
  echo ">> installing requirements from $APP_DIR/requirements.txt"
  pip install --no-cache-dir -r "$APP_DIR/requirements.txt" || true
elif [[ -f "$ALT_DIR/requirements.txt" ]]; then
  echo ">> installing requirements from $ALT_DIR/requirements.txt"
  pip install --no-cache-dir -r "$ALT_DIR/requirements.txt" || true
fi

# 4) Asegurar rembg[gpu] y onnxruntime-gpu (corrige CPU->GPU si fuese necesario)
if python -c "import rembg" 2>/dev/null; then
  echo ">> rembg ya presente"
else
  echo ">> installing rembg[gpu]"
  # quitar onnxruntime CPU si vino desde requirements
  pip uninstall -y onnxruntime || true
  pip install --no-cache-dir onnxruntime-gpu==1.16.3
  pip install --no-cache-dir 'rembg[gpu]==2.0.50'
fi

# 5) Extras comunes que ya usas
pip install --no-cache-dir \
  diffusers==0.24.0 transformers==4.35.2 huggingface_hub==0.19.4 \
  accelerate==0.24.1 safetensors==0.3.1 flask gunicorn einops==0.7.0 timm==0.9.16 \
  opencv-python-headless pillow

# 6) Ubicar app dir (prefiere /srv/app)
if [[ -f "$APP_DIR/api.py" || -f "$APP_DIR/app.py" || -f "$APP_DIR/main.py" ]]; then
  cd "$APP_DIR"
elif [[ -f "$ALT_DIR/api.py" || -f "$ALT_DIR/app.py" || -f "$ALT_DIR/main.py" ]]; then
  cd "$ALT_DIR"
else
  echo "❌ No se encontró api.py/app.py/main.py en $APP_DIR ni $ALT_DIR"
  ls -la "$APP_DIR" || true
  ls -la "$ALT_DIR" || true
  exit 1
fi
echo ">> APP at: $(pwd)"

# 7) Caches de HF (opcional)
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# 8) Preflight imports claros
echo ">> preflight import"
python - <<'PY'
import importlib.util, pathlib, sys
for mod in ("torch","diffusers","rembg"):
    try:
        __import__(mod)
        print(f"[ok] {mod}")
    except Exception as e:
        print(f"[ERR] {mod}: {e}")
p = next((pathlib.Path(n) for n in ("api.py","app.py","main.py") if pathlib.Path(n).exists()), None)
if not p: raise SystemExit("no app file found")
spec = importlib.util.spec_from_file_location(p.stem, p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(f"[ok] loaded {p.name}")
print("[ok] ready")
PY

# 9) Detectar módulo/objeto Flask
MOD="$(ls | grep -E '^(api|app|main)\.py$' | head -n1 | sed 's/\.py$//')"
OBJ="$(python - <<PY
import importlib.util, pathlib
p = pathlib.Path("$MOD.py")
s = importlib.util.spec_from_file_location("$MOD", p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print("app" if hasattr(m,"app") else ("application" if hasattr(m,"application") else ""))
PY
)"
if [[ -z "$MOD" || -z "$OBJ" ]]; then
  echo "❌ No Flask app (esperado app/application)"
  exit 1
fi

echo "🚀 gunicorn ${MOD}:${OBJ} on ${HOST}:${PORT}"
exec gunicorn -w 1 -k gthread --threads 8 -b "${HOST}:${PORT}" "${MOD}:${OBJ}"
