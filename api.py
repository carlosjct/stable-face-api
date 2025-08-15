import os, pathlib
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "faceid_enabled": bool(os.getenv("USE_FACEID", "1") == "1")})

# --- config ---
USE_FACEID = os.getenv("USE_FACEID", "1") == "1"
IP_FACEID_PATH = os.getenv("IP_ADAPTER_FACEID_PATH", "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin")
DEVICE = "cuda"  # o "cpu" si quieres forzar

# --- lazy load ---
pipe = None
ip_adapter = None

def load_pipe():
    global pipe
    if pipe is not None:
        return pipe
    print("⏳ Cargando SDXL...", flush=True)
    # TODO: importa y crea el pipeline aquí, con enable_model_cpu_offload() si usas GPU pequeña
    # pipe = StableDiffusionXLPipeline.from_pretrained(...).to(DEVICE)
    return pipe

def load_faceid_adapter():
    global ip_adapter
    if ip_adapter is not None:
        return ip_adapter
    if not USE_FACEID:
        print("⚠️ USE_FACEID=0 → no se cargará FaceID", flush=True)
        return None
    if not os.path.isfile(IP_FACEID_PATH):
        print(f"⚠️ Peso FaceID no encontrado en {IP_FACEID_PATH}. Arrancando sin FaceID.", flush=True)
        return None
    print("⏳ Cargando IP-Adapter FaceID...", flush=True)
    p = load_pipe()
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
    ip_adapter = IPAdapterFaceID(p, IP_FACEID_PATH, device=DEVICE)
    return ip_adapter

@app.post("/generate")
def generate():
    p = load_pipe()
    fa = load_faceid_adapter()  # puede ser None y seguimos
    data = request.get_json(force=True)
    prompt = data.get("prompt", "a photo")
    # TODO: usa p (y fa si no es None) para generar
    return jsonify({"ok": True, "used_faceid": fa is not None})
