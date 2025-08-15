# api.py
# Flask API para SDXL que devuelve image/png en /generate
# - Carga perezosa del pipeline (mejor arranque)
# - Soporta FaceID (IP-Adapter) si USE_FACEID=1 e IP_ADAPTER_FACEID_PATH existe
# - Devuelve directamente bytes PNG (no base64)
# - Endpoints: GET /  | GET /health | POST /generate

import os
import io
import pathlib
import logging
from contextlib import nullcontext

from flask import Flask, jsonify, request, send_file

import torch
from PIL import Image

# -------------------------
# Configuración
# -------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")

# FaceID / IP-Adapter opcional
USE_FACEID = os.getenv("USE_FACEID", "0") == "1"
IP_FACEID_PATH = os.getenv(
    "IP_ADAPTER_FACEID_PATH",
    "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin",
)

# Offload (útil en GPUs pequeñas). Si lo activas, NO hagas .to("cuda") manual
USE_OFFLOAD = os.getenv("USE_OFFLOAD", "1") == "1"

# Directorio para guardar outputs (por si quieres inspeccionar)
OUT_DIR = pathlib.Path(os.getenv("OUT_DIR", "/workspace/outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Singletons perezosos
# -------------------------
pipe = None
ip_adapter = None


def load_pipe():
    """Carga el pipeline SDXL bajo demanda."""
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import StableDiffusionXLPipeline

    app.logger.info(f"⏳ Cargando SDXL: {MODEL_ID} (device={DEVICE}, dtype={DTYPE})")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )

    # Ajustes de memoria
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if DEVICE == "cuda":
        if USE_OFFLOAD:
            # Offload automático: no hagas pipe.to("cuda")
            try:
                pipe.enable_model_cpu_offload()
                app.logger.info("✔ enable_model_cpu_offload() activado")
            except Exception as e:
                app.logger.warning(f"enable_model_cpu_offload falló: {e}. Usando .to('cuda')")
                pipe.to("cuda")
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    app.logger.info("✔ SDXL listo")
    return pipe


def load_faceid():
    """Carga IP-Adapter FaceID si está habilitado y el archivo existe."""
    global ip_adapter
    if ip_adapter is not None:
        return ip_adapter

    if not USE_FACEID:
        app.logger.info("➡️ USE_FACEID=0 → ejecutando sin FaceID")
        return None

    if not os.path.isfile(IP_FACEID_PATH):
        app.logger.warning(f"⚠️ FaceID weight no encontrado: {IP_FACEID_PATH} → continuo sin FaceID")
        return None

    try:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
    except Exception as e:
        app.logger.warning(f"⚠️ ip_adapter no disponible ({e}) → continuo sin FaceID")
        return None

    p = load_pipe()
    app.logger.info("⏳ Cargando IP-Adapter FaceID…")
    try:
        ip_adapter = IPAdapterFaceID(p, IP_FACEID_PATH, device=DEVICE)
        app.logger.info("✔ FaceID cargado")
    except Exception as e:
        app.logger.warning(f"⚠️ No se pudo inicializar FaceID: {e} → continuo sin FaceID")
        ip_adapter = None

    return ip_adapter


# -------------------------
# Utils
# -------------------------
def pil_to_png_bytes(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def autocast_ctx():
    if DEVICE == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


# -------------------------
# Rutas
# -------------------------
@app.get("/")
def home():
    return jsonify(
        {
            "status": "ok",
            "device": DEVICE,
            "dtype": str(DTYPE),
            "model": MODEL_ID,
            "faceid_enabled": USE_FACEID,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/generate")
def generate():
    """
    Body JSON:
    {
      "prompt": "a studio portrait...",
      "negative_prompt": "",
      "steps": 20,
      "guidance": 7.0,
      "width": 768,
      "height": 768,
      "seed": 1234
    }
    Devuelve image/png (bytes).
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        # Si vienen form-data, intenta tomar prompt del query como fallback
        data = {}

    prompt = data.get("prompt", "a high quality photo")
    negative = data.get("negative_prompt", "")
    steps = int(data.get("steps", 20))
    guidance = float(data.get("guidance", 7.0))
    width = int(data.get("width", 768))
    height = int(data.get("height", 768))
    seed = data.get("seed")

    # Carga perezosa
    p = load_pipe()
    _fa = load_faceid()  # (opcional, no se usa si no hay imagen de referencia)

    # Semilla
    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        try:
            gen = gen.manual_seed(int(seed))
        except Exception:
            pass

    # Inferencia
    with torch.no_grad(), autocast_ctx():
        out = p(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=gen,
        )
        img: Image.Image = out.images[0]

    # (Opcional) Guarda una copia en disco para depurar
    try:
        out_path = OUT_DIR / f"gen_{torch.randint(0, 1_000_000, (1,)).item()}.png"
        img.save(out_path)
    except Exception:
        pass

    # Devuelve bytes PNG
    return send_file(
        pil_to_png_bytes(img),
        mimetype="image/png",
        as_attachment=False,
        download_name="result.png",
        max_age=0,
    )


# -------------------------
# Entry local (opcional)
# -------------------------
if __name__ == "__main__":
    # Para pruebas locales: python api.py
    app.run(host="0.0.0.0", port=3000, debug=False)
