# api.py
# API Flask para SDXL + transparencia automática con rembg
# Endpoints:
#   GET  /           -> info
#   GET  /health     -> ok
#   POST /generate   -> devuelve image/png (RGBA si transparent=true, por defecto)

import io
import os
import logging
from contextlib import nullcontext

from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch

# Diffusers
from diffusers import StableDiffusionXLPipeline

# Background removal
from rembg import remove, new_session

# -------------------------
# Configuración
# -------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

USE_OFFLOAD = os.getenv("USE_OFFLOAD", "0") == "1"
DEFAULT_TRANSPARENT = os.getenv("DEFAULT_TRANSPARENT", "1") == "1"
OUT_DIR = os.getenv("OUT_DIR", "/workspace/outputs")
os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)

# -------------------------
# Singletons
# -------------------------
pipe = None
rembg_session = None


def get_pipe() -> StableDiffusionXLPipeline:
    global pipe
    if pipe is not None:
        return pipe

    log.info(f"⏳ Cargando SDXL: {MODEL_ID} (device={DEVICE}, dtype={DTYPE})")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None,
    )

    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if DEVICE == "cuda":
        if USE_OFFLOAD:
            try:
                pipe.enable_model_cpu_offload()
                log.info("✔ enable_model_cpu_offload() activado")
            except Exception as e:
                log.warning(f"Offload falló ({e}); usando .to('cuda')")
                pipe.to("cuda")
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    log.info("✔ SDXL listo")
    return pipe


def get_rembg_session():
    global rembg_session
    if rembg_session is None:
        log.info("⏳ Cargando modelo rembg (isnet-general-use)...")
        rembg_session = new_session("isnet-general-use")
        log.info("✔ Background removal listo")
    return rembg_session


def pil_to_png_bytes(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def remove_background_rgba(img: Image.Image) -> Image.Image:
    buf_in = io.BytesIO()
    img.save(buf_in, format="PNG")
    out_bytes = remove(buf_in.getvalue(), session=get_rembg_session())
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")


def autocast_ctx():
    return torch.autocast("cuda") if DEVICE == "cuda" else nullcontext()


# -------------------------
# Rutas
# -------------------------
@app.get("/")
def home():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_ID,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "default_transparent": DEFAULT_TRANSPARENT,
            "use_offload": USE_OFFLOAD,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/generate")
def generate():
    data = request.get_json(force=True) if request.data else {}

    # Prompt
    base_prompt = (
        "Ultra detailed, photorealistic, award-winning studio headshot portrait, "
        "sharp focus, realistic skin texture, natural tones, centered composition, 1:1"
    )
    user_prompt = data.get("prompt", "").strip()
    prompt = f"{user_prompt}, {base_prompt}" if user_prompt else base_prompt

    negative = data.get(
        "negative_prompt",
        "blurry, low quality, distorted, cartoon, 3d render, watermark, text, harsh shadows, uneven lighting, artifacts",
    )

    steps = max(int(data.get("steps", 25)), 15)
    guidance = float(data.get("guidance", 7.5))
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    seed = data.get("seed", None)

    transparent = data.get("transparent", DEFAULT_TRANSPARENT)
    bg_hint = data.get("bg_hint", "plain light gray background")
    gen_prompt = f"{prompt}, {bg_hint}" if transparent else prompt

    log.info(
        f"➡️ Generando | steps={steps} guide={guidance} size={width}x{height} "
        f"transparent={transparent} seed={seed}"
    )

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            pass

    p = get_pipe()

    with torch.no_grad(), autocast_ctx():
        result = p(
            prompt=gen_prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        img = result.images[0]

    if transparent:
        try:
            img = remove_background_rgba(img)
        except Exception as e:
            log.warning(f"⚠️ rembg falló ({e}); devolviendo sin transparencia")

    try:
        img.save(os.path.join(OUT_DIR, "last.png"))
    except Exception:
        pass

    return send_file(
        pil_to_png_bytes(img),
        mimetype="image/png",
        as_attachment=False,
        download_name="result.png",
        max_age=0,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
