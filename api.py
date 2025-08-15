import os
import io
import torch
import logging
from flask import Flask, request, send_file, jsonify
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_FACEID = os.environ.get("USE_FACEID", "0") == "1"

pipe = None

def load_pipe():
    global pipe
    logger.info("⏳ Cargando modelo SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None
    )

    if DEVICE == "cuda":
        pipe = pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        logger.info("✔ enable_model_cpu_offload() activado")

    logger.info("✔ SDXL listo")

# Inicializar API Flask
app = Flask(__name__)
load_pipe()

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)

        # Prompt seguro por defecto
        prompt = data.get("prompt", "ultra detailed, 8k, award winning portrait photo of a person, studio lighting, sharp focus")
        negative = data.get("negative_prompt", "blurry, low quality, distorted, cropped, watermark, text")
        
        steps = max(int(data.get("steps", 25)), 15)
        guidance = float(data.get("guidance", 7.5))
        width = int(data.get("width", 768))
        height = int(data.get("height", 768))
        seed = data.get("seed", None)

        logger.info(f"➡️ Generando imagen: prompt='{prompt[:60]}...' steps={steps} guidance={guidance} size={width}x{height} seed={seed}")

        generator = torch.manual_seed(seed) if seed else None

        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator
        ).images[0]

        # Guardar imagen en memoria y devolver como PNG
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype="image/png")

    except Exception as e:
        logger.exception("❌ Error generando imagen")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "SDXL API is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
