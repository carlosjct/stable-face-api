import os
os.environ["HF_HUB_DOWNLOAD_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
import torch
from PIL import Image
import requests
import io
import uuid
import insightface
import cv2
import numpy as np

app = Flask(__name__)

# ---------- CONFIG ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
ip_model_path = "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin"

# ---------- LOAD PIPELINE & ADAPTER ----------
print("⏳ Cargando modelo SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(device)

pipe.enable_model_cpu_offload()

print("⏳ Cargando IP-Adapter-FaceID...")
ip_adapter = IPAdapterFaceID(pipe, ip_model_path, device=device)

print("⏳ Cargando InsightFace...")
face_app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- UTILS ----------
def download_image(url):
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        if img.mode == "P":
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception as e:
        print(f"❌ Error al descargar imagen: {e}")
        return None

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------- API ROUTE ----------
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        steps = int(data.get("steps", 30))
        image_url = data.get("image_url")

        image_embeds = None

        if image_url:
            print("📷 Usando imagen de referencia...")
            face_image = download_image(image_url)
            if face_image is None:
                return jsonify({"error": "No se pudo descargar la imagen"}), 400

            face_np = pil_to_cv2(face_image)
            faces = face_app.get(face_np)
            if not faces:
                return jsonify({"error": "No se detectó rostro válido en la imagen"}), 400

            # ✅ Usar la imagen directamente como PIL
            image_embeds = ip_adapter.get_image_embeds(face_image)
            ip_adapter.set_image_embeds(image_embeds)

        print("🎨 Generando imagen...")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
        ).images[0]

        output_path = f"/tmp/output_{uuid.uuid4().hex}.png"
        result.save(output_path)

        return send_file(output_path, mimetype="image/png")

    except Exception as e:
        print(f"❌ Error en generación: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
