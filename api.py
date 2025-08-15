import io
import os
import math
import base64
import logging
from typing import Optional, Tuple

from flask import Flask, request, send_file, jsonify

import torch
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)

# ---------- Opcionales (no rompen si no están) ----------
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# IP-Adapter FaceID (si existe en vendor)
HAS_FACEID = False
try:
    # Solo si vendorizaste IP-Adapter
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore
    HAS_FACEID = True
except Exception:
    HAS_FACEID = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MODEL_NAME = os.getenv("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# pesos FaceID opcionales
FACEID_CKPT = os.getenv("FACEID_CKPT", "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin")

app = Flask(__name__)

_txt2img = None
_img2img = None
_faceid = None


# --------------- Utilidades ----------------
def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _pil_from_upload() -> Optional[Image.Image]:
    """Lee imagen de:
      - multipart file (field 'image')
      - o JSON: {"image_b64": "..."}  (PNG/JPEG en base64)
      - o JSON: {"image_url": "..."}  (descarga HTTP sencilla)
    """
    if request.files.get("image"):
        return Image.open(request.files["image"].stream).convert("RGB")

    data = {}
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        pass

    b64 = data.get("image_b64")
    if b64:
        try:
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass

    url = data.get("image_url")
    if url:
        try:
            import requests
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            log.warning(f"no pude descargar image_url: {e}")

    return None


def _smart_face_crop(img: Image.Image, target: int = 1024) -> Image.Image:
    """Intenta un recorte tipo headshot: hombros superiores a coronilla.
    Sin dependencias de face-detection: heurística centrada."""
    w, h = img.size
    # recorte centrado cuadrado
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    # subir encuadre para dejar aire arriba (≈ 20%)
    shift = int(side * 0.12)
    img = img.crop((0, shift, side, side)) if shift < side // 2 else img
    img = img.resize((target, target), Image.LANCZOS)
    return img


def _apply_transparency(img: Image.Image) -> Image.Image:
    if not HAS_REMBG:
        return img
    try:
        rgba = rembg_remove(img)
        return rgba
    except Exception as e:
        log.warning(f"rembg fallo: {e}")
        return img


def _boost_prompt(user_prompt: str) -> str:
    booster = (
        " ultra realistic, natural skin texture with pores and fine lines, "
        "photorealistic eyes (no glassiness), micro-contrast, accurate color, "
        "soft studio lighting (no overexposure), no beauty filter, "
        "front-facing headshot from top of shoulders to above the head, centered."
    )
    return f"{user_prompt.strip().rstrip('.')}. {booster}"


def _neg_boost(user_neg: str) -> str:
    extra = (
        " blurry, low contrast, over-smooth skin, plastic skin, waxy, washed-out, "
        "harsh lighting, asymmetrical eyes, misaligned eyes, distortion, artifacts, "
        "profile view, body turned, multiple subjects, collage, duplicate person"
    )
    if user_neg:
        return f"{user_neg}, {extra}"
    return extra


# --------------- Carga de pipelines ----------------
def get_txt2img():
    global _txt2img
    if _txt2img is None:
        log.info(f"⏳ Cargando SDXL txt2img… (device={DEVICE})")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True
        )
        if DEVICE == "cuda":
            pipe.to(DEVICE)
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        _txt2img = pipe
        log.info("✔ txt2img listo")
    return _txt2img


def get_img2img():
    global _img2img
    if _img2img is None:
        log.info(f"⏳ Cargando SDXL img2img… (device={DEVICE})")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True
        )
        if DEVICE == "cuda":
            pipe.to(DEVICE)
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass
        _img2img = pipe
        log.info("✔ img2img listo")
    return _img2img


def get_faceid_adapter(pipe_txt2img):
    """Opcional: IP-Adapter FaceID si existe vendor y ckpt."""
    global _faceid
    if not HAS_FACEID:
        return None
    if not os.path.isfile(FACEID_CKPT):
        log.warning("FACEID ckpt no encontrado; se usará img2img normal.")
        return None
    if _faceid is None:
        try:
            _faceid = IPAdapterFaceID(pipe_txt2img, FACEID_CKPT, device=DEVICE)
            log.info("✔ IP-Adapter FaceID listo")
        except Exception as e:
            log.warning(f"No pude inicializar FaceID: {e}")
            _faceid = None
    return _faceid


# --------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_NAME, "has_rembg": HAS_REMBG, "has_faceid": HAS_FACEID}


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "ultra detailed studio headshot, sharp focus")
    negative = data.get("negative_prompt", "")
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 8.5))
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    seed = data.get("seed")
    transparent = bool(data.get("transparent", False))

    prompt = _boost_prompt(prompt)
    negative = _neg_boost(negative)

    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    pipe = get_txt2img()
    log.info(f"➡️ Txt2Img | steps={steps:02d} guide={guidance:.2f} size={width}x{height} transparent={transparent} seed={seed}")

    png_bytes = None
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width, height=height,
            generator=gen
        )
        img = out.images[0]
        if transparent:
            img = _apply_transparency(img)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        png_bytes = bio.getvalue()
    except Exception as e:
        log.exception("txt2img fallo")
        return jsonify({"ok": False, "error": str(e)}), 500

    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.post("/edit")
def edit_with_headshot():
    """
    Img2Img guiado por headshot.
    Campos:
      - image (multipart) | image_b64 | image_url  (headshot de referencia)
      - prompt (texto de pose/escena)
      - negative_prompt
      - steps (int)
      - guidance (float)
      - strength (float 0..1)  -> fuerza de transformación (más alto = cambia más)
      - identity_strength (0..1) -> peso de identidad (para FaceID; si no hay FaceID se traduce a un strength más bajo)
      - width, height
      - transparent (bool)
    """
    data = request.get_json(silent=True) or {}
    # si viene multipart, request.get_json será None → tomamos defaults
    prompt = (data.get("prompt") or request.form.get("prompt") or
              "professional portrait, realistic texture, soft lighting")
    negative = (data.get("negative_prompt") or request.form.get("negative_prompt") or "")
    steps = int(data.get("steps") or request.form.get("steps") or 30)
    guidance = float(data.get("guidance") or request.form.get("guidance") or 8.0)
    width = int(data.get("width") or request.form.get("width") or 1024)
    height = int(data.get("height") or request.form.get("height") or 1024)
    transparent = (data.get("transparent") if data else None)
    if transparent is None:
        transparent = request.form.get("transparent", "false").lower() == "true"

    # fuerza de difuminado (img2img): más baja = conserva más identidad
    strength = float(data.get("strength") or request.form.get("strength") or 0.28)
    strength = _clamp(strength, 0.05, 0.85)

    identity_strength = float(data.get("identity_strength") or request.form.get("identity_strength") or 0.80)
    identity_strength = _clamp(identity_strength, 0.0, 1.0)

    seed = data.get("seed") or request.form.get("seed")
    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    ref = _pil_from_upload()
    if ref is None:
        return jsonify({"ok": False, "error": "reference image missing (image/image_b64/image_url)"}), 400

    # centramos y escalamos a headshot
    ref_proc = _smart_face_crop(ref, target=max(width, height))

    # prompt booster realista
    prompt = _boost_prompt(prompt)
    negative = _neg_boost(negative)

    png_bytes = None
    try:
        if HAS_FACEID and os.path.isfile(FACEID_CKPT):
            # FaceID: mayor identidad cuanto más grande identity_strength
            base = get_txt2img()
            adapter = get_faceid_adapter(base)
            # el adapter genera variación a partir de prompt y foto de ID (FaceID)
            # Usamos txt2img + FaceID y luego opcionalmente refinamos con img2img suave
            log.info(f"➡️ FaceID | identity_strength={identity_strength:.2f}")
            ImageFile = Image
            # IP-Adapter FaceID espera PIL; algunos forks usan image embeddings,
            # aquí llamamos a su interfaz de conveniencia si está disponible:
            out = adapter.generate(
                prompt=prompt,
                negative_prompt=negative,
                image=ref_proc,
                num_inference_steps=steps,
                guidance_scale=guidance,
                scale=identity_strength,  # peso de identidad
                seed=int(seed) if seed is not None else None,
                width=width, height=height
            )
            img = out[0] if isinstance(out, (list, tuple)) else out

            # opcional refinado muy suave para igualar tonos
            if strength <= 0.35:
                pipe_i2i = get_img2img()
                img = pipe_i2i(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=img,
                    strength=0.20,                 # refinado suave
                    guidance_scale=guidance,
                    num_inference_steps=max(steps - 5, 20),
                    generator=gen
                ).images[0]
        else:
            # Img2img puro (sin FaceID): baja strength para preservar identidad
            log.info(f"➡️ Img2Img | steps={steps:02d} guide={guidance:.2f} strength={strength:.2f} size={width}x{height}")
            pipe_i2i = get_img2img()
            img = pipe_i2i(
                prompt=prompt,
                negative_prompt=negative,
                image=ref_proc,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=gen
            ).images[0]

        if transparent:
            img = _apply_transparency(img)

        bio = io.BytesIO()
        img.save(bio, format="PNG")
        png_bytes = bio.getvalue()
    except Exception as e:
        log.exception("edit fallo")
        return jsonify({"ok": False, "error": str(e)}), 500

    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


# --------- main (para ejecución directa) ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)
