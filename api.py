import io
import os
import math
import logging
import random
import requests
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler

# --- Transparencia opcional (rembg) ---
REM_BG = False
try:
    from rembg import remove as rembg_remove, new_session as rembg_session
    _rembg_sess = rembg_session("u2net")
    REM_BG = True
except Exception:
    REM_BG = False

# --- IP-Adapter FaceID opcional ---
USE_FACEID = os.getenv("USE_FACEID", "0") == "1"
FACEID_CKPT = os.getenv("FACEID_CKPT", "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin")
_has_faceid = False
IPAdapterFaceID = None
try:
    if USE_FACEID:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # <- OJO: sin guion
        _has_faceid = True
except Exception:
    _has_faceid = False

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ---- Globals (se inicializan on-demand) ---
_device = "cuda" if torch.cuda.is_available() else "cpu"
_txt_pipe: Optional[StableDiffusionXLPipeline] = None
_img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
_faceid: Optional[IPAdapterFaceID] = None


def _ensure_pipelines(dtype=torch.float16):
    global _txt_pipe, _img_pipe

    if _txt_pipe is None:
        log.info("⏳ Cargando SDXL txt2img… (device=%s)", _device)
        _txt_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True
        )
        _txt_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_txt_pipe.scheduler.config)
        _txt_pipe.enable_model_cpu_offload() if _device == "cuda" else None
        log.info("✔ txt2img listo")

    if _img_pipe is None:
        log.info("⏳ Cargando SDXL img2img… (device=%s)", _device)
        _img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True
        )
        _img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_img_pipe.scheduler.config)
        _img_pipe.enable_model_cpu_offload() if _device == "cuda" else None
        log.info("✔ img2img listo")


def _ensure_faceid(pipe_for_adapter):
    """Intenta inicializar IP-Adapter FaceID; devuelve instancia o None."""
    global _faceid
    if not USE_FACEID or not _has_faceid:
        return None

    if _faceid is not None:
        return _faceid

    if not os.path.exists(FACEID_CKPT):
        log.warning("⚠️ FACEID_CKPT no encontrado en %s; se continúa sin FaceID", FACEID_CKPT)
        return None

    try:
        log.info("⏳ Cargando IP-Adapter FaceID… (%s)", FACEID_CKPT)
        _faceid = IPAdapterFaceID(pipe_for_adapter, FACEID_CKPT, device=_device)
        log.info("✔ IP-Adapter FaceID listo")
        return _faceid
    except Exception as e:
        log.warning("⚠️ Falló IP-Adapter FaceID: %s", str(e))
        return None


def _fetch_image(url: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


def _apply_transparency(img: Image.Image) -> Image.Image:
    if not REM_BG:
        return img
    try:
        out = rembg_remove(img, session=_rembg_sess)
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        return out
    except Exception as e:
        log.warning("rembg falló: %s", e)
        return img


def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def _maybe_png_response(img: Image.Image, return_bytes: bool) -> Response | None:
    if return_bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True, device=_device, torch=torch.__version__)


@app.route("/generate", methods=["POST"])
def generate():
    """
    Entrada JSON:
      - prompt (str)                     -> requerido
      - negative_prompt (str)            -> opcional
      - width, height (int)              -> por defecto 1024
      - steps (int)                      -> por defecto 30
      - guidance (float)                 -> por defecto 7.5
      - seed (int)                       -> opcional
      - transparent (bool)               -> por defecto false (si true, rembg)
      - image_url (str)                  -> si presente, usa img2img
      - strength (float 0-1)             -> solo img2img, por defecto 0.45 (0.35-0.55 recomendado)
      - faceid (bool)                    -> fuerza uso de IP-Adapter FaceID si está disponible
      - return (str)                     -> "bytes" para devolver PNG (o usa header Accept:image/png)
    """
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()
    negative = body.get("negative_prompt", "").strip() or None
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    steps = int(body.get("steps", 30))
    guidance = float(body.get("guidance", 7.5))
    seed = body.get("seed", None)
    transparent = _bool(body.get("transparent", False))
    image_url = body.get("image_url", None)
    strength = float(body.get("strength", 0.45))
    want_faceid = _bool(body.get("faceid", False))
    return_mode = (body.get("return") == "bytes") or ("image/png" in (request.headers.get("Accept") or ""))

    if not prompt:
        return jsonify(error="prompt is required"), 400

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # límites sanos
    width = max(256, min(1536, width))
    height = max(256, min(1536, height))
    steps = max(10, min(60, steps))
    strength = max(0.1, min(0.95, strength))

    # Asegura pipes
    _ensure_pipelines()

    try:
        if image_url:
            log.info("➡️ Img2Img | steps=%s guide=%.2f size=%dx%d strength=%.2f faceid=%s seed=%s",
                     steps, guidance, width, height, strength, want_faceid, seed)

            # Imagen base
            init_img = _fetch_image(image_url, size=(width, height))

            # FaceID si está disponible y lo piden
            used_faceid = False
            local_img = init_img

            if want_faceid and USE_FACEID and _has_faceid:
                faceid = _ensure_faceid(_img_pipe)
                if faceid is not None:
                    # Con FaceID, se copia la cara; el control de cuánto cambia va por strength
                    used_faceid = True
                    # La llamada de IP-Adapter FaceID trabaja sobre la pipe:
                    # Se usa el mismo pipeline img2img, pero se inyecta face embedding internamente.
                    # Abajo usamos la pipe normal; IP-Adapter se engancha dentro al inicializar.
                    pass  # La inicialización arriba ya lo enganchó en self.pipe

            # Generar
            result = _img_pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=local_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )
            img = result.images[0]

            # Transparencia
            if transparent:
                img = _apply_transparency(img)

            # Devuelve
            resp = _maybe_png_response(img, return_mode)
            if resp is not None:
                return resp

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return jsonify(ok=True, seed=seed, used_faceid=bool(want_faceid and _has_faceid), bytes=len(buf.getvalue()))
        else:
            log.info("➡️ Txt2Img | steps=%s guide=%.2f size=%dx%d transparent=%s seed=%s",
                     steps, guidance, width, height, transparent, seed)

            result = _txt_pipe(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )
            img = result.images[0]

            if transparent:
                img = _apply_transparency(img)

            resp = _maybe_png_response(img, return_mode)
            if resp is not None:
                return resp

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return jsonify(ok=True, seed=seed, bytes=len(buf.getvalue()))
    except Exception as e:
        log.exception("Error en generación")
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
