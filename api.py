import io
import os
import json
import math
import random
import logging
from io import BytesIO
from typing import Optional

import requests
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_file

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("api")

# -----------------------------------------------------------------------------
# ENV / CONFIG
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("DEVICE", "cuda") != "cpu" else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_NAME = os.getenv("MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
MAX_W = int(os.getenv("MAX_W", "1536"))
MAX_H = int(os.getenv("MAX_H", "1536"))

# FaceID checkpoint (si existe, lo usamos)
FACEID_CKPT = os.getenv("FACEID_CKPT", "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin")

# rembg opcional
HAS_REMBG = False
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session  # type: ignore
    _rembg_session = rembg_new_session("u2net")
    HAS_REMBG = True
except Exception as e:
    log.warning(f"rembg no disponible: {e}")
    HAS_REMBG = False

# IP-Adapter FaceID opcional
HAS_FACEID = False
try:
    # Requiere que el paquete ip_adapter esté resolvible (p.ej. vía .pth en site-packages)
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore
    HAS_FACEID = True
except Exception as e:
    log.warning(f"ip_adapter (FaceID) no disponible: {e}")
    HAS_FACEID = False

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _clamp_size(w: int, h: int) -> tuple[int, int]:
    w = max(64, min(MAX_W, int(w // 8 * 8)))
    h = max(64, min(MAX_H, int(h // 8 * 8)))
    return w, h

def _load_pil_from_url(url: str) -> Image.Image:
    # Algunos CDNs piden User-Agent
    resp = requests.get(url, timeout=30, headers={"User-Agent": "curl/8.5"})
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    return img

def _to_rgb_no_alpha(img: Image.Image) -> Image.Image:
    # FaceID / pipelines esperan RGB (sin alpha)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img.convert("RGB"))
        return bg
    return img.convert("RGB")

def _square_center_crop(img: Image.Image, min_side: int = 512) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side < min_side:
        img = img.resize((min_side, min_side), Image.LANCZOS)
    return img

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------------------------------------------------------
# PIPELINES (lazy singletons)
# -----------------------------------------------------------------------------
_base_pipe: Optional[StableDiffusionXLPipeline] = None
_img2img_pipe: Optional[StableDiffusionXLImg2ImgPipeline] = None
_faceid_adapter: Optional["IPAdapterFaceID"] = None

def get_base_pipe() -> StableDiffusionXLPipeline:
    global _base_pipe
    if _base_pipe is None:
        log.info(f"⏳ Cargando SDXL: {MODEL_NAME} (device={DEVICE}, dtype={DTYPE})")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()  # memoria
        _base_pipe = pipe
        log.info("✔ txt2img listo")
    return _base_pipe

def get_img2img_pipe() -> StableDiffusionXLImg2ImgPipeline:
    global _img2img_pipe
    if _img2img_pipe is None:
        log.info("⏳ Cargando SDXL img2img… (device=cuda)")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True,
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        _img2img_pipe = pipe
        log.info("✔ img2img listo")
    return _img2img_pipe

def get_faceid_adapter(base_pipe: StableDiffusionXLPipeline) -> Optional["IPAdapterFaceID"]:
    global _faceid_adapter
    if not HAS_FACEID:
        return None
    if not os.path.isfile(FACEID_CKPT):
        log.warning("FACEID ckpt no encontrado; se usará img2img normal.")
        return None
    if _faceid_adapter is None:
        try:
            _faceid_adapter = IPAdapterFaceID(base_pipe, FACEID_CKPT, device=DEVICE)
            log.info("✔ IP-Adapter FaceID listo")
        except Exception as e:
            log.warning(f"No pude inicializar FaceID: {e}")
            _faceid_adapter = None
    return _faceid_adapter

# -----------------------------------------------------------------------------
# CORE GENERATORS
# -----------------------------------------------------------------------------
def generate_txt2img(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: Optional[int],
) -> Image.Image:
    base = get_base_pipe()
    g = None
    if seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(seed)
    w, h = _clamp_size(width, height)
    out = base(
        prompt=prompt,
        negative_prompt=(negative_prompt or None),
        num_inference_steps=max(10, steps),
        guidance_scale=guidance,
        width=w,
        height=h,
        generator=g,
    ).images[0]
    return out

def refine_with_img2img(
    init_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    strength: float,
    seed: Optional[int],
) -> Image.Image:
    img2img = get_img2img_pipe()
    g = None
    if seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(seed)
    w, h = _clamp_size(width, height)
    init = init_image.convert("RGB").resize((w, h), Image.LANCZOS)
    out = img2img(
        prompt=prompt,
        negative_prompt=(negative_prompt or None),
        image=init,
        num_inference_steps=max(10, steps),
        guidance_scale=guidance,
        strength=min(0.95, max(0.01, strength)),
        generator=g,
    ).images[0]
    return out

def edit_with_headshot(
    image_url: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    strength: float,
    identity_strength: float,
    seed: Optional[int],
) -> Image.Image:
    """
    Usa IP-Adapter FaceID si está disponible; si no, cae a img2img.
    """
    base = get_base_pipe()
    adapter = get_faceid_adapter(base)
    w, h = _clamp_size(width, height)
    g_seed = seed if seed is not None else random.randint(1, 2**31 - 1)

    # Cargar headshot de referencia
    ref = _load_pil_from_url(image_url)
    ref = _to_rgb_no_alpha(ref)
    ref = _square_center_crop(ref, min_side=512)

    if adapter is not None:
        log.info(f"➡️ FaceID | identity_strength={identity_strength:.2f}")
        out = adapter.generate(
            prompt=prompt,
            negative_prompt=(negative_prompt or None),
            num_samples=1,
            width=w,
            height=h,
            num_inference_steps=max(10, steps),
            guidance_scale=guidance,
            seed=g_seed,
            id_images=[ref],  # <— clave: calcular embeddings internamente
            s_scale=max(0.1, min(1.0, identity_strength)),
        )
        img = out[0] if isinstance(out, (list, tuple)) else out
        if not isinstance(img, Image.Image):
            raise RuntimeError("FaceID devolvió un resultado inesperado")
        # Refinado suave opcional para pulir detalles sin perder identidad
        if strength and strength > 0:
            img = refine_with_img2img(
                init_image=img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=w,
                height=h,
                steps=max(10, steps // 2),
                guidance=guidance,
                strength=min(0.35, max(0.05, strength)),
                seed=g_seed,
            )
        return img

    # Fallback: img2img clásico
    log.info("➡️ Fallback img2img (sin FaceID)")
    init = ref.resize((w, h), Image.LANCZOS)
    img = refine_with_img2img(
        init_image=init,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=w,
        height=h,
        steps=steps,
        guidance=guidance,
        strength=min(0.45, max(0.05, strength or 0.25)),
        seed=g_seed,
    )
    return img

def remove_background(img: Image.Image) -> Image.Image:
    if not HAS_REMBG:
        return img  # sin rembg, devolvemos la original
    try:
        b = rembg_remove(img.convert("RGBA"), session=_rembg_session)
        if isinstance(b, bytes):
            b = Image.open(BytesIO(b)).convert("RGBA")
        else:
            b = b.convert("RGBA")
        return b
    except Exception as e:
        log.warning(f"rembg falló: {e}")
        return img

# -----------------------------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "device": DEVICE,
        "model": MODEL_NAME,
        "has_rembg": HAS_REMBG,
        "has_faceid": HAS_FACEID,
    })

@app.route("/generate", methods=["POST"])
def generate():
    """
    JSON:
      - prompt (str) [requerido]
      - negative_prompt (str)
      - width, height (int)
      - steps (int)
      - guidance (float)
      - seed (int)
      - transparent (bool)
      - return: "json" | "png" | "bytes" (default: "json")
      - image_url (str) -> si viene, intentamos FaceID; si no hay, fallback img2img
      - identity_strength (float 0..1) -> FaceID
      - strength (float 0..1) -> refinado img2img
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"ok": False, "error": "JSON inválido"}), 400

    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "prompt requerido"}), 400

    negative_prompt = (data.get("negative_prompt") or "").strip()
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 7.5))
    seed = data.get("seed", None)
    if seed is not None:
        seed = int(seed)
    transparent = bool(data.get("transparent", False))
    ret = (data.get("return") or "json").lower()
    image_url = (data.get("image_url") or "").strip()
    identity_strength = float(data.get("identity_strength", 0.9))
    strength = float(data.get("strength", 0.15))

    log.info(
        f"➡️ Generando | steps={steps} guide={guidance:.2f} size={width}x{height} transparent={transparent} "
        f"seed={seed} image_url={'yes' if image_url else 'no'} faceid={HAS_FACEID}"
    )

    try:
        if image_url:
            img = edit_with_headshot(
                image_url=image_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                strength=strength,
                identity_strength=identity_strength,
                seed=seed,
            )
        else:
            img = generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )

        if transparent:
            img = remove_background(img)
            # Asegura RGBA si pedimos transparencia
            if img.mode != "RGBA":
                img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        if ret == "png":
            # Devuelve imagen directa
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            bio.seek(0)
            return send_file(bio, mimetype="image/png")

        # default: json o bytes
        png_bytes = _pil_to_png_bytes(img)
        if ret == "bytes":
            return send_file(
                io.BytesIO(png_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name="image.png",
            )

        # JSON con tamaño + base64 opcional (apagado por defecto para no inflar)
        return jsonify({
            "ok": True,
            "bytes": len(png_bytes),
            "seed": seed if seed is not None else None,
        })
    except Exception as e:
        log.exception("generate falló")
        return jsonify({"ok": False, "error": str(e)}), 500


# For RunPod health check root
@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "service": "stable-face-api"})


# Desarrollo local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
