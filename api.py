import io
import os
import sys
import json
import math
import random
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# --- logging ---------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
log = logging.getLogger("api")

# --- config ---------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_NAME = os.getenv("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
HF_HOME = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME

# IP-Adapter vendor + checkpoint
HAS_FACEID = False
FACEID_CKPT = os.getenv(
    "FACEID_CKPT",
    "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin"
)

try:
    # vendorizado: /srv/app/IP-Adapter/ip_adapter/...
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore
    HAS_FACEID = True
    log.info("✔ ip_adapter (FaceID) importado")
except Exception as e:
    HAS_FACEID = False
    log.warning(f"ip_adapter no disponible: {e}")

# rembg (para transparent background opcional)
HAS_REMBG = False
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session  # type: ignore
    _rembg_session = rembg_new_session("u2net")
    HAS_REMBG = True
    log.info("✔ rembg listo")
except Exception as e:
    log.warning(f"rembg no disponible: {e}")
    HAS_REMBG = False

# InsightFace para embeddings
_HAS_INSIGHT = False
try:
    from insightface.app import FaceAnalysis  # type: ignore
    _HAS_INSIGHT = True
    log.info("✔ insightface importado")
except Exception as e:
    _HAS_INSIGHT = False
    log.warning(f"insightface no disponible: {e}")


# --- utilidades -----------------------------------------------------------
def _fetch_image(url: str) -> Image.Image:
    """Descarga imagen desde URL a PIL (RGBA si trae alpha)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGBA" if "A" in Image.open(io.BytesIO(r.content)).getbands() else "RGB")
    return img


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        return img.convert("RGB")
    return img


def _maybe_transparent(png: Image.Image, want_transparent: bool) -> Image.Image:
    if want_transparent:
        # si ya viene RGBA lo dejamos, si no convertimos y pasamos rembg si existe
        if png.mode != "RGBA":
            png = png.convert("RGBA")
        if HAS_REMBG:
            try:
                b = io.BytesIO()
                png.save(b, format="PNG")
                b.seek(0)
                out = rembg_remove(b.getvalue(), session=_rembg_session)
                return Image.open(io.BytesIO(out)).convert("RGBA")
            except Exception as e:
                log.warning(f"rembg fallo, continuo sin transparencia: {e}")
        return png
    else:
        # forzamos RGB si no queremos transparencia
        return png.convert("RGB")


# --- pipelínes ------------------------------------------------------------
_base_txt2img: Optional[StableDiffusionXLPipeline] = None
_base_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
_faceid_adapter: Optional[IPAdapterFaceID] = None
_insight_app: Optional["FaceAnalysis"] = None  # type: ignore


def get_txt2img() -> StableDiffusionXLPipeline:
    global _base_txt2img
    if _base_txt2img is None:
        log.info(f"⏳ Cargando SDXL txt2img: {MODEL_NAME} ({DEVICE}, {DTYPE})")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=False,
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=False)
        _base_txt2img = pipe
        log.info("✔ txt2img listo")
    return _base_txt2img


def get_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global _base_img2img
    if _base_img2img is None:
        log.info(f"⏳ Cargando SDXL img2img… ({DEVICE})")
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=False,
        )
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=False)
        _base_img2img = pipe
        log.info("✔ img2img listo")
    return _base_img2img


def get_faceid_adapter() -> Optional[IPAdapterFaceID]:
    """Crea el adaptador FaceID sólo si tenemos vendor + checkpoint y carga ok."""
    global _faceid_adapter
    if not HAS_FACEID:
        return None
    if not os.path.isfile(FACEID_CKPT):
        log.warning("FACEID_CKPT no encontrado; FaceID deshabilitado.")
        return None
    if _faceid_adapter is None:
        try:
            base = get_txt2img()
            _faceid_adapter = IPAdapterFaceID(base, FACEID_CKPT, device=DEVICE)
            log.info("✔ IP-Adapter FaceID listo")
        except Exception as e:
            log.error(f"No pude inicializar FaceID: {e}")
            _faceid_adapter = None
    return _faceid_adapter


def get_insight_app() -> Optional["FaceAnalysis"]:  # type: ignore
    global _insight_app
    if not _HAS_INSIGHT:
        return None
    if _insight_app is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
        try:
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            _insight_app = app
            log.info("✔ InsightFace listo")
        except Exception as e:
            log.error(f"No pude preparar InsightFace: {e}")
            _insight_app = None
    return _insight_app


def compute_faceid_embedding(img_rgba_or_rgb: Image.Image) -> Optional[torch.Tensor]:
    """Devuelve un tensor (1, dim) en CPU con el embedding facial, o None si no hay cara."""
    app = get_insight_app()
    if app is None:
        return None
    img = _ensure_rgb(img_rgba_or_rgb).convert("RGB")
    # InsightFace espera numpy BGR, pero el wrapper `app.get` acepta RGB in PIL->numpy
    import numpy as np  # local import garantizado (numpy<2 ya en venv)
    arr = np.array(img)  # RGB
    faces = app.get(arr)
    if not faces:
        return None
    # Usamos la cara más grande
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding  # np.float32, shape (512,) típico
    if emb is None:
        return None
    t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).unsqueeze(0)  # (1,512)
    return t


# --- Flask app ------------------------------------------------------------
from flask import Flask, request, Response, jsonify, send_file

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "device": DEVICE,
        "model": MODEL_NAME,
        "has_rembg": HAS_REMBG,
        "has_faceid": bool(get_faceid_adapter() is not None)
    })


def _run_txt2img(prompt: str, negative: str, steps: int, guidance: float, w: int, h: int, seed: Optional[int]) -> Image.Image:
    pipe = get_txt2img()
    g = None
    if seed is not None:
        g = torch.Generator(device=pipe._execution_device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        negative_prompt=negative or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(h),
        width=int(w),
        generator=g
    )
    return out.images[0]


def _run_img2img(init_image: Image.Image, prompt: str, negative: str, steps: int, guidance: float, strength: float, seed: Optional[int]) -> Image.Image:
    pipe = get_img2img()
    g = None
    if seed is not None:
        g = torch.Generator(device=pipe._execution_device).manual_seed(int(seed))
    out = pipe(
        image=init_image,
        prompt=prompt,
        negative_prompt=negative or None,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=g
    )
    return out.images[0]


def _run_faceid(headshot: Image.Image, prompt: str, negative: str, steps: int, guidance: float, w: int, h: int, seed: Optional[int], identity_strength: float) -> Optional[Image.Image]:
    """Genera con IP‑Adapter FaceID. Si falla embedding, devuelve None."""
    adapter = get_faceid_adapter()
    if adapter is None:
        return None

    # 1) embedding
    embeds = compute_faceid_embedding(headshot)
    if embeds is None:
        log.error("No se detectó cara en el headshot (FaceID).")
    else:
        log.info(f"✔ embedding de cara obtenido: shape={tuple(embeds.shape)}")

    # 2) generator/seed
    gen = None
    base = get_txt2img()
    if seed is not None:
        gen = torch.Generator(device=base._execution_device).manual_seed(int(seed))

    # 3) ip_adapter_scale mapea a identidad
    ip_scale = float(max(0.0, min(1.0, identity_strength)))

    # 4) El FaceID oficial acepta `faceid_embeds` cuando no pasas `image`.
    #    Si no hay embeds, devolvemos None para que el caller haga fallback.
    if embeds is None:
        return None

    out = adapter.generate(
        prompt=prompt,
        negative_prompt=negative or None,
        faceid_embeds=embeds,           # <── lo clave para evitar el NoneType
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(h),
        width=int(w),
        ip_adapter_scale=ip_scale,
        generator=gen
    )
    # Algunos forks devuelven PIL directamente, otros dict
    if isinstance(out, Image.Image):
        return out
    if hasattr(out, "images"):
        return out.images[0]
    if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], Image.Image):
        return out[0]
    return None


def _png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


@app.post("/generate")
def generate():
    """
    JSON:
      prompt, negative_prompt, width, height, steps, guidance, seed,
      transparent (bool), return: "bytes"|"base64"|"url",
      image_url (opcional),
      strength (img2img), identity_strength (FaceID)
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"ok": False, "error": "JSON inválido"}), 400

    prompt = str(data.get("prompt", "")).strip()
    negative = str(data.get("negative_prompt", "")).strip()
    w = int(data.get("width", 1024))
    h = int(data.get("height", 1024))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 7.5))
    seed = data.get("seed", None)
    seed = int(seed) if seed is not None else None
    want_transparent = bool(data.get("transparent", False))
    ret_kind = data.get("return", "bytes")

    image_url = data.get("image_url", "").strip()
    strength = float(data.get("strength", 0.25))
    identity_strength = float(data.get("identity_strength", 0.0))

    # --- ruta de decisión ---
    img: Optional[Image.Image] = None

    # 1) FaceID si hay headshot + identity_strength>0 + FaceID disponible
    if image_url and identity_strength > 0.0 and (get_faceid_adapter() is not None):
        try:
            ref = _fetch_image(image_url)
            log.info(f"➡️ FaceID | identity_strength={identity_strength:.2f}")
            img = _run_faceid(
                headshot=ref,
                prompt=prompt,
                negative=negative,
                steps=steps,
                guidance=guidance,
                w=w, h=h,
                seed=seed,
                identity_strength=identity_strength
            )
            if img is None:
                log.warning("FaceID falló o no detectó cara; hago fallback a img2img suave")
                # Fallback a img2img (para no dejar al cliente sin imagen)
                init = _ensure_rgb(ref).resize((w, h), Image.LANCZOS)
                img = _run_img2img(init, prompt, negative, steps, guidance, strength=max(0.05, min(0.35, strength)), seed=seed)
        except Exception as e:
            log.exception("edit fallo")
            return jsonify({"ok": False, "error": str(e)}), 500

    # 2) img2img si hay image_url (y NO FaceID)
    elif image_url:
        ref = _fetch_image(image_url)
        init = _ensure_rgb(ref).resize((w, h), Image.LANCZOS)
        img = _run_img2img(init, prompt, negative, steps, guidance, strength=strength, seed=seed)

    # 3) txt2img directo
    else:
        img = _run_txt2img(prompt, negative, steps, guidance, w, h, seed)

    # transparencia opcional (al final)
    img = _maybe_transparent(img, want_transparent)

    # respuesta
    if ret_kind == "bytes":
        payload = _png_bytes(img)
        return Response(payload, mimetype="image/png")
    else:
        # Para mantener compatibilidad — devolvemos stats simples
        b = _png_bytes(img)
        return jsonify({"ok": True, "bytes": len(b), "seed": seed})

# -------------------------------------------------------------------------
# Arranque local (útil si corres `python api.py` dentro del Pod)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
