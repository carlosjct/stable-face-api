
import io
import os
import math
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps
import requests

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,  # refiner / img2img
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)

# ---- opcionales ----
HAS_REMBG = True
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _rembg_session = rembg_new_session("u2net")
except Exception:
    HAS_REMBG = False
    _rembg_session = None

# ---- IP-Adapter FaceID vendorizado (opcional) ----
HAS_FACEID = False
try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore
    HAS_FACEID = True
except Exception:
    HAS_FACEID = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_NAME = os.getenv("SDXL_MODEL", "SG161222/RealVisXL_V4.0")
REFINER_NAME = os.getenv("SDXL_REFINER", "stabilityai/stable-diffusion-xl-refiner-1.0")
USE_REFINER = os.getenv("USE_REFINER", "1") == "1"

# VAE compartido
SDXL_VAE = os.getenv("SDXL_VAE", "madebyollin/sdxl-vae-fp16-fix")

# pesos FaceID (opcional)
FACEID_CKPT = os.getenv(
    "FACEID_CKPT",
    "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin",
)

# caches HF
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/.cache/huggingface")

app = Flask(__name__)

_base: Optional[StableDiffusionXLPipeline] = None
_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
_faceid: Optional["IPAdapterFaceID"] = None  # type: ignore

_vae = None
def _load_vae():
    global _vae
    if _vae is None:
        log.info(f"⏳ Cargando VAE: {SDXL_VAE}")
        try:
            _vae_local = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=DTYPE)
        except Exception as e:
            log.warning(f"No se pudo cargar VAE {SDXL_VAE}: {e}. Usando el del checkpoint.")
            _vae_local = None
        if _vae_local is not None and DEVICE == "cuda":
            _vae_local = _vae_local.to(DEVICE)
        _vae = _vae_local
        log.info("✔ VAE listo")
    return _vae

def _load_base() -> StableDiffusionXLPipeline:
    global _base
    if _base is None:
        log.info(f"⏳ Cargando SDXL base: {MODEL_NAME} (device={DEVICE})")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, use_safetensors=True, variant="fp16", vae=_load_vae()
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
        _base = pipe
        log.info("✔ txt2img listo")
    return _base

def _load_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global _img2img
    if _img2img is None:
        log.info(f"⏳ Cargando SDXL img2img / refiner: {REFINER_NAME} (device={DEVICE})")
        p = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            REFINER_NAME, torch_dtype=DTYPE, use_safetensors=True, variant="fp16", vae=_load_vae()
        )
        p.scheduler = EulerAncestralDiscreteScheduler.from_config(p.scheduler.config)
        if DEVICE == "cuda":
            p.enable_model_cpu_offload()
            p.enable_vae_tiling()
        _img2img = p
        log.info("✔ img2img listo")
    return _img2img

def _load_faceid_adapter(base_pipe: StableDiffusionXLPipeline) -> Optional["IPAdapterFaceID"]:
    global _faceid
    if not HAS_FACEID:
        return None
    if not os.path.isfile(FACEID_CKPT):
        log.warning("FACEID ckpt no encontrado; ignorando FaceID.")
        return None
    if _faceid is None:
        try:
            _faceid = IPAdapterFaceID(base_pipe, FACEID_CKPT, device=DEVICE)
            log.info("✔ IP-Adapter FaceID listo")
        except Exception as e:
            log.warning(f"No pude inicializar FaceID: {e}")
            _faceid = None
    return _faceid

def _fetch_image(url: str, size: Tuple[int, int]) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img = ImageOps.contain(img, size, method=Image.LANCZOS)
    bg = Image.new("RGB", size, (255, 255, 255))
    bg.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return bg

REALISM_ADD = (
    "ultra realistic, photorealistic, 8k, professional studio photography, "
    "natural skin texture, detailed pores, lifelike hair, accurate facial proportions, "
    "soft diffused lighting, sharp focus on eyes, subtle depth of field, "
    "no cgi, no 3d render, no cartoon"
)
REALISM_NEG = (
    "cgi, 3d render, cartoon, illustration, painting, plastic skin, waxy skin, "
    "over-smooth, over-sharpen, low quality, artifacts, text, watermark, extra fingers, "
    "cropped face, disfigured, deformed"
)

def _compose_prompts(prompt: str, negative_prompt: str) -> Tuple[str, str]:
    p = f"{prompt}, {REALISM_ADD}" if prompt else REALISM_ADD
    n = f"{negative_prompt}, {REALISM_NEG}" if negative_prompt else REALISM_NEG
    return p[:1800], n[:1800]

def _maybe_transparent(img: Image.Image, make_transparent: bool) -> Image.Image:
    if not make_transparent:
        return img
    if not HAS_REMBG:
        return img
    try:
        out = rembg_remove(img, session=_rembg_session)
        return out
    except Exception as e:
        log.warning(f"rembg falló: {e}")
        return img

def _to_response(img: Image.Image, as_bytes: bool, fmt: str = "PNG"):
    buf = io.BytesIO()
    fmt = (fmt or "PNG").upper()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs.update({"quality": 92, "optimize": True})
    img.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    if as_bytes:
        return buf
    return send_file(buf, mimetype=f"image/{fmt.lower()}")

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "device": DEVICE,
        "model": MODEL_NAME,
        "use_refiner": USE_REFINER,
        "has_rembg": HAS_REMBG,
        "has_faceid": HAS_FACEID and os.path.isfile(FACEID_CKPT),
    })

@dataclass
class GenParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int]
    transparent: bool
    ret_bytes: bool
    image_url: Optional[str]
    strength: float
    identity_strength: float
    # NUEVO: ajuste de pose sólo con texto (opcional)
    pose_prompt: Optional[str] = None
    pose_strength: float = 0.15     # fuerza del paso img2img de pose
    pose_steps: int = 20            # pasos del paso de pose

def _parse_json() -> GenParams:
    j = request.get_json(force=True, silent=True) or {}
    return GenParams(
        prompt=j.get("prompt", ""),
        negative_prompt=j.get("negative_prompt", ""),
        width=int(j.get("width", 1024)),
        height=int(j.get("height", 1024)),
        steps=int(j.get("steps", 45)),
        guidance=float(j.get("guidance", 5.5)),
        seed=j.get("seed", None),
        transparent=bool(j.get("transparent", False)),
        ret_bytes=j.get("return", "").lower() == "bytes",
        image_url=j.get("image_url"),
        strength=float(j.get("strength", 0.20)),
        identity_strength=float(j.get("identity_strength", 0.85)),
        pose_prompt=j.get("pose_prompt"),
        pose_strength=float(j.get("pose_strength", 0.15)),
        pose_steps=int(j.get("pose_steps", 20)),
    )

def _apply_refiner_if_needed(img: Image.Image, prompt: str, negative_prompt: str, steps: int, guidance: float) -> Image.Image:
    if not USE_REFINER:
        return img
    ref = _load_img2img()
    p, n = _compose_prompts(prompt, negative_prompt)
    with torch.inference_mode():
        out = ref(
            prompt=p,
            negative_prompt=n,
            image=img,
            strength=0.10,
            num_inference_steps=max(10, steps // 2),
            guidance_scale=guidance,
        )
    return out.images[0]

def _txt2img(params: GenParams) -> Image.Image:
    base = _load_base()
    p, n = _compose_prompts(params.prompt, params.negative_prompt)
    g = None
    if params.seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(params.seed))
    log.info(f"➡️ Txt2Img | steps={params.steps} guide={params.guidance:.2f} size={params.width}x{params.height} transparent={params.transparent} seed={params.seed}")
    with torch.inference_mode():
        out = base(
            prompt=p,
            negative_prompt=n,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=g,
        )
    img = out.images[0]
    img = _apply_refiner_if_needed(img, p, n, params.steps, params.guidance)
    img = _maybe_transparent(img, params.transparent)
    return img

def _edit_with_faceid(params: GenParams) -> Optional[Image.Image]:
    base = _load_base()
    adapter = _load_faceid_adapter(base)
    if adapter is None:
        return None
    try:
        src = _fetch_image(params.image_url, (params.width, params.height))
    except Exception as e:
        log.warning(f"No pude descargar image_url: {e}")
        return None
    p, n = _compose_prompts(params.prompt, params.negative_prompt)
    g = None
    if params.seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(params.seed))
    faceid_scale = max(0.1, min(1.2, params.identity_strength * 1.2))
    img2img_strength = max(0.05, min(0.6, params.strength))
    log.info(f"➡️ FaceID | identity={params.identity_strength:.2f} (scale={faceid_scale:.2f}) strength={img2img_strength:.2f}")
    with torch.inference_mode():
        out = adapter.generate(
            prompt=p,
            negative_prompt=n,
            image_pil=src,
            scale=faceid_scale,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=g,
            width=params.width,
            height=params.height,
        )
    img = out.images[0]
    # Ajuste de pose opcional (texto) ANTES del refiner
    img = _apply_pose_tweak_if_requested(img, params)
    # Refiner y transparencia
    img = _apply_refiner_if_needed(img, p, n, params.steps, params.guidance)
    img = _maybe_transparent(img, params.transparent)
    return img

def _img2img_plain(params: GenParams) -> Optional[Image.Image]:
    src = _fetch_image(params.image_url, (params.width, params.height))
    p, n = _compose_prompts(params.prompt, params.negative_prompt)
    g = None
    if params.seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(params.seed))
    pipe = _load_img2img()
    strength = max(0.05, min(0.6, params.strength))
    log.info(f"➡️ Img2Img | strength={strength:.2f}")
    with torch.inference_mode():
        out = pipe(
            prompt=p,
            negative_prompt=n,
            image=src,
            strength=strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=g,
        )
    img = out.images[0]
    img = _apply_pose_tweak_if_requested(img, params)
    img = _maybe_transparent(img, params.transparent)
    return img

# -------- NUEVO: ajuste de pose sólo texto (pequeño img2img) --------
POSE_NEG = "straight gaze, looking at camera, head straight"

def _apply_pose_tweak_if_requested(img: Image.Image, params: GenParams) -> Image.Image:
    if not params.pose_prompt:
        return img
    pipe = _load_img2img()
    # Combinamos prompt con indicaciones de pose, y reforzamos el negativo típico que compite
    pose_p = f"{params.pose_prompt}"
    pose_n = f"{POSE_NEG}, {params.negative_prompt}" if params.negative_prompt else POSE_NEG
    steps = max(10, int(params.pose_steps))
    strength = max(0.05, min(0.35, float(params.pose_strength)))  # bajo para no destruir identidad
    log.info(f"↪️ Pose tweak | strength={strength:.2f} steps={steps} prompt='{pose_p}'")
    with torch.inference_mode():
        out = pipe(
            prompt=pose_p,
            negative_prompt=pose_n,
            image=img,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=max(5.0, params.guidance),  # un poco más de guide ayuda a obedecer
        )
    return out.images[0]

@app.post("/generate")
def generate():
    try:
        params = _parse_json()
        if params.image_url:
            img = _edit_with_faceid(params)
            if img is None:
                img = _img2img_plain(params)
        else:
            img = _txt2img(params)
        fmt = request.args.get("format", "png")
        if params.ret_bytes:
            blob = _to_response(img, as_bytes=True, fmt=fmt)
            return jsonify({"ok": True, "bytes": len(blob.getvalue()), "seed": params.seed})
        else:
            return _to_response(img, as_bytes=False, fmt=fmt)
    except Exception as e:
        log.exception("generate falló")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run("0.0.0.0", 3000, debug=False)
