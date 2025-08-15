
import io
import os
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps, ImageFile

# ------------- Logging -------------
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:api:%(message)s")
log.setLevel(logging.INFO)

# ------------- Env / Defaults -------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_NAME = os.getenv("SDXL_MODEL", "SG161222/RealVisXL_V4.0")
REFINER_NAME = os.getenv("SDXL_REFINER", "stabilityai/stable-diffusion-xl-refiner-1.0")
SDXL_VAE = os.getenv("SDXL_VAE", "madebyollin/sdxl-vae-fp16-fix")

# FaceID adapter (ya lo tienes en disco)
FACEID_ADAPTER_PATH = os.getenv("FACEID_ADAPTER_PATH", "/srv/app/models/ip-adapter-faceid_sdxl.bin")
# Adapter de composición (pose/encuadre)
COMPOSE_ADAPTER_REPO = os.getenv("COMPOSE_ADAPTER_REPO", "h94/IP-Adapter")
COMPOSE_ADAPTER_SUBMODEL = os.getenv("COMPOSE_ADAPTER_SUBMODEL", "ip-adapter-plus_sdxl_vit-h")

# ------------- Diffusers / Pipelines -------------
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)

# IP-Adapter util (nuevo de diffusers>=0.24)
from diffusers.loaders import IPAdapterMixin

# ------------- Robust PIL -------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _fetch_image(url_or_path: str, resize_to: Optional[Tuple[int, int]]=None) -> Image.Image:
    import requests
    if not url_or_path:
        raise ValueError("image_url vacío")
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        r = requests.get(url_or_path, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(url_or_path)

    if img.mode in ("P", "LA", "L"):
        img = img.convert("RGBA")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    if resize_to is not None:
        img = img.resize(resize_to, Image.LANCZOS)
    return img

def _maybe_transparent(img: Image.Image, transparent: bool) -> Image.Image:
    if not transparent:
        return img
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # recorta fondo liso si lo hay
    bg = Image.new("RGBA", img.size, (0,0,0,0))
    return Image.alpha_composite(bg, img)

# ------------- Params -------------
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
    # FaceID
    image_url: Optional[str] = None
    identity_strength: Optional[float] = None
    strength: Optional[float] = None
    # Composition / Pose
    pose_image_url: Optional[str] = None
    pose_weight: Optional[float] = None

# ------------- Global state -------------
_base_pipe: Optional[StableDiffusionXLPipeline] = None
_refiner: Optional[StableDiffusionXLImg2ImgPipeline] = None
_loaded_faceid: bool = False
_loaded_compose: bool = False

def _load_vae():
    vae = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=DTYPE)
    if DEVICE == "cuda":
        vae = vae.to(DEVICE)
    return vae

def _load_base() -> StableDiffusionXLPipeline:
    global _base_pipe
    if _base_pipe is not None:
        return _base_pipe
    log.info(f"⏳ Cargando base SDXL: {MODEL_NAME}")
    vae = _load_vae()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None,
        vae=vae,
    )
    # Scheduler recomendado para RealVis
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    if DEVICE == "cuda":
        pipe = pipe.to(DEVICE)
    _base_pipe = pipe
    return pipe

def _load_refiner() -> StableDiffusionXLImg2ImgPipeline:
    global _refiner
    if _refiner is not None:
        return _refiner
    log.info(f"⏳ Cargando refiner: {REFINER_NAME}")
    vae = _load_vae()  # compartir VAE
    p = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_NAME,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None,
        vae=vae,
    )
    p.scheduler = EulerAncestralDiscreteScheduler.from_config(p.scheduler.config)
    p.enable_vae_tiling()
    if DEVICE == "cuda":
        p = p.to(DEVICE)
    _refiner = p
    return p

# ------------- IP-Adapter loaders -------------
def _load_faceid_adapter(pipe: StableDiffusionXLPipeline):
    global _loaded_faceid
    if _loaded_faceid:
        return
    if not os.path.exists(FACEID_ADAPTER_PATH):
        log.warning("No se encontró FACEID_ADAPTER_PATH, FaceID deshabilitado.")
        return
    # Cargar FaceID
    pipe.load_ip_adapter(FACEID_ADAPTER_PATH, subfolder=None)
    _loaded_faceid = True
    log.info("✔ IP-Adapter FaceID listo")

def _load_compose_adapter(pipe: StableDiffusionXLPipeline):
    global _loaded_compose
    if _loaded_compose:
        return
    # Cargar adapter de composición (encuadre/pose) desde repo
    try:
        pipe.load_ip_adapter(COMPOSE_ADAPTER_REPO, subfolder=COMPOSE_ADAPTER_SUBMODEL)
        _loaded_compose = True
        log.info(f"✔ IP-Adapter Composition listo ({COMPOSE_ADAPTER_SUBMODEL})")
    except Exception as e:
        log.warning(f"No se pudo cargar Composition adapter: {e}")

# ------------- Prompt helpers -------------
def _compose_prompts(p: Optional[str], n: Optional[str]) -> Tuple[str, str]:
    base_p = (p or "").strip()
    base_n = (n or "").strip()
    return base_p, base_n

# ------------- Refiner -------------
def _apply_refiner_if_needed(img: Image.Image, prompt: str, negative: str, steps: int, guidance: float) -> Image.Image:
    try:
        refiner = _load_refiner()
    except Exception:
        return img
    # refiner suave
    strength = 0.10
    with torch.inference_mode():
        out = refiner(
            prompt=prompt,
            negative_prompt=negative,
            image=img,
            num_inference_steps=max(steps // 2, 15),
            guidance_scale=guidance,
            strength=strength,
        )
    return out.images[0]

# ------------- Embeddings helpers -------------
def _get_faceid_embeds(pipe: StableDiffusionXLPipeline, image: Image.Image):
    # diffusers reciente expone get_faceid_embeds via pipe.ip_adapter
    try:
        return pipe.get_ip_adapter_embedder("face_id").get_image_embeds(image).to(DEVICE, DTYPE)
    except Exception:
        # fallback a API genérica del primer adapter cargado
        try:
            return pipe.get_ip_adapter_image_embeds(image).to(DEVICE, DTYPE)
        except Exception as e:
            log.warning(f"FaceID embeds fallo: {e}")
            return None

def _get_compose_embeds(pipe: StableDiffusionXLPipeline, image: Image.Image):
    try:
        return pipe.get_ip_adapter_embedder("image").get_image_embeds(image).to(DEVICE, DTYPE)
    except Exception:
        try:
            return pipe.get_ip_adapter_image_embeds(image).to(DEVICE, DTYPE)
        except Exception as e:
            log.warning(f"Compose embeds fallo: {e}")
            return None

# ------------- Core FaceID + Composition -------------
def _edit_with_faceid_and_pose(params: GenParams) -> Optional[Image.Image]:
    pipe = _load_base()
    _load_faceid_adapter(pipe)
    _load_compose_adapter(pipe)  # opcional, si falla sólo usamos FaceID

    if not params.image_url:
        return None

    # Cargar referencia de identidad (RGB)
    src = _fetch_image(params.image_url)
    if src.mode == "RGBA":
        src_rgb = src.convert("RGB")
    else:
        src_rgb = src.convert("RGB")

    # Embeddings FaceID
    face_embeds = _get_faceid_embeds(pipe, src_rgb)
    if face_embeds is None:
        log.warning("faceid_embeds=None; abortando FaceID.")
        return None

    # Embeddings de composición (opcional)
    compose_embeds = None
    pose_weight = float(params.pose_weight or 0.6)
    if params.pose_image_url:
        try:
            pose_img = _fetch_image(params.pose_image_url, (params.width, params.height) if params.width and params.height else None)
            if pose_img.mode == "RGBA":
                pose_img = pose_img.convert("RGB")
            compose_embeds = _get_compose_embeds(pipe, pose_img)
        except Exception as e:
            log.warning(f"No pude usar pose_image_url: {e}")

    identity = float(params.identity_strength or 0.9)
    face_scale = max(0.1, min(1.5, 0.6 + identity * 0.6))  # 0.6–1.5 aprox
    log.info(f"➡️ FaceID | identity={identity:.2f} (scale={face_scale:.2f})  Pose weight={pose_weight:.2f}")

    g = None
    if params.seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(params.seed))

    ptxt, ntxt = _compose_prompts(params.prompt, params.negative_prompt)

    # Construir lista de embeds y scales
    embeds_list: List[torch.Tensor] = [face_embeds]
    scales: List[float] = [face_scale]
    if compose_embeds is not None:
        embeds_list.append(compose_embeds)
        scales.append(float(pose_weight))

    with torch.inference_mode():
        out = pipe(
            prompt=ptxt,
            negative_prompt=ntxt,
            ip_adapter_image_embeds=embeds_list,
            ip_adapter_scale=scales,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=g,
        )

    img = out.images[0]
    img = _apply_refiner_if_needed(img, ptxt, ntxt, params.steps, params.guidance)
    img = _maybe_transparent(img, params.transparent)
    return img

# ------------- Plain txt2img -------------
def _txt2img(params: GenParams) -> Image.Image:
    pipe = _load_base()
    ptxt, ntxt = _compose_prompts(params.prompt, params.negative_prompt)
    g = None
    if params.seed is not None:
        g = torch.Generator(device=DEVICE).manual_seed(int(params.seed))
    with torch.inference_mode():
        out = pipe(
            prompt=ptxt,
            negative_prompt=ntxt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=g,
        )
    img = out.images[0]
    img = _apply_refiner_if_needed(img, ptxt, ntxt, params.steps, params.guidance)
    img = _maybe_transparent(img, params.transparent)
    return img

# ------------- Flask -------------
app = Flask(__name__)

def _parse_params(js: dict) -> GenParams:
    return GenParams(
        prompt=js.get("prompt", ""),
        negative_prompt=js.get("negative_prompt", ""),
        width=int(js.get("width", 1024)),
        height=int(js.get("height", 1024)),
        steps=int(js.get("steps", 45)),
        guidance=float(js.get("guidance", 5.5)),
        seed=js.get("seed", None),
        transparent=bool(js.get("transparent", False)),
        image_url=js.get("image_url"),
        identity_strength=js.get("identity_strength", 0.9),
        strength=js.get("strength", 0.15),
        pose_image_url=js.get("pose_image_url"),
        pose_weight=js.get("pose_weight", 0.6),
    )

@app.route("/generate", methods=["POST"])
def generate():
    try:
        js = request.get_json(force=True, silent=False)
        params = _parse_params(js)

        if params.image_url:
            img = _edit_with_faceid_and_pose(params)
            if img is None:
                img = _txt2img(params)
        else:
            img = _txt2img(params)

        buf = io.BytesIO()
        fmt = "PNG" if params.transparent else "JPEG"
        img.save(buf, format=fmt)
        buf.seek(0)
        return send_file(buf, mimetype=f"image/{fmt.lower()}")
    except Exception as e:
        log.exception("generate falló")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": MODEL_NAME, "device": DEVICE})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
