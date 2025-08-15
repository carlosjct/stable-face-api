import io
import os
import math
import random
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

import torch
from flask import Flask, request, jsonify, send_file

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLRefinerPipeline,
    DPMSolverMultistepScheduler,
)

# ====== Log ======
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))
log = logging.getLogger("api")

# ====== Config ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.getenv("SDXL_MODEL", "SG161222/RealVisXL_V5.0")
REFINER_NAME = os.getenv("SDXL_REFINER", "stabilityai/stable-diffusion-xl-refiner-1.0")
USE_REFINER = os.getenv("USE_REFINER", "1") == "1"
FACEID_CKPT = os.getenv("FACEID_CKPT", "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin")

HF_HOME = os.getenv("HF_HOME", "/workspace/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)

# ====== Rembg opcional ======
HAS_REMBG = True
try:
    from rembg import remove as rembg_remove
except Exception:
    HAS_REMBG = False

# ====== FaceID opcional ======
HAS_FACEID = False
try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # type: ignore
    HAS_FACEID = True
except Exception:
    HAS_FACEID = False

_base: Optional[StableDiffusionXLPipeline] = None
_img2img: Optional[StableDiffusionXLImg2ImgPipeline] = None
_refiner: Optional[StableDiffusionXLRefinerPipeline] = None
_faceid: Optional[IPAdapterFaceID] = None

# ====== Util ======
def _seed_everything(seed: Optional[int]) -> int:
    if seed is None or seed == 0:
        seed = random.randint(1, 2**31 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return seed, generator

def _download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img

def _ensure_even(x: int) -> int:
    return int(math.floor(x / 8) * 8)

# ====== Carga de pipelines ======
@torch.inference_mode()
def get_pipes() -> Tuple[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, Optional[StableDiffusionXLRefinerPipeline]]:
    global _base, _img2img, _refiner

    if _base is None:
        log.info(f"⏳ Cargando SDXL base: {MODEL_NAME} (device={DEVICE})")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
            torch.backends.cuda.matmul.allow_tf32 = True
        _base = pipe
        log.info("✔ txt2img listo")

    if _img2img is None:
        log.info("⏳ Cargando SDXL img2img…")
        pipe = StableDiffusionXLImg2ImgPipeline(
            vae=_base.vae,
            text_encoder=_base.text_encoder,
            text_encoder_2=_base.text_encoder_2,
            tokenizer=_base.tokenizer,
            tokenizer_2=_base.tokenizer_2,
            unet=_base.unet,
            scheduler=_base.scheduler,
            feature_extractor=_base.feature_extractor,
        )
        if DEVICE == "cuda":
            pipe.to("cuda", dtype=torch.float16)
        _img2img = pipe
        log.info("✔ img2img listo")

    if USE_REFINER and _refiner is None:
        try:
            log.info(f"⏳ Cargando Refiner: {REFINER_NAME}")
            ref = StableDiffusionXLRefinerPipeline.from_pretrained(
                REFINER_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                use_safetensors=True,
            )
            if DEVICE == "cuda":
                ref.enable_model_cpu_offload()
            _refiner = ref
            log.info("✔ refiner listo")
        except Exception as e:
            log.warning(f"No pude cargar refiner: {e}")
            _refiner = None

    return _base, _img2img, _refiner

def get_faceid_adapter(base: StableDiffusionXLPipeline) -> Optional[IPAdapterFaceID]:
    global _faceid
    if not HAS_FACEID:
        return None
    if not os.path.isfile(FACEID_CKPT):
        log.warning("FACEID ckpt no encontrado; se usará img2img normal.")
        return None
    if _faceid is None:
        try:
            _faceid = IPAdapterFaceID(base, FACEID_CKPT, device=DEVICE)
            log.info("✔ IP-Adapter FaceID listo")
        except Exception as e:
            log.warning(f"No pude inicializar FaceID: {e}")
            _faceid = None
    return _faceid

# ====== Boosters de realismo ======
REALISM_POS = ", ultra realistic, RAW photo, DSLR, natural lighting, cinematic, high dynamic range, realistic skin texture, pores, subtle imperfections"
REALISM_NEG = ", cartoon, anime, illustration, painting, cgi, 3d, render, plastic skin, oversmoothed, waxy, lowres, jpeg artifacts"

# ====== Núcleos ======
@torch.inference_mode()
def run_txt2img(prompt: str, negative: str, w: int, h: int, steps: int, guidance: float, seed: Optional[int]) -> Tuple[Image.Image, int]:
    base, _, refiner = get_pipes()
    # Forzar realismo cuando NO se provee imagen de identidad
    p = (prompt or "").strip() + REALISM_POS
    n = (negative or "").strip() + REALISM_NEG

    w = _ensure_even(w); h = _ensure_even(h)
    seed, gen = _seed_everything(seed)

    img = base(
        prompt=p,
        negative_prompt=n,
        width=w, height=h,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]

    if refiner is not None:
        img = refiner(
            prompt=p, negative_prompt=n, image=img,
            guidance_scale=min(7.5, guidance + 0.5),
            num_inference_steps=max(12, steps // 2),
        ).images[0]

    return img, seed

@torch.inference_mode()
def run_faceid_edit(
    image: Image.Image,
    prompt: str,
    negative: str,
    w: int, h: int,
    steps: int,
    guidance: float,
    identity_strength: float,
    strength: float,
    seed: Optional[int],
) -> Tuple[Image.Image, int]:
    base, img2img, refiner = get_pipes()
    adapter = get_faceid_adapter(base)

    p = (prompt or "").strip() + REALISM_POS
    n = (negative or "").strip() + REALISM_NEG
    w = _ensure_even(w); h = _ensure_even(h)
    seed, gen = _seed_everything(seed)

    # Redimensionar referencia a tamaño razonable
    face_img = ImageOps.exif_transpose(image.convert("RGB"))
    face_img = face_img.resize((512, 512), Image.LANCZOS)

    if adapter is not None:
        # ===== IP-Adapter FaceID path =====
        log.info(f"➡️ FaceID | identity_strength={identity_strength:.2f}")
        # En la mayoría de forks, scale ~ [0.5 .. 1.3]. Ajusta con identity_strength.
        face_scale = max(0.6, min(1.3, 0.6 + identity_strength * 0.6))
        out = adapter.generate(
            prompt=p,
            negative_prompt=n,
            face_image=face_img,
            scale=face_scale,
            width=w, height=h,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=gen,
        )
        img = out.images[0] if hasattr(out, "images") else out
        # Refinar con un toquecito de img2img para limpiar bordes
        img = img2img(
            prompt=p, negative_prompt=n,
            image=img, strength=max(0.05, min(0.25, strength)),
            guidance_scale=guidance,
            num_inference_steps=max(18, steps),
            generator=gen,
        ).images[0]
    else:
        # ===== Fallback: img2img suave conservando identidad =====
        log.info(f"➡️ Img2Img fallback | strength={strength:.2f}")
        init = face_img.resize((w, h), Image.LANCZOS)
        img = img2img(
            prompt=p, negative_prompt=n,
            image=init,
            strength=max(0.05, min(0.20, strength)),
            guidance_scale=guidance,
            num_inference_steps=max(25, steps),
            generator=gen,
        ).images[0]

    if refiner is not None:
        img = refiner(
            prompt=p, negative_prompt=n, image=img,
            guidance_scale=min(7.5, guidance + 0.5),
            num_inference_steps=max(12, steps // 2),
        ).images[0]

    return img, seed

def to_png_bytes(img: Image.Image, transparent: bool) -> bytes:
    if transparent and HAS_REMBG:
        try:
            img = rembg_remove(img.convert("RGBA"))
        except Exception as e:
            log.warning(f"rembg fallo: {e}")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

# ====== Flask ======
app = Flask(__name__)

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

@app.post("/generate")
@torch.inference_mode()
def generate():
    data = request.get_json(force=True, silent=True) or {}

    # Inputs
    prompt = data.get("prompt", "")
    negative = data.get("negative_prompt", "")
    image_url = data.get("image_url")  # referencia (opcional)
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 7.5))
    seed = data.get("seed")
    seed = int(seed) if isinstance(seed, (int, str)) and str(seed).isdigit() else None

    transparent = bool(data.get("transparent", False))
    ret = (data.get("return") or "").lower()  # "bytes" para devolver imagen
    # Para identidad/edición
    strength = float(data.get("strength", 0.12))
    identity_strength = float(data.get("identity_strength", 0.95))

    try:
        if image_url:
            ref = _download_image(image_url)
            img, used_seed = run_faceid_edit(
                image=ref,
                prompt=prompt,
                negative=negative,
                w=width, h=height,
                steps=steps,
                guidance=guidance,
                identity_strength=identity_strength,
                strength=strength,
                seed=seed,
            )
        else:
            img, used_seed = run_txt2img(
                prompt=prompt,
                negative=negative,
                w=width, h=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
            )

        if ret == "bytes" or (request.args.get("format") == "png"):
            png = to_png_bytes(img, transparent=transparent)
            return send_file(
                io.BytesIO(png),
                mimetype="image/png",
                as_attachment=False,
                download_name="image.png",
            )
        else:
            # Por defecto, devuelve JSON con stats
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            return jsonify({"ok": True, "bytes": len(bio.getvalue()), "seed": used_seed})
    except Exception as e:
        log.exception("generate falló")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, threaded=True)
