import io
import os
import base64
import logging
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, make_response

import torch
from PIL import Image
import requests

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# rembg (opcional) para transparencia
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# -----------------------------------------------------------------------------
# Globals / Config
# -----------------------------------------------------------------------------
MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if (DEVICE == "cuda") else torch.float32

# Pipelines cargadas bajo demanda
_TXT2IMG: Optional[StableDiffusionXLPipeline] = None
_IMG2IMG: Optional[StableDiffusionXLImg2ImgPipeline] = None
_REMBG_SESSION = None  # on-demand

# -----------------------------------------------------------------------------
# Utilidades comunes
# -----------------------------------------------------------------------------
def _enable_device(pipe):
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            log.info("✔ enable_model_cpu_offload() activado")
        except Exception:
            pipe = pipe.to(DEVICE)
            log.info("✔ .to(cuda) activado (sin offload)")
    else:
        pipe = pipe.to(DEVICE)
        log.info("✔ .to(cpu) activado")
    pipe.set_progress_bar_config(disable=False)
    return pipe

def get_txt2img() -> StableDiffusionXLPipeline:
    global _TXT2IMG
    if _TXT2IMG is not None:
        return _TXT2IMG
    log.info("⏳ Cargando SDXL TXT2IMG: %s (device=%s, dtype=%s)", MODEL_ID, DEVICE, DTYPE)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True
    )
    _TXT2IMG = _enable_device(pipe)
    log.info("✔ SDXL TXT2IMG listo")
    return _TXT2IMG

def get_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global _IMG2IMG
    if _IMG2IMG is not None:
        return _IMG2IMG
    log.info("⏳ Cargando SDXL IMG2IMG: %s (device=%s, dtype=%s)", MODEL_ID, DEVICE, DTYPE)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, use_safetensors=True
    )
    _IMG2IMG = _enable_device(pipe)
    log.info("✔ SDXL IMG2IMG listo")
    return _IMG2IMG

def _get_rembg_session():
    global _REMBG_SESSION
    if not _HAS_REMBG:
        return None
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = rembg_new_session()  # u2net por defecto
            log.info("✔ rembg session creada")
        except Exception as e:
            log.warning("⚠️ rembg no disponible: %s", e)
            _REMBG_SESSION = None
    return _REMBG_SESSION

# ---------------------- CLIP-safe prompt helpers ------------------------------
def _split_tags(text: str) -> list[str]:
    if not text:
        return []
    txt = text.replace("\n", ",").replace(";", ",")
    tags = [t.strip() for t in txt.split(",")]
    return [t for t in tags if t]

def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for t in items:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out

def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(
        text, add_special_tokens=False, truncation=False, return_tensors=None
    )["input_ids"])

def _join_until(tokens_limit: int, parts: list[str], tokenizer) -> str:
    acc = []
    for t in parts:
        candidate = (", ".join(acc + [t])) if acc else t
        if _token_len(tokenizer, candidate) <= tokens_limit:
            acc.append(t)
        else:
            break
    return ", ".join(acc)

def clip_hard_trim(pipe, prompt: str, max_tokens: int = 75) -> str:
    tags = _dedup_preserve_order(_split_tags(prompt))
    tok_a = getattr(pipe, "tokenizer", None)
    tok_b = getattr(pipe, "tokenizer_2", None)

    if tok_a is None:
        # fallback heurístico
        words, count = [], 0
        for t in tags:
            w = max(1, len(t.split()))
            if count + w > max_tokens - 2:
                break
            words.append(t); count += w
        return ", ".join(words)

    trimmed_a = _join_until(max_tokens, tags, tok_a)
    if tok_b is not None:
        safe_tags = _dedup_preserve_order(_split_tags(trimmed_a))
        trimmed_b = _join_until(max_tokens, safe_tags, tok_b)
        return trimmed_b
    return trimmed_a

def build_prompts_clip_safe(pipe, user_prompt: str, user_negative: Optional[str]) -> Tuple[str, str]:
    base_quality = [
        "ultra realistic", "photorealistic", "professional headshot",
        "studio lighting", "soft diffused light", "sharp focus on eyes",
        "natural skin texture", "centered composition", "front-facing",
        "neutral expression"
    ]
    user_tags = _split_tags(user_prompt)
    merged = _dedup_preserve_order(user_tags + base_quality)

    draft = ", ".join(merged)
    final_prompt = clip_hard_trim(pipe, draft, max_tokens=75)

    base_neg = [
        "blurry", "low quality", "cartoon", "3d render", "uncanny", "distorted face",
        "text", "watermark", "logo", "artifacts", "harsh shadows", "uneven lighting"
    ]
    neg_merged = _dedup_preserve_order(_split_tags(user_negative or "") + base_neg)
    final_negative = clip_hard_trim(pipe, ", ".join(neg_merged), max_tokens=60)

    log.info("🧠 prompt(tokens<=75): %s", final_prompt)
    log.info("🧹 negative(tokens<=60): %s", final_negative)
    return final_prompt, final_negative

# --------------------------- imagen helpers -----------------------------------
def _to_png_bytes(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

def _maybe_remove_bg(pil_image, do_transparent: bool) -> Tuple[bytes, bool]:
    """Devuelve (png_bytes, transparent_applied)"""
    if not do_transparent:
        return _to_png_bytes(pil_image), False

    sess = _get_rembg_session()
    if not sess:
        log.warning("⚠️ rembg no disponible; devolviendo imagen con fondo")
        return _to_png_bytes(pil_image), False

    try:
        raw = _to_png_bytes(pil_image)
        out = rembg_remove(raw, session=sess)
        return out, True
    except Exception as e:
        log.warning("⚠️ rembg falló: %s; devolviendo imagen con fondo", e)
        return _to_png_bytes(pil_image), False

def _wants_png() -> bool:
    # Si el cliente pide imagen: Accept: image/png o ?format=png
    if request.args.get("format", "").lower() == "png":
        return True
    acc = request.headers.get("accept", "")
    return "image/png" in acc.lower()

def _load_pil_from_input(image_url: Optional[str], image_base64: Optional[str]) -> Image.Image:
    if image_base64:
        try:
            raw = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError(f"image_base64 inválido: {e}")
    if image_url:
        try:
            r = requests.get(image_url, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"image_url inválido: {e}")
    raise ValueError("Debes enviar image_url o image_base64")

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    try:
        _ = get_txt2img()
        return jsonify(ok=True, device=DEVICE, has_rembg=_HAS_REMBG)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

# ------------------------------- /generate ------------------------------------
@app.post("/generate")
def generate():
    data = request.get_json(force=True, silent=True) or {}
    prompt_in = data.get("prompt", "") or ""
    negative_in = data.get("negative_prompt", "") or ""
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 7.0))
    transparent = data.get("transparent", False) in (True, "true", 1, "1")
    seed = data.get("seed", None)

    log.info("➡️ TXT2IMG | steps=%s guide=%s size=%sx%s transparent=%s seed=%s",
             steps, guidance, width, height, transparent, seed)

    p = get_txt2img()

    # Prompts “CLIP-safe”
    final_prompt, final_negative = build_prompts_clip_safe(p, prompt_in, negative_in)

    # Fondo neutro si no pediste transparencia
    if not transparent:
        if "plain light gray background" not in final_prompt.lower():
            final_prompt = clip_hard_trim(p, final_prompt + ", plain light gray background", 75)

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            pass

    # Inference
    images = p(
        prompt=final_prompt,
        negative_prompt=final_negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator
    ).images

    img = images[0]
    png_bytes, did_transp = _maybe_remove_bg(img, transparent)

    if _wants_png():
        resp = make_response(png_bytes)
        resp.headers["Content-Type"] = "image/png"
        resp.headers["X-Transparent"] = "1" if did_transp else "0"
        return resp

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return jsonify(
        ok=True,
        transparent_applied=bool(did_transp),
        prompt=final_prompt,
        negative_prompt=final_negative,
        image_base64=b64
    )

# --------------------------- /generate_from (NEW) -----------------------------
@app.post("/generate_from")
def generate_from():
    """
    Cambia la pose/encuadre a partir de un headshot (img2img con SDXL).
    JSON:
      - image_url o image_base64 (obligatorio uno)
      - prompt (tu prompt + instrucciones de pose)
      - negative_prompt (opcional)
      - width, height (por defecto 1024x1024)
      - steps (30), guidance (6.5–7.5), strength (0.35–0.6)
      - transparent (bool)
      - seed (opcional)
    """
    data = request.get_json(force=True, silent=True) or {}
    image_url = data.get("image_url")
    image_base64 = data.get("image_base64")
    prompt_in = data.get("prompt", "") or ""
    negative_in = data.get("negative_prompt", "") or ""
    width = int(data.get("width", 1024))
    height = int(data.get("height", 1024))
    steps = int(data.get("steps", 30))
    guidance = float(data.get("guidance", 7.0))
    strength = float(data.get("strength", 0.45))  # 0.35 cambios sutiles, 0.6 más fuertes
    transparent = data.get("transparent", False) in (True, "true", 1, "1")
    seed = data.get("seed", None)

    init_img = _load_pil_from_input(image_url, image_base64)
    init_img = init_img.convert("RGB").resize((width, height), Image.LANCZOS)

    log.info("➡️ IMG2IMG | steps=%s guide=%s strength=%.2f size=%sx%s transparent=%s seed=%s",
             steps, guidance, strength, width, height, transparent, seed)

    p = get_img2img()

    # Prompts “CLIP-safe” y algunos nudges para headshot pasaporte
    pose_nudges = ", centered headshot, shoulders visible, straight on, eyes level, neutral expression"
    final_prompt, final_negative = build_prompts_clip_safe(p, prompt_in + pose_nudges, negative_in)

    generator = None
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        except Exception:
            pass

    out = p(
        prompt=final_prompt,
        negative_prompt=final_negative,
        image=init_img,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    png_bytes, did_transp = _maybe_remove_bg(out, transparent)

    if _wants_png():
        resp = make_response(png_bytes)
        resp.headers["Content-Type"] = "image/png"
        resp.headers["X-Transparent"] = "1" if did_transp else "0"
        return resp

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return jsonify(
        ok=True,
        transparent_applied=bool(did_transp),
        prompt=final_prompt,
        negative_prompt=final_negative,
        image_base64=b64
    )

# -----------------------------------------------------------------------------
# Main (local debug)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")), debug=False)
