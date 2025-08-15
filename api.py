import io
import os
import base64
import logging
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, make_response

import torch
from diffusers import StableDiffusionXLPipeline

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

_PIPE: Optional[StableDiffusionXLPipeline] = None
_REMBG_SESSION = None  # se crea on-demand

# -----------------------------------------------------------------------------
# SDXL Loader
# -----------------------------------------------------------------------------
def get_pipe() -> StableDiffusionXLPipeline:
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    log.info("⏳ Cargando SDXL: %s (device=%s, dtype=%s)", MODEL_ID, DEVICE, DTYPE)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )

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
    _PIPE = pipe
    log.info("✔ SDXL listo")
    return _PIPE

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

# -----------------------------------------------------------------------------
# CLIP-safe prompt utilities (hard cap con tokenizadores reales de SDXL)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Flask App
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.get("/health")
def health():
    try:
        p = get_pipe()
        _ = getattr(p, "tokenizer", None) is not None
        return jsonify(ok=True, device=DEVICE, has_rembg=_HAS_REMBG)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

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

    log.info("➡️ Generando | steps=%s guide=%s size=%sx%s transparent=%s seed=%s",
             steps, guidance, width, height, transparent, seed)

    p = get_pipe()

    # Construye prompts seguros para CLIP
    final_prompt, final_negative = build_prompts_clip_safe(p, prompt_in, negative_in)

    # Si NO quieres transparencia, un fondo neutro ayuda a consistencia
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

    # Si el cliente quiere PNG binario, lo devolvemos como image/png
    if _wants_png():
        resp = make_response(png_bytes)
        resp.headers["Content-Type"] = "image/png"
        resp.headers["X-Transparent"] = "1" if did_transp else "0"
        return resp

    # Si no, base64 en JSON
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
