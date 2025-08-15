import io, os, base64, logging, requests
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response, make_response

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# ---------- IP-Adapter FaceID (si está vendor en tu repo) ----------
_HAS_FACEID = False
try:
    from IP_Adapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceID  # alt nombre
    _HAS_FACEID = True
except Exception:
    try:
        from IP-Adapter.ip_adapter.ip_adapter_faceid import IPAdapterFaceID
        _HAS_FACEID = True
    except Exception:
        try:
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
            _HAS_FACEID = True
        except Exception:
            _HAS_FACEID = False

# ---------- rembg (opcional p/ transparencia) ----------
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ---------- config ----------
MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# intenta detectar el .bin del FaceID
def _find_faceid_bin() -> Optional[str]:
    env = os.getenv("IPADAPTER_FACEID_BIN")
    if env and os.path.isfile(env):
        return env
    candidates = [
        "/workspace/models/ipadapter/ip-adapter-faceid_sdxl.bin",
        "/srv/app/models/ipadapter/ip-adapter-faceid_sdxl.bin",
        "/workspace/ip-adapter-faceid_sdxl.bin",
        "/srv/app/ip-adapter-faceid_sdxl.bin",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

FACEID_BIN = _find_faceid_bin()
if _HAS_FACEID and FACEID_BIN:
    log.info("✔ IP-Adapter FaceID disponible: %s", FACEID_BIN)
elif _HAS_FACEID:
    log.warning("⚠️ IP-Adapter FaceID detectado, pero .bin NO encontrado")
else:
    log.warning("⚠️ IP-Adapter FaceID no importable; usaré fallback img2img si se requiere identidad")

# ---------- globals ----------
_TXT2IMG: Optional[StableDiffusionXLPipeline] = None
_IMG2IMG: Optional[StableDiffusionXLImg2ImgPipeline] = None
_REMBG_SESSION = None
_FACEID: Optional[IPAdapterFaceID] = None

# ---------- helpers ----------
def _ensure_rembg():
    global _REMBG_SESSION
    if not _HAS_REMBG:
        return None
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = rembg_new_session()
            log.info("✔ rembg listo")
        except Exception as e:
            log.warning("rembg no disponible: %s", e)
            _REMBG_SESSION = None
    return _REMBG_SESSION

def _to_png_bytes(pil_img: Image.Image) -> bytes:
    b = io.BytesIO()
    pil_img.save(b, format="PNG")
    return b.getvalue()

def _maybe_remove_bg(pil_img: Image.Image, want_transparent: bool) -> Tuple[bytes, bool]:
    if not want_transparent:
        return _to_png_bytes(pil_img), False
    sess = _ensure_rembg()
    if not sess:
        return _to_png_bytes(pil_img), False
    try:
        raw = _to_png_bytes(pil_img)
        out = rembg_remove(raw, session=sess)
        return out, True
    except Exception as e:
        log.warning("rembg falló: %s", e)
        return _to_png_bytes(pil_img), False

def _wants_png() -> bool:
    if request.args.get("format", "").lower() == "png":
        return True
    return "image/png" in (request.headers.get("accept","").lower())

def _split_tags(s: str) -> list[str]:
    if not s: return []
    s = s.replace("\n", ",").replace(";", ",")
    return [t.strip() for t in s.split(",") if t.strip()]

def _dedup(seq: list[str]) -> list[str]:
    seen=set(); out=[]
    for t in seq:
        k=t.lower()
        if k not in seen:
            seen.add(k); out.append(t)
    return out

def _clip_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])

def _join_until(limit: int, parts: list[str], tok) -> str:
    acc=[]
    for t in parts:
        cand = (", ".join(acc+[t])) if acc else t
        if _clip_count(tok, cand) <= limit:
            acc.append(t)
        else:
            break
    return ", ".join(acc)

def clip_safe(pipe, prompt: str, max_tokens=75) -> str:
    tags = _dedup(_split_tags(prompt))
    t1 = getattr(pipe, "tokenizer", None)
    t2 = getattr(pipe, "tokenizer_2", None)
    if not t1:
        # heurístico
        words=[]; n=0
        for t in tags:
            w=max(1,len(t.split()))
            if n+w>max_tokens-2: break
            words.append(t); n+=w
        return ", ".join(words)
    s = _join_until(max_tokens, tags, t1)
    if t2:
        s = _join_until(max_tokens, _split_tags(s), t2)
    return s

def build_prompts(pipe, ptxt: str, ng: Optional[str]):
    base = [
        "ultra realistic", "photorealistic", "studio headshot",
        "front-facing", "neutral expression", "sharp focus on eyes",
        "natural skin texture", "soft diffused lighting"
    ]
    pos = _dedup(_split_tags(ptxt) + base)
    pos = clip_safe(pipe, ", ".join(pos), 75)

    base_neg = [
        "blurry", "low quality", "cartoon", "3d render", "uncanny", "distorted face",
        "text", "watermark", "logo", "harsh shadows", "over-smoothed skin"
    ]
    neg = _dedup(_split_tags(ng or "") + base_neg)
    neg = clip_safe(pipe, ", ".join(neg), 60)
    return pos, neg

def _download_image(url_or_b64: str) -> Image.Image:
    if url_or_b64.startswith("http"):
        r = requests.get(url_or_b64, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        raw = base64.b64decode(url_or_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

# ---------- pipelines ----------
def get_txt2img() -> StableDiffusionXLPipeline:
    global _TXT2IMG
    if _TXT2IMG: return _TXT2IMG
    log.info("⏳ Cargando SDXL txt2img: %s", MODEL_ID)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, use_safetensors=True)
    if DEVICE=="cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)
    else:
        pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=False)
    _TXT2IMG = pipe
    return _TXT2IMG

def get_img2img() -> StableDiffusionXLImg2ImgPipeline:
    global _IMG2IMG
    if _IMG2IMG: return _IMG2IMG
    log.info("⏳ Cargando SDXL img2img: %s", MODEL_ID)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE, use_safetensors=True)
    if DEVICE=="cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to(DEVICE)
    else:
        pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=False)
    _IMG2IMG = pipe
    return _IMG2IMG

def get_faceid() -> Optional[IPAdapterFaceID]:
    global _FACEID
    if _FACEID is not None:
        return _FACEID
    if not (_HAS_FACEID and FACEID_BIN):
        return None
    # FaceID se monta sobre la pipeline txt2img
    p = get_txt2img()
    try:
        _FACEID = IPAdapterFaceID(p, FACEID_BIN, device=DEVICE)
        log.info("✔ IP-Adapter FaceID inicializado")
        return _FACEID
    except Exception as e:
        log.warning("No pude inicializar FaceID: %s", e)
        return None

# ---------- Flask ----------
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(
        ok=True,
        device=DEVICE,
        has_rembg=_HAS_REMBG,
        has_faceid=bool(_HAS_FACEID),
        faceid_bin=bool(FACEID_BIN)
    )

@app.post("/generate")
def generate():
    data = request.get_json(force=True, silent=True) or {}
    prompt_in = data.get("prompt","") or ""
    negative_in = data.get("negative_prompt","") or ""
    w = int(data.get("width",1024)); h = int(data.get("height",1024))
    steps = int(data.get("steps",30)); guidance = float(data.get("guidance",7.0))
    transparent = data.get("transparent", False) in (True,"true",1,"1")
    seed = data.get("seed", None)

    log.info("➡️ /generate w=%s h=%s steps=%s gs=%.2f transp=%s", w,h,steps,guidance,transparent)
    p = get_txt2img()
    pos, neg = build_prompts(p, prompt_in, negative_in)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(int(seed))

    out = p(
        prompt=pos,
        negative_prompt=neg,
        width=w,
        height=h,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen
    ).images[0]

    png, did = _maybe_remove_bg(out, transparent)
    if _wants_png():
        r = make_response(png); r.headers["Content-Type"]="image/png"; r.headers["X-Transparent"]="1" if did else "0"
        return r
    return jsonify(ok=True, prompt=pos, negative_prompt=neg, transparent_applied=bool(did),
                   image_base64=base64.b64encode(png).decode("ascii"))

@app.post("/face")
def face():
    """
    Genera manteniendo identidad del headshot:
    - FaceID (si el .bin está disponible) → mejor matching y permite cambiar pose por prompt.
    - Fallback: img2img con denoise bajo usando el headshot como init.
    Body JSON:
    {
      "image_url": "http://... o base64",
      "prompt": "... (e.g., thinking, analyzing, business casual, 3/4 torso, looking slightly left)",
      "negative_prompt": "...",
      "width": 1024, "height": 1024,
      "steps": 30, "guidance": 7.0,
      "transparent": true,
      "seed": 42,
      "adapter_strength": 0.9,   # (FaceID) ~0.7-1.2
      "use_init": false,         # si además quieres img2img
      "strength": 0.25           # denoise de img2img si use_init=true
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    src = data.get("image_url") or data.get("image_base64")
    if not src:
        return jsonify(ok=False, error="image_url o image_base64 requerido"), 400

    prompt_in = data.get("prompt","") or ""
    negative_in = data.get("negative_prompt","") or ""
    w = int(data.get("width",1024)); h = int(data.get("height",1024))
    steps = int(data.get("steps",30)); guidance = float(data.get("guidance",7.0))
    transparent = data.get("transparent", False) in (True,"true",1,"1")
    seed = data.get("seed", None)

    adapter_strength = float(data.get("adapter_strength", 0.9))
    use_init = data.get("use_init", False) in (True,"true",1,"1")
    strength = float(data.get("strength", 0.25))

    ref = _download_image(src)
    log.info("➡️ /face steps=%s gs=%.2f faceid=%s init=%s",
             steps, guidance, bool(_HAS_FACEID and FACEID_BIN), use_init)

    # Preferir FaceID si está disponible
    faceid = get_faceid()
    if faceid is not None:
        pipe = get_txt2img()
        pos, neg = build_prompts(pipe, prompt_in, negative_in)

        # Embeddings de la cara
        try:
            image_embeds, uncond = faceid.get_image_embeds(ref)
        except Exception as e:
            log.warning("get_image_embeds falló (%s); haré fallback img2img", e)
            faceid = None
        else:
            gen=None
            if seed is not None:
                gen = torch.Generator(device=DEVICE).manual_seed(int(seed))

            # Opción: también usar init image (ayuda a pelo/tono); si no, puro text2img con FaceID
            if use_init:
                img2img = get_img2img()
                pos2, neg2 = build_prompts(img2img, pos, neg)  # igual, recorte seguro
                out = img2img(
                    prompt=pos2,
                    negative_prompt=neg2,
                    image=ref.resize((w,h), Image.LANCZOS),
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    image_embeds=image_embeds,
                    negative_image_embeds=uncond,
                    ip_adapter_scale=adapter_strength,
                    generator=gen
                ).images[0]
            else:
                out = pipe(
                    prompt=pos,
                    negative_prompt=neg,
                    width=w, height=h,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    image_embeds=image_embeds,
                    negative_image_embeds=uncond,
                    ip_adapter_scale=adapter_strength,
                    generator=gen
                ).images[0]

            png, did = _maybe_remove_bg(out, transparent)
            if _wants_png():
                r = make_response(png); r.headers["Content-Type"]="image/png"; r.headers["X-Transparent"]="1" if did else "0"
                return r
            return jsonify(ok=True, used_faceid=True, transparent_applied=bool(did),
                           image_base64=base64.b64encode(png).decode("ascii"))

    # -------- Fallback: img2img sin FaceID (preserva bastante; menos fiel) --------
    img2img = get_img2img()
    pos, neg = build_prompts(img2img, prompt_in, negative_in)
    gen=None
    if seed is not None:
        gen = torch.Generator(device=DEVICE).manual_seed(int(seed))

    ref_resized = ref.resize((w,h), Image.LANCZOS)
    out = img2img(
        prompt=pos, negative_prompt=neg,
        image=ref_resized,
        strength=strength,                # <= 0.25 para fidelidad facial
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen
    ).images[0]

    png, did = _maybe_remove_bg(out, transparent)
    if _wants_png():
        r = make_response(png); r.headers["Content-Type"]="image/png"; r.headers["X-Transparent"]="1" if did else "0"
        return r
    return jsonify(ok=True, used_faceid=False, transparent_applied=bool(did),
                   image_base64=base64.b64encode(png).decode("ascii"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","3000")), debug=False)
