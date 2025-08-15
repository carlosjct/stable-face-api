# --- deps arriba del archivo (asegúrate de tenerlos) ---
import io
from flask import send_file, request

# ...

def pil_to_png_bytes(img) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.post("/generate")
def generate():
    data = request.get_json(force=True) or {}
    prompt   = data.get("prompt", "a photo of a cat")
    negative = data.get("negative_prompt", "")
    steps    = int(data.get("steps", 25))
    guidance = float(data.get("guidance", 7.0))
    width    = int(data.get("width", 1024))
    height   = int(data.get("height", 1024))
    seed     = data.get("seed")

    # carga perezosa de pipeline y (opcional) faceid
    p  = load_pipe()
    _  = load_faceid()  # puede ser None; no detiene generación

    gen = torch.Generator(device=DEVICE)
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    # autocast para GPU si está disponible
    if DEVICE == "cuda":
        cm = torch.autocast("cuda")
    else:
        from contextlib import nullcontext
        cm = nullcontext()

    with torch.no_grad(), cm:
        out = p(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=gen,
        )
        img = out.images[0]

    # responde bytes PNG directamente
    png_bytes = pil_to_png_bytes(img)
    # filename opcional (el navegador lo muestra inline igualmente)
    return send_file(
        png_bytes,
        mimetype="image/png",
        as_attachment=False,
        download_name="result.png",
        max_age=0,  # evita cache en proxies
    )
