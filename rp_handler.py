# rp_handler.py
import os, io, time, uuid, json, tempfile, requests, torch
import runpod
from PIL import Image
from typing import Optional
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# ---------- Config ----------
MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE   = "cuda"
DTYPE    = torch.bfloat16   # recommandé pour Wan 2.2
DEFAULT_FPS = int(os.getenv("FPS", "24"))

# Cloudflare R2 (S3)
R2_ENDPOINT = os.getenv("R2_ENDPOINT")              # ex: https://<account>.r2.cloudflarestorage.com
R2_BUCKET   = os.getenv("R2_BUCKET")                # ex: x3gen
R2_KEY      = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET   = os.getenv("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE = os.getenv("R2_PUBLIC_BASE")        # ex: https://cdn.domain.com (optionnel)

# ---------- Globals ----------
pipe = None

def _log(msg: str):
    print(f"[wan22] {msg}", flush=True)

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    t0 = time.time()
    _log(f"Loading {MODEL_ID} ...")
    # VAE séparé (float32) + pipeline en bfloat16
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE)
    pipe.to(DEVICE)

    # Économie VRAM : T5 sur CPU (utile sur 24 Go)
    try:
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.to("cpu")
            _log("Text encoder moved to CPU to save VRAM.")
    except Exception as e:
        _log(f"Text encoder move skipped: {e}")

    _log(f"Loaded in {time.time()-t0:.2f}s")
    return pipe

def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def upload_r2(local_path: str, key: str) -> str:
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
    )
    extra = {"ContentType": "video/mp4"}
    s3.upload_file(local_path, R2_BUCKET, key, ExtraArgs=extra)

    if R2_PUBLIC_BASE:
        return f"{R2_PUBLIC_BASE.rstrip('/')}/{key}"
    # fallback si bucket public
    return f"{R2_ENDPOINT.rstrip('/')}/{R2_BUCKET}/{key}"

def clamp_720p(width: int, height: int) -> tuple[int, int]:
    """
    Wan 2.2 720p recommandé:
      - paysage : 1280x704
      - portrait : 704x1280
    On choisit automatiquement selon l'orientation demandée.
    """
    if height > width:
        return 704, 1280
    return 1280, 704

@torch.inference_mode()
def do_infer(inp: dict) -> dict:
    prompt  = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "Missing 'prompt'."}

    neg     = inp.get("negative_prompt") or None
    init    = inp.get("init_image")      # URL -> I2V si fourni
    width   = int(inp.get("width", 1280))
    height  = int(inp.get("height", 704))
    width, height = clamp_720p(width, height)

    frames  = int(inp.get("num_frames", 121))  # ~5s @ 24fps
    steps   = int(inp.get("steps", 50))
    scale   = float(inp.get("guidance", 5.0))
    fps     = int(inp.get("fps", DEFAULT_FPS))
    seed    = inp.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    p = load_model()

    init_image: Optional[Image.Image] = None
    if init:
        init_image = fetch_image(init)

    t1 = time.time()
    with torch.autocast(device_type="cuda", dtype=DTYPE):
        result = p(
            prompt=prompt,
            negative_prompt=neg,
            height=height, width=width,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=scale,
            image=init_image,               # None => T2V ; Image => I2V
            generator=generator,
        )
    vid = result.frames[0]
    infer_s = time.time() - t1

    # Export & upload
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "wan22.mp4")
        export_to_video(vid, out_path, fps=fps)
        key = f"wan22/{uuid.uuid4().hex}.mp4"
        url = upload_r2(out_path, key)

    return {
        "url": url,
        "meta": {
            "mode": "I2V" if init_image else "T2V",
            "width": width, "height": height,
            "frames": frames, "fps": fps,
            "steps": steps, "guidance": scale, "seed": seed,
        },
        "timings": {"infer_s": round(infer_s, 3)}
    }

def handler(job):
    try:
        _in = job.get("input") or {}
        return do_infer(_in)
    except Exception as e:
        _log(f"ERROR: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})