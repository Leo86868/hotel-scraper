"""OpenRouter multimodal API client.

Sends images/videos to OpenRouter (default: MiMo V2 Omni) and returns
the full API response. Includes FFmpeg video compression and base64 encoding.

Moved from projects/ug/core/omni_analyzer.py — shared across pipeline stages.
"""

import base64
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Config from env vars (same names/defaults as former ug.config)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OMNI_MODEL = os.getenv("UG_OMNI_MODEL", "xiaomi/mimo-v2-omni")
MAX_VIDEO_SIZE_MB = int(os.getenv("UG_MAX_VIDEO_SIZE_MB", 10))
COMPRESS_TARGET_MB = int(os.getenv("UG_COMPRESS_TARGET_MB", 9))
COMPRESS_RESOLUTION = int(os.getenv("UG_COMPRESS_RESOLUTION", 540))
DEFAULT_FPS = int(os.getenv("UG_VIDEO_FPS", 2))
TEMP_DIR = os.getenv("AIGC_TEMP_DIR", "/tmp/aigc_temp")

ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/Leo86868/hotel-scraper",
    "X-OpenRouter-Title": "hotel-scraper",
    "X-OpenRouter-Categories": "video-analysis",
}


# ---------------------------------------------------------------------------
# Token estimation (matches MiMo V2 Omni's spatial/temporal architecture)
# ---------------------------------------------------------------------------

def estimate_video_tokens(
    duration: float,
    width: int,
    height: int,
    fps: float = 2.0,
    media_resolution: str = "default",
    mute: bool = False,
) -> int:
    """Estimate input tokens for a video sent to MiMo V2 Omni."""
    PATCH = 16
    MERGE = 2
    T_PATCH = 2
    SPATIAL = PATCH * MERGE
    PIX_PER_TOKEN = SPATIAL ** 2
    MAX_TOTAL_TOKENS = 131072
    TOTAL_MAX_PIX = MAX_TOTAL_TOKENS * PIX_PER_TOKEN
    MIN_PIX = 8192
    MAX_PIX = 8388608
    MAX_FRAMES = 2048
    DEFAULT_MAX_FRAME_TOKEN = 300

    nframes = math.ceil(duration * fps)
    nframes = min(nframes, MAX_FRAMES)
    nframes = max(math.ceil(nframes / T_PATCH) * T_PATCH, T_PATCH)

    max_pix = TOTAL_MAX_PIX * T_PATCH / nframes
    if media_resolution != "max":
        max_pix = min(max_pix, DEFAULT_MAX_FRAME_TOKEN * PIX_PER_TOKEN)
    max_pix = max(MIN_PIX, min(max_pix, MAX_PIX))

    h, w = height, width
    if min(h, w) < SPATIAL:
        if h < w:
            w = int(w * SPATIAL / h)
            h = SPATIAL
        else:
            h = int(h * SPATIAL / w)
            w = SPATIAL

    h_bar = round(h / SPATIAL) * SPATIAL
    w_bar = round(w / SPATIAL) * SPATIAL

    if h_bar * w_bar > max_pix:
        beta = math.sqrt(h * w / max_pix)
        h_bar = math.floor(h / beta / SPATIAL) * SPATIAL
        w_bar = math.floor(w / beta / SPATIAL) * SPATIAL
    elif h_bar * w_bar < MIN_PIX:
        beta = math.sqrt(MIN_PIX / (h * w))
        h_bar = math.ceil(h * beta / SPATIAL) * SPATIAL
        w_bar = math.ceil(w * beta / SPATIAL) * SPATIAL

    grids = nframes // T_PATCH
    tokens_per_grid = int((h_bar / PATCH) * (w_bar / PATCH) / (MERGE ** 2))
    vision = grids * tokens_per_grid
    timestamps = grids * (5 if fps > 2 else 3)
    special = grids * 2 + 2

    audio = 0
    if not mute:
        spec_len = math.floor(duration * 24000 / 240) + 1
        t = math.floor((spec_len - 1) / 2) + 1
        t = math.floor(t / 2) + (1 if t % 2 != 0 else 0)
        audio = math.ceil(t / 4) + 2

    return round(vision + timestamps + special + audio)


# ---------------------------------------------------------------------------
# FFprobe helpers
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """Get duration, width, height from a video file via FFprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return {
        "duration": float(data["format"]["duration"]),
        "width": data["streams"][0]["width"],
        "height": data["streams"][0]["height"],
    }


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Video compression
# ---------------------------------------------------------------------------

def compress_video(
    video_path: str,
    target_mb: float = COMPRESS_TARGET_MB,
    resolution: int = COMPRESS_RESOLUTION,
) -> str:
    """Compress video to target size using FFmpeg. Returns path to compressed file."""
    temp_dir = os.path.join(TEMP_DIR, "compressed")
    os.makedirs(temp_dir, exist_ok=True)

    base_name = Path(video_path).stem
    compressed_path = os.path.join(temp_dir, f"{base_name}_compressed.mp4")

    info = get_video_info(video_path)
    duration = info["duration"]

    target_bits = target_mb * 8 * 1024 * 1024
    target_bitrate_k = int(target_bits / duration / 1000)

    logger.info(
        "Compressing: duration=%.1fs, target_bitrate=%dk, resolution=%dp",
        duration, target_bitrate_k, resolution,
    )

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale=-2:{resolution}",
        "-c:v", "libx264", "-preset", "slow",
        "-b:v", f"{target_bitrate_k}k",
        "-c:a", "aac", "-b:a", "128k",
        "-y", compressed_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    final_size = get_file_size_mb(compressed_path)
    logger.info("Compressed to %.2f MB: %s", final_size, compressed_path)
    return compressed_path


def process_video_file(video_path: str) -> tuple:
    """Check video size and compress if needed. Returns (path_to_use, was_compressed)."""
    size_mb = get_file_size_mb(video_path)
    logger.info("Video size: %.2f MB", size_mb)

    if size_mb > MAX_VIDEO_SIZE_MB:
        logger.info("Exceeds %d MB limit, compressing...", MAX_VIDEO_SIZE_MB)
        compressed = compress_video(video_path)
        return compressed, True

    return video_path, False


# ---------------------------------------------------------------------------
# File encoding
# ---------------------------------------------------------------------------

def encode_file_base64(file_path: str) -> str:
    """Read a file and return its base64-encoded content."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Max file size before compressing for API calls (2MB)
_COMPRESS_THRESHOLD = int(os.getenv("OPENROUTER_COMPRESS_THRESHOLD_KB", 2048)) * 1024
_COMPRESS_MAX_DIM = int(os.getenv("OPENROUTER_COMPRESS_MAX_DIM", 1024))
_COMPRESS_QUALITY = int(os.getenv("OPENROUTER_COMPRESS_QUALITY", 80))


def _encode_image_smart(file_path: str) -> tuple:
    """Encode image to base64, compressing large files in memory.

    Returns (base64_str, mime_type). Does NOT modify the original file.
    """
    file_size = os.path.getsize(file_path)
    if file_size <= _COMPRESS_THRESHOLD:
        return encode_file_base64(file_path), _mime_for_image(file_path)

    # Compress in memory: resize + JPEG quality reduction
    try:
        from PIL import Image as PILImage
        import io

        img = PILImage.open(file_path)
        img.thumbnail((_COMPRESS_MAX_DIM, _COMPRESS_MAX_DIM))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_COMPRESS_QUALITY)
        compressed = buf.getvalue()

        logger.info("Compressed %s for API: %dKB → %dKB",
                     os.path.basename(file_path),
                     file_size // 1024, len(compressed) // 1024)
        return base64.b64encode(compressed).decode("utf-8"), "image/jpeg"
    except Exception as exc:
        logger.warning("Compression failed for %s: %s — using original", file_path, exc)
        return encode_file_base64(file_path), _mime_for_image(file_path)


def _mime_for_image(path: str) -> str:
    """Infer MIME type from image file extension."""
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def call_openrouter(
    prompt: str,
    images: list = None,
    image_urls: list = None,
    videos: list = None,
    model: str = None,
    fps: int = None,
    api_key: str = None,
) -> dict:
    """Send a multimodal request to OpenRouter and return the full response.

    Args:
        prompt: Text prompt describing what analysis to perform.
        images: List of image file paths (will be base64 encoded).
        image_urls: List of public URLs (sent directly, no base64).
        videos: List of video file paths.
        model: OpenRouter model ID (default from env UG_OMNI_MODEL).
        fps: Video frame sampling rate, 2-10 (default from env UG_VIDEO_FPS).
        api_key: API key override (default from env OPENROUTER_API_KEY).

    Returns:
        Full API response dict with choices, usage, model fields.
    """
    api_key = api_key or OPENROUTER_API_KEY
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Export it or pass api_key parameter."
        )

    model = model or OMNI_MODEL
    fps = max(2, min(10, fps or DEFAULT_FPS))
    images = images or []
    image_urls = image_urls or []
    videos = videos or []

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **ATTRIBUTION_HEADERS,
    }

    content = [{"type": "text", "text": prompt}]

    # URL-based images (no base64, much smaller payload)
    for img_url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": img_url},
        })

    # File-based images (base64 encoded, auto-compressed if large)
    for img_path in images:
        logger.info("Processing image: %s", img_path)
        img_b64, mime = _encode_image_smart(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
        })

    compressed_paths = []
    for vid_path in videos:
        logger.info("Processing video: %s", vid_path)
        processed_path, was_compressed = process_video_file(vid_path)
        if was_compressed:
            compressed_paths.append(processed_path)

        vid_b64 = encode_file_base64(processed_path)
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{vid_b64}"},
            "fps": fps,
            "media_resolution": "default",
        })

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 10000,
    }

    logger.info(
        "Sending to OpenRouter: model=%s, fps=%d, parts=%d (text=1, image_urls=%d, images=%d, videos=%d)",
        model, fps, len(content), len(image_urls), len(images), len(videos),
    )

    response = requests.post(url, headers=headers, json=body, timeout=300)

    if response.status_code != 200:
        # Log full error response before raising
        try:
            error_body = response.text[:500]
        except Exception:
            error_body = "(could not read body)"
        logger.error(
            "OpenRouter HTTP %d: %s\nResponse body: %s",
            response.status_code, response.reason, error_body,
        )
        response.raise_for_status()

    result = response.json()

    # Log if response has error field (OpenRouter sometimes returns 200 with error)
    if result.get("error"):
        logger.error("OpenRouter returned error in 200 response: %s", result["error"])

    return result
