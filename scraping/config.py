"""Scraping module configuration.

All tunable parameters loaded from environment variables with sensible defaults.
No thresholds or paths should be hardcoded anywhere else in the module.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Candidate collection (raw images before filtering)
SCRAPE_CANDIDATE_IMAGES = int(os.getenv("SCRAPE_CANDIDATE_IMAGES", 100))

# Final kept images after filtering
TARGET_IMAGES_MIN = int(os.getenv("SCRAPE_TARGET_IMAGES_MIN", 12))
TARGET_IMAGES_MAX = int(os.getenv("SCRAPE_TARGET_IMAGES_MAX", 15))

# Minimum to advance to next pipeline stage
MIN_IMAGES_TO_ADVANCE = int(os.getenv("SCRAPE_MIN_IMAGES", 12))
SKIP_IF_SCRAPED_WITHIN_DAYS = int(os.getenv("SCRAPE_SKIP_DAYS", 15))

# Legacy alias — used by filter pipeline and tests
TARGET_IMAGES = TARGET_IMAGES_MAX

# Filter thresholds
MIN_WIDTH = int(os.getenv("SCRAPE_MIN_WIDTH", 600))
MIN_HEIGHT = int(os.getenv("SCRAPE_MIN_HEIGHT", 600))
MIN_ASPECT_RATIO = float(os.getenv("SCRAPE_MIN_ASPECT", 0.4))
MAX_ASPECT_RATIO = float(os.getenv("SCRAPE_MAX_ASPECT", 2.5))
PERSON_CONFIDENCE_THRESHOLD = float(os.getenv("SCRAPE_PERSON_CONF", 0.35))
AESTHETIC_SCORE_MIN = float(os.getenv("SCRAPE_AESTHETIC_MIN", 6.0))
# Legacy — no longer used (environment CLIP filter disabled, MiMo handles relevance)
CLIP_LUXURY_MIN = float(os.getenv("SCRAPE_CLIP_MIN", 0.20))
CLIP_SIMILARITY_THRESHOLD = float(os.getenv("SCRAPE_CLIP_DEDUP", 0.89))
PHASH_THRESHOLD = int(os.getenv("SCRAPE_PHASH_DEDUP", 8))

# Legacy — people quota now handled by MiMo prompt (max 2 rule)
MAX_PEOPLE_IMAGES = int(os.getenv("SCRAPE_MAX_PEOPLE_IMAGES", 2))

# Category hard caps — only room and food are limited.
# Everything else: uncapped, selected by aesthetic score descending.
CATEGORY_CAP_ROOM = int(os.getenv("SCRAPE_CAP_ROOM", 2))
CATEGORY_CAP_FOOD = int(os.getenv("SCRAPE_CAP_FOOD", 1))

# Dynamic queries — Gemini generates search suffixes per POI
USE_DYNAMIC_QUERIES = os.getenv("SCRAPE_DYNAMIC_QUERIES", "true").lower() in ("true", "1", "yes")

# bb-browser source (replaces SerpAPI/Bing/DDG when enabled)
USE_BB_BROWSER = os.getenv("SCRAPE_USE_BB_BROWSER", "false").lower() in ("true", "1", "yes")
BB_BROWSER_PORT = int(os.getenv("BB_BROWSER_PORT", 9222))
BB_DAEMON_URL = os.getenv("BB_DAEMON_URL", "http://127.0.0.1:19824")

# bb-browser per-source quotas
KP_ALL_TAB_QUOTA = int(os.getenv("KP_ALL_TAB_QUOTA", 30))
KP_OTHER_TAB_QUOTA = int(os.getenv("KP_OTHER_TAB_QUOTA", 10))
KP_TOTAL_CAP = int(os.getenv("KP_TOTAL_CAP", 200))
# Legacy alias — kept for backward compat
KP_IMAGES_PER_CATEGORY = KP_ALL_TAB_QUOTA
GOOGLE_IMAGES_MAX = int(os.getenv("GOOGLE_IMAGES_MAX", 50))
OFFICIAL_SITE_MAX = int(os.getenv("OFFICIAL_SITE_MAX", 50))

# Device detection (MPS for M4 Mac, CUDA for GPU, CPU fallback)
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# Local storage
LOCAL_SCRAPE_DIR = os.path.expanduser(
    os.getenv("AIGC_PICTURES_PATH", "~/AIGC_Pictures")
)
TEMP_DIR = os.getenv("AIGC_TEMP_DIR", "/tmp/aigc_temp")
