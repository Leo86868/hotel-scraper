"""Stage 2: Aspect ratio filter — reject images outside valid ratio range."""

import logging
from typing import List
from PIL import Image
from scraping.config import MIN_ASPECT_RATIO, MAX_ASPECT_RATIO

logger = logging.getLogger(__name__)


def filter_by_aspect_ratio(image_paths: List[str]) -> List[str]:
    """Keep images with width/height ratio between MIN and MAX."""
    kept = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            ratio = w / h if h > 0 else 0
            if MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
                kept.append(path)
            else:
                logger.debug("Rejected %s: ratio=%.2f (range %.1f-%.1f)", path, ratio, MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return kept
