"""Stage 1: Resolution filter — reject images below minimum dimensions."""

import logging
from typing import List
from PIL import Image
from scraping.config import MIN_WIDTH, MIN_HEIGHT

logger = logging.getLogger(__name__)


def filter_by_resolution(image_paths: List[str]) -> List[str]:
    """Keep only images that meet minimum width and height requirements."""
    kept = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                kept.append(path)
            else:
                logger.debug("Rejected %s: %dx%d (min %dx%d)", path, w, h, MIN_WIDTH, MIN_HEIGHT)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return kept
