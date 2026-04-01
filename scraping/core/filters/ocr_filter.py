"""Stage 4: Text/watermark detection — reject images with significant text.

Three rejection paths:
1. Stock photo keyword match (fast-path: Getty, Shutterstock, etc.)
2. Corner/margin text (position-based: any text in outer edges is a watermark)
3. Total text area exceeding 8% (heavy signage/overlay)
"""

import logging
import os
from typing import List

import numpy as np
from PIL import Image
from scraping.core.model_cache import models

logger = logging.getLogger(__name__)

TEXT_AREA_RATIO_SIGN = 0.08       # 8% total text coverage = heavy signage, reject
TEXT_AREA_RATIO_WATERMARK = 0.001  # 0.1% text in margins = watermark, reject
WATERMARK_KEYWORDS = {"getty", "shutterstock", "alamy", "dreamstime", "istock", "adobe", "123rf"}

# Margin zones (normalized 0-1): text here is likely a watermark
MARGIN_TOP = 0.15
MARGIN_BOTTOM = 0.80
MARGIN_LEFT = 0.10
MARGIN_RIGHT = 0.90


def filter_by_text_detection(image_paths: List[str]) -> List[str]:
    """Reject images with watermarks or excessive text overlays."""
    kept = []
    reader = models.easyocr_reader

    for path in image_paths:
        try:
            img = np.array(Image.open(path))
            img_h, img_w = img.shape[:2]
            img_area = img_h * img_w

            results = reader.readtext(img)
            if not results:
                kept.append(path)
                continue

            total_text_area = 0
            reject_reason = None
            basename = os.path.basename(path)

            for (bbox, text, conf) in results:
                if conf < 0.3:
                    continue

                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                text_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                total_text_area += text_area
                text_ratio = text_area / img_area if img_area > 0 else 0

                # Path 1: Stock photo keyword (fast-path)
                if any(kw in text.lower() for kw in WATERMARK_KEYWORDS):
                    reject_reason = f"stock watermark '{text.strip()}'"
                    break

                # Path 2: Corner/margin text (position-based)
                center_y = ((min(ys) + max(ys)) / 2) / img_h
                center_x = ((min(xs) + max(xs)) / 2) / img_w
                in_margin = (center_y > MARGIN_BOTTOM or center_y < MARGIN_TOP or
                             center_x < MARGIN_LEFT or center_x > MARGIN_RIGHT)

                if in_margin and text_ratio >= TEXT_AREA_RATIO_WATERMARK:
                    pos = "bottom" if center_y > 0.5 else "top"
                    if center_x < 0.3:
                        pos += "-left"
                    elif center_x > 0.7:
                        pos += "-right"
                    reject_reason = f"corner watermark '{text.strip()}' at {pos} ({text_ratio*100:.1f}%)"
                    break

            if reject_reason:
                logger.info("Rejected %s: %s", basename, reject_reason)
                continue

            # Path 3: Total text area too high
            total_ratio = total_text_area / img_area if img_area > 0 else 0
            if total_ratio > TEXT_AREA_RATIO_SIGN:
                detected = [t for (_, t, c) in results if c >= 0.3][:3]
                logger.info("Rejected %s: text coverage %.1f%% — '%s'",
                            basename, total_ratio * 100, "', '".join(detected))
                continue

            kept.append(path)
        except Exception as exc:
            logger.warning("OCR failed on %s: %s — keeping image", path, exc)
            kept.append(path)

    return kept
