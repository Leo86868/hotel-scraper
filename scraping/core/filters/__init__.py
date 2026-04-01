"""Filter pipeline for scraping module.

5-stage pipeline:
  1. Resolution (≥600×600)
  2. Aspect ratio (0.4–2.5)
  3. Aesthetic score (≥6.0, LAION ViT-B/32)
  4. Dedup (pHash + CLIP)
  5. MiMo visual relevance (top 30 → KEEP/REJECT, includes watermark + people checks)
Then: top 15 by aesthetic score from MiMo's KEEP list.
"""

import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_base_key(path: str) -> Optional[str]:
    """Extract zero-padded 4-digit key from a filename."""
    filename = os.path.basename(path)
    match = re.search(r'\d{4}', filename)
    return match.group(0) if match else None


def _check_memory():
    """Log warning if available memory is critically low."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 2.0:
            logger.error("Critical: only %.1fGB available — pipeline may OOM", available_gb)
        elif available_gb < 4.0:
            logger.warning("Low memory: %.1fGB available", available_gb)
        else:
            logger.info("Memory check: %.1fGB available", available_gb)
    except ImportError:
        pass


def run_filter_pipeline(image_paths: List[str], target: int = 15,
                        metadata_collector: Dict[str, dict] = None,
                        poi_name: Optional[str] = None,
                        location: Optional[str] = None,
                        description: Optional[str] = None,
                        environment: Optional[str] = None) -> List[str]:
    """Run 5-stage filter pipeline, then select top N by aesthetic score.

    Stages:
      1. Resolution (≥600×600) — PIL, instant
      2. Aspect ratio (0.4–2.5) — PIL, instant
      3. Aesthetic scoring (LAION ViT-B/32, ≥6.0) — ~10s
      4. Deduplication (pHash + CLIP) — ~7s
      5. MiMo visual relevance on top 30 (includes watermark + people checks) — ~45s
    Then: top 15 by aesthetic score from MiMo survivors.

    Person detection, OCR/watermark, and people quota are handled by MiMo.
    """
    from .resolution_filter import filter_by_resolution
    from .aspect_ratio_filter import filter_by_aspect_ratio
    from .aesthetic_filter import filter_by_aesthetic_score
    from .dedup_filter import filter_duplicates
    from .category_balancer import balance_categories
    from scraping.core.model_cache import models

    _check_memory()

    # 4 local stages — no YOLO, no EasyOCR
    stages = [
        ("Resolution", filter_by_resolution, None),
        ("Aspect Ratio", filter_by_aspect_ratio, None),
        ("Aesthetic Score", lambda paths: filter_by_aesthetic_score(
            paths, metadata_collector=metadata_collector), "aesthetic"),
        ("Deduplication", filter_duplicates, None),
    ]

    _unloaders = {
        "aesthetic": models.unload_aesthetic,
    }

    remaining = list(image_paths)
    try:
        for stage_name, stage_func, unload_key in stages:
            before = len(remaining)
            remaining = stage_func(remaining)
            rejected = before - len(remaining)
            logger.info(
                "Stage [%s]: %d/%d rejected, %d remaining",
                stage_name, rejected, before, len(remaining),
            )
            if unload_key:
                _unloaders[unload_key]()

        # Sort by aesthetic score for MiMo + final selection
        def _aes(path):
            if metadata_collector:
                key = extract_base_key(path)
                if key and key in metadata_collector:
                    return metadata_collector[key].get("aesthetic_score", 0)
            return 0
        remaining.sort(key=_aes, reverse=True)

        # MiMo: visual relevance + watermarks + people (on top 30)
        if poi_name:
            from .visual_relevance import filter_by_visual_relevance
            before = len(remaining)
            remaining = filter_by_visual_relevance(
                remaining, poi_name=poi_name,
                location=location, description=description,
                metadata_collector=metadata_collector,
            )
            logger.info(
                "Stage [MiMo]: %d/%d rejected, %d remaining",
                before - len(remaining), before, len(remaining),
            )

        # Select top N by aesthetic score
        before = len(remaining)
        remaining = balance_categories(remaining, target=target,
                                       metadata_collector=metadata_collector)
        logger.info(
            "Stage [Top-%d by Aesthetic]: selected %d/%d",
            target, len(remaining), before,
        )
    finally:
        models.unload_aesthetic()
        models.unload_clip()
        models._gc_cleanup()

    # Prune metadata_collector to only include surviving images
    if metadata_collector is not None:
        surviving_keys = set()
        for path in remaining:
            key = extract_base_key(path)
            if key:
                surviving_keys.add(key)
        for key in list(metadata_collector.keys()):
            if key not in surviving_keys:
                del metadata_collector[key]

    return remaining
