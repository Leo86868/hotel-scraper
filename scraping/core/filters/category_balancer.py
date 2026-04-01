"""Stage 8: Select top images by aesthetic score.

No category quotas, no caps. Pure quality ranking.
"""

import logging
from typing import List, Dict

from scraping.config import TARGET_IMAGES_MAX, TARGET_IMAGES_MIN

logger = logging.getLogger(__name__)


def balance_categories(image_paths: List[str], target: int = None,
                       metadata_collector: Dict[str, dict] = None) -> List[str]:
    """Select top N images by aesthetic score descending.

    Args:
        image_paths: List of local image file paths.
        target: Number of images to keep (default TARGET_IMAGES_MAX=15).
        metadata_collector: Dict with aesthetic_score per base_key (from stage 4).
    """
    from scraping.core.filters import extract_base_key

    if target is None:
        target = TARGET_IMAGES_MAX

    if len(image_paths) <= target:
        if len(image_paths) < TARGET_IMAGES_MIN:
            logger.warning(
                "Only %d images available (need %d minimum)",
                len(image_paths), TARGET_IMAGES_MIN,
            )
        return image_paths

    def _aesthetic_score(path):
        if metadata_collector:
            key = extract_base_key(path)
            if key and key in metadata_collector:
                return metadata_collector[key].get("aesthetic_score", 0)
        return 0

    sorted_paths = sorted(image_paths, key=_aesthetic_score, reverse=True)
    selected = sorted_paths[:target]

    scores = [_aesthetic_score(p) for p in selected]
    logger.info(
        "Aesthetic selection: kept %d/%d (scores: %.2f — %.2f, mean %.2f)",
        len(selected), len(image_paths),
        max(scores) if scores else 0,
        min(scores) if scores else 0,
        sum(scores) / len(scores) if scores else 0,
    )

    if len(selected) < TARGET_IMAGES_MIN:
        logger.warning(
            "Only %d images after selection (need %d minimum)",
            len(selected), TARGET_IMAGES_MIN,
        )

    return selected
