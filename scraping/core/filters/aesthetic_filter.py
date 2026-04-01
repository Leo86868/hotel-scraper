"""Stage 4: Aesthetic scoring — reject low-quality images using LAION predictor."""

import logging
from typing import Dict, List, Optional
import torch
from PIL import Image
from scraping.config import AESTHETIC_SCORE_MIN, DEVICE
from scraping.core.model_cache import models

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def filter_by_aesthetic_score(image_paths: List[str],
                              metadata_collector: Dict[str, dict] = None) -> List[str]:
    """Keep images with LAION aesthetic score >= threshold.

    Args:
        image_paths: List of local image file paths.
        metadata_collector: If provided, saves aesthetic_score per base_key.
    """
    from scraping.core.filters import extract_base_key

    aesthetic_model, clip_model, preprocess = models.aesthetic

    # Score all images in batches
    all_scores = {}
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        tensors = []
        valid_paths = []
        for path in batch_paths:
            try:
                tensors.append(preprocess(Image.open(path).convert("RGB")))
                valid_paths.append(path)
            except Exception as exc:
                logger.warning("Aesthetic scoring failed on %s: %s — keeping image", path, exc)
                all_scores[path] = AESTHETIC_SCORE_MIN  # keep on error
        if not tensors:
            continue
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            features = clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            scores = aesthetic_model(features.float()).squeeze(-1)
        for path, score in zip(valid_paths, scores.cpu().tolist()):
            all_scores[path] = score

    # Filter and collect metadata
    kept = []
    for path in image_paths:
        score = all_scores.get(path, 0.0)
        if score >= AESTHETIC_SCORE_MIN:
            kept.append(path)
            if metadata_collector is not None:
                base_key = extract_base_key(path)
                if base_key:
                    metadata_collector.setdefault(base_key, {})["aesthetic_score"] = round(float(score), 2)
        else:
            logger.debug("Rejected %s: aesthetic=%.2f (min=%.1f)", path, score, AESTHETIC_SCORE_MIN)

    return kept
