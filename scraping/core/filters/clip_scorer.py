"""Stage 5: CLIP environment filter — reject images from wrong environments.

Scores each image against multiple environment types. Rejects only when a
WRONG environment scores higher than the correct one by a margin.
Interior/neutral images (low scores on ALL environments) pass through.
"""

import logging
import os
from typing import Dict, List, Optional
import torch
from PIL import Image
from scraping.config import DEVICE
from scraping.core.model_cache import models

logger = logging.getLogger(__name__)

# Margin: wrong env must beat correct by this much to reject
ENV_MISMATCH_MARGIN = float(os.getenv("SCRAPE_ENV_MARGIN", 0.02))

# Neutral threshold: if max score across all envs < this, image is
# environment-neutral (interior shot) and passes unconditionally
ENV_NEUTRAL_THRESHOLD = float(os.getenv("SCRAPE_ENV_NEUTRAL", 0.21))

# Environment prompts — describe the SETTING, not the hotel
ENVIRONMENT_PROMPTS: Dict[str, List[str]] = {
    "desert": [
        "desert landscape canyon red rock arid terrain",
        "sandstone mesa dry wilderness southwestern scenery",
        "arid desert valley with rocky formations",
    ],
    "tropical": [
        "tropical ocean beach overwater turquoise",
        "palm trees white sand crystal clear water island",
        "lush tropical coastline with coral reef",
    ],
    "coastal": [
        "ocean coastline rocky cliffs seaside waves",
        "coastal bluff overlooking the sea sandy shore",
        "pacific coast highway ocean views rugged shore",
    ],
    "mountain": [
        "snow-capped mountain peaks alpine forest valley",
        "mountain lodge pine trees wilderness winter landscape",
        "rocky mountain range with evergreen forest",
    ],
    "forest": [
        "dense forest tall trees woodland canopy greenery",
        "wooded countryside rolling hills pastoral landscape",
        "forest clearing surrounded by mature trees",
    ],
    "city": [
        "urban city skyline skyscrapers downtown metropolitan",
        "city street buildings traffic modern architecture",
        "dense urban area with high-rise towers",
    ],
    "rural": [
        "rural farmland open fields countryside barn",
        "rolling green hills pastoral ranch landscape",
        "countryside with meadows and scattered farmhouses",
    ],
}


def filter_by_clip_relevance(image_paths: List[str],
                              environment: Optional[str] = None) -> List[str]:
    """Filter images by environment mismatch.

    If environment is provided, rejects images where a wrong environment
    scores higher than the correct one. If not provided (Gemini failed),
    skips filtering entirely (all images pass).
    """
    if not environment or environment not in ENVIRONMENT_PROMPTS:
        logger.info("No environment set — skipping environment filter (all pass)")
        return list(image_paths)

    clip_model, preprocess, tokenize = models.clip

    # Pre-encode all environment prompts
    env_features: Dict[str, torch.Tensor] = {}
    for env_name, prompts in ENVIRONMENT_PROMPTS.items():
        tokens = tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        env_features[env_name] = feat

    kept = []
    rejected_count = 0
    neutral_count = 0

    for path in image_paths:
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                scores = {}
                for env_name, feat in env_features.items():
                    scores[env_name] = (img_feat @ feat.T).max().item()

            max_score = max(scores.values())
            correct_score = scores[environment]
            best_wrong = max(s for n, s in scores.items() if n != environment)

            # Neutral: all scores low — interior shot, can't determine environment
            if max_score < ENV_NEUTRAL_THRESHOLD:
                kept.append(path)
                neutral_count += 1
                continue

            # Mismatch: wrong environment beats correct by margin
            if best_wrong - correct_score > ENV_MISMATCH_MARGIN:
                best_wrong_name = max(
                    ((n, s) for n, s in scores.items() if n != environment),
                    key=lambda x: x[1]
                )[0]
                logger.debug(
                    "Env mismatch %s: %s=%.3f < %s=%.3f (delta=%.3f)",
                    os.path.basename(path), environment, correct_score,
                    best_wrong_name, best_wrong,
                    correct_score - best_wrong,
                )
                rejected_count += 1
                continue

            kept.append(path)

        except Exception as exc:
            logger.warning("Env filter failed on %s: %s — keeping", path, exc)
            kept.append(path)

    logger.info(
        "Environment filter (%s): kept %d, rejected %d env-mismatch, %d neutral pass-through",
        environment, len(kept), rejected_count, neutral_count,
    )
    return kept
