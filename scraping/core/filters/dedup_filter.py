"""Stage 7: Deduplication — reject near-duplicate images via pHash + CLIP similarity."""

import logging
from typing import List
import imagehash
from PIL import Image
from scraping.config import CLIP_SIMILARITY_THRESHOLD, DEVICE, PHASH_THRESHOLD
from scraping.core.model_cache import models

logger = logging.getLogger(__name__)


def filter_duplicates(image_paths: List[str]) -> List[str]:
    """Remove near-duplicate images using pHash then CLIP embedding similarity."""
    if len(image_paths) <= 1:
        return image_paths

    # Pass 1: pHash dedup (fast, catches exact/near-identical)
    phash_kept = _phash_dedup(image_paths)
    logger.info("pHash dedup: %d → %d", len(image_paths), len(phash_kept))

    # Pass 2: CLIP semantic dedup (catches color-graded variants)
    clip_kept = _clip_dedup(phash_kept)
    logger.info("CLIP dedup: %d → %d", len(phash_kept), len(clip_kept))

    return clip_kept


def _phash_dedup(paths: List[str]) -> List[str]:
    """Remove images with near-identical perceptual hashes."""
    kept = []
    hashes = []
    hash_paths = []  # parallel to hashes — tracks which path each hash belongs to
    for path in paths:
        try:
            h = imagehash.phash(Image.open(path))
            min_distance = None
            matched_idx = -1
            for idx, existing in enumerate(hashes):
                d = abs(h - existing)
                if min_distance is None or d < min_distance:
                    min_distance = d
                    matched_idx = idx
            if min_distance is None or min_distance >= PHASH_THRESHOLD:
                kept.append(path)
                hashes.append(h)
                hash_paths.append(path)
            else:
                logger.debug(
                    "pHash duplicate: %s (distance=%d, matched=%s)",
                    path, min_distance, hash_paths[matched_idx]
                )
        except Exception as exc:
            logger.warning("pHash failed on %s: %s — keeping", path, exc)
            kept.append(path)
    return kept


def _clip_dedup(paths: List[str]) -> List[str]:
    """Remove semantically similar images using CLIP embeddings."""
    import torch
    clip_model, preprocess, _ = models.clip
    embeddings = []
    embed_paths = []  # parallel to embeddings — only paths with successful encoding
    failed_paths = []  # paths that failed encoding — always kept

    for path in paths:
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = clip_model.encode_image(img)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())
            embed_paths.append(path)
        except Exception as exc:
            logger.warning("CLIP embed failed on %s: %s — keeping", path, exc)
            failed_paths.append(path)

    if not embeddings:
        return failed_paths

    kept = [embed_paths[0]]
    kept_embs = [embeddings[0]]

    for i in range(1, len(embeddings)):
        sims = [torch.cosine_similarity(embeddings[i], k).item() for k in kept_embs]
        max_sim = max(sims) if sims else 0
        if max_sim < CLIP_SIMILARITY_THRESHOLD:
            kept.append(embed_paths[i])
            kept_embs.append(embeddings[i])
        else:
            matched_idx = sims.index(max_sim)
            logger.debug(
                "CLIP duplicate: %s (sim=%.3f, matched=%s)",
                embed_paths[i], max_sim, kept[matched_idx]
            )

    return kept + failed_paths
