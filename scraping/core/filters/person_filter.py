"""Stage 3: Person detection filter — reject images with prominent people.

Also tags images that contain any detected person (even small/background)
in the metadata_collector for downstream people-quota enforcement.
"""

import logging
import os
from typing import Dict, List, Optional
from scraping.config import PERSON_CONFIDENCE_THRESHOLD
from scraping.core.model_cache import models

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class 0 = person
MIN_PERSON_AREA_RATIO = 0.05  # Person must cover > 5% to reject


def filter_by_person_detection(image_paths: List[str],
                               metadata_collector: Dict[str, dict] = None) -> List[str]:
    """Reject images where YOLO detects a person with high confidence.

    Also tags ALL images with has_people=True/False in metadata_collector
    for downstream people-quota enforcement. Tags are stored per-POI in
    the metadata dict, avoiding module-level state (safe for concurrent POIs).

    A person detected at ANY confidence >= threshold counts as has_people,
    regardless of area. The 5% area threshold only controls hard rejection.
    """
    from scraping.core.filters import extract_base_key

    kept = []
    yolo = models.yolo

    for path in image_paths:
        try:
            results = yolo(path, verbose=False)
            reject = False
            any_person = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == PERSON_CLASS_ID and conf >= PERSON_CONFIDENCE_THRESHOLD:
                        any_person = True
                        # Check area ratio for hard rejection
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_area = (x2 - x1) * (y2 - y1)
                        img_h, img_w = r.orig_shape
                        img_area = img_h * img_w
                        if box_area / img_area >= MIN_PERSON_AREA_RATIO:
                            reject = True
                            break
                if reject:
                    break

            # Store tag in metadata_collector (per-POI, thread-safe)
            if metadata_collector is not None:
                base_key = extract_base_key(path)
                if base_key:
                    metadata_collector.setdefault(base_key, {})["has_people"] = any_person

            if not reject:
                kept.append(path)
            else:
                logger.debug("Rejected %s: person detected (conf=%.2f)", path, conf)
        except Exception as exc:
            logger.warning("YOLO failed on %s: %s — keeping image", path, exc)
            kept.append(path)

    tagged = sum(
        1 for k, v in (metadata_collector or {}).items() if v.get("has_people")
    )
    logger.info("Person tagging: %d/%d images have people (kept %d, rejected %d)",
                tagged, len(image_paths), len(kept), len(image_paths) - len(kept))
    return kept
