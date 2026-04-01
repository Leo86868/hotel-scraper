"""Stage 9: Visual relevance check via MiMo V2 Omni.

Sends the top N candidate images (pre-sorted by aesthetic score) to MiMo
in a single API call. MiMo judges each image as KEEP or REJECT based on
whether it could plausibly be a photo of the target hotel/resort.

This catches what statistical filters miss: plush toys, AI paintings,
random nature photos from wrong environments, construction site photos, etc.
"""

import logging
import os
import re
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# How many images to send to MiMo (top N by aesthetic score)
MIMO_CANDIDATE_COUNT = int(os.getenv("SCRAPE_MIMO_CANDIDATES", 30))

# Max concurrent MiMo API calls across all POIs
_mimo_semaphore = threading.Semaphore(int(os.getenv("MIMO_MAX_CONCURRENT", 3)))


def filter_by_visual_relevance(
    image_paths: List[str],
    poi_name: str,
    location: Optional[str] = None,
    description: Optional[str] = None,
    metadata_collector: Optional[Dict[str, dict]] = None,
) -> List[str]:
    """Filter images using MiMo V2 Omni visual relevance check.

    Args:
        image_paths: Pre-sorted by aesthetic score descending, already filtered.
        poi_name: Hotel/resort name.
        location: Resolved location (e.g. "New York, New York").
        description: 1-2 sentence visual description of the hotel.
        metadata_collector: Dict with aesthetic_score per base_key.

    Returns:
        List of paths that MiMo judged as KEEP, in original order.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — skipping MiMo filter (all pass)")
        return image_paths

    # Take top N candidates
    candidates = image_paths[:MIMO_CANDIDATE_COUNT]
    if not candidates:
        return image_paths

    # Build the prompt
    loc_str = f" in {location}" if location else ""
    desc_str = f" This is: {description}" if description else ""

    prompt = (
        f'You are curating photos for a short-form vertical travel video about '
        f'"{poi_name}"{loc_str}.{desc_str}\n'
        f'The final video is a fast-paced, visually stunning 30-second montage '
        f'designed to make viewers want to visit this place.\n\n'

        f'== STEP 1: FILTER ==\n'
        f'For each image (numbered 1-{len(candidates)}), respond with KEEP or REJECT.\n\n'
        f'REJECT if:\n'
        f'- Not a real photograph (drawings, paintings, AI art, screenshots, maps, logos, clipart)\n'
        f'- Completely unrelated to a hotel or resort (toys, random objects, unrelated buildings)\n'
        f'- Wrong environment type (e.g. forest/nature photos for a city hotel, city skyline for a desert resort)\n'
        f'- Not aspirational or visually appealing (construction sites, parking lots, corporate offices, cluttered back-of-house)\n'
        f'- Subject is people/events rather than the place (wedding groups, staff portraits, conference attendees)\n'
        f'- Visible watermarks, photographer credits, stock logos, website URLs, or large text overlays\n'
        f'- Prominent person as main subject (face clearly visible, person dominates frame). Small/distant people OK.\n'
        f'- Chaotic motion scenes (falling confetti, swirling particles, splashing debris) — these cause AI animation artifacts\n'
        f'- Low quality (blurry, too dark, weird crops)\n\n'
        f'KEEP if it could work in a travel video that makes viewers want to visit.\n'
        f'Be lenient on hotel identity — it just needs to MATCH the environment and vibe.\n\n'

        f'Format for Step 1:\n'
        f'1. KEEP — brief reason\n'
        f'2. REJECT — brief reason\n\n'

        f'== STEP 2: RANK TOP 15 ==\n'
        f'After filtering, select the best 15 from your KEEP list and rank them.\n'
        f'Think about what makes a short-form travel video go viral — variety, wow factor, aspirational feeling.\n\n'
        f'Ranking priorities (highest to lowest):\n'
        f'1. Scenic/landscape shots — pools, ocean views, architecture, dramatic nature, golden hour\n'
        f'2. Interesting & aesthetic hotel facility shots — unique rooms, stunning lobbies, atmospheric restaurants\n'
        f'3. Shots WITHOUT people are preferred over shots with people\n\n'
        f'Category caps (enforce during selection):\n'
        f'- Food/dining close-ups: max 2\n'
        f'- People visible: max 2\n\n'

        f'Format for Step 2:\n'
        f'TOP15: 3, 7, 1, 12, 5, 18, 9, 22, 15, 28, 4, 11, 20, 6, 14\n'
        f'(comma-separated image numbers, best first)\n'
    )

    # Call MiMo via OpenRouter with base64 (exponential backoff retry)
    # Semaphore limits concurrent MiMo calls across parallel POIs
    logger.info("Waiting for MiMo semaphore (max %d concurrent)...",
                _mimo_semaphore._value if hasattr(_mimo_semaphore, '_value') else 3)
    with _mimo_semaphore:
        try:
            import time as _time
            from shared.openrouter import call_openrouter

            max_retries = 3
            backoff_delays = [5, 15, 30]  # seconds between retries
            content = ""
            response = None
            for attempt in range(1, max_retries + 1):
                logger.info("Sending %d images to MiMo (attempt %d/%d)...",
                            len(candidates), attempt, max_retries)
                try:
                    response = call_openrouter(
                        prompt=prompt,
                        images=candidates,
                    )
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        break
                    finish_reason = response.get("choices", [{}])[0].get("finish_reason", "?")
                    error_field = response.get("error", None)
                    logger.warning(
                        "MiMo returned empty response on attempt %d (finish_reason=%s, error=%s, keys=%s)",
                        attempt, finish_reason, error_field, list(response.keys()),
                    )
                except Exception as retry_exc:
                    logger.warning("MiMo attempt %d failed: %s", attempt, retry_exc)
                if attempt < max_retries:
                    delay = backoff_delays[attempt - 1]
                    logger.info("Retrying in %ds...", delay)
                    _time.sleep(delay)

            if not content or not content.strip():
                logger.error("MiMo failed after %d attempts — failing POI", max_retries)
                raise RuntimeError(f"MiMo visual relevance failed after {max_retries} attempts")
            usage = response.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            logger.info(
                "MiMo response: %d tokens (prompt=%d, completion=%d)",
                total_tokens, prompt_tokens, completion_tokens,
            )

            # Parse response: extract KEEP/REJECT + TOP15 ranking
            decisions = _parse_mimo_response(content, len(candidates))
            top15_indices = _parse_top15(content, len(candidates))
            if not top15_indices:
                logger.warning("MiMo response tail (no TOP15 found): ...%s", content[-300:])

            kept_indices = [i for i, decision in enumerate(decisions) if decision == "KEEP"]
            rejected_indices = [i for i, decision in enumerate(decisions) if decision == "REJECT"]

            logger.info(
                "MiMo visual relevance: %d KEEP, %d REJECT out of %d",
                len(kept_indices), len(rejected_indices), len(candidates),
            )

            for i in rejected_indices:
                logger.info("  MiMo REJECT: %s", os.path.basename(candidates[i]))

            # Use MiMo's TOP15 ranking if available, otherwise fall back to kept list
            if top15_indices:
                selected = [candidates[i] for i in top15_indices if i < len(candidates)]
                logger.info("MiMo TOP15 ranking: %s",
                            [os.path.basename(p) for p in selected])
                return selected
            else:
                logger.warning("MiMo did not return TOP15 ranking — falling back to aesthetic sort")
                kept_set = set(candidates[i] for i in kept_indices)
                return [p for p in image_paths if p in kept_set]

        except Exception as exc:
            logger.error("MiMo filter failed: %s — failing POI (fail-closed)", exc)
            raise


def _parse_top15(content: str, expected_count: int) -> List[int]:
    """Parse MiMo's TOP15 ranking line.

    Matches variations like:
      TOP15: 3, 7, 1, 12, ...
      **TOP15:** 3, 7, 1, ...
      Top 15: 3, 7, 1, ...
    Returns list of 0-indexed image indices in ranked order, or empty list.
    """
    for line in content.strip().split("\n"):
        cleaned = line.strip().replace("*", "").replace("#", "").strip()
        if re.match(r'^TOP\s*15\s*[:：]', cleaned, re.IGNORECASE):
            nums_part = re.split(r'[:：]', cleaned, 1)[-1].strip()
            try:
                indices = [int(n.strip()) - 1 for n in nums_part.split(",") if n.strip().isdigit()]
                valid = [i for i in indices if 0 <= i < expected_count]
                if valid:
                    return valid
            except (ValueError, IndexError):
                pass
    return []


def _parse_mimo_response(content: str, expected_count: int) -> List[str]:
    """Parse MiMo's numbered KEEP/REJECT response.

    Returns list of "KEEP" or "REJECT" strings, one per image.
    Defaults to "KEEP" for unparseable lines.
    """
    decisions = ["KEEP"] * expected_count

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match patterns like "1. KEEP — reason" or "1: REJECT - reason"
        match = re.match(r'^(\d+)[.\s:)\-]+\s*(KEEP|REJECT)', line, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # 1-indexed to 0-indexed
            decision = match.group(2).upper()
            if 0 <= idx < expected_count:
                decisions[idx] = decision

    return decisions
