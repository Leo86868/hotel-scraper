# Scraping Gotchas

## SerpAPI Rate Limits
- SerpAPI has a hard limit of **250 searches/month** on the current plan. Track usage before large batches.
- 429 errors from SerpAPI are transient — retry with backoff, don't abort the POI.
- Bing and DuckDuckGo have looser limits but return lower-quality results. SerpAPI (site: precision) is preferred for the first pass.

## Aesthetic Model Threshold
- The aesthetic score threshold is **6.0** for **ViT-B/32**. It was 5.5 for L-14 — do not revert to the old value.
- The model is `openai/clip-vit-base-patch32` — using a different CLIP model changes the score distribution and requires re-calibrating the threshold.
- Do not change this without testing on 50+ images across multiple POIs.

## CLIP Similarity Threshold
- `CLIP_SIMILARITY_THRESHOLD` is **0.89**. This catches near-duplicate images from different sources.
- Setting it lower (e.g., 0.85) causes false positives — legitimately different angles of the same hotel get deduplicated.
- Setting it higher (e.g., 0.93) lets through visually similar images that waste enhancement credits.

## OCR Position-Based Filter False Positives
- The OCR margin zone filter rejects text in the outer margins: **top 15%, bottom 20%, left 10%, right 10%** of the image.
- This catches watermarks and photographer credits but also flags hotel entrance signs, room number plates, and menu boards that fall in margin zones.
- Common false positive patterns: illuminated hotel name on building facade, brass room number plates, spa menu boards near image edges.
- If a POI consistently loses good images to OCR, check the rejection logs before lowering the OCR threshold globally.

## Bing Dedup Gaps
- Bing dedup uses pHash + CLIP similarity. Near-duplicates that fall just below the CLIP threshold (0.89) slip through.
- This is acceptable — the filter pipeline catches most duplicates. Aggressive dedup causes more harm (losing good images) than letting a few near-dupes through to enhancement.
