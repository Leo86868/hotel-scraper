# Scraping Learnings

## What Works
- SerpAPI with `site:` precision for first pass, Bing/DuckDuckGo for volume backfill
- Aesthetic threshold 6.0 with ViT-B/32 — calibrated across 50+ POIs
- CLIP similarity threshold 0.89 — balanced between dedup accuracy and false positives
- Letting near-dupes through to enhancement stage rather than aggressive early dedup
- MiMo visual relevance filter catches plush toys, AI paintings, wrong-environment photos (2026-03-27)
- Gemini dynamic query suffixes produce hotel-specific searches vs hardcoded generic templates (2026-03-27)
- Scraping retry (clear cached suffixes + 50% more candidates) recovered W LA from 11→14 and W Atlanta from 11→15 (2026-03-27)

## What Doesn't Work
- Bing-only scraping: lower quality results, misses niche luxury properties
- Aesthetic threshold below 6.0: lets through too many low-quality images
- CLIP similarity below 0.89: false positive dedup removes legitimately different angles
- CLIP similarity above 0.93: near-dupes slip through and waste enhancement credits
- Lowering OCR threshold globally when one POI has edge-case false positives
- Generic Gemini suffix prompts produce identical suffixes across POIs ("hotel interior", "guest rooms" for every hotel) (2026-03-27)
- DDG is unreliable on remote servers — often returns 0 due to rate limiting from server IP (2026-03-27)
- Re-running scraping without clearing old scrape dirs causes 85%+ pHash dedup (duplicate files from two runs) (2026-03-27)

## Rules (Always Apply)
- Check SerpAPI usage before large batches (250 searches/month limit)
- Retry 429s from SerpAPI with backoff — they're transient
- If a POI loses good images to OCR, check rejection logs before adjusting thresholds
- OCR margin zones: top 15%, bottom 20%, left 10%, right 10% — catches watermarks but also hotel signs
- Always clear old scrape dirs before re-running a POI — stale files cause massive false dedup
- Install google-search-results package on server — SerpAPI silently skips without it (100 images lost)
- MiMo scraping filter is fail-open — if API fails, all images pass. Check logs for "MiMo failed" warnings
