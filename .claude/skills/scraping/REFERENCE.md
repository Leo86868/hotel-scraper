# Scraping Module – Technical Reference

## Pipeline Overview

5-stage filter pipeline + MiMo visual relevance:

```
Download (SerpAPI + Bing + DDG)
  → Stage 1: Resolution (≥600×600)
  → Stage 2: Aspect ratio (0.4–2.5)
  → Stage 3: Aesthetic score (≥6.0, LAION ViT-B/32)
  → Stage 4: Dedup (pHash + CLIP)
  → Stage 5: MiMo visual relevance (top 30 → KEEP/REJECT)
  → Top 15 by aesthetic score
```

---

## Parameter Table

### Pipeline Thresholds (config.py)

| Parameter | Value | Env Var | Description |
|-----------|-------|---------|-------------|
| `TARGET_IMAGES_MAX` | 15 | `SCRAPE_TARGET_IMAGES_MAX` | Images to keep per POI |
| `TARGET_IMAGES_MIN` | 12 | `SCRAPE_TARGET_IMAGES_MIN` | Minimum images to upload |
| `MIN_IMAGES_TO_ADVANCE` | 12 | `SCRAPE_MIN_IMAGES` | Minimum to advance to next stage |
| `SCRAPE_CANDIDATE_IMAGES` | 100 | `SCRAPE_CANDIDATE_IMAGES` | Raw download target |
| `SKIP_IF_SCRAPED_WITHIN_DAYS` | 15 | `SCRAPE_SKIP_DAYS` | Skip recently-scraped POIs |
| `USE_DYNAMIC_QUERIES` | true | `SCRAPE_DYNAMIC_QUERIES` | Use Gemini-generated search suffixes |

### Filter Thresholds

| Parameter | Value | Env Var | Description |
|-----------|-------|---------|-------------|
| `MIN_WIDTH` | 600 | `SCRAPE_MIN_WIDTH` | Minimum image width (px) |
| `MIN_HEIGHT` | 600 | `SCRAPE_MIN_HEIGHT` | Minimum image height (px) |
| `MIN_ASPECT_RATIO` | 0.4 | `SCRAPE_MIN_ASPECT` | Min width/height ratio |
| `MAX_ASPECT_RATIO` | 2.5 | `SCRAPE_MAX_ASPECT` | Max width/height ratio |
| `AESTHETIC_SCORE_MIN` | 6.0 | `SCRAPE_AESTHETIC_MIN` | LAION ViT-B/32 minimum (1-10) |
| `CLIP_SIMILARITY_THRESHOLD` | 0.89 | `SCRAPE_CLIP_DEDUP` | CLIP dedup max similarity |
| `PHASH_THRESHOLD` | 8 | `SCRAPE_PHASH_DEDUP` | pHash min Hamming distance |

### MiMo Visual Relevance

| Parameter | Value | Env Var | Description |
|-----------|-------|---------|-------------|
| `MIMO_CANDIDATE_COUNT` | 30 | `SCRAPE_MIMO_CANDIDATES` | Images sent to MiMo |
| Model | `xiaomi/mimo-v2-omni` | `UG_OMNI_MODEL` | Via OpenRouter API |
| Retry | 2 attempts, 3s delay | (hardcoded) | On API failure |

---

## Download Stage

### Three Sources (parallel)

| Source | Library | Purpose | Volume |
|--------|---------|---------|--------|
| SerpAPI | `serpapi` → Google Images | Precision — hotel's own domain | max 20 |
| Bing | `icrawler.BingImageCrawler` | Primary volume | 5 queries × ~20 |
| DuckDuckGo | `ddgs.DDGS().images()` | Opportunistic bonus | 5 queries × 20 |

### Dynamic Queries (default: on)

Gemini generates 4 search suffixes per POI, cached in `pois_cache.notes.search_suffixes`. Queries:
```
Query 1: "{poi_name}"  (always)
Query 2-5: "{poi_name} {location} {suffix}"  (Gemini-generated)
```

Gemini prompt asks for 4 broad, diverse, non-seasonal, 2-3 word suffixes.

**Fallback** (if Gemini fails or `USE_DYNAMIC_QUERIES=false`):
```
"{poi_name}"
"{poi_name} {location} resort"
"{poi_name} {location} nature view"
"{poi_name} {location} surroundings"
"{poi_name} {location} aerial view"
"{poi_name} {location} scenic drive to"
```

### Gemini Resolution (all cached in pois_cache)

| Field | Prompt | Example |
|-------|--------|---------|
| `location` | "Where is this hotel?" | "New York, New York" |
| `domain` | "Official website domain?" | "marriott.com" |
| `description` | "1-2 sentence visual description" | "A sleek modern skyscraper..." |
| `search_suffixes` | "4 broad search suffix phrases" | ["hotel interiors", "rooms view", ...] |

---

## Filter Stages

### Stage 1: Resolution Filter
- `resolution_filter.py` — PIL dimension check
- Reject if width < 600 OR height < 600
- **Kill rate:** ~15-30%

### Stage 2: Aspect Ratio Filter
- `aspect_ratio_filter.py` — PIL ratio check
- Reject if ratio outside 0.4–2.5
- **Kill rate:** ~3-5%

### Stage 3: Aesthetic Scoring (LAION)
- `aesthetic_filter.py` — LAION ViT-B/32 linear predictor
- Score range 1-10, reject if < 6.0
- Scores saved in `metadata_collector` for ranking
- **Kill rate:** ~35-45%

### Stage 4: Deduplication
- `dedup_filter.py` — two-pass
- Pass 1: pHash Hamming distance ≥ 8
- Pass 2: CLIP ViT-B/32 cosine similarity < 0.89
- **Kill rate:** ~40-55%

### Stage 5: MiMo Visual Relevance
- `visual_relevance.py` — MiMo V2 Omni via OpenRouter
- Takes top 30 by aesthetic score, sends all in single API call
- Handles: content relevance, environment matching, watermarks, people detection, quality assessment
- 2 retry attempts on API failure, fail-open if both fail
- **Kill rate:** ~15-40%

### Final Selection
- `category_balancer.py` — sort by aesthetic score descending, take top 15
- No category quotas, no caps — pure quality ranking

---

## MiMo Responsibilities (replaces YOLO, EasyOCR, people quota)

MiMo handles in a single API call what previously required 3 separate stages:

| Old Stage | Old Model | Now Handled By |
|-----------|-----------|---------------|
| Person detection | YOLOv8m (35s, 100MB) | MiMo prompt: "prominent person" + people rule |
| OCR/watermark | EasyOCR (77s, 200MB) | MiMo prompt: "watermarks, stock logos, text overlays" |
| People quota (max 2) | Post-stage logic | MiMo prompt: "PEOPLE RULE: max 2" |

**Time saved:** ~112s per POI (YOLO 35s + EasyOCR 77s)
**Memory saved:** ~300MB (YOLO 100MB + EasyOCR 200MB)

---

## Inactive Filters (kept in codebase, not in pipeline)

| File | Was | Status |
|------|-----|--------|
| `person_filter.py` | YOLO person detection | Inactive — replaced by MiMo |
| `ocr_filter.py` | EasyOCR watermark detection | Inactive — replaced by MiMo |
| `clip_scorer.py` | CLIP environment filter | Inactive — replaced by MiMo |

---

## Models Used

| Model | Purpose | Memory | Lifecycle |
|-------|---------|--------|-----------|
| CLIP ViT-B/32 | Aesthetic scoring + dedup | ~350 MB | Loaded stage 3, unloaded after stage 4 |
| LAION linear head | Aesthetic scoring | 3 KB | Unloaded after stage 3 |
| MiMo V2 Omni | Visual relevance (API) | 0 (remote) | Stage 5, via OpenRouter |

YOLO and EasyOCR are **no longer loaded** in the pipeline.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "All POIs skipped" | Recently scraped within 15 days. Use `--force`. |
| "< 12 images after filter" | POI has limited online presence. Try alternate name. |
| "MiMo filter failed" | OpenRouter API error. Retries 2x. Fail-open — pipeline continues without relevance check. |
| Missing aesthetic weights | Download `sa_0_4_vit_b_32_linear.pth` to `assets/models/`. |
| "Gemini suffix generation failed" | Falls back to hardcoded query templates. |

---

## Interface Contracts

### Input
```
python3 scraping/scripts/scrape_pipeline.py --poi "Hotel Name" [--force] [--no-upload]
```

### Local Output
```
~/AIGC_Pictures/Scraped/{POI_Name}/
├── candidate_0000.jpg
├── candidate_0001.jpg
└── ...
```

### Supabase Storage
```
pipeline-media/pois/{POI_Name}/scraped/
├── candidate_0000.jpg
├── ...
└── metadata.json  (aesthetic scores per image)
```
