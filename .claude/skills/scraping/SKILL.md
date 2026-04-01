---
name: scraping
description: >
  Scrape hotel images, filter with AI, upload to Supabase Storage + DB. Step 1 of pipeline.
  TRIGGER: "scrape", "add new POI", "get images for", hotel/resort image ingestion.
  NO TRIGGER: web research scraping, dependency downloads, API data fetching.
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, WebSearch, WebFetch
---

# Scraping Module

## What It Does

Scrape hotel images via bb-browser (3 sources), download raw candidates, SCP to VPS for filtering.

**Local Mac (this module):** bb-browser extract URLs → download raw images → tar+SCP to server → create Supabase DB record
**Remote server:** filter (CLIP/YOLO/OCR) → image-opt → video-gen → compile → download

### Three scraping sources (bb-browser mode)

| Source | What it does | Needs LLM? |
|--------|-------------|------------|
| Google KP Tabs | Search hotel → click Photos → iterate tabs (Rooms, Exterior, etc.) | No |
| Google Images | Search hotel → extract image URLs from script tags | No |
| Official Site | Search hotel → MIMO finds Website link → navigate hotel site → MIMO picks subpages → extract images | Yes (MIMO) |

**Legacy mode** (fallback when bb-browser unavailable): SerpAPI + Bing + DuckDuckGo. Toggle: `USE_BB_BROWSER=true` in .env.

## Directory Structure

```
pipeline/scraping/
├── scripts/scrape_pipeline.py  # Main CLI orchestrator (USE THIS)
├── core/
│   ├── bb_source.py            # bb-browser scraping (daemon HTTP API)
│   ├── scraper.py              # Legacy scraping (Bing + DuckDuckGo + SerpAPI)
│   ├── uploader.py             # Supabase Storage upload
│   ├── model_cache.py          # Lazy model loading with unloading
│   └── filters/
│       ├── __init__.py         # Filter pipeline orchestrator
│       ├── resolution_filter.py
│       ├── aspect_ratio_filter.py
│       ├── person_filter.py    # YOLO person detection
│       ├── aesthetic_filter.py # LAION aesthetic scoring
│       ├── clip_scorer.py      # CLIP relevance scoring
│       ├── dedup_filter.py     # pHash + CLIP deduplication
│       ├── ocr_filter.py       # EasyOCR watermark detection
│       └── category_balancer.py # Category quota balancing
├── config.py                   # All thresholds and env vars
├── tests/test_scraping.py
scripts/
└── start_bb_browser.sh          # Auto-launch Chrome + daemon + extension
```

## How to Run

### `--bb-only` mode (DEFAULT for new POIs)

This is the standard workflow. Runs on local Mac, does NOT filter locally — filtering happens on a remote server.

Flow: bb-browser extract URLs → download raw images → tar+SCP to server → create DB record.

```bash
cd "$PIPELINE_ROOT"

# Single POI (most common usage)
python3 scraping/scripts/scrape_pipeline.py --poi "Amangiri" --bb-only

# Multiple POIs
python3 scraping/scripts/scrape_pipeline.py --poi "Hotel A" "Hotel B" --bb-only
```

After SCP completes, run on the remote server to continue:
```bash
# SSH into server
ssh -p $VPS_PORT $VPS_HOST

# On server — process specific POIs
python3 run_batch.py --pois 'Amangiri' --from-stage scraping

# On server — process ALL pending POIs (idle status in DB)
python3 run_batch.py --all-pending --from-stage scraping
```

Configure `VPS_HOST`, `VPS_PORT`, and `VPS_SCRAPE_DIR` in your `.env`.

### Full local pipeline (rarely used)

Only use when the remote server is unavailable or you need filtered images locally.

```bash
# Full pipeline: scrape + filter + upload to Supabase
python3 scraping/scripts/scrape_pipeline.py --poi "Amangiri"

# With bb-browser sources instead of legacy
USE_BB_BROWSER=true python3 scraping/scripts/scrape_pipeline.py --poi "Amangiri"

# Scrape + filter locally, skip Supabase upload
python3 scraping/scripts/scrape_pipeline.py --poi "Hotel A" --no-upload

# Force re-scrape (ignore skip_days)
python3 scraping/scripts/scrape_pipeline.py --poi "Hotel Name" --force
```

## Input/Output

**Input:** POI name (location resolved automatically via Gemini)

**Output:**
- `~/AIGC_Pictures/Scraped/{POI_Name}/` — 12-15 filtered images locally
- `pois/{POI_Name}/scraped/` — images in Supabase Storage
- `optimization_layer` DB record with `scraped_count`, `scraped_pics` path

## bb-browser Mode

### CRITICAL: Use daemon HTTP, NOT CLI

bb-browser CLI (`bb-browser click @ref`) does NOT work for navigation.
The CLI connects directly to Chrome via CDP, and `Input.dispatchMouseEvent`
does not produce DOM events. Only the daemon path works:

```
Python → HTTP POST → daemon (port 19824) → SSE → Chrome Extension → chrome.debugger → Chrome
```

All scraping code MUST use `requests.post("http://localhost:19824/command", json={...})`.
Never use `subprocess.run(["bb-browser", ...])`.

### Prerequisites (auto-handled by start_bb_browser.sh)

1. Chrome running (bb-browser managed instance on port 19825)
2. Daemon running on port 19824
3. Chrome extension connected to daemon

### Three sources

| Source | Method | Quota | LLM |
|--------|--------|-------|-----|
| Google KP Tabs | snapshot → click tabs → eval extract | KP_IMAGES_PER_CATEGORY (default 20) | No |
| Google Images | eval extract from script tags | GOOGLE_IMAGES_MAX (default 50) | No |
| Official Site | snapshot → MIMO picks links → eval extract | OFFICIAL_SITE_MAX (default 50) | MIMO (OpenRouter) |

### Two-step architecture

- **Step 1 (local Mac):** `bb_extract_urls()` → JSON manifest (needs Chrome + daemon)
- **Step 2 (server or local):** `bb_download_from_manifest()` → download images via HTTP (no bb-browser needed)

### Tab management

Every source function must close its tab after finishing.
`_open()` creates a new tab, `_close()` closes it and resets `_current_tab_id`.
Failing to close tabs causes subsequent sources to fail.

## Key Parameters (config.py)

| Parameter | Default | Env var | Description |
|-----------|---------|---------|-------------|
| TARGET_IMAGES_MAX | 15 | SCRAPE_TARGET_IMAGES_MAX | Target images per POI |
| TARGET_IMAGES_MIN | 12 | SCRAPE_TARGET_IMAGES_MIN | Minimum target |
| MIN_IMAGES_TO_ADVANCE | 12 | SCRAPE_MIN_IMAGES | Min to advance to next stage |
| CLIP_SIMILARITY_THRESHOLD | 0.89 | SCRAPE_CLIP_DEDUP | Dedup similarity threshold |
| PHASH_THRESHOLD | 8 | SCRAPE_PHASH_DEDUP | pHash hamming distance |
| PERSON_CONFIDENCE_THRESHOLD | 0.35 | SCRAPE_PERSON_CONF | YOLO person confidence |
| AESTHETIC_SCORE_MIN | 6.0 | SCRAPE_AESTHETIC_MIN | LAION aesthetic minimum |
| CLIP_LUXURY_MIN | 0.20 | SCRAPE_CLIP_MIN | CLIP luxury relevance minimum |
| MAX_PEOPLE_IMAGES | 2 | SCRAPE_MAX_PEOPLE_IMAGES | Max images with detected people |
| SCRAPE_CANDIDATE_IMAGES | 100 | SCRAPE_CANDIDATE_IMAGES | Raw candidates before filtering |
| KP_IMAGES_PER_CATEGORY | 20 | KP_IMAGES_PER_CATEGORY | Max images per KP tab |
| GOOGLE_IMAGES_MAX | 50 | GOOGLE_IMAGES_MAX | Max Google Images URLs |
| OFFICIAL_SITE_MAX | 50 | OFFICIAL_SITE_MAX | Max official site URLs |
| BB_DAEMON_URL | http://127.0.0.1:19824 | BB_DAEMON_URL | bb-browser daemon endpoint |

## OCR Filter — Three Rejection Paths

`ocr_filter.py` runs EasyOCR and rejects an image via any of three paths (in order):

1. **Stock keyword match** (fast-path): text contains Getty, Shutterstock, Alamy, Dreamstime, iStock, Adobe, or 123RF — immediate reject.
2. **Corner/margin position-based detection**: any text bounding box whose center falls in the outer margins (top 15%, bottom 20%, left 10%, right 10% of the image) with area ≥ 0.1% of the image is treated as a watermark — reject. This catches hotel logos, photographer credits, and agency tags regardless of text content.
3. **Total text area threshold**: if all detected text boxes together exceed 8% of image area — reject (heavy signage/overlay).

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| <12 images | POI lacks photos; try different name or lower thresholds |
| All skipped | Recently scraped; use `--force` |
| Upload fails | Check SUPABASE_URL, SUPABASE_KEY credentials |
| KP=0 (Hotels Pack) | Brand searches like "W Hotels MIAMI" show hotel lists, not single KP. Code auto-detects and re-searches best-matching hotel name. |
| Official site=0 | Check logs for "Reload" or empty snapshot — site may block headless browsers. KP+GImg usually sufficient. |

## Architecture Notes — Official Site Domain Resolution

Official site does NOT use Gemini to guess the domain. Flow:
1. Google search hotel name → snapshot search results page
2. MIMO finds the "Website" button or official domain link in the Knowledge Panel
3. Click into official site → extract domain from `window.location.hostname`
4. Fallback to Gemini only if MIMO finds nothing

This avoids domain hallucination (e.g., Gemini returning `desertspringsresort.com` instead of `marriott.com`).

## See Also

- `REFERENCE.md` — Full technical reference
- `scraping/config.py` — All configuration values

## Known Gotchas

@gotchas/common-failures.md
