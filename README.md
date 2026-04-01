# Hotel Image Scraper

Automated pipeline that scrapes, filters, and curates high-quality hotel/resort images from the web. Designed for content pipelines that need diverse, aesthetic, watermark-free hotel imagery.

## What it does

**Input:** Hotel name (e.g., "Aman Tokyo")

**Output:** 12-15 curated images uploaded to Supabase Storage, categorized by scene type (scenic, pool, room, exterior, food).

### Pipeline stages

1. **Search** — Queries Google Images (SerpAPI), Bing (icrawler), and DuckDuckGo for candidate images. Optionally uses Gemini to generate smarter search queries.
2. **Download** — Fetches ~100 candidate images to a temp directory.
3. **Filter** (5-stage):
   - Resolution filter (min 600x600)
   - Aspect ratio filter (0.4 - 2.5)
   - Aesthetic scorer (LAION CLIP-based, threshold 6.0)
   - Dedup (perceptual hash + CLIP embedding similarity)
   - Visual relevance (MiMo V2 via OpenRouter — rejects watermarks, people, irrelevant content)
4. **Category balance** — Selects top-N across scene categories (scenic, pool, room, etc.)
5. **Upload** — Uploads final images to Supabase Storage and registers metadata in the database.

## Setup

```bash
# Clone
git clone https://github.com/Leo86868/hotel-scraper.git
cd hotel-scraper

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Required API keys

| Key | Service | Purpose |
|-----|---------|---------|
| `SUPABASE_URL` + `SUPABASE_KEY` | Supabase | Image storage + metadata DB |
| `OPENROUTER_API_KEY` | OpenRouter | MiMo visual relevance filter |
| `GEMINI_API_KEY` | Google AI | Location resolution, dynamic queries |

### Optional

| Key | Service | Purpose |
|-----|---------|---------|
| `SERPAPI_API_KEY` | SerpAPI | Google Images search (skipped if absent) |

## Usage

```bash
# Scrape a single hotel
python -m scraping.scripts.scrape_pipeline --poi "Aman Tokyo"

# Force re-scrape (ignore skip-days)
python -m scraping.scripts.scrape_pipeline --poi "Aman Tokyo" --force

# Multiple hotels
python -m scraping.scripts.scrape_pipeline --poi "Aman Tokyo"
python -m scraping.scripts.scrape_pipeline --poi "Four Seasons Bora Bora"
```

## Configuration

All tunable parameters have sensible defaults. Override via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPE_CANDIDATE_IMAGES` | 100 | Max candidate images to download |
| `SCRAPE_TARGET_IMAGES_MIN` | 12 | Minimum images to keep |
| `SCRAPE_TARGET_IMAGES_MAX` | 15 | Maximum images to keep |
| `SCRAPE_MIN_IMAGES` | 12 | Minimum to consider scrape successful |
| `SCRAPE_SKIP_DAYS` | 15 | Days before re-scraping same POI |
| `SCRAPE_AESTHETIC_MIN` | 6.0 | LAION aesthetic score threshold |
| `SCRAPE_DYNAMIC_QUERIES` | true | Use Gemini for smarter search queries |
| `MIMO_MAX_CONCURRENT` | 4 | Parallel MiMo filter threads |

## Project structure

```
hotel-scraper/
├── scraping/
│   ├── config.py                  # All configuration constants
│   ├── scripts/
│   │   └── scrape_pipeline.py     # Entry point
│   └── core/
│       ├── scraper.py             # Search + download
│       ├── uploader.py            # Supabase upload
│       ├── model_cache.py         # CLIP/aesthetic model singleton
│       ├── bb_source.py           # Browser-based scraping (optional)
│       ├── kp_extractor.py        # Google Knowledge Panel extraction
│       └── filters/               # 5-stage filter pipeline
├── shared/
│   ├── supabase_client.py         # Database operations
│   ├── supabase_storage_client.py # Storage operations
│   ├── openrouter.py              # MiMo API client
│   ├── poi_utils.py               # POI name utilities
│   └── logging_config.py          # Logging setup
├── assets/models/                 # Pre-trained model weights
├── requirements.txt
└── .env.example
```
