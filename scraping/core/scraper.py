"""Web image search and download using Bing + SerpAPI + DuckDuckGo.

Automatically resolves POI location and domain via Gemini.
Three-source strategy:
  1. SerpAPI site: search (precision — images from hotel's own domain)
  2. Bing broad search (volume — primary source)
  3. DuckDuckGo (opportunistic bonus — rate-limited)
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import requests as http_requests

from scraping.config import SCRAPE_CANDIDATE_IMAGES, USE_DYNAMIC_QUERIES

logger = logging.getLogger(__name__)

# Queries with location (used when location is resolved)
_QUERIES_WITH_LOCATION = [
    "{poi_name}",
    "{poi_name} {location} resort",
    "{poi_name} {location} nature view",
    "{poi_name} {location} surroundings",
    "{poi_name} {location} aerial view",
    "{poi_name} {location} scenic drive to",
]

# Fallback queries (used only if Gemini fails to resolve location)
_QUERIES_FALLBACK = [
    "{poi_name}",
    "{poi_name} hotel resort",
    "{poi_name} nature view",
    "{poi_name} surroundings",
    "{poi_name} aerial view",
    "{poi_name} scenic landscape",
]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_DOWNLOAD_TIMEOUT = 15
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# DuckDuckGo delay between queries to reduce rate limiting
_DDG_QUERY_DELAY = 5


# ------------------------------------------------------------------ #
#  Location & domain resolution
# ------------------------------------------------------------------ #

def resolve_poi_location(poi_name: str) -> Optional[str]:
    """Resolve POI location via Gemini, with pois_cache DB caching.

    Returns location string (e.g. "Barnard, Vermont") or None on failure.
    Checks pois_cache first to avoid redundant API calls.
    """
    # Check cache first
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        cache = db.check_poi_in_cache(poi_name)
        if cache["exists"] and cache["record"] and cache["record"].get("location"):
            location = cache["record"]["location"]
            logger.info("Location from cache: %s → %s", poi_name, location)
            return location
    except Exception as exc:
        logger.warning("Cache lookup failed for %s: %s", poi_name, exc)
        db = None

    # Resolve via Gemini
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — cannot resolve location")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        resp = model.generate_content(
            f'Where is the luxury hotel/resort called "{poi_name}"? '
            f'Reply with ONLY the city and state/country, like: Barnard, Vermont. '
            f'No explanation. If unknown, reply: UNKNOWN'
        )
        location = resp.text.strip()
        if not location or location.upper() == "UNKNOWN":
            logger.warning("Gemini could not resolve location for %s", poi_name)
            return None

        logger.info("Location resolved: %s → %s", poi_name, location)

        # Cache the result in pois_cache
        if db:
            try:
                db.client.table("pois_cache").upsert({
                    "poi_name": poi_name,
                    "location": location,
                }, on_conflict="poi_name").execute()
            except Exception as exc:
                logger.warning("Failed to cache location for %s: %s", poi_name, exc)

        return location

    except Exception as exc:
        logger.warning("Gemini location resolution failed for %s: %s", poi_name, exc)
        return None


def resolve_poi_domain(poi_name: str) -> Optional[str]:
    """Resolve POI official website domain via Gemini, with pois_cache caching.

    Returns domain string (e.g. "aman.com") or None on failure.
    Caches in pois_cache.notes JSON field to avoid DB migration.
    """
    # Check cache first (stored in notes JSON)
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        cache = db.check_poi_in_cache(poi_name)
        if cache["exists"] and cache["record"]:
            notes = cache["record"].get("notes") or ""
            if notes:
                try:
                    notes_data = json.loads(notes) if isinstance(notes, str) else notes
                    domain = notes_data.get("domain")
                    if domain:
                        logger.info("Domain from cache: %s → %s", poi_name, domain)
                        return domain
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as exc:
        logger.warning("Domain cache lookup failed for %s: %s", poi_name, exc)
        db = None

    # Resolve via Gemini
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        resp = model.generate_content(
            f'What is the official website domain for the luxury hotel/resort "{poi_name}"? '
            f'Reply with ONLY the bare domain, like: aman.com or blackberryfarm.com. '
            f'No https://, no www., no explanation. If unknown, reply: UNKNOWN'
        )
        domain = resp.text.strip().lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
        if not domain or domain == "unknown":
            logger.warning("Gemini could not resolve domain for %s", poi_name)
            return None

        logger.info("Domain resolved: %s → %s", poi_name, domain)

        # Cache in notes JSON
        if db:
            try:
                record = db.check_poi_in_cache(poi_name)
                existing_notes = {}
                if record["exists"] and record["record"].get("notes"):
                    try:
                        existing_notes = json.loads(record["record"]["notes"]) if isinstance(record["record"]["notes"], str) else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                existing_notes["domain"] = domain
                db.client.table("pois_cache").update({
                    "notes": json.dumps(existing_notes),
                }).eq("poi_name", poi_name).execute()
            except Exception as exc:
                logger.warning("Failed to cache domain for %s: %s", poi_name, exc)

        return domain

    except Exception as exc:
        logger.warning("Gemini domain resolution failed for %s: %s", poi_name, exc)
        return None


def resolve_poi_description(poi_name: str, location: Optional[str] = None) -> Optional[str]:
    """Get a 1-2 sentence visual description of the hotel and its environment.

    Used as context for the MiMo visual relevance filter.
    Cached in pois_cache.notes JSON under 'description'.
    """
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        cache = db.check_poi_in_cache(poi_name)
        if cache["exists"] and cache["record"]:
            notes = cache["record"].get("notes") or ""
            if notes:
                try:
                    notes_data = json.loads(notes) if isinstance(notes, str) else notes
                    desc = notes_data.get("description")
                    if desc:
                        return desc
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as exc:
        logger.warning("Description cache lookup failed for %s: %s", poi_name, exc)
        db = None

    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        loc_hint = f" in {location}" if location else ""
        resp = model.generate_content(
            f'Describe what the hotel "{poi_name}"{loc_hint} looks like in 1-2 sentences. '
            f'Focus on: the building style, surrounding landscape/environment '
            f'(desert, beach, city skyline, mountains, forest, etc.), and the general vibe. '
            f'Example: "A minimalist concrete resort nestled in red sandstone canyons '
            f'in the Utah desert, surrounded by mesas and arid terrain."'
        )
        desc = resp.text.strip()
        if not desc:
            return None

        logger.info("Description resolved: %s → %s", poi_name, desc[:80])

        if db:
            try:
                record = db.check_poi_in_cache(poi_name)
                existing_notes = {}
                if record["exists"] and record["record"].get("notes"):
                    try:
                        existing_notes = json.loads(record["record"]["notes"]) if isinstance(record["record"]["notes"], str) else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                existing_notes["description"] = desc
                db.client.table("pois_cache").update({
                    "notes": json.dumps(existing_notes),
                }).eq("poi_name", poi_name).execute()
            except Exception as exc:
                logger.warning("Failed to cache description for %s: %s", poi_name, exc)

        return desc

    except Exception as exc:
        logger.warning("Gemini description failed for %s: %s", poi_name, exc)
        return None


def resolve_poi_suffixes(poi_name: str, location: Optional[str] = None) -> Optional[List[str]]:
    """Generate 4 search suffixes via Gemini, tailored to this specific POI.

    Returns list of 4 short phrases (2-4 words each) or None on failure.
    Cached in pois_cache.notes JSON under 'search_suffixes'.
    """
    # Check cache first
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        cache = db.check_poi_in_cache(poi_name)
        if cache["exists"] and cache["record"]:
            notes = cache["record"].get("notes") or ""
            if notes:
                try:
                    notes_data = json.loads(notes) if isinstance(notes, str) else notes
                    suffixes = notes_data.get("search_suffixes")
                    if suffixes and isinstance(suffixes, list) and len(suffixes) >= 4:
                        logger.info("Search suffixes from cache: %s → %s", poi_name, suffixes)
                        return suffixes[:4]
                except (json.JSONDecodeError, TypeError):
                    pass
    except Exception as exc:
        logger.warning("Suffix cache lookup failed for %s: %s", poi_name, exc)
        db = None

    # Generate via Gemini
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        loc_hint = f" in {location}" if location else ""
        resp = model.generate_content(
            f'You are sourcing image search queries for {poi_name}{loc_hint} to find photos '
            f'for a short-form travel video. Generate 4 search suffix phrases that will be '
            f'appended after the hotel name and location.\n'
            f'Rules:\n'
            f'- The 4 suffixes should return visually different types of images\n'
            f'- At most ONE suffix can be about rooms or interiors — the rest should cover outdoor areas, dining, views, pools, architecture, surroundings, etc.\n'
            f'- Use simple, natural search terms that a real traveler would type into Google Images\n'
            f'- Each suffix: 2-3 words\n'
            f'- Do NOT include seasonal or time-specific terms\n'
            f'Output as a JSON array of 4 short phrases (2-3 words each).',
            generation_config={"max_output_tokens": 200, "temperature": 0.5},
        )

        text = resp.text.strip().strip("`").replace("json\n", "")
        suffixes = json.loads(text)
        if not isinstance(suffixes, list) or len(suffixes) < 4:
            logger.warning("Gemini returned invalid suffixes for %s: %s", poi_name, suffixes)
            return None

        suffixes = [str(s).strip() for s in suffixes[:4]]
        logger.info("Search suffixes generated: %s → %s", poi_name, suffixes)

        # Cache
        if db:
            try:
                record = db.check_poi_in_cache(poi_name)
                existing_notes = {}
                if record["exists"] and record["record"].get("notes"):
                    try:
                        existing_notes = json.loads(record["record"]["notes"]) if isinstance(record["record"]["notes"], str) else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                existing_notes["search_suffixes"] = suffixes
                db.client.table("pois_cache").update({
                    "notes": json.dumps(existing_notes),
                }).eq("poi_name", poi_name).execute()
            except Exception as exc:
                logger.warning("Failed to cache suffixes for %s: %s", poi_name, exc)

        return suffixes

    except Exception as exc:
        logger.warning("Gemini suffix generation failed for %s: %s", poi_name, exc)
        return None


def _build_queries(poi_name: str, location: Optional[str] = None,
                   suffixes: Optional[List[str]] = None) -> List[str]:
    """Build search queries for a POI.

    If USE_DYNAMIC_QUERIES and suffixes are provided, builds:
      query 1: "{poi_name}"
      query 2-5: "{poi_name} {location} {suffix}" for each suffix

    Otherwise falls back to hardcoded templates.
    """
    if USE_DYNAMIC_QUERIES and suffixes and len(suffixes) >= 4:
        queries = [poi_name]
        loc = location or ""
        for suffix in suffixes[:4]:
            queries.append(f"{poi_name} {loc} {suffix}".strip())
        logger.info("Dynamic queries: %s", queries)
        return queries

    # Fallback to hardcoded templates
    if location:
        return [
            q.format(poi_name=poi_name, location=location)
            for q in _QUERIES_WITH_LOCATION
        ]
    return [q.format(poi_name=poi_name) for q in _QUERIES_FALLBACK]


# ------------------------------------------------------------------ #
#  Search sources
# ------------------------------------------------------------------ #

def _search_serpapi_site(
    domain: str, poi_name: str, location: Optional[str] = None, max_results: int = 20
) -> List[str]:
    """Search Google Images via SerpAPI with site: operator.

    Returns direct image URLs from the hotel's own domain.
    Requires SERPAPI_API_KEY env var. Skips silently if not configured.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.info("SERPAPI_API_KEY not set — skipping SerpAPI site: search")
        return []

    try:
        from serpapi import GoogleSearch
    except ImportError:
        logger.warning("google-search-results package not installed — skipping SerpAPI")
        return []

    # Build precision query: include location for chain hotels
    query_parts = [f"site:{domain}", poi_name]
    if location:
        # Extract state/region (last part after comma) for brevity
        loc_short = location.split(",")[-1].strip() if "," in location else location
        query_parts.append(loc_short)
    query = " ".join(query_parts)

    try:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        images = results.get("images_results", [])
        urls = [img["original"] for img in images if img.get("original")]
        logger.info("SerpAPI site:%s '%s': %d images", domain, query, len(urls))
        return urls

    except Exception as exc:
        logger.warning("SerpAPI search failed for '%s': %s", query, exc)
        return []


def _search_duckduckgo(query: str, max_results: int = 20) -> List[str]:
    """Search DuckDuckGo Images and return image URLs.

    Returns empty list on any failure — never blocks the pipeline.
    """
    try:
        from ddgs import DDGS
        results = DDGS(timeout=10).images(query, max_results=max_results)
        urls = [r["image"] for r in results if r.get("image")]
        logger.info("DDG '%s': %d URLs", query, len(urls))
        return urls
    except ImportError:
        logger.warning("ddgs package not installed — skipping DuckDuckGo")
        return []
    except Exception as exc:
        logger.warning("DDG search failed for '%s': %s", query, exc)
        return []


def _download_url(url: str, dest_path: str) -> bool:
    """Download a single URL to dest_path. Returns True on success."""
    try:
        resp = http_requests.get(url, timeout=_DOWNLOAD_TIMEOUT,
                                 headers=_DOWNLOAD_HEADERS, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return False
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        # Reject tiny files (likely error pages)
        if os.path.getsize(dest_path) < 5000:
            os.remove(dest_path)
            return False
        return True
    except Exception:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        return False


def _download_urls_to_dir(
    urls: List[str], dest_dir: str, prefix: str, seen_urls: set
) -> int:
    """Download a list of URLs to dest_dir, deduplicating against seen_urls.

    Returns count of successfully downloaded images.
    """
    count = 0
    for url in urls:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        ext = ".jpg"
        for e in (".png", ".webp", ".jpeg", ".jpg"):
            if e in url.lower():
                ext = e
                break
        dest = os.path.join(dest_dir, f"{prefix}_{count:04d}{ext}")
        if _download_url(url, dest):
            count += 1
    return count


# ------------------------------------------------------------------ #
#  Main entry point
# ------------------------------------------------------------------ #

def search_and_download(poi_name: str, output_dir: str, max_images: int = None,
                        location_hint: str = None) -> List[str]:
    """Search for hotel images and download them to output_dir.

    Three-source strategy:
      1. SerpAPI site: search (precision — from hotel's own domain)
      2. Bing broad search (volume — primary source)
      3. DuckDuckGo (opportunistic bonus — rate-limited)

    All sources merge into a single pool, deduplicated, then copied
    to output_dir with sequential candidate names.

    Args:
        poi_name: Hotel/resort name.
        output_dir: Directory to save downloaded images.
        max_images: Maximum total images. Defaults to config.
        location_hint: Override automatic resolution with explicit location.

    Returns:
        List of downloaded file paths.
    """
    from icrawler.builtin import BingImageCrawler

    if max_images is None:
        max_images = SCRAPE_CANDIDATE_IMAGES

    # Resolve location, domain, and dynamic suffixes
    location = location_hint or resolve_poi_location(poi_name)
    domain = resolve_poi_domain(poi_name)
    suffixes = resolve_poi_suffixes(poi_name, location) if USE_DYNAMIC_QUERIES else None

    queries = _build_queries(poi_name, location, suffixes=suffixes)
    os.makedirs(output_dir, exist_ok=True)
    per_query = max(max_images // len(queries), 10)

    logger.info("Scraping '%s' (location=%s, domain=%s) with %d queries, %d per query",
                poi_name, location or "unresolved", domain or "unknown",
                len(queries), per_query)

    all_temp_files = []
    temp_base = tempfile.mkdtemp(prefix="scrape_")

    try:
        # Run all 3 sources in parallel (they hit different servers).
        # Each source gets its own seen_urls set and temp dir.
        # Dedup across sources happens in the pHash/CLIP filter stages.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        serp_files = []
        bing_files = []
        ddg_files = []
        serp_count = 0
        bing_count = 0
        ddg_count = 0

        def _run_serpapi():
            nonlocal serp_count
            if not domain:
                return []
            serp_dir = os.path.join(temp_base, "serpapi")
            os.makedirs(serp_dir, exist_ok=True)
            seen = set()
            urls = _search_serpapi_site(domain, poi_name, location, max_results=20)
            serp_count = _download_urls_to_dir(urls, serp_dir, "serp", seen)
            return [str(f) for f in Path(serp_dir).iterdir()
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]

        def _run_bing():
            nonlocal bing_count
            files = []
            seen = set()
            for i, query in enumerate(queries):
                temp_dir = os.path.join(temp_base, f"bing_q{i}")
                os.makedirs(temp_dir, exist_ok=True)
                try:
                    crawler = BingImageCrawler(
                        storage={"root_dir": temp_dir},
                        log_level=logging.WARNING,
                    )
                    crawler.crawl(keyword=query, max_num=per_query)
                except Exception as exc:
                    logger.warning("Bing search failed for '%s': %s", query, exc)
                    continue
                for f in Path(temp_dir).iterdir():
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS:
                        files.append(str(f))
                        bing_count += 1
                logger.info("Query '%s': %d images (Bing)", query,
                            len(list(Path(temp_dir).iterdir())))
            return files

        def _run_ddg():
            nonlocal ddg_count
            ddg_dir = os.path.join(temp_base, "ddg")
            os.makedirs(ddg_dir, exist_ok=True)
            seen = set()
            for i, query in enumerate(queries):
                if i > 0:
                    time.sleep(_DDG_QUERY_DELAY)
                urls = _search_duckduckgo(query, max_results=20)
                ddg_count += _download_urls_to_dir(urls, ddg_dir, "ddg", seen)
            return [str(f) for f in Path(ddg_dir).iterdir()
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_serp = executor.submit(_run_serpapi)
            future_bing = executor.submit(_run_bing)
            future_ddg = executor.submit(_run_ddg)

            try:
                serp_files = future_serp.result()
            except Exception as exc:
                logger.warning("SerpAPI source failed: %s", exc)

            try:
                bing_files = future_bing.result()
            except Exception as exc:
                logger.warning("Bing source failed: %s", exc)

            try:
                ddg_files = future_ddg.result()
            except Exception as exc:
                logger.warning("DDG source failed: %s", exc)

        all_temp_files = serp_files + bing_files + ddg_files
        logger.info("Sources: SerpAPI=%d, Bing=%d, DDG=%d, total=%d",
                     serp_count, bing_count, ddg_count, len(all_temp_files))

        # Merge all temp files into output_dir with unique names.
        # Start numbering after existing candidates to avoid overwriting
        # prior scrape results (important for retry with force=True).
        existing = [f for f in Path(output_dir).iterdir()
                    if f.is_file() and f.name.startswith("candidate_")]
        start_idx = len(existing)
        merged = []
        for idx, src in enumerate(all_temp_files):
            ext = Path(src).suffix
            dest = os.path.join(output_dir, f"candidate_{start_idx + idx:04d}{ext}")
            shutil.copy2(src, dest)
            merged.append(dest)
        # Return ALL candidates (existing + new) for filter pipeline
        all_candidates = [str(f) for f in Path(output_dir).iterdir()
                          if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]

    finally:
        shutil.rmtree(temp_base, ignore_errors=True)

    logger.info("Downloaded %d new + %d existing = %d total for '%s'",
                len(merged), start_idx, len(all_candidates), poi_name)
    return sorted(all_candidates)
