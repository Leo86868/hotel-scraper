"""Scraping pipeline CLI entry point.

Usage:
    python3 scraping/scripts/scrape_pipeline.py --poi "Hotel Name"
    python3 scraping/scripts/scrape_pipeline.py --poi "Hotel A" "Hotel B" --no-upload
    python3 scraping/scripts/scrape_pipeline.py --poi "Hotel" --force
    python3 scraping/scripts/scrape_pipeline.py --poi "Hotel" --bb-only
"""

import argparse
import logging
import subprocess
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add pipeline root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scraping.config import (
    TARGET_IMAGES, TARGET_IMAGES_MIN, MIN_IMAGES_TO_ADVANCE,
    SKIP_IF_SCRAPED_WITHIN_DAYS, LOCAL_SCRAPE_DIR, USE_BB_BROWSER,
)
from scraping.core.scraper import search_and_download, resolve_poi_location
from scraping.core.filters import run_filter_pipeline
from scraping.core.uploader import upload_and_register
from shared.poi_utils import display_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def should_skip(poi_name: str, force: bool) -> bool:
    """Check if POI was recently scraped and should be skipped."""
    if force:
        return False
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        record = db.get_poi_by_name(poi_name)
        if record and record.get("scraped_count", 0) >= MIN_IMAGES_TO_ADVANCE:
            updated = record.get("updated_at")
            if updated:
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                cutoff = datetime.now(timezone.utc) - timedelta(days=SKIP_IF_SCRAPED_WITHIN_DAYS)
                if updated_dt > cutoff:
                    logger.info(
                        "Skipping %s — already scraped %d images within %d days",
                        poi_name, record["scraped_count"], SKIP_IF_SCRAPED_WITHIN_DAYS,
                    )
                    return True
    except Exception as exc:
        logger.warning("Could not check Supabase for %s: %s", poi_name, exc)
    return False


def process_poi(poi_name: str, no_upload: bool, force: bool, run_id: str = None,
                max_images: int = None) -> bool:
    """Process a single POI through the full scraping pipeline."""
    logger.info("=" * 60)
    logger.info("Processing POI: %s", poi_name)
    logger.info("=" * 60)

    # Strip version suffix for searches: 'Madonna Inn v2' searches as 'Madonna Inn'
    query_name = display_name(poi_name)
    if query_name != poi_name:
        logger.info("Search name: '%s' (versioned storage: '%s')", query_name, poi_name)

    # Step 1: Check if we should skip
    if should_skip(poi_name, force):
        return True

    # Step 2: Search and download images
    local_dir = Path(LOCAL_SCRAPE_DIR) / "Scraped" / poi_name
    logger.info("Step 1/3: Searching and downloading images to %s...", local_dir)

    # Check for pre-existing images (e.g. SCP'd from bb-browser on local machine)
    existing_images = []
    if local_dir.exists():
        existing_images = [
            str(p) for p in local_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") and p.stat().st_size > 1000
        ]

    if existing_images and len(existing_images) >= MIN_IMAGES_TO_ADVANCE:
        logger.info("Found %d pre-existing images in %s — skipping search",
                     len(existing_images), local_dir)
        downloaded = existing_images
    elif USE_BB_BROWSER:
        from scraping.core.bb_source import bb_extract_urls, bb_download_from_manifest
        manifest_path = str(local_dir / "bb_urls.json")
        try:
            bb_extract_urls(query_name, manifest_path)
            downloaded = bb_download_from_manifest(manifest_path, str(local_dir),
                                                   max_images=max_images)
        except (ConnectionError, RuntimeError) as exc:
            logger.error("bb-browser failed for %s: %s — falling back to legacy sources", poi_name, exc)
            downloaded = search_and_download(query_name, str(local_dir), max_images=max_images)
    else:
        downloaded = search_and_download(query_name, str(local_dir), max_images=max_images)

    if not downloaded:
        logger.error("No images found for %s", poi_name)
        return False
    logger.info("Downloaded %d images", len(downloaded))

    if len(downloaded) < MIN_IMAGES_TO_ADVANCE:
        logger.warning(
            "Only downloaded %d images (need %d minimum) for %s",
            len(downloaded), MIN_IMAGES_TO_ADVANCE, poi_name,
        )

    # Resolve location + description for MiMo filter (use real name for search)
    location = resolve_poi_location(query_name)
    from scraping.core.scraper import resolve_poi_description
    description = resolve_poi_description(query_name, location)
    logger.info("Location: %s", location or "unknown")
    logger.info("Description: %s", (description or "unknown")[:100])

    # Step 3: Run filter pipeline (with metadata collection + MiMo context)
    logger.info("Step 2/3: Running filter pipeline...")
    metadata_collector = {}
    kept = run_filter_pipeline(downloaded, target=TARGET_IMAGES,
                               metadata_collector=metadata_collector,
                               poi_name=poi_name,
                               location=location,
                               description=description)
    logger.info("Filter pipeline kept %d/%d images", len(kept), len(downloaded))

    if len(kept) < TARGET_IMAGES_MIN:
        logger.warning(
            "Only %d images passed filters (need %d minimum) for %s — attempting retry",
            len(kept), TARGET_IMAGES_MIN, poi_name,
        )

        # RETRY: clear cached suffixes, re-scrape with more candidates
        retry_kept = _retry_scrape(
            poi_name, local_dir, location, description,
            existing_count=len(downloaded),
            metadata_collector=metadata_collector,
        )
        if retry_kept and len(retry_kept) >= TARGET_IMAGES_MIN:
            kept = retry_kept
            logger.info("Retry succeeded: %d images for %s", len(kept), poi_name)
        else:
            final_count = len(retry_kept) if retry_kept else len(kept)
            logger.error(
                "Only %d images after retry (need %d minimum) for %s — giving up",
                final_count, TARGET_IMAGES_MIN, poi_name,
            )
            return False

    # Step 4: Upload + register (unless --no-upload)
    if no_upload:
        logger.info("Step 3/3: Skipping upload (--no-upload flag)")
    else:
        logger.info("Step 3/3: Uploading to storage + creating DB record...")
        success = upload_and_register(poi_name, kept,
                                      metadata_collector=metadata_collector,
                                      run_id=run_id)
        if not success:
            logger.error("Upload failed for %s", poi_name)
            return False

    logger.info("Completed %s — %d images kept", poi_name, len(kept))
    return True


def _retry_scrape(poi_name, local_dir, location, description,
                   existing_count=0, metadata_collector=None):
    """Retry scraping with fallback queries + more candidates.

    Strategy:
    1. Clear cached Gemini suffixes so fresh ones are generated
    2. Request 50% more candidates than the first attempt
    3. Run the full filter pipeline on ALL images (existing + new)

    Returns the kept images list, or None on failure.
    """
    import json as _json
    from scraping.core.scraper import resolve_poi_suffixes

    logger.info("RETRY SCRAPE [%s]: clearing cached suffixes, requesting more candidates", poi_name)

    # Clear cached suffixes to force fresh Gemini generation
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        cache = db.check_poi_in_cache(poi_name)
        if cache["exists"] and cache["record"]:
            notes_raw = cache["record"].get("notes") or "{}"
            try:
                notes = _json.loads(notes_raw) if isinstance(notes_raw, str) else notes_raw
            except (_json.JSONDecodeError, TypeError):
                notes = {}
            notes.pop("search_suffixes", None)
            db.client.table("pois_cache").update(
                {"notes": _json.dumps(notes)}
            ).eq("poi_name", poi_name).execute()
            logger.info("RETRY: cleared cached suffixes for %s", poi_name)
    except Exception as exc:
        logger.warning("RETRY: failed to clear suffix cache: %s", exc)

    # Re-scrape with 50% more candidates (additive — existing images stay)
    retry_max = max(int(existing_count * 0.5), 50)
    logger.info("RETRY: downloading %d more candidates for %s", retry_max, poi_name)

    new_downloaded = search_and_download(display_name(poi_name), str(local_dir), max_images=retry_max)
    if not new_downloaded:
        logger.error("RETRY: no new images downloaded for %s", poi_name)
        return None

    logger.info("RETRY: now have %d total images for %s", len(new_downloaded), poi_name)

    # Re-run filter pipeline on ALL images (existing + new combined)
    retry_metadata = {}
    retry_kept = run_filter_pipeline(
        new_downloaded, target=TARGET_IMAGES,
        metadata_collector=retry_metadata,
        poi_name=poi_name, location=location, description=description,
    )
    logger.info("RETRY: filter pipeline kept %d/%d images for %s",
                len(retry_kept), len(new_downloaded), poi_name)

    # Update metadata_collector with retry results
    if metadata_collector is not None:
        metadata_collector.clear()
        metadata_collector.update(retry_metadata)

    return retry_kept


VPS_HOST = os.getenv("VPS_HOST", "")
VPS_PORT = os.getenv("VPS_PORT", "22")
VPS_SCRAPE_DIR = os.getenv("VPS_SCRAPE_DIR", "")


def _has_completed_videos(poi_name: str, db) -> bool:
    """Check if POI has completed compiled videos in poi_batches."""
    try:
        resp = (
            db.client.table("poi_batches")
            .select("id")
            .eq("poi_name", poi_name)
            .eq("compilation_status", "completed")
            .limit(1)
            .execute()
        )
        return bool(resp.data)
    except Exception:
        return False


def _resolve_versioned_name(poi_name: str) -> str:
    """If POI already has completed videos, return versioned name (e.g. 'Hotel v2').

    Only versions when the POI has actually completed the full pipeline.
    An existing record with partial progress (e.g. scraped but not compiled)
    is NOT versioned — that's a retry, not a new version.
    """
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()

        if not _has_completed_videos(poi_name, db):
            return poi_name

        # POI has completed videos — find next available version
        version = 2
        while True:
            candidate = f"{poi_name} v{version}"
            if not db.get_poi_by_name(candidate) and not _has_completed_videos(candidate, db):
                logger.info("POI '%s' has completed videos — versioned to '%s'", poi_name, candidate)
                return candidate
            version += 1
    except Exception as exc:
        logger.warning("Could not check existing POI: %s — using original name", exc)
        return poi_name


def process_poi_bb_only(poi_name: str) -> bool:
    """bb-only mode: extract URLs + download locally, SCP to VPS, create DB record.

    The versioned name (e.g. 'Madonna Inn v2') is only used for storage paths
    and DB records. All searches use the original poi_name so Google gets
    a real hotel name, not 'Madonna Inn v2'.
    """
    logger.info("=" * 60)
    logger.info("BB-ONLY MODE: %s", poi_name)
    logger.info("=" * 60)

    # Step 1: Auto-version only if POI has completed videos
    versioned_name = _resolve_versioned_name(poi_name)

    # Step 2: Extract URLs via bb-browser daemon (always search with original name)
    local_dir = Path(LOCAL_SCRAPE_DIR) / "Scraped" / versioned_name
    manifest_path = str(local_dir / "bb_urls.json")

    logger.info("Step 1/4: Extracting URLs via bb-browser (search: '%s')...", poi_name)
    from scraping.core.bb_source import bb_extract_urls, bb_download_from_manifest
    try:
        bb_extract_urls(poi_name, manifest_path)
    except (ConnectionError, RuntimeError) as exc:
        logger.error("bb-browser extraction failed: %s", exc)
        return False

    # Step 3: Download images locally
    logger.info("Step 2/4: Downloading images to %s...", local_dir)
    downloaded = bb_download_from_manifest(manifest_path, str(local_dir))
    if not downloaded:
        logger.error("No images downloaded for %s", poi_name)
        return False
    logger.info("Downloaded %d images", len(downloaded))

    # Step 4: tar + SCP to VPS (3x faster than scp -r)
    remote_dir = f"{VPS_SCRAPE_DIR}/{versioned_name}"
    logger.info("Step 3/4: tar+SCP %d files to VPS (%s)...", len(downloaded), remote_dir)
    import tempfile
    tar_path = None
    try:
        # Create tar archive locally
        tar_fd, tar_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(tar_fd)
        subprocess.run(
            ["tar", "czf", tar_path, "-C", str(local_dir), "."],
            check=True, timeout=60,
            env={**os.environ, "COPYFILE_DISABLE": "1"},
        )
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        logger.info("Tar archive: %.1fMB", tar_size_mb)

        # Create remote dir + SCP tar + extract on VPS
        subprocess.run(
            ["ssh", "-p", VPS_PORT, VPS_HOST, f"mkdir -p \"{remote_dir}\""],
            check=True, timeout=30,
        )
        subprocess.run(
            ["scp", "-P", VPS_PORT, tar_path, f"{VPS_HOST}:/tmp/_scrape_transfer.tar.gz"],
            check=True, timeout=300, capture_output=True, text=True,
        )
        subprocess.run(
            ["ssh", "-p", VPS_PORT, VPS_HOST,
             f"tar xzf /tmp/_scrape_transfer.tar.gz -C \"{remote_dir}\" && rm /tmp/_scrape_transfer.tar.gz"],
            check=True, timeout=60,
        )
        logger.info("tar+SCP complete")
    except subprocess.TimeoutExpired:
        logger.error("tar+SCP timed out")
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("tar+SCP failed: %s", exc.stderr if hasattr(exc, 'stderr') else exc)
        return False
    finally:
        if tar_path and os.path.exists(tar_path):
            os.unlink(tar_path)

    # Step 5: Create Supabase optimization_layer record
    logger.info("Step 4/4: Creating Supabase record for '%s'...", versioned_name)
    try:
        from shared.supabase_client import SupabaseClient
        db = SupabaseClient()
        existing = db.get_poi_by_name(versioned_name)
        if not existing:
            db.create_poi(versioned_name)
            logger.info("Created optimization_layer record: %s", versioned_name)
        else:
            logger.info("Record already exists for %s", versioned_name)
    except Exception as exc:
        logger.error("Failed to create DB record: %s", exc)
        return False

    # Done — print VPS command
    logger.info("=" * 60)
    logger.info("READY. Run on VPS:")
    logger.info("  python3 run_batch.py --pois '%s' --from-stage scraping", versioned_name)
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Scrape hotel images")
    parser.add_argument("--poi", nargs="+", required=True, help="Hotel/POI names to scrape")
    parser.add_argument("--no-upload", action="store_true", help="Skip Drive upload and DB record")
    parser.add_argument("--force", action="store_true", help="Force re-scrape even if recently done")
    parser.add_argument("--bb-only", action="store_true",
                        help="bb-browser only: extract + download locally, SCP to VPS, create DB record")
    args = parser.parse_args()

    results = {}
    for poi in args.poi:
        if args.bb_only:
            success = process_poi_bb_only(poi)
        else:
            success = process_poi(poi, args.no_upload, args.force)
        results[poi] = "OK" if success else "FAILED"

    logger.info("=" * 60)
    logger.info("SUMMARY")
    for poi, status in results.items():
        logger.info("  %s: %s", poi, status)
    logger.info("=" * 60)

    failed = sum(1 for s in results.values() if s == "FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
