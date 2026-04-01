"""Upload filtered images to Supabase Storage and create database record."""

import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)


def _get_clients():
    """Lazy-import shared clients to avoid import-time side effects."""
    pipeline_root = os.path.join(os.path.dirname(__file__), '..', '..')
    if pipeline_root not in sys.path:
        sys.path.insert(0, pipeline_root)

    from shared.supabase_client import SupabaseClient
    from shared.supabase_storage_client import SupabaseStorageClient
    return SupabaseStorageClient(), SupabaseClient()


def upload_and_register(poi_name: str, image_paths: List[str],
                        metadata_collector: dict = None,
                        run_id: str = None) -> bool:
    """Upload images to Supabase Storage and create/update database record.

    1. Upload all images to pois/{poi_name}/scraped/
    2. Write metadata.json sidecar (aesthetic scores + categories)
    3. Create or update optimization_layer record with storage path

    Args:
        poi_name: Hotel/POI name.
        image_paths: List of local image file paths to upload.
        metadata_collector: Optional dict of {base_key: {aesthetic_score, category}}.

    Returns:
        True on success, False on failure.
    """
    if not image_paths:
        logger.warning("No images to upload for %s", poi_name)
        return False

    try:
        storage, db = _get_clients()
    except Exception as exc:
        logger.error("Failed to initialize clients: %s", exc)
        return False

    storage_prefix = f"pois/{poi_name}/scraped"

    # Archive old scraped images if count differs (prevents mixed-run data)
    existing = [f for f in storage.list_files_in_path(storage_prefix) if not f.get("name", "").endswith(".json")]
    if existing and len(existing) != len(image_paths):
        archived = storage.archive_path(storage_prefix, poi_name)
        if archived:
            logger.info("Archived %d old scraped images (count %d → %d)", archived, len(existing), len(image_paths))

    # Upload images (upsert mode — overwrites if exists)
    uploaded_count = 0
    for path in image_paths:
        filename = os.path.basename(path)
        if storage.file_exists(f"{storage_prefix}/{filename}"):
            logger.debug("Already in storage, skipping: %s", filename)
            uploaded_count += 1
            continue
        result = storage.upload_image(path, poi_name, "scraped")
        if result:
            uploaded_count += 1
        else:
            logger.warning("Failed to upload %s", path)

    logger.info("Uploaded %d/%d images for %s", uploaded_count, len(image_paths), poi_name)

    # Write metadata.json sidecar (non-fatal)
    if metadata_collector:
        try:
            import json
            import tempfile
            meta_content = json.dumps(metadata_collector, indent=2)
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            tmp.write(meta_content)
            tmp.close()
            meta_path = f"{storage_prefix}/metadata.json"
            storage.upload_file(tmp.name, meta_path, content_type="application/json")
            os.unlink(tmp.name)
            logger.info("Wrote metadata.json: %d entries", len(metadata_collector))
        except Exception as exc:
            logger.warning("Failed to write metadata.json: %s — continuing", exc)

    # Create or update Supabase record
    # scraped_pics stores the storage path prefix (not a Drive folder ID)
    try:
        existing = db.get_poi_by_name(poi_name)
        if existing:
            db.update_poi(poi_name, {
                "scraped_pics": storage_prefix,
                "scraped_count": uploaded_count,
                "processing_status": "idle",
            })
            logger.info("Updated existing record for %s", poi_name)
        else:
            db.create_poi(poi_name)
            db.update_poi(poi_name, {
                "scraped_pics": storage_prefix,
                "scraped_count": uploaded_count,
                "processing_status": "idle",
            })
            logger.info("Created new record for %s", poi_name)
    except Exception as exc:
        logger.error("Failed to update Supabase for %s: %s", poi_name, exc)
        return False

    return True
