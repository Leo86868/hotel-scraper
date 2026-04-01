"""Supabase Storage client for pipeline file operations.

Provides upload, download, listing, and public URL generation
for pipeline media files (images, videos) via Supabase Storage.
Mirrors the GoogleDriveClient interface for easy migration.
"""

import errno as errno_module
import logging
import mimetypes
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "pipeline-media"


class SupabaseStorageClient:
    """Supabase Storage client for pipeline media files."""

    def __init__(self, bucket: str = DEFAULT_BUCKET):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        self.client = create_client(url, key)
        self.bucket = bucket
        self._base_url = f"{url}/storage/v1/object/public/{bucket}"

    def _build_path(self, poi_name: str, stage: str, filename: str,
                    run_id: str = None) -> str:
        """Build storage path: pois/{poi_name}/[{run_id}/]{stage}/{filename}."""
        parts = ["pois", poi_name]
        if run_id:
            parts.append(run_id)
        parts.extend([stage, filename])
        return "/".join(parts)

    # ------------------------------------------------------------------ #
    #  Upload
    # ------------------------------------------------------------------ #

    def upload_file(self, local_path: str, storage_path: str,
                    content_type: str = None) -> Optional[dict]:
        """Upload a file to storage. Returns {path, url} or None on failure.

        Retries on Errno 35 (Resource temporarily unavailable) up to 3 times.
        After final failure, checks if file actually exists in storage
        (ghost upload detection — the upload may succeed server-side
        despite a client-side socket error).
        """
        if not os.path.isfile(local_path):
            logger.error("File not found: %s", local_path)
            return None

        if content_type is None:
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = "application/octet-stream"

        MAX_RETRIES = 3
        RETRY_DELAY = 2.0

        for attempt in range(MAX_RETRIES):
            try:
                with open(local_path, "rb") as f:
                    self.client.storage.from_(self.bucket).upload(
                        storage_path, f,
                        {"content-type": content_type, "upsert": "true"},
                    )
                url = self.get_public_url(storage_path)
                logger.info("Uploaded %s → %s", os.path.basename(local_path), storage_path)
                return {"path": storage_path, "url": url}

            except OSError as e:
                if e.errno == errno_module.EAGAIN or e.errno == 35:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            "Upload attempt %d/%d got Errno 35 for %s — retrying in %.1fs",
                            attempt + 1, MAX_RETRIES, storage_path, RETRY_DELAY,
                        )
                        time.sleep(RETRY_DELAY)
                        continue
                    # Final attempt — check if file actually made it
                    logger.warning("Errno 35 on final attempt for %s — checking existence", storage_path)
                    if self.file_exists(storage_path):
                        url = self.get_public_url(storage_path)
                        logger.warning("Ghost upload confirmed for %s — file exists despite Errno 35", storage_path)
                        return {"path": storage_path, "url": url}
                    logger.error("Upload truly failed for %s after %d attempts", storage_path, MAX_RETRIES)
                    return None
                # Non-Errno-35 OSError — check existence as safety net
                logger.warning("OSError uploading %s: %s — checking existence", storage_path, e)
                if self.file_exists(storage_path):
                    url = self.get_public_url(storage_path)
                    logger.warning("File exists despite OSError — counting as success")
                    return {"path": storage_path, "url": url}
                return None

            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "Upload attempt %d/%d failed for %s: %s — retrying",
                        attempt + 1, MAX_RETRIES, storage_path, exc,
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                # Final attempt — check existence
                logger.warning("Upload failed for %s: %s — checking existence", storage_path, exc)
                if self.file_exists(storage_path):
                    url = self.get_public_url(storage_path)
                    logger.warning("Ghost upload confirmed — counting as success")
                    return {"path": storage_path, "url": url}
                logger.error("Upload truly failed for %s", storage_path)
                return None

        return None

    def upload_image(self, image_path: str, poi_name: str, stage: str,
                     filename: str = None, run_id: str = None) -> Optional[dict]:
        """Upload image to pois/{poi_name}/[run_id/]{stage}/{filename}."""
        if filename is None:
            filename = os.path.basename(image_path)
        storage_path = self._build_path(poi_name, stage, filename, run_id)
        content_type, _ = mimetypes.guess_type(image_path)
        if not content_type or not content_type.startswith("image/"):
            content_type = "image/jpeg"
        return self.upload_file(image_path, storage_path, content_type)

    def upload_video(self, video_path: str, poi_name: str, stage: str,
                     filename: str = None, run_id: str = None) -> Optional[dict]:
        """Upload video to pois/{poi_name}/[run_id/]{stage}/{filename}."""
        if filename is None:
            filename = os.path.basename(video_path)
        storage_path = self._build_path(poi_name, stage, filename, run_id)
        content_type, _ = mimetypes.guess_type(video_path)
        if not content_type or not content_type.startswith("video/"):
            content_type = "video/mp4"
        return self.upload_file(video_path, storage_path, content_type)

    # ------------------------------------------------------------------ #
    #  Download
    # ------------------------------------------------------------------ #

    def download_file(self, storage_path: str, local_path: str) -> bool:
        """Download a file from storage. Returns True on success."""
        try:
            data = self.client.storage.from_(self.bucket).download(storage_path)
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded %s → %s", storage_path, local_path)
            return True
        except Exception as exc:
            logger.error("Download failed for %s: %s", storage_path, exc)
            return False

    # ------------------------------------------------------------------ #
    #  Public URLs
    # ------------------------------------------------------------------ #

    def get_public_url(self, storage_path: str) -> str:
        """Get permanent public CDN URL. Computed client-side, no API call."""
        return f"{self._base_url}/{storage_path}"

    # ------------------------------------------------------------------ #
    #  List / Query
    # ------------------------------------------------------------------ #

    def list_files(self, prefix: str) -> List[dict]:
        """List files under a path prefix. Returns list of file metadata dicts."""
        try:
            # Split prefix into folder path for the API
            parts = prefix.rstrip("/").rsplit("/", 1)
            if len(parts) == 2:
                folder, search_prefix = parts[0], parts[1]
            else:
                folder, search_prefix = parts[0], ""

            results = self.client.storage.from_(self.bucket).list(
                folder,
                {"search": search_prefix} if search_prefix else {},
            )
            return [f for f in (results or []) if f.get("name")]
        except Exception as exc:
            logger.error("List failed for prefix '%s': %s", prefix, exc)
            return []

    def list_files_in_path(self, path: str) -> List[dict]:
        """List all files directly inside a path."""
        try:
            results = self.client.storage.from_(self.bucket).list(path)
            return [f for f in (results or []) if f.get("name")]
        except Exception as exc:
            logger.error("List failed for path '%s': %s", path, exc)
            return []

    def file_exists(self, storage_path: str) -> bool:
        """Check if a file exists at the given path."""
        parts = storage_path.rsplit("/", 1)
        if len(parts) != 2:
            return False
        folder, filename = parts
        files = self.list_files_in_path(folder)
        return any(f.get("name") == filename for f in files)

    def count_files(self, prefix: str) -> int:
        """Count files under a path prefix."""
        return len(self.list_files_in_path(prefix))

    # ------------------------------------------------------------------ #
    #  Archive (move old files before re-run)
    # ------------------------------------------------------------------ #

    def archive_path(self, source_path: str, poi_name: str) -> int:
        """Move all files in source_path to an archive subfolder.

        Moves to pois/{poi_name}/archive/{YYYYMMDD_HHMMSS}/{stage}/
        Returns count of files moved. Skips metadata.json.
        """
        from datetime import datetime

        files = self.list_files_in_path(source_path)
        if not files:
            return 0

        stage = source_path.rstrip("/").rsplit("/", 1)[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_prefix = f"pois/{poi_name}/archive/{timestamp}/{stage}"

        moved = 0
        for f in files:
            fname = f.get("name", "")
            if not fname or fname.endswith(".json"):
                continue
            src = f"{source_path}/{fname}"
            dst = f"{archive_prefix}/{fname}"
            try:
                self.client.storage.from_(self.bucket).move(src, dst)
                moved += 1
            except Exception as exc:
                logger.warning("Failed to archive %s → %s: %s", src, dst, exc)

        if moved:
            logger.info("Archived %d files: %s → %s", moved, source_path, archive_prefix)
        return moved

    def move_file(self, source_path: str, dest_path: str) -> bool:
        """Move a file within the same bucket. Returns True on success."""
        try:
            self.client.storage.from_(self.bucket).move(source_path, dest_path)
            logger.info("Moved %s → %s", source_path, dest_path)
            return True
        except Exception as exc:
            logger.warning("Failed to move %s → %s: %s", source_path, dest_path, exc)
            return False

    # ------------------------------------------------------------------ #
    #  Delete
    # ------------------------------------------------------------------ #

    def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage. Returns True on success."""
        try:
            self.client.storage.from_(self.bucket).remove([storage_path])
            logger.info("Deleted %s", storage_path)
            return True
        except Exception as exc:
            logger.error("Delete failed for %s: %s", storage_path, exc)
            return False

    def delete_files(self, storage_paths: List[str]) -> int:
        """Delete multiple files. Returns count of successful deletions."""
        if not storage_paths:
            return 0
        try:
            self.client.storage.from_(self.bucket).remove(storage_paths)
            logger.info("Deleted %d files", len(storage_paths))
            return len(storage_paths)
        except Exception as exc:
            logger.error("Bulk delete failed: %s", exc)
            return 0
