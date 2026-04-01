"""Supabase client for all pipeline database operations.

Provides synchronous access to optimization_layer, poi_batches,
compiled_videos, and music_library tables via supabase-py.
"""

import logging
from supabase import create_client, Client
from typing import Optional, List
from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseClient:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        self.client: Client = create_client(url, key)

    # ------------------------------------------------------------------ #
    #  Optimization Layer
    # ------------------------------------------------------------------ #

    def get_poi_by_name(self, poi_name: str) -> Optional[dict]:
        """Get POI record by name. Returns None if not found."""
        response = (
            self.client.table("optimization_layer")
            .select("*")
            .eq("poi_name", poi_name)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None

    def create_poi(self, poi_name: str) -> dict:
        """Create new POI record. Returns created record."""
        response = (
            self.client.table("optimization_layer")
            .insert({"poi_name": poi_name})
            .execute()
        )
        if not response.data:
            raise ValueError(f"Failed to create POI '{poi_name}': empty response from Supabase")
        return response.data[0]

    def update_poi(self, poi_name: str, fields: dict) -> dict:
        """Update POI fields by name. Returns updated record."""
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        response = (
            self.client.table("optimization_layer")
            .update(fields)
            .eq("poi_name", poi_name)
            .execute()
        )
        if not response.data:
            raise ValueError(f"Failed to update POI '{poi_name}': no matching record found")
        return response.data[0]

    def get_pending_image_optimization(self, min_threshold: int = None) -> List[dict]:
        """POIs with scraped_count >= threshold, img_count == 0, and processing_status == 'idle'."""
        threshold = min_threshold or int(os.getenv("MIN_TO_ADVANCE", 12))
        response = (
            self.client.table("optimization_layer")
            .select("*")
            .gte("scraped_count", threshold)
            .eq("img_count", 0)
            .eq("processing_status", "idle")
            .execute()
        )
        return response.data or []

    def get_pending_video_generation(self, min_threshold: int = None) -> List[dict]:
        """POIs with img_count >= threshold, clip_count == 0, and processing_status == 'idle'."""
        threshold = min_threshold or int(os.getenv("MIN_TO_ADVANCE", 12))
        response = (
            self.client.table("optimization_layer")
            .select("*")
            .gte("img_count", threshold)
            .eq("clip_count", 0)
            .eq("processing_status", "idle")
            .execute()
        )
        return response.data or []

    def claim_poi_for_processing(self, poi_name: str) -> bool:
        """Atomically claim a POI for processing.

        Uses a single UPDATE with WHERE clause — if another worker already
        claimed this POI, the WHERE won't match and response.data will be
        empty. Postgres guarantees row-level atomicity.

        Returns True if claimed successfully, False if already claimed.
        """
        response = (
            self.client.table("optimization_layer")
            .update({
                "processing_status": "processing",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("poi_name", poi_name)
            .eq("processing_status", "idle")
            .execute()
        )
        return len(response.data or []) > 0

    def release_poi(self, poi_name: str, status: str = "idle") -> None:
        """Release POI back to idle or mark with another status."""
        self.update_poi(poi_name, {"processing_status": status})

    def reset_stale_processing(self, max_age_minutes: int = 120) -> int:
        """Reset POIs stuck in 'processing' for longer than max_age_minutes.

        Returns the number of POIs reset.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        ).isoformat()
        response = (
            self.client.table("optimization_layer")
            .update({"processing_status": "idle"})
            .eq("processing_status", "processing")
            .lt("updated_at", cutoff)
            .execute()
        )
        count = len(response.data or [])
        if count:
            logger.info("Reset %d stale POIs from 'processing' to 'idle'", count)
        return count

    def reset_stale_runs(self, max_age_minutes: int = 120) -> dict:
        """Reset pipeline runs and steps stuck in 'running' for too long.

        Marks them as 'stale' (not 'failed') to distinguish crashed runs
        from runs that explicitly reported failure.

        Returns dict with counts: {"runs": N, "steps": N}.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        ).isoformat()
        now = datetime.now(timezone.utc).isoformat()

        # Reset stale pipeline_runs
        runs_resp = (
            self.client.table("pipeline_runs")
            .update({"status": "stale", "completed_at": now})
            .eq("status", "running")
            .lt("started_at", cutoff)
            .execute()
        )
        runs_count = len(runs_resp.data or [])

        # Reset stale pipeline_run_steps
        steps_resp = (
            self.client.table("pipeline_run_steps")
            .update({"status": "stale", "completed_at": now})
            .eq("status", "running")
            .lt("started_at", cutoff)
            .execute()
        )
        steps_count = len(steps_resp.data or [])

        if runs_count or steps_count:
            logger.info(
                "Reset %d stale run(s) and %d stale step(s) to 'stale'",
                runs_count, steps_count,
            )
        return {"runs": runs_count, "steps": steps_count}

    # ------------------------------------------------------------------ #
    #  POI Batches
    # ------------------------------------------------------------------ #

    def get_or_create_poi_batch(self, poi_name: str, folder_id: str, overlay_text: str = None) -> dict:
        """Get existing batch or create new one. Returns record."""
        response = (
            self.client.table("poi_batches")
            .select("*")
            .eq("poi_name", poi_name)
            .execute()
        )
        if response.data:
            return response.data[0]

        # Look up the optimization_layer id for the foreign key
        poi_record = self.get_poi_by_name(poi_name)
        insert_payload = {
            "poi_name": poi_name,
            "clips_folder": folder_id,
        }
        if overlay_text:
            insert_payload["simple_overlay_text"] = overlay_text
        if poi_record:
            insert_payload["optimization_layer_id"] = poi_record["id"]

        response = (
            self.client.table("poi_batches")
            .insert(insert_payload)
            .execute()
        )
        return response.data[0]

    def get_ready_for_vibe(self) -> List[dict]:
        """POI batches with compilation_status == 'pending'."""
        response = (
            self.client.table("poi_batches")
            .select("*")
            .eq("compilation_status", "pending")
            .execute()
        )
        return response.data or []

    def update_poi_batch_overlay(self, poi_batch_id: str, overlay_text: str) -> dict:
        """Write generated overlay text back to poi_batches."""
        response = (
            self.client.table("poi_batches")
            .update({"simple_overlay_text": overlay_text})
            .eq("id", poi_batch_id)
            .execute()
        )
        return response.data[0] if response.data else {}

    def update_poi_batch_status(self, poi_batch_id: str, status: str) -> dict:
        """Update compilation_status on poi_batches record."""
        response = (
            self.client.table("poi_batches")
            .update({"compilation_status": status})
            .eq("id", poi_batch_id)
            .execute()
        )
        return response.data[0] if response.data else {}

    def get_existing_compiled_videos(self, poi_batch_id: str) -> List[dict]:
        """Get existing compiled_videos for a POI batch."""
        response = (
            self.client.table("compiled_videos")
            .select("*")
            .eq("poi_id", poi_batch_id)
            .execute()
        )
        return response.data or []

    def compiled_video_exists(self, poi_batch_id: str, variation: str) -> bool:
        """Check if a specific variation already exists for this POI batch."""
        response = (
            self.client.table("compiled_videos")
            .select("id")
            .eq("poi_id", poi_batch_id)
            .eq("video_variation", variation)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    # ------------------------------------------------------------------ #
    #  Compiled Videos
    # ------------------------------------------------------------------ #

    def create_compiled_video(self, poi_batch_id: str, fields: dict) -> dict:
        """Create compiled video record. Returns created record."""
        payload = {"poi_id": poi_batch_id}
        payload.update(fields)
        response = (
            self.client.table("compiled_videos")
            .insert(payload)
            .execute()
        )
        if not response.data:
            raise ValueError(f"Failed to create compiled video for batch '{poi_batch_id}': empty response")
        return response.data[0]

    def get_videos_for_poi(self, poi_batch_id: str) -> List[dict]:
        """Get all compiled videos for a POI batch."""
        response = (
            self.client.table("compiled_videos")
            .select("*")
            .eq("poi_id", poi_batch_id)
            .execute()
        )
        return response.data or []

    def update_video(self, video_id: str, fields: dict) -> dict:
        """Update video record fields."""
        response = (
            self.client.table("compiled_videos")
            .update(fields)
            .eq("id", video_id)
            .execute()
        )
        if not response.data:
            raise ValueError(f"Failed to update video '{video_id}': no matching record found")
        return response.data[0]

    # ------------------------------------------------------------------ #
    #  Download Pipeline
    # ------------------------------------------------------------------ #

    def get_videos_for_download(self, poi_name: str = None) -> List[dict]:
        """Get compiled videos eligible for download.

        Filters: publishing_status='draft', compressed_video_url is null.
        Optionally filters by POI name via poi_batches join.

        Args:
            poi_name: If provided, only return videos for this POI.

        Returns:
            List of compiled_videos records.
        """
        query = (
            self.client.table("compiled_videos")
            .select("*, poi_batches!inner(poi_name)")
            .eq("publishing_status", "draft")
            .is_("compressed_video_url", "null")
        )
        if poi_name:
            query = query.eq("poi_batches.poi_name", poi_name)

        response = query.execute()
        # Flatten the poi_name from the join
        results = []
        for row in (response.data or []):
            batch_info = row.pop("poi_batches", {})
            row["poi_name"] = batch_info.get("poi_name", "")
            results.append(row)
        return results

    def create_text_overlay_variation(self, compiled_video_id: str,
                                      overlay_text: str, final_output_url: str) -> dict:
        """Create text overlay variation record for a compiled video."""
        response = (
            self.client.table("text_overlay_variations")
            .insert({
                "compiled_video_id": compiled_video_id,
                "overlay_text": overlay_text,
                "final_output_url": final_output_url,
                "status": "completed",
            })
            .execute()
        )
        return response.data[0]

    def get_overlay_url(self, video_id: str) -> Optional[str]:
        """Get the final_output_url from text_overlay_variations for a video.

        Args:
            video_id: The compiled_videos ID.

        Returns:
            The final_output_url string, or None.
        """
        response = (
            self.client.table("text_overlay_variations")
            .select("final_output_url")
            .eq("compiled_video_id", video_id)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0].get("final_output_url")
        return None

    # ------------------------------------------------------------------ #
    #  POI Cache (discover-pois dedup)
    # ------------------------------------------------------------------ #

    def check_poi_in_cache(self, poi_name: str, skip_days: int = 15) -> dict:
        """Check if a POI exists in pois_cache and how recently it was synced.

        Returns:
            Dict with keys:
                - exists (bool): Whether the POI is in cache.
                - status (str): "new", "skip", or "rescrape".
                - days_ago (int | None): Days since first_synced_at, or None.
                - record (dict | None): The raw cache record, or None.
        """
        response = (
            self.client.table("pois_cache")
            .select("*")
            .eq("poi_name", poi_name)
            .execute()
        )
        if not response.data:
            return {"exists": False, "status": "new", "days_ago": None, "record": None}

        record = response.data[0]
        synced_at = record.get("first_synced_at")
        if not synced_at:
            return {"exists": True, "status": "rescrape", "days_ago": None, "record": record}

        synced_dt = datetime.fromisoformat(synced_at.replace("Z", "+00:00"))
        days_ago = (datetime.now(timezone.utc) - synced_dt).days

        if days_ago < skip_days:
            status = "skip"
        else:
            status = "rescrape"

        return {"exists": True, "status": status, "days_ago": days_ago, "record": record}

    # ------------------------------------------------------------------ #
    #  Music
    # ------------------------------------------------------------------ #

    def get_music_files(self, count: int = 3) -> List[dict]:
        """Get music tracks from library.

        Orders by created_at descending as a deterministic fallback
        (Supabase REST API does not support random ordering).
        """
        response = (
            self.client.table("music_library")
            .select("*")
            .order("created_at", desc=True)
            .limit(count)
            .execute()
        )
        return response.data or []

    def create_music_track(self, fields: dict) -> dict:
        """Insert a new music track into music_library."""
        response = (
            self.client.table("music_library")
            .insert(fields)
            .execute()
        )
        return response.data[0]

    def get_music_track_by_name(self, music_name: str) -> Optional[dict]:
        """Check if a track already exists by name (dedup)."""
        response = (
            self.client.table("music_library")
            .select("*")
            .eq("music_name", music_name)
            .execute()
        )
        return response.data[0] if response.data else None

    def update_music_track(self, track_id: str, fields: dict) -> dict:
        """Update an existing music track."""
        response = (
            self.client.table("music_library")
            .update(fields)
            .eq("id", track_id)
            .execute()
        )
        return response.data[0] if response.data else {}

    # ------------------------------------------------------------------ #
    #  Pipeline Runs
    # ------------------------------------------------------------------ #

    def create_run(self, run_id: str, poi_names: List[str],
                   from_stage: str = "scraping", config: dict = None,
                   triggered_by: str = "manual",
                   parent_run_id: str = None) -> dict:
        """Create a new pipeline_runs record for a batch.

        Args:
            run_id: Unique run identifier (e.g. "run_20260317_143052_a3b2c1").
            poi_names: List of POI names in this batch.
            from_stage: Starting stage name.
            config: Optional config snapshot (workers, thresholds, etc).
            triggered_by: "manual", "cron", or "rerun".
            parent_run_id: If this is a rerun, the original run_id.

        Returns:
            Created pipeline_runs record.
        """
        payload = {
            "run_id": run_id,
            "poi_names": poi_names,
            "poi_count": len(poi_names),
            "from_stage": from_stage,
            "triggered_by": triggered_by,
        }
        if config:
            payload["config"] = config
        if parent_run_id:
            payload["parent_run_id"] = parent_run_id
        response = (
            self.client.table("pipeline_runs")
            .insert(payload)
            .execute()
        )
        return response.data[0]

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a pipeline_runs record by run_id."""
        response = (
            self.client.table("pipeline_runs")
            .select("*")
            .eq("run_id", run_id)
            .execute()
        )
        return response.data[0] if response.data else None

    def complete_run(self, run_id: str, summary: dict = None) -> dict:
        """Mark a pipeline run as completed with summary stats."""
        fields = {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if summary:
            fields["summary"] = summary
        return self._update_run(run_id, fields)

    def fail_run(self, run_id: str, error_message: str = None,
                 summary: dict = None) -> dict:
        """Mark a pipeline run as failed."""
        fields = {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if error_message:
            fields["error_message"] = error_message
        if summary:
            fields["summary"] = summary
        return self._update_run(run_id, fields)

    def _update_run(self, run_id: str, fields: dict) -> dict:
        """Update pipeline_runs fields by run_id."""
        response = (
            self.client.table("pipeline_runs")
            .update(fields)
            .eq("run_id", run_id)
            .execute()
        )
        return response.data[0] if response.data else {}

    def list_runs(self, limit: int = 10) -> List[dict]:
        """Get recent pipeline runs ordered by started_at desc."""
        response = (
            self.client.table("pipeline_runs")
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []

    # ------------------------------------------------------------------ #
    #  Pipeline Run Steps
    # ------------------------------------------------------------------ #

    def create_run_step(self, run_id: str, poi_name: str, stage: str) -> dict:
        """Create a pipeline_run_steps record (status='running').

        Args:
            run_id: The parent run identifier.
            poi_name: POI being processed.
            stage: Stage name (scraping, image-opt, video-gen, vibe, download).

        Returns:
            Created pipeline_run_steps record.
        """
        response = (
            self.client.table("pipeline_run_steps")
            .insert({
                "run_id": run_id,
                "poi_name": poi_name,
                "stage": stage,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
            .execute()
        )
        return response.data[0]

    def complete_run_step(self, run_id: str, poi_name: str, stage: str,
                          status: str, duration_s: float = None,
                          metrics: dict = None,
                          error_message: str = None) -> dict:
        """Update a run step with final status and optional metrics.

        Args:
            run_id: The parent run identifier.
            poi_name: POI that was processed.
            stage: Stage name.
            status: Final status (passed, failed, skipped, gate_fail).
            duration_s: How long the stage took in seconds.
            metrics: Optional snapshot (e.g. {scraped_count: 15}).
            error_message: Error details if failed.

        Returns:
            Updated pipeline_run_steps record.
        """
        fields = {
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if duration_s is not None:
            fields["duration_s"] = duration_s
        if metrics:
            fields["metrics"] = metrics
        if error_message:
            fields["error_message"] = error_message
        response = (
            self.client.table("pipeline_run_steps")
            .update(fields)
            .eq("run_id", run_id)
            .eq("poi_name", poi_name)
            .eq("stage", stage)
            .execute()
        )
        return response.data[0] if response.data else {}

    def get_failed_pois(self, run_id: str) -> List[str]:
        """Get POI names that failed in a given run (for --rerun).

        Returns:
            List of POI names with at least one failed step.
        """
        response = (
            self.client.table("pipeline_run_steps")
            .select("poi_name")
            .eq("run_id", run_id)
            .eq("status", "failed")
            .execute()
        )
        # Deduplicate (a POI might fail at multiple stages)
        return list({r["poi_name"] for r in (response.data or [])})
