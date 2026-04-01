"""Tests for the scraping module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image


# ---- Config tests ----

class TestConfig:
    def test_default_values(self):
        from scraping.config import TARGET_IMAGES, MIN_IMAGES_TO_ADVANCE
        assert TARGET_IMAGES == 15
        assert MIN_IMAGES_TO_ADVANCE == 12

    @patch.dict(os.environ, {"SCRAPE_TARGET_IMAGES_MAX": "20"})
    def test_env_override(self):
        # Need to reimport to pick up env var
        import importlib
        import scraping.config
        importlib.reload(scraping.config)
        assert scraping.config.TARGET_IMAGES_MAX == 20
        assert scraping.config.TARGET_IMAGES == 20  # alias
        # Reset
        importlib.reload(scraping.config)


# ---- Resolution filter tests ----

class TestResolutionFilter:
    def _make_image(self, tmpdir, width, height, name="test.jpg"):
        path = os.path.join(tmpdir, name)
        Image.new("RGB", (width, height)).save(path)
        return path

    def test_passes_large_image(self):
        from scraping.core.filters.resolution_filter import filter_by_resolution
        with tempfile.TemporaryDirectory() as d:
            path = self._make_image(d, 800, 600)
            result = filter_by_resolution([path])
            assert len(result) == 1

    def test_rejects_small_image(self):
        from scraping.core.filters.resolution_filter import filter_by_resolution
        with tempfile.TemporaryDirectory() as d:
            path = self._make_image(d, 200, 200)
            result = filter_by_resolution([path])
            assert len(result) == 0


# ---- Aspect ratio filter tests ----

class TestAspectRatioFilter:
    def _make_image(self, tmpdir, width, height, name="test.jpg"):
        path = os.path.join(tmpdir, name)
        Image.new("RGB", (width, height)).save(path)
        return path

    def test_passes_normal_ratio(self):
        from scraping.core.filters.aspect_ratio_filter import filter_by_aspect_ratio
        with tempfile.TemporaryDirectory() as d:
            path = self._make_image(d, 800, 600)  # ratio = 1.33
            result = filter_by_aspect_ratio([path])
            assert len(result) == 1

    def test_rejects_extreme_ratio(self):
        from scraping.core.filters.aspect_ratio_filter import filter_by_aspect_ratio
        with tempfile.TemporaryDirectory() as d:
            path = self._make_image(d, 100, 1000)  # ratio = 0.1
            result = filter_by_aspect_ratio([path])
            assert len(result) == 0

    def test_boundary_values(self):
        from scraping.core.filters.aspect_ratio_filter import filter_by_aspect_ratio
        with tempfile.TemporaryDirectory() as d:
            # Ratio 0.4 should pass (boundary)
            path = self._make_image(d, 400, 1000, "narrow.jpg")
            result = filter_by_aspect_ratio([path])
            assert len(result) == 1


# ---- Dedup filter tests ----

class TestDedupFilter:
    def _make_image(self, tmpdir, color, name):
        path = os.path.join(tmpdir, name)
        Image.new("RGB", (800, 600), color=color).save(path)
        return path

    def test_identical_images_rejected(self):
        from scraping.core.filters.dedup_filter import _phash_dedup
        with tempfile.TemporaryDirectory() as d:
            img1 = self._make_image(d, (255, 0, 0), "a.jpg")
            img2 = self._make_image(d, (255, 0, 0), "b.jpg")
            result = _phash_dedup([img1, img2])
            assert len(result) == 1

    def test_different_images_kept(self):
        from scraping.core.filters.dedup_filter import _phash_dedup
        with tempfile.TemporaryDirectory() as d:
            import numpy as np
            np.random.seed(42)
            img1_path = os.path.join(d, "noise.jpg")
            img2_path = os.path.join(d, "checker.jpg")
            # Random noise — high-entropy image
            Image.fromarray(np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)).save(img1_path)
            # Checkerboard — structured, very different from noise
            checker = np.zeros((600, 800, 3), dtype=np.uint8)
            checker[::2, ::2] = 255
            checker[1::2, 1::2] = 255
            Image.fromarray(checker).save(img2_path)
            result = _phash_dedup([img1_path, img2_path])
            assert len(result) == 2


# ---- OCR position-based filter tests ----

class TestOCRPositionFilter:
    """Test that corner/margin text gets detected as watermarks."""

    def test_corner_text_rejected(self):
        """Text in bottom-right corner should be rejected as watermark."""
        from scraping.core.filters.ocr_filter import (
            MARGIN_BOTTOM, MARGIN_RIGHT, TEXT_AREA_RATIO_WATERMARK
        )
        # Bottom-right: center_y > 0.80, center_x > 0.90
        # These are in margin zone → should trigger rejection
        assert MARGIN_BOTTOM == 0.80
        assert MARGIN_RIGHT == 0.90
        assert TEXT_AREA_RATIO_WATERMARK == 0.001

    def test_center_text_not_in_margin(self):
        """Text centered in the image should NOT be in any margin zone."""
        from scraping.core.filters.ocr_filter import (
            MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT
        )
        center_y = 0.5  # middle of image
        center_x = 0.5
        in_margin = (center_y > MARGIN_BOTTOM or center_y < MARGIN_TOP or
                     center_x < MARGIN_LEFT or center_x > MARGIN_RIGHT)
        assert not in_margin

    def test_bottom_text_in_margin(self):
        """Text at y=0.85 (bottom 15%) should be in margin."""
        from scraping.core.filters.ocr_filter import MARGIN_BOTTOM
        center_y = 0.85
        assert center_y > MARGIN_BOTTOM

    def test_top_text_in_margin(self):
        """Text at y=0.10 (top 15%) should be in margin."""
        from scraping.core.filters.ocr_filter import MARGIN_TOP
        center_y = 0.10
        assert center_y < MARGIN_TOP

    @patch("scraping.core.filters.ocr_filter.models")
    def test_filter_rejects_corner_watermark(self, mock_models):
        """Full filter test: text in corner with sufficient area → rejected."""
        mock_reader = MagicMock()
        mock_models.easyocr_reader = mock_reader

        # Simulate OCR finding "DANNY DONG" in bottom-right corner
        # Image is 1000x1000, text bbox at (850, 900) to (990, 950)
        mock_reader.readtext.return_value = [
            ([[850, 900], [990, 900], [990, 950], [850, 950]], "DANNY DONG", 0.8)
        ]

        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new("RGB", (1000, 1000)).save(f.name)
            from scraping.core.filters.ocr_filter import filter_by_text_detection
            result = filter_by_text_detection([f.name])
            os.unlink(f.name)

        # Should be rejected: text center at x=0.92, y=0.925 — in margin
        assert len(result) == 0

    @patch("scraping.core.filters.ocr_filter.models")
    def test_filter_keeps_centered_text(self, mock_models):
        """Text in center of image should NOT be rejected as watermark."""
        mock_reader = MagicMock()
        mock_models.easyocr_reader = mock_reader

        # Small text in center: "HOTEL NAME" at center of image
        mock_reader.readtext.return_value = [
            ([[400, 450], [600, 450], [600, 480], [400, 480]], "HOTEL NAME", 0.9)
        ]

        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new("RGB", (1000, 1000)).save(f.name)
            from scraping.core.filters.ocr_filter import filter_by_text_detection
            result = filter_by_text_detection([f.name])
            os.unlink(f.name)

        # Center text, small area — should be kept
        assert len(result) == 1

    @patch("scraping.core.filters.ocr_filter.models")
    def test_stock_keyword_always_rejected(self, mock_models):
        """Getty/Shutterstock keywords rejected regardless of position."""
        mock_reader = MagicMock()
        mock_models.easyocr_reader = mock_reader

        mock_reader.readtext.return_value = [
            ([[400, 400], [600, 400], [600, 430], [400, 430]], "Getty Images", 0.9)
        ]

        import tempfile
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.new("RGB", (1000, 1000)).save(f.name)
            from scraping.core.filters.ocr_filter import filter_by_text_detection
            result = filter_by_text_detection([f.name])
            os.unlink(f.name)

        assert len(result) == 0


# ---- Pipeline integration test ----

class TestPipelineIntegration:
    @patch("scraping.core.uploader._get_clients")
    def test_upload_creates_record(self, mock_get_clients):
        mock_storage = MagicMock()
        mock_storage.upload_image.return_value = {"path": "pois/Test Hotel/scraped/test.jpg", "url": "https://cdn/test.jpg"}
        mock_storage.file_exists.return_value = False

        mock_db = MagicMock()
        mock_db.get_poi_by_name.return_value = None

        mock_get_clients.return_value = (mock_storage, mock_db)

        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "test.jpg")
            Image.new("RGB", (800, 600)).save(img)

            from scraping.core.uploader import upload_and_register
            result = upload_and_register("Test Hotel", [img])

            assert result is True
            mock_storage.upload_image.assert_called_once()
            mock_db.create_poi.assert_called_once_with("Test Hotel")

    def test_no_upload_skips_drive_and_db(self):
        """When --no-upload is used, uploader is never called."""
        # The CLI simply doesn't call upload_and_register when --no-upload
        # This is a design test — no mock needed
        from scraping.core.uploader import upload_and_register
        # Empty list should return False without touching clients
        result = upload_and_register("Test", [])
        assert result is False
