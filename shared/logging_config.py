"""Structured logging for the pipeline.

Usage:
    from shared.logging_config import setup_logging, set_poi_context

    # At startup (run_pipeline.py, run_batch.py):
    setup_logging()

    # When starting work on a POI:
    set_poi_context("Amangiri")

    # All subsequent logger calls automatically include [Amangiri]:
    logger.info("Stage 2: uploading image 3/15")
    # Output: 2026-03-16 19:30:00 [INFO ] [Amangiri] Stage 2: uploading image 3/15
"""

import logging
import sys
import threading
from datetime import datetime

# Thread-local storage for POI context — safe for concurrent use
_local = threading.local()


def set_poi_context(poi_name: str):
    """Set the current POI name for this thread's log messages."""
    _local.poi_name = poi_name


def clear_poi_context():
    """Clear POI context after a POI's work is done."""
    _local.poi_name = None


class POIContextFilter(logging.Filter):
    """Injects POI name into every log record from this thread."""

    def filter(self, record):
        record.poi = getattr(_local, "poi_name", None) or "-"
        return True


def setup_logging(level=logging.INFO, log_file=None):
    """Configure structured logging for the pipeline.

    Format: TIMESTAMP [LEVEL] [POI] MESSAGE
    Example: 2026-03-16 19:30:00 [INFO ] [Amangiri] Uploaded 15 images

    Args:
        level: Logging level (default INFO)
        log_file: Optional path to write logs to file as well as stdout
    """
    fmt = "%(asctime)s [%(levelname)-5s] [%(poi)-20s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    poi_filter = POIContextFilter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(poi_filter)

    handlers = [console]

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(poi_filter)
        handlers.append(file_handler)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)


def get_batch_log_path():
    """Generate a log file path for the current batch run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/tmp/pipeline_batch_{timestamp}.log"
