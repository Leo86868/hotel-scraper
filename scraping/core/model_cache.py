"""Lazy-loading model cache for ML models used in the filter pipeline.

Loads each model on first access and caches for reuse. This avoids loading
6-8GB of models multiple times when running the full filter pipeline.
"""

import logging
import os
import threading
from typing import Optional
from scraping.config import DEVICE

logger = logging.getLogger(__name__)

class ModelCache:
    """Thread-safe singleton cache for heavy ML models.

    Uses per-model locks to prevent double-load races and
    unload-while-in-use when multiple POIs run concurrently.
    """

    _instance: Optional['ModelCache'] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._yolo_model = None
        self._easyocr_reader = None
        self._aesthetic_model = None
        self._aesthetic_clip_model = None
        self._aesthetic_preprocess = None
        # Per-model locks prevent concurrent load/unload races
        self._clip_lock = threading.Lock()
        self._yolo_lock = threading.Lock()
        self._easyocr_lock = threading.Lock()
        self._aesthetic_lock = threading.Lock()
        logger.info("ModelCache initialized (device=%s)", DEVICE)

    @property
    def clip(self):
        """CLIP model + preprocess + tokenize. Returns (model, preprocess, tokenize)."""
        with self._clip_lock:
            if self._clip_model is None:
                import clip
                logger.info("Loading CLIP model (ViT-B/32) on %s...", DEVICE)
                self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
                self._clip_tokenizer = clip.tokenize
                logger.info("CLIP model loaded")
            return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    @property
    def yolo(self):
        """YOLOv8m model for person detection."""
        with self._yolo_lock:
            if self._yolo_model is None:
                from ultralytics import YOLO
                logger.info("Loading YOLOv8m model...")
                _model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'models', 'yolov8m.pt')
                self._yolo_model = YOLO(_model_path)
                logger.info("YOLOv8m model loaded")
            return self._yolo_model

    @property
    def easyocr_reader(self):
        """EasyOCR reader. Falls back to CPU if MPS fails."""
        with self._easyocr_lock:
            if self._easyocr_reader is None:
                import easyocr
                gpu = DEVICE in ("cuda", "mps")
                try:
                    logger.info("Loading EasyOCR (gpu=%s)...", gpu)
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=gpu)
                    logger.info("EasyOCR loaded (gpu=%s)", gpu)
                except Exception as exc:
                    logger.warning("EasyOCR failed on %s, falling back to CPU: %s", DEVICE, exc)
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    logger.info("EasyOCR loaded (CPU fallback)")
            return self._easyocr_reader

    @property
    def aesthetic(self):
        """LAION aesthetic scorer using ViT-B/32 (reuses CLIP model).

        Returns (linear_model, clip_model, preprocess) — same interface
        as before but uses the already-loaded ViT-B/32 instead of
        loading a separate ViT-L-14 (saves 924MB + 5s load time).
        """
        with self._aesthetic_lock:
            if self._aesthetic_model is None:
                import torch
                import torch.nn as nn

                logger.info("Loading LAION aesthetic predictor (ViT-B/32 linear)...")

                try:
                    # Reuse the CLIP ViT-B/32 model (loaded once, shared)
                    clip_model, preprocess, _ = self.clip

                    # Simple linear predictor: 512-dim → 1 score
                    aesthetic_model = nn.Linear(512, 1)
                    _weights_name = 'sa_0_4_vit_b_32_linear.pth'
                    # Search multiple paths: scraping/models/ (VPS), assets/models/ (local)
                    _candidates = [
                        os.path.join(os.path.dirname(__file__), '..', 'models', _weights_name),
                        os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'models', _weights_name),
                    ]
                    weights_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
                    if os.path.exists(weights_path):
                        state_dict = torch.load(weights_path, map_location=DEVICE)
                        aesthetic_model.load_state_dict(state_dict)
                        logger.info("Loaded B/32 aesthetic weights from %s", weights_path)
                    else:
                        logger.warning(
                            "Aesthetic weights not found at %s — "
                            "scores will be random. Download from LAION.",
                            weights_path
                        )
                    aesthetic_model = aesthetic_model.to(DEVICE).eval()

                    self._aesthetic_model = aesthetic_model
                    self._aesthetic_clip_model = clip_model
                    self._aesthetic_preprocess = preprocess
                    logger.info("LAION aesthetic predictor loaded (B/32, 2KB)")
                except Exception:
                    self._gc_cleanup()
                    raise

            return self._aesthetic_model, self._aesthetic_clip_model, self._aesthetic_preprocess

    def _gc_cleanup(self):
        """Force garbage collection and clear MPS/CUDA cache."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_yolo(self):
        """Unload YOLO model after person detection stage."""
        with self._yolo_lock:
            if self._yolo_model is not None:
                logger.info("Unloading YOLOv8m model")
                self._yolo_model = None
                self._gc_cleanup()

    def unload_aesthetic(self):
        """Unload aesthetic linear layer. Does NOT unload CLIP (shared)."""
        with self._aesthetic_lock:
            if self._aesthetic_model is not None:
                logger.info("Unloading aesthetic linear layer (CLIP stays loaded)")
                self._aesthetic_model = None
                # Don't unload _aesthetic_clip_model/_aesthetic_preprocess —
                # they're references to the shared CLIP model, not owned copies
                self._aesthetic_clip_model = None
                self._aesthetic_preprocess = None

    def unload_easyocr(self):
        """Unload EasyOCR reader after text/watermark stage."""
        with self._easyocr_lock:
            if self._easyocr_reader is not None:
                logger.info("Unloading EasyOCR reader")
                self._easyocr_reader = None
                self._gc_cleanup()

    def unload_clip(self):
        """Unload CLIP model after category balance (last use)."""
        with self._clip_lock:
            if self._clip_model is not None:
                logger.info("Unloading CLIP model")
                self._clip_model = None
                self._clip_preprocess = None
                self._clip_tokenizer = None
                self._gc_cleanup()

    def reset(self):
        """Clear all cached models (for testing)."""
        self.__init__()
        self._initialized = False
        ModelCache._instance = None


# Module-level convenience
models = ModelCache()
