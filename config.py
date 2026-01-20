import os
import json
from pathlib import Path

# Get the absolute path to the project directory
BASE_DIR = Path(__file__).resolve().parent

class Config:
    # Face Analysis Settings (defaults)
    DETECTION_SIZE = (640, 640)
    SIMILARITY_THRESHOLD = 0.65  # Real-time matching threshold (lowered to catch more matches in video)
    DEDUP_SIMILARITY_THRESHOLD = 0.60  # Post-processing dedup threshold (lower = more aggressive merging)
    MIN_FACE_QUALITY = 0.70      # Minimum face detection confidence (increased from implicit 0.6)
    BLUR_DETECTION_THRESHOLD = 0.30  # Blur score threshold (0.30 = reasonably sharp, 0.15 = very lenient)
    TIME_WINDOW_SECONDS = 3600  # 1 Hour

    # Deduplication / Stability
    IOU_DEDUP_THRESHOLD = 0.4   # IoU threshold to merge overlapping detections (stricter: was 0.3)
    MIN_REPEAT_SECONDS = 30     # Minimum seconds between counting the same person again
    USE_POST_PROCESSING = True  # Enable post-processing deduplication using clustering

    # Person Detection (YOLO) - Fallback when face is not visible
    ENABLE_PERSON_DETECTION = True   # Enable YOLO person detection for people without visible faces
    PERSON_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for person detection
    USE_ROI_CROP = False  # Crop frame to ROI before detection (faster but may miss edge detections)

    # Face Bounding Box Padding (for extraction)
    # Expands the detected bbox to include more context (ears, forehead, chin)
    BBOX_PADDING_TOP = 0.30     # Padding above face (% of face height)
    BBOX_PADDING_BOTTOM = 0.25  # Padding below face (% of face height)
    BBOX_PADDING_LEFT = 0.20    # Padding left side (% of face width)
    BBOX_PADDING_RIGHT = 0.20   # Padding right side (% of face width)

    # Storage (using absolute paths)
    DB_PATH = str(BASE_DIR / "faces_db.json")   # Simple JSON DB for this example
    SAVE_IMAGES = True          # Save face crops for probing?
    IMAGE_OUTPUT_DIR = str(BASE_DIR / "detected_faces")

    # Settings file (persistable)
    SETTINGS_PATH = str(BASE_DIR / "app_settings.json")

    @staticmethod
    def setup_dirs():
        if Config.SAVE_IMAGES and not os.path.exists(Config.IMAGE_OUTPUT_DIR):
            os.makedirs(Config.IMAGE_OUTPUT_DIR)

    @staticmethod
    def expand_bbox(bbox, frame_h, frame_w):
        """
        Expand a bounding box with padding to include more context.
        
        Args:
            bbox: [x1, y1, x2, y2] coordinates
            frame_h: Frame height
            frame_w: Frame width
            
        Returns:
            Expanded bbox [x1, y1, x2, y2]
        """
        import numpy as np
        
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Calculate face dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Apply padding
        x1 -= int(face_w * Config.BBOX_PADDING_LEFT)
        x2 += int(face_w * Config.BBOX_PADDING_RIGHT)
        y1 -= int(face_h * Config.BBOX_PADDING_TOP)
        y2 += int(face_h * Config.BBOX_PADDING_BOTTOM)
        
        # Clip to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)
        
        return np.array([x1, y1, x2, y2], dtype=bbox.dtype)

    @staticmethod
    def load_settings():
        """Load persisted settings from disk and apply to Config class attributes."""
        if not os.path.exists(Config.SETTINGS_PATH):
            return
        try:
            with open(Config.SETTINGS_PATH, 'r') as f:
                data = json.load(f)
            # Apply known settings if present
            Config.SIMILARITY_THRESHOLD = float(data.get('SIMILARITY_THRESHOLD', Config.SIMILARITY_THRESHOLD))
            Config.DEDUP_SIMILARITY_THRESHOLD = float(data.get('DEDUP_SIMILARITY_THRESHOLD', Config.DEDUP_SIMILARITY_THRESHOLD))
            Config.IOU_DEDUP_THRESHOLD = float(data.get('IOU_DEDUP_THRESHOLD', Config.IOU_DEDUP_THRESHOLD))
            Config.MIN_REPEAT_SECONDS = float(data.get('MIN_REPEAT_SECONDS', Config.MIN_REPEAT_SECONDS))
            Config.TIME_WINDOW_SECONDS = int(data.get('TIME_WINDOW_SECONDS', Config.TIME_WINDOW_SECONDS))
            Config.MIN_FACE_QUALITY = float(data.get('MIN_FACE_QUALITY', Config.MIN_FACE_QUALITY))
            Config.BLUR_DETECTION_THRESHOLD = float(data.get('BLUR_DETECTION_THRESHOLD', Config.BLUR_DETECTION_THRESHOLD))
            Config.BBOX_PADDING_TOP = float(data.get('BBOX_PADDING_TOP', Config.BBOX_PADDING_TOP))
            Config.BBOX_PADDING_BOTTOM = float(data.get('BBOX_PADDING_BOTTOM', Config.BBOX_PADDING_BOTTOM))
            Config.BBOX_PADDING_LEFT = float(data.get('BBOX_PADDING_LEFT', Config.BBOX_PADDING_LEFT))
            Config.BBOX_PADDING_RIGHT = float(data.get('BBOX_PADDING_RIGHT', Config.BBOX_PADDING_RIGHT))
        except Exception:
            # If reading fails, ignore and continue with defaults
            pass

    @staticmethod
    def save_settings(new_settings: dict):
        """Persist a subset of configuration options to SETTINGS_PATH."""
        out = {
            'SIMILARITY_THRESHOLD': Config.SIMILARITY_THRESHOLD,
            'DEDUP_SIMILARITY_THRESHOLD': Config.DEDUP_SIMILARITY_THRESHOLD,
            'IOU_DEDUP_THRESHOLD': Config.IOU_DEDUP_THRESHOLD,
            'MIN_REPEAT_SECONDS': Config.MIN_REPEAT_SECONDS,
            'TIME_WINDOW_SECONDS': Config.TIME_WINDOW_SECONDS,
            'MIN_FACE_QUALITY': Config.MIN_FACE_QUALITY,
            'BLUR_DETECTION_THRESHOLD': Config.BLUR_DETECTION_THRESHOLD,
            'BBOX_PADDING_TOP': Config.BBOX_PADDING_TOP,
            'BBOX_PADDING_BOTTOM': Config.BBOX_PADDING_BOTTOM,
            'BBOX_PADDING_LEFT': Config.BBOX_PADDING_LEFT,
            'BBOX_PADDING_RIGHT': Config.BBOX_PADDING_RIGHT
        }
        out.update({k: v for k, v in new_settings.items() if k in out})
        try:
            with open(Config.SETTINGS_PATH, 'w') as f:
                json.dump(out, f)
            # Reload to apply
            Config.load_settings()
            return True
        except Exception:
            return False