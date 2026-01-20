import numpy as np
import time
import uuid
import cv2
from .interfaces import IFaceRecognizer, IStorage
from .line_crossing import LineCrossingTracker
from .clustering import ImprovedDeduplication, FaceClustering
from .face_logger import FaceLogger
from .face_clustering import FaceClusterManager
from config import Config

# Person detector for fallback when face is not visible
_person_detector = None

def get_person_detector(use_gpu=True):
    """Lazy load person detector"""
    global _person_detector
    
    # Check if person detection is enabled
    if not (Config.ENABLE_PERSON_DETECTION if hasattr(Config, 'ENABLE_PERSON_DETECTION') else True):
        return None
    
    if _person_detector is None:
        try:
            from .person_detector import PersonDetector
            confidence = Config.PERSON_DETECTION_CONFIDENCE if hasattr(Config, 'PERSON_DETECTION_CONFIDENCE') else 0.5
            _person_detector = PersonDetector(use_gpu=use_gpu, confidence=confidence)
        except Exception as e:
            print(f"[PersonDetector] Failed to initialize: {e}")
            _person_detector = False  # Mark as failed, don't retry
    return _person_detector if _person_detector else None

# Lazy import for tracking database to avoid circular dependency
_tracking_db = None
_events_db = None

def get_tracking_db():
    """Lazy load tracking database"""
    global _tracking_db
    if _tracking_db is None:
        from .tracking_db import TrackingDatabase
        _tracking_db = TrackingDatabase()
    return _tracking_db

def get_events_db():
    """Lazy load events database"""
    global _events_db
    if _events_db is None:
        from .events_db import EventsDB
        _events_db = EventsDB()
    return _events_db

class UniquePeopleCounter:
    def __init__(self, recognizer: IFaceRecognizer, storage: IStorage, camera_id=None):
        self.recognizer = recognizer
        self.storage = storage
        self.camera_id = camera_id
        # Cache stores: {'embedding': ndarray, 'last_seen': float, 'id': str, 'quality': float, 'count': int, 'gallery': [embeddings], 'last_bbox': bbox, 'centroid': (x, y)}
        self.active_faces = [] 
        
        # Camera type for entry/exit tracking
        self.camera_type = "entry"  # Can be "entry" or "exit"
        self.watchlist_threshold = 0.4  # Default 40% match threshold
        
        # Pending detections buffer for watchlist matching delay
        self.pending_detections = []  # List of (detection_time, face_data, embedding)
        self.watchlist_delay_seconds = 1.0  # Wait 1 second before finalizing detection
        
        # Person tracking (people without visible faces)
        self.active_persons = []  # Tracks people detected by body but no face
        self.person_detection_enabled = Config.ENABLE_PERSON_DETECTION if hasattr(Config, 'ENABLE_PERSON_DETECTION') else True
        
        # Face logging system (lazy initialized)
        self._logger = None
        
        # Face clustering system - groups faces of same person
        self.cluster_manager = FaceClusterManager(similarity_threshold=Config.SIMILARITY_THRESHOLD)
        
        # Line crossing tracker (disabled by default)
        self.line_crossing_enabled = False
        self.line_crossing_tracker = None
        self.line_crossing_count = 0  # Count only increments on line crossing
        
        # Multi-line tracker for named lines (A, B, etc.)
        self.multi_line_tracker = None
        
        # Face quality thresholds
        self.min_face_quality = Config.MIN_FACE_QUALITY
        self.blur_threshold = Config.BLUR_DETECTION_THRESHOLD
        
        # ROI and ROT (Region of Tracking) for area-based counting
        self.roi_points = []  # ROI as polygon (normalized 0-1 coords)
        self.rot_points = []  # ROT as polygon (normalized 0-1 coords)
        self.roi_enabled = False  # Only count if face is in ROI
        
        # Time window reset tracking
        self.time_window_seconds = 3600  # Default 1 hour
        self.window_start_time = time.time()
        self.last_reset_time = time.time()
        
        # Improved deduplication with multi-view gallery support
        self.deduplicator = ImprovedDeduplication(
            similarity_threshold=0.70,  # Lowered to 0.70 for video (was 0.75 for still images)
            min_face_quality=0.70  # Minimum detection confidence
        )
        self.clustering = FaceClustering(similarity_threshold=0.70)
        
        # Spatial tracking for centroid-based ID persistence
        self.prev_frame_boxes = {}  # {cam_frame_id: [bbox]} for continuity
        
        # Minimum detection confidence (0.0 to 1.0)
        self.min_face_score = 0.70  # Increased from 0.6

    @property
    def logger(self):
        """Lazy initialize logger on first access"""
        if self._logger is None:
            self._logger = FaceLogger()
        return self._logger

    def _log_face_detection(self, event_type, face_id, quality, confidence, details=None):
        """Log detailed face detection event"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_msg = f"[{timestamp}] {event_type} | Face ID: {face_id} | Quality: {quality:.2f} | Confidence: {confidence:.2f}"
        if details:
            log_msg += f" | {details}"
        print(log_msg)  # Console logging for debugging
        # Could also write to file or database for persistent logging

    def _log_matching_attempt(self, face_id, detected_embedding_norm, matched_face_id, similarity_score, matched=True):
        """Log face matching attempt"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        status = "MATCH" if matched else "NO_MATCH"
        log_msg = f"[{timestamp}] MATCH_ATTEMPT | Detected: {face_id} | vs Existing: {matched_face_id} | Similarity: {similarity_score:.3f} | Result: {status}"
        print(log_msg)

    def _calculate_blur(self, frame, bbox):
        """
        Calculate blur metric using Laplacian variance.
        Higher value = sharper, Lower value = blurrier
        Returns: blur_score (0-1 normalized)
        """
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            # Empirically for video face crops:
            # Sharp faces: variance ~50-200
            # Normal faces: variance ~20-50  
            # Blurry faces: variance <20
            # Use divisor of 50 to map: 50->1.0, 20->0.4, 5->0.1
            blur_score = min(1.0, max(0.0, laplacian_var / 50.0))
            
            return blur_score
        except Exception:
            return 0.5  # Default to middle value if calculation fails

    def _calculate_similarity(self, embedding1, embedding2):
        """Computes Cosine Similarity assuming inputs are normalized."""
        # If vectors are normalized, dot product suffices
        return float(np.dot(embedding1, embedding2))

    def _normalize_similarity_to_confidence(self, raw_similarity: float) -> float:
        """
        Convert raw cosine similarity to human-readable confidence percentage.
        
        InsightFace embeddings typically give:
        - Same person, same image: ~0.75-0.85
        - Same person, different images: ~0.55-0.75
        - Different people: ~0.20-0.45
        
        We map this to:
        - 0.75+ raw -> 95-100% confidence (definitely same person)
        - 0.55-0.75 raw -> 70-95% confidence (likely same person)
        - 0.40-0.55 raw -> 40-70% confidence (possibly same person)
        - below 0.40 raw -> 0-40% confidence (likely different)
        
        Using a piecewise linear mapping for intuitive results.
        """
        if raw_similarity >= 0.75:
            # High match: 0.75-1.0 -> 95-100%
            normalized = 0.95 + (raw_similarity - 0.75) * 0.2  # 0.75->95%, 1.0->100%
        elif raw_similarity >= 0.55:
            # Good match: 0.55-0.75 -> 70-95%
            normalized = 0.70 + (raw_similarity - 0.55) * 1.25  # 0.55->70%, 0.75->95%
        elif raw_similarity >= 0.40:
            # Possible match: 0.40-0.55 -> 40-70%
            normalized = 0.40 + (raw_similarity - 0.40) * 2.0  # 0.40->40%, 0.55->70%
        else:
            # Low match: 0.0-0.40 -> 0-40%
            normalized = raw_similarity  # Linear mapping for low scores
        
        return min(1.0, max(0.0, normalized))

    def _raw_threshold_from_confidence(self, confidence_percent: float) -> float:
        """
        Convert human-readable confidence percentage back to raw threshold.
        Inverse of _normalize_similarity_to_confidence.
        """
        if confidence_percent >= 0.95:
            return 0.75 + (confidence_percent - 0.95) / 0.2
        elif confidence_percent >= 0.70:
            return 0.55 + (confidence_percent - 0.70) / 1.25
        elif confidence_percent >= 0.40:
            return 0.40 + (confidence_percent - 0.40) / 2.0
        else:
            return confidence_percent

    def _iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x1,y1,x2,y2]"""
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0

        boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _dedupe_faces(self, faces):
        """Improved deduplication using NMS and quality filtering."""
        if not faces:
            return []
        
        # Step 1: Filter low-quality detections
        faces = self.deduplicator.filter_low_quality_faces(faces)
        if not faces:
            return []
        
        # Step 2: Remove overlapping bboxes (likely same face detected multiple times)
        faces = self.deduplicator.deduplicate_by_bbox(faces, iou_threshold=0.4)
        
        return faces

    def _cleanup_cache(self, time_window_seconds=None):
        """Removes faces that haven't been seen in the specified time window (defaults to Config.TIME_WINDOW_SECONDS)."""
        now = time.time()
        tw = time_window_seconds or Config.TIME_WINDOW_SECONDS
        self.active_faces = [
            f for f in self.active_faces 
            if (now - f['last_seen']) < tw
        ]
        # Cleanup line crossing history
        if self.line_crossing_tracker:
            active_ids = {f['id'] for f in self.active_faces}
            self.line_crossing_tracker.cleanup_old_faces(active_ids, now)

    def enable_line_crossing(self, frame_width=640, frame_height=480, line_position=0.5):
        """Enable line-crossing counting mode."""
        self.line_crossing_enabled = True
        self.line_crossing_tracker = LineCrossingTracker(frame_width, frame_height, line_position)
        self.line_crossing_count = 0

    def disable_line_crossing(self):
        """Disable line-crossing counting mode."""
        self.line_crossing_enabled = False
        self.line_crossing_tracker = None

    def set_line_position(self, position):
        """Adjust line position (0.0 to 1.0)."""
        if self.line_crossing_tracker:
            self.line_crossing_tracker.set_line_position(position)

    def _point_in_polygon(self, point, polygon):
        """Check if point (x, y) is inside polygon using ray casting algorithm.
        Handles both tuple format (x, y) and dict format {'x': x, 'y': y} for polygon points.
        """
        if not polygon or len(polygon) < 3:
            return True  # No valid polygon, consider point as inside
        
        x, y = point
        n = len(polygon)
        inside = False
        
        # Helper to extract coordinates from various formats
        def get_coords(p):
            if isinstance(p, dict):
                return (p.get('x', 0), p.get('y', 0))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                return (p[0], p[1])
            return (0, 0)
        
        p1x, p1y = get_coords(polygon[0])
        for i in range(1, n + 1):
            p2x, p2y = get_coords(polygon[i % n])
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_bbox_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _find_spatially_nearby_face(self, bbox, iou_threshold=0.3, time_threshold=2.0):
        """
        Find a face from previous frame that overlaps with current bbox (spatial continuity).
        Uses both IoU and centroid distance for more robust matching.
        """
        import time as time_module
        now = time_module.time()
        
        best_match = None
        best_score = 0
        
        for cached_face in self.active_faces:
            if 'last_bbox' not in cached_face:
                continue
            
            # Only consider recent detections
            last_seen = cached_face.get('last_seen', 0)
            if now - last_seen > time_threshold:
                continue
            
            # Calculate IoU
            iou = self._bbox_iou(bbox, cached_face['last_bbox'])
            
            # Also calculate centroid distance as backup
            curr_cx, curr_cy = self._get_bbox_centroid(bbox)
            cached_bbox = cached_face['last_bbox']
            cached_cx, cached_cy = self._get_bbox_centroid(cached_bbox)
            
            # Normalize distance by frame size (assume 640 width)
            distance = ((curr_cx - cached_cx) ** 2 + (curr_cy - cached_cy) ** 2) ** 0.5
            normalized_dist = distance / 640.0
            
            # Combined score: high IoU or low distance
            score = max(iou, 1.0 - min(normalized_dist * 5, 1.0))  # 5x weight for distance
            
            if score > best_score and score > iou_threshold:
                best_score = score
                best_match = cached_face
        
        return best_match

    def _is_frontal_face(self, face):
        """Check if face is frontal (not extreme side profile or up/down)"""
        # InsightFace provides pose as (yaw, pitch, roll)
        # Yaw: left-right rotation, Pitch: up-down, Roll: tilt
        pose = getattr(face, 'pose', [0, 0, 0])
        if pose is None or len(pose) < 3:
            return True  # Assume frontal if no pose info
        
        yaw, pitch, roll = pose[0], pose[1], pose[2]
        
        # Allow moderate head turns (yaw) but require reasonable pitch
        is_frontal = (
            abs(float(yaw)) < 40 and    # Yaw < 40 degrees (left/right)
            abs(float(pitch)) < 35 and  # Pitch < 35 degrees (up/down)
            abs(float(roll)) < 35       # Roll < 35 degrees (tilt)
        )
        return is_frontal

    def _get_face_size_info(self, bbox):
        """Calculate face size and categorize it"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Categorize based on area
        if area > 20000:
            category = "close"  # Large face, likely close to camera
        elif area > 5000:
            category = "medium"  # Medium distance
        else:
            category = "far"    # Small face, likely far from camera
        
        return {
            'width': float(width),
            'height': float(height),
            'area': float(area),
            'category': category
        }

    def set_roi(self, roi_points):
        """Set ROI as list of normalized points (0.0 to 1.0)."""
        self.roi_points = roi_points

    def set_rot(self, rot_points):
        """Set ROT (perimeter) as list of normalized points (0.0 to 1.0)."""
        self.rot_points = rot_points

    def _is_in_roi(self, bbox, frame_height, frame_width):
        """Check if face centroid is within ROI or ROT region."""
        if not hasattr(self, 'roi_points') and not hasattr(self, 'rot_points'):
            return True  # No ROI/ROT set, count all
        
        # Calculate face centroid in normalized coordinates
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / frame_width
        cy = (y1 + y2) / 2 / frame_height
        
        # Check ROI if set
        if hasattr(self, 'roi_points') and self.roi_points:
            if not self._point_in_polygon((cx, cy), self.roi_points):
                return False
        
        # Check ROT if set
        if hasattr(self, 'rot_points') and self.rot_points:
            if not self._point_in_polygon((cx, cy), self.rot_points):
                return False
        
        return True

    def set_time_window(self, time_window_seconds: int):
        """Set time window for unique counting and reset timer"""
        self.time_window_seconds = time_window_seconds
        
    def check_and_reset_window(self, clear_logs: bool = True):
        """Check if time window has expired and reset if needed"""
        now = time.time()
        elapsed = now - self.last_reset_time
        
        if elapsed >= self.time_window_seconds:
            # Archive stats to events DB before reset
            try:
                events_db = get_events_db()
                events_db.reset_camera_counts(self.camera_id)
                
                # Clear detection logs after timer expires
                if clear_logs:
                    cleared = events_db.clear_detection_logs(self.camera_id)
                    print(f"[Reset] Camera {self.camera_id} - Cleared {cleared} detection logs")
            except Exception as e:
                print(f"[Reset] Error archiving/clearing stats: {e}")
            
            # Reset counts
            self.last_reset_time = now
            self.window_start_time = now
            # Also reset person tracking
            self.active_persons = []
            print(f"[Reset] Camera {self.camera_id} - Time window reset after {elapsed:.0f}s")
            return True
        return False

    def _crop_frame_to_roi(self, frame):
        """
        If ROI is set, crop the frame to the ROI bounding box for faster detection.
        Returns: (cropped_frame, offset_x, offset_y, scale_x, scale_y) or (frame, 0, 0, 1, 1) if no ROI
        """
        if not self.roi_points or len(self.roi_points) < 3:
            return frame, 0, 0, 1.0, 1.0
        
        frame_height, frame_width = frame.shape[:2]
        
        # Get bounding box of ROI polygon
        xs = [p.get('x', p[0]) if isinstance(p, dict) else p[0] for p in self.roi_points]
        ys = [p.get('y', p[1]) if isinstance(p, dict) else p[1] for p in self.roi_points]
        
        # Convert from normalized to pixel coordinates
        min_x = int(min(xs) * frame_width)
        max_x = int(max(xs) * frame_width)
        min_y = int(min(ys) * frame_height)
        max_y = int(max(ys) * frame_height)
        
        # Add padding (10% of ROI size)
        pad_x = int((max_x - min_x) * 0.1)
        pad_y = int((max_y - min_y) * 0.1)
        
        # Clamp to frame bounds
        min_x = max(0, min_x - pad_x)
        max_x = min(frame_width, max_x + pad_x)
        min_y = max(0, min_y - pad_y)
        max_y = min(frame_height, max_y + pad_y)
        
        # Crop frame
        cropped = frame[min_y:max_y, min_x:max_x]
        
        return cropped, min_x, min_y, 1.0, 1.0

    def _adjust_bbox_for_crop(self, bbox, offset_x, offset_y):
        """Adjust bbox coordinates back to full frame coordinates after ROI crop."""
        x1, y1, x2, y2 = bbox
        return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]

    def _cleanup_persons(self, time_window_seconds=None):
        """Remove persons not seen recently."""
        now = time.time()
        tw = time_window_seconds or Config.TIME_WINDOW_SECONDS
        
    def _match_against_watchlist(self, embedding) -> tuple:
        """
        Match a face embedding against all watchlist entries.
        Returns: (matched_person_id, watchlist_name, raw_score, normalized_confidence) or (None, None, 0, 0)
        """
        watchlist_threshold = getattr(self, 'watchlist_threshold', 0.40)
        # Convert user's confidence threshold to raw similarity threshold
        # User sets 40% which means raw ~0.40, 60% means raw ~0.50, etc.
        raw_threshold = self._raw_threshold_from_confidence(watchlist_threshold)
        
        best_match = None
        best_name = None
        best_raw_score = 0
        best_confidence = 0
        
        try:
            from core.watchlist import WatchlistDB
            watchlist_db_local = WatchlistDB()
            watchlist_embeddings = watchlist_db_local.get_all_embeddings()
            
            for wl_person in watchlist_embeddings:
                wl_embedding = wl_person.get('embedding')
                if wl_embedding is None or len(wl_embedding) == 0:
                    continue
                
                # Normalize watchlist embedding
                wl_norm = np.linalg.norm(wl_embedding)
                if wl_norm > 0:
                    wl_embedding = wl_embedding / wl_norm
                
                raw_score = self._calculate_similarity(embedding, wl_embedding)
                
                if raw_score > best_raw_score and raw_score >= raw_threshold:
                    best_raw_score = raw_score
                    best_match = wl_person['person_id']
                    best_name = wl_person.get('name') or wl_person['person_id']
                    best_confidence = self._normalize_similarity_to_confidence(raw_score)
            
            watchlist_db_local.close()
            
            if best_match:
                print(f"[WATCHLIST] Matched '{best_name}' with {best_confidence*100:.1f}% confidence (raw: {best_raw_score:.3f})")
            
        except Exception as e:
            print(f"[Watchlist] Match error: {e}")
        
        return best_match, best_name, best_raw_score, best_confidence

    def _cleanup_persons_internal(self, time_window_seconds=None):
        """Remove persons not seen recently."""
        now = time.time()
        tw = time_window_seconds or Config.TIME_WINDOW_SECONDS
        self.active_persons = [
            p for p in self.active_persons 
            if (now - p.get('last_seen', 0)) < tw
        ]

    def process_frame(self, frame, time_window_seconds=None):
        # Check and reset time window if needed
        if time_window_seconds:
            self.time_window_seconds = time_window_seconds
        self.check_and_reset_window()
        
        frame_height, frame_width = frame.shape[:2]
        
        # Optionally crop to ROI for faster detection
        use_roi_crop = self.roi_points and len(self.roi_points) >= 3 and Config.USE_ROI_CROP if hasattr(Config, 'USE_ROI_CROP') else False
        
        if use_roi_crop:
            detection_frame, offset_x, offset_y, _, _ = self._crop_frame_to_roi(frame)
        else:
            detection_frame = frame
            offset_x, offset_y = 0, 0
        
        # 1. Detect faces
        faces = self.recognizer.get_faces(detection_frame)
        self._cleanup_cache(time_window_seconds=time_window_seconds)
        self._cleanup_persons(time_window_seconds=time_window_seconds)

        # 1.a Deduplicate overlapping detections in the same frame
        faces = self._dedupe_faces(faces)
        
        # 1.b Filter faces by ROI (only process faces inside ROI)
        if self.roi_points and len(self.roi_points) >= 3:
            roi_filtered_faces = []
            for face in faces:
                bbox = face.bbox
                if use_roi_crop:
                    # Adjust back to full frame
                    bbox = self._adjust_bbox_for_crop(bbox, offset_x, offset_y)
                    face.bbox = np.array(bbox)
                
                if self._is_in_roi(bbox, frame_height, frame_width):
                    roi_filtered_faces.append(face)
                else:
                    self._log_face_detection("FILTERED_ROI", f"face", 0, 0, "Outside ROI region")
            faces = roi_filtered_faces

        results = []
        now = time.time()
        matched_ids_in_frame = set()  # avoid multiple increments in same frame
        face_bboxes = []  # Track face locations for person detection

        for face_idx, face in enumerate(faces):
            face_bboxes.append(list(face.bbox))
            
            # FILTER: Ignore blurry or low-confidence detections
            det_score = getattr(face, 'det_score', 0.0)
            
            # Use dynamic quality threshold (read from Config so settings changes apply immediately)
            quality_threshold = Config.MIN_FACE_QUALITY
            
            if det_score < quality_threshold:
                # Skip low-quality/blurry faces
                self._log_face_detection("FILTERED_OUT", f"face_{face_idx}", det_score, det_score, 
                                        f"Below quality threshold ({quality_threshold:.2f})")
                continue
            
            # Check blur using Laplacian variance
            blur_score = self._calculate_blur(frame, face.bbox)
            blur_threshold = Config.BLUR_DETECTION_THRESHOLD
            
            if blur_score < blur_threshold:
                # Skip blurry faces
                self._log_face_detection("FILTERED_OUT", f"face_{face_idx}", det_score, blur_score,
                                        f"Too blurry (score: {blur_score:.3f}, threshold: {blur_threshold:.2f})")
                continue

            embedding = np.array(face.embedding, dtype=float)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding = embedding / norm  # normalize

            # Get face size info for logging
            size_info = self._get_face_size_info(face.bbox)
            pose = getattr(face, 'pose', [0, 0, 0])
            is_frontal = self._is_frontal_face(face)
            
            best_score = -1
            best_match = None

            # 2. SPATIAL CONTINUITY CHECK (Priority: If bbox overlaps with previous frame, keep same ID)
            spatial_match = self._find_spatially_nearby_face(face.bbox, iou_threshold=0.3)
            if spatial_match:
                # High confidence match based on spatial proximity
                best_match = spatial_match
                best_score = 0.76  # Set high score to force match
            else:
                # MATCHING LOGIC: Check against all poses in the person's gallery
                for cached_face in self.active_faces:
                    # Initialize gallery if it doesn't exist
                    if 'gallery' not in cached_face:
                        cached_face['gallery'] = [cached_face['embedding']]
                    
                    # Check the new face against every pose we've saved for this person
                    for stored_pose in cached_face['gallery']:
                        score = self._calculate_similarity(embedding, stored_pose)
                        # Log each matching attempt
                        if score > best_score:
                            self._log_matching_attempt(f"face_{face_idx}", embedding, cached_face['id'], score, matched=True)
                        if score > best_score:
                            best_score = score
                            best_match = cached_face

            # 3. DECISION LOGIC - Use dynamic similarity threshold from Config
            matching_threshold = Config.SIMILARITY_THRESHOLD  # Default 0.75, but can be adjusted
            if best_match and best_score > matching_threshold:
                self._log_face_detection("MATCHED", best_match['id'], det_score, best_score, 
                                        f"Similarity: {best_score:.3f}")
                matched_id = best_match['id']
                is_new = False
                
                # Check if this matched face is in watchlist (may have been added since first detection)
                wl_match, wl_name, wl_raw, wl_conf = self._match_against_watchlist(embedding)
                if wl_match and wl_name:
                    # Store watchlist info in the cached face record
                    best_match['watchlist_name'] = wl_name
                    best_match['watchlist_confidence'] = wl_conf

                prev_last_seen = best_match.get('last_seen', 0)
                # UPDATE: Rolling Average Embedding then normalize
                alpha = 0.25  # Increased from 0.2 for better adaptation
                new_emb = (1 - alpha) * best_match['embedding'] + alpha * embedding
                emb_norm = np.linalg.norm(new_emb)
                if emb_norm > 0:
                    best_match['embedding'] = new_emb / emb_norm
                best_match['last_seen'] = now
                best_match['last_bbox'] = face.bbox  # Store for spatial tracking
                best_match['centroid'] = self._get_bbox_centroid(face.bbox)
                
                # ADD TO GALLERY: If this is a significantly different pose (0.70-0.85 similarity)
                # Add it to the gallery so we recognize them from this angle next time
                if best_score < 0.85 and best_score > 0.70:
                    best_match['gallery'].append(embedding)
                    # Keep gallery small to maintain speed (max 5 poses)
                    if len(best_match['gallery']) > 5:
                        best_match['gallery'].pop(0)

                # Only increment if:
                # 1. Not already incremented in this frame
                # 2. Cooldown period has passed (prevent rapid re-counting)
                # 3. If line-crossing enabled, check if face crossed the line
                # 4. Face is within ROI/ROT if set
                time_since_last_count = now - prev_last_seen
                should_increment = (
                    matched_id not in matched_ids_in_frame and
                    time_since_last_count > Config.MIN_REPEAT_SECONDS and
                    self._is_in_roi(face.bbox, frame.shape[0], frame.shape[1])
                )

                # Check line crossing if enabled
                if should_increment and self.line_crossing_enabled and self.line_crossing_tracker:
                    crossed, direction = self.line_crossing_tracker.check_line_crossing(
                        matched_id, face.bbox, now
                    )
                    should_increment = crossed
                elif self.line_crossing_enabled and self.line_crossing_tracker:
                    # Even if not incrementing, track position
                    self.line_crossing_tracker.check_line_crossing(matched_id, face.bbox, now)

                if should_increment:
                    best_match['count'] = best_match.get('count', 1) + 1
                    if self.line_crossing_enabled:
                        self.line_crossing_count += 1
                    try:
                        self.storage.increment_face_count(matched_id)
                    except Exception:
                        pass
                    # Log count increment
                    self._log_face_detection("COUNT_INCREMENT", matched_id, best_match.get('quality', 0), 
                                            best_match.get('confidence', 0), 
                                            f"Total count: {best_match['count']}")
                    matched_ids_in_frame.add(matched_id)
                    
                    # Check if face is in ROI
                    in_roi = self._is_in_roi(face.bbox, frame.shape[0], frame.shape[1])
                    
                    # Save thumbnail and fullframe for detection log
                    thumbnail_path = None
                    fullframe_path = None
                    try:
                        thumbnail_path = self.storage.save_face_image(frame, face.bbox, matched_id)
                        fullframe_path = self.storage.save_fullframe_with_bbox(frame, face.bbox, matched_id)
                    except Exception as e:
                        print(f"[Storage] Error saving images: {e}")
                    
                    # EVENTS DB: Record detection event with images
                    if self.camera_id:
                        try:
                            camera_name = getattr(self, 'camera_name', f'Camera {self.camera_id}')
                            # Get watchlist info if available
                            wl_name = best_match.get('watchlist_name') if best_match else None
                            wl_conf = best_match.get('watchlist_confidence') if best_match else None
                            
                            events_db = get_events_db()
                            events_db.record_detection(
                                person_id=matched_id,
                                camera_id=self.camera_id,
                                camera_name=camera_name,
                                confidence=float(det_score),
                                bbox={'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]), 
                                     'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])},
                                in_roi=in_roi,
                                event_type='detection',
                                thumbnail_path=thumbnail_path,
                                fullframe_path=fullframe_path,
                                watchlist_name=wl_name,
                                watchlist_confidence=wl_conf
                            )
                        except Exception as e:
                            print(f"[Events] Error recording detection: {e}")
                    
                    # TRACKING: Update person location in tracking database
                    if self.camera_id:
                        try:
                            camera_name = getattr(self, 'camera_name', f'Camera {self.camera_id}')
                            tracking_db = get_tracking_db()
                            is_unique = tracking_db.add_or_update_person(
                                person_id=matched_id,
                                camera_id=self.camera_id,
                                camera_name=camera_name,
                                confidence=float(det_score),
                                bbox={'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]), 
                                     'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])}
                            )
                            
                            # Check if person is in watchlist
                            try:
                                from core.watchlist import WatchlistDB
                                watchlist_db = WatchlistDB()
                                if watchlist_db.is_in_watchlist(matched_id):
                                    watchlist_db.record_detection(
                                        person_id=matched_id, 
                                        camera_id=self.camera_id,
                                        camera_name=camera_name, 
                                        confidence=float(det_score),
                                        thumbnail_path=thumbnail_path,
                                        fullframe_path=fullframe_path,
                                        bbox={'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]), 
                                             'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])}
                                    )
                                    print(f"[WATCHLIST ALERT] {matched_id} detected at {camera_name}")
                                watchlist_db.close()
                            except Exception as e:
                                print(f"[Watchlist] Error: {e}")
                        except Exception as e:
                            print(f"[Tracking] Error updating person: {e}")
            else:
                # NEW VISITOR: Not found in the last hour
                # Only create a new person ID if the face is clear and looking at the camera (frontal)
                if not self._is_frontal_face(face):
                    # Skip creating a new ID from extreme side-view or odd angles
                    continue
                
                # Check if within ROI/ROT
                if not self._is_in_roi(face.bbox, frame.shape[0], frame.shape[1]):
                    # Face outside ROI/ROT, skip
                    continue
                
                # WATCHLIST MATCHING: Check if this face matches anyone in the watchlist
                watchlist_match, watchlist_name, watchlist_raw_score, watchlist_confidence = self._match_against_watchlist(embedding)
                
                # DEDUPLICATION CHECK: Before creating new ID, check if there's a recent detection
                # at the same location in the events DB
                existing_person = None
                if self.camera_id:
                    try:
                        events_db = get_events_db()
                        is_dup, existing_id = events_db._is_duplicate_detection(
                            self.camera_id, 
                            {'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]),
                             'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])},
                            time_window_seconds=3
                        )
                        if is_dup and existing_id:
                            existing_person = existing_id
                            print(f"[Dedup] Found existing person at same location: {existing_id}")
                    except Exception as e:
                        print(f"[Dedup] Error checking: {e}")
                
                if existing_person:
                    # Use existing person ID instead of creating new one
                    matched_id = existing_person
                    is_new = False
                    # Update the cached face if we have it
                    for cached in self.active_faces:
                        if cached['id'] == existing_person:
                            cached['last_seen'] = now
                            cached['last_bbox'] = face.bbox
                            break
                elif watchlist_match:
                    # Use watchlist person ID - this is a known person!
                    matched_id = watchlist_match
                    new_id = watchlist_match
                    is_new = True  # Still treat as new detection for this session
                    print(f"[WATCHLIST] Using watchlist ID: {watchlist_match} (Name: {watchlist_name})")
                else:
                    new_id = str(uuid.uuid4())[:8]
                    matched_id = new_id
                    is_new = True
                
                confidence = float(getattr(face, 'det_score', 0.0))
                # Use minimum of det_score and blur_score as quality metric
                quality = min(confidence, blur_score)
                
                # Log face detection
                if is_new:
                    self._log_face_detection("NEW_PERSON", matched_id, quality, confidence, 
                                            f"First detection (Cluster ID will be assigned)")
                else:
                    self._log_face_detection("DEDUPLICATED", matched_id, quality, confidence, 
                                            f"Matched to existing at same location")
                    # For deduplicated faces, skip creating new records
                    matched_ids_in_frame.add(matched_id)
                    results.append({
                        'bbox': list(face.bbox) if hasattr(face.bbox, '__iter__') else face.bbox,
                        'id': matched_id,
                        'is_new': False,
                        'score': best_score if best_score > 0 else confidence,
                        'count': 1,
                        'type': 'face'
                    })
                    continue

                # Save crop for probing/visual evidence
                img_path = self.storage.save_face_image(frame, face.bbox, matched_id)
                
                # Save fullframe with bbox for detection log
                fullframe_path = None
                try:
                    fullframe_path = self.storage.save_fullframe_with_bbox(frame, face.bbox, matched_id)
                except Exception as e:
                    print(f"[Storage] Error saving fullframe: {e}")
                
                # Add to clustering system
                cluster_id = self.cluster_manager.add_face_event(
                    face_id=matched_id,
                    embedding=embedding,
                    quality=quality,
                    confidence=confidence,
                    timestamp=now,
                    image_path=img_path
                )

                # Add to in-memory cache (embedding already normalized)
                new_record = {
                    'embedding': embedding,
                    'gallery': [embedding],  # Start gallery with this frontal pose
                    'last_seen': now,
                    'id': new_id,
                    'quality': quality,
                    'confidence': confidence,
                    'cluster_id': cluster_id,
                    'timestamp': now,
                    'count': 1,
                    'last_bbox': face.bbox,
                    'centroid': self._get_bbox_centroid(face.bbox),
                    'image_path': img_path
                }
                self.active_faces.append(new_record)
                matched_ids_in_frame.add(new_id)

                # TRACKING: Add new person to tracking database
                if self.camera_id:
                    try:
                        camera_name = getattr(self, 'camera_name', f'Camera {self.camera_id}')
                        camera_type = getattr(self, 'camera_type', 'entry')
                        
                        # Check if in ROI
                        in_roi = self._is_in_roi(face.bbox, frame.shape[0], frame.shape[1])
                        
                        # EVENTS DB: Record entry event for new person with embedding and watchlist info
                        events_db = get_events_db()
                        events_db.record_detection(
                            person_id=new_id,
                            camera_id=self.camera_id,
                            camera_name=camera_name,
                            confidence=confidence,
                            bbox={'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]), 
                                 'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])},
                            in_roi=in_roi,
                            event_type='first_detection',
                            thumbnail_path=img_path,
                            fullframe_path=fullframe_path,
                            embedding=embedding.tobytes(),  # Store embedding for watchlist matching
                            watchlist_name=watchlist_name,
                            watchlist_confidence=watchlist_confidence if watchlist_match else None
                        )
                        
                        # GLOBAL ENTRY/EXIT TRACKING based on camera type
                        if camera_type == 'entry':
                            events_db.record_global_entry(new_id, self.camera_id, camera_name)
                            print(f"[ENTRY] Person {watchlist_name or new_id} ENTERED at {camera_name}")
                        elif camera_type == 'exit':
                            events_db.record_global_exit(new_id, self.camera_id, camera_name)
                            print(f"[EXIT] Person {watchlist_name or new_id} EXITED at {camera_name}")
                        
                        # Tracking DB
                        tracking_db = get_tracking_db()
                        tracking_db.add_or_update_person(
                            person_id=new_id,
                            camera_id=self.camera_id,
                            camera_name=camera_name,
                            embedding=embedding.tobytes(),
                            confidence=confidence,
                            bbox={'x1': float(face.bbox[0]), 'y1': float(face.bbox[1]), 
                                 'x2': float(face.bbox[2]), 'y2': float(face.bbox[3])}
                        )
                    except Exception as e:
                        print(f"[Tracking] Error adding new person: {e}")

                # Persist to JSON DB
                db_rec = {
                    'embedding': embedding,
                    'timestamp': now,
                    'id': new_id,
                    'cluster_id': cluster_id,
                    'image_path': img_path,
                    'score': float(confidence),
                    'quality': quality,
                    'confidence': confidence,
                    'count': 1
                }
                if self.camera_id is not None:
                    db_rec['camera_id'] = self.camera_id
                self.storage.save_face(db_rec)
                matched_ids_in_frame.add(matched_id)

            # Get current count for the id (from cache if possible)
            current_count = None
            for c in self.active_faces:
                if c['id'] == matched_id:
                    current_count = c.get('count', 1)
                    break

            results.append({
                'bbox': list(face.bbox) if hasattr(face.bbox, '__iter__') else face.bbox,
                'id': matched_id,
                'is_new': is_new,
                'score': best_score if not is_new else 1.0,
                'count': current_count if current_count is not None else 1,
                'type': 'face'
            })
            
            # Update multi-line tracker with this detection
            if self.multi_line_tracker:
                try:
                    crossing_events = self.multi_line_tracker.update(
                        matched_id, 
                        list(face.bbox),
                        now
                    )
                    # Record crossing events to database
                    if crossing_events and self.camera_id:
                        for event in crossing_events:
                            try:
                                camera_name = getattr(self, 'camera_name', f'Camera {self.camera_id}')
                                events_db = get_events_db()
                                events_db.record_entry_exit(
                                    person_id=event['person_id'],
                                    camera_id=self.camera_id,
                                    camera_name=camera_name,
                                    event_type=event['direction'],
                                    direction=f"Line {event['line_name']}"
                                )
                                print(f"[LINE CROSSING] {event['person_id']} crossed line {event['line_name']} ({event['direction']})")
                            except Exception as e:
                                print(f"[LINE CROSSING] Error recording: {e}")
                except Exception as e:
                    print(f"[MultiLine] Error updating tracker: {e}")

        # PERSON DETECTION: Detect people without visible faces
        person_results = self._process_person_detections(frame, face_bboxes, now, matched_ids_in_frame)
        results.extend(person_results)

        # Total count includes both faces and persons without faces
        total_count = len(self.active_faces) + len(self.active_persons)
        return results, total_count

    def _process_person_detections(self, frame, face_bboxes, now, matched_ids_in_frame):
        """
        Detect people using YOLO and track those without visible faces.
        Returns list of person detection results.
        """
        results = []
        
        if not self.person_detection_enabled:
            return results
        
        person_detector = get_person_detector()
        if not person_detector or not person_detector.enabled:
            return results
        
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Detect persons with ROI filtering
            persons = person_detector.detect_persons(frame, self.roi_points if self.roi_points else None)
            
            for person in persons:
                person_bbox = person['bbox']
                person_centroid = person['centroid']
                
                # Check if this person already has a detected face
                has_face = False
                for face_bbox in face_bboxes:
                    if self._is_face_in_person_bbox(face_bbox, person_bbox):
                        has_face = True
                        break
                
                if has_face:
                    # Already counted via face detection
                    continue
                
                # This is a person without a visible face - track them
                person_id = self._find_or_create_person(person_bbox, person_centroid, now)
                
                if person_id and person_id not in matched_ids_in_frame:
                    matched_ids_in_frame.add(person_id)
                    
                    results.append({
                        'bbox': person_bbox,
                        'id': person_id,
                        'is_new': False,  # Will be set by _find_or_create_person
                        'score': person['confidence'],
                        'count': 1,
                        'type': 'person',  # Body detection, no face
                        'face_visible': False
                    })
                    
                    # Record in events DB
                    if self.camera_id:
                        try:
                            camera_name = getattr(self, 'camera_name', f'Camera {self.camera_id}')
                            events_db = get_events_db()
                            
                            # Save person crop (thumbnail)
                            thumbnail_path = None
                            fullframe_path = None
                            try:
                                thumbnail_path = self.storage.save_face_image(frame, np.array(person_bbox), person_id)
                                # Save fullframe with body bbox drawn
                                fullframe_path = self._save_fullframe_with_body_bbox(frame, person_bbox, person_id)
                            except Exception as e:
                                print(f"[PersonDetection] Error saving images: {e}")
                            
                            events_db.record_detection(
                                person_id=person_id,
                                camera_id=self.camera_id,
                                camera_name=camera_name,
                                confidence=float(person['confidence']),
                                bbox={'x1': person_bbox[0], 'y1': person_bbox[1], 
                                     'x2': person_bbox[2], 'y2': person_bbox[3]},
                                in_roi=True,
                                event_type='body_detection',  # Mark as body detection
                                thumbnail_path=thumbnail_path,
                                fullframe_path=fullframe_path
                            )
                            
                            # Also update tracking DB for person without face
                            try:
                                tracking_db = get_tracking_db()
                                tracking_db.add_or_update_person(
                                    person_id=person_id,
                                    camera_id=self.camera_id,
                                    camera_name=camera_name,
                                    confidence=float(person['confidence']),
                                    bbox={'x1': person_bbox[0], 'y1': person_bbox[1], 
                                         'x2': person_bbox[2], 'y2': person_bbox[3]}
                                )
                            except:
                                pass
                        except Exception as e:
                            print(f"[PersonDetection] Error recording: {e}")
        except Exception as e:
            print(f"[PersonDetection] Error: {e}")
        
        return results
    
    def _is_face_in_person_bbox(self, face_bbox, person_bbox):
        """Check if a face detection is within a person detection."""
        fx1, fy1, fx2, fy2 = face_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Face center
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        
        # Check if face center is within person bbox (with some margin)
        margin = 0.1 * (px2 - px1)  # 10% margin
        return (px1 - margin <= face_cx <= px2 + margin and 
                py1 <= face_cy <= py2 + (py2 - py1) * 0.3)  # Face should be in upper 30%
    
    def _find_or_create_person(self, bbox, centroid, now):
        """
        Find existing person by spatial proximity or create new one.
        """
        # Try to match with existing tracked person by IoU
        best_match = None
        best_iou = 0.3  # Minimum IoU threshold
        
        for person in self.active_persons:
            if 'last_bbox' in person:
                iou = self._bbox_iou(bbox, person['last_bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = person
        
        if best_match:
            # Update existing person
            best_match['last_seen'] = now
            best_match['last_bbox'] = bbox
            best_match['centroid'] = centroid
            best_match['count'] = best_match.get('count', 1) + 1
            return best_match['id']
        
        # Create new person entry
        new_id = f"person_{str(uuid.uuid4())[:6]}"
        new_person = {
            'id': new_id,
            'last_seen': now,
            'last_bbox': bbox,
            'centroid': centroid,
            'count': 1,
            'first_seen': now,
            'type': 'person'
        }
        self.active_persons.append(new_person)
        
        print(f"[PersonDetection] New person without face: {new_id}")
        return new_id

    def _save_fullframe_with_body_bbox(self, frame, bbox, person_id):
        """Save full frame with body bounding box drawn (for person detection without face)"""
        if not Config.SAVE_IMAGES:
            return None
        
        import os
        filename = f"{Config.IMAGE_OUTPUT_DIR}/body_{person_id}_{int(time.time())}.jpg"
        
        # Make a copy to avoid modifying the original frame
        frame_copy = frame.copy()
        
        # Draw body bounding box in orange (to distinguish from green face bbox)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange color
        
        # Add label indicating body detection
        label = f"Body: {person_id[:8]} (No Face)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(frame_copy, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 165, 255), -1)
        
        # Draw text
        cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imwrite(filename, frame_copy)
        return filename

    def get_cluster_summary(self):
        """
        Get clustering summary of current active faces.
        Returns groups of likely same people based on similarity.
        """
        if not self.active_faces:
            return []

        # Prepare faces for clustering
        faces_for_clustering = []
        for face in self.active_faces:
            faces_for_clustering.append({
                'id': face['id'],
                'embedding': face['embedding'],
                'bbox': None,
                'score': face.get('quality', 0.9),
                'det_score': face.get('quality', 0.9)
            })

        # Perform clustering
        clusters = self.clustering.cluster_faces(faces_for_clustering)

        # Convert to summary format
        summary = []
        for cluster in clusters:
            summary.append({
                'representative_id': cluster['representative_id'],
                'members': cluster['member_ids'],
                'cluster_size': cluster['size'],
                'avg_confidence': cluster['avg_score']
            })

        return summary
    def get_detailed_faces(self):
        """
        Get detailed information about all detected faces with cluster info.
        Returns list of face events with timestamps, quality, confidence, cluster_id.
        """
        detailed_faces = []
        clusters = self.cluster_manager.get_all_clusters()
        
        for cluster_id, face_events in clusters.items():
            for face_event in face_events:
                detailed_faces.append({
                    'face_id': face_event['face_id'],
                    'cluster_id': cluster_id,
                    'quality': face_event['quality'],
                    'confidence': face_event['confidence'],
                    'timestamp': face_event['timestamp'],
                    'image_path': face_event['image_path']
                })
        
        return detailed_faces
    
    def get_unique_people_count(self):
        """Get count of unique people (clusters) instead of unique face detections"""
        # Periodically merge similar clusters to catch same person split into multiple clusters
        self.cluster_manager.merge_similar_clusters(merge_threshold=Config.DEDUP_SIMILARITY_THRESHOLD)
        return self.cluster_manager.get_unique_count()