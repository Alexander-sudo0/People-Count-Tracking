"""
Person Detector using YOLO for detecting people when face is not visible.
Works alongside face detection to catch people looking down, away, or with covered faces.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

# Try to import ultralytics (YOLO)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[PersonDetector] YOLO not available. Install with: pip install ultralytics")


class PersonDetector:
    """
    Detects people using YOLO body detection.
    Used as fallback when face detection fails (person looking away, face covered, etc.)
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5, use_gpu: bool = True):
        """
        Initialize the person detector.
        
        Args:
            model_name: YOLO model to use (yolov8n.pt is fastest, yolov8s.pt is more accurate)
            confidence: Minimum confidence threshold for detection
            use_gpu: Whether to use GPU acceleration
        """
        self.confidence = confidence
        self.model = None
        self._enabled = False
        
        if not YOLO_AVAILABLE:
            print("[PersonDetector] YOLO not available, person detection disabled")
            return
        
        try:
            self.model = YOLO(model_name)
            # Set device
            if use_gpu:
                self.model.to('cuda')
            else:
                self.model.to('cpu')
            self._enabled = True
            print(f"[PersonDetector] Initialized with {model_name}")
        except Exception as e:
            print(f"[PersonDetector] Failed to load model: {e}")
            self._enabled = False
    
    @property
    def enabled(self) -> bool:
        return self._enabled and self.model is not None
    
    def detect_persons(self, frame: np.ndarray, roi_points: Optional[List] = None) -> List[Dict]:
        """
        Detect people in the frame.
        
        Args:
            frame: BGR image from OpenCV
            roi_points: Optional list of normalized ROI polygon points
            
        Returns:
            List of detected persons with bbox, confidence, centroid
        """
        if not self.enabled:
            return []
        
        try:
            # Run detection
            results = self.model(frame, conf=self.confidence, classes=[0], verbose=False)  # class 0 is 'person'
            
            detections = []
            frame_height, frame_width = frame.shape[:2]
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Calculate centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Check if in ROI
                    in_roi = True
                    if roi_points and len(roi_points) >= 3:
                        # Convert to normalized coordinates
                        cx_norm = cx / frame_width
                        cy_norm = cy / frame_height
                        in_roi = self._point_in_polygon((cx_norm, cy_norm), roi_points)
                    
                    if not in_roi:
                        continue
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'centroid': (float(cx), float(cy)),
                        'area': float((x2 - x1) * (y2 - y1)),
                        'type': 'person'
                    })
            
            return detections
            
        except Exception as e:
            print(f"[PersonDetector] Detection error: {e}")
            return []
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Dict]) -> bool:
        """Check if point is inside polygon using ray casting."""
        x, y = point
        n = len(polygon)
        inside = False
        
        # Handle both dict format {'x': 0.5, 'y': 0.5} and tuple format (0.5, 0.5)
        def get_point(p):
            if isinstance(p, dict):
                return p.get('x', 0), p.get('y', 0)
            return p
        
        p1x, p1y = get_point(polygon[0])
        for i in range(1, n + 1):
            p2x, p2y = get_point(polygon[i % n])
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def get_head_region(self, person_bbox: List[float]) -> List[float]:
        """
        Estimate head region from person bounding box.
        Useful for face detection crop.
        
        Args:
            person_bbox: [x1, y1, x2, y2] of person
            
        Returns:
            [x1, y1, x2, y2] of estimated head region
        """
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1
        
        # Head is typically in the top 20-25% of the body bbox
        head_y1 = y1
        head_y2 = y1 + height * 0.25
        
        # Center horizontally with some padding
        head_x1 = x1 + width * 0.2
        head_x2 = x2 - width * 0.2
        
        return [head_x1, head_y1, head_x2, head_y2]


class HybridDetector:
    """
    Combines face detection (InsightFace) with person detection (YOLO).
    Ensures detection works even when face is not visible.
    """
    
    def __init__(self, face_recognizer, use_gpu: bool = True, enable_person_detection: bool = True):
        """
        Args:
            face_recognizer: InsightFaceRecognizer instance
            use_gpu: Use GPU for YOLO
            enable_person_detection: Enable/disable person detection fallback
        """
        self.face_recognizer = face_recognizer
        self.person_detector = None
        self.enable_person_detection = enable_person_detection
        
        if enable_person_detection:
            self.person_detector = PersonDetector(use_gpu=use_gpu)
    
    def get_detections(self, frame: np.ndarray, roi_points: Optional[List] = None) -> Dict:
        """
        Get both face and person detections.
        
        Returns:
            {
                'faces': [...],  # InsightFace face objects
                'persons': [...],  # YOLO person detections without matched face
                'face_count': int,
                'person_only_count': int  # Persons detected but no face
            }
        """
        result = {
            'faces': [],
            'persons': [],
            'face_count': 0,
            'person_only_count': 0
        }
        
        frame_height, frame_width = frame.shape[:2]
        
        # 1. Get face detections
        faces = self.face_recognizer.get_faces(frame)
        
        # Filter faces by ROI if set
        if roi_points and len(roi_points) >= 3:
            filtered_faces = []
            for face in faces:
                cx = (face.bbox[0] + face.bbox[2]) / 2 / frame_width
                cy = (face.bbox[1] + face.bbox[3]) / 2 / frame_height
                if self._point_in_polygon((cx, cy), roi_points):
                    filtered_faces.append(face)
            faces = filtered_faces
        
        result['faces'] = faces
        result['face_count'] = len(faces)
        
        # 2. Get person detections (if enabled)
        if self.person_detector and self.person_detector.enabled:
            persons = self.person_detector.detect_persons(frame, roi_points)
            
            # Find persons without matched faces
            face_bboxes = [list(f.bbox) for f in faces]
            unmatched_persons = []
            
            for person in persons:
                has_face = False
                person_bbox = person['bbox']
                
                # Check if any face is within this person's bbox
                for face_bbox in face_bboxes:
                    if self._is_face_in_person(face_bbox, person_bbox):
                        has_face = True
                        break
                
                if not has_face:
                    unmatched_persons.append(person)
            
            result['persons'] = unmatched_persons
            result['person_only_count'] = len(unmatched_persons)
        
        return result
    
    def _is_face_in_person(self, face_bbox: List[float], person_bbox: List[float]) -> bool:
        """Check if face bbox is inside or overlaps significantly with person bbox."""
        fx1, fy1, fx2, fy2 = face_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Check if face center is within person bbox
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        
        if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
            return True
        
        # Also check IoU > 0.3
        inter_x1 = max(fx1, px1)
        inter_y1 = max(fy1, py1)
        inter_x2 = min(fx2, px2)
        inter_y2 = min(fy2, py2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            face_area = (fx2 - fx1) * (fy2 - fy1)
            if face_area > 0 and inter_area / face_area > 0.3:
                return True
        
        return False
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List) -> bool:
        """Check if point is inside polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        
        def get_point(p):
            if isinstance(p, dict):
                return p.get('x', 0), p.get('y', 0)
            return p
        
        p1x, p1y = get_point(polygon[0])
        for i in range(1, n + 1):
            p2x, p2y = get_point(polygon[i % n])
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
