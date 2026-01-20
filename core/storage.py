import json
import time
import os
import cv2
import numpy as np
from .interfaces import IStorage
from config import Config

class JSONStorage(IStorage):
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []

    def save_face(self, face_data):
        # face_data dict: {embedding, timestamp, id, image_path, camera_id}
        # Ensure count exists and embeddings are JSON serializable
        record = face_data.copy()
        record.setdefault('count', 1)
        if isinstance(record.get('embedding'), np.ndarray):
            record['embedding'] = record['embedding'].tolist()

        self.data.append(record)
        # In production, use SQLite. For PoC, dumping JSON is fine but slow at scale.
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f)

    def increment_face_count(self, face_id):
        """Increment the occurrence count for a face and persist to disk"""
        for rec in self.data:
            if rec.get('id') == face_id:
                rec['count'] = rec.get('count', 1) + 1
                # update timestamp to last seen
                rec['timestamp'] = time.time()
                with open(self.db_path, 'w') as f:
                    json.dump(self.data, f)
                return True
        return False

    def load_recent_faces(self, time_window_seconds):
        current_time = time.time()
        # Filter data in memory
        recent = [
            d for d in self.data 
            if (current_time - d['timestamp']) < time_window_seconds
        ]
        return recent

    def load_recent_faces_by_camera(self, camera_id, time_window_seconds):
        """Return recent faces for a specific camera_id"""
        current_time = time.time()
        recent = [
            d for d in self.data
            if d.get('camera_id') == camera_id and (current_time - d['timestamp']) < time_window_seconds
        ]
        return recent
    
    def save_face_image(self, frame, bbox, face_id):
        if not Config.SAVE_IMAGES:
            return None
        
        filename = f"{Config.IMAGE_OUTPUT_DIR}/face_{face_id}_{int(time.time())}.jpg"
        
        # Expand bbox with padding to include ears, forehead, etc.
        h, w, _ = frame.shape
        expanded_bbox = Config.expand_bbox(bbox, h, w)
        
        # Extract face region with padding
        x1, y1, x2, y2 = expanded_bbox.astype(int)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            cv2.imwrite(filename, face_img)
            return filename
        return None
    
    def save_fullframe_with_bbox(self, frame, bbox, face_id):
        """Save full frame with bounding box drawn around the face"""
        if not Config.SAVE_IMAGES:
            return None
        
        filename = f"{Config.IMAGE_OUTPUT_DIR}/fullframe_{face_id}_{int(time.time())}.jpg"
        
        # Make a copy to avoid modifying the original frame
        frame_copy = frame.copy()
        
        # Draw bounding box on the frame
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with face ID
        label = f"ID: {face_id[:8]}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(frame_copy, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), (0, 255, 0), -1)
        cv2.putText(frame_copy, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Save the frame
        cv2.imwrite(filename, frame_copy)
        return filename
    
    def clear_faces(self, camera_id=None):
        """Clear face data for a specific camera or all cameras"""
        if camera_id:
            # Clear faces for specific camera only
            self.data = [d for d in self.data if d.get('camera_id') != camera_id]
        else:
            # Clear all faces
            self.data = []
        
        # Persist to disk
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f)