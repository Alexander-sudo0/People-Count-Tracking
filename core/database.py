"""
Database module for persistent storage of cameras, ROI/perimeter configs, and face detections with timestamps.
Uses JSON for simplicity but structured for scalability to SQL if needed.
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import threading

class Database:
    """Persistent database for camera configs and face detections"""
    
    def __init__(self, db_dir="db"):
        self.db_dir = db_dir
        self.cameras_file = os.path.join(db_dir, "cameras.json")
        self.faces_file = os.path.join(db_dir, "faces.json")
        self.lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(db_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        if not os.path.exists(self.cameras_file):
            self._save_cameras({})
        if not os.path.exists(self.faces_file):
            self._save_faces([])
    
    # ==================== CAMERA MANAGEMENT ====================
    
    def _load_cameras(self) -> Dict[str, Dict]:
        """Load all camera configurations"""
        try:
            with open(self.cameras_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Database] Error loading cameras: {e}")
            return {}
    
    def _save_cameras(self, cameras: Dict):
        """Save all camera configurations"""
        try:
            with self.lock:
                with open(self.cameras_file, 'w') as f:
                    json.dump(cameras, f, indent=2)
        except Exception as e:
            print(f"[Database] Error saving cameras: {e}")
    
    def save_camera(self, camera_id: str, config: Dict):
        """Save or update camera configuration with ROI/ROT"""
        cameras = self._load_cameras()
        cameras[camera_id] = {
            'id': camera_id,
            'name': config.get('name', f'Camera {camera_id}'),
            'source': config.get('source', '0'),
            'fps': config.get('fps', 5),
            'time_window_seconds': config.get('time_window_seconds', 3600),
            'counting_enabled': config.get('counting_enabled', True),
            'line_crossing_enabled': config.get('line_crossing_enabled', False),
            'line_position': config.get('line_position', 0.5),
            'roi_points': config.get('roi_points', []),  # Perimeter polygon (normalized coords)
            'rot_points': config.get('rot_points', []),  # Region of tracking polygon
            'created_at': config.get('created_at', datetime.now().isoformat()),
            'updated_at': datetime.now().isoformat()
        }
        self._save_cameras(cameras)
    
    def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera configuration"""
        cameras = self._load_cameras()
        return cameras.get(camera_id)
    
    def get_all_cameras(self) -> Dict[str, Dict]:
        """Get all camera configurations"""
        return self._load_cameras()
    
    def delete_camera(self, camera_id: str):
        """Delete camera configuration"""
        cameras = self._load_cameras()
        if camera_id in cameras:
            del cameras[camera_id]
            self._save_cameras(cameras)
    
    # ==================== FACE DETECTION MANAGEMENT ====================
    
    def _load_faces(self) -> List[Dict]:
        """Load all face detections"""
        try:
            with open(self.faces_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Database] Error loading faces: {e}")
            return []
    
    def _save_faces(self, faces: List[Dict]):
        """Save all face detections"""
        try:
            with self.lock:
                with open(self.faces_file, 'w') as f:
                    json.dump(faces, f, indent=2)
        except Exception as e:
            print(f"[Database] Error saving faces: {e}")
    
    def save_face_detection(self, face_data: Dict) -> str:
        """
        Save a face detection with timestamp.
        Returns face_id.
        
        Args:
            face_data: {
                'camera_id': str,
                'face_id': str (unique identifier for this face),
                'embedding': list (converted from numpy array),
                'bbox': [x1, y1, x2, y2],
                'score': float (detection confidence),
                'image_path': str (path to cropped face image),
                'count': int (number of times seen)
            }
        """
        faces = self._load_faces()
        
        detection_record = {
            'id': f"{face_data.get('camera_id', 'unknown')}__{face_data.get('face_id', 'unknown')}",
            'camera_id': face_data.get('camera_id', 'unknown'),
            'face_id': face_data.get('face_id'),
            'embedding': face_data.get('embedding'),
            'bbox': face_data.get('bbox'),
            'score': face_data.get('score', 0.0),
            'image_path': face_data.get('image_path'),
            'count': face_data.get('count', 1),
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'detections': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'bbox': face_data.get('bbox'),
                    'score': face_data.get('score', 0.0)
                }
            ]
        }
        
        # Check if face already exists (same camera + same face_id)
        existing_idx = None
        for idx, face in enumerate(faces):
            if (face.get('camera_id') == face_data.get('camera_id') and 
                face.get('face_id') == face_data.get('face_id')):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # Update existing face
            faces[existing_idx]['count'] += 1
            faces[existing_idx]['last_seen'] = datetime.now().isoformat()
            faces[existing_idx]['detections'].append(detection_record['detections'][0])
        else:
            # Add new face
            faces.append(detection_record)
        
        self._save_faces(faces)
        return detection_record['id']
    
    def get_faces_by_camera(self, camera_id: str, time_window_seconds: Optional[int] = None) -> List[Dict]:
        """
        Get all faces detected by a camera, optionally filtered by time window.
        
        Args:
            camera_id: Camera identifier
            time_window_seconds: Optional time window (e.g., 3600 for last hour)
        
        Returns:
            List of face records with detections
        """
        faces = self._load_faces()
        camera_faces = [f for f in faces if f.get('camera_id') == camera_id]
        
        if time_window_seconds:
            cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
            filtered = []
            for face in camera_faces:
                last_seen = datetime.fromisoformat(face.get('last_seen', datetime.now().isoformat()))
                if last_seen >= cutoff_time:
                    filtered.append(face)
            return filtered
        
        return camera_faces
    
    def get_face_timeline(self, camera_id: str, face_id: str) -> Optional[Dict]:
        """
        Get complete detection timeline for a specific face in a camera.
        
        Args:
            camera_id: Camera identifier
            face_id: Face identifier
        
        Returns:
            Face record with all detection timestamps and locations
        """
        faces = self._load_faces()
        for face in faces:
            if face.get('camera_id') == camera_id and face.get('face_id') == face_id:
                return face
        return None
    
    def get_faces_in_time_range(self, camera_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Get all unique faces detected in a specific time range for a camera.
        
        Args:
            camera_id: Camera identifier
            start_time: Start datetime
            end_time: End datetime
        
        Returns:
            List of face records active in that time range
        """
        faces = self._load_faces()
        camera_faces = [f for f in faces if f.get('camera_id') == camera_id]
        
        filtered = []
        for face in camera_faces:
            last_seen = datetime.fromisoformat(face.get('last_seen', datetime.now().isoformat()))
            first_seen = datetime.fromisoformat(face.get('first_seen', datetime.now().isoformat()))
            
            # Check if face was active in the time range
            if (first_seen <= end_time and last_seen >= start_time):
                filtered.append(face)
        
        return filtered
    
    def increment_face_count(self, camera_id: str, face_id: str, bbox: Optional[List] = None):
        """
        Increment detection count for a face and log the detection.
        
        Args:
            camera_id: Camera identifier
            face_id: Face identifier
            bbox: Optional bounding box of current detection
        """
        faces = self._load_faces()
        
        for face in faces:
            if face.get('camera_id') == camera_id and face.get('face_id') == face_id:
                face['count'] = face.get('count', 1) + 1
                face['last_seen'] = datetime.now().isoformat()
                
                # Log detection timestamp
                detection = {
                    'timestamp': datetime.now().isoformat(),
                    'bbox': bbox,
                    'score': 0.0
                }
                if 'detections' not in face:
                    face['detections'] = []
                face['detections'].append(detection)
                
                self._save_faces(faces)
                return
        
        print(f"[Database] Face not found: camera={camera_id}, face_id={face_id}")
    
    def get_face_statistics(self, camera_id: str) -> Dict:
        """
        Get statistics about faces detected by a camera.
        
        Returns:
            {
                'total_unique_faces': int,
                'total_detections': int,
                'busiest_hour': str,
                'most_frequent_face': dict,
                'detection_timeline': list of (timestamp, count) tuples
            }
        """
        faces = self.get_faces_by_camera(camera_id)
        
        total_unique = len(faces)
        total_detections = sum(f.get('count', 0) for f in faces)
        
        # Find most frequent face
        most_frequent = max(faces, key=lambda f: f.get('count', 0)) if faces else None
        
        # Build hourly statistics
        hourly_stats = {}
        for face in faces:
            for detection in face.get('detections', []):
                try:
                    ts = datetime.fromisoformat(detection.get('timestamp'))
                    hour_key = ts.strftime('%Y-%m-%d %H:00')
                    hourly_stats[hour_key] = hourly_stats.get(hour_key, 0) + 1
                except:
                    pass
        
        busiest_hour = max(hourly_stats, key=hourly_stats.get) if hourly_stats else None
        
        return {
            'total_unique_faces': total_unique,
            'total_detections': total_detections,
            'busiest_hour': busiest_hour,
            'busiest_hour_count': hourly_stats.get(busiest_hour, 0) if busiest_hour else 0,
            'most_frequent_face': most_frequent,
            'hourly_stats': hourly_stats
        }
    
    # ==================== CLEANUP & MAINTENANCE ====================
    
    def cleanup_old_faces(self, camera_id: str, days: int = 30):
        """
        Remove faces older than specified days from the database.
        
        Args:
            camera_id: Camera identifier (or None for all cameras)
            days: Number of days to keep
        """
        faces = self._load_faces()
        cutoff_time = datetime.now() - timedelta(days=days)
        
        filtered_faces = []
        removed_count = 0
        
        for face in faces:
            if camera_id and face.get('camera_id') != camera_id:
                filtered_faces.append(face)
                continue
            
            last_seen = datetime.fromisoformat(face.get('last_seen', datetime.now().isoformat()))
            if last_seen >= cutoff_time:
                filtered_faces.append(face)
            else:
                removed_count += 1
        
        if removed_count > 0:
            self._save_faces(filtered_faces)
            print(f"[Database] Cleaned up {removed_count} old face records for camera {camera_id}")
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        cameras = self._load_cameras()
        faces = self._load_faces()
        
        total_detections = sum(f.get('count', 0) for f in faces)
        
        camera_stats = {}
        for cam_id in cameras:
            cam_faces = [f for f in faces if f.get('camera_id') == cam_id]
            camera_stats[cam_id] = {
                'unique_faces': len(cam_faces),
                'total_detections': sum(f.get('count', 0) for f in cam_faces)
            }
        
        return {
            'total_cameras': len(cameras),
            'total_unique_faces': len(faces),
            'total_detections': total_detections,
            'camera_stats': camera_stats,
            'db_files': {
                'cameras': os.path.getsize(self.cameras_file) if os.path.exists(self.cameras_file) else 0,
                'faces': os.path.getsize(self.faces_file) if os.path.exists(self.faces_file) else 0
            }
        }
