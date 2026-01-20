import uuid
from .recognizer import InsightFaceRecognizer
from .counter import UniquePeopleCounter
from .storage import JSONStorage
from .database import Database
from utils.stream_loader import RTSPStreamLoader

class CameraManager:
    def __init__(self, use_gpu=False):
        self.cameras = {} # { camera_id: { 'loader': ..., 'counter': ... } }
        self.use_gpu = use_gpu
        # Initialize shared resources (Recognizer/Storage) to save memory
        self.storage = JSONStorage()
        self.recognizer = InsightFaceRecognizer(use_gpu=use_gpu)
        # Initialize persistent database for camera configs and face detections
        self.db = Database()
        
        # Load persisted camera configurations
        self._load_persisted_cameras()
    
    def _load_persisted_cameras(self):
        """Load previously saved camera configurations from database"""
        persisted_cameras = self.db.get_all_cameras()
        for cam_id, cam_config in persisted_cameras.items():
            try:
                src_val = int(cam_config['source']) if str(cam_config['source']).isdigit() else cam_config['source']
                loader = RTSPStreamLoader(src=src_val, process_fps=cam_config.get('fps', 5))
                counter = UniquePeopleCounter(self.recognizer, self.storage, camera_id=cam_id)
                
                # Restore ROI and ROT if they exist
                roi_points = cam_config.get('roi_points', [])
                rot_points = cam_config.get('rot_points', [])
                if roi_points:
                    counter.set_roi(roi_points)
                if rot_points:
                    counter.set_rot(rot_points)
                
                # Set camera name on counter for proper logging
                counter.camera_name = cam_config.get('name', f"Camera {len(self.cameras) + 1}")
                
                # Set camera type for entry/exit tracking
                counter.camera_type = cam_config.get('camera_type', 'entry')
                
                # Set watchlist threshold
                counter.watchlist_threshold = cam_config.get('watchlist_threshold', 0.4)
                
                # Restore line crossing settings
                if cam_config.get('line_crossing_enabled'):
                    counter.enable_line_crossing(line_position=cam_config.get('line_position', 0.5))
                
                self.cameras[cam_id] = {
                    "source": cam_config['source'],
                    "loader": loader,
                    "counter": counter,
                    "name": cam_config.get('name', f"Camera {len(self.cameras) + 1}"),
                    "process_fps": cam_config.get('fps', 5),
                    "counting_enabled": cam_config.get('counting_enabled', True),
                    "time_window_seconds": cam_config.get('time_window_seconds'),
                    "line_crossing_enabled": cam_config.get('line_crossing_enabled', False),
                    "line_position": cam_config.get('line_position', 0.5),
                    "roi_points": roi_points,
                    "rot_points": rot_points,
                    "camera_type": cam_config.get('camera_type', 'entry'),
                    "watchlist_threshold": cam_config.get('watchlist_threshold', 0.4)
                }
                print(f"[CameraManager] Restored camera {cam_id}: {cam_config.get('name')}")
            except Exception as e:
                print(f"[CameraManager] Error loading camera {cam_id}: {e}")

    def add_camera(self, source, name=None, process_fps=5, counting_enabled=True, time_window_seconds=None, camera_type="entry", watchlist_threshold=0.4):
        cam_id = str(uuid.uuid4())[:8]
        
        # Create dedicated loader for this stream
        # Handle numeric source (webcam) vs string (RTSP)
        src_val = int(source) if str(source).isdigit() else source
        loader = RTSPStreamLoader(src=src_val, process_fps=process_fps)
        
        # Create dedicated counter logic (camera-aware)
        counter = UniquePeopleCounter(self.recognizer, self.storage, camera_id=cam_id)
        counter.camera_name = name or f"Camera {len(self.cameras) + 1}"  # Add camera name
        counter.camera_type = camera_type  # Set camera type for entry/exit tracking
        counter.watchlist_threshold = watchlist_threshold  # Set watchlist matching threshold
        
        camera_config = {
            "source": source,
            "loader": loader,
            "counter": counter,
            "name": name or f"Camera {len(self.cameras) + 1}",
            "process_fps": process_fps,
            "counting_enabled": counting_enabled,
            "time_window_seconds": time_window_seconds or 3600,
            "line_crossing_enabled": False,
            "line_position": 0.5,
            "roi_points": [],
            "rot_points": [],
            "camera_type": camera_type,
            "watchlist_threshold": watchlist_threshold
        }
        
        self.cameras[cam_id] = camera_config
        
        # Persist camera configuration to database
        self.db.save_camera(cam_id, {
            'name': camera_config['name'],
            'source': source,
            'fps': process_fps,
            'counting_enabled': counting_enabled,
            'time_window_seconds': time_window_seconds or 3600,
            'line_crossing_enabled': False,
            'line_position': 0.5,
            'roi_points': [],
            'rot_points': [],
            'camera_type': camera_type,
            'watchlist_threshold': watchlist_threshold
        })
        
        return cam_id

    def update_camera(self, cam_id, name=None, counting_enabled=None):
        if cam_id not in self.cameras:
            return False
        if name is not None:
            self.cameras[cam_id]['name'] = name
        if counting_enabled is not None:
            self.cameras[cam_id]['counting_enabled'] = bool(counting_enabled)
        return True

    def set_frequency(self, cam_id, process_fps):
        """Update process FPS for a camera without restarting the loader thread."""
        if cam_id not in self.cameras:
            return False
        cam = self.cameras[cam_id]
        cam['process_fps'] = process_fps
        cam['loader'].process_fps = process_fps
        cam['loader'].interval = 1.0 / process_fps if process_fps > 0 else cam['loader'].interval
        return True

    def set_time_window(self, cam_id, time_window_seconds):
        """Set the time window (seconds) for uniqueness counting for a camera."""
        if cam_id not in self.cameras:
            return False
        self.cameras[cam_id]['time_window_seconds'] = int(time_window_seconds)
        return True

    def set_line_crossing(self, cam_id, enabled, line_position=None):
        """Enable/disable line crossing detection for a camera."""
        if cam_id not in self.cameras:
            return False
        cam = self.cameras[cam_id]
        cam['line_crossing_enabled'] = bool(enabled)
        
        if enabled:
            # Initialize line crossing on counter
            counter = cam['counter']
            # Get frame dimensions from loader if available
            cap = cam['loader'].capture
            if cap and cap.isOpened():
                w = int(cap.get(3))
                h = int(cap.get(4))
            else:
                w, h = 640, 480
            
            lp = line_position if line_position is not None else cam.get('line_position', 0.5)
            counter.enable_line_crossing(w, h, lp)
            cam['line_position'] = lp
        else:
            cam['counter'].disable_line_crossing()
        
        return True

    def set_line_position(self, cam_id, line_position):
        """Set line crossing position (0.0 to 1.0)."""
        if cam_id not in self.cameras:
            return False
        cam = self.cameras[cam_id]
        cam['line_position'] = max(0, min(1, line_position))
        if cam.get('line_crossing_enabled'):
            cam['counter'].set_line_position(cam['line_position'])
        return True


    def update_camera(self, cam_id, name=None, counting_enabled=None):
        if cam_id not in self.cameras:
            return False
        if name is not None:
            self.cameras[cam_id]['name'] = name
        if counting_enabled is not None:
            self.cameras[cam_id]['counting_enabled'] = bool(counting_enabled)
        return True

    def set_frequency(self, cam_id, process_fps):
        """Update process FPS for a camera without restarting the loader thread."""
        if cam_id not in self.cameras:
            return False
        cam = self.cameras[cam_id]
        cam['process_fps'] = process_fps
        cam['loader'].process_fps = process_fps
        cam['loader'].interval = 1.0 / process_fps if process_fps > 0 else cam['loader'].interval
        return True

    def get_frame(self, cam_id):
        if cam_id not in self.cameras:
            # Return empty defaults if camera doesn't exist
            return None, [], 0
            
        cam = self.cameras[cam_id]
        frame = cam['loader'].get_frame()
        
        results = []
        count = len(cam['counter'].active_faces) # Get current count from cache
        
        if frame is not None and cam.get('counting_enabled', True):
            # Run the counting logic using camera's time window (fallback to global)
            tw = cam.get('time_window_seconds') or None
            results, count = cam['counter'].process_frame(frame, time_window_seconds=tw)
            
        return frame, results, count

    def get_active_cameras(self):
        return [{
            "id": k,
            "name": v['name'],
            "source": v['source'],
            "process_fps": v.get('process_fps', 5),
            "counting_enabled": v.get('counting_enabled', True),
            "time_window_seconds": v.get('time_window_seconds')
        } for k, v in self.cameras.items()]

    def set_roi(self, cam_id, roi_points):
        """Set ROI (Region of Interest) for a camera - as polygon with normalized coords."""
        if cam_id not in self.cameras:
            return False
        
        cam = self.cameras[cam_id]
        cam['roi_points'] = roi_points
        cam['counter'].set_roi(roi_points)
        
        # Persist to database
        self.db.save_camera(cam_id, {
            'name': cam['name'],
            'source': cam['source'],
            'fps': cam['process_fps'],
            'counting_enabled': cam['counting_enabled'],
            'time_window_seconds': cam['time_window_seconds'],
            'line_crossing_enabled': cam.get('line_crossing_enabled', False),
            'line_position': cam.get('line_position', 0.5),
            'roi_points': roi_points,
            'rot_points': cam.get('rot_points', [])
        })
        
        return True

    def set_rot(self, cam_id, rot_points):
        """Set ROT (Region of Tracking/perimeter) for a camera - as polygon with normalized coords."""
        if cam_id not in self.cameras:
            return False
        
        cam = self.cameras[cam_id]
        cam['rot_points'] = rot_points
        cam['counter'].set_rot(rot_points)
        
        # Persist to database
        self.db.save_camera(cam_id, {
            'name': cam['name'],
            'source': cam['source'],
            'fps': cam['process_fps'],
            'counting_enabled': cam['counting_enabled'],
            'time_window_seconds': cam['time_window_seconds'],
            'line_crossing_enabled': cam.get('line_crossing_enabled', False),
            'line_position': cam.get('line_position', 0.5),
            'roi_points': cam.get('roi_points', []),
            'rot_points': rot_points
        })
        
        return True

    def delete_camera(self, cam_id):
        """Delete a camera and stop its stream"""
        if cam_id not in self.cameras:
            return False
        
        try:
            # Stop the stream loader
            cam = self.cameras[cam_id]
            if 'loader' in cam and cam['loader']:
                cam['loader'].stop()
            
            # Remove from active cameras
            del self.cameras[cam_id]
            
            # Delete from database
            self.db.delete_camera(cam_id)
            
            return True
        except Exception as e:
            print(f"Error deleting camera {cam_id}: {e}")
            return False

    def get_camera(self, cam_id):
        """Get details of a specific camera"""
        if cam_id not in self.cameras:
            return None
        
        cam = self.cameras[cam_id]
        return {
            'id': cam_id,
            'name': cam.get('name'),
            'source': cam.get('source'),
            'process_fps': cam.get('process_fps'),
            'counting_enabled': cam.get('counting_enabled'),
            'time_window_seconds': cam.get('time_window_seconds'),
            'line_crossing_enabled': cam.get('line_crossing_enabled', False),
            'line_position': cam.get('line_position', 0.5),
            'roi_points': cam.get('roi_points', []),
            'rot_points': cam.get('rot_points', [])
        }

    def update_camera_name(self, cam_id, new_name):
        """Update camera name"""
        if cam_id not in self.cameras:
            return False
        
        cam = self.cameras[cam_id]
        cam['name'] = new_name
        
        # Persist to database
        self.db.save_camera(cam_id, {
            'name': new_name,
            'source': cam['source'],
            'fps': cam['process_fps'],
            'counting_enabled': cam['counting_enabled'],
            'time_window_seconds': cam['time_window_seconds'],
            'line_crossing_enabled': cam.get('line_crossing_enabled', False),
            'line_position': cam.get('line_position', 0.5),
            'roi_points': cam.get('roi_points', []),
            'rot_points': cam.get('rot_points', [])
        })
        
        return True
