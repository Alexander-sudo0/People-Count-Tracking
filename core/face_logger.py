"""
Detailed logging system for face detection and identification.
Tracks every decision made by the counter for debugging.
"""
import json
import os
from datetime import datetime
from collections import deque


class FaceLogger:
    """Logs all face detection and identification events"""
    
    def __init__(self, max_logs=1000):
        self.logs = deque(maxlen=max_logs)  # Keep last 1000 events
        self.max_logs = max_logs
        self.log_file = "face_detection.log"
    
    def log_detection(self, event_data):
        """Log a face detection/identification event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            **event_data
        }
        self.logs.append(event)
        
        # Also write to file for persistence
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"[FaceLogger] Error writing to log file: {e}")
    
    def log_face_found(self, face_id, det_score, pose, bbox, size_info, is_frontal, reason=""):
        """Log when a face is detected"""
        self.log_detection({
            'event': 'face_detected',
            'face_id': face_id,
            'detection_score': float(det_score),
            'pose': [float(p) for p in pose] if pose else None,
            'is_frontal': is_frontal,
            'bbox': [float(x) for x in bbox] if bbox else None,
            'face_size': size_info,  # {'width': w, 'height': h, 'area': a, 'category': 'close/medium/far'}
            'reason': reason
        })
    
    def log_match(self, face_id, matched_person_id, similarity_score, gallery_size, method="embedding"):
        """Log when a face is matched to an existing person"""
        self.log_detection({
            'event': 'face_matched',
            'face_id': face_id,
            'matched_person_id': matched_person_id,
            'similarity_score': float(similarity_score),
            'gallery_size': gallery_size,
            'match_method': method,  # 'spatial', 'embedding', 'gallery'
            'result': 'success'
        })
    
    def log_new_person(self, face_id, det_score, pose, reason=""):
        """Log when a new person ID is created"""
        self.log_detection({
            'event': 'new_person_created',
            'face_id': face_id,
            'detection_score': float(det_score),
            'pose': [float(p) for p in pose] if pose else None,
            'reason': reason
        })
    
    def log_no_match(self, face_index, best_score, threshold, reason=""):
        """Log when a face doesn't match anyone"""
        self.log_detection({
            'event': 'no_match_found',
            'face_index': face_index,
            'best_similarity_score': float(best_score),
            'threshold': threshold,
            'reason': reason
        })
    
    def log_filtered_face(self, reason):
        """Log when a face is filtered out (skipped)"""
        self.log_detection({
            'event': 'face_filtered',
            'reason': reason
        })
    
    def log_increment(self, face_id, person_id, reason=""):
        """Log when a person count is incremented"""
        self.log_detection({
            'event': 'count_incremented',
            'face_id': face_id,
            'person_id': person_id,
            'reason': reason
        })
    
    def get_logs(self, event_type=None, limit=100):
        """Get logs, optionally filtered by event type"""
        logs_list = list(self.logs)
        if event_type:
            logs_list = [log for log in logs_list if log.get('event') == event_type]
        
        return logs_list[-limit:]  # Return last N logs
    
    def get_stats(self):
        """Get statistics about logged events"""
        logs_list = list(self.logs)
        stats = {
            'total_events': len(logs_list),
            'events_by_type': {},
            'recent_faces': [],
            'distinct_people': set()
        }
        
        for log in logs_list:
            event_type = log.get('event', 'unknown')
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
            
            if log.get('matched_person_id'):
                stats['distinct_people'].add(log['matched_person_id'])
            if log.get('face_id'):
                stats['recent_faces'].append(log['face_id'])
        
        stats['distinct_people'] = list(stats['distinct_people'])
        stats['recent_faces'] = list(set(stats['recent_faces']))[-20:]  # Last 20 unique faces
        
        return stats
    
    def clear_logs(self):
        """Clear all logs"""
        self.logs.clear()
        try:
            os.remove(self.log_file)
        except:
            pass
