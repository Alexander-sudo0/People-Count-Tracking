import numpy as np
from config import Config


class LineCrossingTracker:
    """Track face line crossings to count people entering/exiting a region."""

    def __init__(self, frame_width=640, frame_height=480, line_position=0.5):
        """
        Initialize tracker with frame dimensions and line position.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            line_position: Line position as fraction (0.0 to 1.0) of frame width (0=left, 1=right)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.line_position = line_position  # Fraction of width
        self.line_x = int(frame_width * line_position)
        
        # Entry/exit counts
        self.entry_count = 0  # Left to right crossings
        self.exit_count = 0   # Right to left crossings
        
        # Track face positions: {face_id: {'prev_x': float, 'crossed': bool, 'last_cross_time': float}}
        self.face_history = {}
        
    def get_line_x(self):
        """Get the x-coordinate of the crossing line."""
        return self.line_x
    
    def set_line_position(self, position):
        """Update line position (0.0 to 1.0 fraction of frame width)."""
        self.line_position = max(0, min(1, position))
        self.line_x = int(self.frame_width * self.line_position)
    
    def reset_counts(self):
        """Reset entry and exit counts."""
        self.entry_count = 0
        self.exit_count = 0
    
    def get_face_centroid(self, bbox):
        """Calculate centroid (x, y) of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def check_line_crossing(self, face_id, current_bbox, current_time):
        """
        Check if a face has crossed the line.
        
        Returns:
            (crossed: bool, direction: str)  # direction: 'left_to_right' or 'right_to_left'
        """
        current_x, _ = self.get_face_centroid(current_bbox)
        
        if face_id not in self.face_history:
            # First detection of this face
            self.face_history[face_id] = {
                'prev_x': current_x,
                'crossed': False,
                'last_cross_time': 0
            }
            return False, None
        
        history = self.face_history[face_id]
        prev_x = history['prev_x']
        
        # Check if line was crossed
        crossed_line = False
        direction = None
        
        if prev_x < self.line_x <= current_x:
            # Crossed from left to right (entry)
            crossed_line = True
            direction = 'left_to_right'
            self.entry_count += 1
        elif prev_x > self.line_x >= current_x:
            # Crossed from right to left (exit)
            crossed_line = True
            direction = 'right_to_left'
            self.exit_count += 1
        
        # Update history
        history['prev_x'] = current_x
        history['last_cross_time'] = current_time
        
        return crossed_line, direction
    
    def cleanup_old_faces(self, active_face_ids, current_time):
        """Remove faces from history that are no longer active."""
        # Keep only faces that are still being tracked
        self.face_history = {
            fid: hist for fid, hist in self.face_history.items()
            if fid in active_face_ids
        }
