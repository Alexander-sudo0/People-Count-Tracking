"""
Enhanced Line Crossing Tracker with named lines (A, B, etc.)
Supports multiple lines, directional counting, and polygon-based lines
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class CrossingLine:
    """Represents a crossing line with start and end points"""
    name: str  # Line name (A, B, C, etc.)
    start: Tuple[float, float]  # Normalized (x, y) start point
    end: Tuple[float, float]  # Normalized (x, y) end point
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR color
    direction_positive: str = "in"  # Direction name when crossing from left/top
    direction_negative: str = "out"  # Direction name when crossing from right/bottom
    entry_count: int = 0
    exit_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'start': self.start,
            'end': self.end,
            'color': list(self.color),
            'direction_positive': self.direction_positive,
            'direction_negative': self.direction_negative,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'CrossingLine':
        return CrossingLine(
            name=data['name'],
            start=tuple(data['start']),
            end=tuple(data['end']),
            color=tuple(data.get('color', [0, 255, 0])),
            direction_positive=data.get('direction_positive', 'in'),
            direction_negative=data.get('direction_negative', 'out'),
            entry_count=data.get('entry_count', 0),
            exit_count=data.get('exit_count', 0)
        )


class MultiLineCrossingTracker:
    """
    Track crossings of multiple named lines.
    Supports entry/exit counting with directional awareness.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Named lines: {line_name: CrossingLine}
        self.lines: Dict[str, CrossingLine] = {}
        
        # Track person positions: {person_id: {'prev_position': (x, y), 'crossings': [...]}}
        self.person_history: Dict[str, Dict] = {}
        
        # Recent crossing events
        self.crossing_events: List[Dict] = []
        
    def add_line(self, name: str, start: Tuple[float, float], end: Tuple[float, float],
                 direction_positive: str = "in", direction_negative: str = "out",
                 color: Tuple[int, int, int] = None) -> CrossingLine:
        """
        Add a named crossing line.
        
        Args:
            name: Line identifier (A, B, Entry, Exit, etc.)
            start: (x, y) normalized start point (0.0 to 1.0)
            end: (x, y) normalized end point (0.0 to 1.0)
            direction_positive: Name for crossing in positive direction
            direction_negative: Name for crossing in negative direction
            color: BGR color for visualization
        """
        if color is None:
            # Generate color based on line index
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]
            color = colors[len(self.lines) % len(colors)]
        
        line = CrossingLine(
            name=name,
            start=start,
            end=end,
            color=color,
            direction_positive=direction_positive,
            direction_negative=direction_negative
        )
        self.lines[name] = line
        return line
    
    def remove_line(self, name: str) -> bool:
        """Remove a named line"""
        if name in self.lines:
            del self.lines[name]
            return True
        return False
    
    def clear_lines(self):
        """Remove all lines"""
        self.lines.clear()
    
    def get_lines(self) -> List[Dict]:
        """Get all lines as dictionaries"""
        return [line.to_dict() for line in self.lines.values()]
    
    def set_lines(self, lines_data: List[Dict]):
        """Set lines from dictionary data"""
        self.clear_lines()
        for line_data in lines_data:
            line = CrossingLine.from_dict(line_data)
            self.lines[line.name] = line
    
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Get centroid of bounding box in pixel coordinates"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _get_normalized_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """Get normalized centroid (0-1 range)"""
        cx, cy = self._get_centroid(bbox)
        return (cx / self.frame_width, cy / self.frame_height)
    
    def _cross_product_2d(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """2D cross product (determinant)"""
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    def _check_line_crossing(self, prev_pos: Tuple[float, float], 
                              curr_pos: Tuple[float, float],
                              line: CrossingLine) -> Optional[str]:
        """
        Check if movement from prev_pos to curr_pos crosses the line.
        
        Returns:
            Direction string if crossed, None if not crossed
        """
        # Line segment points
        p1 = line.start
        p2 = line.end
        
        # Movement segment
        p3 = prev_pos
        p4 = curr_pos
        
        # Calculate cross products to determine if segments intersect
        d1 = self._cross_product_2d(
            (p2[0] - p1[0], p2[1] - p1[1]),
            (p3[0] - p1[0], p3[1] - p1[1])
        )
        d2 = self._cross_product_2d(
            (p2[0] - p1[0], p2[1] - p1[1]),
            (p4[0] - p1[0], p4[1] - p1[1])
        )
        d3 = self._cross_product_2d(
            (p4[0] - p3[0], p4[1] - p3[1]),
            (p1[0] - p3[0], p1[1] - p3[1])
        )
        d4 = self._cross_product_2d(
            (p4[0] - p3[0], p4[1] - p3[1]),
            (p2[0] - p3[0], p2[1] - p3[1])
        )
        
        # Check if segments intersect
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            # Line was crossed - determine direction
            # Positive = crossed from "left" of line to "right"
            if d1 > 0:
                return line.direction_positive
            else:
                return line.direction_negative
        
        return None
    
    def update(self, person_id: str, bbox: List[float], current_time: float = None) -> List[Dict]:
        """
        Update tracking for a person and check for line crossings.
        
        Args:
            person_id: Unique identifier for the person
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            current_time: Timestamp (uses time.time() if not provided)
            
        Returns:
            List of crossing events that occurred
        """
        if current_time is None:
            current_time = time.time()
        
        current_pos = self._get_normalized_centroid(bbox)
        events = []
        
        if person_id in self.person_history:
            prev_pos = self.person_history[person_id]['prev_position']
            
            # Check each line for crossings
            for line_name, line in self.lines.items():
                direction = self._check_line_crossing(prev_pos, current_pos, line)
                
                if direction:
                    # Line was crossed!
                    event = {
                        'person_id': person_id,
                        'line_name': line_name,
                        'direction': direction,
                        'timestamp': current_time,
                        'position': current_pos
                    }
                    events.append(event)
                    self.crossing_events.append(event)
                    
                    # Update line counters
                    if direction == line.direction_positive:
                        line.entry_count += 1
                    else:
                        line.exit_count += 1
                    
                    # Keep crossing history for this person
                    self.person_history[person_id]['crossings'].append(event)
        
        # Update person position
        self.person_history[person_id] = {
            'prev_position': current_pos,
            'last_seen': current_time,
            'crossings': self.person_history.get(person_id, {}).get('crossings', [])
        }
        
        return events
    
    def get_line_counts(self) -> Dict[str, Dict]:
        """Get entry/exit counts for all lines"""
        return {
            name: {
                'name': name,
                'entry_count': line.entry_count,
                'exit_count': line.exit_count,
                'net_count': line.entry_count - line.exit_count
            }
            for name, line in self.lines.items()
        }
    
    def get_recent_crossings(self, minutes: int = 5) -> List[Dict]:
        """Get crossing events from the last N minutes"""
        cutoff = time.time() - (minutes * 60)
        return [e for e in self.crossing_events if e['timestamp'] >= cutoff]
    
    def get_person_crossings(self, person_id: str) -> List[Dict]:
        """Get all crossings for a specific person"""
        if person_id in self.person_history:
            return self.person_history[person_id].get('crossings', [])
        return []
    
    def reset_counts(self):
        """Reset all line crossing counts"""
        for line in self.lines.values():
            line.entry_count = 0
            line.exit_count = 0
        self.crossing_events.clear()
    
    def cleanup_old_persons(self, max_age_seconds: float = 300):
        """Remove persons not seen recently"""
        cutoff = time.time() - max_age_seconds
        self.person_history = {
            pid: data for pid, data in self.person_history.items()
            if data['last_seen'] >= cutoff
        }
    
    def get_pixel_coordinates(self, line: CrossingLine) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert normalized line coordinates to pixel coordinates"""
        start_px = (
            int(line.start[0] * self.frame_width),
            int(line.start[1] * self.frame_height)
        )
        end_px = (
            int(line.end[0] * self.frame_width),
            int(line.end[1] * self.frame_height)
        )
        return start_px, end_px
    
    def draw_lines(self, frame: np.ndarray, thickness: int = 2, show_counts: bool = True) -> np.ndarray:
        """Draw all crossing lines on the frame"""
        import cv2
        
        for line in self.lines.values():
            start_px, end_px = self.get_pixel_coordinates(line)
            
            # Draw line
            cv2.line(frame, start_px, end_px, line.color, thickness)
            
            # Draw label
            label_pos = (
                (start_px[0] + end_px[0]) // 2,
                (start_px[1] + end_px[1]) // 2 - 10
            )
            cv2.putText(frame, line.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, line.color, 2)
            
            if show_counts:
                count_text = f"In:{line.entry_count} Out:{line.exit_count}"
                count_pos = (label_pos[0] - 40, label_pos[1] + 25)
                cv2.putText(frame, count_text, count_pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, line.color, 1)
        
        return frame


# Convenience function to create entry/exit lines
def create_entry_exit_lines(tracker: MultiLineCrossingTracker,
                            entry_position: float = 0.3,
                            exit_position: float = 0.7,
                            orientation: str = 'vertical') -> None:
    """
    Create default entry (A) and exit (B) lines.
    
    Args:
        tracker: MultiLineCrossingTracker instance
        entry_position: Normalized position for entry line (0-1)
        exit_position: Normalized position for exit line (0-1)
        orientation: 'vertical' or 'horizontal'
    """
    if orientation == 'vertical':
        # Vertical lines (crossing left-right)
        tracker.add_line(
            name='A',
            start=(entry_position, 0.0),
            end=(entry_position, 1.0),
            direction_positive='entry',
            direction_negative='exit',
            color=(0, 255, 0)  # Green
        )
        tracker.add_line(
            name='B',
            start=(exit_position, 0.0),
            end=(exit_position, 1.0),
            direction_positive='exit',
            direction_negative='entry',
            color=(0, 0, 255)  # Red
        )
    else:
        # Horizontal lines (crossing top-bottom)
        tracker.add_line(
            name='A',
            start=(0.0, entry_position),
            end=(1.0, entry_position),
            direction_positive='entry',
            direction_negative='exit',
            color=(0, 255, 0)
        )
        tracker.add_line(
            name='B',
            start=(0.0, exit_position),
            end=(1.0, exit_position),
            direction_positive='exit',
            direction_negative='entry',
            color=(0, 0, 255)
        )
