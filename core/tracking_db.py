"""
Enhanced Database Schema for Person Tracking System

This module provides comprehensive person tracking with location history,
movement analytics, and timestamp validation to prevent duplicate counting.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json


class TrackingDatabase:
    """Enhanced database with person tracking capabilities"""

    def __init__(self, db_path: str = "db/tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database with enhanced tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Person tracking table - stores person state
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    face_embedding BLOB,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_camera_id TEXT,
                    total_detections INTEGER DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Detection history - stores all detections with timestamps
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    camera_id TEXT,
                    camera_name TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    bbox_data TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                )
            ''')

            # Location history - tracks person's latest location per camera
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    camera_id TEXT,
                    camera_name TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP NULL,
                    duration_seconds INTEGER,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                )
            ''')

            # Movement events - tracks zone/line crossings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movement_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    camera_id TEXT,
                    event_type TEXT,
                    direction TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                )
            ''')

            # Analytics aggregation table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracking_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    camera_id TEXT,
                    unique_visitors INTEGER,
                    total_detections INTEGER,
                    avg_dwell_time_seconds REAL,
                    peak_hour INTEGER
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_last_seen ON persons(last_seen)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_history(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_camera ON detection_history(camera_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_person ON location_history(person_id)')

            conn.commit()

    def add_or_update_person(self, person_id: str, camera_id: str, camera_name: str,
                            embedding: Optional[bytes] = None, confidence: float = 0.0,
                            bbox: Optional[Dict] = None) -> bool:
        """
        Add new person or update existing person's tracking data
        Returns True if this is a new unique detection (not duplicate)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if person exists
            cursor.execute('SELECT person_id, last_seen, last_camera_id FROM persons WHERE person_id = ?',
                          (person_id,))
            result = cursor.fetchone()

            current_time = datetime.now()

            if result:
                # Person exists - check for duplicate detection
                last_seen = datetime.fromisoformat(result[1])
                last_camera = result[2]

                # Prevent duplicate if same camera within 30 seconds
                time_diff = (current_time - last_seen).total_seconds()
                if last_camera == camera_id and time_diff < 30:
                    return False  # Duplicate detection

                # Update person record
                cursor.execute('''
                    UPDATE persons 
                    SET last_seen = ?, last_camera_id = ?, total_detections = total_detections + 1
                    WHERE person_id = ?
                ''', (current_time, camera_id, person_id))

            else:
                # New person
                cursor.execute('''
                    INSERT INTO persons (person_id, face_embedding, last_camera_id)
                    VALUES (?, ?, ?)
                ''', (person_id, embedding, camera_id))

            # Add detection to history
            cursor.execute('''
                INSERT INTO detection_history 
                (person_id, camera_id, camera_name, confidence, bbox_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, camera_id, camera_name, confidence, json.dumps(bbox) if bbox else None))

            # Update location history
            self._update_location_history(cursor, person_id, camera_id, camera_name, current_time)

            conn.commit()
            return True  # Valid new detection

    def _update_location_history(self, cursor, person_id: str, camera_id: str,
                                 camera_name: str, timestamp: datetime):
        """Update location history when person moves between cameras"""
        # Check if person has an open location entry for different camera
        cursor.execute('''
            SELECT id, camera_id, entry_time 
            FROM location_history 
            WHERE person_id = ? AND exit_time IS NULL
            ORDER BY entry_time DESC LIMIT 1
        ''', (person_id,))

        result = cursor.fetchone()

        if result:
            loc_id, prev_camera, entry_time = result
            entry_dt = datetime.fromisoformat(entry_time)

            if prev_camera != camera_id:
                # Person moved to different camera - close previous location
                duration = (timestamp - entry_dt).total_seconds()
                cursor.execute('''
                    UPDATE location_history 
                    SET exit_time = ?, duration_seconds = ?
                    WHERE id = ?
                ''', (timestamp, int(duration), loc_id))

                # Create new location entry
                cursor.execute('''
                    INSERT INTO location_history (person_id, camera_id, camera_name, entry_time)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, camera_id, camera_name, timestamp))
        else:
            # First location for this person
            cursor.execute('''
                INSERT INTO location_history (person_id, camera_id, camera_name, entry_time)
                VALUES (?, ?, ?, ?)
            ''', (person_id, camera_id, camera_name, timestamp))

    def get_person_current_location(self, person_id: str) -> Optional[Dict]:
        """Get person's current camera location and last seen time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.person_id, p.last_camera_id, p.last_seen, p.total_detections,
                       l.camera_name, l.entry_time
                FROM persons p
                LEFT JOIN location_history l ON p.person_id = l.person_id AND l.exit_time IS NULL
                WHERE p.person_id = ?
            ''', (person_id,))

            result = cursor.fetchone()
            if result:
                return {
                    'person_id': result[0],
                    'last_camera': result[1],
                    'last_seen': result[2],
                    'total_detections': result[3],
                    'camera_name': result[4] or 'Unknown',
                    'entry_time': result[5]
                }
            return None

    def get_person_movement_history(self, person_id: str, limit: int = 50) -> List[Dict]:
        """Get person's movement history across cameras"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT camera_id, camera_name, entry_time, exit_time, duration_seconds
                FROM location_history
                WHERE person_id = ?
                ORDER BY entry_time DESC
                LIMIT ?
            ''', (person_id, limit))

            return [{
                'camera_id': row[0],
                'camera_name': row[1],
                'entry_time': row[2],
                'exit_time': row[3],
                'duration_seconds': row[4]
            } for row in cursor.fetchall()]

    def get_all_tracked_people(self, active_only: bool = False, minutes: int = 60) -> List[Dict]:
        """Get all tracked people with their latest location"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            query = '''
                SELECT p.person_id, p.last_camera_id, p.last_seen, p.first_seen, 
                       p.total_detections, l.camera_name
                FROM persons p
                LEFT JOIN location_history l ON p.person_id = l.person_id AND l.exit_time IS NULL
            '''

            if active_only:
                query += f" WHERE p.last_seen >= datetime('now', '-{minutes} minutes')"

            query += " ORDER BY p.last_seen DESC"

            cursor.execute(query)

            results = []
            for row in cursor.fetchall():
                # Get all cameras visited
                cursor.execute('''
                    SELECT DISTINCT camera_id FROM location_history WHERE person_id = ?
                ''', (row[0],))
                cameras_visited = [r[0] for r in cursor.fetchall()]

                results.append({
                    'person_id': row[0],
                    'last_camera': row[1],
                    'last_camera_name': row[5] or row[1],
                    'last_seen': row[2],
                    'first_seen': row[3],
                    'total_detections': row[4],
                    'cameras_visited': cameras_visited
                })

            return results

    def get_tracking_stats(self) -> Dict:
        """Get overall tracking statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total unique people
            cursor.execute('SELECT COUNT(*) FROM persons')
            total_people = cursor.fetchone()[0]

            # Active people (seen in last hour)
            cursor.execute('''
                SELECT COUNT(*) FROM persons 
                WHERE last_seen >= datetime('now', '-60 minutes')
            ''')
            active_people = cursor.fetchone()[0]

            # Total cameras
            cursor.execute('SELECT COUNT(DISTINCT camera_id) FROM location_history')
            total_cameras = cursor.fetchone()[0]

            # Total detections today
            cursor.execute('''
                SELECT COUNT(*) FROM detection_history 
                WHERE DATE(timestamp) = DATE('now')
            ''')
            detections_today = cursor.fetchone()[0]

            return {
                'total_people': total_people,
                'active_people': active_people,
                'total_cameras': total_cameras,
                'detections_today': detections_today
            }

    def record_movement_event(self, person_id: str, camera_id: str,
                             event_type: str, direction: str = None,
                             metadata: Dict = None):
        """Record movement events (entry, exit, zone crossing)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO movement_events 
                (person_id, camera_id, event_type, direction, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, camera_id, event_type, direction, json.dumps(metadata) if metadata else None))
            conn.commit()
