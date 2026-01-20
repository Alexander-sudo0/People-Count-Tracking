"""
Events Database - SQLite storage for all face detection and tracking events
Stores unique people counts, entry/exit events, camera detections, and movement history
"""
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

class EventsDB:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path='db/events.db'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path='db/events.db'):
        if self._initialized:
            return
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._initialized = True

    def _create_tables(self):
        """Create all event tables"""
        cursor = self.conn.cursor()
        
        # Persons table - stores unique people with embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_camera_id TEXT,
                last_camera_name TEXT,
                embedding BLOB,
                total_detections INTEGER DEFAULT 1,
                metadata TEXT
            )
        ''')
        
        # Detection events - every time a person is detected
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                camera_name TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                in_roi INTEGER DEFAULT 0,
                event_type TEXT DEFAULT 'detection',
                thumbnail_path TEXT,
                fullframe_path TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        ''')
        
        # Entry/Exit events - when person crosses ROI boundary
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_exit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                camera_name TEXT,
                event_type TEXT NOT NULL,
                event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                direction TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
        ''')
        
        # Camera stats - aggregate counts per camera with time windows
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                camera_name TEXT,
                unique_count INTEGER DEFAULT 0,
                entry_count INTEGER DEFAULT 0,
                exit_count INTEGER DEFAULT 0,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Global entry/exit tracking - synced across entry and exit cameras
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_counts (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_entries INTEGER DEFAULT 0,
                total_exits INTEGER DEFAULT 0,
                currently_inside INTEGER DEFAULT 0,
                last_entry_time TIMESTAMP,
                last_exit_time TIMESTAMP,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize global counts if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO global_counts (id, total_entries, total_exits, currently_inside)
            VALUES (1, 0, 0, 0)
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_person ON detection_events(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_camera ON detection_events(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_events(detected_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_exit_person ON entry_exit_events(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entry_exit_time ON entry_exit_events(event_time)')
        
        # Migrate: Add thumbnail_path and fullframe_path columns if missing
        self._migrate_add_image_columns(cursor)
        
        self.conn.commit()
    
    def _migrate_add_image_columns(self, cursor):
        """Add image path columns to tables if they don't exist"""
        # Check if columns already exist in detection_events
        cursor.execute("PRAGMA table_info(detection_events)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'thumbnail_path' not in columns:
            cursor.execute('ALTER TABLE detection_events ADD COLUMN thumbnail_path TEXT')
            print("[EventsDB] Migrated: Added thumbnail_path column to detection_events")
            
        if 'fullframe_path' not in columns:
            cursor.execute('ALTER TABLE detection_events ADD COLUMN fullframe_path TEXT')
            print("[EventsDB] Migrated: Added fullframe_path column to detection_events")
        
        # Add primary_thumbnail_path to persons table
        cursor.execute("PRAGMA table_info(persons)")
        person_columns = [col[1] for col in cursor.fetchall()]
        
        if 'primary_thumbnail_path' not in person_columns:
            cursor.execute('ALTER TABLE persons ADD COLUMN primary_thumbnail_path TEXT')
            print("[EventsDB] Migrated: Added primary_thumbnail_path column to persons")
        
        if 'primary_fullframe_path' not in person_columns:
            cursor.execute('ALTER TABLE persons ADD COLUMN primary_fullframe_path TEXT')
            print("[EventsDB] Migrated: Added primary_fullframe_path column to persons")
        
        # Add watchlist_name and watchlist_confidence columns to detection_events
        cursor.execute("PRAGMA table_info(detection_events)")
        detection_columns = [col[1] for col in cursor.fetchall()]
        
        if 'watchlist_name' not in detection_columns:
            cursor.execute('ALTER TABLE detection_events ADD COLUMN watchlist_name TEXT')
            print("[EventsDB] Migrated: Added watchlist_name column to detection_events")
        
        if 'watchlist_confidence' not in detection_columns:
            cursor.execute('ALTER TABLE detection_events ADD COLUMN watchlist_confidence REAL')
            print("[EventsDB] Migrated: Added watchlist_confidence column to detection_events")

    def _is_duplicate_detection(self, camera_id: str, bbox: dict, time_window_seconds: int = 2) -> tuple:
        """
        Check if there's a recent detection at a similar location.
        Returns (is_duplicate, existing_person_id) if duplicate found.
        """
        if not bbox:
            return False, None
        
        cursor = self.conn.cursor()
        
        # Get recent detections from same camera within time window
        cursor.execute('''
            SELECT person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM detection_events
            WHERE camera_id = ?
            AND detected_at >= datetime('now', '-' || ? || ' seconds')
            ORDER BY detected_at DESC
            LIMIT 20
        ''', (camera_id, time_window_seconds))
        
        recent = cursor.fetchall()
        
        new_x1 = bbox.get('x1', 0)
        new_y1 = bbox.get('y1', 0)
        new_x2 = bbox.get('x2', 0)
        new_y2 = bbox.get('y2', 0)
        new_cx = (new_x1 + new_x2) / 2
        new_cy = (new_y1 + new_y2) / 2
        
        for row in recent:
            old_x1, old_y1, old_x2, old_y2 = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
            old_cx = (old_x1 + old_x2) / 2
            old_cy = (old_y1 + old_y2) / 2
            
            # Check if centroids are close (within 50 pixels)
            distance = ((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5
            if distance < 50:
                return True, row['person_id']
        
        return False, None

    def record_detection(self, person_id: str, camera_id: str, camera_name: str = None,
                         confidence: float = 1.0, bbox: dict = None, in_roi: bool = False,
                         event_type: str = 'detection', thumbnail_path: str = None,
                         fullframe_path: str = None, skip_dedup: bool = False,
                         embedding: bytes = None,
                         watchlist_name: str = None, watchlist_confidence: float = None) -> int:
        """Record a face detection event with optional image paths, embedding, and watchlist info"""
        cursor = self.conn.cursor()
        
        bbox_x1 = bbox.get('x1', 0) if bbox else 0
        bbox_y1 = bbox.get('y1', 0) if bbox else 0
        bbox_x2 = bbox.get('x2', 0) if bbox else 0
        bbox_y2 = bbox.get('y2', 0) if bbox else 0
        
        # Check for spatial duplicates if not skipped
        if not skip_dedup and bbox:
            is_dup, existing_id = self._is_duplicate_detection(camera_id, bbox, time_window_seconds=3)
            if is_dup and existing_id:
                # Update existing person instead of creating new entry
                print(f"[EventsDB] Deduplicated: {person_id} -> {existing_id} (same location)")
                person_id = existing_id
        
        cursor.execute('''
            INSERT INTO detection_events 
            (person_id, camera_id, camera_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, in_roi, event_type, thumbnail_path, fullframe_path, watchlist_name, watchlist_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, camera_id, camera_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
              1 if in_roi else 0, event_type, thumbnail_path, fullframe_path, watchlist_name, watchlist_confidence))
        
        event_id = cursor.lastrowid
        
        # Update person's last seen and store primary thumbnail/fullframe (if this is first detection)
        # Also store embedding if provided
        if embedding is not None:
            cursor.execute('''
                INSERT INTO persons (person_id, last_seen, last_camera_id, last_camera_name, total_detections, primary_thumbnail_path, primary_fullframe_path, embedding)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    last_camera_id = ?,
                    last_camera_name = ?,
                    total_detections = total_detections + 1,
                    primary_thumbnail_path = COALESCE(persons.primary_thumbnail_path, ?),
                    primary_fullframe_path = COALESCE(persons.primary_fullframe_path, ?),
                    embedding = COALESCE(persons.embedding, ?)
            ''', (person_id, camera_id, camera_name, thumbnail_path, fullframe_path, embedding,
                  camera_id, camera_name, thumbnail_path, fullframe_path, embedding))
        else:
            cursor.execute('''
                INSERT INTO persons (person_id, last_seen, last_camera_id, last_camera_name, total_detections, primary_thumbnail_path, primary_fullframe_path)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, 1, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    last_camera_id = ?,
                    last_camera_name = ?,
                    total_detections = total_detections + 1,
                    primary_thumbnail_path = COALESCE(persons.primary_thumbnail_path, ?),
                    primary_fullframe_path = COALESCE(persons.primary_fullframe_path, ?)
            ''', (person_id, camera_id, camera_name, thumbnail_path, fullframe_path, 
                  camera_id, camera_name, thumbnail_path, fullframe_path))
        
        self.conn.commit()
        return event_id

    def record_entry_exit(self, person_id: str, camera_id: str, camera_name: str,
                          event_type: str, direction: str = None) -> int:
        """Record an entry or exit event"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO entry_exit_events (person_id, camera_id, camera_name, event_type, direction)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, camera_id, camera_name, event_type, direction))
        self.conn.commit()
        return cursor.lastrowid

    def get_unique_count(self, camera_id: str, time_window_seconds: int = 3600) -> int:
        """Get unique people count within time window for a camera"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT person_id) as count
            FROM detection_events
            WHERE camera_id = ?
            AND detected_at >= datetime('now', '-' || ? || ' seconds')
        ''', (camera_id, time_window_seconds))
        result = cursor.fetchone()
        return result['count'] if result else 0

    def get_unique_count_in_roi(self, camera_id: str, time_window_seconds: int = 3600) -> int:
        """Get unique people count within ROI and time window"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT person_id) as count
            FROM detection_events
            WHERE camera_id = ?
            AND in_roi = 1
            AND detected_at >= datetime('now', '-' || ? || ' seconds')
        ''', (camera_id, time_window_seconds))
        result = cursor.fetchone()
        return result['count'] if result else 0

    def get_person_embedding(self, person_id: str):
        """Get the face embedding for a person (for watchlist matching)"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM persons WHERE person_id = ?', (person_id,))
        row = cursor.fetchone()
        if row and row['embedding']:
            import numpy as np
            return np.frombuffer(row['embedding'], dtype=np.float32)
        return None

    # ========== GLOBAL ENTRY/EXIT TRACKING ==========
    
    def record_global_entry(self, person_id: str, camera_id: str, camera_name: str = None):
        """Record a person entry and update global counts"""
        cursor = self.conn.cursor()
        
        # Record the entry event
        cursor.execute('''
            INSERT INTO entry_exit_events (person_id, camera_id, camera_name, event_type, direction)
            VALUES (?, ?, ?, 'entry', 'in')
        ''', (person_id, camera_id, camera_name))
        
        # Update global counts
        cursor.execute('''
            UPDATE global_counts 
            SET total_entries = total_entries + 1,
                currently_inside = currently_inside + 1,
                last_entry_time = CURRENT_TIMESTAMP
            WHERE id = 1
        ''')
        
        self.conn.commit()
        return self.get_global_counts()
    
    def record_global_exit(self, person_id: str, camera_id: str, camera_name: str = None):
        """Record a person exit and update global counts"""
        cursor = self.conn.cursor()
        
        # Record the exit event
        cursor.execute('''
            INSERT INTO entry_exit_events (person_id, camera_id, camera_name, event_type, direction)
            VALUES (?, ?, ?, 'exit', 'out')
        ''', (person_id, camera_id, camera_name))
        
        # Update global counts (ensure currently_inside doesn't go below 0)
        cursor.execute('''
            UPDATE global_counts 
            SET total_exits = total_exits + 1,
                currently_inside = MAX(0, currently_inside - 1),
                last_exit_time = CURRENT_TIMESTAMP
            WHERE id = 1
        ''')
        
        self.conn.commit()
        return self.get_global_counts()
    
    def get_global_counts(self) -> dict:
        """Get the global entry/exit/currently inside counts"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT total_entries, total_exits, currently_inside, last_entry_time, last_exit_time, last_reset
            FROM global_counts WHERE id = 1
        ''')
        row = cursor.fetchone()
        if row:
            return {
                'total_entries': row['total_entries'] or 0,
                'total_exits': row['total_exits'] or 0,
                'currently_inside': row['currently_inside'] or 0,
                'last_entry_time': row['last_entry_time'],
                'last_exit_time': row['last_exit_time'],
                'last_reset': row['last_reset']
            }
        return {'total_entries': 0, 'total_exits': 0, 'currently_inside': 0}
    
    def reset_global_counts(self):
        """Reset global entry/exit counts"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE global_counts 
            SET total_entries = 0, total_exits = 0, currently_inside = 0,
                last_reset = CURRENT_TIMESTAMP
            WHERE id = 1
        ''')
        self.conn.commit()
        print("[EventsDB] Global counts reset")
        return True

    def get_entry_exit_counts(self, camera_id: str, time_window_seconds: int = 3600) -> dict:
        """Get entry and exit counts within time window"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN event_type = 'entry' THEN 1 ELSE 0 END) as entries,
                SUM(CASE WHEN event_type = 'exit' THEN 1 ELSE 0 END) as exits
            FROM entry_exit_events
            WHERE camera_id = ?
            AND event_time >= datetime('now', '-' || ? || ' seconds')
        ''', (camera_id, time_window_seconds))
        result = cursor.fetchone()
        return {
            'entries': result['entries'] or 0,
            'exits': result['exits'] or 0,
            'current': (result['entries'] or 0) - (result['exits'] or 0)
        }

    def get_person_location(self, person_id: str) -> Optional[dict]:
        """Get person's current location (last camera)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT person_id, last_camera_id, last_camera_name, 
                   datetime(last_seen, 'localtime') as last_seen, 
                   datetime(first_seen, 'localtime') as first_seen, 
                   total_detections
            FROM persons
            WHERE person_id = ?
        ''', (person_id,))
        row = cursor.fetchone()
        if row:
            last_seen = row['last_seen']
            first_seen = row['first_seen']
            # Convert to ISO format for JavaScript
            if last_seen and 'T' not in str(last_seen):
                last_seen = str(last_seen).replace(' ', 'T')
            if first_seen and 'T' not in str(first_seen):
                first_seen = str(first_seen).replace(' ', 'T')
            return {
                'person_id': row['person_id'],
                'last_camera_id': row['last_camera_id'],
                'last_camera_name': row['last_camera_name'],
                'last_seen': last_seen,
                'first_seen': first_seen,
                'total_detections': row['total_detections']
            }
        return None

    def get_person_history(self, person_id: str, limit: int = 100) -> List[dict]:
        """Get person's detection history across all cameras with images"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT camera_id, camera_name, datetime(detected_at, 'localtime') as detected_at, 
                   confidence, event_type, in_roi,
                   thumbnail_path, fullframe_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM detection_events
            WHERE person_id = ?
            ORDER BY detected_at DESC
            LIMIT ?
        ''', (person_id, limit))
        
        history = []
        for row in cursor.fetchall():
            detected_at = row['detected_at']
            if detected_at:
                try:
                    from datetime import datetime as dt
                    dt_obj = dt.strptime(detected_at, '%Y-%m-%d %H:%M:%S')
                    detected_at = dt_obj.isoformat()
                except:
                    pass
            
            history.append({
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'detected_at': detected_at,
                'confidence': row['confidence'],
                'event_type': row['event_type'],
                'in_roi': bool(row['in_roi']),
                'thumbnail_path': row['thumbnail_path'],
                'fullframe_path': row['fullframe_path'],
                'bbox': {
                    'x1': row['bbox_x1'],
                    'y1': row['bbox_y1'],
                    'x2': row['bbox_x2'],
                    'y2': row['bbox_y2']
                }
            })
        return history

    def get_person_entry_exit_history(self, person_id: str, limit: int = 50) -> List[dict]:
        """Get person's entry/exit events"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT camera_id, camera_name, event_type, event_time, direction
            FROM entry_exit_events
            WHERE person_id = ?
            ORDER BY event_time DESC
            LIMIT ?
        ''', (person_id, limit))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'event_type': row['event_type'],
                'event_time': row['event_time'],
                'direction': row['direction']
            })
        return events

    def get_camera_people(self, camera_id: str, since_hours: int = 24, limit: int = 50) -> List[dict]:
        """Get unique people detected by a specific camera"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                de.person_id,
                datetime(MIN(de.detected_at), 'localtime') as first_seen,
                datetime(MAX(de.detected_at), 'localtime') as last_seen,
                COUNT(*) as detection_count,
                MAX(de.confidence) as max_confidence
            FROM detection_events de
            WHERE de.camera_id = ?
            AND de.detected_at >= datetime('now', '-' || ? || ' hours')
            GROUP BY de.person_id
            ORDER BY last_seen DESC
            LIMIT ?
        ''', (camera_id, since_hours, limit))
        
        people = []
        for row in cursor.fetchall():
            first_seen = row['first_seen']
            last_seen = row['last_seen']
            
            # Convert to ISO format
            if first_seen:
                try:
                    from datetime import datetime as dt
                    dt_obj = dt.strptime(first_seen, '%Y-%m-%d %H:%M:%S')
                    first_seen = dt_obj.isoformat()
                except:
                    pass
            if last_seen:
                try:
                    from datetime import datetime as dt
                    dt_obj = dt.strptime(last_seen, '%Y-%m-%d %H:%M:%S')
                    last_seen = dt_obj.isoformat()
                except:
                    pass
            
            people.append({
                'person_id': row['person_id'],
                'first_seen': first_seen,
                'last_seen': last_seen,
                'detection_count': row['detection_count'],
                'confidence': row['max_confidence']
            })
        return people

    def search_people(self, query: str = None, camera_id: str = None, 
                      since_hours: int = 24, limit: int = 100) -> List[dict]:
        """Search people with optional filters"""
        cursor = self.conn.cursor()
        
        sql = '''
            SELECT 
                p.person_id,
                p.first_seen,
                p.last_seen,
                p.last_camera_id,
                p.last_camera_name,
                p.total_detections,
                (SELECT GROUP_CONCAT(DISTINCT camera_name) 
                 FROM detection_events de 
                 WHERE de.person_id = p.person_id) as cameras_visited
            FROM persons p
            WHERE p.last_seen >= datetime('now', '-' || ? || ' hours')
        '''
        params = [since_hours]
        
        if query:
            sql += ' AND p.person_id LIKE ?'
            params.append(f'%{query}%')
        
        if camera_id:
            sql += ' AND p.last_camera_id = ?'
            params.append(camera_id)
        
        sql += ' ORDER BY p.last_seen DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        
        people = []
        for row in cursor.fetchall():
            cameras = row['cameras_visited'].split(',') if row['cameras_visited'] else []
            people.append({
                'person_id': row['person_id'],
                'first_seen': row['first_seen'],
                'last_seen': row['last_seen'],
                'last_camera_id': row['last_camera_id'],
                'last_camera_name': row['last_camera_name'],
                'total_detections': row['total_detections'],
                'cameras_visited': list(set(cameras))
            })
        return people

    def get_recent_events(self, camera_id: str = None, limit: int = 50) -> List[dict]:
        """Get recent detection and entry/exit events"""
        cursor = self.conn.cursor()
        
        # Get entry/exit events
        sql = '''
            SELECT 
                person_id, camera_id, camera_name, event_type, event_time as event_at
            FROM entry_exit_events
        '''
        params = []
        if camera_id:
            sql += ' WHERE camera_id = ?'
            params.append(camera_id)
        sql += ' ORDER BY event_time DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'person_id': row['person_id'],
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'event_type': row['event_type'],
                'event_at': row['event_at']
            })
        return events

    def get_recent_detections(self, camera_id: str = None, limit: int = 20) -> List[dict]:
        """Get recent face detections with images for the detection log UI"""
        cursor = self.conn.cursor()
        
        sql = '''
            SELECT 
                d.id, d.person_id, d.camera_id, d.camera_name, 
                datetime(d.detected_at, 'localtime') as detected_at,
                d.confidence, d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                d.in_roi, d.event_type, d.thumbnail_path, d.fullframe_path,
                d.watchlist_name, d.watchlist_confidence
            FROM detection_events d
        '''
        params = []
        if camera_id:
            sql += ' WHERE d.camera_id = ?'
            params.append(camera_id)
        sql += ' ORDER BY d.detected_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        
        detections = []
        for row in cursor.fetchall():
            # Convert SQLite datetime string to ISO format for proper frontend parsing
            detected_at = row['detected_at']
            if detected_at:
                # SQLite returns datetime as "YYYY-MM-DD HH:MM:SS", convert to ISO
                try:
                    from datetime import datetime as dt
                    dt_obj = dt.strptime(detected_at, '%Y-%m-%d %H:%M:%S')
                    detected_at = dt_obj.isoformat()
                except:
                    pass
            
            # Get watchlist info - handle column may not exist
            watchlist_name = None
            watchlist_confidence = None
            try:
                watchlist_name = row['watchlist_name']
                watchlist_confidence = row['watchlist_confidence']
            except:
                pass
            
            detections.append({
                'id': row['id'],
                'person_id': row['person_id'],
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'detected_at': detected_at,
                'confidence': row['confidence'],
                'bbox': {
                    'x1': row['bbox_x1'],
                    'y1': row['bbox_y1'],
                    'x2': row['bbox_x2'],
                    'y2': row['bbox_y2']
                },
                'in_roi': bool(row['in_roi']),
                'event_type': row['event_type'],
                'thumbnail_path': row['thumbnail_path'],
                'fullframe_path': row['fullframe_path'],
                'watchlist_name': watchlist_name,
                'watchlist_confidence': watchlist_confidence
            })
        return detections

    def get_person_thumbnail(self, person_id: str) -> Optional[str]:
        """Get the primary thumbnail for a person (stored on person record for persistence)"""
        cursor = self.conn.cursor()
        
        # First try to get the primary thumbnail from persons table (persists across resets)
        cursor.execute('SELECT primary_thumbnail_path FROM persons WHERE person_id = ?', (person_id,))
        row = cursor.fetchone()
        if row and row['primary_thumbnail_path']:
            return row['primary_thumbnail_path']
        
        # Fall back to most recent from detection_events
        cursor.execute('''
            SELECT thumbnail_path FROM detection_events 
            WHERE person_id = ? AND thumbnail_path IS NOT NULL
            ORDER BY detected_at DESC LIMIT 1
        ''', (person_id,))
        row = cursor.fetchone()
        return row['thumbnail_path'] if row else None

    def reset_camera_counts(self, camera_id: str) -> bool:
        """Reset counts for a camera (called when time window expires)"""
        cursor = self.conn.cursor()
        # Archive current stats before reset
        cursor.execute('''
            INSERT INTO camera_stats (camera_id, camera_name, unique_count, entry_count, exit_count, window_start, window_end)
            SELECT 
                ? as camera_id,
                (SELECT camera_name FROM detection_events WHERE camera_id = ? LIMIT 1) as camera_name,
                COUNT(DISTINCT person_id) as unique_count,
                (SELECT COUNT(*) FROM entry_exit_events WHERE camera_id = ? AND event_type = 'entry') as entry_count,
                (SELECT COUNT(*) FROM entry_exit_events WHERE camera_id = ? AND event_type = 'exit') as exit_count,
                MIN(detected_at) as window_start,
                MAX(detected_at) as window_end
            FROM detection_events
            WHERE camera_id = ?
        ''', (camera_id, camera_id, camera_id, camera_id, camera_id))
        self.conn.commit()
        return True

    def clear_detection_logs(self, camera_id: str, keep_persons: bool = True) -> int:
        """
        Clear detection event logs for a camera (used after timer reset).
        Keeps person data intact for continuous tracking - only clears the count window.
        
        Args:
            camera_id: Camera to clear logs for
            keep_persons: If True, keep person records and their primary thumbnails (default: True)
        
        Returns:
            Number of detection events cleared
        """
        cursor = self.conn.cursor()
        
        # Get count of detections to be deleted
        cursor.execute('SELECT COUNT(*) as count FROM detection_events WHERE camera_id = ?', (camera_id,))
        count = cursor.fetchone()['count']
        
        if keep_persons:
            # IMPORTANT: Only delete detection events for THIS camera, keep all person data
            # This allows tracking to continue across timer resets
            cursor.execute('DELETE FROM detection_events WHERE camera_id = ?', (camera_id,))
            
            # Delete entry/exit events for this camera only
            cursor.execute('DELETE FROM entry_exit_events WHERE camera_id = ?', (camera_id,))
            
            # DO NOT delete persons - they are tracked across all cameras
            # Persons will be updated when they are detected again
        else:
            # Full cleanup mode - also remove orphaned persons
            cursor.execute('DELETE FROM detection_events WHERE camera_id = ?', (camera_id,))
            cursor.execute('DELETE FROM entry_exit_events WHERE camera_id = ?', (camera_id,))
            
            # Only clean up persons with no remaining detections
            cursor.execute('''
                DELETE FROM persons WHERE person_id NOT IN (
                    SELECT DISTINCT person_id FROM detection_events
                )
            ''')
        
        self.conn.commit()
        print(f"[EventsDB] Cleared {count} detection logs for camera {camera_id} (keep_persons={keep_persons})")
        return count

    def get_all_people_with_stats(self, since_hours: int = 24, limit: int = 200, include_all: bool = True) -> List[dict]:
        """
        Get all unique people across all cameras with aggregated stats.
        
        Args:
            since_hours: Time window in hours (default 24)
            limit: Maximum number of results
            include_all: If True, include all people regardless of last_seen time (for tracking persistence)
        """
        cursor = self.conn.cursor()
        
        if include_all:
            # Get ALL people, ordered by last_seen (most recent first)
            cursor.execute('''
                SELECT 
                    p.person_id,
                    datetime(p.first_seen, 'localtime') as first_seen,
                    datetime(p.last_seen, 'localtime') as last_seen,
                    p.total_detections,
                    p.last_camera_id,
                    p.last_camera_name,
                    (SELECT COUNT(DISTINCT camera_id) FROM detection_events WHERE person_id = p.person_id) as cameras_visited,
                    (SELECT MAX(confidence) FROM detection_events WHERE person_id = p.person_id) as max_confidence
                FROM persons p
                ORDER BY p.last_seen DESC
                LIMIT ?
            ''', (limit,))
        else:
            # Filter by since_hours
            cursor.execute('''
                SELECT 
                    p.person_id,
                    datetime(p.first_seen, 'localtime') as first_seen,
                    datetime(p.last_seen, 'localtime') as last_seen,
                    p.total_detections,
                    p.last_camera_id,
                    p.last_camera_name,
                    (SELECT COUNT(DISTINCT camera_id) FROM detection_events WHERE person_id = p.person_id) as cameras_visited,
                    (SELECT MAX(confidence) FROM detection_events WHERE person_id = p.person_id) as max_confidence
                FROM persons p
                WHERE p.last_seen >= datetime('now', '-' || ? || ' hours')
                ORDER BY p.last_seen DESC
                LIMIT ?
            ''', (since_hours, limit))
        
        people = []
        for row in cursor.fetchall():
            last_seen = row['last_seen']
            first_seen = row['first_seen']
            # Convert to ISO format for JavaScript
            if last_seen and 'T' not in str(last_seen):
                last_seen = str(last_seen).replace(' ', 'T')
            if first_seen and 'T' not in str(first_seen):
                first_seen = str(first_seen).replace(' ', 'T')
            people.append({
                'person_id': row['person_id'],
                'first_seen': first_seen,
                'last_seen': last_seen,
                'total_detections': row['total_detections'] or 0,
                'current_camera': row['last_camera_name'],
                'cameras_visited': row['cameras_visited'] or 1,
                'confidence': row['max_confidence'] or 0.0
            })
        return people

    def get_person_camera_visits(self, person_id: str) -> List[dict]:
        """Get all cameras visited by a person with detection counts"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                camera_id,
                camera_name,
                COUNT(*) as detection_count,
                MIN(detected_at) as first_detection,
                MAX(detected_at) as last_detection
            FROM detection_events
            WHERE person_id = ?
            GROUP BY camera_id, camera_name
            ORDER BY last_detection DESC
        ''', (person_id,))
        
        cameras = []
        for row in cursor.fetchall():
            cameras.append({
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'] or row['camera_id'],
                'detection_count': row['detection_count'],
                'first_detection': row['first_detection'],
                'last_detection': row['last_detection'],
                'is_active': False  # Will be updated in API
            })
        return cameras

    def get_stats_summary(self) -> dict:
        """Get overall statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM persons')
        total_people = cursor.fetchone()['total']
        
        cursor.execute('''
            SELECT COUNT(DISTINCT person_id) as active
            FROM detection_events
            WHERE detected_at >= datetime('now', '-1 hour')
        ''')
        active_people = cursor.fetchone()['active']
        
        cursor.execute('SELECT COUNT(DISTINCT camera_id) as cameras FROM detection_events')
        total_cameras = cursor.fetchone()['cameras']
        
        cursor.execute('SELECT COUNT(*) as events FROM detection_events')
        total_events = cursor.fetchone()['events']
        
        return {
            'total_people': total_people,
            'active_people': active_people,
            'total_cameras': total_cameras,
            'total_events': total_events
        }

    def close(self):
        """Close database connection"""
        self.conn.close()


# Singleton instance
_events_db = None

def get_events_db() -> EventsDB:
    """Get singleton EventsDB instance"""
    global _events_db
    if _events_db is None:
        _events_db = EventsDB()
    return _events_db
