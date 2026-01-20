"""
Watchlist Database - Track persons of interest with photos and embeddings
"""
import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class WatchlistDB:
    def __init__(self, db_path='db/watchlist.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create watchlist tables"""
        cursor = self.conn.cursor()
        
        # Watchlist persons table with photo and embedding support
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                person_id TEXT PRIMARY KEY,
                name TEXT,
                photo_path TEXT,
                thumbnail_path TEXT,
                embedding BLOB,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                alert_enabled INTEGER DEFAULT 1,
                category TEXT DEFAULT 'general',
                last_location TEXT,
                last_camera_id TEXT,
                last_seen TIMESTAMP
            )
        ''')
        
        # Migration: Add category column if missing
        try:
            cursor.execute("SELECT category FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN category TEXT DEFAULT 'general'")
            print("[WatchlistDB] Migrated: Added category column")
        
        # Migration: Add photo_path column if missing
        try:
            cursor.execute("SELECT photo_path FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN photo_path TEXT")
            print("[WatchlistDB] Migrated: Added photo_path column")
        
        # Migration: Add thumbnail_path column if missing
        try:
            cursor.execute("SELECT thumbnail_path FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN thumbnail_path TEXT")
            print("[WatchlistDB] Migrated: Added thumbnail_path column")
        
        # Migration: Add embedding column if missing
        try:
            cursor.execute("SELECT embedding FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN embedding BLOB")
            print("[WatchlistDB] Migrated: Added embedding column")
        
        # Migration: Add last_location column if missing
        try:
            cursor.execute("SELECT last_location FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN last_location TEXT")
            print("[WatchlistDB] Migrated: Added last_location column")
        
        # Migration: Add last_camera_id column if missing
        try:
            cursor.execute("SELECT last_camera_id FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN last_camera_id TEXT")
            print("[WatchlistDB] Migrated: Added last_camera_id column")
        
        # Migration: Add last_seen column if missing
        try:
            cursor.execute("SELECT last_seen FROM watchlist LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN last_seen TIMESTAMP")
            print("[WatchlistDB] Migrated: Added last_seen column")
        
        # Watchlist detections table with image paths
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                camera_id TEXT,
                camera_name TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                thumbnail_path TEXT,
                fullframe_path TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                FOREIGN KEY (person_id) REFERENCES watchlist(person_id)
            )
        ''')
        
        # Custom categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#808080',
                description TEXT,
                alert_priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default categories if not exist
        default_categories = [
            ('general', '#6366f1', 'General watchlist', 1),
            ('vip', '#22c55e', 'VIP - Very Important Persons', 2),
            ('banned', '#ef4444', 'Banned persons - Alert immediately', 3),
            ('employee', '#3b82f6', 'Employees', 1),
            ('visitor', '#f59e0b', 'Regular visitors', 1)
        ]
        for cat_name, color, desc, priority in default_categories:
            cursor.execute('''
                INSERT OR IGNORE INTO watchlist_categories (name, color, description, alert_priority)
                VALUES (?, ?, ?, ?)
            ''', (cat_name, color, desc, priority))
        
        # Create indexes (only if columns exist)
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_name ON watchlist(name)')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_det_time ON watchlist_detections(detected_at)')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_category ON watchlist(category)')
        except sqlite3.OperationalError:
            pass
        
        self.conn.commit()

    def add_to_watchlist(self, person_id: str, name: str = None, notes: str = None, 
                         alert_enabled: bool = True, photo_path: str = None,
                         thumbnail_path: str = None, embedding: np.ndarray = None,
                         category: str = 'general'):
        """Add a person to the watchlist with optional photo and embedding"""
        cursor = self.conn.cursor()
        
        embedding_bytes = embedding.tobytes() if embedding is not None else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO watchlist 
            (person_id, name, notes, alert_enabled, photo_path, thumbnail_path, embedding, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, name, notes, 1 if alert_enabled else 0, 
              photo_path, thumbnail_path, embedding_bytes, category))
        self.conn.commit()
        return True

    def update_watchlist_person(self, person_id: str, name: str = None, notes: str = None,
                                alert_enabled: bool = None, category: str = None):
        """Update a watchlist person's details"""
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if name is not None:
            updates.append('name = ?')
            params.append(name)
        if notes is not None:
            updates.append('notes = ?')
            params.append(notes)
        if alert_enabled is not None:
            updates.append('alert_enabled = ?')
            params.append(1 if alert_enabled else 0)
        if category is not None:
            updates.append('category = ?')
            params.append(category)
        
        if not updates:
            return False
        
        params.append(person_id)
        cursor.execute(f'''
            UPDATE watchlist SET {', '.join(updates)} WHERE person_id = ?
        ''', params)
        self.conn.commit()
        return True

    def set_photo(self, person_id: str, photo_path: str, thumbnail_path: str = None):
        """Set photo for a watchlist person"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE watchlist SET photo_path = ?, thumbnail_path = ? WHERE person_id = ?
        ''', (photo_path, thumbnail_path, person_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def set_embedding(self, person_id: str, embedding: np.ndarray):
        """Set face embedding for a watchlist person (for face matching)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE watchlist SET embedding = ? WHERE person_id = ?
        ''', (embedding.tobytes(), person_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_embedding(self, person_id: str) -> Optional[np.ndarray]:
        """Get face embedding for a watchlist person"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT embedding FROM watchlist WHERE person_id = ?', (person_id,))
        row = cursor.fetchone()
        if row and row['embedding']:
            return np.frombuffer(row['embedding'], dtype=np.float32)
        return None

    def get_all_embeddings(self) -> List[Dict]:
        """Get all watchlist embeddings for face matching"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT person_id, name, embedding FROM watchlist WHERE embedding IS NOT NULL')
        
        results = []
        for row in cursor.fetchall():
            if row['embedding']:
                results.append({
                    'person_id': row['person_id'],
                    'name': row['name'],
                    'embedding': np.frombuffer(row['embedding'], dtype=np.float32)
                })
        return results

    def remove_from_watchlist(self, person_id: str):
        """Remove a person from the watchlist"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM watchlist_detections WHERE person_id = ?', (person_id,))
        cursor.execute('DELETE FROM watchlist WHERE person_id = ?', (person_id,))
        self.conn.commit()
        return True

    def is_in_watchlist(self, person_id: str):
        """Check if a person is in the watchlist"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT person_id FROM watchlist WHERE person_id = ?', (person_id,))
        return cursor.fetchone() is not None

    def get_watchlist(self, category: str = None) -> List[Dict]:
        """Get all persons in the watchlist"""
        cursor = self.conn.cursor()
        
        # Use simpler query that handles missing columns gracefully
        try:
            sql = '''
                SELECT 
                    w.person_id, 
                    w.name, 
                    w.photo_path,
                    w.thumbnail_path,
                    w.added_at, 
                    w.notes, 
                    w.alert_enabled,
                    w.category,
                    w.last_location,
                    w.last_camera_id,
                    w.last_seen,
                    COUNT(wd.id) as detection_count,
                    MAX(wd.detected_at) as last_detected
                FROM watchlist w
                LEFT JOIN watchlist_detections wd ON w.person_id = wd.person_id
            '''
        except Exception:
            # Fallback for older schema
            sql = '''
                SELECT 
                    w.person_id, 
                    w.name, 
                    NULL as photo_path,
                    NULL as thumbnail_path,
                    w.added_at, 
                    w.notes, 
                    w.alert_enabled,
                    w.category,
                    NULL as last_location,
                    NULL as last_camera_id,
                    NULL as last_seen,
                    0 as detection_count,
                    NULL as last_detected
                FROM watchlist w
            '''
        params = []
        
        if category:
            sql += ' WHERE w.category = ?'
            params.append(category)
        
        sql += ' GROUP BY w.person_id ORDER BY w.added_at DESC'
        
        try:
            cursor.execute(sql, params)
        except Exception as e:
            print(f"[WatchlistDB] Query error, trying fallback: {e}")
            # Very simple fallback
            cursor.execute('SELECT person_id, name, notes, alert_enabled, added_at FROM watchlist')
            watchlist = []
            for row in cursor.fetchall():
                watchlist.append({
                    'person_id': row['person_id'],
                    'name': row['name'],
                    'notes': row['notes'],
                    'alert_enabled': bool(row['alert_enabled']),
                    'added_at': row['added_at'],
                    'photo_path': None,
                    'thumbnail_path': None,
                    'category': 'general',
                    'last_location': None,
                    'last_camera_id': None,
                    'last_seen': None,
                    'detection_count': 0
                })
            return watchlist
        
        watchlist = []
        for row in cursor.fetchall():
            watchlist.append({
                'person_id': row['person_id'],
                'name': row['name'],
                'photo_path': row['photo_path'],
                'thumbnail_path': row['thumbnail_path'],
                'added_at': row['added_at'],
                'notes': row['notes'],
                'alert_enabled': bool(row['alert_enabled']),
                'category': row['category'] or 'general',
                'last_location': row['last_location'],
                'last_camera_id': row['last_camera_id'],
                'last_seen': row['last_seen'] or row['last_detected'],
                'detection_count': row['detection_count']
            })
        return watchlist

    def get_person(self, person_id: str) -> Optional[Dict]:
        """Get a specific watchlist person"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                w.*,
                (SELECT COUNT(*) FROM watchlist_detections WHERE person_id = w.person_id) as detection_count
            FROM watchlist w
            WHERE w.person_id = ?
        ''', (person_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'person_id': row['person_id'],
                'name': row['name'],
                'photo_path': row['photo_path'],
                'thumbnail_path': row['thumbnail_path'],
                'added_at': row['added_at'],
                'notes': row['notes'],
                'alert_enabled': bool(row['alert_enabled']),
                'category': row['category'],
                'last_location': row['last_location'],
                'last_camera_id': row['last_camera_id'],
                'last_seen': row['last_seen'],
                'detection_count': row['detection_count']
            }
        return None

    def search_watchlist(self, query: str) -> List[Dict]:
        """Search watchlist by name or person_id"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                w.person_id, 
                w.name, 
                w.photo_path,
                w.thumbnail_path,
                w.notes,
                w.category,
                w.last_location,
                w.last_seen,
                COUNT(wd.id) as detection_count
            FROM watchlist w
            LEFT JOIN watchlist_detections wd ON w.person_id = wd.person_id
            WHERE w.person_id LIKE ? OR w.name LIKE ?
            GROUP BY w.person_id
        ''', (f'%{query}%', f'%{query}%'))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'person_id': row['person_id'],
                'name': row['name'],
                'photo_path': row['photo_path'],
                'thumbnail_path': row['thumbnail_path'],
                'notes': row['notes'],
                'category': row['category'],
                'last_location': row['last_location'],
                'last_seen': row['last_seen'],
                'detection_count': row['detection_count']
            })
        return results

    def record_detection(self, person_id: str, camera_id: str, camera_name: str, 
                         confidence: float = 1.0, thumbnail_path: str = None,
                         fullframe_path: str = None, bbox: dict = None):
        """Record a detection of a watchlist person with images"""
        if not self.is_in_watchlist(person_id):
            return False
        
        cursor = self.conn.cursor()
        
        bbox_x1 = bbox.get('x1', 0) if bbox else 0
        bbox_y1 = bbox.get('y1', 0) if bbox else 0
        bbox_x2 = bbox.get('x2', 0) if bbox else 0
        bbox_y2 = bbox.get('y2', 0) if bbox else 0
        
        cursor.execute('''
            INSERT INTO watchlist_detections 
            (person_id, camera_id, camera_name, confidence, thumbnail_path, fullframe_path,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, camera_id, camera_name, confidence, thumbnail_path, fullframe_path,
              bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        
        # Update last seen location
        cursor.execute('''
            UPDATE watchlist SET last_location = ?, last_camera_id = ?, last_seen = CURRENT_TIMESTAMP
            WHERE person_id = ?
        ''', (camera_name, camera_id, person_id))
        
        self.conn.commit()
        return True

    def get_person_detections(self, person_id: str, limit: int = 50) -> List[Dict]:
        """Get detection history for a person with images"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT camera_id, camera_name, detected_at, confidence,
                   thumbnail_path, fullframe_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM watchlist_detections
            WHERE person_id = ?
            ORDER BY detected_at DESC
            LIMIT ?
        ''', (person_id, limit))
        
        detections = []
        for row in cursor.fetchall():
            detections.append({
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'detected_at': row['detected_at'],
                'confidence': row['confidence'],
                'thumbnail_path': row['thumbnail_path'],
                'fullframe_path': row['fullframe_path'],
                'bbox': {
                    'x1': row['bbox_x1'],
                    'y1': row['bbox_y1'],
                    'x2': row['bbox_x2'],
                    'y2': row['bbox_y2']
                }
            })
        return detections

    def get_watchlist_alerts(self, since_minutes: int = 5) -> List[Dict]:
        """Get recent detections of watchlist persons"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                w.person_id,
                w.name,
                w.notes,
                w.thumbnail_path as person_thumbnail,
                wd.camera_id,
                wd.camera_name,
                wd.detected_at,
                wd.confidence,
                wd.thumbnail_path,
                wd.fullframe_path
            FROM watchlist w
            JOIN watchlist_detections wd ON w.person_id = wd.person_id
            WHERE w.alert_enabled = 1
            AND datetime(wd.detected_at) >= datetime('now', '-' || ? || ' minutes')
            ORDER BY wd.detected_at DESC
        ''', (since_minutes,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'person_id': row['person_id'],
                'name': row['name'],
                'notes': row['notes'],
                'person_thumbnail': row['person_thumbnail'],
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'detected_at': row['detected_at'],
                'confidence': row['confidence'],
                'thumbnail_path': row['thumbnail_path'],
                'fullframe_path': row['fullframe_path']
            })
        return alerts

    def match_face_in_watchlist(self, embedding: np.ndarray, threshold: float = 0.4) -> Optional[Dict]:
        """Match a face embedding against watchlist embeddings"""
        watchlist_embeddings = self.get_all_embeddings()
        
        if not watchlist_embeddings:
            return None
        
        best_match = None
        best_similarity = 0
        
        for wl_person in watchlist_embeddings:
            wl_embedding = wl_person['embedding']
            # Cosine similarity
            similarity = np.dot(embedding, wl_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(wl_embedding) + 1e-8
            )
            
            if similarity > best_similarity and similarity >= (1 - threshold):
                best_similarity = similarity
                best_match = {
                    'person_id': wl_person['person_id'],
                    'name': wl_person['name'],
                    'similarity': float(similarity)
                }
        
        return best_match

    def get_person_location_history(self, person_id: str) -> List[Dict]:
        """Get location history timeline for a person"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                camera_id,
                camera_name,
                detected_at,
                thumbnail_path
            FROM watchlist_detections
            WHERE person_id = ?
            ORDER BY detected_at DESC
        ''', (person_id,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'camera_id': row['camera_id'],
                'camera_name': row['camera_name'],
                'detected_at': row['detected_at'],
                'thumbnail_path': row['thumbnail_path']
            })
        return history

    def get_categories(self) -> List[Dict]:
        """Get all watchlist categories with details"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                wc.name,
                wc.color,
                wc.description,
                wc.alert_priority,
                (SELECT COUNT(*) FROM watchlist WHERE category = wc.name) as person_count
            FROM watchlist_categories wc
            ORDER BY wc.alert_priority DESC, wc.name
        ''')
        
        categories = []
        for row in cursor.fetchall():
            categories.append({
                'name': row['name'] if isinstance(row, dict) else row[0],
                'color': row['color'] if isinstance(row, dict) else row[1],
                'description': row['description'] if isinstance(row, dict) else row[2],
                'alert_priority': row['alert_priority'] if isinstance(row, dict) else row[3],
                'person_count': row['person_count'] if isinstance(row, dict) else row[4]
            })
        return categories

    def add_category(self, name: str, color: str = '#808080', description: str = None, 
                     alert_priority: int = 1) -> bool:
        """Add a new watchlist category"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO watchlist_categories (name, color, description, alert_priority)
                VALUES (?, ?, ?, ?)
            ''', (name, color, description, alert_priority))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Category already exists

    def update_category(self, name: str, color: str = None, description: str = None,
                        alert_priority: int = None) -> bool:
        """Update an existing category"""
        cursor = self.conn.cursor()
        updates = []
        params = []
        
        if color is not None:
            updates.append('color = ?')
            params.append(color)
        if description is not None:
            updates.append('description = ?')
            params.append(description)
        if alert_priority is not None:
            updates.append('alert_priority = ?')
            params.append(alert_priority)
        
        if not updates:
            return False
        
        params.append(name)
        cursor.execute(f'''
            UPDATE watchlist_categories SET {', '.join(updates)} WHERE name = ?
        ''', params)
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_category(self, name: str) -> bool:
        """Delete a category (moves persons to 'general')"""
        cursor = self.conn.cursor()
        # Move persons to general first
        cursor.execute('UPDATE watchlist SET category = ? WHERE category = ?', ('general', name))
        cursor.execute('DELETE FROM watchlist_categories WHERE name = ?', (name,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self):
        """Close database connection"""
        self.conn.close()
