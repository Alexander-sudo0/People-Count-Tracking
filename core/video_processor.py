import cv2
import numpy as np
import time
import uuid
from .recognizer import IFaceRecognizer
from .storage import IStorage
from .face_clustering import FaceClusterManager
from config import Config


class VideoProcessor:
    """Process offline video files to extract unique faces."""

    def __init__(self, recognizer: IFaceRecognizer, storage: IStorage):
        self.recognizer = recognizer
        self.storage = storage
        # Use same thresholds as live processing
        self.min_face_score = Config.MIN_FACE_QUALITY
        self.blur_threshold = Config.BLUR_DETECTION_THRESHOLD
        # Clustering system for grouping same person
        self.cluster_manager = FaceClusterManager(similarity_threshold=Config.SIMILARITY_THRESHOLD)

    def _calculate_similarity(self, embedding1, embedding2):
        """Computes Cosine Similarity assuming inputs are normalized."""
        return float(np.dot(embedding1, embedding2))

    def _iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes [x1,y1,x2,y2]"""
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0

        boxAArea = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        boxBArea = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def _calculate_blur(self, frame, bbox):
        """Calculate blur metric using Laplacian variance. Same as live processing."""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                return 0.0
            
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, max(0.0, laplacian_var / 50.0))
            return blur_score
        except Exception:
            return 0.5

    def _dedupe_faces(self, faces):
        """Simple NMS-like dedupe using IoU and detection score."""
        if not faces:
            return []
        ordered = sorted(faces, key=lambda f: getattr(f, 'det_score', 0.0), reverse=True)
        picked = []
        for f in ordered:
            skip = False
            fb = np.array(f.bbox).astype(float)
            for p in picked:
                pb = np.array(p.bbox).astype(float)
                if self._iou(fb, pb) > Config.IOU_DEDUP_THRESHOLD:
                    skip = True
                    break
            if not skip:
                picked.append(f)
        return picked

    def process_video(self, video_path, process_every_n_frames=5):
        """
        Process a video file and extract unique faces.
        
        Returns:
            {
                'total_frames': int,
                'unique_faces': [
                    {'id': str, 'count': int, 'image': base64_str},
                    ...
                ]
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_count = 0
        unique_faces_cache = []  # [{id, embedding, count, image_path}, ...]
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue

                # Detect faces in frame
                faces = self.recognizer.get_faces(frame)
                faces = self._dedupe_faces(faces)

                for face in faces:
                    det_score = getattr(face, 'det_score', 0.0)
                    
                    # Filter by quality threshold (same as live processing)
                    if det_score < self.min_face_score:
                        continue
                    
                    # Filter by blur (same as live processing)
                    blur_score = self._calculate_blur(frame, face.bbox)
                    if blur_score < self.blur_threshold:
                        continue

                    embedding = np.array(face.embedding, dtype=float)
                    norm = np.linalg.norm(embedding)
                    if norm == 0:
                        continue
                    embedding = embedding / norm

                    # Find match in cache
                    best_score = -1
                    best_idx = -1
                    for idx, cached in enumerate(unique_faces_cache):
                        score = self._calculate_similarity(embedding, cached['embedding'])
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx >= 0 and best_score > Config.SIMILARITY_THRESHOLD:
                        # Increment existing face count
                        unique_faces_cache[best_idx]['count'] += 1
                    else:
                        # New unique face
                        face_id = str(uuid.uuid4())[:8]
                        img_path = self._save_face_image(frame, face.bbox, face_id)
                        unique_faces_cache.append({
                            'id': face_id,
                            'embedding': embedding,
                            'count': 1,
                            'image_path': img_path
                        })

        finally:
            cap.release()

        # Cluster faces to find unique people (post-processing)
        # Add all faces to cluster manager for final clustering
        now = time.time()
        for face in unique_faces_cache:
            self.cluster_manager.add_face_event(
                face_id=face['id'],
                embedding=face['embedding'],
                quality=1.0,  # All passed quality/blur checks
                confidence=1.0,
                timestamp=now,
                image_path=face['image_path']
            )
        
        # Merge similar clusters (aggressive merging for batch processing)
        self.cluster_manager.merge_similar_clusters(merge_threshold=Config.DEDUP_SIMILARITY_THRESHOLD)
        
        # Get cluster summary
        cluster_summary = self.cluster_manager.get_cluster_summary()
        
        # Build response with cluster information
        unique_faces = []
        for f in unique_faces_cache:
            img_url = None
            if f.get('image_path'):
                import os
                img_url = f"/static/faces/{os.path.basename(f['image_path'])}"
            
            # Get cluster this face belongs to
            cluster_id = self.cluster_manager.face_to_cluster.get(f['id'], 'unknown')
            
            unique_faces.append({
                'id': f['id'],
                'cluster_id': cluster_id,
                'count': f['count'],
                'image': img_url
            })

        # Sort by count descending
        unique_faces.sort(key=lambda x: x['count'], reverse=True)

        return {
            'total_frames': frame_count,
            'unique_faces': unique_faces,
            'clusters': cluster_summary,
            'unique_people_count': len(cluster_summary),  # Unique count = number of clusters
            'total_face_events': len(unique_faces_cache)
        }

    def _save_face_image(self, frame, bbox, face_id):
        """Save face crop to disk with padding."""
        if not Config.SAVE_IMAGES:
            return None
        
        filename = f"{Config.IMAGE_OUTPUT_DIR}/video_face_{face_id}_{int(time.time())}.jpg"
        
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
