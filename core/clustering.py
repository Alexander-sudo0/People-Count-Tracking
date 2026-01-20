"""
Advanced clustering and deduplication for face recognition.
Uses DBSCAN-like approach for better grouping of similar faces.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats


class FaceClustering:
    """Clusters similar faces using improved algorithms"""
    
    def __init__(self, similarity_threshold=0.75, min_cluster_size=2):
        """
        Args:
            similarity_threshold: Minimum similarity score to consider faces as same person (0.0-1.0)
                                 Stricter: 0.75+ (fewer false positives)
                                 Lenient: 0.65- (more matches but risk of false positives)
            min_cluster_size: Minimum faces in a group to consider it valid
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.face_clusters = {}  # {cluster_id: [face_ids]}
        self.face_to_cluster = {}  # {face_id: cluster_id}
        self.cluster_representatives = {}  # {cluster_id: representative_embedding}

    def calculate_similarity(self, emb1, emb2):
        """Cosine similarity between two normalized embeddings"""
        return float(np.dot(emb1, emb2))

    def cluster_faces(self, faces):
        """
        Cluster faces using DBSCAN-like approach.
        
        Args:
            faces: List of dicts with keys: 'id', 'embedding', 'bbox', 'score'
        
        Returns:
            List of cluster groups with representative info
        """
        if len(faces) < 2:
            # Single face, no clustering needed
            return [{
                'representative_id': faces[0]['id'],
                'representative_embedding': faces[0]['embedding'],
                'member_ids': [faces[0]['id']],
                'size': 1,
                'avg_score': faces[0].get('score', 0.9)
            }] if faces else []

        # Build similarity matrix
        embeddings = np.array([f['embedding'] for f in faces])
        n = len(embeddings)
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Cluster using connectivity
        clusters = self._dbscan_clustering(similarity_matrix, faces)
        return clusters

    def _dbscan_clustering(self, similarity_matrix, faces):
        """
        DBSCAN-like clustering on similarity matrix.
        Points are similar faces, clusters are groups of the same person.
        """
        n = len(faces)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            # Find all faces similar to this one
            cluster_indices = [i]
            visited[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                # Find neighbors (similar faces)
                for j in range(n):
                    if not visited[j] and similarity_matrix[current, j] >= self.similarity_threshold:
                        visited[j] = True
                        cluster_indices.append(j)
                        queue.append(j)

            # Create cluster if valid
            if len(cluster_indices) >= 1:  # Even single faces are clusters
                cluster_faces = [faces[idx] for idx in cluster_indices]
                cluster_info = self._create_cluster_info(cluster_faces)
                clusters.append(cluster_info)

        return clusters

    def _create_cluster_info(self, cluster_faces):
        """Create representative info for a cluster"""
        embeddings = np.array([f['embedding'] for f in cluster_faces])
        scores = np.array([f.get('score', 0.9) for f in cluster_faces])
        
        # Use highest quality face as representative
        best_idx = np.argmax(scores)
        representative_embedding = embeddings[best_idx]
        representative_id = cluster_faces[best_idx]['id']
        
        # Compute average embedding for better matching
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        return {
            'representative_id': representative_id,
            'representative_embedding': representative_embedding,
            'average_embedding': avg_embedding,
            'member_ids': [f['id'] for f in cluster_faces],
            'size': len(cluster_faces),
            'avg_score': float(np.mean(scores)),
            'face_data': cluster_faces
        }

    def merge_clusters(self, cluster1, cluster2):
        """Merge two clusters if they're similar enough"""
        emb1 = cluster1['average_embedding']
        emb2 = cluster2['average_embedding']
        sim = self.calculate_similarity(emb1, emb2)
        
        if sim >= self.similarity_threshold:
            # Merge: use better quality as representative
            if cluster1['avg_score'] >= cluster2['avg_score']:
                merged = cluster1.copy()
                merged['member_ids'].extend(cluster2['member_ids'])
                merged['size'] = len(merged['member_ids'])
            else:
                merged = cluster2.copy()
                merged['member_ids'].extend(cluster1['member_ids'])
                merged['size'] = len(merged['member_ids'])
            return merged
        return None


class ImprovedDeduplication:
    """Improved deduplication with quality filtering and temporal consistency"""
    
    def __init__(self, similarity_threshold=0.75, min_face_quality=0.70):
        """
        Args:
            similarity_threshold: Stricter threshold for face matching (default 0.75)
            min_face_quality: Minimum face detection score to consider (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.min_face_quality = min_face_quality
        self.clustering = FaceClustering(similarity_threshold)

    def filter_low_quality_faces(self, faces):
        """Remove low-quality detections"""
        return [f for f in faces if f.get('det_score', 0.0) >= self.min_face_quality]

    def deduplicate_by_bbox(self, faces, iou_threshold=0.5):
        """
        Remove overlapping bboxes (likely same face detected multiple times).
        Keep the highest confidence detection.
        """
        if not faces:
            return []

        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        filtered = []

        for face in faces:
            skip = False
            bbox1 = face.bbox
            for kept_face in filtered:
                bbox2 = kept_face.bbox
                iou = self._iou(bbox1, bbox2)
                if iou > iou_threshold:
                    skip = True
                    break
            if not skip:
                filtered.append(face)

        return filtered

    def _iou(self, bbox1, bbox2):
        """Compute IoU between two bboxes"""
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

    def adaptive_threshold(self, detection_confidence):
        """
        Adapt similarity threshold based on detection confidence.
        Higher confidence = stricter matching required.
        """
        # Confidence 0.6 -> threshold 0.72
        # Confidence 0.95 -> threshold 0.78
        base_threshold = 0.75
        confidence_factor = max(0, (detection_confidence - 0.6) * 0.1)
        return min(0.80, base_threshold + confidence_factor)
