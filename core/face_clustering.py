"""
Face clustering system to group detected face events by similarity.
A cluster represents a unique person detected multiple times.
"""
import numpy as np
from collections import defaultdict
import time


class FaceClusterManager:
    """Manages clustering of face events into groups of the same person"""
    
    def __init__(self, similarity_threshold=0.70):
        self.similarity_threshold = similarity_threshold
        self.clusters = {}  # cluster_id -> list of face events
        self.face_to_cluster = {}  # face_id -> cluster_id
        self.next_cluster_id = 0
    
    def add_face_event(self, face_id, embedding, quality, confidence, timestamp, image_path=None):
        """
        Add a face event and assign it to a cluster.
        
        Returns:
            cluster_id: The cluster this face belongs to
        """
        if embedding is None:
            # Unclusterable - create its own cluster
            cluster_id = self._create_cluster()
        else:
            # Try to find matching cluster
            cluster_id = self._find_matching_cluster(embedding)
            if cluster_id is None:
                cluster_id = self._create_cluster()
        
        # Store face-to-cluster mapping
        self.face_to_cluster[face_id] = cluster_id
        
        # Add face event to cluster
        face_event = {
            'face_id': face_id,
            'embedding': embedding,
            'quality': quality,
            'confidence': confidence,
            'timestamp': timestamp,
            'image_path': image_path,
            'added_at': time.time()
        }
        
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        
        self.clusters[cluster_id].append(face_event)
        
        return cluster_id
    
    def _create_cluster(self):
        """Create a new cluster"""
        cluster_id = f"cluster_{self.next_cluster_id}"
        self.next_cluster_id += 1
        return cluster_id
    
    def _find_matching_cluster(self, embedding):
        """Find a cluster whose faces are similar to this embedding"""
        if embedding is None or len(self.clusters) == 0:
            return None
        
        best_cluster = None
        best_similarity = 0
        
        # Check each cluster
        for cluster_id, face_events in self.clusters.items():
            # Compare with best face in cluster
            for face_event in face_events:
                cluster_embedding = face_event['embedding']
                if cluster_embedding is None:
                    continue
                
                similarity = self._calculate_similarity(embedding, cluster_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
        
        # Only assign to cluster if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            return best_cluster
        
        return None
    
    def _calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        if emb1 is None or emb2 is None:
            return 0
        
        emb1 = np.array(emb1, dtype=np.float64)
        emb2 = np.array(emb2, dtype=np.float64)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def get_unique_count(self):
        """Get number of unique clusters (unique people)"""
        return len(self.clusters)
    
    def get_cluster_faces(self, cluster_id):
        """Get all face events in a cluster"""
        return self.clusters.get(cluster_id, [])
    
    def get_all_clusters(self):
        """Get all clusters with their face events"""
        return self.clusters
    
    def get_cluster_summary(self):
        """Get summary of all clusters"""
        summary = []
        for cluster_id, face_events in self.clusters.items():
            summary.append({
                'cluster_id': cluster_id,
                'face_count': len(face_events),
                'quality_avg': np.mean([f['quality'] for f in face_events]) if face_events else 0,
                'first_seen': min([f['timestamp'] for f in face_events]) if face_events else 0,
                'last_seen': max([f['timestamp'] for f in face_events]) if face_events else 0,
            })
        return summary
    
    def merge_similar_clusters(self, merge_threshold=0.65):
        """
        Post-processing: Merge clusters whose faces are very similar.
        This helps catch same person split into multiple clusters.
        
        Args:
            merge_threshold: Similarity threshold for merging (lower = more aggressive)
        """
        if len(self.clusters) < 2:
            return  # Nothing to merge
        
        cluster_ids = list(self.clusters.keys())
        merges = []  # Track which clusters to merge
        
        # Compare all cluster pairs
        for i, cluster_a in enumerate(cluster_ids):
            if cluster_a not in self.clusters:
                continue  # Already merged
                
            for cluster_b in cluster_ids[i+1:]:
                if cluster_b not in self.clusters:
                    continue  # Already merged
                
                # Get best similarity between clusters
                best_sim = self._get_cluster_similarity(cluster_a, cluster_b)
                
                if best_sim >= merge_threshold:
                    # Merge cluster_b into cluster_a
                    print(f"[CLUSTER_MERGE] Merging {cluster_b} into {cluster_a} (similarity: {best_sim:.3f})")
                    
                    # Move all faces from cluster_b to cluster_a
                    self.clusters[cluster_a].extend(self.clusters[cluster_b])
                    
                    # Update face_to_cluster mapping
                    for face_event in self.clusters[cluster_b]:
                        self.face_to_cluster[face_event['face_id']] = cluster_a
                    
                    # Remove cluster_b
                    del self.clusters[cluster_b]
    
    def _get_cluster_similarity(self, cluster_a, cluster_b):
        """
        Get the best similarity between two clusters.
        Uses the most similar pair of faces from each cluster.
        """
        if cluster_a not in self.clusters or cluster_b not in self.clusters:
            return 0
        
        best_similarity = 0
        
        for face_a in self.clusters[cluster_a]:
            if face_a['embedding'] is None:
                continue
            for face_b in self.clusters[cluster_b]:
                if face_b['embedding'] is None:
                    continue
                
                sim = self._calculate_similarity(face_a['embedding'], face_b['embedding'])
                if sim > best_similarity:
                    best_similarity = sim
        
        return best_similarity
