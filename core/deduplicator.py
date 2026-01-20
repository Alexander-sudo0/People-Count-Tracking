"""
Post-processing deduplication using clustering to merge similar faces.
Useful for handling duplicates from blur, low quality, and repeated detections.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from config import Config


class FaceDeduplicator:
    """Deduplicates faces in the database by clustering similar embeddings"""
    
    def __init__(self, similarity_threshold=0.70):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_faces(self, faces_list):
        """
        Cluster similar faces and merge duplicates.
        
        Args:
            faces_list: List of face dicts with 'embedding', 'id', 'count', 'quality'
        
        Returns:
            dict: Mapping of old_id -> primary_id (for deduplication)
            dict: Updated face records with merged counts
        """
        if not faces_list or len(faces_list) < 2:
            return {}, {f['id']: f for f in faces_list}
        
        # Filter faces with valid embeddings
        valid_faces = []
        for f in faces_list:
            emb = f.get('embedding')
            # Check if embedding exists and is not None
            if emb is None:
                continue
            # Convert to numpy array if needed
            if not isinstance(emb, np.ndarray):
                try:
                    emb = np.array(emb, dtype=np.float64)
                except (ValueError, TypeError):
                    continue
            else:
                emb = np.array(emb, dtype=np.float64)
            
            # Check for NaN or inf values
            if not np.all(np.isfinite(emb)):
                continue
            
            # Store the cleaned embedding back
            f['embedding'] = emb
            valid_faces.append(f)
        
        if not valid_faces or len(valid_faces) < 2:
            # Return faces as-is if not enough valid embeddings
            return {}, {f['id']: f for f in faces_list}
        
        # Extract embeddings into a 2D array
        embeddings = np.array([f['embedding'] for f in valid_faces], dtype=np.float64)
        
        # Ensure it's 2D
        if embeddings.ndim != 2:
            return {}, {f['id']: f for f in faces_list}
        
        # Normalize embeddings to ensure valid cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Ensure normalized embeddings are finite
        if not np.all(np.isfinite(embeddings)):
            return {}, {f['id']: f for f in faces_list}
        # Calculate distance matrix (1 - cosine similarity)
        # For normalized embeddings: distance = 1 - dot_product
        dot_product = np.dot(embeddings, embeddings.T)
        
        # Double-clamp to handle numerical precision issues
        # First clamp the dot product to [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate distances
        distances = 1.0 - dot_product
        
        # Ensure distances are in valid range [0, 2] (hard clamp to remove any negatives)
        distances = np.clip(distances, 0.0, 2.0)
        
        # Final check: ensure no negative values
        min_dist = np.min(distances)
        if min_dist < 0:
            print(f"[WARNING] Found negative distances ({min_dist}), clamping to 0")
            distances = np.maximum(distances, 0.0)
        
        # Ensure distance matrix is valid 2D float array
        distances = np.asarray(distances, dtype=np.float64)
        
        # Cluster using DBSCAN
        # eps = 1 - threshold (e.g., threshold 0.70 -> eps 0.30)
        eps = 1 - self.similarity_threshold
        
        try:
            clusterer = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
            labels = clusterer.fit_predict(distances)
            
            # Debug logging
            unique_labels = set(labels)
            num_clusters = len(unique_labels)
            if num_clusters > len(valid_faces) // 2:
                print(f"[DEBUG] DBSCAN clustering: {len(valid_faces)} faces -> {num_clusters} clusters (threshold={self.similarity_threshold}, eps={eps})")
                print(f"[DEBUG] Distance matrix stats: min={np.min(distances):.4f}, max={np.max(distances):.4f}, mean={np.mean(distances):.4f}")
        except Exception as e:
            print(f"[ERROR] DBSCAN failed: {e}")
            print(f"Distance matrix shape: {distances.shape}, dtype: {distances.dtype}")
            print(f"Distance range: [{np.min(distances)}, {np.max(distances)}]")
            print(f"Contains NaN: {np.any(np.isnan(distances))}")
            print(f"Contains inf: {np.any(np.isinf(distances))}")
            raise
        
        # Group faces by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_faces[idx])
        
        # Merge faces in each cluster
        id_mapping = {}  # old_id -> primary_id
        merged_faces = {}
        
        for cluster_id, cluster_faces in clusters.items():
            # Sort by quality (descending) and take best as primary
            sorted_faces = sorted(cluster_faces, key=lambda f: f.get('quality', 0), reverse=True)
            primary_face = sorted_faces[0]
            primary_id = primary_face['id']
            
            # Merge all faces in cluster into primary
            merged_count = 0
            merged_quality = primary_face.get('quality', 1.0)
            merged_embedding = primary_face['embedding'].copy()
            
            for face in sorted_faces:
                merged_count += face.get('count', 1)
                id_mapping[face['id']] = primary_id
                
                # Update embedding as weighted average (weighted by quality)
                if face['id'] != primary_id:
                    weight = face.get('quality', 0.5) / (1 + face.get('quality', 0.5))
                    merged_embedding = (1 - weight) * merged_embedding + weight * face['embedding']
            
            # Normalize merged embedding
            norm = np.linalg.norm(merged_embedding)
            if norm > 0:
                merged_embedding = merged_embedding / norm
            
            merged_faces[primary_id] = {
                'id': primary_id,
                'embedding': merged_embedding,
                'count': merged_count,
                'quality': merged_quality,
                'merged_count': len(sorted_faces),  # How many faces were merged
                'original_ids': [f['id'] for f in sorted_faces]
            }
        
        # Add faces without valid embeddings as-is, preserving all original data
        for face in faces_list:
            if face['id'] not in merged_faces:
                # Keep original face intact
                merged_faces[face['id']] = face
        
        return id_mapping, merged_faces
    
    def get_unique_count(self, merged_faces):
        """Get the total unique count from merged faces"""
        return sum(f.get('count', 1) for f in merged_faces.values())
    
    def deduplicate_with_threshold_analysis(self, faces_list, quality_threshold=None):
        """
        Deduplicate and also filter by quality.
        
        Args:
            faces_list: List of face dicts
            quality_threshold: Override default quality threshold
        
        Returns:
            dict: Deduplication results with statistics
        """
        if quality_threshold is None:
            quality_threshold = Config.MIN_FACE_QUALITY
        
        # Calculate total count before any processing
        total_count_before = sum(f.get('count', 1) for f in faces_list)
        
        # Filter by quality
        high_quality_faces = [f for f in faces_list if f.get('quality', 1.0) >= quality_threshold]
        low_quality_faces = [f for f in faces_list if f.get('quality', 1.0) < quality_threshold]
        
        # Deduplicate high quality faces
        id_mapping, merged_high = self.deduplicate_faces(high_quality_faces)
        
        # Build final merged_faces dict
        merged_faces = {}
        
        # Add merged high quality faces
        merged_faces.update(merged_high)
        
        # Add low quality faces (as-is, no merging)
        for low_face in low_quality_faces:
            if low_face['id'] not in merged_faces:
                merged_faces[low_face['id']] = low_face
        
        # Calculate total count after processing
        total_count_after = sum(f.get('count', 1) for f in merged_faces.values())
        
        duplicates_removed = len(faces_list) - len(merged_faces)
        
        # Sanity check: if no duplicates were removed, counts should be equal
        if duplicates_removed == 0 and total_count_before != total_count_after:
            print(f"[WARNING] No duplicates removed but count changed: {total_count_before} -> {total_count_after}")
            # If this happens, use the before count
            total_count_after = total_count_before
        
        return {
            'merged_faces': merged_faces,
            'id_mapping': id_mapping,
            'high_quality_count': len(high_quality_faces),
            'low_quality_count': len(low_quality_faces),
            'unique_before': len(faces_list),
            'unique_after': len(merged_faces),
            'duplicates_removed': duplicates_removed,
            'total_count_before': total_count_before,
            'total_count_after': total_count_after
        }
