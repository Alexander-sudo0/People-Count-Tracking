# Face Clustering & Enhanced Display Implementation

## Overview
This implementation provides comprehensive face detection, clustering, and detailed visualization of identified people in the peoplecount system. Similar faces are automatically grouped into clusters representing unique individuals.

---

## 1. Backend Components

### A. Face Clustering System (`core/face_clustering.py`)
- **Purpose**: Groups detected faces into clusters representing unique people
- **Key Features**:
  - Similarity-based clustering using cosine distance
  - Tracks cluster metadata: face_count, quality_avg, first_seen, last_seen
  - Assigns unique cluster_id to group faces of the same person
  - Provides API methods: `add_face_event()`, `get_unique_count()`, `get_cluster_summary()`

### B. Enhanced Counter (`core/counter.py`)
- **Integrated FaceClusterManager**: All detected faces are assigned to clusters
- **Detailed Logging System**:
  - `_log_face_detection()`: Logs detection events with quality and confidence scores
  - `_log_matching_attempt()`: Logs each face matching attempt with similarity scores
  - Console output shows real-time detection and matching information

- **Face Record Enhancement**: Each face now includes:
  - `cluster_id`: Which cluster (unique person) this face belongs to
  - `confidence`: Detection confidence score (0-1)
  - `quality`: Face quality/blur metric (0-1)
  - `timestamp`: Exact time of detection
  - `image_path`: Path to saved face image

- **Matching Logic Improvements**:
  - Gallery-based matching with multiple poses per person
  - Spatial continuity tracking to maintain ID persistence
  - Dynamic quality threshold applied per-frame

### C. Configuration Management (`config.py`)
- **Persistent Settings**:
  - `MIN_FACE_QUALITY`: Minimum acceptable detection quality (default: 0.70)
  - `BLUR_DETECTION_THRESHOLD`: Threshold for blur detection (default: 0.60)
  - `DEDUP_SIMILARITY_THRESHOLD`: Post-processing threshold for dedup (default: 0.65)
  - `SIMILARITY_THRESHOLD`: Real-time matching threshold (default: 0.75)

- **Dynamic Application**: Settings applied on each frame, allowing changes without restart

### D. Enhanced REST API (`web_app.py`)
- **Endpoint: `/faces/{cam_id}`**
  - Returns detailed face information grouped by clusters
  - Response structure:
    ```json
    {
      "faces": [
        {
          "face_id": "abc123",
          "cluster_id": "cluster_0",
          "image": "/static/faces/abc123.jpg",
          "quality": 0.92,
          "confidence": 0.95,
          "timestamp": 1234567890,
          "captured_at": "2026-01-11 14:30:45"
        }
      ],
      "clusters": [
        {
          "cluster_id": "cluster_0",
          "face_count": 3,
          "quality_avg": 0.90,
          "first_seen": "2026-01-11 14:20:00",
          "last_seen": "2026-01-11 14:35:00"
        }
      ],
      "unique_people_count": 5,
      "total_face_events": 12
    }
    ```

---

## 2. Frontend Display (`templates/dashboard.html`)

### Enhanced Face Viewing Panel
- **Panel Title**: "Identified Faces (Clustered)"
- **Summary Statistics**:
  - Unique People: Number of distinct clusters
  - Total Face Events: Total face detections

- **Clustered Display**:
  - Faces grouped by cluster_id
  - Each cluster shows:
    - Cluster ID
    - Number of detections in cluster
    - Average quality score
    - First and last detection time
  
  - Each face shows:
    - Face thumbnail image
    - Cluster ID
    - Quality score
    - Confidence score
    - Capture date and time (formatted as YYYY-MM-DD HH:MM:SS)

### UI Styling
- Color-coded sections:
  - **Blue**: Summary statistics
  - **Green**: Cluster headers
  - **Gray**: Individual face cards
- Responsive grid layout with full-width face images
- Increased panel height for better visibility

---

## 3. Logging & Debugging

### Console Logging Output Examples
```
[2026-01-11 14:30:45] FILTERED_OUT | Face ID: face_0 | Quality: 0.55 | Confidence: 0.55 | Below quality threshold (0.70)
[2026-01-11 14:30:46] MATCH_ATTEMPT | Detected: face_1 | vs Existing: abc123 | Similarity: 0.82 | Result: MATCH
[2026-01-11 14:30:46] MATCHED | Face ID: abc123 | Quality: 0.92 | Confidence: 0.92 | Similarity: 0.82
[2026-01-11 14:30:46] COUNT_INCREMENT | Face ID: abc123 | Quality: 0.92 | Confidence: 0.92 | Total count: 2
[2026-01-11 14:30:47] NEW_PERSON | Face ID: xyz789 | Quality: 0.88 | Confidence: 0.88 | First detection (Cluster ID will be assigned)
```

### Debug Information Captured
- Face detection quality and confidence scores
- Matching attempts with similarity scores
- Cluster assignments and grouping decisions
- Count increment reasons and timing
- Filtered-out faces and reasons (quality, blur, etc.)

---

## 4. Key Metrics & Terminology

- **Face Event**: A single detection of a face in a frame
- **Cluster**: A group of face events determined to be the same person
- **Unique People Count**: Number of clusters = number of unique individuals detected
- **Quality Score**: Measure of face clarity/blur (0-1)
- **Confidence Score**: Detection confidence from face recognition model (0-1)
- **Cluster ID**: Unique identifier for a group of similar faces (e.g., "cluster_0")

---

## 5. Configuration Example

### Adjusting Sensitivity (via Settings Panel)
- **Increase Min Face Quality** → More stringent, fewer blurry faces captured
- **Decrease Min Face Quality** → More lenient, captures more faces including lower quality
- **Adjust Blur Detection Threshold** → Controls blur sensitivity
- **Adjust Dedup Similarity** → Controls how aggressive duplicate detection is

All settings persist in `app_settings.json` and apply immediately without restart.

---

## 6. Technical Flow

### Face Detection → Clustering → Display
```
1. Frame input to recognizer.get_faces()
   ↓
2. Filter by quality threshold (Config.MIN_FACE_QUALITY)
   ↓
3. Deduplicate overlapping detections in same frame
   ↓
4. Match against active faces using cosine similarity (threshold: 0.70)
   ↓
5. For matches: Update embedding via rolling average, add to gallery
   ↓
6. For new faces: Generate new face_id, assign cluster_id, save image
   ↓
7. All face events logged with timestamps, quality, confidence
   ↓
8. Cluster manager groups similar faces (threshold: 0.75)
   ↓
9. API endpoint /faces/{cam_id} returns:
   - Individual face events with metadata
   - Cluster summaries with grouping info
   - Unique people count (= number of clusters)
   ↓
10. Dashboard displays faces grouped by cluster with all metadata
```

---

## 7. Database Schema

### Face Record (in faces_db.json)
```json
{
  "id": "abc123",
  "cluster_id": "cluster_0",
  "embedding": [normalized 512-d array],
  "timestamp": 1234567890,
  "image_path": "detected_faces/abc123.jpg",
  "quality": 0.92,
  "confidence": 0.95,
  "count": 3,
  "camera_id": "camera_1"
}
```

### Cluster Data (managed in memory by FaceClusterManager)
```json
{
  "cluster_id": "cluster_0",
  "face_ids": ["id1", "id2", "id3"],
  "quality_avg": 0.90,
  "first_seen": 1234567800,
  "last_seen": 1234567900,
  "face_count": 3
}
```

---

## 8. Usage Examples

### Viewing Faces Dashboard
1. Select a camera from the dropdown
2. Click "View Faces" button
3. See summary: "Unique People: 5, Total Face Events: 12"
4. Browse faces grouped by cluster:
   - "cluster_0": 3 detections, avg quality 0.90, first/last times
   - "cluster_1": 2 detections, avg quality 0.88, first/last times
   - etc.

### Interpreting Face Details
```
Cluster: cluster_0
Detections: 3 | Avg Quality: 0.90
First: 2026-01-11 14:20:00 | Last: 2026-01-11 14:35:00

Face 1: Quality: 0.92 | Conf: 0.95 | Captured: 2026-01-11 14:20:15
Face 2: Quality: 0.89 | Conf: 0.93 | Captured: 2026-01-11 14:27:30
Face 3: Quality: 0.88 | Conf: 0.92 | Captured: 2026-01-11 14:35:00
```

---

## 9. Troubleshooting

### Issue: Too many faces being created
- **Solution**: Increase `MIN_FACE_QUALITY` in settings to filter blurry faces

### Issue: Same person split into multiple clusters
- **Solution**: Check `SIMILARITY_THRESHOLD` in config (lower = more aggressive matching)

### Issue: Faces not being counted
- **Check**:
  1. Is face quality above threshold?
  2. Is face within ROI (if ROI is set)?
  3. Has cooldown period elapsed (MIN_REPEAT_SECONDS)?
  4. Check console logs for detailed reason

### Issue: Dashboard showing zero unique people
- **Check**: 
  1. Camera is selected
  2. Faces are being detected (check console logs)
  3. Cluster manager has initialized faces (should happen on first detection)

---

## 10. Future Enhancements

Possible improvements:
- Export face clusters as groups for batch processing
- Manual cluster merging/splitting UI
- Historical cluster tracking across sessions
- Person identification with labels
- Advanced filtering (by date range, quality range, etc.)
- Cluster comparison visualization

---

## Implementation Completion

✅ **Backend**: Face clustering, enhanced metadata, API endpoints
✅ **Frontend**: Dashboard display with cluster grouping, detailed face info
✅ **Logging**: Comprehensive console logging of detections and matching
✅ **Configuration**: Persistent settings with dynamic application
✅ **Database**: Face records with cluster_id, quality, confidence, timestamp

**Status**: Feature complete and ready for testing.
