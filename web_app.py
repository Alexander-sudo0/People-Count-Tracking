import cv2
import numpy as np
import uvicorn
import time
from fastapi import FastAPI, Request, Form, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from core.camera_manager import CameraManager
from core.video_processor import VideoProcessor
from config import Config
import os
import tempfile
import logging
from pathlib import Path

# Get the absolute path to the script directory
BASE_DIR = Path(__file__).resolve().parent

# Add cuDNN to PATH for GPU support (Windows)
cudnn_path = r"C:\Program Files\NVIDIA\CUDNN\v9.14\bin\12.9"
if os.path.exists(cudnn_path) and cudnn_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')
    print(f"[INFO] Added cuDNN to PATH: {cudnn_path}")

# Suppress FFmpeg/OpenCV verbose logging to reduce HEVC codec warnings
os.environ['FFREPORT'] = 'file=/dev/null'
logging.getLogger('cv2').setLevel(logging.WARNING)

# Suppress OpenCV build warnings
cv2.setLogLevel(0)

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Ensure output directories exist before mounting static files
Config.setup_dirs()
# Serve face crops as static files
app.mount("/static/faces", StaticFiles(directory=Config.IMAGE_OUTPUT_DIR), name="faces")

# Initialize Manager (Global State)
# In production, use a Dependency Injection container
# Enable GPU for faster processing
manager = CameraManager(use_gpu=True) 

# Initialize Video Processor
video_processor = VideoProcessor(manager.recognizer, manager.storage)

# Load persisted settings (if any)
Config.load_settings()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "cameras": manager.get_active_cameras()
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    # Provide current settings to template
    settings = {
        'SIMILARITY_THRESHOLD': Config.SIMILARITY_THRESHOLD,
        'IOU_DEDUP_THRESHOLD': Config.IOU_DEDUP_THRESHOLD,
        'MIN_REPEAT_SECONDS': Config.MIN_REPEAT_SECONDS,
        'TIME_WINDOW_SECONDS': Config.TIME_WINDOW_SECONDS
    }
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})

@app.get("/video_analysis", response_class=HTMLResponse)
async def video_analysis_page(request: Request):
    return templates.TemplateResponse("video_analysis.html", {"request": request})

@app.get("/api/settings")
async def get_settings():
    return JSONResponse({
        'SIMILARITY_THRESHOLD': Config.SIMILARITY_THRESHOLD,
        'MIN_FACE_QUALITY': Config.MIN_FACE_QUALITY,
        'BLUR_DETECTION_THRESHOLD': Config.BLUR_DETECTION_THRESHOLD,
        'IOU_DEDUP_THRESHOLD': Config.IOU_DEDUP_THRESHOLD,
        'MIN_REPEAT_SECONDS': Config.MIN_REPEAT_SECONDS,
        'TIME_WINDOW_SECONDS': Config.TIME_WINDOW_SECONDS,
        'USE_POST_PROCESSING': Config.USE_POST_PROCESSING,
        'BBOX_PADDING_TOP': Config.BBOX_PADDING_TOP,
        'BBOX_PADDING_BOTTOM': Config.BBOX_PADDING_BOTTOM,
        'BBOX_PADDING_LEFT': Config.BBOX_PADDING_LEFT,
        'BBOX_PADDING_RIGHT': Config.BBOX_PADDING_RIGHT,
        'ENABLE_PERSON_DETECTION': Config.ENABLE_PERSON_DETECTION if hasattr(Config, 'ENABLE_PERSON_DETECTION') else True,
        'PERSON_DETECTION_CONFIDENCE': Config.PERSON_DETECTION_CONFIDENCE if hasattr(Config, 'PERSON_DETECTION_CONFIDENCE') else 0.5
    })

@app.put("/api/camera/{cam_id}/person_detection")
async def set_person_detection(cam_id: str, request: Request):
    """Enable/disable person detection for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        data = await request.json()
        enabled = data.get('enabled', True)
        
        cam = manager.cameras[cam_id]
        counter = cam['counter']
        counter.person_detection_enabled = bool(enabled)
        
        return JSONResponse({
            'status': 'success',
            'person_detection_enabled': counter.person_detection_enabled
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/cameras")
async def get_cameras():
    """Get list of all active cameras"""
    cameras = []
    for cam_id, cam in manager.cameras.items():
        cameras.append({
            'id': cam_id,
            'name': cam.get('name', cam_id),
            'counting_enabled': cam.get('counting_enabled', True),
            'camera_type': cam.get('camera_type', 'entry')
        })
    return JSONResponse({'cameras': cameras})

@app.post("/api/settings")
async def update_settings(
    similarity_threshold: float = Form(None),
    min_face_quality: float = Form(None),
    blur_threshold: float = Form(None),
    bbox_padding_top: float = Form(None),
    bbox_padding_bottom: float = Form(None),
    bbox_padding_left: float = Form(None),
    bbox_padding_right: float = Form(None)
):
    """Update face quality, similarity, and bbox padding settings"""
    settings_to_save = {}
    
    if similarity_threshold is not None and 0.0 <= similarity_threshold <= 1.0:
        Config.SIMILARITY_THRESHOLD = similarity_threshold
        settings_to_save['SIMILARITY_THRESHOLD'] = similarity_threshold
    if min_face_quality is not None and 0.0 <= min_face_quality <= 1.0:
        Config.MIN_FACE_QUALITY = min_face_quality
        settings_to_save['MIN_FACE_QUALITY'] = min_face_quality
    if blur_threshold is not None and 0.0 <= blur_threshold <= 1.0:
        Config.BLUR_DETECTION_THRESHOLD = blur_threshold
        settings_to_save['BLUR_DETECTION_THRESHOLD'] = blur_threshold
    if bbox_padding_top is not None and 0.0 <= bbox_padding_top <= 1.0:
        Config.BBOX_PADDING_TOP = bbox_padding_top
        settings_to_save['BBOX_PADDING_TOP'] = bbox_padding_top
    if bbox_padding_bottom is not None and 0.0 <= bbox_padding_bottom <= 1.0:
        Config.BBOX_PADDING_BOTTOM = bbox_padding_bottom
        settings_to_save['BBOX_PADDING_BOTTOM'] = bbox_padding_bottom
    if bbox_padding_left is not None and 0.0 <= bbox_padding_left <= 1.0:
        Config.BBOX_PADDING_LEFT = bbox_padding_left
        settings_to_save['BBOX_PADDING_LEFT'] = bbox_padding_left
    if bbox_padding_right is not None and 0.0 <= bbox_padding_right <= 1.0:
        Config.BBOX_PADDING_RIGHT = bbox_padding_right
        settings_to_save['BBOX_PADDING_RIGHT'] = bbox_padding_right
    
    # Persist settings
    if settings_to_save:
        Config.save_settings(settings_to_save)
        print(f"[INFO] Settings saved: {settings_to_save}")
    
    return JSONResponse({'status': 'settings updated'})

@app.post("/api/deduplicate")
async def deduplicate_faces(cam_id: str = None, quality_threshold: float = None):
    """
    Post-process faces to deduplicate using clustering.
    Merges similar faces and returns statistics.
    """
    from core.deduplicator import FaceDeduplicator
    
    try:
        deduplicator = FaceDeduplicator(similarity_threshold=Config.DEDUP_SIMILARITY_THRESHOLD)
        
        # Get all faces (from all cameras if cam_id not specified)
        if cam_id and cam_id in manager.cameras:
            cam = manager.cameras[cam_id]
            # Get active faces from counter
            faces_data = []
            for face in cam['counter'].active_faces:
                embedding = face.get('embedding')
                # Convert to numpy array if it's a list
                if embedding is not None and not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                faces_data.append({
                    'id': face.get('id'),
                    'embedding': embedding,
                    'count': face.get('count', 1),
                    'quality': face.get('quality', 1.0)
                })
        else:
            # Get all faces from all cameras
            faces_data = []
            for cam in manager.cameras.values():
                for face in cam['counter'].active_faces:
                    embedding = face.get('embedding')
                    # Convert to numpy array if it's a list
                    if embedding is not None and not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)
                    faces_data.append({
                        'id': face.get('id'),
                        'embedding': embedding,
                        'count': face.get('count', 1),
                        'quality': face.get('quality', 1.0)
                    })
        
        if not faces_data:
            return JSONResponse({'error': 'No faces found'}, status_code=400)
        
        print(f"[INFO] Running deduplication on {len(faces_data)} faces with threshold={Config.SIMILARITY_THRESHOLD}")
        
        # Run deduplication
        result = deduplicator.deduplicate_with_threshold_analysis(
            faces_data, 
            quality_threshold=quality_threshold
        )
        
        # Debug logging
        unique_before = result['unique_before']
        unique_after = result['unique_after']
        total_before = result['total_count_before']
        total_after = result['total_count_after']
        
        if unique_before == unique_after and total_before != total_after:
            print(f"[WARNING] Count mismatch: No dedup but count changed!")
            print(f"  Before: unique={unique_before}, total={total_before}")
            print(f"  After: unique={unique_after}, total={total_after}")
            print(f"  Faces data: {[(f['id'], f.get('count', '?')) for f in faces_data]}")
            print(f"  Merged faces: {[(f['id'], f.get('count', '?')) for f in result['merged_faces'].values()]}")
        
        return JSONResponse({
            'status': 'success',
            'statistics': {
                'unique_before': result['unique_before'],
                'unique_after': result['unique_after'],
                'duplicates_removed': result['duplicates_removed'],
                'total_count_before': result['total_count_before'],
                'total_count_after': result['total_count_after'],
                'high_quality_faces': result['high_quality_count'],
                'low_quality_faces': result['low_quality_count'],
                'deduplication_rate': f"{(result['duplicates_removed'] / result['unique_before'] * 100):.1f}%" if result['unique_before'] > 0 else "0%"
            },
            'merged_summary': {
                'merged_into_groups': len(result['merged_faces']),
                'some_faces_merged': result['duplicates_removed'] > 0
            }
        })
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Deduplication failed: {error_msg}")
        traceback.print_exc()
        return JSONResponse({'error': error_msg}, status_code=500)

@app.post("/api/reset_frequency")
async def reset_frequency(cam_id: str = None):
    """Reset the frequency count for all cameras or a specific camera"""
    try:
        cameras_reset = 0
        
        if cam_id and cam_id in manager.cameras:
            # Reset specific camera
            cam = manager.cameras[cam_id]
            cam['counter'].active_faces = []  # Clear active faces
            cam['counter'].line_crossing_count = 0  # Reset line crossing count if enabled
            # Clear persistent storage
            try:
                manager.storage.clear_faces(cam_id)
            except:
                pass
            cameras_reset = 1
        else:
            # Reset all cameras
            for cam_id, cam in manager.cameras.items():
                cam['counter'].active_faces = []
                cam['counter'].line_crossing_count = 0
                try:
                    manager.storage.clear_faces(cam_id)
                except:
                    pass
                cameras_reset += 1
        
        return JSONResponse({
            'status': 'success',
            'cameras_reset': cameras_reset,
            'message': f'Frequency reset for {cameras_reset} camera(s)'
        })
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] Frequency reset failed: {error_msg}")
        traceback.print_exc()
        return JSONResponse({'error': error_msg}, status_code=500)

@app.post("/api/process_video")
async def process_video(file: UploadFile = File(...)):
    """Process an uploaded video and extract unique faces."""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Process video
        result = video_processor.process_video(tmp_path, process_every_n_frames=5)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        if result is None:
            return JSONResponse({'error': 'Failed to process video'}, status_code=400)
        
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/add_camera")
async def add_camera(source: str = Form(...), name: str = Form(None), process_fps: int = Form(5), counting_enabled: str = Form("true"), time_window_hours: int = Form(1), camera_type: str = Form("entry"), watchlist_threshold: float = Form(0.4)):
    enabled = str(counting_enabled).lower() in ("1", "true", "yes", "on")
    time_secs = int(time_window_hours) * 3600
    manager.add_camera(source, name=name, process_fps=process_fps, counting_enabled=enabled, time_window_seconds=time_secs, camera_type=camera_type, watchlist_threshold=watchlist_threshold)
    return JSONResponse({"status": "success", "cameras": manager.get_active_cameras()})

@app.post("/api/camera/{cam_id}/update")
async def update_camera(cam_id: str, name: str = Form(None), counting_enabled: str = Form(None)):
    enabled = None
    if counting_enabled is not None:
        enabled = str(counting_enabled).lower() in ("1", "true", "yes", "on")
    ok = manager.update_camera(cam_id, name=name, counting_enabled=enabled)
    return JSONResponse({"ok": ok, "cameras": manager.get_active_cameras()})

@app.delete("/api/camera/{cam_id}")
async def delete_camera(cam_id: str):
    """Delete a camera and stop processing"""
    ok = manager.delete_camera(cam_id)
    return JSONResponse({"ok": ok, "cameras": manager.get_active_cameras()})

@app.get("/api/camera/{cam_id}")
async def get_camera(cam_id: str):
    """Get details of a specific camera"""
    cam_details = manager.get_camera(cam_id)
    if cam_details:
        return JSONResponse(cam_details)
    return JSONResponse({"error": "Camera not found"}, status_code=404)

@app.post("/api/camera/{cam_id}/frequency")
async def set_frequency(cam_id: str, process_fps: int = Form(...)):
    ok = manager.set_frequency(cam_id, process_fps)
    return JSONResponse({"ok": ok, "cameras": manager.get_active_cameras()})

@app.post("/api/camera/{cam_id}/line_crossing")
async def set_line_crossing(cam_id: str, enabled: str = Form(...), line_position: float = Form(0.5)):
    en = str(enabled).lower() in ("1", "true", "yes", "on")
    ok = manager.set_line_crossing(cam_id, en, line_position)
    return JSONResponse({"ok": ok})

@app.delete("/api/camera/{cam_id}/line_crossing")
async def delete_line_crossing(cam_id: str):
    """Disable/clear line crossing for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        # Disable line crossing
        manager.set_line_crossing(cam_id, False, 0.5)
        
        return JSONResponse({'status': 'success', 'message': 'Line crossing disabled'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/camera/{cam_id}/line_position")
async def set_line_position(cam_id: str, line_position: float = Form(...)):
    ok = manager.set_line_position(cam_id, line_position)
    return JSONResponse({"ok": ok})

@app.post("/api/camera/{cam_id}/roi")
async def set_roi(cam_id: str, roi_points: str = Form(""), rot_points: str = Form("")):
    """Set ROI (Region of Interest) and ROT (Region of Tracking) for a camera"""
    import json
    try:
        roi_pts = json.loads(roi_points) if roi_points else []
        rot_pts = json.loads(rot_points) if rot_points else []
        
        # Save using manager which persists to database
        if roi_pts:
            manager.set_roi(cam_id, roi_pts)
        if rot_pts:
            manager.set_rot(cam_id, rot_pts)
        
        return JSONResponse({"ok": True})
    except Exception as e:
        print(f"Error setting ROI: {e}")
    return JSONResponse({"ok": False})

@app.get("/api/faces/{cam_id}")
async def list_faces(cam_id: str):
    """Return recent faces for a camera with detailed cluster information"""
    if cam_id not in manager.cameras:
        return JSONResponse({"faces": [], "clusters": []})
    
    cam = manager.cameras[cam_id]
    
    # Get detailed faces from counter (with cluster info)
    detailed_faces = cam['counter'].get_detailed_faces()
    
    # Get cluster summary
    cluster_summary = cam['counter'].cluster_manager.get_cluster_summary()
    
    # Enrich detailed faces with image URLs and formatting
    faces = []
    for face_detail in detailed_faces:
        img = face_detail.get('image_path')
        img_url = None
        if img:
            import os
            img_url = f"/static/faces/{os.path.basename(img)}"
        
        faces.append({
            "face_id": face_detail['face_id'],
            "cluster_id": face_detail['cluster_id'],
            "image": img_url,
            "quality": round(face_detail['quality'], 2),
            "confidence": round(face_detail['confidence'], 2),
            "timestamp": face_detail['timestamp'],
            "captured_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(face_detail['timestamp']))
        })
    
    # Format clusters
    clusters = []
    for cluster in cluster_summary:
        clusters.append({
            "cluster_id": cluster['cluster_id'],
            "face_count": cluster['face_count'],
            "quality_avg": round(cluster['quality_avg'], 2),
            "first_seen": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cluster['first_seen'])),
            "last_seen": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cluster['last_seen']))
        })
    
    return JSONResponse({
        "faces": faces,
        "clusters": clusters,
        "unique_people_count": cam['counter'].get_unique_people_count(),
        "total_face_events": len(faces)
    })

@app.get("/api/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    """Generates the MJPEG stream for a specific camera"""
    return StreamingResponse(generate_frames(cam_id), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats/{cam_id}")
async def get_stats(cam_id: str):
    """API to fetch pure JSON stats for the frontend to poll"""
    from core.events_db import get_events_db
    
    if cam_id in manager.cameras:
        cam = manager.cameras[cam_id]
        
        # Get time window for this camera
        time_window = cam.get('time_window_seconds', 900)  # Default 15 minutes
        camera_type = cam.get('camera_type', 'entry')
        
        # Get unique count from events DB for accurate counting
        events_db = get_events_db()
        unique_count = events_db.get_unique_count(cam_id, time_window)
        
        # Also get real-time active faces count
        active_count = len(cam['counter'].active_faces)
        
        # Use the higher of the two as the displayed count
        count = max(unique_count, active_count)
        
        # Get GLOBAL entry/exit counts (synced across all cameras)
        global_counts = events_db.get_global_counts()
        entry_count = global_counts['total_entries']
        exit_count = global_counts['total_exits']
        current_count = global_counts['currently_inside']
        
        return {
            "unique_count": count,
            "unique_count_1hr": count,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "current_count": max(0, current_count),
            "process_fps": cam.get('process_fps', 5),
            "counting_enabled": cam.get('counting_enabled', True),
            "name": cam.get('name'),
            "time_window_seconds": cam.get('time_window_seconds'),
            "line_crossing_enabled": cam.get('line_crossing_enabled', False),
            "line_position": cam.get('line_position', 0.5),
            "camera_type": camera_type
        }
    return {"unique_count": 0, "unique_count_1hr": 0, "entry_count": 0, "exit_count": 0, "current_count": 0}

@app.get("/api/stats/global")
async def get_global_stats():
    """Get global entry/exit counts (synced across all entry and exit cameras)"""
    from core.events_db import get_events_db
    try:
        events_db = get_events_db()
        global_counts = events_db.get_global_counts()
        return JSONResponse(global_counts)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/stats/global/reset")
async def reset_global_stats():
    """Reset global entry/exit counts"""
    from core.events_db import get_events_db
    try:
        events_db = get_events_db()
        events_db.reset_global_counts()
        return JSONResponse({'status': 'reset'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/clustering/{cam_id}")
async def get_clustering(cam_id: str):
    """Get clustering information - groups of likely same people"""
    if cam_id in manager.cameras:
        cam = manager.cameras[cam_id]
        clusters = cam['counter'].get_cluster_summary()
        return {
            "total_faces": len(cam['counter'].active_faces),
            "clusters": clusters,
            "estimated_unique_people": len(clusters)
        }
    return {"total_faces": 0, "clusters": [], "estimated_unique_people": 0}

@app.get("/logs")
async def get_logs(event_type: str = None, limit: int = 100):
    """Get face detection logs - useful for debugging"""
    if manager.cameras:
        # Get logs from first active camera's counter
        first_cam = list(manager.cameras.values())[0]
        logger = first_cam['counter'].logger
        logs = logger.get_logs(event_type=event_type, limit=limit)
        return {
            "logs": logs,
            "total_count": len(logs),
            "stats": logger.get_stats()
        }
    return {"logs": [], "total_count": 0, "stats": {}}

@app.get("/logs/export")
async def export_logs():
    """Export logs as JSON file for analysis"""
    if manager.cameras:
        first_cam = list(manager.cameras.values())[0]
        logger = first_cam['counter'].logger
        all_logs = logger.get_logs(limit=10000)
        return JSONResponse(all_logs)
    return JSONResponse([])

def generate_frames(cam_id):
    """Generator function for MJPEG with error recovery"""
    error_skip_count = 0
    no_frame_count = 0
    max_no_frame = 100  # Show error message after this many failed attempts
    
    while True:
        try:
            frame, results, count = manager.get_frame(cam_id)
            
            if frame is None:
                no_frame_count += 1
                # Generate a black error frame if camera not connected
                if no_frame_count > max_no_frame:
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Camera Not Connected", (120, 220), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(error_frame, f"Camera ID: {cam_id}", (180, 260), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.5)
                    no_frame_count = 0  # Reset to show message periodically
                    continue
                time.sleep(0.05)
                continue
            
            # Reset no frame counter when we get a valid frame
            no_frame_count = 0
            
            # Simple frame validation
            if not isinstance(frame, np.ndarray) or len(frame.shape) < 2:
                error_skip_count += 1
                if error_skip_count > 20:
                    error_skip_count = 0
                time.sleep(0.05)
                continue
            
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            if frame_height <= 0 or frame_width <= 0:
                error_skip_count += 1
                time.sleep(0.05)
                continue
            
            error_skip_count = 0
            
            # Draw count
            cv2.putText(frame, f"Count: {count}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw detections - differentiate face vs body
            face_count = 0
            body_count = 0
            if results:
                for res in results:
                    try:
                        bbox = res.get('bbox')
                        if bbox is not None:
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            detection_type = res.get('type', 'face')
                            
                            if detection_type == 'person' or res.get('face_visible') == False:
                                # Body detection (no face) - draw in orange
                                color = (0, 165, 255)  # Orange
                                label = f"Body: {res.get('id', '?')[:6]}"
                                body_count += 1
                            else:
                                # Face detection - green for new, blue for existing
                                color = (0, 255, 0) if res.get('is_new') else (255, 0, 0)
                                label = f"Face: {res.get('id', '?')[:6]}"
                                face_count += 1
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw ID label
                            cv2.putText(frame, label, (x1, y1 - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as e:
                        pass
            
            # Draw detection type counts
            if body_count > 0:
                cv2.putText(frame, f"Faces: {face_count} | Bodies: {body_count}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Draw line crossing if enabled
            if cam_id in manager.cameras:
                cam = manager.cameras[cam_id]
                
                # Draw multi-line crossings if available
                counter = cam.get('counter')
                if hasattr(counter, 'multi_line_tracker') and counter.multi_line_tracker:
                    try:
                        for line in counter.multi_line_tracker.lines.values():
                            start_px, end_px = counter.multi_line_tracker.get_pixel_coordinates(line)
                            cv2.line(frame, start_px, end_px, line.color, 2)
                            # Draw label
                            label_pos = ((start_px[0] + end_px[0]) // 2, 
                                        (start_px[1] + end_px[1]) // 2 - 10)
                            cv2.putText(frame, f"{line.name}: In={line.entry_count} Out={line.exit_count}", 
                                       label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line.color, 1)
                    except Exception as e:
                        pass
                elif cam.get('line_crossing_enabled') and hasattr(counter, 'line_crossing_tracker') and counter.line_crossing_tracker:
                    try:
                        line_x = int(cam.get('line_position', 0.5) * frame_width)
                        cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 255, 0), 2)
                        cv2.putText(frame, "CROSSING LINE", (line_x + 5, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    except:
                        pass
                
                # Draw ROI if set
                try:
                    roi_points = cam.get('roi_points')
                    if roi_points and len(roi_points) > 0:
                        roi_pts = np.array([(int(p[0]*frame_width), int(p[1]*frame_height)) for p in roi_points], dtype=np.int32)
                        cv2.polylines(frame, [roi_pts], True, (0, 255, 0), 2)
                except:
                    pass
                
                # Draw ROT if set
                try:
                    rot_points = cam.get('rot_points')
                    if rot_points and len(rot_points) > 0:
                        rot_pts = np.array([(int(p[0]*frame_width), int(p[1]*frame_height)) for p in rot_points], dtype=np.int32)
                        cv2.polylines(frame, [rot_pts], True, (0, 0, 255), 2)
                except:
                    pass
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print(f"[generate_frames] Error processing frame for {cam_id}: {e}")
            error_skip_count += 1
            time.sleep(0.1)
            if error_skip_count > 20:
                error_skip_count = 0


# ========== TRACKING API ENDPOINTS ==========

from core.tracking_db import TrackingDatabase
from core.events_db import get_events_db

# Initialize tracking database
tracking_db = TrackingDatabase()

@app.get("/api/tracking/people")
async def get_all_tracked_people(hours: int = 24, limit: int = 200, active_only: bool = False, minutes: int = 60):
    """Get all people across all cameras with aggregated stats for people-first view"""
    try:
        import os
        from datetime import datetime, timedelta
        events_db = get_events_db()
        
        # Get all people with stats from events DB
        people = events_db.get_all_people_with_stats(since_hours=hours, limit=limit)
        
        # Get watchlist data for name lookups
        watchlist_names = {}
        if watchlist_db:
            try:
                watchlist = watchlist_db.get_watchlist()
                for wl_person in watchlist:
                    if wl_person.get('name'):
                        watchlist_names[wl_person['person_id']] = wl_person['name']
            except:
                pass
        
        result = []
        for person in people:
            # Get thumbnail
            thumbnail = events_db.get_person_thumbnail(person['person_id'])
            
            # Check if person is currently live (seen in last 30 seconds)
            is_live = False
            last_seen = person.get('last_seen')
            if last_seen:
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                    is_live = (datetime.now() - last_seen_dt.replace(tzinfo=None)) < timedelta(seconds=30)
                except:
                    pass
            
            # Skip inactive people if active_only is set
            if active_only and not is_live:
                continue
            
            # Get watchlist name if available
            watchlist_name = watchlist_names.get(person['person_id'])
            
            result.append({
                'person_id': person['person_id'],
                'name': watchlist_name,  # Watchlist name if available
                'first_seen': person.get('first_seen'),
                'last_seen': last_seen,
                'total_detections': person.get('total_detections', 0),
                'confidence': person.get('confidence', 0.0),
                'thumbnail_url': f"/static/faces/{os.path.basename(thumbnail)}" if thumbnail else None,
                'is_live': is_live,
                'current_camera': person.get('current_camera'),
                'cameras_visited': person.get('cameras_visited', 1),
                'in_watchlist': person['person_id'] in watchlist_names
            })
        
        return JSONResponse({
            'people': result,
            'total': len(result)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get("/api/tracking/person/{person_id}")
async def get_person_location(person_id: str):
    """Get person's current location and tracking info"""
    try:
        location = tracking_db.get_person_current_location(person_id)
        if location:
            return JSONResponse(location)
        return JSONResponse({'error': 'Person not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get("/api/tracking/person/{person_id}/history")
async def get_person_history(person_id: str, limit: int = 50):
    """Get person's movement history across cameras"""
    try:
        history = tracking_db.get_person_movement_history(person_id, limit=limit)
        return JSONResponse(history)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get("/api/tracking/stats")
async def get_tracking_stats():
    """Get overall tracking statistics"""
    try:
        stats = tracking_db.get_tracking_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post("/api/tracking/event")
async def record_tracking_event(
    person_id: str = Form(...),
    camera_id: str = Form(...),
    event_type: str = Form(...),
    direction: str = Form(None),
    metadata: str = Form(None)
):
    """Record a movement event (entry, exit, zone crossing)"""
    try:
        import json
        metadata_dict = json.loads(metadata) if metadata else None
        tracking_db.record_movement_event(
            person_id=person_id,
            camera_id=camera_id,
            event_type=event_type,
            direction=direction,
            metadata=metadata_dict
        )
        return JSONResponse({'status': 'success'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


# Watchlist endpoints
from core.watchlist import WatchlistDB
try:
    watchlist_db = WatchlistDB()
    print("[INFO] WatchlistDB initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize WatchlistDB: {e}")
    import traceback
    traceback.print_exc()
    watchlist_db = None

@app.get("/api/watchlist")
async def get_watchlist():
    """Get all persons in the watchlist"""
    if watchlist_db is None:
        return JSONResponse({'error': 'Watchlist database not initialized'}, status_code=500)
    try:
        watchlist = watchlist_db.get_watchlist()
        # Add thumbnail URLs
        for person in watchlist:
            if person.get('thumbnail_path'):
                person['thumbnail_url'] = f"/static/faces/{os.path.basename(person['thumbnail_path'])}"
            if person.get('photo_path'):
                person['photo_url'] = f"/static/faces/{os.path.basename(person['photo_path'])}"
        return JSONResponse({'watchlist': watchlist})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/watchlist/add")
async def add_to_watchlist(
    person_id: str = Form(...),
    name: str = Form(None),
    notes: str = Form(None),
    alert_enabled: str = Form("true"),
    category: str = Form("general")
):
    """Add a person to the watchlist"""
    if watchlist_db is None:
        return JSONResponse({'error': 'Watchlist database not initialized'}, status_code=500)
    try:
        enabled = str(alert_enabled).lower() in ("1", "true", "yes", "on")
        
        # Get embedding from events database for face matching
        embedding = None
        thumbnail_path = None
        try:
            from core.events_db import EventsDB
            events_db = EventsDB()
            embedding = events_db.get_person_embedding(person_id)
            thumbnail_path = events_db.get_person_thumbnail(person_id)
            if embedding is not None:
                print(f"[Watchlist] Got embedding for {person_id}, shape: {embedding.shape}")
            else:
                print(f"[Watchlist] No embedding found for {person_id}")
        except Exception as e:
            print(f"[Watchlist] Error getting embedding: {e}")
        
        watchlist_db.add_to_watchlist(
            person_id, name, notes, enabled, 
            category=category,
            embedding=embedding,
            thumbnail_path=thumbnail_path
        )
        return JSONResponse({'status': 'success'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.delete("/api/watchlist/{person_id}")
async def remove_from_watchlist(person_id: str):
    """Remove a person from the watchlist"""
    try:
        watchlist_db.remove_from_watchlist(person_id)
        return JSONResponse({'status': 'success'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/{person_id}/detections")
async def get_watchlist_detections(person_id: str):
    """Get detection history for a watchlist person"""
    try:
        detections = watchlist_db.get_person_detections(person_id)
        # Add image URLs
        for detection in detections:
            if detection.get('thumbnail_path'):
                detection['thumbnail_url'] = f"/static/faces/{os.path.basename(detection['thumbnail_path'])}"
            if detection.get('fullframe_path'):
                detection['fullframe_url'] = f"/static/faces/{os.path.basename(detection['fullframe_path'])}"
        return JSONResponse({'detections': detections})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/alerts")
async def get_watchlist_alerts(minutes: int = 5):
    """Get recent watchlist person detections"""
    try:
        alerts = watchlist_db.get_watchlist_alerts(minutes)
        return JSONResponse({'alerts': alerts})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/{person_id}")
async def get_watchlist_person(person_id: str):
    """Get details of a watchlist person"""
    try:
        person = watchlist_db.get_person(person_id)
        if person:
            # Add image URLs
            if person.get('thumbnail_path'):
                person['thumbnail_url'] = f"/static/faces/{os.path.basename(person['thumbnail_path'])}"
            if person.get('photo_path'):
                person['photo_url'] = f"/static/faces/{os.path.basename(person['photo_path'])}"
            return JSONResponse({'person': person})
        return JSONResponse({'error': 'Person not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.put("/api/watchlist/{person_id}")
async def update_watchlist_person(
    person_id: str,
    name: str = Form(None),
    notes: str = Form(None),
    alert_enabled: str = Form(None),
    category: str = Form(None)
):
    """Update a watchlist person's details"""
    try:
        enabled = None
        if alert_enabled is not None:
            enabled = str(alert_enabled).lower() in ("1", "true", "yes", "on")
        
        watchlist_db.update_watchlist_person(person_id, name=name, notes=notes, 
                                             alert_enabled=enabled, category=category)
        return JSONResponse({'status': 'success'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/watchlist/{person_id}/photo")
async def upload_watchlist_photo(person_id: str, photo: UploadFile = File(...)):
    """Upload a photo for a watchlist person"""
    try:
        # Save the uploaded photo
        photo_filename = f"watchlist_{person_id}_{int(time.time())}.jpg"
        photo_path = os.path.join(Config.IMAGE_OUTPUT_DIR, photo_filename)
        
        contents = await photo.read()
        
        # Save original photo
        with open(photo_path, 'wb') as f:
            f.write(contents)
        
        # Create thumbnail
        import cv2
        import numpy as np
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Create thumbnail (150x150)
            h, w = img.shape[:2]
            size = min(h, w)
            y_offset = (h - size) // 2
            x_offset = (w - size) // 2
            cropped = img[y_offset:y_offset+size, x_offset:x_offset+size]
            thumb = cv2.resize(cropped, (150, 150))
            
            thumb_filename = f"watchlist_{person_id}_{int(time.time())}_thumb.jpg"
            thumb_path = os.path.join(Config.IMAGE_OUTPUT_DIR, thumb_filename)
            cv2.imwrite(thumb_path, thumb)
            
            # Try to extract face embedding
            try:
                faces = manager.recognizer.get_faces(img)
                if faces:
                    embedding = faces[0].embedding
                    watchlist_db.set_embedding(person_id, embedding)
                    print(f"[Watchlist] Saved embedding for {person_id}")
            except Exception as e:
                print(f"Could not extract embedding: {e}")
            
            # Update database
            watchlist_db.set_photo(person_id, photo_path, thumb_path)
            
            return JSONResponse({
                'status': 'success',
                'photo_url': f"/static/faces/{photo_filename}",
                'thumbnail_url': f"/static/faces/{thumb_filename}"
            })
        
        return JSONResponse({'error': 'Invalid image'}, status_code=400)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/watchlist/{person_id}/regenerate-embedding")
async def regenerate_watchlist_embedding(person_id: str):
    """Regenerate face embedding for a watchlist person from their photo"""
    try:
        person = watchlist_db.get_person(person_id)
        if not person:
            return JSONResponse({'error': 'Person not found'}, status_code=404)
        
        # Try to get photo path
        photo_path = person.get('photo_path') or person.get('thumbnail_path')
        if not photo_path or not os.path.exists(photo_path):
            return JSONResponse({'error': 'No photo available for this person'}, status_code=400)
        
        # Read image and extract embedding
        img = cv2.imread(photo_path)
        if img is None:
            return JSONResponse({'error': 'Could not read image'}, status_code=400)
        
        faces = manager.recognizer.get_faces(img)
        if not faces:
            return JSONResponse({'error': 'No face detected in the photo'}, status_code=400)
        
        # Store the embedding
        embedding = faces[0].embedding
        watchlist_db.set_embedding(person_id, embedding)
        
        return JSONResponse({
            'status': 'success',
            'message': f'Embedding regenerated for {person.get("name", person_id)}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/search/{query}")
async def search_watchlist(query: str):
    """Search watchlist by name or person_id"""
    try:
        results = watchlist_db.search_watchlist(query)
        # Add thumbnail URLs
        for person in results:
            if person.get('thumbnail_path'):
                person['thumbnail_url'] = f"/static/faces/{os.path.basename(person['thumbnail_path'])}"
        return JSONResponse({'results': results})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/{person_id}/location-history")
async def get_watchlist_location_history(person_id: str):
    """Get location history timeline for a watchlist person"""
    try:
        history = watchlist_db.get_person_location_history(person_id)
        # Add thumbnail URLs
        for detection in history:
            if detection.get('thumbnail_path'):
                detection['thumbnail_url'] = f"/static/faces/{os.path.basename(detection['thumbnail_path'])}"
        return JSONResponse({'history': history})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/watchlist/add-from-tracking")
async def add_to_watchlist_from_tracking(
    person_id: str = Form(...),
    name: str = Form(None),
    notes: str = Form(None),
    category: str = Form("general")
):
    """Add a tracked person to the watchlist with their existing thumbnail and embedding"""
    try:
        events_db = get_events_db()
        
        # Get the person's latest thumbnail from events
        thumbnail = events_db.get_person_thumbnail(person_id)
        
        # First try to get stored embedding from events database
        embedding = events_db.get_person_embedding(person_id)
        
        # If no stored embedding, try to extract from thumbnail
        if embedding is None and thumbnail:
            try:
                img = cv2.imread(thumbnail)
                if img is not None:
                    faces = manager.recognizer.get_faces(img)
                    if faces:
                        embedding = faces[0].embedding
                        print(f"[Watchlist] Extracted embedding from thumbnail for {person_id}")
            except Exception as e:
                print(f"Could not extract embedding: {e}")
        else:
            print(f"[Watchlist] Using stored embedding for {person_id}")
        
        # Add to watchlist
        watchlist_db.add_to_watchlist(
            person_id=person_id,
            name=name or person_id,
            notes=notes,
            alert_enabled=True,
            thumbnail_path=thumbnail,
            embedding=embedding,
            category=category
        )
        
        return JSONResponse({'status': 'success', 'person_id': person_id})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/watchlist/categories")
async def get_watchlist_categories():
    """Get all watchlist categories with details"""
    if watchlist_db is None:
        return JSONResponse({'error': 'Watchlist database not initialized'}, status_code=500)
    try:
        categories = watchlist_db.get_categories()
        return JSONResponse({'categories': categories})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/watchlist/categories")
async def add_watchlist_category(request: Request):
    """Create a new watchlist category"""
    if watchlist_db is None:
        return JSONResponse({'error': 'Watchlist database not initialized'}, status_code=500)
    try:
        data = await request.json()
        name = data.get('name')
        if not name:
            return JSONResponse({'error': 'Category name is required'}, status_code=400)
        
        color = data.get('color', '#808080')
        description = data.get('description')
        alert_priority = data.get('alert_priority', 1)
        
        success = watchlist_db.add_category(name, color, description, alert_priority)
        if success:
            return JSONResponse({'status': 'success', 'category': name})
        else:
            return JSONResponse({'error': 'Category already exists'}, status_code=409)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.put("/api/watchlist/categories/{category_name}")
async def update_watchlist_category(category_name: str, request: Request):
    """Update a watchlist category"""
    try:
        data = await request.json()
        color = data.get('color')
        description = data.get('description')
        alert_priority = data.get('alert_priority')
        
        success = watchlist_db.update_category(category_name, color, description, alert_priority)
        if success:
            return JSONResponse({'status': 'success'})
        else:
            return JSONResponse({'error': 'Category not found or no changes made'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.delete("/api/watchlist/categories/{category_name}")
async def delete_watchlist_category(category_name: str):
    """Delete a watchlist category (moves persons to general)"""
    try:
        if category_name == 'general':
            return JSONResponse({'error': 'Cannot delete the general category'}, status_code=400)
        
        success = watchlist_db.delete_category(category_name)
        if success:
            return JSONResponse({'status': 'success'})
        else:
            return JSONResponse({'error': 'Category not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

# Camera Timer Status endpoint
@app.get("/api/camera/{cam_id}/timer")
async def get_camera_timer(cam_id: str):
    """Get the remaining time until camera count resets"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        counter = cam['counter']
        
        current_time = time.time()
        window_start = getattr(counter, 'window_start_time', current_time)
        time_window = counter.time_window_seconds
        
        elapsed = current_time - window_start
        remaining = max(0, time_window - elapsed)
        
        return JSONResponse({
            'time_window_seconds': time_window,
            'elapsed_seconds': int(elapsed),
            'remaining_seconds': int(remaining),
            'reset_at': window_start + time_window
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.put("/api/camera/{cam_id}/timer")
async def set_camera_timer(cam_id: str, request: Request):
    """Set the time window for a camera's count reset"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        data = await request.json()
        time_window_seconds = data.get('time_window_seconds')
        reset_now = data.get('reset_now', False)
        
        if time_window_seconds is None and not reset_now:
            return JSONResponse({'error': 'time_window_seconds is required'}, status_code=400)
        
        cam = manager.cameras[cam_id]
        counter = cam['counter']
        
        # Update time window if provided
        if time_window_seconds is not None:
            time_window_seconds = max(60, int(time_window_seconds))  # Minimum 1 minute
            counter.time_window_seconds = time_window_seconds
            cam['time_window_seconds'] = time_window_seconds
            
            # Persist to database
            manager.db.save_camera(cam_id, {
                'name': cam['name'],
                'source': cam['source'],
                'fps': cam['process_fps'],
                'counting_enabled': cam['counting_enabled'],
                'time_window_seconds': time_window_seconds,
                'line_crossing_enabled': cam.get('line_crossing_enabled', False),
                'line_position': cam.get('line_position', 0.5),
                'roi_points': cam.get('roi_points', []),
                'rot_points': cam.get('rot_points', [])
            })
        
        # Reset timer if requested
        if reset_now:
            counter.window_start_time = time.time()
            counter.last_reset_time = time.time()
            counter.active_faces = []  # Clear active faces
            
            # Clear events for this camera
            try:
                from core.events_db import EventsDB
                events_db = EventsDB()
                events_db.reset_camera_counts(cam_id)
            except Exception as e:
                print(f"[Timer Reset] Error clearing events: {e}")
        
        return JSONResponse({
            'status': 'success',
            'time_window_seconds': counter.time_window_seconds,
            'reset_performed': reset_now
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

# Camera ROI endpoints
@app.get("/api/camera/{cam_id}/roi")
async def get_camera_roi(cam_id: str):
    """Get the current ROI and ROT points for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        roi_points = cam.get('roi_points', [])
        rot_points = cam.get('rot_points', [])
        line_crossing_enabled = cam.get('line_crossing_enabled', False)
        
        return JSONResponse({
            'roi_points': roi_points,
            'rot_points': rot_points,
            'line_crossing_enabled': line_crossing_enabled
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.delete("/api/camera/{cam_id}/roi")
async def delete_camera_roi(cam_id: str):
    """Delete/clear the ROI for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        # Clear ROI in manager
        manager.set_roi(cam_id, [])
        
        return JSONResponse({'status': 'success', 'message': 'ROI cleared'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.delete("/api/camera/{cam_id}/rot")
async def delete_camera_rot(cam_id: str):
    """Delete/clear the entry/exit line (ROT) for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        # Clear ROT in manager
        manager.set_rot(cam_id, [])
        
        # Also disable line crossing when clearing ROT
        manager.set_line_crossing(cam_id, False, 0.5)
        
        return JSONResponse({'status': 'success', 'message': 'Entry/Exit line cleared'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


# ========== LINE CROSSING ENDPOINTS ==========

@app.get("/api/camera/{cam_id}/lines")
async def get_camera_lines(cam_id: str):
    """Get all crossing lines for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        counter = cam.get('counter')
        
        if hasattr(counter, 'multi_line_tracker') and counter.multi_line_tracker:
            lines = counter.multi_line_tracker.get_lines()
            counts = counter.multi_line_tracker.get_line_counts()
            return JSONResponse({
                'lines': lines,
                'counts': counts
            })
        
        return JSONResponse({'lines': [], 'counts': {}})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/camera/{cam_id}/lines")
async def add_camera_line(cam_id: str, request: Request):
    """Add a crossing line to a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        data = await request.json()
        name = data.get('name', 'A')
        start = data.get('start', [0.5, 0.0])
        end = data.get('end', [0.5, 1.0])
        direction_positive = data.get('direction_positive', 'entry')
        direction_negative = data.get('direction_negative', 'exit')
        color = data.get('color')
        
        cam = manager.cameras[cam_id]
        counter = cam.get('counter')
        
        # Initialize multi-line tracker if not exists
        if not hasattr(counter, 'multi_line_tracker') or counter.multi_line_tracker is None:
            from core.line_crossing_v2 import MultiLineCrossingTracker
            counter.multi_line_tracker = MultiLineCrossingTracker()
        
        # Add the line
        color_tuple = tuple(color) if color else None
        counter.multi_line_tracker.add_line(
            name=name,
            start=tuple(start),
            end=tuple(end),
            direction_positive=direction_positive,
            direction_negative=direction_negative,
            color=color_tuple
        )
        
        return JSONResponse({'status': 'success', 'line': name})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.delete("/api/camera/{cam_id}/lines/{line_name}")
async def delete_camera_line(cam_id: str, line_name: str):
    """Delete a crossing line from a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        counter = cam.get('counter')
        
        if hasattr(counter, 'multi_line_tracker') and counter.multi_line_tracker:
            success = counter.multi_line_tracker.remove_line(line_name)
            if success:
                return JSONResponse({'status': 'success'})
            return JSONResponse({'error': 'Line not found'}, status_code=404)
        
        return JSONResponse({'error': 'No lines configured'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/camera/{cam_id}/line-crossings")
async def get_line_crossings(cam_id: str, minutes: int = 5):
    """Get recent line crossing events for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        counter = cam.get('counter')
        
        if hasattr(counter, 'multi_line_tracker') and counter.multi_line_tracker:
            crossings = counter.multi_line_tracker.get_recent_crossings(minutes)
            counts = counter.multi_line_tracker.get_line_counts()
            return JSONResponse({
                'crossings': crossings,
                'counts': counts
            })
        
        return JSONResponse({'crossings': [], 'counts': {}})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/camera/{cam_id}/lines/reset")
async def reset_line_counts(cam_id: str):
    """Reset all line crossing counts for a camera"""
    try:
        if cam_id not in manager.cameras:
            return JSONResponse({'error': 'Camera not found'}, status_code=404)
        
        cam = manager.cameras[cam_id]
        counter = cam.get('counter')
        
        if hasattr(counter, 'multi_line_tracker') and counter.multi_line_tracker:
            counter.multi_line_tracker.reset_counts()
            return JSONResponse({'status': 'success'})
        
        return JSONResponse({'status': 'success', 'message': 'No lines to reset'})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


# Events DB endpoints
from core.events_db import get_events_db

@app.get("/api/tracking/cameras")
async def get_tracking_cameras_with_counts():
    """Get all cameras with unique person count for tracking page"""
    try:
        events_db = get_events_db()
        
        cameras_data = []
        for cam_id, cam in manager.cameras.items():
            # Get unique count for this camera
            unique_count = events_db.get_unique_count(cam_id, time_window_seconds=3600)
            
            cameras_data.append({
                'id': cam_id,
                'name': cam.get('name', cam_id),
                'unique_count': unique_count,
                'is_active': True
            })
        
        return JSONResponse({'cameras': cameras_data})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/tracking/camera/{cam_id}/people")
async def get_camera_unique_people(cam_id: str, hours: int = 24, limit: int = 50):
    """Get unique people detected by a specific camera with thumbnails"""
    try:
        import os
        events_db = get_events_db()
        
        # Get people detected by this camera
        people = events_db.get_camera_people(cam_id, since_hours=hours, limit=limit)
        
        # Add thumbnail URLs
        for person in people:
            thumbnail = events_db.get_person_thumbnail(person['person_id'])
            if thumbnail:
                person['thumbnail_url'] = f"/static/faces/{os.path.basename(thumbnail)}"
            else:
                person['thumbnail_url'] = None
        
        return JSONResponse({
            'camera_id': cam_id,
            'people': people,
            'total': len(people)
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/tracking/person/{person_id}/live")
async def get_person_live_tracking(person_id: str):
    """Get real-time tracking info for a person - current camera or last known location"""
    try:
        import os
        events_db = get_events_db()
        
        # Get person's current/latest location
        location = events_db.get_person_location(person_id)
        
        if not location:
            return JSONResponse({'error': 'Person not found'}, status_code=404)
        
        # Check if person was seen in last 30 seconds (considered "live")
        from datetime import datetime, timedelta
        last_seen_str = location.get('last_seen')
        is_live = False
        
        if last_seen_str:
            try:
                last_seen = datetime.fromisoformat(last_seen_str.replace('Z', '+00:00'))
                is_live = (datetime.now() - last_seen.replace(tzinfo=None)) < timedelta(seconds=30)
            except:
                is_live = False
        
        # Get thumbnail
        thumbnail = events_db.get_person_thumbnail(person_id)
        
        # Get recent detections across cameras
        history = events_db.get_person_history(person_id, limit=10)
        for det in history:
            if det.get('thumbnail_path'):
                det['thumbnail_url'] = f"/static/faces/{os.path.basename(det['thumbnail_path'])}"
            if det.get('fullframe_path'):
                det['fullframe_url'] = f"/static/faces/{os.path.basename(det['fullframe_path'])}"
        
        # Check watchlist status
        from core.watchlist import WatchlistDB
        watchlist_db = WatchlistDB()
        watchlist_info = watchlist_db.get_person(person_id)
        watchlist_db.close()
        
        return JSONResponse({
            'person_id': person_id,
            'is_live': is_live,
            'current_camera_id': location.get('last_camera_id'),
            'current_camera_name': location.get('last_camera_name'),
            'last_seen': location.get('last_seen'),
            'first_seen': location.get('first_seen'),
            'total_detections': location.get('total_detections', 0),
            'thumbnail_url': f"/static/faces/{os.path.basename(thumbnail)}" if thumbnail else None,
            'recent_detections': history,
            'watchlist': {
                'in_watchlist': watchlist_info is not None,
                'name': watchlist_info.get('name') if watchlist_info else None,
                'category': watchlist_info.get('category') if watchlist_info else None
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/people")
async def get_all_people(query: str = None, camera_id: str = None, hours: int = 24, limit: int = 100):
    """Search and get all tracked people with filters and thumbnails"""
    try:
        events_db = get_events_db()
        people = events_db.search_people(query=query, camera_id=camera_id, since_hours=hours, limit=limit)
        
        # Add thumbnail URLs for each person
        for person in people:
            thumbnail = events_db.get_person_thumbnail(person['person_id'])
            if thumbnail:
                import os
                person['thumbnail_url'] = f"/static/faces/{os.path.basename(thumbnail)}"
            else:
                person['thumbnail_url'] = None
        
        return JSONResponse({'people': people})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/person/{person_id}")
async def get_person_details(person_id: str):
    """Get person location, details, and thumbnail"""
    try:
        events_db = get_events_db()
        location = events_db.get_person_location(person_id)
        if location:
            # Add thumbnail
            thumbnail = events_db.get_person_thumbnail(person_id)
            if thumbnail:
                import os
                location['thumbnail_url'] = f"/static/faces/{os.path.basename(thumbnail)}"
            else:
                location['thumbnail_url'] = None
            return JSONResponse(location)
        return JSONResponse({'error': 'Person not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/person/{person_id}/history")
async def get_person_detection_history(person_id: str, limit: int = 100):
    """Get person's detection history across cameras with images"""
    try:
        events_db = get_events_db()
        history = events_db.get_person_history(person_id, limit=limit)
        events = events_db.get_person_entry_exit_history(person_id, limit=50)
        
        # Convert image paths to URLs
        import os
        for detection in history:
            if detection.get('thumbnail_path'):
                detection['thumbnail_url'] = f"/static/faces/{os.path.basename(detection['thumbnail_path'])}"
            else:
                detection['thumbnail_url'] = None
            if detection.get('fullframe_path'):
                detection['fullframe_url'] = f"/static/faces/{os.path.basename(detection['fullframe_path'])}"
            else:
                detection['fullframe_url'] = None
        
        return JSONResponse({
            'detections': history,
            'entry_exit_events': events
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/recent")
async def get_recent_events(camera_id: str = None, limit: int = 50):
    """Get recent entry/exit events"""
    try:
        events_db = get_events_db()
        events = events_db.get_recent_events(camera_id=camera_id, limit=limit)
        return JSONResponse({'events': events})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/detections/{cam_id}")
async def get_camera_detections(cam_id: str, limit: int = 20):
    """Get recent face detections with images for a camera's detection log"""
    try:
        events_db = get_events_db()
        detections = events_db.get_recent_detections(camera_id=cam_id, limit=limit)
        
        # Get watchlist data for name lookups
        watchlist_names = {}
        if watchlist_db:
            try:
                watchlist = watchlist_db.get_watchlist()
                for wl_person in watchlist:
                    if wl_person.get('name'):
                        watchlist_names[wl_person['person_id']] = wl_person['name']
            except:
                pass
        
        # Convert image paths to URLs and add watchlist names
        import os
        for detection in detections:
            if detection.get('thumbnail_path'):
                detection['thumbnail_url'] = f"/static/faces/{os.path.basename(detection['thumbnail_path'])}"
            else:
                detection['thumbnail_url'] = None
            if detection.get('fullframe_path'):
                detection['fullframe_url'] = f"/static/faces/{os.path.basename(detection['fullframe_path'])}"
            else:
                detection['fullframe_url'] = None
            
            # Use watchlist_name from detection if available (stored in DB), otherwise lookup
            if detection.get('watchlist_name'):
                detection['name'] = detection['watchlist_name']
                detection['in_watchlist'] = True
                detection['match_confidence'] = detection.get('watchlist_confidence')
            else:
                detection['name'] = watchlist_names.get(detection['person_id'])
                detection['in_watchlist'] = detection['person_id'] in watchlist_names
                detection['match_confidence'] = None
        
        return JSONResponse({'detections': detections})
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/stats")
async def get_events_stats():
    """Get overall events statistics"""
    try:
        events_db = get_events_db()
        stats = events_db.get_stats_summary()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.get("/api/events/camera/{cam_id}/counts")
async def get_camera_counts(cam_id: str, time_window: int = 3600):
    """Get unique count and entry/exit for a camera within time window"""
    try:
        events_db = get_events_db()
        unique_count = events_db.get_unique_count(cam_id, time_window)
        unique_in_roi = events_db.get_unique_count_in_roi(cam_id, time_window)
        entry_exit = events_db.get_entry_exit_counts(cam_id, time_window)
        return JSONResponse({
            'unique_count': unique_count,
            'unique_in_roi': unique_in_roi,
            'entry_count': entry_exit['entries'],
            'exit_count': entry_exit['exits'],
            'current_count': entry_exit['current'],
            'time_window_seconds': time_window
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/camera/{cam_id}/reset")
async def reset_camera_counts(cam_id: str, clear_logs: bool = True):
    """Reset counts for a camera and clear detection logs for counting page only"""
    try:
        if cam_id in manager.cameras:
            cam = manager.cameras[cam_id]
            counter = cam['counter']
            
            # Reset timing
            counter.last_reset_time = time.time()
            counter.window_start_time = time.time()
            
            # Reset in-memory face tracking
            counter.active_faces = []
            counter.face_ids = set() if hasattr(counter, 'face_ids') else set()
            
            # Reset line crossing counts if enabled
            if hasattr(counter, 'line_crossing_count'):
                counter.line_crossing_count = 0
            if hasattr(counter, 'line_crossing_tracker') and counter.line_crossing_tracker:
                counter.line_crossing_tracker.entry_count = 0
                counter.line_crossing_tracker.exit_count = 0
                counter.line_crossing_tracker.crossed_ids = set()
            
            # Reset database - archive stats
            events_db = get_events_db()
            events_db.reset_camera_counts(cam_id)
            
            # RESET GLOBAL COUNTS (entry/exit/currently inside)
            events_db.reset_global_counts()
            
            # CLEAR DETECTION LOGS for counting page only (not tracking history)
            if clear_logs:
                cleared = events_db.clear_detection_logs(cam_id)
                print(f"[Reset] Cleared {cleared} detection logs for camera {cam_id}")
            
            return JSONResponse({'status': 'reset', 'camera_id': cam_id, 'logs_cleared': clear_logs})
        return JSONResponse({'error': 'Camera not found'}, status_code=404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get("/api/tracking/person/{person_id}/details")
async def get_person_full_details(person_id: str):
    """Get complete person details including cameras visited and recent detections"""
    try:
        import os
        from datetime import datetime, timedelta
        events_db = get_events_db()
        
        # Get person location
        location = events_db.get_person_location(person_id)
        if not location:
            return JSONResponse({'error': 'Person not found'}, status_code=404)
        
        # Check if live
        is_live = False
        last_seen = location.get('last_seen')
        if last_seen:
            try:
                last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                is_live = (datetime.now() - last_seen_dt.replace(tzinfo=None)) < timedelta(seconds=30)
            except:
                pass
        
        # Get thumbnail
        thumbnail = events_db.get_person_thumbnail(person_id)
        
        # Get all cameras visited by this person
        cameras = events_db.get_person_camera_visits(person_id)
        
        # Mark active cameras (person seen in last 30 seconds)
        for cam in cameras:
            cam['is_active'] = False
            last_det = cam.get('last_detection')
            if last_det:
                try:
                    last_det_dt = datetime.fromisoformat(last_det.replace('Z', '+00:00'))
                    cam['is_active'] = (datetime.now() - last_det_dt.replace(tzinfo=None)) < timedelta(seconds=30)
                except:
                    pass
        
        # Get recent detections
        history = events_db.get_person_history(person_id, limit=20)
        for det in history:
            if det.get('thumbnail_path'):
                det['thumbnail_url'] = f"/static/faces/{os.path.basename(det['thumbnail_path'])}"
            else:
                det['thumbnail_url'] = None
            if det.get('fullframe_path'):
                det['fullframe_url'] = f"/static/faces/{os.path.basename(det['fullframe_path'])}"
            else:
                det['fullframe_url'] = None
        
        # Check watchlist status
        watchlist_info = None
        try:
            from core.watchlist import WatchlistDB
            wl_db = WatchlistDB()
            watchlist_info = wl_db.get_person(person_id)
            wl_db.close()
        except:
            pass
        
        return JSONResponse({
            'person_id': person_id,
            'is_live': is_live,
            'current_camera_id': location.get('last_camera_id'),
            'current_camera_name': location.get('last_camera_name'),
            'last_seen': last_seen,
            'first_seen': location.get('first_seen'),
            'total_detections': location.get('total_detections', 0),
            'thumbnail_url': f"/static/faces/{os.path.basename(thumbnail)}" if thumbnail else None,
            'cameras': cameras,
            'recent_detections': history,
            'watchlist': {
                'in_watchlist': watchlist_info is not None,
                'name': watchlist_info.get('name') if watchlist_info else None,
                'category': watchlist_info.get('category') if watchlist_info else None
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)


if __name__ == "__main__":
    # Use '0.0.0.0' to make it accessible on the local network
    uvicorn.run(app, host="0.0.0.0", port=8000)