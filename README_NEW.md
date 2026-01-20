# OptiExacta - AI-Powered People Analytics

## System Architecture

### Frontend (React + TypeScript + Tailwind)
- **Homepage**: Landing page with navigation to Counting and Tracking
- **People Counting**: Real-time camera feeds with ROI, face detection, visitor counting
- **People Tracking**: Movement history, location tracking, analytics

### Backend (FastAPI + Python)
- **Face Recognition**: InsightFace with GPU acceleration (CUDA)
- **Tracking Database**: SQLite with person history, location, and movement events
- **REST API**: Full API for counting, tracking, camera management

### Database Schema
- **persons**: Person state, embeddings, last seen location
- **detection_history**: All detections with timestamps
- **location_history**: Camera-to-camera movement
- **movement_events**: Zone crossings, entry/exit events
- **tracking_analytics**: Aggregated analytics data

## Features

### People Counting
- ✅ Real-time face detection and recognition
- ✅ Unique visitor tracking with deduplication
- ✅ ROI (Region of Interest) zone selection
- ✅ Entry/Exit counting
- ✅ Multi-camera support (RTSP, USB webcam)
- ✅ GPU acceleration with CUDA
- ✅ Adjustable FPS processing

### People Tracking (NEW)
- ✅ Track person's latest camera location
- ✅ Movement history across cameras
- ✅ Last seen timestamp with validation
- ✅ Duplicate detection prevention (30s window)
- ✅ Person movement analytics
- ✅ Camera visit history
- ✅ Real-time active person monitoring

## Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+
- NVIDIA GPU with CUDA 12.x (optional, for GPU acceleration)
- cuDNN v9+ (for GPU)

### Installation

1. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

2. **Install Frontend Dependencies**
```bash
cd frontend
npm install
```

3. **Run the Application**

**Option 1: Using start script (Windows)**
```bash
start.bat
```

**Option 2: Manual start**

Terminal 1 - Backend:
```bash
python web_app.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## API Endpoints

### Camera Management
- `GET /api/cameras` - List all cameras
- `POST /add_camera` - Add new camera
- `POST /camera/{cam_id}/delete` - Delete camera
- `POST /camera/{cam_id}/roi` - Set ROI zone
- `GET /stats/{cam_id}` - Get camera stats

### Tracking API (NEW)
- `GET /api/tracking/people` - Get all tracked people
- `GET /api/tracking/person/{id}` - Get person's current location
- `GET /api/tracking/person/{id}/history` - Get movement history
- `GET /api/tracking/stats` - Get overall tracking statistics
- `POST /api/tracking/event` - Record movement event

## Configuration

Edit `config.py` for:
- Face recognition thresholds
- Processing FPS
- Deduplication settings
- ROI padding
- Time windows

## Tech Stack

**Frontend:**
- React 19 + TypeScript
- Vite 7
- Tailwind CSS 3
- ShadCN UI Components
- React Router 6
- Axios

**Backend:**
- FastAPI
- InsightFace (buffalo_s model)
- ONNX Runtime (CUDA)
- OpenCV
- SQLite

**AI/ML:**
- Face Detection: SCRFD (500M parameters)
- Face Recognition: ArcFace (600K identities)
- Gender/Age: GenderAge model
- Landmarks: 2D/3D facial landmarks

## Project Structure

```
pepCount-IF/
├── frontend/              # React frontend
│   ├── src/
│   │   ├── _comps/       # Reusable components
│   │   ├── components/   # UI components
│   │   ├── pages/        # Page components
│   │   └── lib/          # Utilities
│   └── package.json
├── core/                 # Backend core modules
│   ├── camera_manager.py
│   ├── recognizer.py
│   ├── tracking_db.py   # NEW: Tracking database
│   └── ...
├── web_app.py           # FastAPI application
├── config.py            # Configuration
├── requirements.txt     # Python dependencies
└── start.bat           # Quick start script
```

## Scalability Considerations

### Performance
- GPU acceleration reduces processing time by 10-20x
- FPS throttling to balance accuracy vs performance
- Deduplication prevents duplicate counting
- Indexed database queries for fast lookups

### Database
- SQLite for simple deployment
- Can migrate to PostgreSQL for high concurrency
- Partitioning by date for large datasets
- Regular cleanup of old detection history

### Multi-Camera
- Async processing per camera
- Shared face recognition model
- Centralized tracking database
- Camera-agnostic person IDs

### Future Enhancements
- Real-time WebSocket updates
- Heat map visualization
- Path prediction algorithms
- Integration with access control systems
- Cloud deployment (AWS/Azure)
- Distributed processing with Redis
- Video archival with S3

## License
All rights reserved - OptiExacta 2026
