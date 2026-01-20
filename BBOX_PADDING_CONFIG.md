# Face Bounding Box Padding Configuration

## Overview
The face bounding box padding settings control how much context around the detected face is included when saving face crops. This allows capturing ears, forehead, chin, and other features that might be cut off with tight bounding boxes.

## Settings Parameters

All parameters are stored in `config.py` and persist in `app_settings.json`:

```python
BBOX_PADDING_TOP = 0.30      # Padding above face (% of face height)
BBOX_PADDING_BOTTOM = 0.25   # Padding below face (% of face height)
BBOX_PADDING_LEFT = 0.20     # Padding left side (% of face width)
BBOX_PADDING_RIGHT = 0.20    # Padding right side (% of face width)
```

## How It Works

### Original Bounding Box
The face detection model returns a bounding box `[x1, y1, x2, y2]` that tightly fits the detected face.

### Expanded Bounding Box
With padding, the bbox is expanded by:
- **Top**: `face_height × BBOX_PADDING_TOP`
- **Bottom**: `face_height × BBOX_PADDING_BOTTOM`
- **Left**: `face_width × BBOX_PADDING_LEFT`
- **Right**: `face_width × BBOX_PADDING_RIGHT`

### Example
For a face with dimensions 100x120 pixels:
- Original bbox: [100, 50, 200, 170]
- Top padding (0.30): 120 × 0.30 = 36 pixels
- Bottom padding (0.25): 120 × 0.25 = 30 pixels
- Left padding (0.20): 100 × 0.20 = 20 pixels
- Right padding (0.20): 100 × 0.20 = 20 pixels

**Expanded bbox**: [80, 14, 220, 200]

## Current Default Values

| Parameter | Value | Effect |
|-----------|-------|--------|
| BBOX_PADDING_TOP | 0.30 | Captures forehead + 30% of face height above |
| BBOX_PADDING_BOTTOM | 0.25 | Captures chin + 25% of face height below |
| BBOX_PADDING_LEFT | 0.20 | Captures left side + 20% of face width |
| BBOX_PADDING_RIGHT | 0.20 | Captures right side + 20% of face width |

**Result**: ~36% more pixel area around faces (fixes ear/forehead cutoff)

## Adjustment Guide

### If capturing too much background:
**Decrease the padding values**
- Reduce BBOX_PADDING_TOP to 0.15-0.20 (less forehead)
- Reduce BBOX_PADDING_LEFT/RIGHT to 0.10-0.15 (less sides)
- Reduce BBOX_PADDING_BOTTOM to 0.15-0.20 (less chin)

### If still cutting off features (ears, forehead):
**Increase the padding values**
- Increase BBOX_PADDING_TOP to 0.40-0.50 (more forehead)
- Increase BBOX_PADDING_LEFT/RIGHT to 0.25-0.35 (more sides)
- Increase BBOX_PADDING_BOTTOM to 0.30-0.40 (more chin)

### Balanced settings (recommended):
```python
BBOX_PADDING_TOP = 0.35
BBOX_PADDING_BOTTOM = 0.30
BBOX_PADDING_LEFT = 0.25
BBOX_PADDING_RIGHT = 0.25
```

## Where Padding is Applied

### Live Stream Processing
- When saving face crops in `core/storage.py::save_face_image()`
- Applied to detected faces from cameras
- Results in files like `detected_faces/face_abc123_1234567890.jpg`

### Offline Video Processing
- When processing uploaded video files in `core/video_processor.py::_save_face_image()`
- Applied to all detected faces in batch processing
- Results in files like `detected_faces/video_face_xyz789_1234567890.jpg`

## Configuration Storage

Settings are persisted to `app_settings.json`:
```json
{
    "BBOX_PADDING_TOP": 0.30,
    "BBOX_PADDING_BOTTOM": 0.25,
    "BBOX_PADDING_LEFT": 0.20,
    "BBOX_PADDING_RIGHT": 0.20,
    ...
}
```

### Load/Save Flow
1. App starts → `Config.load_settings()` reads from `app_settings.json`
2. User adjusts padding via Settings UI
3. Frontend sends POST to `/api/settings`
4. Backend updates Config class + calls `Config.save_settings()`
5. Next frame uses new padding values
6. On app restart, `load_settings()` reapplies saved values

## API Reference

### GET /api/settings
Returns current configuration including padding values:
```json
{
    "BBOX_PADDING_TOP": 0.30,
    "BBOX_PADDING_BOTTOM": 0.25,
    "BBOX_PADDING_LEFT": 0.20,
    "BBOX_PADDING_RIGHT": 0.20,
    ...
}
```

### POST /api/settings
Update padding values (sent as form data):
```
bbox_padding_top=0.35&bbox_padding_bottom=0.30&bbox_padding_left=0.25&bbox_padding_right=0.25
```

## Boundary Clipping

The expanded bbox is automatically clipped to frame boundaries:
- If `x1 < 0` → clipped to 0
- If `y1 < 0` → clipped to 0
- If `x2 > frame_width` → clipped to frame_width
- If `y2 > frame_height` → clipped to frame_height

This ensures padding never extends beyond the video frame.

## Impact on Performance

- **Larger padding** = Larger face crop images = More disk space
- **Default (0.30/0.25/0.20)** ≈ 35% more storage vs. tight bbox
- **High padding (0.50+)** ≈ 100%+ more storage but captures full context

## Troubleshooting

### Problem: Faces still cut off at edges
- Check frame resolution vs. face position
- Faces very close to image edge may still be clipped
- Increase padding values, but monitor for boundary clipping

### Problem: Too much background captured
- Reduce padding values
- Try 0.15 for top/bottom, 0.10 for sides

### Problem: Settings not applying
- Restart the application after changing
- Check `app_settings.json` was written successfully
- Verify no read-only permissions on `app_settings.json`

## Technical Implementation

### Config Method: expand_bbox()
```python
@staticmethod
def expand_bbox(bbox, frame_h, frame_w):
    """Expand a bounding box with padding to include more context."""
    x1, y1, x2, y2 = bbox.astype(int)
    face_w = x2 - x1
    face_h = y2 - y1
    
    # Apply padding
    x1 -= int(face_w * Config.BBOX_PADDING_LEFT)
    x2 += int(face_w * Config.BBOX_PADDING_RIGHT)
    y1 -= int(face_h * Config.BBOX_PADDING_TOP)
    y2 += int(face_h * Config.BBOX_PADDING_BOTTOM)
    
    # Clip to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)
    
    return np.array([x1, y1, x2, y2])
```

This method is called by:
- `core/storage.py::save_face_image()` - Live stream face crops
- `core/video_processor.py::_save_face_image()` - Batch video processing
