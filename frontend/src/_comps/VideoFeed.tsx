import { useState, useRef, useEffect, useCallback } from 'react'
import { Pencil, X, Trash2, Clock, MapPin, Minus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import axios from 'axios'
import ROIDrawer from './ROIDrawer'
import TimerSettings from './TimerSettings'

interface VideoFeedProps {
  cameraId: string
  showTimer?: boolean
  showROIOverlay?: boolean
}

interface TimerData {
  time_window_seconds: number
  elapsed_seconds: number
  remaining_seconds: number
}

interface ROIData {
  roi_points: { x: number; y: number }[]
  rot_points: { x: number; y: number }[]
  line_crossing_enabled?: boolean
}

export default function VideoFeed({ cameraId, showTimer = true, showROIOverlay = true }: VideoFeedProps) {
  const [showROIDrawer, setShowROIDrawer] = useState(false)
  const [timerData, setTimerData] = useState<TimerData | null>(null)
  const [roiData, setRoiData] = useState<ROIData | null>(null)
  const [lineCrossingEnabled, setLineCrossingEnabled] = useState(false)
  const [roiMode, setRoiMode] = useState(false)
  const [roiPoints, setRoiPoints] = useState<{ x: number; y: number }[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLImageElement>(null)

  // Fetch timer data
  const fetchTimerData = useCallback(async () => {
    if (!showTimer) return
    try {
      const res = await axios.get(`/api/camera/${cameraId}/timer`)
      setTimerData(res.data)
    } catch (error) {
      console.error('Error fetching timer:', error)
    }
  }, [cameraId, showTimer])

  // Fetch ROI data
  const fetchROIData = useCallback(async () => {
    try {
      const res = await axios.get(`/api/camera/${cameraId}/roi`)
      setRoiData(res.data)
      setLineCrossingEnabled(res.data.line_crossing_enabled || false)
    } catch (error) {
      console.error('Error fetching ROI:', error)
    }
  }, [cameraId])

  // Initial fetch and timer refresh
  useEffect(() => {
    fetchTimerData()
    fetchROIData()
    
    const timerInterval = setInterval(fetchTimerData, 1000)
    const roiInterval = setInterval(fetchROIData, 5000)
    
    return () => {
      clearInterval(timerInterval)
      clearInterval(roiInterval)
    }
  }, [fetchTimerData, fetchROIData])

  // Draw ROI overlay
  useEffect(() => {
    if (!showROIOverlay || !overlayCanvasRef.current || !containerRef.current) return
    
    const canvas = overlayCanvasRef.current
    const container = containerRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Match canvas size to container
    canvas.width = container.offsetWidth
    canvas.height = container.offsetHeight
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw ROI polygon
    if (roiData?.roi_points && roiData.roi_points.length > 2) {
      ctx.beginPath()
      ctx.strokeStyle = '#22c55e'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      
      roiData.roi_points.forEach((point, idx) => {
        // Scale points to canvas size (assume original 640x480)
        const x = (point.x / 640) * canvas.width
        const y = (point.y / 480) * canvas.height
        
        if (idx === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      ctx.closePath()
      ctx.stroke()
      
      ctx.fillStyle = 'rgba(34, 197, 94, 0.1)'
      ctx.fill()
    }
    
    // Draw Entry/Exit line (ROT)
    if (roiData?.rot_points && roiData.rot_points.length >= 2) {
      ctx.beginPath()
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 3
      ctx.setLineDash([])
      
      const start = roiData.rot_points[0]
      const end = roiData.rot_points[1]
      
      const x1 = (start.x / 640) * canvas.width
      const y1 = (start.y / 480) * canvas.height
      const x2 = (end.x / 640) * canvas.width
      const y2 = (end.y / 480) * canvas.height
      
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
      
      // Draw direction arrow
      const angle = Math.atan2(y2 - y1, x2 - x1)
      const midX = (x1 + x2) / 2
      const midY = (y1 + y2) / 2
      
      ctx.beginPath()
      ctx.moveTo(midX, midY)
      ctx.lineTo(midX - 10 * Math.cos(angle - Math.PI / 6), midY - 10 * Math.sin(angle - Math.PI / 6))
      ctx.moveTo(midX, midY)
      ctx.lineTo(midX - 10 * Math.cos(angle + Math.PI / 6), midY - 10 * Math.sin(angle + Math.PI / 6))
      ctx.stroke()
    }
  }, [roiData, showROIOverlay])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!roiMode || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setRoiPoints([...roiPoints, { x, y }])
  }

  const saveROI = async () => {
    if (roiPoints.length < 3) return

    try {
      const formData = new FormData()
      formData.append('roi_points', JSON.stringify(roiPoints))
      await axios.post(`/api/camera/${cameraId}/roi`, formData)
      setRoiMode(false)
      setRoiPoints([])
      fetchROIData()
    } catch (error) {
      console.error('Error saving ROI:', error)
    }
  }

  const clearROI = async () => {
    try {
      await axios.delete(`/api/camera/${cameraId}/roi`)
      setRoiPoints([])
      fetchROIData()
    } catch (error) {
      console.error('Error clearing ROI:', error)
    }
  }

  const clearROT = async () => {
    try {
      await axios.delete(`/api/camera/${cameraId}/rot`)
      fetchROIData()
    } catch (error) {
      console.error('Error clearing ROT:', error)
    }
  }

  const clearLineCrossing = async () => {
    try {
      await axios.delete(`/api/camera/${cameraId}/line_crossing`)
      setLineCrossingEnabled(false)
      fetchROIData()
    } catch (error) {
      console.error('Error clearing line crossing:', error)
    }
  }

  const clearAll = async () => {
    try {
      await axios.delete(`/api/camera/${cameraId}/roi`)
      await axios.delete(`/api/camera/${cameraId}/rot`)
      await axios.delete(`/api/camera/${cameraId}/line_crossing`)
      setRoiPoints([])
      setLineCrossingEnabled(false)
      fetchROIData()
    } catch (error) {
      console.error('Error clearing all:', error)
    }
  }

  const hasROI = roiData?.roi_points && roiData.roi_points.length > 0
  const hasROT = roiData?.rot_points && roiData.rot_points.length > 0

  return (
    <div ref={containerRef} className="relative w-full h-full bg-black rounded-lg overflow-hidden">
      <img
        ref={videoRef}
        src={`/api/video_feed/${cameraId}`}
        alt="Live Feed"
        className="w-full h-full object-contain"
      />
      
      {/* ROI Overlay Canvas */}
      {showROIOverlay && !showROIDrawer && (
        <canvas
          ref={overlayCanvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      )}
      
      {roiMode && (
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="absolute top-0 left-0 w-full h-full cursor-crosshair"
          style={{ background: 'rgba(0,0,0,0.3)' }}
        />
      )}
      
      {/* ROI Drawer Overlay */}
      {showROIDrawer && (
        <ROIDrawer
          cameraId={cameraId}
          videoWidth={640}
          videoHeight={480}
          onSave={() => {
            setShowROIDrawer(false)
            fetchROIData()
          }}
          onClose={() => setShowROIDrawer(false)}
        />
      )}
      
      {/* Timer Display */}
      {showTimer && timerData && (
        <div className="absolute top-4 left-4 bg-black/70 backdrop-blur px-3 py-2 rounded-lg">
          <div className="flex items-center gap-2 text-white">
            <Clock className="w-4 h-4 text-yellow-400" />
            <span className="text-sm font-mono">
              Reset in: {formatTime(timerData.remaining_seconds)}
            </span>
            <TimerSettings 
              cameraId={cameraId} 
              currentTimeWindow={timerData.time_window_seconds}
              onTimerUpdated={fetchTimerData}
            />
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Window: {formatTime(timerData.time_window_seconds)}
          </div>
        </div>
      )}
      
      {/* ROI/ROT/Line Crossing Status Indicators */}
      {(hasROI || hasROT || lineCrossingEnabled) && !showROIDrawer && (
        <div className="absolute bottom-4 left-4 flex gap-2 flex-wrap">
          {hasROI && (
            <div className="flex items-center gap-1 bg-green-600/80 text-white px-2 py-1 rounded text-xs">
              <MapPin className="w-3 h-3" />
              ROI Active
              <button 
                onClick={clearROI}
                className="ml-1 hover:text-red-300"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          {hasROT && (
            <div className="flex items-center gap-1 bg-red-600/80 text-white px-2 py-1 rounded text-xs">
              Entry/Exit Line
              <button 
                onClick={clearROT}
                className="ml-1 hover:text-red-300"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          {lineCrossingEnabled && (
            <div className="flex items-center gap-1 bg-cyan-600/80 text-white px-2 py-1 rounded text-xs">
              <Minus className="w-3 h-3" />
              Line Crossing
              <button 
                onClick={clearLineCrossing}
                className="ml-1 hover:text-red-300"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
        </div>
      )}
      
      {/* Toolbar */}
      <div className="absolute top-4 right-4 flex gap-2">
        {!roiMode && !showROIDrawer && (
          <>
            <Button 
              size="sm" 
              onClick={() => setShowROIDrawer(true)}
              className="bg-purple-600 hover:bg-purple-700"
            >
              <Pencil className="w-4 h-4 mr-1" />
              Draw ROI
            </Button>
            {(hasROI || hasROT || lineCrossingEnabled) && (
              <Button 
                size="sm" 
                variant="destructive"
                onClick={clearAll}
              >
                <Trash2 className="w-4 h-4 mr-1" />
                Clear All
              </Button>
            )}
          </>
        )}
        {roiMode && (
          <>
            <Button size="sm" onClick={saveROI} disabled={roiPoints.length < 3}>
              Save ROI ({roiPoints.length})
            </Button>
            <Button size="sm" variant="secondary" onClick={clearROI}>
              Clear
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => {
                setRoiMode(false)
                setRoiPoints([])
              }}
            >
              <X className="w-4 h-4" />
            </Button>
          </>
        )}
      </div>
    </div>
  )
}
