import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { 
  Pencil, Square, Minus, Trash2, Save, X, 
  MousePointer, RotateCcw 
} from 'lucide-react'
import axios from 'axios'

interface Point {
  x: number
  y: number
}

interface ROIDrawerProps {
  cameraId: string
  videoWidth: number
  videoHeight: number
  onSave?: () => void
  onClose?: () => void
}

type DrawingMode = 'none' | 'polygon' | 'box' | 'line'

export default function ROIDrawer({ 
  cameraId, 
  videoWidth, 
  videoHeight, 
  onSave,
  onClose 
}: ROIDrawerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [mode, setMode] = useState<DrawingMode>('none')
  const [roiPoints, setRoiPoints] = useState<Point[]>([])
  const [linePoints, setLinePoints] = useState<Point[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState<Point | null>(null)
  const [currentPoint, setCurrentPoint] = useState<Point | null>(null)

  // Redraw canvas when points change
  useEffect(() => {
    drawCanvas()
  }, [roiPoints, linePoints, currentPoint, mode])

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw ROI polygon/box
    if (roiPoints.length > 0) {
      ctx.beginPath()
      ctx.strokeStyle = '#22c55e'  // Green
      ctx.fillStyle = 'rgba(34, 197, 94, 0.2)'
      ctx.lineWidth = 2

      ctx.moveTo(roiPoints[0].x * canvas.width, roiPoints[0].y * canvas.height)
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i].x * canvas.width, roiPoints[i].y * canvas.height)
      }
      
      // Close polygon if complete
      if (mode !== 'polygon' || roiPoints.length >= 3) {
        ctx.closePath()
        ctx.fill()
      }
      ctx.stroke()

      // Draw points
      roiPoints.forEach((point, index) => {
        ctx.beginPath()
        ctx.fillStyle = index === 0 ? '#22c55e' : '#ffffff'
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 6, 0, 2 * Math.PI)
        ctx.fill()
        ctx.strokeStyle = '#22c55e'
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }

    // Draw current point for box mode
    if (mode === 'box' && startPoint && currentPoint) {
      ctx.beginPath()
      ctx.strokeStyle = '#3b82f6'  // Blue
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])

      const x = startPoint.x * canvas.width
      const y = startPoint.y * canvas.height
      const w = (currentPoint.x - startPoint.x) * canvas.width
      const h = (currentPoint.y - startPoint.y) * canvas.height

      ctx.rect(x, y, w, h)
      ctx.fill()
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw entry/exit line
    if (linePoints.length > 0) {
      ctx.beginPath()
      ctx.strokeStyle = '#ef4444'  // Red
      ctx.lineWidth = 3
      ctx.setLineDash([])

      ctx.moveTo(linePoints[0].x * canvas.width, linePoints[0].y * canvas.height)
      if (linePoints.length > 1) {
        ctx.lineTo(linePoints[1].x * canvas.width, linePoints[1].y * canvas.height)
      } else if (currentPoint && mode === 'line') {
        ctx.lineTo(currentPoint.x * canvas.width, currentPoint.y * canvas.height)
      }
      ctx.stroke()

      // Draw arrow indicating direction
      if (linePoints.length === 2) {
        const midX = ((linePoints[0].x + linePoints[1].x) / 2) * canvas.width
        const midY = ((linePoints[0].y + linePoints[1].y) / 2) * canvas.height
        
        ctx.fillStyle = '#ef4444'
        ctx.font = '12px sans-serif'
        ctx.fillText('Entry/Exit Line', midX - 40, midY - 10)
      }

      // Draw endpoints
      linePoints.forEach((point) => {
        ctx.beginPath()
        ctx.fillStyle = '#ef4444'
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 8, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    // Draw current point for line mode
    if (mode === 'line' && linePoints.length === 1 && currentPoint) {
      ctx.beginPath()
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.moveTo(linePoints[0].x * canvas.width, linePoints[0].y * canvas.height)
      ctx.lineTo(currentPoint.x * canvas.width, currentPoint.y * canvas.height)
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [roiPoints, linePoints, currentPoint, mode, startPoint])

  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) }
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getCanvasPoint(e)

    if (mode === 'polygon') {
      // Add point to polygon
      setRoiPoints([...roiPoints, point])
    } else if (mode === 'line') {
      if (linePoints.length < 2) {
        setLinePoints([...linePoints, point])
      }
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode === 'box') {
      const point = getCanvasPoint(e)
      setStartPoint(point)
      setIsDrawing(true)
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getCanvasPoint(e)
    setCurrentPoint(point)
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode === 'box' && isDrawing && startPoint) {
      const endPoint = getCanvasPoint(e)
      // Create rectangle from start and end points
      const minX = Math.min(startPoint.x, endPoint.x)
      const minY = Math.min(startPoint.y, endPoint.y)
      const maxX = Math.max(startPoint.x, endPoint.x)
      const maxY = Math.max(startPoint.y, endPoint.y)

      setRoiPoints([
        { x: minX, y: minY },
        { x: maxX, y: minY },
        { x: maxX, y: maxY },
        { x: minX, y: maxY }
      ])
      setIsDrawing(false)
      setStartPoint(null)
      setMode('none')
    }
  }

  const handleSave = async () => {
    try {
      const formData = new FormData()
      formData.append('roi_points', JSON.stringify(roiPoints))
      formData.append('rot_points', JSON.stringify([]))

      await axios.post(`/api/camera/${cameraId}/roi`, formData)
      
      // Also set line crossing if line is defined
      if (linePoints.length === 2) {
        const lineY = (linePoints[0].y + linePoints[1].y) / 2
        await axios.post(`/api/camera/${cameraId}/line_crossing`, new URLSearchParams({
          enabled: 'true',
          line_position: lineY.toString()
        }))
      }

      onSave?.()
    } catch (error) {
      console.error('Error saving ROI:', error)
    }
  }

  const handleClear = () => {
    setRoiPoints([])
    setLinePoints([])
    setMode('none')
  }

  const handleUndo = () => {
    if (mode === 'polygon' && roiPoints.length > 0) {
      setRoiPoints(roiPoints.slice(0, -1))
    } else if (mode === 'line' && linePoints.length > 0) {
      setLinePoints(linePoints.slice(0, -1))
    }
  }

  return (
    <Card className="bg-gray-900/95 border-gray-700 absolute inset-0 z-50 m-4">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Pencil className="w-5 h-5 text-purple-400" />
            Draw ROI & Entry/Exit Line
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {/* Toolbar */}
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
            <Button
              variant={mode === 'none' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setMode('none')}
              title="Select"
            >
              <MousePointer className="w-4 h-4" />
            </Button>
            <Button
              variant={mode === 'box' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => { setMode('box'); setRoiPoints([]) }}
              title="Draw Box ROI"
            >
              <Square className="w-4 h-4" />
              <span className="ml-1 text-xs">Box</span>
            </Button>
            <Button
              variant={mode === 'polygon' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => { setMode('polygon'); setRoiPoints([]) }}
              title="Draw Polygon ROI"
            >
              <Pencil className="w-4 h-4" />
              <span className="ml-1 text-xs">Polygon</span>
            </Button>
            <Button
              variant={mode === 'line' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => { setMode('line'); setLinePoints([]) }}
              title="Draw Entry/Exit Line"
            >
              <Minus className="w-4 h-4" />
              <span className="ml-1 text-xs">Line</span>
            </Button>
          </div>

          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleUndo}
              title="Undo"
            >
              <RotateCcw className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              className="text-red-400 hover:text-red-300"
              title="Clear All"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex-1" />

          <Button
            variant="default"
            size="sm"
            onClick={handleSave}
            className="bg-green-600 hover:bg-green-700"
          >
            <Save className="w-4 h-4 mr-1" />
            Save ROI
          </Button>
        </div>

        {/* Instructions */}
        <div className="text-xs text-gray-500 mb-3">
          {mode === 'box' && '🔲 Click and drag to draw a rectangular ROI'}
          {mode === 'polygon' && '✏️ Click to add points. Close the polygon by clicking near the first point.'}
          {mode === 'line' && '➖ Click to set the start and end points of the entry/exit line'}
          {mode === 'none' && '👆 Select a drawing tool above to start'}
        </div>

        {/* Canvas overlay (positioned over video) */}
        <div className="relative bg-gray-800 rounded-lg overflow-hidden">
          {/* Video background */}
          <img
            src={`/api/video_feed/${cameraId}`}
            alt="Camera feed"
            className="w-full h-auto"
            style={{ maxHeight: 'calc(100vh - 300px)' }}
          />
          
          {/* Drawing canvas */}
          <canvas
            ref={canvasRef}
            width={videoWidth || 640}
            height={videoHeight || 480}
            className="absolute inset-0 w-full h-full cursor-crosshair"
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => setCurrentPoint(null)}
          />
        </div>

        {/* Status */}
        <div className="flex items-center gap-4 mt-3 text-sm">
          {roiPoints.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded" />
              <span className="text-gray-400">
                ROI: {roiPoints.length} points
              </span>
            </div>
          )}
          {linePoints.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded" />
              <span className="text-gray-400">
                Line: {linePoints.length}/2 points
              </span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
