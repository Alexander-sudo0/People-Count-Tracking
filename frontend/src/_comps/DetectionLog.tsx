import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Clock, User, Camera, Maximize2 } from 'lucide-react'
import axios from 'axios'

interface Detection {
  id: number
  person_id: string
  name?: string | null
  in_watchlist?: boolean
  match_confidence?: number | null
  camera_id: string
  camera_name: string
  detected_at: string
  confidence: number
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
  in_roi: boolean
  event_type: string
  thumbnail_url: string | null
  fullframe_url: string | null
}

interface DetectionLogProps {
  cameraId: string
}

export default function DetectionLog({ cameraId }: DetectionLogProps) {
  const [detections, setDetections] = useState<Detection[]>([])
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null)
  const [showFullframe, setShowFullframe] = useState(false)

  useEffect(() => {
    if (!cameraId) return

    const fetchDetections = async () => {
      try {
        const res = await axios.get(`/api/events/detections/${cameraId}?limit=20`)
        setDetections(res.data.detections || [])
      } catch (error) {
        console.error('Error fetching detections:', error)
      }
    }

    fetchDetections()
    const interval = setInterval(fetchDetections, 3000)
    return () => clearInterval(interval)
  }, [cameraId])

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    })
  }

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    })
  }

  return (
    <>
      <Card className="bg-gray-800/50 border-gray-700 mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <User className="w-5 h-5 text-blue-400" />
            Detection Log
            <span className="text-sm text-gray-500 font-normal">
              ({detections.length} recent)
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {detections.length === 0 ? (
            <div className="text-center text-gray-500 py-6">
              <User className="w-10 h-10 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No detections yet</p>
            </div>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 2xl:grid-cols-10 gap-2">
              {detections.map((detection) => (
                <div
                  key={detection.id}
                  className="bg-gray-900/50 rounded border border-gray-700 overflow-hidden cursor-pointer hover:border-purple-500 transition-all group"
                  onClick={() => {
                    setSelectedDetection(detection)
                    setShowFullframe(true)
                  }}
                >
                  {/* Thumbnail */}
                  <div className="relative aspect-square bg-gray-800">
                    {detection.thumbnail_url ? (
                      <img
                        src={detection.thumbnail_url}
                        alt={`Face ${detection.person_id}`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-600">
                        <User className="w-6 h-6" />
                      </div>
                    )}
                    {/* Overlay on hover */}
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Maximize2 className="w-4 h-4 text-white" />
                    </div>
                    {/* ROI Badge */}
                    {detection.in_roi && (
                      <div className="absolute top-0.5 right-0.5 bg-green-500 text-[10px] px-1 py-0.5 rounded text-white font-medium">
                        ROI
                      </div>
                    )}
                    {/* Event type badge */}
                    {detection.event_type === 'first_detection' && (
                      <div className="absolute top-0.5 left-0.5 bg-blue-500 text-[10px] px-1 py-0.5 rounded text-white font-medium">
                        NEW
                      </div>
                    )}
                  </div>
                  {/* Info - Compact */}
                  <div className="p-1.5">
                    <div className="text-[10px] font-mono text-purple-400 truncate">
                      {detection.name ? (
                        <span className="text-green-400 font-semibold">{detection.name}</span>
                      ) : (
                        <span>{detection.person_id.substring(0, 6)}</span>
                      )}
                      {detection.in_watchlist && (
                        <span className="ml-0.5 text-yellow-500" title="In Watchlist">⭐</span>
                      )}
                    </div>
                    <div className="text-[10px] text-gray-500 flex items-center gap-0.5">
                      <Clock className="w-2.5 h-2.5" />
                      {formatTime(detection.detected_at)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Fullframe Modal */}
      <Dialog open={showFullframe} onOpenChange={setShowFullframe}>
        <DialogContent className="max-w-4xl bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <User className="w-5 h-5 text-purple-400" />
              Detection Details
            </DialogTitle>
          </DialogHeader>
          {selectedDetection && (
            <div className="space-y-4">
              {/* Full frame image */}
              <div className="relative bg-gray-800 rounded-lg overflow-hidden">
                {selectedDetection.fullframe_url ? (
                  <img
                    src={selectedDetection.fullframe_url}
                    alt="Full frame"
                    className="w-full h-auto max-h-[60vh] object-contain"
                  />
                ) : (
                  <div className="w-full h-64 flex items-center justify-center text-gray-600">
                    <Camera className="w-16 h-16" />
                  </div>
                )}
              </div>

              {/* Detection info */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Person</div>
                  <div className="font-mono text-sm text-purple-400">
                    {selectedDetection.name ? (
                      <span className="text-green-400 font-semibold flex items-center gap-1">
                        {selectedDetection.name}
                        {selectedDetection.in_watchlist && (
                          <span className="text-yellow-500">⭐</span>
                        )}
                      </span>
                    ) : (
                      selectedDetection.person_id
                    )}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Detected At</div>
                  <div className="text-sm">
                    {formatDate(selectedDetection.detected_at)}
                    <br />
                    {formatTime(selectedDetection.detected_at)}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Detection Confidence</div>
                  <div className="text-sm font-semibold text-green-400">
                    {(selectedDetection.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Status</div>
                  <div className="flex gap-2">
                    {selectedDetection.in_roi && (
                      <span className="bg-green-500/20 text-green-400 text-xs px-2 py-1 rounded">
                        In ROI
                      </span>
                    )}
                    {selectedDetection.event_type === 'first_detection' && (
                      <span className="bg-blue-500/20 text-blue-400 text-xs px-2 py-1 rounded">
                        First Detection
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Thumbnail comparison */}
              {selectedDetection.thumbnail_url && (
                <div className="flex items-start gap-4">
                  <div>
                    <div className="text-xs text-gray-500 mb-2">Face Thumbnail</div>
                    <img
                      src={selectedDetection.thumbnail_url}
                      alt="Face thumbnail"
                      className="w-24 h-24 object-cover rounded-lg border border-gray-700"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
