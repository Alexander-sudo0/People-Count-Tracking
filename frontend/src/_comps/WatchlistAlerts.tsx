import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { AlertTriangle, User, Camera, Clock, Star, X, Maximize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import axios from 'axios'

interface WatchlistAlert {
  person_id: string
  name: string
  notes: string | null
  person_thumbnail: string | null
  camera_id: string
  camera_name: string
  detected_at: string
  confidence: number
  thumbnail_url: string | null
  fullframe_url: string | null
}

interface WatchlistAlertsProps {
  cameraId?: string
}

export default function WatchlistAlerts({ cameraId }: WatchlistAlertsProps) {
  const [alerts, setAlerts] = useState<WatchlistAlert[]>([])
  const [selectedAlert, setSelectedAlert] = useState<WatchlistAlert | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [dismissed, setDismissed] = useState<Set<string>>(new Set())

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await axios.get('/api/watchlist/alerts?minutes=5')
        const allAlerts = res.data.alerts || []
        
        // Convert paths to URLs
        const processedAlerts = allAlerts.map((alert: WatchlistAlert & { thumbnail_path?: string; fullframe_path?: string; person_thumbnail?: string }) => ({
          ...alert,
          thumbnail_url: alert.thumbnail_path 
            ? `/static/faces/${alert.thumbnail_path.split('/').pop() || alert.thumbnail_path.split('\\').pop()}`
            : null,
          fullframe_url: alert.fullframe_path
            ? `/static/faces/${alert.fullframe_path.split('/').pop() || alert.fullframe_path.split('\\').pop()}`
            : null,
          person_thumbnail_url: alert.person_thumbnail
            ? `/static/faces/${alert.person_thumbnail.split('/').pop() || alert.person_thumbnail.split('\\').pop()}`
            : null,
        }))
        
        // Filter by camera if specified and exclude dismissed
        const filtered = cameraId 
          ? processedAlerts.filter((a: WatchlistAlert) => a.camera_id === cameraId)
          : processedAlerts
        
        // Exclude dismissed alerts
        const active = filtered.filter((a: WatchlistAlert) => 
          !dismissed.has(`${a.person_id}-${a.detected_at}`)
        )
        
        setAlerts(active)
      } catch (error) {
        console.error('Error fetching watchlist alerts:', error)
      }
    }

    fetchAlerts()
    const interval = setInterval(fetchAlerts, 3000)
    return () => clearInterval(interval)
  }, [cameraId, dismissed])

  const dismissAlert = (alert: WatchlistAlert) => {
    const key = `${alert.person_id}-${alert.detected_at}`
    setDismissed(prev => new Set(prev).add(key))
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    })
  }

  if (alerts.length === 0) return null

  return (
    <>
      <Card className="bg-yellow-900/20 border-yellow-500/50 mb-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2 text-yellow-400">
            <AlertTriangle className="w-4 h-4 animate-pulse" />
            Watchlist Alert{alerts.length > 1 ? 's' : ''} ({alerts.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {alerts.slice(0, 5).map((alert, index) => (
            <div
              key={`${alert.person_id}-${alert.detected_at}-${index}`}
              className="flex items-center gap-3 p-3 bg-yellow-600/10 border border-yellow-500/30 rounded-lg cursor-pointer hover:bg-yellow-600/20 transition-colors"
              onClick={() => {
                setSelectedAlert(alert)
                setShowDetails(true)
              }}
            >
              {/* Thumbnail */}
              <div className="w-12 h-12 rounded-lg bg-gray-800 overflow-hidden flex-shrink-0">
                {alert.thumbnail_url ? (
                  <img
                    src={alert.thumbnail_url}
                    alt={alert.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none'
                    }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <User className="w-6 h-6 text-gray-600" />
                  </div>
                )}
              </div>
              
              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                  <span className="font-semibold text-yellow-300 truncate">
                    {alert.name || alert.person_id.substring(0, 8)}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-400 mt-1">
                  <span className="flex items-center gap-1">
                    <Camera className="w-3 h-3" />
                    {alert.camera_name}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatTime(alert.detected_at)}
                  </span>
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-white"
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation()
                    setSelectedAlert(alert)
                    setShowDetails(true)
                  }}
                >
                  <Maximize2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-red-400"
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation()
                    dismissAlert(alert)
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
          ))}
          
          {alerts.length > 5 && (
            <div className="text-center text-xs text-gray-500 pt-2">
              +{alerts.length - 5} more alerts
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alert Details Modal */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="max-w-3xl bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-yellow-400">
              <AlertTriangle className="w-5 h-5" />
              Watchlist Person Detected
            </DialogTitle>
          </DialogHeader>
          
          {selectedAlert && (
            <div className="space-y-4">
              {/* Person Info */}
              <div className="flex gap-6 p-4 bg-yellow-600/10 border border-yellow-500/30 rounded-lg">
                {/* Photo */}
                <div className="w-24 h-24 rounded-xl bg-gray-800 overflow-hidden flex-shrink-0">
                  {selectedAlert.thumbnail_url ? (
                    <img
                      src={selectedAlert.thumbnail_url}
                      alt={selectedAlert.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="w-12 h-12 text-gray-600" />
                    </div>
                  )}
                </div>
                
                {/* Details */}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                    <h3 className="text-xl font-bold text-yellow-300">
                      {selectedAlert.name || 'Unknown'}
                    </h3>
                  </div>
                  <div className="font-mono text-sm text-purple-400 mb-2">
                    ID: {selectedAlert.person_id}
                  </div>
                  {selectedAlert.notes && (
                    <div className="text-sm text-gray-400 bg-gray-800/50 p-2 rounded">
                      {selectedAlert.notes}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Detection Info */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Camera</div>
                  <div className="flex items-center gap-2">
                    <Camera className="w-4 h-4 text-blue-400" />
                    <span>{selectedAlert.camera_name}</span>
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Detected At</div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-green-400" />
                    <span>{new Date(selectedAlert.detected_at).toLocaleString()}</span>
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Confidence</div>
                  <div className="text-green-400 font-semibold">
                    {(selectedAlert.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              
              {/* Full Frame */}
              {(selectedAlert.fullframe_url || selectedAlert.thumbnail_url) && (
                <div className="bg-gray-800 rounded-lg overflow-hidden">
                  <img
                    src={selectedAlert.fullframe_url || selectedAlert.thumbnail_url || ''}
                    alt="Detection frame"
                    className="w-full h-auto max-h-[50vh] object-contain"
                  />
                </div>
              )}
              
              {/* Actions */}
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => {
                    dismissAlert(selectedAlert)
                    setShowDetails(false)
                  }}
                >
                  Dismiss Alert
                </Button>
                <Button
                  className="flex-1 bg-purple-600 hover:bg-purple-700"
                  onClick={() => {
                    window.location.href = `/tracking?person=${selectedAlert.person_id}`
                  }}
                >
                  Track Person
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
