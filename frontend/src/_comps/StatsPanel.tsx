import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Clock, RotateCcw, LogIn, LogOut, Users } from 'lucide-react'
import { Button } from '@/components/ui/button'
import axios from 'axios'

interface StatsData {
  unique_count: number
  entry_count: number
  exit_count: number
  current_count: number
  line_crossing_enabled?: boolean
}

interface TimerData {
  time_window_seconds: number
  elapsed_seconds: number
  remaining_seconds: number
}

interface StatsPanelProps {
  stats: StatsData
  cameraId?: string
  cameraType?: string
}

export default function StatsPanel({ stats, cameraId, cameraType = 'entry' }: StatsPanelProps) {
  const [timerData, setTimerData] = useState<TimerData | null>(null)

  useEffect(() => {
    if (!cameraId) return

    const fetchTimer = async () => {
      try {
        const res = await axios.get(`/api/camera/${cameraId}/timer`)
        setTimerData(res.data)
      } catch (error) {
        console.error('Error fetching timer:', error)
      }
    }

    fetchTimer()
    const interval = setInterval(fetchTimer, 1000)
    return () => clearInterval(interval)
  }, [cameraId])

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    
    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleReset = async () => {
    if (!cameraId) return
    try {
      await axios.post(`/api/camera/${cameraId}/reset`)
    } catch (error) {
      console.error('Error resetting counts:', error)
    }
  }

  return (
    <div className="border-t border-gray-700 bg-gray-900/80 p-6">
      {/* Camera Type & Timer Bar */}
      <div className="mb-4 p-3 bg-gray-800/50 rounded-lg flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Camera Type Badge */}
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${
            cameraType === 'entry' 
              ? 'bg-green-500/20 border border-green-500/50' 
              : 'bg-red-500/20 border border-red-500/50'
          }`}>
            {cameraType === 'entry' ? (
              <LogIn className="w-4 h-4 text-green-400" />
            ) : (
              <LogOut className="w-4 h-4 text-red-400" />
            )}
            <span className={`font-semibold uppercase text-sm ${
              cameraType === 'entry' ? 'text-green-400' : 'text-red-400'
            }`}>
              {cameraType === 'entry' ? 'Entry Camera' : 'Exit Camera'}
            </span>
          </div>

          {/* Timer */}
          {timerData && cameraId && (
            <>
              <div className="h-6 w-px bg-gray-600" />
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">Resets in: </span>
                <span className="font-mono text-lg text-white font-semibold">
                  {formatTime(timerData.remaining_seconds)}
                </span>
              </div>
            </>
          )}
        </div>
        
        <Button
          variant="outline"
          size="sm"
          onClick={handleReset}
          className="border-gray-600 hover:bg-gray-700"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Reset All
        </Button>
      </div>

      {/* Stats Cards - Always show synced entry/exit/currently inside */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-gradient-to-br from-green-600/20 to-green-800/20 border-green-500/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400">Entered</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-400">{stats.entry_count}</div>
            <p className="text-xs text-gray-500 mt-1">Total people who entered</p>
          </CardContent>
        </Card>
        <Card className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 border-purple-500/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400">Currently Inside</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-purple-400">{stats.current_count}</div>
            <p className="text-xs text-gray-500 mt-1">People currently in area</p>
          </CardContent>
        </Card>
        <Card className="bg-gradient-to-br from-red-600/20 to-red-800/20 border-red-500/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-400">Exited</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-400">{stats.exit_count}</div>
            <p className="text-xs text-gray-500 mt-1">Total people who exited</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
