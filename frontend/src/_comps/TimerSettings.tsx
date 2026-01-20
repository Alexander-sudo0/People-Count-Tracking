import { useState, useEffect } from 'react'
import { Clock, Save, RotateCcw, Settings2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import axios from 'axios'

interface TimerSettingsProps {
  cameraId: string
  currentTimeWindow: number // in seconds
  onTimerUpdated?: () => void
}

const PRESET_TIMES = [
  { label: '5 minutes', value: 300 },
  { label: '15 minutes', value: 900 },
  { label: '30 minutes', value: 1800 },
  { label: '1 hour', value: 3600 },
  { label: '2 hours', value: 7200 },
  { label: '4 hours', value: 14400 },
  { label: '8 hours', value: 28800 },
  { label: '12 hours', value: 43200 },
  { label: '24 hours', value: 86400 },
]

export default function TimerSettings({ cameraId, currentTimeWindow, onTimerUpdated }: TimerSettingsProps) {
  const [open, setOpen] = useState(false)
  const [hours, setHours] = useState(Math.floor(currentTimeWindow / 3600))
  const [minutes, setMinutes] = useState(Math.floor((currentTimeWindow % 3600) / 60))
  const [saving, setSaving] = useState(false)
  const [resetting, setResetting] = useState(false)

  useEffect(() => {
    setHours(Math.floor(currentTimeWindow / 3600))
    setMinutes(Math.floor((currentTimeWindow % 3600) / 60))
  }, [currentTimeWindow])

  const handlePresetChange = (value: string) => {
    const seconds = parseInt(value)
    setHours(Math.floor(seconds / 3600))
    setMinutes(Math.floor((seconds % 3600) / 60))
  }

  const handleSave = async () => {
    const totalSeconds = hours * 3600 + minutes * 60
    if (totalSeconds < 60) {
      alert('Minimum time window is 1 minute')
      return
    }

    setSaving(true)
    try {
      await axios.put(`/api/camera/${cameraId}/timer`, {
        time_window_seconds: totalSeconds
      })
      onTimerUpdated?.()
      setOpen(false)
    } catch (error) {
      console.error('Error updating timer:', error)
      alert('Failed to update timer')
    } finally {
      setSaving(false)
    }
  }

  const handleResetNow = async () => {
    if (!confirm('This will reset the count immediately and clear all current detections. Continue?')) {
      return
    }

    setResetting(true)
    try {
      await axios.put(`/api/camera/${cameraId}/timer`, {
        reset_now: true
      })
      onTimerUpdated?.()
    } catch (error) {
      console.error('Error resetting timer:', error)
      alert('Failed to reset timer')
    } finally {
      setResetting(false)
    }
  }

  const formatCurrentTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    if (h > 0) {
      return m > 0 ? `${h}h ${m}m` : `${h}h`
    }
    return `${m}m`
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="text-white hover:bg-white/20">
          <Settings2 className="w-4 h-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px] bg-gray-900 text-white border-gray-700">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-yellow-400" />
            Timer Settings
          </DialogTitle>
          <DialogDescription className="text-gray-400">
            Set the time window for counting unique people. Count resets automatically when the timer expires.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6 py-4">
          {/* Current Setting */}
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-sm text-gray-400">Current Time Window</p>
            <p className="text-2xl font-bold text-yellow-400">{formatCurrentTime(currentTimeWindow)}</p>
          </div>

          {/* Preset Selection */}
          <div className="space-y-2">
            <Label>Quick Presets</Label>
            <Select onValueChange={handlePresetChange}>
              <SelectTrigger className="bg-gray-800 border-gray-600">
                <SelectValue placeholder="Select a preset time" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-600">
                {PRESET_TIMES.map((preset) => (
                  <SelectItem key={preset.value} value={preset.value.toString()}>
                    {preset.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Custom Time */}
          <div className="space-y-2">
            <Label>Custom Time</Label>
            <div className="flex gap-4">
              <div className="flex-1">
                <Label className="text-xs text-gray-400">Hours</Label>
                <Input
                  type="number"
                  min={0}
                  max={168}
                  value={hours}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setHours(Math.max(0, parseInt(e.target.value) || 0))}
                  className="bg-gray-800 border-gray-600"
                />
              </div>
              <div className="flex-1">
                <Label className="text-xs text-gray-400">Minutes</Label>
                <Input
                  type="number"
                  min={0}
                  max={59}
                  value={minutes}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMinutes(Math.max(0, Math.min(59, parseInt(e.target.value) || 0)))}
                  className="bg-gray-800 border-gray-600"
                />
              </div>
            </div>
            <p className="text-xs text-gray-500">
              New time: {hours}h {minutes}m = {hours * 3600 + minutes * 60} seconds
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button 
              onClick={handleSave} 
              disabled={saving}
              className="flex-1 bg-blue-600 hover:bg-blue-700"
            >
              <Save className="w-4 h-4 mr-2" />
              {saving ? 'Saving...' : 'Save Timer'}
            </Button>
            <Button 
              onClick={handleResetNow} 
              disabled={resetting}
              variant="destructive"
              className="flex-1"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              {resetting ? 'Resetting...' : 'Reset Now'}
            </Button>
          </div>
          
          <p className="text-xs text-gray-500 text-center">
            "Reset Now" will immediately reset the count to zero and start a new counting window.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}
