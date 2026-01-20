import { useState } from 'react'
import { Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import axios from 'axios'

interface AddCameraFormProps {
  onCameraAdded: () => void
}

export default function AddCameraForm({ onCameraAdded }: AddCameraFormProps) {
  const [newCamera, setNewCamera] = useState({
    name: '',
    source: '',
    process_fps: 15,
    camera_type: 'entry',
    watchlist_threshold: 40,
  })

  const addCamera = async () => {
    if (!newCamera.name || !newCamera.source) return

    try {
      const formData = new FormData()
      formData.append('name', newCamera.name)
      formData.append('source', newCamera.source)
      formData.append('process_fps', newCamera.process_fps.toString())
      formData.append('counting_enabled', 'true')
      formData.append('time_window_hours', '24')
      formData.append('camera_type', newCamera.camera_type)
      formData.append('watchlist_threshold', (newCamera.watchlist_threshold / 100).toString())

      await axios.post('/api/add_camera', formData)
      setNewCamera({ name: '', source: '', process_fps: 5, camera_type: 'entry', watchlist_threshold: 40 })
      onCameraAdded()
    } catch (error) {
      console.error('Error adding camera:', error)
    }
  }

  return (
    <div className="p-4 border-b border-gray-700">
      <h3 className="text-sm font-semibold text-gray-400 mb-3">ADD CAMERA</h3>
      <div className="space-y-2">
        <Input
          placeholder="Camera Name"
          value={newCamera.name}
          onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
          className="bg-gray-800 border-gray-700"
        />
        <div>
          <Input
            placeholder="RTSP URL or Device ID"
            value={newCamera.source}
            onChange={(e) => setNewCamera({ ...newCamera, source: e.target.value })}
            className="bg-gray-800 border-gray-700"
          />
          <p className="text-xs text-gray-500 mt-1">
            Examples: rtsp://192.168.1.10:554/stream or 0 for webcam
          </p>
        </div>
        <Input
          type="number"
          placeholder="FPS"
          value={newCamera.process_fps}
          onChange={(e) => setNewCamera({ ...newCamera, process_fps: parseInt(e.target.value) })}
          className="bg-gray-800 border-gray-700"
        />
        <Select
          value={newCamera.camera_type}
          onValueChange={(value) => setNewCamera({ ...newCamera, camera_type: value })}
        >
          <SelectTrigger className="bg-gray-800 border-gray-700">
            <SelectValue placeholder="Camera Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="entry">Entry Camera</SelectItem>
            <SelectItem value="exit">Exit Camera</SelectItem>
          </SelectContent>
        </Select>
        <div className="space-y-1">
          <label className="text-xs text-gray-400">
            Watchlist Match Threshold: {newCamera.watchlist_threshold}%
          </label>
          <input
            type="range"
            value={newCamera.watchlist_threshold}
            onChange={(e) => setNewCamera({ ...newCamera, watchlist_threshold: parseInt(e.target.value) })}
            min={20}
            max={90}
            step={5}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <p className="text-xs text-gray-500">
            Lower = more matches (may have false positives), Higher = stricter matching
          </p>
        </div>
        <Button onClick={addCamera} className="w-full" size="sm">
          <Plus className="w-4 h-4 mr-1" />
          Add Camera
        </Button>
      </div>
    </div>
  )
}
