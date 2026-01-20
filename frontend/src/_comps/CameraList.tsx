import { Video, Trash2, LogIn, LogOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import axios from 'axios'

interface CameraData {
  id: string
  name: string
  source: string
  counting_enabled: boolean
  process_fps: number
  camera_type?: string
}

interface CameraListProps {
  cameras: CameraData[]
  selectedCamera: string | null
  onSelectCamera: (id: string) => void
  onCameraDeleted: () => void
}

export default function CameraList({ 
  cameras, 
  selectedCamera, 
  onSelectCamera, 
  onCameraDeleted 
}: CameraListProps) {
  const deleteCamera = async (cameraId: string) => {
    try {
      await axios.delete(`/api/camera/${cameraId}`)
      onCameraDeleted()
    } catch (error) {
      console.error('Error deleting camera:', error)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <h3 className="text-sm font-semibold text-gray-400 mb-3">CAMERAS ({cameras.length})</h3>
      <div className="space-y-2">
        {cameras.map((cam) => (
          <div
            key={cam.id}
            className={`p-3 rounded-lg cursor-pointer transition-all ${
              selectedCamera === cam.id
                ? 'bg-blue-600/20 border border-blue-500'
                : 'bg-gray-800 border border-gray-700 hover:bg-gray-750'
            }`}
            onClick={() => onSelectCamera(cam.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <Video className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <span className="font-medium truncate">{cam.name}</span>
                {/* Camera Type Badge */}
                {cam.camera_type === 'entry' && (
                  <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/50">
                    <LogIn className="w-2.5 h-2.5" />
                    ENTRY
                  </span>
                )}
                {cam.camera_type === 'exit' && (
                  <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/50">
                    <LogOut className="w-2.5 h-2.5" />
                    EXIT
                  </span>
                )}
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteCamera(cam.id)
                }}
                className="h-6 w-6 p-0 flex-shrink-0"
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
            <div className="text-xs text-gray-500 mt-1 truncate">{cam.source}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
