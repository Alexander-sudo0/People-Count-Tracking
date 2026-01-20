import AddCameraForm from './AddCameraForm'
import CameraList from './CameraList'

interface CameraData {
  id: string
  name: string
  source: string
  counting_enabled: boolean
  process_fps: number
  camera_type?: string
}

interface SidebarProps {
  cameras: CameraData[]
  selectedCamera: string | null
  onSelectCamera: (id: string) => void
  onCameraAdded: () => void
  onCameraDeleted: () => void
}

export default function Sidebar({
  cameras,
  selectedCamera,
  onSelectCamera,
  onCameraAdded,
  onCameraDeleted,
}: SidebarProps) {
  return (
    <div className="flex flex-col h-full">
      <AddCameraForm onCameraAdded={onCameraAdded} />
      <CameraList
        cameras={cameras}
        selectedCamera={selectedCamera}
        onSelectCamera={onSelectCamera}
        onCameraDeleted={onCameraDeleted}
      />
    </div>
  )
}
