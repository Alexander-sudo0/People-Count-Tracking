import { useState, useEffect } from 'react'
import axios from 'axios'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import Header from '@/_comps/Header'
import Footer from '@/_comps/Footer'
import Sidebar from '@/_comps/Sidebar'
import VideoFeed from '@/_comps/VideoFeed'
import StatsPanel from '@/_comps/StatsPanel'
import EmptyState from '@/_comps/EmptyState'
import DetectionLog from '@/_comps/DetectionLog'
import WatchlistAlerts from '@/_comps/WatchlistAlerts'

interface CameraData {
  id: string
  name: string
  source: string
  counting_enabled: boolean
  process_fps: number
  camera_type?: string
}

interface StatsData {
  unique_count: number
  entry_count: number
  exit_count: number
  current_count: number
}

export default function PeopleCounting() {
  const [cameras, setCameras] = useState<CameraData[]>([])
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [stats, setStats] = useState<StatsData>({
    unique_count: 0,
    entry_count: 0,
    exit_count: 0,
    current_count: 0,
  })

  // Fetch cameras
  useEffect(() => {
    fetchCameras()
  }, [])

  // Fetch stats for selected camera
  useEffect(() => {
    if (!selectedCamera) return

    const interval = setInterval(() => {
      axios
        .get(`/api/stats/${selectedCamera}`)
        .then((res) => setStats(res.data))
        .catch(console.error)
    }, 2000)

    return () => clearInterval(interval)
  }, [selectedCamera])

  const fetchCameras = async () => {
    try {
      const res = await axios.get('/api/cameras')
      setCameras(res.data.cameras || [])
      if (res.data.cameras && res.data.cameras.length > 0 && !selectedCamera) {
        setSelectedCamera(res.data.cameras[0].id)
      }
    } catch (error) {
      console.error('Error fetching cameras:', error)
      setCameras([])
    }
  }

  const handleCameraDeleted = () => {
    if (selectedCamera) {
      const deletedCamera = cameras.find((c) => c.id === selectedCamera)
      if (deletedCamera) {
        setSelectedCamera(null)
      }
    }
    fetchCameras()
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-white">
      <Header />

      <div className="flex-1 flex overflow-hidden relative">
        {/* Collapsible Sidebar */}
        <div
          className={`${
            sidebarOpen ? 'w-72' : 'w-0'
          } transition-all duration-300 ease-in-out overflow-hidden flex-shrink-0`}
        >
          <aside className="w-72 h-full border-r border-gray-700 bg-gray-900/50 flex flex-col">
            <Sidebar
              cameras={cameras}
              selectedCamera={selectedCamera}
              onSelectCamera={setSelectedCamera}
              onCameraAdded={fetchCameras}
              onCameraDeleted={handleCameraDeleted}
            />
          </aside>
        </div>

        {/* Toggle Button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className={`absolute top-1/2 -translate-y-1/2 z-20 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-r-lg p-1.5 transition-all duration-300 ${
            sidebarOpen ? 'left-72' : 'left-0'
          }`}
          title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          {sidebarOpen ? (
            <ChevronLeft className="w-4 h-4 text-gray-300" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-300" />
          )}
        </button>

        {/* Main Content */}
        <main className="flex-1 flex flex-col bg-gray-900 overflow-y-auto">
          {selectedCamera ? (
            <>
              {/* Side-by-side layout when sidebar is closed */}
              {!sidebarOpen ? (
                <div className="flex-1 flex flex-col lg:flex-row p-4 gap-4">
                  {/* Left: Video Feed */}
                  <div className="lg:w-1/2 flex-shrink-0">
                    <div className="h-full min-h-[400px] flex items-center justify-center">
                      <VideoFeed cameraId={selectedCamera} />
                    </div>
                  </div>
                  
                  {/* Right: Stats + Logs */}
                  <div className="lg:w-1/2 flex flex-col gap-4 overflow-y-auto">
                    <StatsPanel 
                      stats={stats} 
                      cameraId={selectedCamera}
                      cameraType={cameras.find(c => c.id === selectedCamera)?.camera_type}
                    />
                    <WatchlistAlerts cameraId={selectedCamera} />
                    <DetectionLog cameraId={selectedCamera} />
                  </div>
                </div>
              ) : (
                /* Standard stacked layout when sidebar is open */
                <>
                  {/* Video Feed Section */}
                  <div className="flex-1 p-6 flex items-center justify-center relative min-h-[400px]">
                    <VideoFeed cameraId={selectedCamera} />
                  </div>
                  
                  {/* Stats Panel */}
                  <StatsPanel 
                    stats={stats} 
                    cameraId={selectedCamera}
                    cameraType={cameras.find(c => c.id === selectedCamera)?.camera_type}
                  />
                  
                  {/* Detection Section */}
                  <div className="px-6 pb-6 space-y-4">
                    <WatchlistAlerts cameraId={selectedCamera} />
                    <DetectionLog cameraId={selectedCamera} />
                  </div>
                </>
              )}
            </>
          ) : (
            <EmptyState />
          )}
        </main>
      </div>

      <Footer />
    </div>
  )
}
