import { useState, useEffect } from 'react'
import { 
  Users as UsersIcon, Clock, Camera, Image, 
  Maximize2, UserPlus, AlertTriangle, ArrowLeft, Radio,
  Eye, EyeOff, Star, Grid, List, User, Search, RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import axios from 'axios'
import Header from '@/_comps/Header'
import Footer from '@/_comps/Footer'

interface PersonData {
  person_id: string
  name: string | null
  first_seen: string
  last_seen: string
  total_detections: number
  confidence: number
  thumbnail_url: string | null
  is_live: boolean
  current_camera: string | null
  cameras_visited: number
  in_watchlist: boolean
}

interface CameraVisit {
  camera_id: string
  camera_name: string
  detection_count: number
  first_detection: string
  last_detection: string
  is_active: boolean
}

interface Detection {
  id: number
  camera_id: string
  camera_name: string
  detected_at: string
  confidence: number
  event_type: string
  in_roi: boolean
  thumbnail_url: string | null
  fullframe_url: string | null
}

interface PersonDetails {
  person_id: string
  is_live: boolean
  current_camera_id: string | null
  current_camera_name: string | null
  last_seen: string
  first_seen: string
  total_detections: number
  thumbnail_url: string | null
  cameras: CameraVisit[]
  recent_detections: Detection[]
  watchlist: {
    in_watchlist: boolean
    name: string | null
    category: string | null
  }
}

type ViewMode = 'people' | 'person-details' | 'camera-detections'
type DisplayMode = 'grid' | 'list'

export default function PeopleTracking() {
  const [viewMode, setViewMode] = useState<ViewMode>('people')
  const [displayMode, setDisplayMode] = useState<DisplayMode>('grid')
  const [people, setPeople] = useState<PersonData[]>([])
  const [selectedPerson, setSelectedPerson] = useState<PersonData | null>(null)
  const [personDetails, setPersonDetails] = useState<PersonDetails | null>(null)
  const [selectedCamera, setSelectedCamera] = useState<CameraVisit | null>(null)
  const [cameraDetections, setCameraDetections] = useState<Detection[]>([])
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null)
  const [showFullframe, setShowFullframe] = useState(false)
  const [showAddToWatchlist, setShowAddToWatchlist] = useState(false)
  const [addWatchlistName, setAddWatchlistName] = useState('')
  const [addWatchlistNotes, setAddWatchlistNotes] = useState('')
  const [addWatchlistCategory, setAddWatchlistCategory] = useState('general')
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Fetch all people on mount
  useEffect(() => {
    fetchPeople()
    const interval = setInterval(fetchPeople, 5000)
    return () => clearInterval(interval)
  }, [])

  // Fetch person details when selected
  useEffect(() => {
    if (selectedPerson && viewMode === 'person-details') {
      fetchPersonDetails(selectedPerson.person_id)
      const interval = setInterval(() => fetchPersonDetails(selectedPerson.person_id), 3000)
      return () => clearInterval(interval)
    }
  }, [selectedPerson, viewMode])

  const fetchPeople = async () => {
    try {
      const res = await axios.get('/api/tracking/people')
      setPeople(res.data.people || [])
    } catch (error) {
      console.error('Error fetching people:', error)
    }
  }

  const fetchPersonDetails = async (personId: string) => {
    setLoading(true)
    try {
      const res = await axios.get(`/api/tracking/person/${personId}/details`)
      setPersonDetails(res.data)
    } catch (error) {
      console.error('Error fetching person details:', error)
    } finally {
      setLoading(false)
    }
  }

  const handlePersonClick = (person: PersonData) => {
    setSelectedPerson(person)
    setViewMode('person-details')
  }

  const handleCameraClick = (camera: CameraVisit) => {
    if (!personDetails) return
    setSelectedCamera(camera)
    // Filter detections for this camera
    const camDetections = personDetails.recent_detections.filter(
      d => d.camera_id === camera.camera_id || d.camera_name === camera.camera_name
    )
    setCameraDetections(camDetections)
    setViewMode('camera-detections')
  }

  const handleBackClick = () => {
    if (viewMode === 'camera-detections') {
      setViewMode('person-details')
      setSelectedCamera(null)
      setCameraDetections([])
    } else if (viewMode === 'person-details') {
      setViewMode('people')
      setSelectedPerson(null)
      setPersonDetails(null)
    }
  }

  const addToWatchlist = async () => {
    if (!selectedPerson) return
    
    try {
      const formData = new FormData()
      formData.append('person_id', selectedPerson.person_id)
      formData.append('name', addWatchlistName || selectedPerson.person_id)
      if (addWatchlistNotes) formData.append('notes', addWatchlistNotes)
      formData.append('category', addWatchlistCategory)
      
      await axios.post('/api/watchlist/add-from-tracking', formData)
      
      setShowAddToWatchlist(false)
      setAddWatchlistName('')
      setAddWatchlistNotes('')
      setAddWatchlistCategory('general')
      
      // Refresh person details
      if (selectedPerson) {
        fetchPersonDetails(selectedPerson.person_id)
      }
    } catch (error) {
      console.error('Error adding to watchlist:', error)
      alert('Failed to add to watchlist')
    }
  }

  const formatTime = (timestamp: string) => {
    if (!timestamp) return 'Unknown'
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffSecs = Math.floor(diffMs / 1000)
    
    if (diffSecs < 30) return 'Just now'
    if (diffSecs < 60) return `${diffSecs}s ago`
    const diffMins = Math.floor(diffSecs / 60)
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return `${Math.floor(diffHours / 24)}d ago`
  }

  const filteredPeople = people.filter(p => 
    p.person_id.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const livePeople = people.filter(p => p.is_live)
  const totalDetections = people.reduce((sum, p) => sum + p.total_detections, 0)

  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-white">
      <Header />

      <main className="flex-1 container mx-auto px-6 py-6">
        {/* Breadcrumb / Navigation */}
        <div className="flex items-center gap-4 mb-6">
          {viewMode !== 'people' && (
            <Button variant="ghost" size="sm" onClick={handleBackClick}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          )}
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <UsersIcon className="w-4 h-4" />
            <span 
              className={viewMode === 'people' ? 'text-purple-400 font-semibold' : 'cursor-pointer hover:text-white'}
              onClick={() => { setViewMode('people'); setSelectedPerson(null); setSelectedCamera(null); }}
            >
              All People
            </span>
            {selectedPerson && (
              <>
                <span>/</span>
                <span 
                  className={viewMode === 'person-details' ? 'text-purple-400 font-semibold' : 'cursor-pointer hover:text-white'}
                  onClick={() => { setViewMode('person-details'); setSelectedCamera(null); }}
                >
                  Person {selectedPerson.person_id.substring(0, 8)}
                </span>
              </>
            )}
            {selectedCamera && viewMode === 'camera-detections' && (
              <>
                <span>/</span>
                <span className="text-purple-400 font-semibold">
                  {selectedCamera.camera_name}
                </span>
              </>
            )}
          </div>
        </div>

        {/* VIEW: All People */}
        {viewMode === 'people' && (
          <div className="space-y-6">
            {/* Header with Search and View Toggle */}
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h1 className="text-2xl font-bold flex items-center gap-3">
                  <UsersIcon className="w-7 h-7 text-purple-400" />
                  People Tracking
                </h1>
                <p className="text-gray-400">Track and monitor all detected individuals</p>
              </div>
              
              <div className="flex items-center gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
                  <Input
                    placeholder="Search by ID..."
                    value={searchQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
                    className="pl-10 w-64 bg-gray-800 border-gray-700"
                  />
                </div>
                
                {/* View Toggle */}
                <div className="flex bg-gray-800 rounded-lg p-1">
                  <button
                    onClick={() => setDisplayMode('grid')}
                    className={`p-2 rounded ${displayMode === 'grid' ? 'bg-purple-600' : 'hover:bg-gray-700'}`}
                  >
                    <Grid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setDisplayMode('list')}
                    className={`p-2 rounded ${displayMode === 'list' ? 'bg-purple-600' : 'hover:bg-gray-700'}`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>

                {/* Refresh */}
                <Button variant="ghost" size="sm" onClick={fetchPeople}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-3xl font-bold text-purple-400">{people.length}</div>
                <div className="text-sm text-gray-400">Total People</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-3xl font-bold text-green-400">{livePeople.length}</div>
                <div className="text-sm text-gray-400 flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  Currently Live
                </div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-3xl font-bold text-blue-400">{totalDetections}</div>
                <div className="text-sm text-gray-400">Total Detections</div>
              </div>
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="text-3xl font-bold text-yellow-400">
                  {people.reduce((sum, p) => sum + p.cameras_visited, 0)}
                </div>
                <div className="text-sm text-gray-400">Camera Visits</div>
              </div>
            </div>

            {/* People Grid/List */}
            {filteredPeople.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <UsersIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No people found</p>
                <p className="text-sm mt-1">People will appear here when detected</p>
              </div>
            ) : displayMode === 'grid' ? (
              /* Grid View */
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {filteredPeople.map((person) => (
                  <Card
                    key={person.person_id}
                    className="bg-gray-900/50 border-gray-700 cursor-pointer hover:border-purple-500 transition-all group overflow-hidden"
                    onClick={() => handlePersonClick(person)}
                  >
                    {/* Thumbnail */}
                    <div className="aspect-square bg-gray-800 relative">
                      {person.thumbnail_url ? (
                        <img
                          src={person.thumbnail_url}
                          alt={`Person ${person.person_id}`}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none'
                          }}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <User className="w-12 h-12 text-gray-600" />
                        </div>
                      )}
                      {/* Live indicator */}
                      {person.is_live && (
                        <div className="absolute top-2 left-2 bg-green-500 text-xs px-2 py-1 rounded text-white font-medium flex items-center gap-1">
                          <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                          LIVE
                        </div>
                      )}
                      {/* Overlay on hover */}
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                        <span className="text-xs bg-purple-600 px-2 py-1 rounded">
                          <Radio className="w-3 h-3 inline mr-1" />
                          View Details
                        </span>
                      </div>
                    </div>
                    <CardContent className="p-3">
                      <div className="font-mono text-xs text-purple-400 truncate mb-2">
                        {person.name ? (
                          <span className="text-green-400 font-semibold">{person.name}</span>
                        ) : (
                          <span>{person.person_id.substring(0, 12)}...</span>
                        )}
                        {person.in_watchlist && (
                          <span className="ml-1 text-yellow-500" title="In Watchlist">⭐</span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <div className="text-gray-500">Detections</div>
                          <div className="font-semibold">{person.total_detections}</div>
                        </div>
                        <div>
                          <div className="text-gray-500">Cameras</div>
                          <div className="font-semibold">{person.cameras_visited}</div>
                        </div>
                      </div>
                      {person.current_camera && (
                        <div className="mt-2 text-xs">
                          <div className="text-gray-500">Current Location</div>
                          <div className="text-blue-400 truncate">{person.current_camera}</div>
                        </div>
                      )}
                      <div className="mt-2 text-xs text-gray-500">
                        Last seen: {formatTime(person.last_seen)}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              /* List View */
              <div className="bg-gray-900/50 rounded-lg overflow-hidden">
                {/* Table Header */}
                <div className="grid grid-cols-7 gap-4 p-4 bg-gray-800/50 text-sm font-semibold text-gray-400">
                  <div className="col-span-2">Person</div>
                  <div>Status</div>
                  <div>Current Camera</div>
                  <div>Detections</div>
                  <div>Cameras</div>
                  <div>Last Seen</div>
                </div>
                {/* Table Body */}
                {filteredPeople.map((person) => (
                  <div
                    key={person.person_id}
                    className="grid grid-cols-7 gap-4 p-4 border-t border-gray-800 hover:bg-gray-800/30 cursor-pointer transition-colors"
                    onClick={() => handlePersonClick(person)}
                  >
                    {/* Person Info */}
                    <div className="col-span-2 flex items-center gap-3">
                      <div className="w-10 h-10 bg-gray-800 rounded-lg overflow-hidden flex-shrink-0">
                        {person.thumbnail_url ? (
                          <img
                            src={person.thumbnail_url}
                            alt="Person"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <User className="w-5 h-5 text-gray-600" />
                          </div>
                        )}
                      </div>
                      <div className="font-mono text-sm text-purple-400 truncate">
                        {person.name ? (
                          <span className="text-green-400 font-semibold">{person.name}</span>
                        ) : (
                          <span>{person.person_id.substring(0, 16)}...</span>
                        )}
                        {person.in_watchlist && (
                          <span className="ml-1 text-yellow-500" title="In Watchlist">⭐</span>
                        )}
                      </div>
                    </div>
                    
                    {/* Status */}
                    <div className="flex items-center">
                      {person.is_live ? (
                        <span className="flex items-center gap-1 text-green-400 text-sm">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                          Live
                        </span>
                      ) : (
                        <span className="text-gray-500 text-sm">Offline</span>
                      )}
                    </div>
                    
                    {/* Current Camera */}
                    <div className="flex items-center text-sm text-blue-400 truncate">
                      {person.current_camera || '-'}
                    </div>
                    
                    {/* Detections */}
                    <div className="flex items-center text-sm">{person.total_detections}</div>
                    
                    {/* Cameras */}
                    <div className="flex items-center text-sm">{person.cameras_visited}</div>
                    
                    {/* Last Seen */}
                    <div className="flex items-center text-sm text-gray-400">
                      {formatTime(person.last_seen)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* VIEW: Person Details with Cameras */}
        {viewMode === 'person-details' && selectedPerson && (
          <div className="space-y-6 max-w-6xl mx-auto">
            {/* Loading State */}
            {loading && !personDetails && (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
              </div>
            )}
            
            {/* Live Status Banner */}
            {personDetails && (
              <div className={`p-4 rounded-lg border ${
                personDetails.is_live 
                  ? 'bg-green-600/10 border-green-500/30' 
                  : 'bg-gray-800/50 border-gray-700'
              }`}>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-4">
                    {personDetails.is_live ? (
                      <>
                        <div className="w-10 h-10 bg-green-500/20 rounded-full flex items-center justify-center">
                          <Eye className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                            <span className="text-green-400 font-semibold">Currently Visible</span>
                          </div>
                          <div className="text-sm text-gray-400">
                            Live at <span className="text-white">{personDetails.current_camera_name}</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
                          <EyeOff className="w-5 h-5 text-gray-400" />
                        </div>
                        <div>
                          <div className="text-gray-400 font-medium">Not Currently Visible</div>
                          <div className="text-sm text-gray-500">
                            Last seen at <span className="text-gray-300">{personDetails.current_camera_name}</span>
                            {' · '}{formatTime(personDetails.last_seen)}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                  
                  {/* Watchlist Status */}
                  {personDetails.watchlist?.in_watchlist ? (
                    <div className="flex items-center gap-2 bg-yellow-600/20 px-3 py-2 rounded-lg">
                      <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                      <div>
                        <div className="text-yellow-400 font-medium text-sm">
                          {personDetails.watchlist.name}
                        </div>
                        <div className="text-xs text-yellow-600">
                          {personDetails.watchlist.category}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Button 
                      onClick={() => setShowAddToWatchlist(true)}
                      className="bg-yellow-600 hover:bg-yellow-700"
                    >
                      <UserPlus className="w-4 h-4 mr-2" />
                      Add to Watchlist
                    </Button>
                  )}
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left: Person Info */}
              <Card className="bg-gray-900/50 border-gray-700">
                <CardContent className="p-6">
                  {/* Large Thumbnail */}
                  <div className="aspect-square bg-gray-800 rounded-xl overflow-hidden mb-4">
                    {personDetails?.thumbnail_url ? (
                      <img
                        src={personDetails.thumbnail_url}
                        alt="Person"
                        className="w-full h-full object-cover"
                      />
                    ) : selectedPerson.thumbnail_url ? (
                      <img
                        src={selectedPerson.thumbnail_url}
                        alt="Person"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <User className="w-16 h-16 text-gray-600" />
                      </div>
                    )}
                  </div>

                  {/* Person ID */}
                  <div className="text-center mb-4">
                    <div className="font-mono text-sm text-purple-400 mb-1">
                      {selectedPerson.person_id}
                    </div>
                    {personDetails?.watchlist?.in_watchlist && (
                      <div className="text-lg font-semibold text-yellow-400">
                        {personDetails.watchlist.name}
                      </div>
                    )}
                  </div>

                  {/* Stats */}
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Total Detections</span>
                      <span className="font-semibold">{personDetails?.total_detections || selectedPerson.total_detections}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Cameras Visited</span>
                      <span className="font-semibold">{personDetails?.cameras?.length || selectedPerson.cameras_visited}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">First Seen</span>
                      <span>{personDetails?.first_seen ? new Date(personDetails.first_seen).toLocaleString() : '-'}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Last Seen</span>
                      <span className="text-green-400">{formatTime(personDetails?.last_seen || selectedPerson.last_seen)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Right: Cameras and Detections */}
              <div className="lg:col-span-2 space-y-6">
                {/* Cameras Section */}
                <div>
                  <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                    <Camera className="w-5 h-5 text-blue-400" />
                    Cameras Visited
                  </h2>
                  
                  {personDetails?.cameras && personDetails.cameras.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {personDetails.cameras.map((camera) => (
                        <Card
                          key={camera.camera_id}
                          className="bg-gray-900/50 border-gray-700 cursor-pointer hover:border-blue-500 transition-all group"
                          onClick={() => handleCameraClick(camera)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-blue-600/20 rounded-lg flex items-center justify-center">
                                  <Camera className="w-5 h-5 text-blue-400" />
                                </div>
                                <div>
                                  <h3 className="font-semibold">{camera.camera_name}</h3>
                                  <p className="text-xs text-gray-500">ID: {camera.camera_id}</p>
                                </div>
                              </div>
                              {camera.is_active && (
                                <div className="flex items-center gap-1 text-green-400 text-xs">
                                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                  Active
                                </div>
                              )}
                            </div>
                            
                            <div className="grid grid-cols-2 gap-3 text-sm">
                              <div>
                                <div className="text-gray-500 text-xs">Detections</div>
                                <div className="font-semibold text-blue-400">{camera.detection_count}</div>
                              </div>
                              <div>
                                <div className="text-gray-500 text-xs">Last Seen</div>
                                <div className="font-semibold">{formatTime(camera.last_detection)}</div>
                              </div>
                            </div>
                            
                            <Button size="sm" variant="ghost" className="w-full mt-3 group-hover:bg-blue-600/20">
                              View Detections <ArrowLeft className="w-4 h-4 ml-1 rotate-180" />
                            </Button>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500 bg-gray-900/30 rounded-lg">
                      <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>No camera data available</p>
                    </div>
                  )}
                </div>

                {/* Recent Detections */}
                <div>
                  <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                    <Image className="w-5 h-5 text-purple-400" />
                    Recent Detections
                  </h2>

                  {personDetails?.recent_detections && personDetails.recent_detections.length > 0 ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {personDetails.recent_detections.slice(0, 6).map((detection, index) => (
                        <Card
                          key={detection.id || index}
                          className="bg-gray-900/50 border-gray-700 overflow-hidden cursor-pointer hover:border-purple-500 transition-all group"
                          onClick={() => {
                            setSelectedDetection(detection)
                            setShowFullframe(true)
                          }}
                        >
                          {/* Thumbnail */}
                          <div className="aspect-square bg-gray-800 relative">
                            {detection.thumbnail_url ? (
                              <img
                                src={detection.thumbnail_url}
                                alt={`Detection ${index + 1}`}
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center">
                                <User className="w-8 h-8 text-gray-600" />
                              </div>
                            )}
                            {/* Overlay on hover */}
                            <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                              <Maximize2 className="w-6 h-6 text-white" />
                            </div>
                            {/* ROI Badge */}
                            {detection.in_roi && (
                              <div className="absolute top-1 right-1 bg-green-500 text-xs px-1.5 py-0.5 rounded text-white font-medium">
                                ROI
                              </div>
                            )}
                          </div>
                          {/* Info */}
                          <CardContent className="p-2">
                            <div className="text-xs text-gray-400 truncate flex items-center gap-1">
                              <Camera className="w-3 h-3" />
                              {detection.camera_name}
                            </div>
                            <div className="text-xs text-gray-500 flex items-center gap-1 mt-1">
                              <Clock className="w-3 h-3" />
                              {formatTime(detection.detected_at)}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500 bg-gray-900/30 rounded-lg">
                      <Image className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>No detection history available</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW: Camera Detections for Person */}
        {viewMode === 'camera-detections' && selectedCamera && selectedPerson && (
          <div className="space-y-6 max-w-5xl mx-auto">
            {/* Camera Info Header */}
            <div className="bg-gray-800/50 rounded-lg p-6">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 bg-blue-600/20 rounded-xl flex items-center justify-center">
                  <Camera className="w-7 h-7 text-blue-400" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold">{selectedCamera.camera_name}</h1>
                  <p className="text-gray-400">
                    Detections for Person {selectedPerson.person_id.substring(0, 8)}
                  </p>
                </div>
                <div className="ml-auto flex items-center gap-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">{selectedCamera.detection_count}</div>
                    <div className="text-xs text-gray-500">Detections</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-semibold">{formatTime(selectedCamera.last_detection)}</div>
                    <div className="text-xs text-gray-500">Last Seen</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Detection Grid */}
            {cameraDetections.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {cameraDetections.map((detection, index) => (
                  <Card
                    key={detection.id || index}
                    className="bg-gray-900/50 border-gray-700 overflow-hidden cursor-pointer hover:border-purple-500 transition-all group"
                    onClick={() => {
                      setSelectedDetection(detection)
                      setShowFullframe(true)
                    }}
                  >
                    {/* Thumbnail */}
                    <div className="aspect-square bg-gray-800 relative">
                      {detection.thumbnail_url ? (
                        <img
                          src={detection.thumbnail_url}
                          alt={`Detection ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <User className="w-8 h-8 text-gray-600" />
                        </div>
                      )}
                      {/* Overlay on hover */}
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <Maximize2 className="w-6 h-6 text-white" />
                      </div>
                      {/* ROI Badge */}
                      {detection.in_roi && (
                        <div className="absolute top-1 right-1 bg-green-500 text-xs px-1.5 py-0.5 rounded text-white font-medium">
                          ROI
                        </div>
                      )}
                    </div>
                    {/* Info */}
                    <CardContent className="p-3">
                      <div className="text-sm font-medium mb-1">
                        {new Date(detection.detected_at).toLocaleString()}
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-500">
                          {(detection.confidence * 100).toFixed(0)}% confidence
                        </span>
                        {detection.event_type === 'first_detection' && (
                          <span className="bg-blue-500 px-1.5 py-0.5 rounded text-white">NEW</span>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500 bg-gray-900/30 rounded-lg">
                <Image className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No detections found for this camera</p>
              </div>
            )}
          </div>
        )}
      </main>

      <Footer />

      {/* Fullframe Modal */}
      <Dialog open={showFullframe} onOpenChange={setShowFullframe}>
        <DialogContent className="max-w-4xl bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Image className="w-5 h-5 text-purple-400" />
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
                ) : selectedDetection.thumbnail_url ? (
                  <img
                    src={selectedDetection.thumbnail_url}
                    alt="Thumbnail"
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
                  <div className="text-xs text-gray-500 mb-1">Camera</div>
                  <div className="text-sm">{selectedDetection.camera_name}</div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Detected At</div>
                  <div className="text-sm">
                    {new Date(selectedDetection.detected_at).toLocaleString()}
                  </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-lg">
                  <div className="text-xs text-gray-500 mb-1">Confidence</div>
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
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Add to Watchlist Modal */}
      <Dialog open={showAddToWatchlist} onOpenChange={setShowAddToWatchlist}>
        <DialogContent className="max-w-md bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-500" />
              Add to Watchlist
            </DialogTitle>
          </DialogHeader>
          {selectedPerson && (
            <div className="space-y-4">
              {/* Person preview */}
              <div className="flex items-center gap-4 p-3 bg-gray-800 rounded-lg">
                <div className="w-16 h-16 bg-gray-700 rounded-lg overflow-hidden">
                  {selectedPerson.thumbnail_url ? (
                    <img
                      src={selectedPerson.thumbnail_url}
                      alt="Person"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <User className="w-8 h-8 text-gray-600" />
                    </div>
                  )}
                </div>
                <div>
                  <div className="font-mono text-sm text-purple-400">
                    {selectedPerson.person_id.substring(0, 12)}...
                  </div>
                  <div className="text-xs text-gray-500">
                    {selectedPerson.total_detections} detections
                  </div>
                </div>
              </div>

              {/* Form */}
              <div className="space-y-3">
                <Input
                  placeholder="Name (optional)"
                  value={addWatchlistName}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAddWatchlistName(e.target.value)}
                  className="bg-gray-800 border-gray-700"
                />
                <Input
                  placeholder="Notes (optional)"
                  value={addWatchlistNotes}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAddWatchlistNotes(e.target.value)}
                  className="bg-gray-800 border-gray-700"
                />
                <select
                  value={addWatchlistCategory}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setAddWatchlistCategory(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
                >
                  <option value="general">General</option>
                  <option value="vip">VIP</option>
                  <option value="banned">Banned</option>
                  <option value="employee">Employee</option>
                </select>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={addToWatchlist}
                  className="flex-1 bg-yellow-600 hover:bg-yellow-700"
                >
                  <UserPlus className="w-4 h-4 mr-2" />
                  Add to Watchlist
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowAddToWatchlist(false)}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
