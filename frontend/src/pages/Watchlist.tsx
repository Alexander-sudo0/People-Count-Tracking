import { useState, useEffect, useRef } from 'react'
import {
  AlertTriangle,
  X,
  Plus,
  Eye,
  Bell,
  BellOff,
  Search,
  MapPin,
  Camera,
  Clock,
  Upload,
  User,
  History,
  Image as ImageIcon
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import Header from '@/_comps/Header'
import Footer from '@/_comps/Footer'
import axios from 'axios'

interface WatchlistPerson {
  person_id: string
  name: string | null
  photo_path: string | null
  thumbnail_path: string | null
  thumbnail_url?: string
  photo_url?: string
  added_at: string
  notes: string | null
  alert_enabled: boolean
  category: string
  last_location: string | null
  last_camera_id: string | null
  last_seen: string | null
  detection_count: number
}

interface Detection {
  camera_id: string
  camera_name: string
  detected_at: string
  confidence: number
  thumbnail_path: string | null
  thumbnail_url?: string
  fullframe_path: string | null
  fullframe_url?: string
}

interface LocationHistory {
  camera_id: string
  camera_name: string
  detected_at: string
  thumbnail_path: string | null
  thumbnail_url?: string
}

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<WatchlistPerson[]>([])
  const [filteredList, setFilteredList] = useState<WatchlistPerson[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedPerson, setSelectedPerson] = useState<WatchlistPerson | null>(null)
  const [personDetections, setPersonDetections] = useState<Detection[]>([])
  const [locationHistory, setLocationHistory] = useState<LocationHistory[]>([])
  const [showAddForm, setShowAddForm] = useState(false)
  const [showPersonModal, setShowPersonModal] = useState(false)
  const [selectedFullframe, setSelectedFullframe] = useState<string | null>(null)
  const [categories, setCategories] = useState<string[]>(['general', 'vip', 'banned', 'employee'])
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const [newPerson, setNewPerson] = useState({
    person_id: '',
    name: '',
    notes: '',
    category: 'general',
    alert_enabled: true,
    photo: null as File | null,
    photoPreview: '' as string
  })

  const fetchWatchlist = async () => {
    try {
      const res = await axios.get('/api/watchlist')
      setWatchlist(res.data.watchlist || [])
    } catch (error) {
      console.error('Error fetching watchlist:', error)
    }
  }

  const fetchCategories = async () => {
    try {
      const res = await axios.get('/api/watchlist/categories')
      const cats = res.data.categories || []
      setCategories(['all', 'general', 'vip', 'banned', 'employee', ...cats.filter((c: string) => !['general', 'vip', 'banned', 'employee'].includes(c))])
    } catch (error) {
      console.error('Error fetching categories:', error)
    }
  }

  useEffect(() => {
    fetchWatchlist()
    fetchCategories()
    const interval = setInterval(fetchWatchlist, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      setFilteredList(
        watchlist.filter(
          (p) =>
            p.name?.toLowerCase().includes(query) ||
            p.person_id.toLowerCase().includes(query) ||
            p.last_location?.toLowerCase().includes(query)
        )
      )
    } else if (selectedCategory) {
      setFilteredList(watchlist.filter((p) => p.category === selectedCategory))
    } else {
      setFilteredList(watchlist)
    }
  }, [searchQuery, watchlist, selectedCategory])

  const fetchPersonDetails = async (person: WatchlistPerson) => {
    setSelectedPerson(person)
    setShowPersonModal(true)
    
    try {
      // Fetch detection history
      const detRes = await axios.get(`/api/watchlist/${person.person_id}/detections`)
      setPersonDetections(detRes.data.detections || [])
      
      // Fetch location history
      const locRes = await axios.get(`/api/watchlist/${person.person_id}/location-history`)
      setLocationHistory(locRes.data.history || [])
    } catch (error) {
      console.error('Error fetching person details:', error)
    }
  }

  const handlePhotoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setNewPerson({ 
        ...newPerson, 
        photo: file,
        photoPreview: URL.createObjectURL(file)
      })
    }
  }

  const addToWatchlist = async () => {
    if (!newPerson.person_id && !newPerson.name) return

    try {
      const personId = newPerson.person_id || `person_${Date.now()}`
      const formData = new FormData()
      formData.append('person_id', personId)
      if (newPerson.name) formData.append('name', newPerson.name)
      if (newPerson.notes) formData.append('notes', newPerson.notes)
      formData.append('category', newPerson.category)
      formData.append('alert_enabled', newPerson.alert_enabled ? 'true' : 'false')

      await axios.post('/api/watchlist/add', formData)
      
      // Upload photo if provided
      if (newPerson.photo) {
        const photoFormData = new FormData()
        photoFormData.append('photo', newPerson.photo)
        await axios.post(`/api/watchlist/${personId}/photo`, photoFormData)
      }
      
      setNewPerson({ 
        person_id: '', 
        name: '', 
        notes: '', 
        category: 'general',
        alert_enabled: true, 
        photo: null,
        photoPreview: ''
      })
      setShowAddForm(false)
      fetchWatchlist()
    } catch (error) {
      console.error('Error adding to watchlist:', error)
    }
  }

  const uploadPhoto = async (personId: string, file: File) => {
    try {
      const formData = new FormData()
      formData.append('photo', file)
      await axios.post(`/api/watchlist/${personId}/photo`, formData)
      fetchWatchlist()
    } catch (error) {
      console.error('Error uploading photo:', error)
    }
  }

  const removeFromWatchlist = async (personId: string) => {
    if (!confirm('Remove this person from watchlist?')) return
    try {
      await axios.delete(`/api/watchlist/${personId}`)
      setShowPersonModal(false)
      fetchWatchlist()
    } catch (error) {
      console.error('Error removing from watchlist:', error)
    }
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    return `${Math.floor(diffHours / 24)}d ago`
  }

  const formatDateTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      vip: 'bg-green-600',
      banned: 'bg-red-600',
      employee: 'bg-blue-600',
      general: 'bg-gray-600'
    }
    return colors[category] || 'bg-gray-600'
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header currentPage="watchlist" />
      
      <main className="flex-1 p-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <AlertTriangle className="w-8 h-8 text-yellow-500" />
            Watchlist Management
          </h1>
          <Button
            onClick={() => setShowAddForm(!showAddForm)}
            className="bg-yellow-600 hover:bg-yellow-700"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Person
          </Button>
        </div>

        {/* Search and Filter Bar */}
        <div className="flex gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search by name, ID, or location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-800 border-gray-700"
            />
          </div>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
          >
            <option value="">All Categories</option>
            {categories.filter(c => c !== 'all').map((cat) => (
              <option key={cat} value={cat}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* Add Person Form */}
        {showAddForm && (
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <CardTitle className="text-lg">Add New Person to Watchlist</CardTitle>
            </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-4">
                <Input
                  placeholder="Person ID (auto-generated if empty)"
                  value={newPerson.person_id}
                  onChange={(e) => setNewPerson({ ...newPerson, person_id: e.target.value })}
                  className="bg-gray-800 border-gray-700"
                />
                <Input
                  placeholder="Name"
                  value={newPerson.name}
                  onChange={(e) => setNewPerson({ ...newPerson, name: e.target.value })}
                  className="bg-gray-800 border-gray-700"
                />
                <Input
                  placeholder="Notes"
                  value={newPerson.notes}
                  onChange={(e) => setNewPerson({ ...newPerson, notes: e.target.value })}
                  className="bg-gray-800 border-gray-700"
                />
                <select
                  value={newPerson.category}
                  onChange={(e) => setNewPerson({ ...newPerson, category: e.target.value })}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
                >
                  <option value="general">General</option>
                  <option value="vip">VIP</option>
                  <option value="banned">Banned</option>
                  <option value="employee">Employee</option>
                </select>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="add-alert-enabled"
                    checked={newPerson.alert_enabled}
                    onChange={(e) => setNewPerson({ ...newPerson, alert_enabled: e.target.checked })}
                    className="rounded"
                  />
                  <label htmlFor="add-alert-enabled" className="text-sm text-gray-300">
                    Enable alerts when detected
                  </label>
                </div>
              </div>
              
              {/* Photo Upload */}
              <div className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-600 rounded-lg">
                {newPerson.photoPreview ? (
                  <div className="relative">
                    <img 
                      src={newPerson.photoPreview} 
                      alt="Preview" 
                      className="w-32 h-32 object-cover rounded-lg"
                    />
                    <button
                      onClick={() => setNewPerson({ ...newPerson, photo: null, photoPreview: '' })}
                      className="absolute -top-2 -right-2 bg-red-600 rounded-full p-1"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <>
                    <Upload className="w-12 h-12 text-gray-500 mb-2" />
                    <p className="text-gray-400 text-sm mb-2">Upload Photo</p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handlePhotoSelect}
                      className="hidden"
                      ref={fileInputRef}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Select Image
                    </Button>
                  </>
                )}
              </div>
            </div>
            
            <div className="flex gap-2">
              <Button onClick={addToWatchlist} className="flex-1 bg-yellow-600 hover:bg-yellow-700">
                Add to Watchlist
              </Button>
              <Button onClick={() => setShowAddForm(false)} variant="outline">
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Watchlist Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filteredList.length === 0 && (
          <div className="col-span-full text-center py-12 text-gray-500">
            <AlertTriangle className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg">No persons in watchlist</p>
            <p className="text-sm">Add persons to track their detections</p>
          </div>
        )}

        {filteredList.map((person) => (
          <Card
            key={person.person_id}
            className="bg-gray-800/50 border-gray-700 hover:border-yellow-600 transition-all cursor-pointer"
            onClick={() => fetchPersonDetails(person)}
          >
            <CardContent className="p-4">
              <div className="flex gap-3">
                {/* Photo */}
                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0">
                  {person.thumbnail_url ? (
                    <img
                      src={person.thumbnail_url}
                      alt={person.name || person.person_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <User className="w-8 h-8 text-gray-500" />
                  )}
                </div>
                
                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-white truncate">
                      {person.name || person.person_id}
                    </span>
                    {person.alert_enabled ? (
                      <Bell className="w-3 h-3 text-yellow-500 flex-shrink-0" />
                    ) : (
                      <BellOff className="w-3 h-3 text-gray-500 flex-shrink-0" />
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`text-xs px-2 py-0.5 rounded ${getCategoryColor(person.category)}`}>
                      {person.category}
                    </span>
                  </div>
                  
                  <div className="text-xs text-gray-400 space-y-1">
                    <div className="flex items-center gap-1">
                      <Eye className="w-3 h-3" />
                      {person.detection_count} detections
                    </div>
                    {person.last_location && (
                      <div className="flex items-center gap-1 truncate">
                        <MapPin className="w-3 h-3 flex-shrink-0" />
                        {person.last_location}
                      </div>
                    )}
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatTime(person.last_seen)}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Person Detail Modal */}
      <Dialog open={showPersonModal} onOpenChange={setShowPersonModal}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-yellow-500" />
              {selectedPerson?.name || selectedPerson?.person_id}
            </DialogTitle>
          </DialogHeader>

          {selectedPerson && (
            <div className="space-y-6">
              {/* Person Info */}
              <div className="flex gap-6">
                <div className="w-32 h-32 bg-gray-800 rounded-lg flex items-center justify-center overflow-hidden">
                  {selectedPerson.thumbnail_url || selectedPerson.photo_url ? (
                    <img
                      src={selectedPerson.photo_url || selectedPerson.thumbnail_url}
                      alt={selectedPerson.name || selectedPerson.person_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <User className="w-16 h-16 text-gray-500" />
                  )}
                </div>
                
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className={`px-3 py-1 rounded ${getCategoryColor(selectedPerson.category)}`}>
                      {selectedPerson.category.toUpperCase()}
                    </span>
                    {selectedPerson.alert_enabled && (
                      <span className="flex items-center gap-1 text-yellow-500 text-sm">
                        <Bell className="w-4 h-4" /> Alerts On
                      </span>
                    )}
                  </div>
                  
                  <div className="text-gray-400 text-sm">
                    <p>ID: {selectedPerson.person_id}</p>
                    <p>Added: {formatDateTime(selectedPerson.added_at)}</p>
                    {selectedPerson.notes && <p>Notes: {selectedPerson.notes}</p>}
                  </div>
                  
                  {selectedPerson.last_location && (
                    <div className="flex items-center gap-2 text-green-400">
                      <MapPin className="w-4 h-4" />
                      <span>Last seen at: {selectedPerson.last_location}</span>
                      <span className="text-gray-500">({formatTime(selectedPerson.last_seen)})</span>
                    </div>
                  )}
                  
                  {/* Upload Photo Button */}
                  <div className="flex gap-2 pt-2">
                    <input
                      type="file"
                      accept="image/*"
                      id={`upload-${selectedPerson.person_id}`}
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0]
                        if (file) uploadPhoto(selectedPerson.person_id, file)
                      }}
                    />
                    <label htmlFor={`upload-${selectedPerson.person_id}`}>
                      <Button variant="outline" size="sm" asChild>
                        <span>
                          <Upload className="w-4 h-4 mr-1" />
                          Update Photo
                        </span>
                      </Button>
                    </label>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => removeFromWatchlist(selectedPerson.person_id)}
                    >
                      Remove from Watchlist
                    </Button>
                  </div>
                </div>
              </div>

              {/* Location Timeline */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <History className="w-5 h-5" />
                  Location History
                </h3>
                {locationHistory.length === 0 ? (
                  <p className="text-gray-500">No location history</p>
                ) : (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {locationHistory.map((loc, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-3 p-2 bg-gray-800 rounded-lg"
                      >
                        <div className="w-10 h-10 bg-gray-700 rounded overflow-hidden">
                          {loc.thumbnail_url ? (
                            <img src={loc.thumbnail_url} alt="" className="w-full h-full object-cover" />
                          ) : (
                            <Camera className="w-6 h-6 m-2 text-gray-500" />
                          )}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <MapPin className="w-4 h-4 text-green-400" />
                            <span className="font-medium">{loc.camera_name}</span>
                          </div>
                          <span className="text-xs text-gray-400">{formatDateTime(loc.detected_at)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Detection History with Images */}
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <ImageIcon className="w-5 h-5" />
                  Detection Images
                </h3>
                {personDetections.length === 0 ? (
                  <p className="text-gray-500">No detection images</p>
                ) : (
                  <div className="grid grid-cols-4 md:grid-cols-6 gap-2 max-h-64 overflow-y-auto">
                    {personDetections.map((det, idx) => (
                      <div
                        key={idx}
                        className="relative group cursor-pointer"
                        onClick={() => det.fullframe_url && setSelectedFullframe(det.fullframe_url)}
                      >
                        <div className="aspect-square bg-gray-800 rounded overflow-hidden">
                          {det.thumbnail_url ? (
                            <img
                              src={det.thumbnail_url}
                              alt=""
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <User className="w-8 h-8 text-gray-600" />
                            </div>
                          )}
                        </div>
                        <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-xs p-1 text-center truncate">
                          {det.camera_name}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Fullframe Modal */}
      <Dialog open={!!selectedFullframe} onOpenChange={() => setSelectedFullframe(null)}>
        <DialogContent className="max-w-5xl bg-black border-gray-700">
          <DialogHeader>
            <DialogTitle>Detection Full Frame</DialogTitle>
          </DialogHeader>
          {selectedFullframe && (
            <img
              src={selectedFullframe}
              alt="Full frame"
              className="w-full h-auto max-h-[80vh] object-contain"
            />
          )}
        </DialogContent>
      </Dialog>
      </main>
      
      <Footer />
    </div>
  )
}
