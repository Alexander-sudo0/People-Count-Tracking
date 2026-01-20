import { useState, useEffect } from 'react'
import { AlertTriangle, X, Plus, Eye, Bell, BellOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import axios from 'axios'

interface WatchlistPerson {
  person_id: string
  name: string | null
  added_at: string
  notes: string | null
  alert_enabled: boolean
  detection_count: number
  last_seen: string | null
}

interface WatchlistProps {
  onAddPerson?: (personId: string) => void
}

export default function Watchlist({ onAddPerson }: WatchlistProps) {
  const [watchlist, setWatchlist] = useState<WatchlistPerson[]>([])
  const [showAddForm, setShowAddForm] = useState(false)
  const [newPerson, setNewPerson] = useState({
    person_id: '',
    name: '',
    notes: '',
    alert_enabled: true,
  })

  useEffect(() => {
    fetchWatchlist()
    const interval = setInterval(fetchWatchlist, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchWatchlist = async () => {
    try {
      const res = await axios.get('/api/watchlist')
      setWatchlist(res.data.watchlist || [])
    } catch (error) {
      console.error('Error fetching watchlist:', error)
    }
  }

  const addToWatchlist = async () => {
    if (!newPerson.person_id) return

    try {
      const formData = new FormData()
      formData.append('person_id', newPerson.person_id)
      if (newPerson.name) formData.append('name', newPerson.name)
      if (newPerson.notes) formData.append('notes', newPerson.notes)
      formData.append('alert_enabled', newPerson.alert_enabled ? 'true' : 'false')

      await axios.post('/api/watchlist/add', formData)
      setNewPerson({ person_id: '', name: '', notes: '', alert_enabled: true })
      setShowAddForm(false)
      fetchWatchlist()
      if (onAddPerson) onAddPerson(newPerson.person_id)
    } catch (error) {
      console.error('Error adding to watchlist:', error)
    }
  }

  const removeFromWatchlist = async (personId: string) => {
    try {
      await axios.delete(`/api/watchlist/${personId}`)
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

  return (
    <Card className="bg-gray-800/50 border-gray-700">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-500" />
          Watchlist ({watchlist.length})
        </CardTitle>
        <Button
          size="sm"
          onClick={() => setShowAddForm(!showAddForm)}
          className="bg-yellow-600 hover:bg-yellow-700"
        >
          <Plus className="w-4 h-4 mr-1" />
          Add Person
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {showAddForm && (
          <div className="p-4 bg-gray-900/50 rounded-lg border border-gray-700 space-y-3">
            <Input
              placeholder="Person ID"
              value={newPerson.person_id}
              onChange={(e) => setNewPerson({ ...newPerson, person_id: e.target.value })}
              className="bg-gray-800 border-gray-700"
            />
            <Input
              placeholder="Name (optional)"
              value={newPerson.name}
              onChange={(e) => setNewPerson({ ...newPerson, name: e.target.value })}
              className="bg-gray-800 border-gray-700"
            />
            <Input
              placeholder="Notes (optional)"
              value={newPerson.notes}
              onChange={(e) => setNewPerson({ ...newPerson, notes: e.target.value })}
              className="bg-gray-800 border-gray-700"
            />
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="alert-enabled"
                checked={newPerson.alert_enabled}
                onChange={(e) => setNewPerson({ ...newPerson, alert_enabled: e.target.checked })}
                className="rounded"
              />
              <label htmlFor="alert-enabled" className="text-sm text-gray-300">
                Enable alerts
              </label>
            </div>
            <div className="flex gap-2">
              <Button onClick={addToWatchlist} className="flex-1 bg-yellow-600 hover:bg-yellow-700">
                Add to Watchlist
              </Button>
              <Button onClick={() => setShowAddForm(false)} variant="outline">
                Cancel
              </Button>
            </div>
          </div>
        )}

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {watchlist.length === 0 && !showAddForm && (
            <div className="text-center py-8 text-gray-500">
              <AlertTriangle className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No persons in watchlist</p>
            </div>
          )}

          {watchlist.map((person) => (
            <div
              key={person.person_id}
              className="p-3 bg-gray-900/50 rounded-lg border border-gray-700 hover:border-yellow-600 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-white">
                      {person.name || person.person_id}
                    </span>
                    {person.alert_enabled ? (
                      <Bell className="w-3 h-3 text-yellow-500" />
                    ) : (
                      <BellOff className="w-3 h-3 text-gray-500" />
                    )}
                  </div>
                  {person.name && (
                    <div className="text-xs text-gray-500">ID: {person.person_id}</div>
                  )}
                  {person.notes && (
                    <div className="text-sm text-gray-400 mt-1">{person.notes}</div>
                  )}
                </div>
                <button
                  onClick={() => removeFromWatchlist(person.person_id)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <Eye className="w-3 h-3" />
                    {person.detection_count} detections
                  </span>
                  <span>Last seen: {formatTime(person.last_seen)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
