import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import PeopleCounting from './pages/PeopleCounting'
import PeopleTracking from './pages/PeopleTracking'
import WatchlistPage from './pages/Watchlist'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/counting" element={<PeopleCounting />} />
        <Route path="/tracking" element={<PeopleTracking />} />
        <Route path="/watchlist" element={<WatchlistPage />} />
      </Routes>
    </div>
  )
}

export default App
