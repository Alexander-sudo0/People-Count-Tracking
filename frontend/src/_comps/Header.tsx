import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { AlertTriangle } from 'lucide-react'

interface HeaderProps {
  currentPage?: 'counting' | 'tracking' | 'watchlist' | 'home'
}

export default function Header({ currentPage = 'home' }: HeaderProps) {
  return (
    <header className="border-b border-gray-700 bg-gray-900/90 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">O</span>
          </div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            OptiExacta
          </h1>
        </Link>
        <nav className="flex gap-4">
          <Link to="/counting">
            <Button variant={currentPage === 'counting' ? 'default' : 'ghost'}>
              People Counting
            </Button>
          </Link>
          <Link to="/tracking">
            <Button variant={currentPage === 'tracking' ? 'default' : 'ghost'}>
              People Tracking
            </Button>
          </Link>
          <Link to="/watchlist">
            <Button 
              variant={currentPage === 'watchlist' ? 'default' : 'ghost'}
              className={currentPage !== 'watchlist' ? 'text-yellow-400 hover:text-yellow-300' : ''}
            >
              <AlertTriangle className="w-4 h-4 mr-2" />
              Watchlist
            </Button>
          </Link>
        </nav>
      </div>
    </header>
  )
}
