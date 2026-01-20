import { Link } from 'react-router-dom'
import { Users, MapPin, ArrowRight, AlertTriangle } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import Header from '@/_comps/Header'
import Footer from '@/_comps/Footer'

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header currentPage="home" />

      {/* Hero Section */}
      <main className="flex-1 container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            AI-Powered People Analytics
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Advanced computer vision system for real-time people counting and tracking across multiple camera feeds
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          <Link to="/counting" className="group">
            <Card className="h-full bg-gradient-to-br from-gray-800/50 to-gray-900/50 border-gray-700 hover:border-blue-500 transition-all duration-300 transform hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mb-6 group-hover:bg-blue-500/30 transition-colors">
                  <Users className="w-8 h-8 text-blue-400" />
                </div>
                <h3 className="text-2xl font-bold mb-3 text-white">People Counting</h3>
                <p className="text-gray-400 mb-6">
                  Real-time people counting with face detection, unique visitor tracking, ROI zones, and multi-camera support
                </p>
                <div className="flex items-center text-blue-400 font-semibold">
                  <span>Get Started</span>
                  <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-2 transition-transform" />
                </div>
                <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-gray-800/50 p-3 rounded-lg">
                    <div className="text-gray-500">Features</div>
                    <div className="text-white font-semibold">Face Detection</div>
                  </div>
                  <div className="bg-gray-800/50 p-3 rounded-lg">
                    <div className="text-gray-500">ROI</div>
                    <div className="text-white font-semibold">Custom Zones</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>

          <Link to="/tracking" className="group">
            <Card className="h-full bg-gradient-to-br from-gray-800/50 to-gray-900/50 border-gray-700 hover:border-purple-500 transition-all duration-300 transform hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 bg-purple-500/20 rounded-full flex items-center justify-center mb-6 group-hover:bg-purple-500/30 transition-colors">
                  <MapPin className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-2xl font-bold mb-3 text-white">People Tracking</h3>
                <p className="text-gray-400 mb-6">
                  Track individual movement across cameras, location history, last seen timestamps, and movement analytics
                </p>
                <div className="flex items-center text-purple-400 font-semibold">
                  <span>Get Started</span>
                  <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-2 transition-transform" />
                </div>
                <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
                  <div className="bg-gray-800/50 p-3 rounded-lg">
                    <div className="text-gray-500">Track</div>
                    <div className="text-white font-semibold">Movement Path</div>
                  </div>
                  <div className="bg-gray-800/50 p-3 rounded-lg">
                    <div className="text-gray-500">History</div>
                    <div className="text-white font-semibold">Location Data</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        </div>

        {/* Watchlist Feature Card */}
        <div className="max-w-5xl mx-auto mt-8">
          <Link to="/watchlist" className="group block">
            <Card className="bg-gradient-to-r from-yellow-900/30 to-orange-900/30 border-yellow-700/50 hover:border-yellow-500 transition-all duration-300">
              <CardContent className="p-6 flex items-center gap-6">
                <div className="w-14 h-14 bg-yellow-500/20 rounded-full flex items-center justify-center group-hover:bg-yellow-500/30 transition-colors flex-shrink-0">
                  <AlertTriangle className="w-7 h-7 text-yellow-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-1 text-white">Watchlist Management</h3>
                  <p className="text-gray-400">
                    Track persons of interest with photos, get alerts on detection, view location history and search across all cameras
                  </p>
                </div>
                <ArrowRight className="w-6 h-6 text-yellow-400 group-hover:translate-x-2 transition-transform" />
              </CardContent>
            </Card>
          </Link>
        </div>

        {/* Tech Stack */}
        <div className="mt-20 text-center">
          <p className="text-gray-500 mb-4">Powered by</p>
          <div className="flex justify-center gap-8 text-gray-400">
            <span>InsightFace</span>
            <span>•</span>
            <span>CUDA GPU</span>
            <span>•</span>
            <span>FastAPI</span>
            <span>•</span>
            <span>React</span>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
