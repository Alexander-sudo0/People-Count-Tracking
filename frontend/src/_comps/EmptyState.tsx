import { Camera } from 'lucide-react'

export default function EmptyState({ message = 'Select a camera to view live feed' }) {
  return (
    <div className="flex-1 flex items-center justify-center text-gray-500">
      <div className="text-center">
        <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
        <p className="text-lg">{message}</p>
      </div>
    </div>
  )
}
