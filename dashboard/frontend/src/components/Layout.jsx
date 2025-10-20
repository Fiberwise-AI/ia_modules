import { Link, useLocation } from 'react-router-dom'
import {
  Layers,
  Activity,
  BarChart3,
  Puzzle,
  Plus
} from 'lucide-react'
import clsx from 'clsx'

const navigation = [
  { name: 'Pipelines', href: '/pipelines', icon: Layers },
  { name: 'Metrics', href: '/metrics', icon: BarChart3 },
  { name: 'Plugins', href: '/plugins', icon: Puzzle },
]

export default function Layout({ children }) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200">
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center h-16 px-6 border-b border-gray-200">
            <Activity className="w-8 h-8 text-primary-600" />
            <span className="ml-2 text-xl font-bold text-gray-900">
              IA Modules
            </span>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-4 space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname.startsWith(item.href)
              const Icon = item.icon

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    'flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                    isActive
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-700 hover:bg-gray-100'
                  )}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {item.name}
                </Link>
              )
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="text-xs text-gray-500">
              <div>IA Modules Dashboard</div>
              <div className="mt-1">v0.1.0</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="py-6 px-8">
          {children}
        </main>
      </div>
    </div>
  )
}
