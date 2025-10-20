import { useState, useEffect } from 'react'
import { Puzzle } from 'lucide-react'
import { plugins as pluginsApi } from '../services/api'

export default function PluginsBrowser() {
  const [plugins, setPlugins] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadPlugins()
  }, [])

  async function loadPlugins() {
    try {
      const response = await pluginsApi.list()
      setPlugins(response.data.plugins || [])
    } catch (error) {
      console.error('Failed to load plugins:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Plugins</h1>

      {loading ? (
        <div className="card text-center py-8">Loading plugins...</div>
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {plugins.map((plugin) => (
            <div key={plugin.name} className="card">
              <div className="flex items-start">
                <Puzzle className="w-8 h-8 text-primary-500 mr-3" />
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900">{plugin.name}</h3>
                  <p className="text-sm text-gray-500 mt-1">{plugin.description || 'No description'}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <span className="px-2 py-1 text-xs font-medium rounded bg-gray-100 text-gray-700">
                      {plugin.type}
                    </span>
                    <span className="text-xs text-gray-500">v{plugin.version}</span>
                  </div>
                  {plugin.author && (
                    <p className="text-xs text-gray-500 mt-2">by {plugin.author}</p>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
