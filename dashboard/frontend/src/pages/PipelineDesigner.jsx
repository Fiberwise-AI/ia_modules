import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Save, Play } from 'lucide-react'
import { pipelines as pipelinesApi } from '../services/api'

export default function PipelineDesigner() {
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [config, setConfig] = useState(JSON.stringify({
    name: "new_pipeline",
    steps: [],
    flow: {
      start_at: "",
      paths: []
    }
  }, null, 2))

  async function savePipeline() {
    try {
      const parsedConfig = JSON.parse(config)
      const response = await pipelinesApi.create({
        name,
        description,
        config: parsedConfig,
        tags: []
      })
      alert('Pipeline created successfully!')
      navigate('/pipelines')
    } catch (error) {
      console.error('Failed to create pipeline:', error)
      alert('Failed to create pipeline: ' + error.message)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Pipeline Designer</h1>
        <div className="flex gap-2">
          <button onClick={savePipeline} className="btn-primary flex items-center">
            <Save className="w-4 h-4 mr-2" />
            Save
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Pipeline Details</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                placeholder="My Pipeline"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                rows="3"
                placeholder="Pipeline description..."
              />
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Configuration (JSON)</h2>
          <textarea
            value={config}
            onChange={(e) => setConfig(e.target.value)}
            className="w-full font-mono text-sm px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            rows="20"
          />
          <p className="mt-2 text-xs text-gray-500">
            Visual designer coming soon! For now, edit JSON directly.
          </p>
        </div>
      </div>
    </div>
  )
}
