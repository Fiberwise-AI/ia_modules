import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Plus, Play, Edit, Trash2, Search } from 'lucide-react'
import { pipelines as pipelinesApi } from '../services/api'
import { format } from 'date-fns'

export default function PipelineList() {
  const [pipelines, setPipelines] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [stats, setStats] = useState(null)

  useEffect(() => {
    loadPipelines()
    loadStats()
  }, [search])

  async function loadPipelines() {
    try {
      setLoading(true)
      const response = await pipelinesApi.list({ search })
      setPipelines(response.data)
    } catch (error) {
      console.error('Failed to load pipelines:', error)
    } finally {
      setLoading(false)
    }
  }

  async function loadStats() {
    try {
      const response = await fetch('/api/stats')
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  async function executePipeline(id) {
    try {
      const response = await pipelinesApi.execute(id, {})
      const { execution_id } = response.data
      window.location.href = `/monitor/${execution_id}`
    } catch (error) {
      console.error('Failed to execute pipeline:', error)
      alert('Failed to execute pipeline: ' + error.message)
    }
  }

  async function deletePipeline(id) {
    if (!confirm('Are you sure you want to delete this pipeline?')) return

    try {
      await pipelinesApi.delete(id)
      loadPipelines()
    } catch (error) {
      console.error('Failed to delete pipeline:', error)
      alert('Failed to delete pipeline: ' + error.message)
    }
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Pipelines</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage and execute your data pipelines
          </p>
        </div>
        <Link
          to="/pipelines/new"
          className="btn-primary flex items-center"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Pipeline
        </Link>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Total Pipelines</div>
            <div className="mt-1 text-2xl font-bold text-gray-900">{stats.total_pipelines}</div>
          </div>
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Active Executions</div>
            <div className="mt-1 text-2xl font-bold text-primary-600">{stats.active_executions}</div>
          </div>
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Executions Today</div>
            <div className="mt-1 text-2xl font-bold text-gray-900">{stats.total_executions_today}</div>
          </div>
          <div className="card">
            <div className="text-sm font-medium text-gray-500">Telemetry</div>
            <div className="mt-1 text-2xl font-bold text-green-600">
              {stats.telemetry_enabled ? 'Enabled' : 'Disabled'}
            </div>
          </div>
        </div>
      )}

      {/* Search */}
      <div className="card mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search pipelines..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Pipeline List */}
      <div className="card">
        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading pipelines...</div>
        ) : pipelines.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500">No pipelines found</p>
            <Link to="/pipelines/new" className="text-primary-600 hover:text-primary-700 mt-2 inline-block">
              Create your first pipeline
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tags
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Updated
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {pipelines.map((pipeline) => (
                  <tr key={pipeline.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{pipeline.name}</div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-500">{pipeline.description || '—'}</div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-1">
                        {pipeline.tags?.map((tag) => (
                          <span key={tag} className="px-2 py-1 text-xs font-medium rounded bg-gray-100 text-gray-700">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {format(new Date(pipeline.updated_at), 'MMM d, yyyy')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => executePipeline(pipeline.id)}
                        className="text-primary-600 hover:text-primary-900 mr-3"
                        title="Execute"
                      >
                        <Play className="w-4 h-4" />
                      </button>
                      <Link
                        to={`/pipelines/${pipeline.id}/edit`}
                        className="text-gray-600 hover:text-gray-900 mr-3"
                        title="Edit"
                      >
                        <Edit className="w-4 h-4" />
                      </Link>
                      <button
                        onClick={() => deletePipeline(pipeline.id)}
                        className="text-red-600 hover:text-red-900"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
