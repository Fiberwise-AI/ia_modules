import React from 'react'
import { CheckCircle, XCircle, Clock, Play, AlertCircle } from 'lucide-react'

export default function ExecutionStatusCard({ execution }) {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={24} className="text-green-500" />
      case 'failed':
        return <XCircle size={24} className="text-red-500" />
      case 'running':
        return <Play size={24} className="text-blue-500 animate-pulse" />
      case 'pending':
        return <Clock size={24} className="text-yellow-600" />
      default:
        return <AlertCircle size={24} className="text-gray-500" />
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'running':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'pending':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getDuration = () => {
    const start = new Date(execution.started_at)
    const end = execution.completed_at ? new Date(execution.completed_at) : new Date()
    const durationMs = end - start
    const seconds = Math.floor(durationMs / 1000)

    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    return `${minutes}m ${seconds % 60}s`
  }

  return (
    <div className={`border-2 rounded-lg p-6 ${getStatusColor(execution.status)}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {getStatusIcon(execution.status)}
          <div>
            <h1 className="text-2xl font-bold">
              {execution.pipeline_name || 'Pipeline Execution'}
            </h1>
            <p className="text-sm opacity-75 font-mono">{execution.job_id}</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-bold">{Math.round(execution.progress * 100)}%</div>
          <div className="text-sm opacity-75">{execution.status}</div>
        </div>
      </div>

      {execution.current_step && (
        <div className="mt-4 flex items-center space-x-2">
          <Play size={16} className="animate-pulse" />
          <span className="text-sm font-medium">Currently executing: {execution.current_step}</span>
        </div>
      )}

      <div className="mt-4">
        <div className="w-full bg-white bg-opacity-30 rounded-full h-3">
          <div
            className="bg-current h-3 rounded-full transition-all duration-300"
            style={{ width: `${execution.progress * 100}%` }}
          ></div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm flex-wrap gap-2">
        <span>Started: {new Date(execution.started_at).toLocaleString()}</span>
        <span>Duration: {getDuration()}</span>
        {execution.completed_at && (
          <span>Completed: {new Date(execution.completed_at).toLocaleString()}</span>
        )}
      </div>
    </div>
  )
}
