import React from 'react'
import { CheckCircle, XCircle, Play, Clock, AlertCircle } from 'lucide-react'

export default function StepDetailCard({ step }) {
  const getStepIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={20} />
      case 'failed':
        return <XCircle className="text-red-500" size={20} />
      case 'running':
        return <Play className="text-blue-500 animate-pulse" size={20} />
      default:
        return <Clock className="text-gray-400" size={20} />
    }
  }

  const getStepColor = (status) => {
    switch (status) {
      case 'completed':
        return 'border-green-200 bg-green-50'
      case 'failed':
        return 'border-red-200 bg-red-50'
      case 'running':
        return 'border-blue-200 bg-blue-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  const getStatusBadge = (status) => {
    const colors = {
      completed: 'bg-green-100 text-green-800 border-green-300',
      failed: 'bg-red-100 text-red-800 border-red-300',
      running: 'bg-blue-100 text-blue-800 border-blue-300',
      pending: 'bg-gray-100 text-gray-800 border-gray-300',
    }

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${colors[status] || colors.pending}`}>
        {status?.charAt(0).toUpperCase() + status?.slice(1)}
      </span>
    )
  }

  // Safe date formatting function
  const formatDate = (dateValue) => {
    if (!dateValue) return 'N/A'
    
    try {
      const date = new Date(dateValue)
      // Check if date is valid
      if (isNaN(date.getTime())) {
        console.warn('Invalid date:', dateValue)
        return 'Invalid Date'
      }
      return date.toLocaleString()
    } catch (error) {
      console.error('Error formatting date:', dateValue, error)
      return 'Invalid Date'
    }
  }

  const formatTime = (dateValue) => {
    if (!dateValue) return 'N/A'
    
    try {
      const date = new Date(dateValue)
      if (isNaN(date.getTime())) return 'Invalid Time'
      return date.toLocaleTimeString()
    } catch (error) {
      return 'Invalid Time'
    }
  }

  return (
    <div className={`border rounded-lg p-5 ${getStepColor(step.status)} transition-all hover:shadow-md`}>
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 mt-1">{getStepIcon(step.status)}</div>
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-start justify-between mb-3">
            <div>
              <h4 className="text-lg font-semibold text-gray-900 mb-1">
                {step.step_name}
              </h4>
              <div className="flex items-center gap-2">
                {getStatusBadge(step.status)}
                {step.step_type && (
                  <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded border border-gray-300">
                    {step.step_type}
                  </span>
                )}
              </div>
            </div>
            <div className="text-right">
              {step.execution_time_ms != null && (
                <div className="text-lg font-bold text-gray-900">
                  {step.execution_time_ms.toFixed(0)}ms
                </div>
              )}
              {step.retry_count > 0 && (
                <div className="text-xs text-orange-600 flex items-center gap-1 mt-1">
                  <AlertCircle size={12} />
                  Retried {step.retry_count}x
                </div>
              )}
            </div>
          </div>

          {/* Error Display */}
          {step.error_message && (
            <div className="mb-3 p-3 bg-red-100 border border-red-300 rounded text-sm text-red-900">
              <strong className="block mb-1">Error:</strong>
              <div className="font-mono text-xs whitespace-pre-wrap break-words">
                {step.error_message}
              </div>
            </div>
          )}

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
            <div className="bg-white bg-opacity-60 rounded p-3 border border-gray-200">
              <div className="text-xs text-gray-500 mb-1">Started</div>
              <div className="text-sm font-medium text-gray-900">
                {formatTime(step.started_at)}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                {step.started_at ? new Date(step.started_at).toLocaleDateString() : 'N/A'}
              </div>
            </div>

            {step.completed_at && (
              <div className="bg-white bg-opacity-60 rounded p-3 border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Completed</div>
                <div className="text-sm font-medium text-gray-900">
                  {formatTime(step.completed_at)}
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {new Date(step.completed_at).toLocaleDateString()}
                </div>
              </div>
            )}

            {step.execution_time_ms != null && (
              <div className="bg-white bg-opacity-60 rounded p-3 border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">Duration</div>
                <div className="text-sm font-medium text-gray-900">
                  {step.execution_time_ms < 1000 
                    ? `${step.execution_time_ms.toFixed(0)}ms`
                    : `${(step.execution_time_ms / 1000).toFixed(2)}s`
                  }
                </div>
              </div>
            )}

            {(step.tokens || step.cost != null) && (
              <div className="bg-white bg-opacity-60 rounded p-3 border border-gray-200">
                <div className="text-xs text-gray-500 mb-1">
                  {step.tokens ? 'Tokens' : 'Cost'}
                </div>
                <div className="text-sm font-medium text-gray-900">
                  {step.tokens || (step.cost != null ? `$${step.cost.toFixed(4)}` : 'N/A')}
                </div>
              </div>
            )}
          </div>

          {/* Data Preview */}
          <div className="space-y-2">
            {/* Input Preview */}
            {step.input_data && Object.keys(step.input_data).length > 0 && (
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900 flex items-center gap-2 py-2">
                  <span className="text-blue-600">▸</span>
                  <span>Input Data</span>
                  <span className="text-xs text-gray-500">
                    ({Object.keys(step.input_data).length} {Object.keys(step.input_data).length === 1 ? 'key' : 'keys'})
                  </span>
                </summary>
                <div className="mt-2 ml-6">
                  <pre className="text-xs text-gray-900 bg-white p-3 rounded border overflow-x-auto max-h-48 overflow-y-auto">
                    {JSON.stringify(step.input_data, null, 2)}
                  </pre>
                </div>
              </details>
            )}

            {/* Output Preview */}
            {step.output_data && Object.keys(step.output_data).length > 0 && (
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900 flex items-center gap-2 py-2">
                  <span className="text-green-600">▸</span>
                  <span>Output Data</span>
                  <span className="text-xs text-gray-500">
                    ({Object.keys(step.output_data).length} {Object.keys(step.output_data).length === 1 ? 'key' : 'keys'})
                  </span>
                </summary>
                <div className="mt-2 ml-6">
                  <pre className="text-xs text-gray-900 bg-white p-3 rounded border overflow-x-auto max-h-48 overflow-y-auto">
                    {JSON.stringify(step.output_data, null, 2)}
                  </pre>
                </div>
              </details>
            )}

            {/* Metadata Preview */}
            {step.metadata && Object.keys(step.metadata).length > 0 && (
              <details className="group">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900 flex items-center gap-2 py-2">
                  <span className="text-purple-600">▸</span>
                  <span>Metadata</span>
                  <span className="text-xs text-gray-500">
                    ({Object.keys(step.metadata).length} {Object.keys(step.metadata).length === 1 ? 'key' : 'keys'})
                  </span>
                </summary>
                <div className="mt-2 ml-6">
                  <pre className="text-xs text-gray-900 bg-white p-3 rounded border overflow-x-auto max-h-48 overflow-y-auto">
                    {JSON.stringify(step.metadata, null, 2)}
                  </pre>
                </div>
              </details>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
