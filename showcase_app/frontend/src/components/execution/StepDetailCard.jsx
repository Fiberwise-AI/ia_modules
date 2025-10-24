import React from 'react'
import { CheckCircle, XCircle, Play, Clock } from 'lucide-react'

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

  return (
    <div className={`border rounded-lg p-4 ${getStepColor(step.status)}`}>
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-1">{getStepIcon(step.status)}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-lg font-semibold text-gray-900">
              {step.step_name}
            </h4>
            {step.duration_ms != null && (
              <span className="text-sm text-gray-600">{step.duration_ms.toFixed(0)}ms</span>
            )}
          </div>

          {step.error && (
            <div className="mb-3 p-2 bg-red-100 border border-red-300 rounded text-sm text-red-900">
              <strong>Error:</strong> {step.error}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm mb-3">
            <div>
              <span className="text-gray-500">Started:</span>
              <span className="ml-2 text-gray-900">
                {new Date(step.started_at).toLocaleTimeString()}
              </span>
            </div>
            {step.completed_at && (
              <div>
                <span className="text-gray-500">Completed:</span>
                <span className="ml-2 text-gray-900">
                  {new Date(step.completed_at).toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>

          {/* Step Input */}
          {step.input_data && Object.keys(step.input_data).length > 0 && (
            <details className="mb-2">
              <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                Input Data
              </summary>
              <pre className="mt-2 text-xs text-gray-900 bg-white p-3 rounded border overflow-x-auto max-h-48 overflow-y-auto">
                {JSON.stringify(step.input_data, null, 2)}
              </pre>
            </details>
          )}

          {/* Step Output */}
          {step.output_data && (
            <details>
              <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                Output Data
              </summary>
              <pre className="mt-2 text-xs text-gray-900 bg-white p-3 rounded border overflow-x-auto max-h-48 overflow-y-auto">
                {JSON.stringify(step.output_data, null, 2)}
              </pre>
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
