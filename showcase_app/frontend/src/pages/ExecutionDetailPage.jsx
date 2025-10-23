import React, { useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { executionAPI } from '../services/api'
import { CheckCircle, XCircle, Clock, Play, ArrowLeft, AlertCircle, ArrowDown, Circle } from 'lucide-react'
import { useExecutionWebSocket } from '../hooks/useWebSocket'

export default function ExecutionDetailPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: execution, isLoading } = useQuery({
    queryKey: ['execution', jobId],
    queryFn: async () => {
      const response = await executionAPI.get(jobId)
      return response.data
    },
    refetchInterval: false, // Disabled - using WebSocket instead
  })

  // Use WebSocket for real-time updates
  const handleWebSocketUpdate = useCallback((data) => {
    queryClient.setQueryData(['execution', jobId], (old) => ({
      ...old,
      ...data
    }))
  }, [jobId, queryClient])

  const { isConnected } = useExecutionWebSocket(jobId, handleWebSocketUpdate)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading execution details...</div>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-600">Execution not found</div>
      </div>
    )
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={24} />
      case 'failed':
        return <XCircle className="text-red-500" size={24} />
      case 'running':
        return <Play className="text-blue-500 animate-pulse" size={24} />
      default:
        return <Clock className="text-gray-500" size={24} />
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
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/executions')}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft size={20} />
            <span>Back to Executions</span>
          </button>
        </div>
      </div>

      {/* Status Card */}
      <div className={`border-2 rounded-lg p-6 ${getStatusColor(execution.status)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {getStatusIcon(execution.status)}
            <div>
              <h1 className="text-2xl font-bold">Pipeline Execution</h1>
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

      {/* Error Message */}
      {execution.error && (
        <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="text-red-500 flex-shrink-0 mt-0.5" size={20} />
            <div className="flex-1">
              <h3 className="text-red-900 font-semibold mb-1">Execution Failed</h3>
              <pre className="text-red-800 text-sm whitespace-pre-wrap">{execution.error}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Flow Visualization */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Pipeline Flow</h2>
          <p className="text-sm text-gray-600">Visual representation of step execution</p>
        </div>
        <div className="p-8">
          {execution.steps && execution.steps.length > 0 ? (
            <PipelineFlowDiagram steps={execution.steps} />
          ) : (
            <div className="text-center py-8 text-gray-500">
              No step execution data available
            </div>
          )}
        </div>
      </div>

      {/* Step Details List */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Step Execution Details</h2>
        </div>
        <div className="p-6">
          {execution.steps && execution.steps.length > 0 ? (
            <div className="space-y-4">
              {execution.steps.map((step, index) => (
                <StepDetailCard key={index} step={step} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No step execution data available
            </div>
          )}
        </div>
      </div>

      {/* Input/Output Data */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800">Input Data</h3>
          </div>
          <div className="p-6">
            <pre className="text-sm text-gray-900 bg-gray-50 p-4 rounded border overflow-x-auto">
              {JSON.stringify(execution.input_data, null, 2)}
            </pre>
          </div>
        </div>

        {execution.output_data && (
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800">Final Output</h3>
            </div>
            <div className="p-6">
              <pre className="text-sm text-gray-900 bg-gray-50 p-4 rounded border overflow-x-auto max-h-96 overflow-y-auto">
                {JSON.stringify(execution.output_data, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function PipelineFlowDiagram({ steps }) {
  return (
    <div className="flex flex-col items-center space-y-4 max-w-3xl mx-auto">
      {steps.map((step, index) => (
        <React.Fragment key={index}>
          <FlowNode step={step} index={index} />
          {index < steps.length - 1 && <FlowConnector />}
        </React.Fragment>
      ))}
    </div>
  )
}

function FlowNode({ step, index }) {
  const getNodeColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 border-green-400 text-green-900'
      case 'failed':
        return 'bg-red-100 border-red-400 text-red-900'
      case 'running':
        return 'bg-blue-100 border-blue-400 text-blue-900 animate-pulse'
      default:
        return 'bg-gray-100 border-gray-400 text-gray-900'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-600" size={32} />
      case 'failed':
        return <XCircle className="text-red-600" size={32} />
      case 'running':
        return <Play className="text-blue-600" size={32} />
      default:
        return <Circle className="text-gray-400" size={32} />
    }
  }

  return (
    <div className={`w-full border-2 rounded-xl p-6 ${getNodeColor(step.status)} transition-all duration-300`}>
      <div className="flex items-center space-x-4">
        {getStatusIcon(step.status)}
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-sm font-medium opacity-75">Step {index + 1}</div>
              <h3 className="text-xl font-bold">{step.step_name}</h3>
            </div>
            {step.duration_ms !== null && (
              <div className="text-right">
                <div className="text-2xl font-bold">{step.duration_ms.toFixed(0)}ms</div>
                <div className="text-xs opacity-75">duration</div>
              </div>
            )}
          </div>

          {step.error && (
            <div className="mt-2 p-3 bg-red-200 bg-opacity-50 rounded-lg">
              <div className="font-semibold text-sm">Error:</div>
              <div className="text-sm">{step.error}</div>
            </div>
          )}

          <div className="mt-2 flex items-center space-x-4 text-sm opacity-75">
            <span>Started: {new Date(step.started_at).toLocaleTimeString()}</span>
            {step.completed_at && (
              <span>Completed: {new Date(step.completed_at).toLocaleTimeString()}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function FlowConnector() {
  return (
    <div className="flex flex-col items-center py-2">
      <ArrowDown className="text-gray-400" size={32} strokeWidth={2} />
    </div>
  )
}

function StepDetailCard({ step }) {
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
            {step.duration_ms !== null && (
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
