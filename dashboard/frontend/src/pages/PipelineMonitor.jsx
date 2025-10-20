import { useState, useEffect, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { CheckCircle, XCircle, Loader, Clock } from 'lucide-react'
import WebSocketService from '../services/websocket'
import { executions } from '../services/api'
import { format } from 'date-fns'

export default function PipelineMonitor() {
  const { executionId } = useParams()
  const [status, setStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [metrics, setMetrics] = useState({})
  const wsRef = useRef(null)

  useEffect(() => {
    loadStatus()

    // Connect WebSocket
    wsRef.current = new WebSocketService()
    wsRef.current.connect(executionId)

    // Listen for updates
    wsRef.current.on('execution_started', (msg) => {
      console.log('Execution started:', msg)
    })

    wsRef.current.on('step_started', (msg) => {
      addLog('info', `Step started: ${msg.data.step_name}`)
    })

    wsRef.current.on('step_completed', (msg) => {
      addLog('success', `Step completed: ${msg.data.step_name} (${msg.data.duration_seconds}s)`)
      loadStatus()
    })

    wsRef.current.on('step_failed', (msg) => {
      addLog('error', `Step failed: ${msg.data.step_name} - ${msg.data.error}`)
      loadStatus()
    })

    wsRef.current.on('log_message', (msg) => {
      addLog(msg.data.level, msg.data.message)
    })

    wsRef.current.on('progress_update', (msg) => {
      setMetrics(prev => ({ ...prev, ...msg.data }))
    })

    wsRef.current.on('metrics_update', (msg) => {
      setMetrics(prev => ({ ...prev, ...msg.data }))
    })

    wsRef.current.on('execution_completed', (msg) => {
      addLog('success', `Execution completed in ${msg.data.duration_seconds}s`)
      loadStatus()
    })

    wsRef.current.on('execution_failed', (msg) => {
      addLog('error', `Execution failed: ${msg.data.error}`)
      loadStatus()
    })

    return () => {
      if (wsRef.current) {
        wsRef.current.disconnect()
      }
    }
  }, [executionId])

  async function loadStatus() {
    try {
      const response = await executions.getStatus(executionId)
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to load status:', error)
    }
  }

  function addLog(level, message) {
    setLogs(prev => [...prev, {
      level,
      message,
      timestamp: new Date().toISOString()
    }])
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
      case 'cancelled':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'running':
        return <Loader className="w-5 h-5 text-primary-500 animate-spin" />
      default:
        return <Clock className="w-5 h-5 text-gray-400" />
    }
  }

  function getStatusColor(status) {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50'
      case 'failed':
      case 'cancelled':
        return 'text-red-600 bg-red-50'
      case 'running':
        return 'text-primary-600 bg-primary-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Pipeline Execution</h1>
            <p className="mt-1 text-sm text-gray-500">ID: {executionId}</p>
          </div>
          <div className={`px-4 py-2 rounded-lg font-medium ${getStatusColor(status.status)}`}>
            <div className="flex items-center">
              {getStatusIcon(status.status)}
              <span className="ml-2 capitalize">{status.status}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="card">
          <div className="text-sm font-medium text-gray-500">Duration</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            {metrics.duration_seconds?.toFixed(2) || status.duration_seconds?.toFixed(2) || '—'}s
          </div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-500">Progress</div>
          <div className="mt-1 text-2xl font-bold text-primary-600">
            {metrics.progress_percent?.toFixed(1) || status.progress_percent?.toFixed(1) || 0}%
          </div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-500">Items Processed</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            {metrics.items_processed?.toLocaleString() || '—'}
          </div>
        </div>
        <div className="card">
          <div className="text-sm font-medium text-gray-500">Cost</div>
          <div className="mt-1 text-2xl font-bold text-gray-900">
            ${metrics.cost_usd?.toFixed(2) || '0.00'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Steps Progress */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Steps</h2>
          <div className="space-y-3">
            {status.steps && status.steps.length > 0 ? (
              status.steps.map((step, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center flex-1">
                    {getStatusIcon(step.status)}
                    <span className="ml-3 text-sm font-medium text-gray-900">
                      {step.step_name}
                    </span>
                  </div>
                  {step.duration_seconds && (
                    <span className="text-sm text-gray-500">
                      {step.duration_seconds.toFixed(2)}s
                    </span>
                  )}
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">No steps yet</p>
            )}
          </div>
        </div>

        {/* Live Logs */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Live Logs</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <div key={index} className="text-sm">
                  <span className="text-gray-400">
                    {format(new Date(log.timestamp), 'HH:mm:ss')}
                  </span>
                  <span className={`ml-2 ${
                    log.level === 'error' ? 'text-red-600' :
                    log.level === 'success' ? 'text-green-600' :
                    'text-gray-700'
                  }`}>
                    {log.message}
                  </span>
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500">No logs yet</p>
            )}
          </div>
        </div>
      </div>

      {/* Output */}
      {status.output_data && (
        <div className="card mt-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Output</h2>
          <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm">
            {JSON.stringify(status.output_data, null, 2)}
          </pre>
        </div>
      )}

      {/* Error */}
      {status.error && (
        <div className="card mt-6 bg-red-50 border-red-200">
          <h2 className="text-lg font-semibold text-red-900 mb-2">Error</h2>
          <p className="text-sm text-red-700">{status.error}</p>
        </div>
      )}
    </div>
  )
}
