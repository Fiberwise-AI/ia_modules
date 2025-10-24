import React, { useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { executionAPI, pipelinesAPI } from '../services/api'
import { useExecutionWebSocket } from '../hooks/useWebSocket'
import axios from 'axios'

import ExecutionHeader from '../components/execution/ExecutionHeader'
import ExecutionStatusCard from '../components/execution/ExecutionStatusCard'
import ExecutionError from '../components/execution/ExecutionError'
import PipelineGraphSection from '../components/execution/PipelineGraphSection'
import StepDetailsList from '../components/execution/StepDetailsList'
import DataViewer from '../components/execution/DataViewer'
import ExecutionTimeline from '../components/execution/ExecutionTimeline'
import SpanTimeline from '../components/telemetry/SpanTimeline'
import CheckpointList from '../components/checkpoint/CheckpointList'
import ConversationHistory from '../components/memory/ConversationHistory'
import ReplayComparison from '../components/replay/ReplayComparison'
import DecisionTimeline from '../components/decision/DecisionTimeline'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5555'

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
    refetchInterval: false,
  })

  const { data: pipeline } = useQuery({
    queryKey: ['pipeline', execution?.pipeline_id],
    queryFn: async () => {
      if (!execution?.pipeline_id) return null
      const response = await pipelinesAPI.get(execution.pipeline_id)
      return response.data
    },
    enabled: !!execution?.pipeline_id,
  })

  // Fetch telemetry spans
  const { data: telemetryData } = useQuery({
    queryKey: ['telemetry', jobId],
    queryFn: async () => {
      const response = await axios.get(`${API_URL}/api/telemetry/timeline/${jobId}`)
      return response.data
    },
    enabled: !!jobId,
    retry: false
  })

  const handleWebSocketUpdate = useCallback((data) => {
    queryClient.setQueryData(['execution', jobId], (old) => ({
      ...old,
      ...data
    }))
  }, [jobId, queryClient])

  useExecutionWebSocket(jobId, handleWebSocketUpdate)

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

  return (
    <div className="space-y-6">
      <ExecutionHeader onBack={() => navigate('/executions')} />

      <ExecutionStatusCard execution={execution} />

      <ExecutionError error={execution.error} />

      {/* Execution Timeline - Gantt Chart */}
      <ExecutionTimeline execution={execution} />

      <PipelineGraphSection pipeline={pipeline} execution={execution} />

      {/* Telemetry Spans Timeline */}
      {telemetryData?.timeline && telemetryData.timeline.length > 0 && (
        <SpanTimeline jobId={jobId} spans={telemetryData.timeline} />
      )}

      {/* Checkpoints */}
      <CheckpointList jobId={jobId} />

      {/* Memory / Conversation History */}
      <div className="bg-white rounded-lg shadow-md border border-gray-200">
        <ConversationHistory sessionId={jobId} />
      </div>

      {/* Replay Comparison */}
      <div className="bg-white rounded-lg shadow-md border border-gray-200">
        <ReplayComparison jobId={jobId} />
      </div>

      {/* Decision Trail */}
      <div className="bg-white rounded-lg shadow-md border border-gray-200">
        <DecisionTimeline jobId={jobId} />
      </div>

      <StepDetailsList steps={execution.steps} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DataViewer
          title="Input Data"
          data={execution.input_data}
        />
        <DataViewer
          title="Final Output"
          data={execution.output_data}
          maxHeight="max-h-96 overflow-y-auto"
        />
      </div>
    </div>
  )
}
