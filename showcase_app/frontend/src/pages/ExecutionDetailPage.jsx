import React, { useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { executionAPI, pipelinesAPI } from '../services/api'
import { useExecutionWebSocket } from '../hooks/useWebSocket'
import axios from 'axios'
import ExecutionHeader from '../components/execution/ExecutionHeader'
import ExecutionStatusCard from '../components/execution/ExecutionStatusCard'
import ExecutionError from '../components/execution/ExecutionError'
import ExecutionTimeline from '../components/execution/ExecutionTimeline'
import PipelineGraphSection from '../components/execution/PipelineGraphSection'
import SpanTimeline from '../components/telemetry/SpanTimeline'
import CheckpointList from '../components/checkpoint/CheckpointList'
import ConversationHistory from '../components/memory/ConversationHistory'
import ReplayComparison from '../components/replay/ReplayComparison'
import DecisionTimeline from '../components/decision/DecisionTimeline'
import StepDetailsList from '../components/execution/StepDetailsList'
import DataViewer from '../components/execution/DataViewer'

const API_URL = import.meta.env.VITE_API_URL || ''

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
        <div className="text-gray-600 dark:text-gray-400">Loading execution details...</div>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-600 dark:text-red-400">Execution not found</div>
      </div>
    )
  }

  const hasSteps = execution.steps && execution.steps.length > 0
  const hasSpans = telemetryData?.timeline && telemetryData.timeline.length > 0

  return (
    <div className="space-y-4">
      <ExecutionHeader onBack={() => navigate('/executions')} />
      <ExecutionStatusCard execution={execution} />
      <ExecutionError error={execution.error} />

      {hasSteps && <ExecutionTimeline execution={execution} />}

      <PipelineGraphSection pipeline={pipeline} execution={execution} />

      {hasSpans && (
        <SpanTimeline jobId={jobId} spans={telemetryData.timeline} />
      )}

      {hasSteps && <StepDetailsList steps={execution.steps} />}

      <DataViewer title="Input Data" data={execution.input_data} />
      <DataViewer
        title="Final Output"
        data={execution.output_data}
        maxHeight="max-h-96 overflow-y-auto"
      />

      <CheckpointList jobId={jobId} />
      <ConversationHistory sessionId={jobId} />
      <ReplayComparison jobId={jobId} />
      <DecisionTimeline jobId={jobId} />
    </div>
  )
}
