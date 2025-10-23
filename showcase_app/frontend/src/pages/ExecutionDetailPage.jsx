import React, { useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { executionAPI, pipelinesAPI } from '../services/api'
import { useExecutionWebSocket } from '../hooks/useWebSocket'

import ExecutionHeader from '../components/execution/ExecutionHeader'
import ExecutionStatusCard from '../components/execution/ExecutionStatusCard'
import ExecutionError from '../components/execution/ExecutionError'
import PipelineGraphSection from '../components/execution/PipelineGraphSection'
import StepDetailsList from '../components/execution/StepDetailsList'
import DataViewer from '../components/execution/DataViewer'

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

      <PipelineGraphSection pipeline={pipeline} execution={execution} />

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
