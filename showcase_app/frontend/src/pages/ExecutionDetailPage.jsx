import React, { useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { executionAPI, pipelinesAPI } from '../services/api'
import { useExecutionWebSocket } from '../hooks/useWebSocket'
import axios from 'axios'
import DragDropContainer from '../components/common/DragDropContainer'

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

  const handleTemplateImport = (templateItems, templateName) => {
    console.log('Imported template:', templateName, templateItems)
    // Could show a toast or update the page title
  }

  return (
    <DragDropContainer
      onLayoutChange={(layout) => console.log('Layout changed:', layout)}
      onTemplateImport={handleTemplateImport}
      layoutKey="execution-detail-layout"
      execution={execution}
      pipeline={pipeline}
      jobId={jobId}
      telemetryData={telemetryData}
    >
      {/* Pass execution data to all components that need it */}
      <div id="execution-header" data-component="ExecutionHeader" data-props={{ onBack: () => navigate('/executions') }} />
      <div id="execution-status" data-component="ExecutionStatusCard" data-props={{ execution }} />
      <div id="execution-error" data-component="ExecutionError" data-props={{ error: execution.error }} />
      <div id="execution-timeline" data-component="ExecutionTimeline" data-props={{ execution }} />
      <div id="pipeline-graph" data-component="PipelineGraphSection" data-props={{ pipeline, execution }} />

      {telemetryData?.timeline && telemetryData.timeline.length > 0 && (
        <div id="span-timeline" data-component="SpanTimeline" data-props={{ jobId, spans: telemetryData.timeline }} />
      )}

      <div id="checkpoints" data-component="CheckpointList" data-props={{ jobId }} />
      <div id="conversation-history" data-component="ConversationHistory" data-props={{ sessionId: jobId }} />
      <div id="replay-comparison" data-component="ReplayComparison" data-props={{ jobId }} />
      <div id="decision-timeline" data-component="DecisionTimeline" data-props={{ jobId }} />
      <div id="step-details" data-component="StepDetailsList" data-props={{ steps: execution.steps }} />
      <div id="input-data-viewer" data-component="DataViewer" data-props={{ title: "Input Data", data: execution.input_data }} />
      <div id="output-data-viewer" data-component="DataViewer" data-props={{
        title: "Final Output",
        data: execution.output_data,
        maxHeight: "max-h-96 overflow-y-auto"
      }} />
    </DragDropContainer>
  )
}
