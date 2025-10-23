import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { pipelinesAPI, executionAPI } from '../services/api'
import { Play, FileCode, Tag, Calendar, X } from 'lucide-react'

// Hardcoded test data for pipelines that require input
const DEFAULT_INPUTS = {
  'Conditional Processing Pipeline': {
    threshold: 0.8,
    test_data: [
      { quality_score: 0.95, content: 'high quality data 1' },
      { quality_score: 0.88, content: 'high quality data 2' },
      { quality_score: 0.92, content: 'high quality data 3' },
      { quality_score: 0.65, content: 'low quality data 1' },
      { quality_score: 0.45, content: 'low quality data 2' },
    ],
  },
  'Parallel Data Processing Pipeline': {
    loaded_data: [
      { id: 1, value: 'test data 1' },
      { id: 2, value: 'test data 2' },
      { id: 3, value: 'test data 3' },
      { id: 4, value: 'test data 4' },
      { id: 5, value: 'test data 5' },
      { id: 6, value: 'test data 6' },
    ],
  },
}

export default function PipelinesPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [executionDialog, setExecutionDialog] = useState(null)
  const [inputData, setInputData] = useState('')

  const { data: pipelines, isLoading } = useQuery({
    queryKey: ['pipelines'],
    queryFn: async () => {
      const response = await pipelinesAPI.list()
      return response.data
    },
  })

  const executeMutation = useMutation({
    mutationFn: async ({ pipelineId, inputData }) => {
      const response = await executionAPI.start(pipelineId, inputData || {})
      return response.data
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries(['executions'])
      setExecutionDialog(null)
      navigate(`/executions/${data.job_id}`)
    },
  })

  const handleExecute = (pipeline) => {
    // Get default input for this pipeline
    const defaultInput = DEFAULT_INPUTS[pipeline.name] || {}
    setInputData(JSON.stringify(defaultInput, null, 2))
    setExecutionDialog(pipeline)
  }

  const handleConfirmExecution = () => {
    try {
      const parsedInput = JSON.parse(inputData)
      console.log('Executing pipeline:', executionDialog.id, 'with input:', parsedInput)
      executeMutation.mutate({ pipelineId: executionDialog.id, inputData: parsedInput })
    } catch (e) {
      console.error('JSON parse error:', e)
      alert('Invalid JSON input: ' + e.message)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-600">Loading pipelines...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Pipelines</h1>
          <p className="text-gray-600 mt-1">Manage and execute your pipelines</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {pipelines?.map((pipeline) => (
          <PipelineCard
            key={pipeline.id}
            pipeline={pipeline}
            onExecute={() => handleExecute(pipeline)}
            isExecuting={executeMutation.isPending}
          />
        ))}
      </div>

      {executeMutation.isSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-green-800">
            Pipeline execution started! Job ID: {executeMutation.data.job_id}
          </p>
        </div>
      )}

      {/* Execution Dialog */}
      {executionDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-800">
                Execute: {executionDialog.name}
              </h2>
              <button
                onClick={() => setExecutionDialog(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 overflow-y-auto max-h-96">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Input Data (JSON)
              </label>
              <textarea
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                className="w-full h-64 p-3 border border-gray-300 rounded-lg font-mono text-sm"
                placeholder="{}"
              />
              <p className="text-xs text-gray-500 mt-2">
                Edit the JSON input data for this pipeline execution
              </p>
            </div>

            <div className="flex items-center justify-end gap-3 p-6 border-t bg-gray-50">
              <button
                onClick={() => setExecutionDialog(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmExecution}
                disabled={executeMutation.isPending}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center"
              >
                <Play size={16} className="mr-2" />
                Execute Pipeline
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PipelineCard({ pipeline, onExecute, isExecuting }) {
  return (
    <div className="bg-white rounded-lg shadow hover:shadow-lg transition p-6">
      <div className="flex items-start justify-between mb-4">
        <div className="bg-primary-100 rounded-lg p-3">
          <FileCode className="text-primary-600" size={24} />
        </div>
      </div>

      <h3 className="text-lg font-semibold text-gray-800 mb-2">{pipeline.name}</h3>
      <p className="text-gray-600 text-sm mb-4 line-clamp-2">{pipeline.description}</p>

      <div className="space-y-2 mb-4">
        <div className="flex items-center text-sm text-gray-500">
          <Tag size={16} className="mr-2" />
          <div className="flex flex-wrap gap-1">
            {pipeline.tags.map((tag) => (
              <span key={tag} className="bg-gray-100 px-2 py-1 rounded text-xs">
                {tag}
              </span>
            ))}
          </div>
        </div>

        <div className="flex items-center text-sm text-gray-500">
          <Calendar size={16} className="mr-2" />
          Created {new Date(pipeline.created_at).toLocaleDateString()}
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <button
          onClick={onExecute}
          disabled={isExecuting}
          className="flex-1 bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-primary-700 transition disabled:opacity-50 flex items-center justify-center"
        >
          <Play size={16} className="mr-2" />
          Execute
        </button>
      </div>
    </div>
  )
}
