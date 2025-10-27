import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { pipelinesAPI, executionAPI } from '../services/api'
import { Play, FileCode, Tag, Calendar, X, Inbox, Edit } from 'lucide-react'
import { LoadingSpinner, ButtonSpinner } from '../components/ui/spinner'
import { SkeletonCard } from '../components/ui/skeleton'
import { EmptyState } from '../components/ui/empty-state'
import { useToast } from '../hooks/useToast'

// Hardcoded test data for pipelines that require input
const DEFAULT_INPUTS = {
  'Simple Three-Step Pipeline': {
    topic: 'artificial intelligence',
  },
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
  'Iterative Content Generation Pipeline': {
    topic: 'machine learning',
    max_revisions: 3,
  },
  'Agent-Based Processing Pipeline': {
    input_data: {},
  },
  'Human-in-the-Loop Test Pipeline': {},
}

export default function PipelinesPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const toast = useToast()
  const [executionDialog, setExecutionDialog] = useState(null)
  const [inputData, setInputData] = useState('')

  const { data: pipelines, isLoading, error } = useQuery({
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
      toast.success(`Pipeline execution started! Job ID: ${data.job_id}`)
      navigate(`/executions/${data.job_id}`)
    },
    onError: (error) => {
      toast.error(`Failed to start pipeline: ${error.message}`)
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
      toast.error(`Invalid JSON input: ${e.message}`)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Pipelines</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Manage and execute your pipelines</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Pipelines</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Manage and execute your pipelines</p>
        </div>
        <EmptyState
          icon={FileCode}
          title="Failed to load pipelines"
          description={error.message || "We couldn't load your pipelines. Please try again."}
          action={() => window.location.reload()}
          actionLabel="Retry"
        />
      </div>
    )
  }

  if (!pipelines || pipelines.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Pipelines</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Manage and execute your pipelines</p>
        </div>
        <EmptyState
          icon={Inbox}
          title="No pipelines yet"
          description="Create your first pipeline to get started with IA Modules."
          action={() => navigate('/editor')}
          actionLabel="Create Pipeline"
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">Pipelines</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Manage and execute your pipelines</p>
        </div>
        <button
          onClick={() => navigate('/editor')}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
        >
          <FileCode size={16} />
          New Pipeline
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {pipelines?.map((pipeline) => (
          <PipelineCard
            key={pipeline.id}
            pipeline={pipeline}
            onExecute={() => handleExecute(pipeline)}
            onEdit={() => navigate(`/editor/${pipeline.id}`)}
            isExecuting={executeMutation.isPending}
          />
        ))}
      </div>

      {/* Execution Dialog */}
      {executionDialog && (
        <div className="fixed inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full my-8 flex flex-col max-h-[90vh]">
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">
                Execute: {executionDialog.name}
              </h2>
              <button
                onClick={() => setExecutionDialog(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Input Data (JSON)
              </label>
              <textarea
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                className="w-full h-64 p-3 border border-gray-300 dark:border-gray-600 rounded-lg font-mono text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                placeholder="{}"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Edit the JSON input data for this pipeline execution
              </p>
            </div>

            <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex-shrink-0">
              <button
                onClick={() => setExecutionDialog(null)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmExecution}
                disabled={executeMutation.isPending}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2 transition"
              >
                {executeMutation.isPending ? (
                  <>
                    <ButtonSpinner />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    Execute Pipeline
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function PipelineCard({ pipeline, onExecute, onEdit, isExecuting }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow hover:shadow-lg dark:hover:shadow-2xl transition p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-start justify-between mb-4">
        <div className="bg-primary-100 dark:bg-primary-900/30 rounded-lg p-3">
          <FileCode className="text-primary-600 dark:text-primary-400" size={24} />
        </div>
      </div>

      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mb-2">{pipeline.name}</h3>
      <p className="text-gray-600 dark:text-gray-400 text-sm mb-4 line-clamp-2">{pipeline.description}</p>

      <div className="space-y-2 mb-4">
        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
          <Tag size={16} className="mr-2" />
          <div className="flex flex-wrap gap-1">
            {pipeline.tags.map((tag) => (
              <span key={tag} className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded text-xs">
                {tag}
              </span>
            ))}
          </div>
        </div>

        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
          <Calendar size={16} className="mr-2" />
          Created {new Date(pipeline.created_at).toLocaleDateString()}
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <button
          onClick={onEdit}
          className="px-3 py-2 border border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition flex items-center gap-2"
        >
          <Edit size={16} />
          Edit
        </button>
        <button
          onClick={onExecute}
          disabled={isExecuting}
          className="flex-1 bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-primary-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isExecuting ? (
            <>
              <ButtonSpinner />
              Executing...
            </>
          ) : (
            <>
              <Play size={16} />
              Execute
            </>
          )}
        </button>
      </div>
    </div>
  )
}
