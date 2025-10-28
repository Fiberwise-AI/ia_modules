import { useState, useEffect } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Code, Eye, Columns, Save, Play, ArrowLeft, FolderOpen, X, ChevronDown, ChevronUp } from 'lucide-react';
import VisualCanvas from '../components/editor/VisualCanvas';
import CodeEditor from '../components/editor/CodeEditor';
import ExecutionDetailsModal from '../components/execution/ExecutionDetailsModal';
import { pipelinesAPI, executionAPI, hitlAPI } from '../services/api';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import HITLInteractionModal from '../components/hitl/HITLInteractionModal';


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
    task: 'Analyze customer feedback data',
    text: 'The product quality is excellent but shipping was delayed. Customer service was responsive and helpful.',
  },
  'Human-in-the-Loop Test Pipeline': {},
};

const VIEW_MODES = {
  VISUAL: 'visual',
  CODE: 'code',
  SPLIT: 'split',
};

export default function PipelineEditorPage() {
  const navigate = useNavigate();
  const { pipelineId } = useParams();
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [viewMode, setViewMode] = useState(VIEW_MODES.VISUAL);
  const [pipelineConfig, setPipelineConfig] = useState(getDefaultPipeline());
  const [codeValue, setCodeValue] = useState(JSON.stringify(pipelineConfig, null, 2));
  const [hasChanges, setHasChanges] = useState(false);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [showExecutionDialog, setShowExecutionDialog] = useState(false);
  const [inputData, setInputData] = useState('');
  const [showExecutionsTable, setShowExecutionsTable] = useState(true);
  const [selectedExecution, setSelectedExecution] = useState(null);
  const [selectedHITLInteraction, setSelectedHITLInteraction] = useState(null);

  const { data: executions = [] } = useQuery({
    queryKey: ['executions', pipelineId],
    queryFn: async () => {
      if (!pipelineId) return [];
      const response = await executionAPI.list();
      return response.data.filter(e => e.pipeline_id === pipelineId);
    },
    enabled: !!pipelineId,
    refetchInterval: 2000, // Refetch every 2 seconds for real-time updates
  });

  // Fetch pending HITL interactions for this pipeline
  const { data: hitlInteractions = [] } = useQuery({
    queryKey: ['hitl-interactions', pipelineId],
    queryFn: async () => {
      if (!pipelineId) return [];
      const response = await hitlAPI.getPending(null, pipelineId);
      return response.data;
    },
    enabled: !!pipelineId,
    refetchInterval: 5000, // Poll every 5 seconds for new interactions
  });

  // Load existing pipeline if pipelineId is provided
  const { data: existingPipeline, isLoading: isLoadingPipeline } = useQuery({
    queryKey: ['pipeline', pipelineId],
    queryFn: async () => {
      if (!pipelineId) return null;
      const response = await pipelinesAPI.get(pipelineId);
      return response.data;
    },
    enabled: !!pipelineId,
  });

  // Load all pipelines for the selector
  const { data: allPipelines } = useQuery({
    queryKey: ['pipelines'],
    queryFn: async () => {
      const response = await pipelinesAPI.list();
      return response.data;
    },
  });

  // Initialize from existing pipeline or URL params
  useEffect(() => {
    if (existingPipeline) {
      setPipelineConfig(existingPipeline.config);
      setCodeValue(JSON.stringify(existingPipeline.config, null, 2));
      setHasChanges(false);
    } else if (!pipelineId) {
      // Check if pipeline data is passed via URL params
      const pipelineData = searchParams.get('data');
      if (pipelineData) {
        try {
          const config = JSON.parse(decodeURIComponent(pipelineData));
          setPipelineConfig(config);
          setCodeValue(JSON.stringify(config, null, 2));
          setHasChanges(false);
        } catch (e) {
          console.error('Failed to parse pipeline data from URL:', e);
        }
      }
    }
  }, [existingPipeline, pipelineId, searchParams]);

  useEffect(() => {
    // Sync code value when pipeline config changes from visual editor
    setCodeValue(JSON.stringify(pipelineConfig, null, 2));
  }, [pipelineConfig]);

  // WebSocket listener for execution updates
  useEffect(() => {
    if (!pipelineId) return;

    const ws = new WebSocket(`${import.meta.env.VITE_WS_URL || 'ws://localhost:5555'}/ws/pipeline/${pipelineId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Invalidate executions query to refetch data
      if (data.type === 'execution_completed' || data.type === 'execution_failed' || data.type === 'execution_paused') {
        queryClient.invalidateQueries(['executions', pipelineId]);
        queryClient.invalidateQueries(['hitl-interactions', pipelineId]);
      }
    };

    return () => ws.close();
  }, [pipelineId, queryClient]);

  const handleVisualChange = (newConfig) => {
    setPipelineConfig(newConfig);
    setHasChanges(true);
  };

  const handleCodeChange = (newCode) => {
    setCodeValue(newCode);
    setHasChanges(true);

    // Try to parse and update pipeline config
    try {
      const parsed = JSON.parse(newCode);
      setPipelineConfig(parsed);
    } catch (e) {
      // Invalid JSON, don't update visual view
      console.log('Invalid JSON, keeping visual view unchanged');
    }
  };

  const handleLoadPipeline = (pipeline) => {
    const config = pipeline.config || pipeline;
    setPipelineConfig(config);
    setCodeValue(JSON.stringify(config, null, 2));
    setHasChanges(false);
    setShowLoadDialog(false);
    // Update URL to reflect loaded pipeline
    navigate(`/editor/${pipeline.id}`, { replace: true });
  };

  const handleSave = async () => {
    try {
      // Validate JSON
      const config = JSON.parse(codeValue);

      let response;
      if (pipelineId) {
        // Update existing pipeline
        response = await pipelinesAPI.update(pipelineId, config);
      } else {
        // Create new pipeline
        response = await pipelinesAPI.create(config);
      }

      if (response.status === 200 || response.status === 201) {
        const result = response.data;
        alert(`Pipeline saved successfully! ID: ${result.id}`);
        setHasChanges(false);
        // Update URL if this was a new pipeline
        if (!pipelineId) {
          navigate(`/editor/${result.id}`, { replace: true });
        }
      } else {
        throw new Error('Failed to save pipeline');
      }
    } catch (error) {
      alert(`Error saving pipeline: ${error.message}`);
    }
  };

  const handleRun = () => {
    const config = JSON.parse(codeValue);
    const pipelineName = config.name || existingPipeline?.name;
    const defaultInput = DEFAULT_INPUTS[pipelineName] || {};
    setInputData(JSON.stringify(defaultInput, null, 2));
    setShowExecutionDialog(true);
  };

  const handleConfirmExecution = async () => {
    try {
      setIsExecuting(true);
      setExecutionResult(null);
      setShowExecutionDialog(false);

      const parsedInput = JSON.parse(inputData);
      const response = await executionAPI.start(pipelineId, parsedInput);
      setExecutionResult(response.data);

      // Invalidate executions query to refetch the list
      queryClient.invalidateQueries({ queryKey: ['executions', pipelineId] });
    } catch (error) {
      alert(`Error executing pipeline: ${error.message}`);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleHITLResponse = async (interactionId, humanInput) => {
    await hitlAPI.respond(interactionId, humanInput);
    // Invalidate queries to refresh data
    queryClient.invalidateQueries({ queryKey: ['hitl-interactions', pipelineId] });
    queryClient.invalidateQueries({ queryKey: ['executions', pipelineId] });
  };

  const handleHITLCancel = async (interactionId) => {
    await hitlAPI.cancel(interactionId);
    queryClient.invalidateQueries({ queryKey: ['hitl-interactions', pipelineId] });
    queryClient.invalidateQueries({ queryKey: ['executions', pipelineId] });
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/pipelines')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold">
              {existingPipeline ? existingPipeline.name : 'Pipeline Editor'}
            </h1>
            {existingPipeline && (
              <p className="text-sm text-gray-600">{existingPipeline.description}</p>
            )}
          </div>
          {hasChanges && <span className="text-sm text-orange-600">● Unsaved changes</span>}
        </div>

        <div className="flex items-center gap-3">
          {/* Load Pipeline Button */}
          <button
            onClick={() => setShowLoadDialog(true)}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <FolderOpen className="w-4 h-4" />
            Load
          </button>
          {/* View Mode Selector */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode(VIEW_MODES.VISUAL)}
              className={`px-3 py-2 rounded flex items-center gap-2 transition-colors ${
                viewMode === VIEW_MODES.VISUAL
                  ? 'bg-white shadow-sm text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Eye className="w-4 h-4" />
              Visual
            </button>
            <button
              onClick={() => setViewMode(VIEW_MODES.CODE)}
              className={`px-3 py-2 rounded flex items-center gap-2 transition-colors ${
                viewMode === VIEW_MODES.CODE
                  ? 'bg-white shadow-sm text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Code className="w-4 h-4" />
              Code
            </button>
            <button
              onClick={() => setViewMode(VIEW_MODES.SPLIT)}
              className={`px-3 py-2 rounded flex items-center gap-2 transition-colors ${
                viewMode === VIEW_MODES.SPLIT
                  ? 'bg-white shadow-sm text-blue-600'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Columns className="w-4 h-4" />
              Split
            </button>
          </div>

          {/* Actions */}
          <button
            onClick={handleSave}
            disabled={!hasChanges}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          <button
            onClick={handleRun}
            disabled={isExecuting}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Run
          </button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="flex-1 overflow-hidden">
        {viewMode === VIEW_MODES.VISUAL && (
          <VisualCanvas pipelineConfig={pipelineConfig} pipelineId={pipelineId} onConfigChange={handleVisualChange} />
        )}

        {viewMode === VIEW_MODES.CODE && (
          <div className="h-full p-4">
            <CodeEditor value={codeValue} onChange={handleCodeChange} language="json" />
          </div>
        )}

        {viewMode === VIEW_MODES.SPLIT && (
          <div className="h-full flex">
            <div className="w-1/2 border-r">
              <VisualCanvas pipelineConfig={pipelineConfig} pipelineId={pipelineId} onConfigChange={handleVisualChange} />
            </div>
            <div className="w-1/2 p-4">
              <CodeEditor value={codeValue} onChange={handleCodeChange} language="json" />
            </div>
          </div>
        )}
      </div>

      {/* Execution Results Panel */}
      {pipelineId && executions && executions.length > 0 && (
        <div className="border-t bg-gray-50">
          <button
            onClick={() => setShowExecutionsTable(!showExecutionsTable)}
            className="w-full flex items-center justify-between p-4 hover:bg-gray-100 transition"
          >
            <h3 className="font-semibold text-gray-900">Recent Executions ({executions.length})</h3>
            {showExecutionsTable ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>
          {showExecutionsTable && (
            <div className="px-4 pb-4 max-h-96 overflow-y-auto">
              <div className="bg-white border rounded overflow-hidden">
            <table className="min-w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Job ID</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Status</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Started</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Duration</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {executions.map((execution) => {
                  const executionHITL = hitlInteractions.filter(h => h.execution_id === execution.job_id);
                  const hasPendingApproval = executionHITL.length > 0 && execution.status === 'waiting_for_human';

                  return (
                  <tr key={execution.job_id} className={`hover:bg-gray-50 ${hasPendingApproval ? 'bg-yellow-50' : ''}`}>
                    <td className="px-4 py-2 text-xs font-mono text-gray-900">
                      {execution.job_id.slice(0, 8)}...
                    </td>
                    <td className="px-4 py-2 text-xs">
                      <span className={`px-2 py-1 rounded text-xs ${
                        execution.status === 'completed' ? 'bg-green-100 text-green-800' :
                        execution.status === 'failed' ? 'bg-red-100 text-red-800' :
                        execution.status === 'waiting_for_human' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {execution.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-600">
                      {new Date(execution.started_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-600">
                      {execution.execution_time_ms ? `${(execution.execution_time_ms / 1000).toFixed(2)}s` : '-'}
                    </td>
                    <td className="px-4 py-2">
                      <button
                        onClick={async () => {
                          const response = await executionAPI.get(execution.job_id);
                          const exec = response.data;

                          // If waiting for human, get or construct HITL interaction
                          let hitl = executionHITL[0];
                          if (!hitl && exec.status === 'waiting_for_human' && exec.output_data?.status === 'human_input_required') {
                            // Fetch HITL interaction by execution ID
                            try {
                              const hitlResponse = await hitlAPI.getPending(exec.job_id);
                              hitl = hitlResponse.data[0];
                            } catch (e) {
                              console.error('Failed to fetch HITL interaction:', e);
                            }
                          }

                          setSelectedExecution({
                            execution: exec,
                            hitlInteraction: hitl || null
                          });
                        }}
                        className="text-xs text-blue-600 hover:text-blue-800"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
            </div>
          )}
        </div>
      )}

      {/* Execution Dialog */}
      {showExecutionDialog && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full my-8 flex flex-col max-h-[90vh]">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-800">Execute Pipeline</h2>
              <button
                onClick={() => setShowExecutionDialog(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Input Data (JSON)
              </label>
              <textarea
                value={inputData}
                onChange={(e) => setInputData(e.target.value)}
                className="w-full h-64 p-3 border rounded-lg font-mono text-sm"
                placeholder="{}"
              />
              <p className="text-xs text-gray-500 mt-2">
                Edit the JSON input data for this pipeline execution
              </p>
            </div>

            <div className="flex items-center justify-end gap-3 p-6 border-t bg-gray-50">
              <button
                onClick={() => setShowExecutionDialog(false)}
                className="px-4 py-2 border rounded-lg text-gray-700 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmExecution}
                disabled={isExecuting}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
              >
                <Play size={16} />
                Execute Pipeline
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Load Pipeline Dialog */}
      {showLoadDialog && (
        <div className="fixed inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full my-8 flex flex-col max-h-[80vh]">
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-100">
                Load Pipeline
              </h2>
              <button
                onClick={() => setShowLoadDialog(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition"
              >
                ×
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              {allPipelines && allPipelines.length > 0 ? (
                <div className="space-y-3">
                  {allPipelines.map((pipeline) => (
                    <div
                      key={pipeline.id}
                      onClick={() => handleLoadPipeline(pipeline)}
                      className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition"
                    >
                      <h3 className="font-semibold text-gray-800 dark:text-gray-100">
                        {pipeline.name}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {pipeline.description}
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        {pipeline.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-xs rounded"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-600 dark:text-gray-400">No pipelines available</p>
                  <button
                    onClick={() => navigate('/pipelines')}
                    className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                  >
                    Go to Pipelines Page
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Execution Details Modal */}
      {selectedExecution && (
        <ExecutionDetailsModal
          execution={selectedExecution.execution}
          hitlInteraction={selectedExecution.hitlInteraction}
          onHITLApprove={(interaction) => setSelectedHITLInteraction(interaction)}
          onClose={() => setSelectedExecution(null)}
        />
      )}

      {/* HITL Interaction Modal */}
      {selectedHITLInteraction && (
        <HITLInteractionModal
          interaction={selectedHITLInteraction}
          onRespond={handleHITLResponse}
          onCancel={handleHITLCancel}
          onClose={() => setSelectedHITLInteraction(null)}
        />
      )}
    </div>
  );
}

function getDefaultPipeline() {
  return {
    name: 'New Pipeline',
    description: 'Create your pipeline here',
    steps: [],
    flow: {
      start_at: '',
      paths: [],
    },
  };
}
