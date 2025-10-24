import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Code, Eye, Columns, Save, Play, ArrowLeft } from 'lucide-react';
import VisualCanvas from '../components/editor/VisualCanvas';
import CodeEditor from '../components/editor/CodeEditor';

const VIEW_MODES = {
  VISUAL: 'visual',
  CODE: 'code',
  SPLIT: 'split',
};

export default function PipelineEditorPage() {
  const navigate = useNavigate();
  const [viewMode, setViewMode] = useState(VIEW_MODES.VISUAL);
  const [pipelineConfig, setPipelineConfig] = useState(getDefaultPipeline());
  const [codeValue, setCodeValue] = useState(JSON.stringify(pipelineConfig, null, 2));
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    // Sync code value when pipeline config changes from visual editor
    setCodeValue(JSON.stringify(pipelineConfig, null, 2));
  }, [pipelineConfig]);

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

  const handleSave = async () => {
    try {
      // Validate JSON
      const config = JSON.parse(codeValue);

      // TODO: Save to backend
      const response = await fetch('http://localhost:5555/api/pipelines', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Pipeline saved successfully! ID: ${result.id}`);
        setHasChanges(false);
      } else {
        throw new Error('Failed to save pipeline');
      }
    } catch (error) {
      alert(`Error saving pipeline: ${error.message}`);
    }
  };

  const handleRun = async () => {
    try {
      const config = JSON.parse(codeValue);

      const response = await fetch('http://localhost:5555/api/execute/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline: config,
          input_data: {},
        }),
      });

      if (response.ok) {
        const result = await response.json();
        navigate(`/executions/${result.job_id}`);
      } else {
        throw new Error('Failed to execute pipeline');
      }
    } catch (error) {
      alert(`Error executing pipeline: ${error.message}`);
    }
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
          <h1 className="text-2xl font-bold">Pipeline Editor</h1>
          {hasChanges && <span className="text-sm text-orange-600">● Unsaved changes</span>}
        </div>

        <div className="flex items-center gap-3">
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
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Run
          </button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="flex-1 overflow-hidden">
        {viewMode === VIEW_MODES.VISUAL && (
          <VisualCanvas pipelineConfig={pipelineConfig} onConfigChange={handleVisualChange} />
        )}

        {viewMode === VIEW_MODES.CODE && (
          <div className="h-full p-4">
            <CodeEditor value={codeValue} onChange={handleCodeChange} language="json" />
          </div>
        )}

        {viewMode === VIEW_MODES.SPLIT && (
          <div className="h-full flex">
            <div className="w-1/2 border-r">
              <VisualCanvas pipelineConfig={pipelineConfig} onConfigChange={handleVisualChange} />
            </div>
            <div className="w-1/2 p-4">
              <CodeEditor value={codeValue} onChange={handleCodeChange} language="json" />
            </div>
          </div>
        )}
      </div>
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
